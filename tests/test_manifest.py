"""T1-01 contracts: immutable dataset manifests, cached partition hashes and
experiment-to-dataset binding.

The manifest is the single source of truth for "which data did this run
see".  These tests pin the content-addressing contract: identical content
produces identical ``dataset_id`` (regardless of insertion order), any
content or schema change produces a different id, partition hashes are
cached across builds, and experiments must either bind to the current
dataset_id or be rejected explicitly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

import ashare_data.manifest as manifest_mod
import ashare_data.sync as sync
from ashare_data.config import DataConfig
from ashare_data.db import AshareDB
from ashare_data.manifest import (
    DatasetIdMismatch,
    build_dataset_manifest,
    check_dataset_id,
    compute_partition_fingerprint,
    compute_table_fingerprint,
    latest_manifest,
    resolve_dataset_id,
    save_manifest,
    schema_digest,
    verify_dataset,
)
from tests.conftest import make_bars

# Every data table created by create_schema must be fingerprinted.
EXPECTED_MANIFEST_TABLES = {
    "stocks",
    "daily_bar",
    "constituents",
    "trade_calendar",
    "fundamental_pit",
    "margin_balance",
    "sw_industry_index",
    "sw_industry_member",
}


def _manifest(db: AshareDB, config: DataConfig, **kwargs):
    return build_dataset_manifest(db, config, source_versions={"test": "1"}, **kwargs)


def test_manifest_records_schema_rows_dates_source_and_partitions(populated_db):
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        manifest = _manifest(db, populated_db)
    # Bound to the authority, not a literal (A02, p26 §4.2 / captain ruling
    # 2): this asserts "the record carries the CURRENT manifest generation",
    # which stays correct across bumps instead of rotting into a
    # maintenance-only edit. Strength is unchanged — still exact equality.
    # The literal pin of the generation itself lives in
    # ``tests/test_runspec.py::test_a02_version_chain_wired_into_the_version_registry``.
    assert manifest.manifest_version == manifest_mod.MANIFEST_VERSION
    assert manifest.dataset_id
    assert manifest.merkle_root
    assert manifest.source_versions["test"] == "1"
    assert set(t.table for t in manifest.tables) == EXPECTED_MANIFEST_TABLES
    daily = manifest.table("daily_bar")
    assert daily.row_count == 40 * 3
    assert daily.date_range == ("20240101", "20240223")
    # Year partitions of the daily table carry per-partition hashes.
    assert {p.partition_key for p in daily.partitions} == {"2024"}
    assert all(len(p.hash) == 32 and p.row_count > 0 for p in daily.partitions)


def test_manifest_is_content_addressed_and_stable(populated_db):
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        first = _manifest(db, populated_db)
        second = _manifest(db, populated_db)
    # Content-addressing contract: id, root, hashes and metadata are
    # stable; only the created_at timestamp may differ across builds.
    assert first.dataset_id == second.dataset_id
    assert first.merkle_root == second.merkle_root
    assert first.tables == second.tables
    assert first.source_versions == second.source_versions
    assert first.manifest_version == second.manifest_version


def test_manifest_identical_content_ignores_insertion_order(tmp_path):
    def build(shuffle: bool) -> str:
        cfg = DataConfig(
            data_dir=tmp_path,
            duckdb_path=tmp_path / f"db_{shuffle}.duckdb",
            parquet_dir=tmp_path / f"parquet_{shuffle}",
            start_date="2024-01-01",
            end_date="2024-12-31",
            index_codes=["000300.SH"],
            min_listed_sessions=1,
        )
        dates, ts_codes, bars = make_bars(n_dates=20)
        rows = bars.to_dict("records")
        if shuffle:
            rng = __import__("numpy").random.default_rng(7)
            rows = [rows[i] for i in rng.permutation(len(rows))]
        with AshareDB(cfg.duckdb_path) as db:
            db.create_schema(cfg)
            db.upsert_daily(rows, cfg)
        with AshareDB(cfg.duckdb_path, read_only=True) as db:
            return _manifest(db, cfg).dataset_id

    assert build(shuffle=False) == build(shuffle=True)


def test_manifest_changes_when_rows_change(populated_db):
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        before = _manifest(db, populated_db)
    with AshareDB(populated_db.duckdb_path) as db:
        db.execute(
            "INSERT INTO daily_bar VALUES "
            "('000001.SZ', '20240301', 10, 10, 10, 10, 10, 1, 10, 1, 1)"
        )
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        after = _manifest(db, populated_db)
    assert after.dataset_id != before.dataset_id
    # Only the touched table's fingerprint changed; the untouched tables
    # keep their exact partition hashes (content-addressing per table).
    for table in before.tables:
        if table.table == "daily_bar":
            assert table != after.table("daily_bar")
        else:
            assert table == after.table(table.table)


def test_manifest_changes_when_schema_changes(populated_db):
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        before = _manifest(db, populated_db)
        digest_before = schema_digest(db, "daily_bar")
    with AshareDB(populated_db.duckdb_path) as db:
        db.execute("ALTER TABLE daily_bar ADD COLUMN extra DOUBLE")
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        after = _manifest(db, populated_db)
        digest_after = schema_digest(db, "daily_bar")
    assert digest_before != digest_after
    assert after.dataset_id != before.dataset_id


def test_partition_hashes_are_cached_across_builds(populated_db, monkeypatch):
    from ashare_data import manifest as manifest_module

    calls = {"n": 0}
    real = manifest_module._compute_partition_hash

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(manifest_module, "_compute_partition_hash", counting)
    with AshareDB(populated_db.duckdb_path) as db:
        _manifest(db, populated_db)  # warm the cache
        calls["n"] = 0
        second = _manifest(db, populated_db)  # cache hit: no recompute
    assert calls["n"] == 0
    assert second.dataset_id
    with AshareDB(populated_db.duckdb_path) as db:
        cold = _manifest(db, populated_db, use_cache=False)
    assert cold.dataset_id == second.dataset_id
    assert calls["n"] > 0


def test_cache_invalidated_by_row_count_change(populated_db):
    with AshareDB(populated_db.duckdb_path) as db:
        before = _manifest(db, populated_db)
        db.execute(
            "INSERT INTO daily_bar VALUES "
            "('000001.SZ', '20240302', 10, 10, 10, 10, 10, 1, 10, 1, 1)"
        )
        after = _manifest(db, populated_db)
    assert after.dataset_id != before.dataset_id
    assert after.table("daily_bar").row_count == before.table("daily_bar").row_count + 1


def test_save_latest_and_resolve_roundtrip(populated_db_without_manifest):
    # P8-05: the shared populated_db fixture now persists a manifest, so
    # this save/resolve round-trip runs against the manifest-free variant.
    config = populated_db_without_manifest
    with AshareDB(config.duckdb_path) as db:
        manifest = _manifest(db, config)
        save_manifest(db, config, manifest)
        save_manifest(db, config, manifest)  # idempotent
        assert latest_manifest(db, config) == manifest
        assert resolve_dataset_id(db, config) == manifest.dataset_id
        rows = db.query("SELECT COUNT(*) AS n FROM dataset_manifest").iloc[0]["n"]
        assert rows == 1


def test_verify_dataset_fast_and_full_agree(populated_db):
    with AshareDB(populated_db.duckdb_path) as db:
        manifest = _manifest(db, populated_db)
        fast = verify_dataset(db, populated_db, expected=manifest, verify="fast")
        full = verify_dataset(db, populated_db, expected=manifest, verify="full")
    assert fast.ok and full.ok
    assert all(entry.ok for entry in fast.tables.values())
    assert set(fast.tables) == EXPECTED_MANIFEST_TABLES


def test_verify_dataset_detects_drift(populated_db):
    with AshareDB(populated_db.duckdb_path) as db:
        manifest = _manifest(db, populated_db)
    with AshareDB(populated_db.duckdb_path) as db:
        db.execute(
            "INSERT INTO daily_bar VALUES "
            "('000001.SZ', '20240303', 10, 10, 10, 10, 10, 1, 10, 1, 1)"
        )
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        fast = verify_dataset(db, populated_db, expected=manifest, verify="fast")
        full = verify_dataset(db, populated_db, expected=manifest, verify="full")
    assert not fast.ok
    assert not full.ok
    assert not fast.tables["daily_bar"].ok


def test_check_dataset_id_contract():
    check_dataset_id(None, "abc")  # legacy artifacts without an id pass
    check_dataset_id("abc", "abc")
    with pytest.raises(DatasetIdMismatch):
        check_dataset_id("abc", "def")


def test_sync_records_dataset_id(tmp_path):
    cfg_path = tmp_path / "ashare_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "duckdb_path": str(tmp_path / "data" / "ashare.duckdb"),
                "parquet_dir": str(tmp_path / "data" / "parquet"),
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
        ),
        encoding="utf-8",
    )
    result = sync.sync_all(cfg_path, offline=True, limit=3)
    dataset_id = result["dataset_id"]
    assert isinstance(dataset_id, str) and len(dataset_id) == 64

    cfg = DataConfig(
        data_dir=tmp_path / "data",
        duckdb_path=tmp_path / "data" / "ashare.duckdb",
        parquet_dir=tmp_path / "data" / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    with AshareDB(cfg.duckdb_path, read_only=True) as db:
        assert resolve_dataset_id(db, cfg) == dataset_id
        manifest = latest_manifest(db, cfg)
        assert manifest is not None
        assert manifest.dataset_id == dataset_id

    from ashare_model.data_loader import AshareDataLoader
    from ashare_data.config import ModelConfig
    from ashare_data.universe import UniverseDevelopmentFallbackWarning

    # p19: the loader's formal price source is the v2 database; mirror the
    # synced bars into it (sync itself remains the legacy pipeline).
    from tests.conftest import populate_v2_for_config

    with AshareDB(cfg.duckdb_path, read_only=True) as db:
        synced_bars = db.query(
            "SELECT ts_code, trade_date, open, high, low, close, pre_close, "
            "volume, amount FROM daily_bar"
        )
    populate_v2_for_config(
        cfg,
        sorted(synced_bars["trade_date"].astype(str).unique().tolist()),
        sorted(synced_bars["ts_code"].unique().tolist()),
        synced_bars,
    )

    # The offline sync writes only snapshot constituents, so the loader
    # must use the explicit development universe fallback (strict mode
    # requires historical PIT intervals).
    loader = AshareDataLoader(
        cfg, ModelConfig(), allow_development_universe_fallback=True
    )
    # IP-10 03: the fallback warning is a pinned contract signal, not
    # unaccounted noise -- the loader MUST warn when it falls back to the
    # development universe (fail-closed discipline, AGENTS 5.3).
    with pytest.warns(UniverseDevelopmentFallbackWarning):
        loader.load_data()
    assert loader.dataset_id == dataset_id


def test_loader_exposes_dataset_id_when_absent(populated_db_without_manifest):
    # P8-05: the shared populated_db fixture now persists a manifest; the
    # "no manifest saved" case uses the manifest-free variant.
    # A02 (p26 I2 + §6-3, acceptance 1): the pre-A02 binding — "no manifest,
    # bind None and carry on" — is abolished on the formal path, which must
    # refuse an unconfirmable identity; it survives only under the explicitly
    # requested legacy audit mode, and there it leaves a machine marker.
    from ashare_data.config import ModelConfig
    from ashare_model.data_loader import AshareDataLoader

    with pytest.raises(DatasetIdMismatch):
        AshareDataLoader(populated_db_without_manifest, ModelConfig()).load_data()
    legacy = AshareDataLoader(
        populated_db_without_manifest,
        ModelConfig(),
        dataset_identity_mode="query",
    ).load_data()
    assert legacy.dataset_id is None  # no manifest saved yet
    assert legacy.dataset_identity_status == "degraded"


def test_manifest_cli_persists_dataset_id(populated_db, monkeypatch):
    """``python -m ashare_data.manifest`` is the migration entry for
    pre-T1-01 databases: it builds and persists the manifest so the
    current data gets a non-empty dataset_id (P0-03 acceptance)."""

    import ashare_data.config as config_module
    from ashare_data import manifest as manifest_module

    monkeypatch.setattr(config_module, "load_config", lambda *a, **k: {})
    monkeypatch.setattr(config_module, "make_data_config", lambda raw, root: populated_db)
    rc = manifest_module.main([])
    assert rc == 0
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        assert resolve_dataset_id(db, populated_db) is not None
        manifest = latest_manifest(db, populated_db)
        assert manifest is not None
        assert manifest.dataset_id == resolve_dataset_id(db, populated_db)


def test_partition_fingerprint_rejects_unknown_table(populated_db):
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        with pytest.raises(ValueError):
            compute_partition_fingerprint(db, populated_db, "no_such_table", "ALL")
        with pytest.raises(ValueError):
            compute_table_fingerprint(db, populated_db, "no_such_table")


# --- A02 (t12): content-confirmed identity vs latest-saved id -----------------
#
# RED block for the A02 implementation contract
# (reports/alphagpt-ab/A02_contract_implementation_plan.md §5, p26 §6):
# the identity a formal run uses must be *content-confirmed*, and the
# legacy "latest saved manifest" API must never stand in for it.


def test_a02_content_confirmation_api_exists_and_is_distinct(populated_db):
    """A02 acceptance 1 (p26 I1): 'latest saved id' and 'content-confirmed
    id' are two different, self-describing APIs — the formal entry uses the
    second one only."""

    from ashare_data import manifest as manifest_mod

    confirmer = getattr(manifest_mod, "confirm_dataset_identity", None)
    assert confirmer is not None, (
        "A02 requires a content-confirming identity API "
        "(manifest.confirm_dataset_identity) distinct from resolve_dataset_id"
    )
    assert callable(confirmer)
    # The legacy query keeps its meaning (latest saved row) so old callers
    # cannot silently start reporting a confirmed identity.
    assert callable(manifest_mod.resolve_dataset_id)
    assert confirmer is not manifest_mod.resolve_dataset_id


def test_a02_stale_manifest_is_refused_by_content_confirmation(populated_db):
    """A02 acceptance 1/6 (p26 I2/I5, R1/R2): after the content moves on,
    the confirmation API must refuse the previously saved id instead of
    handing it back as 'the current dataset'."""

    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        saved = _manifest(db, populated_db)
    with AshareDB(populated_db.duckdb_path) as db:
        save_manifest(db, populated_db, saved)

    # Content drifts after the manifest was published (append one session).
    with AshareDB(populated_db.duckdb_path) as db:
        db.execute(
            "INSERT INTO daily_bar VALUES "
            "('000001.SZ', '20240301', 10, 10, 10, 10, 10, 1, 10, 1, 1)"
        )

    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        # The legacy query still reports the saved row…
        assert resolve_dataset_id(db, populated_db) == saved.dataset_id
        # …while the formal confirmation path must refuse it.
        with pytest.raises(Exception) as exc_info:
            from ashare_data import manifest as manifest_mod

            manifest_mod.confirm_dataset_identity(db, populated_db)
    assert saved.dataset_id[:12] in str(exc_info.value) or "daily_bar" in str(
        exc_info.value
    )


def test_a02_composite_identity_is_deterministic(populated_db):
    """A02 acceptance 2 (p26 I3/I4): the confirmed identity is deterministic
    for identical content (the composite v2-side coverage cases R3/R4/R5 are
    driven by the contract plan's fixtures)."""

    from ashare_data import manifest as manifest_mod

    confirmer = getattr(manifest_mod, "confirm_dataset_identity", None)
    assert confirmer is not None, (
        "A02 requires manifest.confirm_dataset_identity (composite identity)"
    )
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        first = confirmer(db, populated_db)
        again = confirmer(db, populated_db)
    assert first, "composite identity must be a non-empty string"
    assert first == again, "identical content must reproduce the same identity"


def test_a02_fast_capability_is_declared_not_implied(populated_db):
    """A02 acceptance 4 (p26 I8, R10): the verification modes must expose a
    machine-readable capability statement so 'fast' can never be presented
    as complete content confirmation."""

    from ashare_data import manifest as manifest_mod

    capability = getattr(manifest_mod, "verification_capability", None)
    assert capability is not None, (
        "A02 requires a machine-readable fast/full capability declaration "
        "(manifest.verification_capability)"
    )
    declared = capability("fast")
    assert declared["content_confirmation"] is False
    assert capability("full")["content_confirmation"] is True


def test_a02_composite_identity_covers_both_databases(populated_db):
    """A02 acceptance 2 (p26 I3, R3/R5): the composite identity is composed
    from BOTH databases' content ids through the single hash authority.

    The derivation lives in ``ashare_model.runspec`` (it must call the hash
    authority and ``ashare_data`` is the bottom §9 layer); the publication
    version and the coverage lists stay in ``ashare_data.manifest``."""

    from ashare_data import manifest as manifest_mod
    from ashare_model.runspec import composite_snapshot_id
    from dataclasses import replace

    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        primary = _manifest(db, populated_db)
    v2 = replace(primary, dataset_id="v2-" + primary.dataset_id)

    composite = composite_snapshot_id(primary, v2)
    assert composite and composite != primary.dataset_id
    assert composite_snapshot_id(primary, v2) == composite  # deterministic (I4)
    # Changing ONLY the v2 side must change the composite identity (R3/R5).
    assert (
        composite_snapshot_id(primary, replace(v2, dataset_id="v2-changed"))
        != composite
    )
    # A missing v2 side is representable and distinct (no silent fabrication).
    assert composite_snapshot_id(primary, None) != composite
    # The publication-protocol version is part of the identity (R15/R16).
    assert composite_snapshot_id(primary, v2, publication_version="0") != composite
    assert manifest_mod.SNAPSHOT_PUBLICATION_VERSION == "1"


def test_a02_composite_domain_coverage_is_explicit(populated_db):
    """A02 acceptance 2 (p26 I3/I3-EXT, R15): included and explicitly
    NOT-included input domains must be machine-readable, so no domain can be
    added or dropped silently."""

    from ashare_data import manifest as manifest_mod

    coverage = getattr(manifest_mod, "composite_snapshot_domains", None)
    assert coverage is not None, (
        "A02 requires manifest.composite_snapshot_domains() (I3-EXT)"
    )
    declared = coverage()
    included = set(declared["included"])
    not_included = set(declared["not_included"])
    assert included and not_included
    assert included.isdisjoint(not_included)
    for domain in (
        "v2:earnings_forecast_pit",
        "v2:earnings_express_pit",
        "v2:repurchase_event_pit",
        "v2:holder_number_pit",
        "v2:index_weight_pit(content)",
    ):
        assert domain in not_included, domain
    assert declared["publication_version"] == (
        manifest_mod.SNAPSHOT_PUBLICATION_VERSION,
    )


# --- A02 publication protocol (OB-1 fix + p26 §3/I7) ---------------------------
#
# RED block for the publication generations: the observed OB-1 defect was that
# ``latest_manifest`` ordered generations by ``created_at DESC, dataset_id
# DESC`` — a second-granularity timestamp whose tiebreak is content-arbitrary.
# Two manifests published within the same second could order the WRONG way,
# and ``ON CONFLICT (dataset_id) DO NOTHING`` made the wrong selection
# permanent (the database could never be confirmed again).  The publication
# protocol replaces timestamp ordering with an explicit monotonic generation
# (p26 §3/I7: one publication = one immutable generation).


def test_a02_publication_generation_beats_same_second_hex_tiebreak(populated_db):
    """OB-1 regression: a same-second stale row with a hex-greater
    dataset_id must never win over the newest publication generation.

    The fabricated stale row reproduces the exact OB-1 shape observed on
    the synthetic database in tests/test_industry_pit.py: a manifest row
    whose created_at equals the newest publication's and whose dataset_id
    hex-sorts above it.  The legacy ``ORDER BY created_at DESC,
    dataset_id DESC`` resolves that tie towards the stale row; the
    publication generation must decide instead (p26 §3/I7, I9)."""

    import json as _json

    from ashare_data.manifest import (
        confirm_dataset_identity,
        latest_publication,
        publish_snapshot,
    )

    with AshareDB(populated_db.duckdb_path) as db:
        publication = publish_snapshot(db, populated_db)
        # The stale same-second hex-greater row: the publication's own
        # manifest with its identity replaced by a hex-maximal id and saved
        # under the SAME created_at second (the OB-1 trap).
        stale_payload = _json.loads(
            _json.dumps(publication.manifest.to_dict(), sort_keys=True)
        )
        stale_payload["dataset_id"] = "f" * 64
        db.execute(
            "INSERT INTO dataset_manifest VALUES (?, ?, ?, ?) "
            "ON CONFLICT (dataset_id) DO NOTHING",
            [
                "f" * 64,
                publication.manifest.manifest_version,
                publication.manifest.created_at,
                _json.dumps(stale_payload, ensure_ascii=False, sort_keys=True),
            ],
        )
        latest = latest_manifest(db, populated_db)
        newest = latest_publication(db, populated_db)
    assert newest is not None
    assert newest.publication_id == publication.publication_id
    # The newest PUBLICATION decides, not the hex-greater legacy row.
    assert latest is not None
    assert latest.dataset_id == publication.dataset_id
    assert latest.dataset_id != "f" * 64

    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        # And the confirmation path consumes the same winner.
        assert (
            confirm_dataset_identity(db, populated_db)
            == publication.dataset_id
        )


def test_a02_republishing_restores_confirmability(populated_db):
    """The engineer's OB-1 trap, end to end: rewrite a manifest-covered
    table AFTER a publication, republish (routinely within the same
    created_at second), and the database must be confirmable again —
    republishing is never a dead end (p26 §3/I7)."""

    from ashare_data.manifest import (
        confirm_dataset_identity,
        latest_publication,
        publish_snapshot,
    )

    with AshareDB(populated_db.duckdb_path) as db:
        db.execute(
            "INSERT INTO sw_industry_index VALUES "
            "('801010', '新一代', '20240101', 100.0)"
        )
        first = publish_snapshot(db, populated_db)
        # A second generation of the same content lands in the same second:
        # the manifest library dedups (same dataset_id), the publication
        # ledger records a new generation (append-only events).
        second = publish_snapshot(db, populated_db)
        assert second.publication_id > first.publication_id
        assert second.dataset_id == first.dataset_id
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        assert (
            confirm_dataset_identity(db, populated_db) == second.dataset_id
        )
        assert (
            latest_publication(db, populated_db).publication_id
            == second.publication_id
        )


def test_a02_publication_rehashes_value_mutations_preserving_count_and_max(
    populated_db,
):
    """R2 (p26 §6, I8): a value mutation that preserves (count, max) must
    change the PUBLISHED identity.  The publication entry may never serve
    the stale partition hash from ``dataset_manifest_cache`` — publishing
    must re-hash (that is what makes a published generation
    content-confirmed)."""

    from ashare_data.manifest import publish_snapshot

    with AshareDB(populated_db.duckdb_path) as db:
        first = publish_snapshot(db, populated_db)
        # In-place value mutation: row count and MAX(trade_date) unchanged,
        # so the partition cache entry stays (count, max)-valid and would be
        # served to a cached build.
        db.execute(
            "UPDATE daily_bar SET close = close + 1.0 "
            "WHERE ts_code = '000001.SZ' AND trade_date = "
            "(SELECT MIN(trade_date) FROM daily_bar)"
        )
        second = publish_snapshot(db, populated_db)
    assert second.dataset_id != first.dataset_id
    assert second.publication_id > first.publication_id


def test_a02_publication_history_is_append_only(populated_db):
    """R11 (p26 §6, I9): publishing new generations never rewrites the
    existing manifest library rows or publication rows — history is
    append-only, and old identities stay exactly as they were."""

    from ashare_data.manifest import publish_snapshot

    def library_rows(db):
        return {
            str(row["dataset_id"]): (
                str(row["manifest_version"]),
                str(row["created_at"]),
                str(row["manifest"]),
            )
            for _, row in db.query(
                "SELECT dataset_id, manifest_version, created_at, manifest "
                "FROM dataset_manifest"
            ).iterrows()
        }

    def publication_rows(db):
        return {
            (int(row["publication_id"]), str(row["member_library"])): (
                str(row["dataset_id"]),
                str(row["manifest"]),
            )
            for _, row in db.query(
                "SELECT publication_id, member_library, dataset_id, manifest "
                "FROM snapshot_publication"
            ).iterrows()
        }

    with AshareDB(populated_db.duckdb_path) as db:
        first = publish_snapshot(db, populated_db)
        before_library = library_rows(db)
        before_publications = publication_rows(db)
        db.execute(
            "INSERT INTO daily_bar VALUES "
            "('000001.SZ', '20240301', 10, 10, 10, 10, 10, 1, 10, 1, 1)"
        )
        second = publish_snapshot(db, populated_db)
        after_library = library_rows(db)
        after_publications = publication_rows(db)

    # Every pre-existing row is byte-identical; new rows only appear.
    for dataset_id, row in before_library.items():
        assert after_library[dataset_id] == row
    assert second.dataset_id in after_library
    assert second.dataset_id not in before_library or (
        after_library[second.dataset_id] == before_library[second.dataset_id]
    )
    for key, row in before_publications.items():
        assert after_publications[key] == row
    assert (
        second.publication_id,
        "primary",
    ) not in before_publications
    assert len(after_publications) == len(before_publications) + 1


def test_a02_publications_expose_the_protocol_generation(populated_db):
    """The publication record carries the protocol identity (member
    library, publication version, monotonic id) so a generation is
    attributable and machine-classifiable (p26 §3/I7, I12)."""

    from ashare_data import manifest as manifest_mod
    from ashare_data.manifest import publish_snapshot

    with AshareDB(populated_db.duckdb_path) as db:
        publication = publish_snapshot(db, populated_db)
        assert publication.member_library == "primary"
        assert publication.publication_id >= 1
        assert (
            publication.publication_version
            == manifest_mod.SNAPSHOT_PUBLICATION_VERSION
        )
        assert publication.manifest_version == manifest_mod.MANIFEST_VERSION
        assert publication.manifest.dataset_id == publication.dataset_id
