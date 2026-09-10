"""p21 §3 (amended 2026-09-08): loader-side E2 factor-cache wiring tests.

Assertion source: the approved p21 §3 amendment (captain, 89dd121
authorization) — the E2 content-addressed factor-tensor cache is wired
into ``AshareDataLoader.load_data`` with a fail-closed identity derived
from store state; a disabled identity leaves the load path
byte-identical to pre-wiring behavior, and a hit reproduces the exact
tensor the miss path would have computed.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from ashare_data.config import ModelConfig, resolve_v2_duckdb_path
from ashare_model.data_loader import AshareDataLoader
from ashare_model.load_cache import loader_factor_cache_identity


@pytest.fixture
def identity_store(tmp_path):
    """A minimal v2 store carrying every identity-source table."""

    path = tmp_path / "v2.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE source_ingest_runs (ingest_id VARCHAR)")
    con.execute("INSERT INTO source_ingest_runs VALUES ('ing-1'), ('ing-2')")
    con.execute(
        "CREATE TABLE industry_membership_pit (available_date VARCHAR)"
    )
    con.execute("INSERT INTO industry_membership_pit VALUES ('20260821')")
    for table in (
        "income_statement_pit",
        "balance_sheet_pit",
        "cashflow_statement_pit",
    ):
        con.execute(
            f"CREATE TABLE {table} (available_date VARCHAR)"
        )
        con.execute(f"INSERT INTO {table} VALUES ('20260630')")
    con.close()
    return path


DATES = ["20240102", "20240103", "20240104"]
CODES = ["000001.SZ", "600000.SH"]


def test_identity_binds_every_store_dimension(identity_store):
    """The derivation must react to every store-state dimension: the
    ingest set, the industry membership generation, the financial
    generations, the loaded window and the universe."""

    base = loader_factor_cache_identity(
        identity_store, "ds-1", DATES, CODES
    )
    assert base is not None
    assert base.dataset_id == "ds-1"
    assert base.window_id == "20240102..20240104:3"
    assert set(base.source_snapshot_ids) == {"ing-1", "ing-2"}
    assert base.factor_compute_version and base.feature_registry_version
    assert base.industry_membership_version and base.financial_pit_version
    assert base.universe_hash

    # A different ingest set (one more snapshot) must change the identity.
    con = duckdb.connect(str(identity_store))
    con.execute("INSERT INTO source_ingest_runs VALUES ('ing-3')")
    con.close()
    grown = loader_factor_cache_identity(identity_store, "ds-1", DATES, CODES)
    assert grown is not None and grown != base

    # A different dataset id / window / universe each change the identity.
    assert (
        loader_factor_cache_identity(identity_store, "ds-2", DATES, CODES)
        != base
    )
    assert (
        loader_factor_cache_identity(
            identity_store, "ds-1", DATES[:-1], CODES
        )
        != base
    )
    assert (
        loader_factor_cache_identity(
            identity_store, "ds-1", DATES, CODES[:1]
        )
        != base
    )


def test_identity_none_when_ingest_source_missing(tmp_path):
    """A store without ``source_ingest_runs`` cannot support the
    identity — the derivation returns None (cache disabled), never an
    approximated key."""

    path = tmp_path / "bare.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE daily_bar_raw (trade_date VARCHAR)")
    con.close()
    assert (
        loader_factor_cache_identity(path, "ds-1", DATES, CODES) is None
    )


def test_identity_none_when_ingest_source_empty(identity_store):
    con = duckdb.connect(str(identity_store))
    con.execute("DELETE FROM source_ingest_runs")
    con.close()
    assert (
        loader_factor_cache_identity(identity_store, "ds-1", DATES, CODES)
        is None
    )


def _insert_ingest_row(con, ingest_id: str) -> None:
    """Insert one ledger row into the REAL source_ingest_runs table (the
    publication schema creates it with the full p17 §4.1 columns; the
    identity derivation only reads ingest_id)."""

    con.execute(
        "INSERT INTO source_ingest_runs (ingest_id, provider, endpoint, "
        "parameters_json, fields_json, started_at, status, row_count, "
        "raw_payload_sha256, raw_path, code_commit, sdk_version, "
        "hit_row_limit) VALUES (?, 'test', 'test', '{}', '{}', "
        "'2026-01-01 00:00:00', 'succeeded', 0, ?, 'test', 'test', "
        "'test', false)",
        [ingest_id, ingest_id],
    )


def test_loader_load_reuses_cached_tensor(populated_db):
    """Integration: with every identity source present in the toy v2
    store, the first load is a miss (tensor computed + stored) and the
    second load is a hit returning the identical tensor."""

    cfg = populated_db
    v2_path = str(resolve_v2_duckdb_path(cfg))
    con = duckdb.connect(v2_path)
    _insert_ingest_row(con, "toy-ing-1")

    def ensure_marker_table(table: str) -> None:
        """Create a minimal marker table only when the fixture's v2 store
        does not already carry the real one (the derivation reads
        COUNT/MAX(available_date), so an existing table needs no rows)."""

        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema='main' AND table_name = ?",
            [table],
        ).fetchone()[0]
        if not exists:
            con.execute(
                f"CREATE TABLE {table} (available_date VARCHAR)"
            )
            con.execute(f"INSERT INTO {table} VALUES ('20260101')")

    for table in (
        "industry_membership_pit",
        "income_statement_pit",
        "balance_sheet_pit",
        "cashflow_statement_pit",
    ):
        ensure_marker_table(table)
    con.close()
    # A02 (p26 I2/I7): seeding the v2 identity sources moved the v2 content
    # on from the fixture's published generation, so the composite
    # generation is republished before the formal load (production
    # discipline: drift is refused, never silently absorbed).
    from tests.conftest import publish_manifest

    publish_manifest(cfg)

    first = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert first.factor_cache_outcome == "miss"
    first_tensor = first.factor_tensor.numpy().copy()

    # The miss stored exactly one content-addressed entry.
    cache_files = list((cfg.data_dir / "factor_tensor_cache").glob("*.npz"))
    assert len(cache_files) == 1

    second = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert second.factor_cache_outcome == "hit"
    assert np.array_equal(second.factor_tensor.numpy(), first_tensor)

    # The hit wrote nothing new: one content-addressed entry serves every
    # identical-identity load.
    assert (
        len(list((cfg.data_dir / "factor_tensor_cache").glob("*.npz"))) == 1
    )


def test_a02_cache_key_shares_the_composite_identity_authority(populated_db):
    """R9 (p26 §6, I6, A02 acceptance 8): the factor-cache identity's
    dataset element and the RunSpec's dataset identity are the SAME
    composite string — the loader's confirmed binding, never a second
    near-identity assembled separately by the cache layer."""

    from ashare_data.db import AshareDB
    from ashare_model.runspec import confirm_composite_snapshot
    from tests.conftest import publish_manifest

    cfg = populated_db
    v2_path = str(resolve_v2_duckdb_path(cfg))
    con = duckdb.connect(v2_path)
    _insert_ingest_row(con, "r9-ing-1")
    con.close()
    publish_manifest(cfg)

    loaded = AshareDataLoader(cfg, ModelConfig()).load_data()
    with AshareDB(cfg.duckdb_path, read_only=True) as db, AshareDB(
        v2_path, read_only=True
    ) as v2:
        confirmed = confirm_composite_snapshot(db, cfg, v2)
    # The loader's binding IS the composite derivation (one authority).
    assert loaded.dataset_id == confirmed.composite_id
    assert loaded.dataset_id != confirmed.primary.dataset_id

    identity = loader_factor_cache_identity(
        v2_path, loaded.dataset_id, loaded.dates, loaded.ts_codes
    )
    assert identity is not None
    # The cache key's dataset element is that same string — not a
    # separately-derived primary id or a re-hashed variant.
    assert identity.dataset_id == loaded.dataset_id
    assert identity.dataset_id != confirmed.primary.dataset_id


def test_loader_cache_disabled_without_identity_sources(populated_db):
    """Without the identity sources the load path is exactly pre-wiring:
    outcome ``disabled`` and no cache file is written."""

    loader = AshareDataLoader(populated_db, ModelConfig()).load_data()
    assert loader.factor_cache_outcome == "disabled"
    assert not (populated_db.data_dir / "factor_tensor_cache").exists()
