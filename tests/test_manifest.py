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
    assert manifest.manifest_version == "1"
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


def test_save_latest_and_resolve_roundtrip(populated_db):
    with AshareDB(populated_db.duckdb_path) as db:
        manifest = _manifest(db, populated_db)
        save_manifest(db, populated_db, manifest)
        save_manifest(db, populated_db, manifest)  # idempotent
        assert latest_manifest(db, populated_db) == manifest
        assert resolve_dataset_id(db, populated_db) == manifest.dataset_id
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

    # The offline sync writes only snapshot constituents, so the loader
    # must use the explicit development universe fallback (strict mode
    # requires historical PIT intervals).
    loader = AshareDataLoader(
        cfg, ModelConfig(), allow_development_universe_fallback=True
    )
    loader.load_data()
    assert loader.dataset_id == dataset_id


def test_loader_exposes_dataset_id_when_absent(populated_db):
    from ashare_data.config import ModelConfig
    from ashare_model.data_loader import AshareDataLoader

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    assert loader.dataset_id is None  # no manifest saved yet


def test_partition_fingerprint_rejects_unknown_table(populated_db):
    with AshareDB(populated_db.duckdb_path, read_only=True) as db:
        with pytest.raises(ValueError):
            compute_partition_fingerprint(db, populated_db, "no_such_table", "ALL")
        with pytest.raises(ValueError):
            compute_table_fingerprint(db, populated_db, "no_such_table")
