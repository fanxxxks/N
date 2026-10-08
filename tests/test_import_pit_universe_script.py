"""Offline decision-logic tests for the destructive-rebuild safety added
to ``scripts/import_pit_universe.py`` and
``scripts/check_fundamental_scope.py`` (P1 + P2 review follow-up).

The scripts' happy paths need BaoStock/AkShare network access, so these
tests exercise only the extracted, dependency-free decision functions:

* ``rebuild_constituents_table`` — a dry run must not touch the database;
  the real run must back up the deleted rows before the full-table
  replace and abort (fail closed) when the backup cannot be written.
* ``run_purge`` — the same backup-first / fail-closed contract for the
  fundamental-scope purge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import check_fundamental_scope as scope_script  # noqa: E402
import import_pit_universe as pit_script  # noqa: E402

from ashare_data.config import DataConfig  # noqa: E402
from ashare_data.db import AshareDB  # noqa: E402
from ashare_data.fundamentals import persisted_scope_codes  # noqa: E402


def _cfg(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )


_OLD_CONSTITUENTS = [
    {"index_code": "000300.SH", "ts_code": "000001.SZ",
     "in_date": "20230101", "out_date": "99991231"},
    {"index_code": "000300.SH", "ts_code": "600999.SH",
     "in_date": "20200101", "out_date": "20221231"},
]

_NEW_CONSTITUENTS = [
    {"index_code": "000300.SH", "ts_code": "600000.SH",
     "in_date": "20230101", "out_date": "99991231"},
]


def _seed_dirty_scope_tables(cfg: DataConfig) -> AshareDB:
    """A pre-P2-01 database: one valid stock, one legacy B-share row in
    ``stocks`` (inserted directly, as the upsert validator rejects it),
    and one out-of-scope fundamentals row."""
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_stocks([{"ts_code": "000001.SZ", "name": "a", "list_date": None}],
                     cfg)
    db.execute(
        "INSERT INTO stocks (ts_code, name, list_date, is_st) "
        "VALUES ('900935.SH', 'b', '19950727', FALSE)"
    )
    db.upsert_fundamentals(
        [
            {"ts_code": "000001.SZ", "report_date": "20230331",
             "announce_date": "20230430"},
            {"ts_code": "430047.BJ", "report_date": "20230331",
             "announce_date": "20230430"},
        ],
        cfg,
    )
    return db


def _codes(db: AshareDB, table: str) -> set[str]:
    return {
        str(r[0])
        for r in db.query(f"SELECT DISTINCT ts_code FROM {table}").itertuples(
            index=False
        )
    }


# --------------------------------------------------------------------------
# import_pit_universe.rebuild_constituents_table
# --------------------------------------------------------------------------


def test_rebuild_constituents_dry_run_leaves_database_untouched(
    tmp_path: Path,
):
    cfg = _cfg(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_constituents(_OLD_CONSTITUENTS, cfg)
    before = {
        tuple(r)
        for r in db.query(
            "SELECT index_code, ts_code, in_date, out_date FROM constituents"
        ).itertuples(index=False, name=None)
    }
    db.close()
    backup_dir = tmp_path / "backups"
    # A read-only connection proves the dry run issues no write at all.
    with AshareDB(cfg.duckdb_path, read_only=True) as rodb:
        outcome = pit_script.rebuild_constituents_table(
            rodb, cfg, _NEW_CONSTITUENTS, dry_run=True, backup_dir=backup_dir
        )
    assert outcome["existing_rows"] == 2
    assert outcome["inserted_rows"] == 1
    assert list(backup_dir.glob("*.json")) == []
    with AshareDB(cfg.duckdb_path) as check:
        after = {
            tuple(r)
            for r in check.query(
                "SELECT index_code, ts_code, in_date, out_date "
                "FROM constituents"
            ).itertuples(index=False, name=None)
        }
    assert after == before


def test_rebuild_constituents_backs_up_deleted_rows_before_replace(
    tmp_path: Path,
):
    cfg = _cfg(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_constituents(_OLD_CONSTITUENTS, cfg)
    backup_dir = tmp_path / "backups"

    outcome = pit_script.rebuild_constituents_table(
        db, cfg, _NEW_CONSTITUENTS, dry_run=False, backup_dir=backup_dir
    )
    files = list(backup_dir.glob("constituents_backup_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["table"] == "constituents"
    assert payload["row_count"] == 2
    assert {(r["ts_code"], r["in_date"]) for r in payload["rows"]} == {
        ("000001.SZ", "20230101"),
        ("600999.SH", "20200101"),
    }
    assert outcome["backed_up_rows"] == 2
    assert outcome["backup_path"] == str(files[0])
    # The table was replaced with exactly the imported intervals.
    assert _codes(db, "constituents") == {"600000.SH"}


def test_rebuild_constituents_aborts_when_backup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _cfg(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_constituents(_OLD_CONSTITUENTS, cfg)

    def _boom(path, payload):
        raise pit_script.BackupError("disk full")

    monkeypatch.setattr(pit_script, "write_backup_json", _boom)
    with pytest.raises(pit_script.BackupError):
        pit_script.rebuild_constituents_table(
            db, cfg, _NEW_CONSTITUENTS,
            dry_run=False, backup_dir=tmp_path / "backups",
        )
    # Fail-closed: the DELETE never ran.
    assert _codes(db, "constituents") == {"000001.SZ", "600999.SH"}


def test_write_backup_json_fails_closed_on_unwritable_path(tmp_path: Path):
    blocker = tmp_path / "blocked.json"
    blocker.mkdir()  # a directory where the backup file should land
    with pytest.raises(pit_script.BackupError):
        pit_script.write_backup_json(
            blocker, {"table": "constituents", "rows": []}
        )


# --------------------------------------------------------------------------
# check_fundamental_scope.run_purge
# --------------------------------------------------------------------------


def test_scope_purge_backs_up_deleted_rows(tmp_path: Path):
    cfg = _cfg(tmp_path)
    db = _seed_dirty_scope_tables(cfg)
    scope = persisted_scope_codes(db, cfg)
    backup_dir = tmp_path / "backups"

    result = scope_script.run_purge(
        db, cfg, scope=scope, invalid_stocks=["900935.SH"],
        backup_dir=backup_dir,
    )
    files = list(backup_dir.glob("fundamental_purge_backup_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert {r["ts_code"] for r in payload["fundamental_pit"]["rows"]} == {
        "430047.BJ"
    }
    assert {r["ts_code"] for r in payload["stocks"]["rows"]} == {"900935.SH"}
    assert payload["invalid_stocks_codes"] == ["900935.SH"]
    assert result["purge"]["removed_rows"] == 1
    assert result["stocks_purge"]["removed_rows"] == 1
    assert result["backup_rows"] == {"fundamental_pit": 1, "stocks": 1}
    assert _codes(db, cfg.fundamentals_table) == {"000001.SZ"}
    assert _codes(db, cfg.stocks_table) == {"000001.SZ"}


def test_scope_purge_aborts_when_backup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = _cfg(tmp_path)
    db = _seed_dirty_scope_tables(cfg)
    scope = persisted_scope_codes(db, cfg)

    def _boom(path, payload):
        raise scope_script.BackupError("disk full")

    monkeypatch.setattr(scope_script, "write_backup_json", _boom)
    with pytest.raises(scope_script.BackupError):
        scope_script.run_purge(
            db, cfg, scope=scope, invalid_stocks=["900935.SH"],
            backup_dir=tmp_path / "backups",
        )
    # Fail-closed: neither the fundamentals purge nor the stocks deletes
    # ran.
    assert _codes(db, cfg.fundamentals_table) == {"000001.SZ", "430047.BJ"}
    assert _codes(db, cfg.stocks_table) == {"000001.SZ", "900935.SH"}
