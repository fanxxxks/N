"""P2-01 contract tests: fundamentals-table scope.

Contract (``docs/p2_data_tier_contract.md`` §3): the ``fundamental_pit``
table only ever holds rows whose ``ts_code`` is a valid A-share code AND
belongs to the persisted stock scope (``stocks`` table ∪ PIT
``constituents`` ∪ cached daily-bar codes).  Non-stock instruments and
misattributed ``.BJ`` codes (NEEQ/delisted symbols mapped by prefix) can
never enter the table; the ``stocks`` table itself rejects non-A-share
codes on upsert; the scope report/purge path migrates existing dirty
rows with measured before/after counts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ashare_data import config as config_module
from ashare_data.akshare_client import AkShareClient
from ashare_data.db import AshareDB
from ashare_data.fundamentals import (
    fundamental_scope_stats,
    persisted_scope_codes,
    purge_out_of_scope_fundamentals,
    sync_fundamentals,
)


def _data_config(tmp_path: Path) -> config_module.DataConfig:
    return config_module.DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )


def _earnings_frame() -> pd.DataFrame:
    rows = []
    for code in ("000001.SZ", "600000.SH", "430047.BJ", "830001.BJ",
                 "900935.SH", "000300.SZ"):
        rows.append(
            {
                "ts_code": code,
                "report_date": "20230331",
                "announce_date": "20230430",
                "eps_cum": 1.0,
                "bvps": 8.0,
                "roe": 12.0,
                "gross_margin": 30.0,
                "net_margin": 10.0,
                "revenue_cum": 1e8,
                "profit_cum": 1e7,
                "revenue_yoy": 5.0,
                "profit_yoy": 6.0,
            }
        )
    return pd.DataFrame(rows)


class _FakeClient:
    """Offline-free client stub: the whole-market earnings endpoint only.

    P13 (whitelist §10.1 case 2, docs/p13_fundamental_fields_contract.md):
    sync_fundamentals now also binds the cash-flow/balance-sheet bulk
    endpoints (fundamentals.py), so the stub must carry them -- returning
    empty frames keeps these scope tests exactly as narrow as before
    (assertion strength unchanged; coverage only extended to the new
    binding seam)."""

    offline = False

    def __init__(self, earnings: pd.DataFrame):
        self._earnings = earnings

    def get_earnings_report(self, quarter: str) -> pd.DataFrame:
        return self._earnings.copy()

    def get_cash_flow_report(self, quarter: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_balance_sheet(self, quarter: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_financial_indicator(self, ts_code: str, start_year: int) -> pd.DataFrame:
        return pd.DataFrame()

    def get_dividend_detail(self, ts_code: str) -> pd.DataFrame:
        return pd.DataFrame()


def test_sync_fundamentals_restricts_bulk_rows_to_universe(tmp_path: Path):
    """Bulk earnings rows outside the sync universe (BJ/NEEQ codes,
    B-shares, index symbols) must never be upserted (P2-01)."""
    cfg = _data_config(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    stats = sync_fundamentals(
        _FakeClient(_earnings_frame()),
        db,
        cfg,
        universe=["000001.SZ", "600000.SH"],
    )
    assert stats["fundamental_rows"] > 0
    stored = {
        str(r[0])
        for r in db.query(
            f"SELECT DISTINCT ts_code FROM {cfg.fundamentals_table}"
        ).itertuples(index=False)
    }
    assert stored == {"000001.SZ", "600000.SH"}
    # The parquet cache holds only in-scope rows too.
    cached = list((cfg.parquet_dir / "fundamental").glob("earnings_*.parquet"))
    assert cached, "earnings cache should exist after the sync"
    for path in cached:
        codes = set(pd.read_parquet(path)["ts_code"].astype(str))
        assert codes <= {"000001.SZ", "600000.SH"}


def test_upsert_stocks_rejects_non_a_share_codes(tmp_path: Path):
    cfg = _data_config(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_stocks(
        [
            {"ts_code": "000001.SZ", "name": "平安银行", "list_date": "19910403"},
            {"ts_code": "900935.SH", "name": "900935.SH", "list_date": "19950727"},
            {"ts_code": "000300.SZ", "name": "沪深300", "list_date": None},
        ],
        cfg,
    )
    stored = {
        str(r[0])
        for r in db.query(f"SELECT DISTINCT ts_code FROM {cfg.stocks_table}").itertuples(index=False)
    }
    assert stored == {"000001.SZ"}


def test_persisted_scope_codes_unions_stocks_constituents_and_cached_bars(
    tmp_path: Path,
):
    cfg = _data_config(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_stocks(
        [{"ts_code": "000001.SZ", "name": "a", "list_date": None},
         {"ts_code": "900935.SH", "name": "b", "list_date": None}],
        cfg,
    )
    db.upsert_constituents(
        [
            {"index_code": "000300.SH", "ts_code": "600000.SH",
             "in_date": "20230101", "out_date": "20231231"},
        ],
        cfg,
    )
    daily = cfg.parquet_dir / "daily"
    daily.mkdir(parents=True)
    pd.DataFrame().to_parquet(daily / "601398.SH.parquet")
    scope = persisted_scope_codes(db, cfg)
    assert scope == {"000001.SZ", "600000.SH", "601398.SH"}


def test_scope_stats_and_purge_move_out_of_scope_rows(tmp_path: Path):
    cfg = _data_config(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_stocks(
        [{"ts_code": "000001.SZ", "name": "a", "list_date": None}], cfg
    )
    rows = [
        {"ts_code": "000001.SZ", "report_date": "20230331",
         "announce_date": "20230430"},
        {"ts_code": "430047.BJ", "report_date": "20230331",
         "announce_date": "20230430"},
        {"ts_code": "900935.SH", "report_date": "20230331",
         "announce_date": "20230430"},
    ]
    db.upsert_fundamentals(rows, cfg)

    stats = fundamental_scope_stats(db, cfg)
    assert stats["total_rows"] == 3
    assert stats["in_scope_rows"] == 1
    assert stats["out_of_scope_rows"] == 2
    assert stats["out_of_scope_codes"] == ["430047.BJ", "900935.SH"]

    result = purge_out_of_scope_fundamentals(db, cfg)
    assert result["removed_rows"] == 2
    assert result["rows_before"] == 3 and result["rows_after"] == 1
    remaining = {
        str(r[0])
        for r in db.query(
            f"SELECT DISTINCT ts_code FROM {cfg.fundamentals_table}"
        ).itertuples(index=False)
    }
    assert remaining == {"000001.SZ"}
    # Idempotent: a second purge removes nothing.
    again = purge_out_of_scope_fundamentals(db, cfg)
    assert again["removed_rows"] == 0


def test_scope_stats_with_no_fundamentals_table(tmp_path: Path):
    cfg = _data_config(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    stats = fundamental_scope_stats(db, cfg)
    assert stats["total_rows"] == 0
    assert stats["out_of_scope_rows"] == 0


def test_cli_purge_writes_audit_artifact(tmp_path: Path):
    """The scope CLI's --purge writes the measured before/after audit to
    data/fundamental_scope.json and leaves the table clean."""
    import json as _json
    import subprocess
    import sys

    import yaml

    cfg = _data_config(tmp_path)
    db = AshareDB(cfg.duckdb_path)
    db.create_schema(cfg)
    db.upsert_stocks(
        [{"ts_code": "000001.SZ", "name": "a", "list_date": None}],
        cfg,
    )
    # Simulate a legacy B-share row: the upsert validator already rejects
    # it, so legacy data is inserted directly (as in pre-P2-01 databases).
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
    # Release the file lock: DuckDB allows one writer connection, and the
    # subprocess below opens the same database read-write.
    db.close()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(cfg.data_dir),
                "duckdb_path": str(cfg.duckdb_path),
                "parquet_dir": str(cfg.parquet_dir),
                "index_codes": ["000300.SH"],
                "min_listed_sessions": 1,
            }
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "fundamental_scope.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_fundamental_scope.py",
            "--config",
            str(config_path),
            "--purge",
            "--output",
            str(audit_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    payload = _json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["scope"]["total_rows"] == 2
    assert payload["scope"]["out_of_scope_rows"] == 1
    assert payload["purge"]["removed_rows"] == 1
    assert payload["purge"]["rows_after"] == 1
    assert payload["stocks_purge"] == {
        "removed_rows": 1, "rows_before": 2, "rows_after": 1,
    }
    with AshareDB(cfg.duckdb_path) as check:
        remaining = {
            str(r[0])
            for r in check.query(
                f"SELECT DISTINCT ts_code FROM {cfg.fundamentals_table}"
            ).itertuples(index=False)
        }
        stocks = {
            str(r[0])
            for r in check.query(
                f"SELECT DISTINCT ts_code FROM {cfg.stocks_table}"
            ).itertuples(index=False)
        }
    assert remaining == {"000001.SZ"}
    assert stocks == {"000001.SZ"}
