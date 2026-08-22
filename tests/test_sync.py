from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import ashare_data.sync as sync
from ashare_data.config import DataConfig
from ashare_data.db import AshareDB


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )


def test_parquet_cache_path(tmp_path: Path):
    cfg = _config(tmp_path)
    assert sync._parquet_cache_path(cfg, "000001.SZ") == tmp_path / "parquet" / "daily" / "000001.SZ.parquet"


def test_parquet_cache_roundtrip(tmp_path: Path):
    cfg = _config(tmp_path)
    df = pd.DataFrame({"trade_date": ["20240101"], "close": [10.0]})
    sync._write_cached_bars(cfg, "000001.SZ", df)
    out = sync._read_cached_bars(cfg, "000001.SZ")
    assert out.equals(df)


def test_read_cached_bars_missing_and_corrupt(tmp_path: Path):
    cfg = _config(tmp_path)
    assert sync._read_cached_bars(cfg, "missing").empty
    path = sync._parquet_cache_path(cfg, "bad")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a parquet", encoding="utf-8")
    assert sync._read_cached_bars(cfg, "bad").empty


def test_calendar_stock_and_bar_rows():
    assert sync._calendar_rows(["20240101"]) == [{"trade_date": "20240101", "is_open": True}]
    stocks = pd.DataFrame(
        [{"ts_code": "000001.SZ", "name": "ST 风险", "industry": None, "list_date": None}]
    )
    rows = sync._stock_rows(stocks)
    assert rows[0]["is_st"] is True

    bars = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240101",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1000.0,
            }
        ]
    )
    bar_rows = sync._bar_rows(bars)
    assert len(bar_rows) == 1
    assert bar_rows[0]["pre_close"] == 10.0


def test_sync_all_offline(tmp_path: Path):
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
    assert result["calendar_days"] == 4
    assert result["stocks"] == 2
    # 300001.SZ appears in the constituents fixture but not in the stock
    # list fixture: the authoritative stock-list intersection removes it, so
    # it is neither synced nor attempted.
    assert result["universe"] == 2
    assert result["constituent_snapshot_symbols"] >= 2
    assert result["pit_constituent_rows_written"] == 0
    assert result["daily_rows"] > 0
    assert result["failures"] == []

    with AshareDB(tmp_path / "data" / "ashare.duckdb") as db:
        assert db.query("SELECT COUNT(*) AS n FROM daily_bar").iloc[0]["n"] > 0
        codes = set(db.query("SELECT DISTINCT ts_code FROM daily_bar")["ts_code"])
        assert codes == {"000001.SZ", "600000.SH"}
        assert db.query("SELECT COUNT(*) AS n FROM constituents").iloc[0]["n"] == 0


def test_purge_stale_daily_rows(tmp_path: Path):
    cfg = _config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_daily(
            [
                {
                    "ts_code": c,
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "pre_close": 9.9,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
                for c in ("000001.SZ", "600000.SH", "000300.SZ")
            ],
            cfg,
        )
        removed = sync._purge_stale_daily_rows(
            db, cfg, ["000001.SZ"], failures=["600000.SH"]
        )
        assert removed == 1  # only the index code row is removed
        remaining = db.query("SELECT ts_code FROM daily_bar")["ts_code"].tolist()
        assert set(remaining) == {"000001.SZ", "600000.SH"}
