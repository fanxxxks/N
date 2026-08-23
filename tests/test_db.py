from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_data.config import DataConfig
from ashare_data.db import AshareDB


def test_schema_is_created(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        tables = db.query(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        )
        names = set(tables["table_name"].astype(str))
        assert {
            data_config.stocks_table,
            data_config.daily_table,
            data_config.constituents_table,
            data_config.calendar_table,
            data_config.factor_table,
        }.issubset(names)


def test_upsert_stocks_merges_and_never_deletes(data_config: DataConfig):
    """H4 contract: a sync snapshot merges, never deletes.

    Rows absent from the current snapshot (delisted stocks, PIT-import
    backfills) survive a routine sync, and a backfilled list_date is
    preserved whenever the snapshot carries none — the production
    contract restored by scripts/import_pit_universe.py must not be
    silently erased by sync_all.
    """
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_stocks(
            [
                {"ts_code": "000001.SZ", "name": "A", "industry": None, "list_date": "20200101", "is_st": False},
                {"ts_code": "600000.SH", "name": "B", "industry": None, "list_date": "20200101", "is_st": False},
            ],
            data_config,
        )
        # Second snapshot: A renamed / re-ST'd with a missing list_date,
        # C is newly listed, B is absent (delisted / backfilled row).
        db.upsert_stocks(
            [
                {"ts_code": "000001.SZ", "name": "A2", "industry": "bank", "list_date": None, "is_st": True},
                {"ts_code": "300001.SZ", "name": "C", "industry": None, "list_date": "20240201", "is_st": False},
            ],
            data_config,
        )
        df = (
            db.query(f"SELECT * FROM {data_config.stocks_table}")
            .sort_values("ts_code")
            .reset_index(drop=True)
        )
        assert df["ts_code"].tolist() == ["000001.SZ", "300001.SZ", "600000.SH"]
        a = df[df["ts_code"] == "000001.SZ"].iloc[0]
        assert a["name"] == "A2" and a["industry"] == "bank"
        assert a["list_date"] == "20200101"  # preserved: snapshot had none
        assert bool(a["is_st"]) is True
        b = df[df["ts_code"] == "600000.SH"].iloc[0]
        assert b["name"] == "B" and b["list_date"] == "20200101"  # untouched
        c = df[df["ts_code"] == "300001.SZ"].iloc[0]
        assert c["list_date"] == "20240201"
        # A non-null snapshot list_date wins for a currently listed stock.
        db.upsert_stocks(
            [{"ts_code": "000001.SZ", "name": "A3", "industry": "bank", "list_date": "20190101", "is_st": False}],
            data_config,
        )
        df = db.query(
            f"SELECT list_date, name FROM {data_config.stocks_table} WHERE ts_code = '000001.SZ'"
        )
        assert df.iloc[0]["list_date"] == "20190101"
        assert df.iloc[0]["name"] == "A3"


def test_upsert_daily_uses_on_conflict_update(data_config: DataConfig):
    rows = [
        {
            "ts_code": "000001.SZ",
            "trade_date": "20240101",
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
    ]
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(rows, data_config)
        rows[0]["close"] = 99.0
        db.upsert_daily(rows, data_config)
        df = db.query(f"SELECT close FROM {data_config.daily_table}")
        assert len(df) == 1
        assert df.iloc[0]["close"] == 99.0


def test_upsert_calendar_constituents_and_factors(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_calendar([{"trade_date": "20240101", "is_open": True}], data_config)
        db.upsert_constituents(
            [{"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20240101", "out_date": "99991231"}],
            data_config,
        )
        db.upsert_factors(
            [{"ts_code": "000001.SZ", "trade_date": "20240101", "factor_name": "RET_1", "value": 0.1}],
            data_config,
        )
        assert db.query(f"SELECT COUNT(*) AS n FROM {data_config.calendar_table}").iloc[0]["n"] == 1
        assert db.query(f"SELECT COUNT(*) AS n FROM {data_config.constituents_table}").iloc[0]["n"] == 1
        assert db.query(f"SELECT COUNT(*) AS n FROM {data_config.factor_table}").iloc[0]["n"] == 1


def test_empty_upserts_are_noops(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_stocks([], data_config)
        db.upsert_daily([], data_config)
        db.upsert_constituents([], data_config)
        db.upsert_calendar([], data_config)
        db.upsert_factors([], data_config)
        assert db.query(f"SELECT COUNT(*) AS n FROM {data_config.stocks_table}").iloc[0]["n"] == 0
