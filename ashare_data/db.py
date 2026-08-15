"""DuckDB persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import duckdb


class AshareDB:
    """Small wrapper around a local DuckDB file."""

    def __init__(self, path: str | Path, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path), read_only=read_only)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "AshareDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def execute(self, sql: str, params: Iterable[Any] | None = None) -> None:
        if params is None:
            self._conn.execute(sql)
        else:
            self._conn.execute(sql, list(params))

    def query(self, sql: str, params: Iterable[Any] | None = None):
        if params is None:
            return self._conn.execute(sql).fetchdf()
        return self._conn.execute(sql, list(params)).fetchdf()

    def create_schema(self, config) -> None:
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.stocks_table} (
                ts_code VARCHAR PRIMARY KEY,
                name VARCHAR,
                industry VARCHAR,
                list_date VARCHAR,
                is_st BOOLEAN
            )
            """
        )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.daily_table} (
                ts_code VARCHAR,
                trade_date VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                pre_close DOUBLE,
                volume DOUBLE,
                amount DOUBLE,
                turnover_rate DOUBLE,
                adj_factor DOUBLE,
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.constituents_table} (
                index_code VARCHAR,
                ts_code VARCHAR,
                in_date VARCHAR,
                out_date VARCHAR,
                PRIMARY KEY (index_code, ts_code)
            )
            """
        )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.calendar_table} (
                trade_date VARCHAR PRIMARY KEY,
                is_open BOOLEAN
            )
            """
        )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.factor_table} (
                ts_code VARCHAR,
                trade_date VARCHAR,
                factor_name VARCHAR,
                value DOUBLE,
                PRIMARY KEY (ts_code, trade_date, factor_name)
            )
            """
        )

    def upsert_stocks(self, rows: list[dict[str, Any]], config) -> None:
        if not rows:
            return
        self.execute(f"DELETE FROM {config.stocks_table}")
        import pandas as pd

        df = pd.DataFrame(rows)
        self._conn.register("_stocks_df", df)
        self.execute(
            f"""
            INSERT INTO {config.stocks_table}
            SELECT ts_code, name, industry, list_date, is_st FROM _stocks_df
            """
        )

    def upsert_daily(self, rows: list[dict[str, Any]], config) -> None:
        if not rows:
            return
        import pandas as pd

        df = pd.DataFrame(rows)
        self._conn.register("_daily_df", df)
        self.execute(
            f"""
            INSERT INTO {config.daily_table}
            SELECT ts_code, trade_date, open, high, low, close, pre_close,
                   volume, amount, turnover_rate, adj_factor
            FROM _daily_df
            ON CONFLICT (ts_code, trade_date) DO UPDATE SET
                open=EXCLUDED.open,
                high=EXCLUDED.high,
                low=EXCLUDED.low,
                close=EXCLUDED.close,
                pre_close=EXCLUDED.pre_close,
                volume=EXCLUDED.volume,
                amount=EXCLUDED.amount,
                turnover_rate=EXCLUDED.turnover_rate,
                adj_factor=EXCLUDED.adj_factor
            """
        )

    def upsert_constituents(self, rows: list[dict[str, Any]], config) -> None:
        if not rows:
            return
        import pandas as pd

        df = pd.DataFrame(rows)
        self._conn.register("_cons_df", df)
        self.execute(
            f"""
            INSERT INTO {config.constituents_table}
            SELECT index_code, ts_code, in_date, out_date FROM _cons_df
            ON CONFLICT (index_code, ts_code) DO UPDATE SET
                in_date=EXCLUDED.in_date,
                out_date=EXCLUDED.out_date
            """
        )

    def upsert_calendar(self, rows: list[dict[str, Any]], config) -> None:
        if not rows:
            return
        import pandas as pd

        df = pd.DataFrame(rows)
        self._conn.register("_cal_df", df)
        self.execute(
            f"""
            INSERT INTO {config.calendar_table}
            SELECT trade_date, is_open FROM _cal_df
            ON CONFLICT (trade_date) DO UPDATE SET is_open=EXCLUDED.is_open
            """
        )

    def upsert_factors(self, rows: list[dict[str, Any]], config) -> None:
        if not rows:
            return
        import pandas as pd

        df = pd.DataFrame(rows)
        self._conn.register("_factor_df", df)
        self.execute(
            f"""
            INSERT INTO {config.factor_table}
            SELECT ts_code, trade_date, factor_name, value FROM _factor_df
            ON CONFLICT (ts_code, trade_date, factor_name) DO UPDATE SET
                value=EXCLUDED.value
            """
        )
