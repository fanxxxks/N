"""DuckDB persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import duckdb


def sql_quoted_list(values: Iterable[Any]) -> str:
    """A properly quoted SQL ``IN (...)`` fragment (one code path for all
    hand-rolled list quoting in the project)."""

    return ",".join(f"'{value}'" for value in values)


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
                PRIMARY KEY (index_code, ts_code, in_date)
            )
            """
        )
        self._ensure_constituents_primary_key(config)
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
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.fundamentals_table} (
                ts_code VARCHAR,
                report_date VARCHAR,
                announce_date VARCHAR,
                dividend_announce VARCHAR,
                eps_cum DOUBLE,
                bvps DOUBLE,
                roe DOUBLE,
                roa DOUBLE,
                gross_margin DOUBLE,
                net_margin DOUBLE,
                revenue_cum DOUBLE,
                profit_cum DOUBLE,
                revenue_yoy DOUBLE,
                profit_yoy DOUBLE,
                debt_ratio DOUBLE,
                dividend_yield DOUBLE,
                net_operate_cash_flow DOUBLE,
                total_assets DOUBLE,
                PRIMARY KEY (ts_code, report_date)
            )
            """
        )
        # P13 additive migration (docs/p13_fundamental_fields_contract.md
        # §5.1): CREATE TABLE IF NOT EXISTS cannot add columns to a
        # pre-P13 table, so the two family-⑤ fields are added idempotently
        # on every create_schema call.  Existing rows keep their values and
        # read NULL in the new columns until the P13 backfill fills them.
        for _p13_column in ("net_operate_cash_flow", "total_assets"):
            self.execute(
                f"ALTER TABLE {config.fundamentals_table} "
                f"ADD COLUMN IF NOT EXISTS {_p13_column} DOUBLE"
            )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.margin_table} (
                ts_code VARCHAR,
                trade_date VARCHAR,
                rzye DOUBLE,
                PRIMARY KEY (ts_code, trade_date)
            )
            """
        )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.sw_index_table} (
                index_code VARCHAR,
                industry_name VARCHAR,
                trade_date VARCHAR,
                close DOUBLE,
                PRIMARY KEY (index_code, trade_date)
            )
            """
        )
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.sw_member_table} (
                index_code VARCHAR,
                ts_code VARCHAR,
                PRIMARY KEY (index_code, ts_code)
            )
            """
        )

    def upsert_stocks(self, rows: list[dict[str, Any]], config) -> None:
        """Merge-upsert the stock snapshot: update overlapping rows, insert
        new ones, never delete.

        The exchange snapshot is not a PIT universe.  Rows absent from the
        current snapshot (delisted stocks, PIT-import backfills) must
        survive a routine ``sync_all``, or the production contract restored
        by ``scripts/import_pit_universe.py`` is silently erased.  A
        backfilled ``list_date`` is preserved whenever the snapshot carries
        none; a non-null snapshot value wins for currently listed stocks.
        Non-A-share codes (B-shares, index symbols) are rejected on upsert
        (P2-01): the stocks table is the persisted stock scope and must
        never accumulate instruments the pipeline cannot trade.
        """
        if not rows:
            return
        import pandas as pd

        from .processor import is_valid_a_share_code

        rows = [
            row
            for row in rows
            if is_valid_a_share_code(str(row.get("ts_code") or ""))
        ]
        if not rows:
            return

        columns = ["ts_code", "name", "industry", "list_date", "is_st"]
        df = pd.DataFrame(rows).reindex(columns=columns)
        self._conn.register("_stocks_df", df)
        self.execute("BEGIN TRANSACTION")
        try:
            self.execute(
                f"""
                UPDATE {config.stocks_table} s SET
                    name = d.name,
                    industry = d.industry,
                    list_date = COALESCE(
                        NULLIF(CAST(d.list_date AS VARCHAR), ''), s.list_date
                    ),
                    is_st = d.is_st
                FROM _stocks_df d
                WHERE s.ts_code = d.ts_code
                """
            )
            self.execute(
                f"""
                INSERT INTO {config.stocks_table}
                SELECT d.ts_code, d.name, d.industry, d.list_date, d.is_st
                FROM _stocks_df d
                WHERE NOT EXISTS (
                    SELECT 1 FROM {config.stocks_table} s
                    WHERE s.ts_code = d.ts_code
                )
                """
            )
            self.execute("COMMIT")
        except Exception:
            self.execute("ROLLBACK")
            raise

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

        from .universe import validate_membership_intervals

        df = validate_membership_intervals(pd.DataFrame(rows))
        existing = self.query(
            f"SELECT index_code, ts_code, in_date, out_date "
            f"FROM {config.constituents_table}"
        )
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["index_code", "ts_code", "in_date"], keep="last"
        )
        validate_membership_intervals(combined)
        self._conn.register("_cons_df", df)
        self.execute(
            f"""
            INSERT INTO {config.constituents_table}
            SELECT index_code, ts_code, in_date, out_date FROM _cons_df
            ON CONFLICT (index_code, ts_code, in_date) DO UPDATE SET
                out_date=EXCLUDED.out_date
            """
        )

    def _ensure_constituents_primary_key(self, config) -> None:
        """Migrate the legacy two-column PK without inventing interval data."""

        table = config.constituents_table
        constraints = self.query(
            """
            SELECT constraint_column_names
            FROM duckdb_constraints()
            WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'
            """,
            [table],
        )
        current: list[str] = []
        if not constraints.empty:
            value = constraints.iloc[0]["constraint_column_names"]
            current = [str(column) for column in value]
        expected = ["index_code", "ts_code", "in_date"]
        if current == expected:
            return

        migration_table = f"{table}__pit_pk_migration"
        self.execute("BEGIN TRANSACTION")
        try:
            self.execute(f"DROP TABLE IF EXISTS {migration_table}")
            self.execute(
                f"""
                CREATE TABLE {migration_table} (
                    index_code VARCHAR,
                    ts_code VARCHAR,
                    in_date VARCHAR,
                    out_date VARCHAR,
                    PRIMARY KEY (index_code, ts_code, in_date)
                )
                """
            )
            self.execute(
                f"""
                INSERT INTO {migration_table}
                SELECT index_code, ts_code, in_date, out_date FROM {table}
                """
            )
            self.execute(f"DROP TABLE {table}")
            self.execute(f"ALTER TABLE {migration_table} RENAME TO {table}")
            self.execute("COMMIT")
        except Exception:
            self.execute("ROLLBACK")
            raise

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

    def upsert_fundamentals(self, rows: list[dict[str, Any]], config) -> None:
        """Insert/update quarterly fundamental rows by (ts_code, report_date).

        NULL fields never overwrite existing values, so a dividend-only row
        cannot wipe the earnings fields of the same report period.  Each
        source owns its effective date: ``announce_date`` is written by the
        earnings/indicator rows and ``dividend_announce`` (the ex-dividend
        date) by the dividend rows, so merging sources never moves any
        field's point-in-time visibility.
        """

        if not rows:
            return
        import pandas as pd

        cols = [
            "ts_code",
            "report_date",
            "announce_date",
            "dividend_announce",
            "eps_cum",
            "bvps",
            "roe",
            "roa",
            "gross_margin",
            "net_margin",
            "revenue_cum",
            "profit_cum",
            "revenue_yoy",
            "profit_yoy",
            "debt_ratio",
            "dividend_yield",
            "net_operate_cash_flow",
            "total_assets",
        ]
        df = pd.DataFrame(rows).reindex(columns=cols)
        for date_col in ("announce_date", "dividend_announce"):
            df[date_col] = (
                df[date_col].astype("string").where(df[date_col].notna(), None)
            )
        self._conn.register("_fund_df", df)
        updates = ",\n".join(
            f"{c}=COALESCE(EXCLUDED.{c}, {config.fundamentals_table}.{c})"
            for c in cols[2:]
        )
        self.execute(
            f"""
            INSERT INTO {config.fundamentals_table}
            SELECT ts_code, report_date,
                   CAST(announce_date AS VARCHAR),
                   CAST(dividend_announce AS VARCHAR),
                   eps_cum, bvps, roe, roa, gross_margin, net_margin,
                   revenue_cum, profit_cum, revenue_yoy, profit_yoy,
                   debt_ratio, dividend_yield,
                   net_operate_cash_flow, total_assets
            FROM _fund_df
            ON CONFLICT (ts_code, report_date) DO UPDATE SET
                {updates}
            """
        )

    def upsert_margin(self, rows: list[dict[str, Any]], config) -> None:
        """Insert/update margin-financing balances by (ts_code, trade_date)."""

        if not rows:
            return
        import pandas as pd

        df = pd.DataFrame(rows)[["ts_code", "trade_date", "rzye"]]
        self._conn.register("_margin_df", df)
        self.execute(
            f"""
            INSERT INTO {config.margin_table}
            SELECT ts_code, trade_date, rzye FROM _margin_df
            ON CONFLICT (ts_code, trade_date) DO UPDATE SET rzye=EXCLUDED.rzye
            """
        )

    def upsert_sw_index(self, rows: list[dict[str, Any]], config) -> None:
        """Insert/update Shenwan industry index closes."""

        if not rows:
            return
        import pandas as pd

        df = pd.DataFrame(rows)[["index_code", "industry_name", "trade_date", "close"]]
        self._conn.register("_sw_index_df", df)
        self.execute(
            f"""
            INSERT INTO {config.sw_index_table}
            SELECT index_code, industry_name, trade_date, close FROM _sw_index_df
            ON CONFLICT (index_code, trade_date) DO UPDATE SET
                industry_name=EXCLUDED.industry_name,
                close=EXCLUDED.close
            """
        )

    def replace_sw_members(self, index_code: str, ts_codes: list[str], config) -> None:
        """Replace the (current-snapshot) member list of one industry index."""

        import pandas as pd

        self.execute(
            f"DELETE FROM {config.sw_member_table} WHERE index_code = ?",
            [index_code],
        )
        if not ts_codes:
            return
        df = pd.DataFrame({"index_code": index_code, "ts_code": ts_codes})
        self._conn.register("_sw_member_df", df)
        self.execute(
            f"""
            INSERT INTO {config.sw_member_table}
            SELECT index_code, ts_code FROM _sw_member_df
            """
        )
