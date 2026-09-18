"""Shared pytest fixtures and session-level run logging."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import BacktestConfig, DataConfig
from ashare_data.db import AshareDB


def publish_manifest(data_config: DataConfig) -> str:
    """Publish the composite snapshot generation for a test database PAIR
    (A02 p26 I2/I7).

    A formal load consumes the *content-confirmed composite* identity, so a
    test that seeds its own databases must publish exactly as production
    does — one publication id carrying BOTH member manifests (primary +
    tushare_v2).  Returns the published ``composite_id``.

    The member manifests are built UNCACHED (``use_cache=False`` inside
    ``publish_composite_snapshot``): the confirmation path always
    recomputes, so a generation published from cached partition hashes
    can name content the confirm step then rejects (the OB-1 family).
    The v2 side is completed to the full 19-table store schema first so
    the member manifest can fingerprint every domain table.
    """

    from ashare_data.config import resolve_v2_duckdb_path
    from ashare_model.runspec import publish_composite_snapshot

    v2_path = ensure_v2_publication_schema(resolve_v2_duckdb_path(data_config))
    with AshareDB(data_config.duckdb_path) as primary, AshareDB(
        v2_path, read_only=True
    ) as v2:
        return publish_composite_snapshot(
            primary, data_config, v2
        ).composite_id


def ensure_v2_publication_schema(v2_path: Path) -> Path:
    """Create the COMPLETE 19-table v2 store schema (idempotent).

    The v2 dataset manifest fingerprints every domain table, so a synthetic
    v2 database used for composite publications / confirmations must carry
    all of them.  The DDL authority is ``TushareStore.create_schema`` (no
    second DDL copy here); ``CREATE TABLE IF NOT EXISTS`` leaves tables the
    scenario helpers already populated untouched.
    """

    from ashare_data.tushare_store import TushareStore

    v2_path = Path(v2_path)
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    with AshareDB(v2_path) as db:
        TushareStore(
            db,
            v2_path.parent,
            session_resolver=lambda date: date,
            code_commit="tests",
            sdk_version="tests",
        ).create_schema()
    return v2_path


def with_price_semantics(raw: dict[str, "np.ndarray"]) -> dict[str, "np.ndarray"]:
    """Attach the p19 §2 cache keys to a synthetic execution-price dict.

    The adjusted label basis mirrors the raw prices (no corporate actions
    in the synthetic data) and the exact constraint tables default to
    unconstrained; scenario tests override the specific cells they target.
    """

    open_ = np.asarray(raw["open"], dtype=np.float64)
    out = dict(raw)
    out.setdefault("adjusted_open", open_.copy())
    n, t = open_.shape
    out.setdefault("up_limit", np.full((n, t), 1e12))
    out.setdefault("down_limit", np.full((n, t), 1e-9))
    out.setdefault("suspended", np.zeros((n, t)))
    return out
from ashare_logging import (
    export_log_txt,
    guard_process_exit,
    setup_run_logging,
    worker_log_suffix,
)


DEFAULT_CODES = ["000001.SZ", "600000.SH", "300001.SZ"]

# Loader-facing column subset of the p17 §5 v2 price tables (DDL authority:
# ashare_data.tushare_store / test_tushare_store; the loader reads exactly
# these named columns, so the synthetic source carries exactly these).
_V2_PRICE_DDL = {
    "daily_bar_raw": """
        CREATE TABLE IF NOT EXISTS daily_bar_raw (
            ts_code VARCHAR, trade_date VARCHAR,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            pre_close DOUBLE, vol DOUBLE, amount DOUBLE
        )""",
    "daily_adj_factor": """
        CREATE TABLE IF NOT EXISTS daily_adj_factor (
            ts_code VARCHAR, trade_date VARCHAR, adj_factor DOUBLE
        )""",
    "daily_price_limit": """
        CREATE TABLE IF NOT EXISTS daily_price_limit (
            ts_code VARCHAR, trade_date VARCHAR,
            pre_close DOUBLE, up_limit DOUBLE, down_limit DOUBLE
        )""",
    "daily_suspension": """
        CREATE TABLE IF NOT EXISTS daily_suspension (
            ts_code VARCHAR, trade_date VARCHAR, suspend_type VARCHAR
        )""",
    "daily_basic_pit": """
        CREATE TABLE IF NOT EXISTS daily_basic_pit (
            ts_code VARCHAR, trade_date VARCHAR, turnover_rate DOUBLE,
            circ_mv DOUBLE
        )""",
}


def populate_v2_for_config(
    data_config: DataConfig,
    dates: list[str],
    ts_codes: list[str],
    bars: pd.DataFrame,
    **kwargs,
) -> Path:
    """Resolve the config's v2 price source path and populate it.

    A02/O3: this helper used to re-implement the v2 resolution rule locally,
    which had drifted from the authoritative ``resolve_v2_duckdb_path``
    (the local copy accepted a separator-bearing relative value that the
    authority refuses fail-closed).  It now calls the single authority, so
    tests and production resolve v2 paths through one rule only.
    """

    from ashare_data.config import resolve_v2_duckdb_path

    return populate_v2_price_tables(
        resolve_v2_duckdb_path(data_config), dates, ts_codes, bars, **kwargs
    )


V2_INDUSTRY_DDL = {
    "industry_membership_pit": """
        CREATE TABLE IF NOT EXISTS industry_membership_pit (
            ts_code VARCHAR, index_code VARCHAR, level INTEGER,
            in_date VARCHAR, out_date VARCHAR, is_new VARCHAR,
            industry_name VARCHAR, available_date VARCHAR
        )""",
    "industry_taxonomy": """
        CREATE TABLE IF NOT EXISTS industry_taxonomy (
            index_code VARCHAR, industry_name VARCHAR, level VARCHAR,
            industry_code VARCHAR, src VARCHAR
        )""",
}


def populate_v2_margin_table(v2_path: Path, rows: list[dict]) -> Path:
    """Populate the v2 ``margin_detail_pit`` table (p17 §5.2 table 18)
    with explicit contract rows for the capital-frame tests.  Rows carry
    ``ts_code / trade_date / available_date / rzye`` at minimum; the
    remaining p17 columns stay NULL.  The DDL is imported from the store
    authority (no second schema copy)."""

    from ashare_data.tushare_store import _MARGIN_DETAIL_PIT_DDL

    v2_path = Path(v2_path)
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    with AshareDB(v2_path) as db:
        db.execute(_MARGIN_DETAIL_PIT_DDL)
        db.execute("DELETE FROM margin_detail_pit")
        for row in rows:
            db.execute(
                "INSERT INTO margin_detail_pit "
                "(ts_code, trade_date, rzye, rzmre, available_date, ingest_id, "
                "source_row_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    row["ts_code"], row["trade_date"], row.get("rzye"),
                    row.get("rzmre"), row["available_date"],
                    row.get("ingest_id", "test_ingest"),
                    row.get("source_row_hash", f"test-{row['ts_code']}-{row['trade_date']}"),
                ],
            )
    return v2_path


def populate_v2_industry_tables(
    v2_path: Path,
    membership_rows: list[dict] | None = None,
    taxonomy_rows: list[dict] | None = None,
) -> Path:
    """Populate the v2 industry membership/taxonomy tables (p17 §5.2 table 8
    loader-facing subset + taxonomy) with explicit contract rows.

    ``membership_rows`` follow the p17 勘误 #2 expanded-row shape:
    ``ts_code / index_code / level / in_date / out_date (nullable) /
    is_new / industry_name / available_date``.  Default = tables exist but
    stay empty (loader degrades to an all-NaN industry frame, the same
    visible behavior as today's no-membership databases).
    """

    v2_path = Path(v2_path)
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    with AshareDB(v2_path) as db:
        for ddl in V2_INDUSTRY_DDL.values():
            db.execute(ddl)
        db.execute("DELETE FROM industry_membership_pit")
        db.execute("DELETE FROM industry_taxonomy")
        columns = {
            "industry_membership_pit": [
                "ts_code", "index_code", "level", "in_date", "out_date",
                "is_new", "industry_name", "available_date",
            ],
            "industry_taxonomy": [
                "index_code", "industry_name", "level", "industry_code", "src",
            ],
        }
        for table, rows in (
            ("industry_membership_pit", list(membership_rows or [])),
            ("industry_taxonomy", list(taxonomy_rows or [])),
        ):
            if rows:
                frame = pd.DataFrame(rows).reindex(columns=columns[table])
                db.register(f"_{table}_rows", frame)
                db.execute(
                    f"INSERT INTO {table} SELECT * FROM _{table}_rows"
                )
    return v2_path


V2_STATEMENT_COLUMNS = {
    "income_statement_pit": [
        "ts_code", "end_date", "ann_date", "f_ann_date", "available_date",
        "report_type", "comp_type", "update_flag", "end_type",
        "total_revenue", "revenue", "oper_cost", "total_profit",
        "n_income", "n_income_attr_p", "ingest_id",
    ],
    "balance_sheet_pit": [
        "ts_code", "end_date", "ann_date", "f_ann_date", "available_date",
        "report_type", "comp_type", "update_flag", "end_type",
        "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
        "total_hldr_eqy_inc_min_int", "ingest_id",
    ],
    "cashflow_statement_pit": [
        "ts_code", "end_date", "ann_date", "f_ann_date", "available_date",
        "report_type", "comp_type", "update_flag", "end_type",
        "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
        "c_cash_equ_end_period", "ingest_id",
    ],
}

V2_STATEMENT_TABLES = {
    "income": "income_statement_pit",
    "balance": "balance_sheet_pit",
    "cashflow": "cashflow_statement_pit",
}


def _assert_v2_statement_ddl_matches_columns() -> None:
    """Drift guard for the verbatim DDL below (SQL-injection hardening:
    no dynamic table/column names may enter SQL text, so the CREATE
    statements are spelled out literally; this asserts the literal
    spellings against V2_STATEMENT_COLUMNS)."""

    varchar_columns = {
        "ts_code", "end_date", "ann_date", "f_ann_date",
        "available_date", "report_type", "comp_type", "update_flag",
        "end_type", "ingest_id",
    }
    for table, columns in V2_STATEMENT_COLUMNS.items():
        expected = ", ".join(
            f"{col} VARCHAR" if col in varchar_columns else f"{col} DOUBLE"
            for col in columns
        )
        assert table in _V2_STATEMENT_DDL, table
        assert _V2_STATEMENT_DDL[table].endswith(f"({expected})"), table


_V2_STATEMENT_DDL: dict[str, str] = {
    "income_statement_pit": (
        "CREATE TABLE IF NOT EXISTS income_statement_pit ("
        "ts_code VARCHAR, end_date VARCHAR, ann_date VARCHAR, "
        "f_ann_date VARCHAR, available_date VARCHAR, report_type VARCHAR, "
        "comp_type VARCHAR, update_flag VARCHAR, end_type VARCHAR, "
        "total_revenue DOUBLE, revenue DOUBLE, oper_cost DOUBLE, "
        "total_profit DOUBLE, n_income DOUBLE, n_income_attr_p DOUBLE, "
        "ingest_id VARCHAR)"
    ),
    "balance_sheet_pit": (
        "CREATE TABLE IF NOT EXISTS balance_sheet_pit ("
        "ts_code VARCHAR, end_date VARCHAR, ann_date VARCHAR, "
        "f_ann_date VARCHAR, available_date VARCHAR, report_type VARCHAR, "
        "comp_type VARCHAR, update_flag VARCHAR, end_type VARCHAR, "
        "total_assets DOUBLE, total_liab DOUBLE, "
        "total_hldr_eqy_exc_min_int DOUBLE, "
        "total_hldr_eqy_inc_min_int DOUBLE, ingest_id VARCHAR)"
    ),
    "cashflow_statement_pit": (
        "CREATE TABLE IF NOT EXISTS cashflow_statement_pit ("
        "ts_code VARCHAR, end_date VARCHAR, ann_date VARCHAR, "
        "f_ann_date VARCHAR, available_date VARCHAR, report_type VARCHAR, "
        "comp_type VARCHAR, update_flag VARCHAR, end_type VARCHAR, "
        "n_cashflow_act DOUBLE, n_cashflow_inv_act DOUBLE, "
        "n_cash_flows_fnc_act DOUBLE, c_cash_equ_end_period DOUBLE, "
        "ingest_id VARCHAR)"
    ),
}


def populate_v2_statement_tables(
    v2_path: Path,
    income_rows: list[dict] | None = None,
    balance_rows: list[dict] | None = None,
    cashflow_rows: list[dict] | None = None,
) -> Path:
    """Populate the three v2 financial version tables (p17 §5.3, loader-facing
    column subset) with explicit contract rows (APPEND semantics: the helper
    deletes and re-inserts for scenario isolation)."""

    v2_path = Path(v2_path)
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_v2_statement_ddl_matches_columns()
    income = list(income_rows or [])
    balance = list(balance_rows or [])
    cashflow = list(cashflow_rows or [])
    with AshareDB(v2_path) as db:
        # SQL-injection hardening: verbatim statements only — no dynamic
        # table name ever enters SQL text (the drift guard above pins the
        # literal column spellings to V2_STATEMENT_COLUMNS).
        db.execute(
            "CREATE TABLE IF NOT EXISTS income_statement_pit ("
            "ts_code VARCHAR, end_date VARCHAR, ann_date VARCHAR, "
            "f_ann_date VARCHAR, available_date VARCHAR, report_type VARCHAR, "
            "comp_type VARCHAR, update_flag VARCHAR, end_type VARCHAR, "
            "total_revenue DOUBLE, revenue DOUBLE, oper_cost DOUBLE, "
            "total_profit DOUBLE, n_income DOUBLE, n_income_attr_p DOUBLE, "
            "ingest_id VARCHAR)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS balance_sheet_pit ("
            "ts_code VARCHAR, end_date VARCHAR, ann_date VARCHAR, "
            "f_ann_date VARCHAR, available_date VARCHAR, report_type VARCHAR, "
            "comp_type VARCHAR, update_flag VARCHAR, end_type VARCHAR, "
            "total_assets DOUBLE, total_liab DOUBLE, "
            "total_hldr_eqy_exc_min_int DOUBLE, "
            "total_hldr_eqy_inc_min_int DOUBLE, ingest_id VARCHAR)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS cashflow_statement_pit ("
            "ts_code VARCHAR, end_date VARCHAR, ann_date VARCHAR, "
            "f_ann_date VARCHAR, available_date VARCHAR, report_type VARCHAR, "
            "comp_type VARCHAR, update_flag VARCHAR, end_type VARCHAR, "
            "n_cashflow_act DOUBLE, n_cashflow_inv_act DOUBLE, "
            "n_cash_flows_fnc_act DOUBLE, c_cash_equ_end_period DOUBLE, "
            "ingest_id VARCHAR)"
        )
        db.execute("DELETE FROM income_statement_pit")
        db.execute("DELETE FROM balance_sheet_pit")
        db.execute("DELETE FROM cashflow_statement_pit")
        if income:
            frame = pd.DataFrame(income).reindex(
                columns=V2_STATEMENT_COLUMNS["income_statement_pit"]
            )
            db.register("_income_rows", frame)
            db.execute(
                "INSERT INTO income_statement_pit SELECT * FROM _income_rows"
            )
        if balance:
            frame = pd.DataFrame(balance).reindex(
                columns=V2_STATEMENT_COLUMNS["balance_sheet_pit"]
            )
            db.register("_balance_rows", frame)
            db.execute(
                "INSERT INTO balance_sheet_pit SELECT * FROM _balance_rows"
            )
        if cashflow:
            frame = pd.DataFrame(cashflow).reindex(
                columns=V2_STATEMENT_COLUMNS["cashflow_statement_pit"]
            )
            db.register("_cashflow_rows", frame)
            db.execute(
                "INSERT INTO cashflow_statement_pit "
                "SELECT * FROM _cashflow_rows"
            )
    return v2_path


def populate_v2_price_tables(
    v2_path: Path,
    dates: list[str],
    ts_codes: list[str],
    bars: pd.DataFrame,
    *,
    adj_factor: float | dict[tuple[str, str], float] = 1.0,
    limit_band: tuple[float, float] = (2.0, 0.5),
    limit_overrides: dict[tuple[str, str], tuple[float, float]] | None = None,
    drop_limit_for: set[tuple[str, str]] | None = None,
    drop_adj_for: set[tuple[str, str]] | None = None,
    suspensions: list[dict] | None = None,
    turnover_rate: float | dict[tuple[str, str], float] = 1.0,
    circ_mv: float | dict[tuple[str, str], float | None] | None = None,
) -> Path:
    """Populate the v2 price source (p19 §2 formal loader input) in tmp DuckDB.

    ``bars`` is the shared synthetic long frame (``make_bars``); every table
    is written for every (code, date) cell unless explicitly dropped via
    ``drop_limit_for`` / ``drop_adj_for`` (fail-closed scenarios).  Default
    limit bands are deliberately wide (never blocks) so only tests that
    target the exact constraint rule observe blocking.
    """

    v2_path = Path(v2_path)
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    required_bar_columns = (
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
    )
    if bars is None or bars.empty:
        bars = pd.DataFrame(columns=list(required_bar_columns))
    close_of = {
        (row.ts_code, str(row.trade_date)): float(row.close)
        for row in bars.itertuples(index=False)
    }
    limit_overrides = limit_overrides or {}
    drop_limit_for = drop_limit_for or set()
    drop_adj_for = drop_adj_for or set()
    bar_rows: list[dict] = []
    adj_rows: list[dict] = []
    limit_rows: list[dict] = []
    basic_rows: list[dict] = []
    for code in ts_codes:
        for date in dates:
            source = bars[(bars["ts_code"] == code) & (bars["trade_date"] == date)]
            if source.empty:
                continue
            row = source.iloc[0]
            bar_rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "pre_close": float(row["pre_close"]),
                    "vol": float(row["volume"]),
                    "amount": float(row["amount"]),
                }
            )
            if (code, date) not in drop_adj_for:
                value = (
                    adj_factor.get((code, date), 1.0)
                    if isinstance(adj_factor, dict)
                    else adj_factor
                )
                adj_rows.append(
                    {"ts_code": code, "trade_date": date, "adj_factor": value}
                )
            if (code, date) not in drop_limit_for:
                up, down = limit_overrides.get(
                    (code, date),
                    (
                        close_of[(code, date)] * limit_band[0],
                        close_of[(code, date)] * limit_band[1],
                    ),
                )
                limit_rows.append(
                    {
                        "ts_code": code,
                        "trade_date": date,
                        "pre_close": close_of[(code, date)],
                        "up_limit": float(up),
                        "down_limit": float(down),
                    }
                )
            value = (
                turnover_rate.get((code, date), 1.0)
                if isinstance(turnover_rate, dict)
                else turnover_rate
            )
            circ = (
                circ_mv.get((code, date))
                if isinstance(circ_mv, dict)
                else circ_mv
            )
            basic_rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "turnover_rate": value,
                    "circ_mv": circ,
                }
            )
    with AshareDB(v2_path) as db:
        for ddl in _V2_PRICE_DDL.values():
            db.execute(ddl)
        db.execute("DELETE FROM daily_bar_raw")
        db.execute("DELETE FROM daily_adj_factor")
        db.execute("DELETE FROM daily_price_limit")
        db.execute("DELETE FROM daily_suspension")
        db.execute("DELETE FROM daily_basic_pit")
        for table, rows in (
            ("daily_bar_raw", bar_rows),
            ("daily_adj_factor", adj_rows),
            ("daily_price_limit", limit_rows),
            ("daily_suspension", list(suspensions or [])),
            ("daily_basic_pit", basic_rows),
        ):
            if rows:
                db.register(f"_{table}_rows", pd.DataFrame(rows))
                db.execute(
                    f"INSERT INTO {table} SELECT * FROM _{table}_rows"
                )
    return v2_path


def pytest_configure(config: pytest.Config) -> None:
    """IP-10 02 (04-TC-07): itemized, targeted warning filter -- NOT a
    global silencer (AGENTS 10.1).  The OSQP PendingDeprecationWarning
    ("The default value of raise_error will change to True in the
    future.") fires 553x per full run from CVXPY's internal solver
    construction (ashare_portfolio/optimizer.py passes
    ``solver=cp.OSQP``; osqp's own API default is unreachable from our
    call site without changing solver semantics -- the explicit
    ``raise_error`` pin is the portfolio lane's / upstream's decision).
    The exact message + category match keeps every OTHER
    PendingDeprecationWarning visible; the disposition is itemized in
    docs/test_runtime_measurement_log.md."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The default value of raise_error will change to True in the future\\.:PendingDeprecationWarning",
    )


def make_bars(
    n_dates: int = 40,
    ts_codes: list[str] | None = None,
    end: str | None = None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Synthetic bars on a business-day axis.

    Default axis starts at 2024-01-01 (the historical fixture window).
    ``end`` anchors the axis so the LAST session is ``end`` (or the last
    business day before it) — the t23 today-relative variant for CLI
    smoke fixtures: the G8 freshness gate (p16 contract, t13) evaluates
    bar staleness against the real clock, so smoke fixtures must never
    rot with it.
    """

    ts_codes = ts_codes or DEFAULT_CODES
    if end is None:
        dates = pd.bdate_range("2024-01-01", periods=n_dates)
    else:
        dates = pd.bdate_range(end=end, periods=n_dates)
    dates = dates.strftime("%Y%m%d").tolist()
    rows = []
    for code in ts_codes:
        for i, date in enumerate(dates):
            close = 10.0 + i * 0.1 + (0.5 if code == "000001.SZ" else 0.0)
            open_ = close * 0.99
            pre_close = close * 0.99
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": open_,
                    "high": close * 1.02,
                    "low": open_ * 0.98,
                    "close": close,
                    "pre_close": pre_close,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * close,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                    # Exact-limit columns (p17 §5.2 table 5; t22): the
                    # generic fixtures carry a wide 10% band so the event
                    # family sees finite inputs; event-specific tests set
                    # explicit prices.
                    "close_raw": close,
                    "high_raw": close * 1.02,
                    "low_raw": open_ * 0.98,
                    "up_limit": round(pre_close * 1.10, 2),
                    "down_limit": round(pre_close * 0.90, 2),
                }
            )
    return dates, ts_codes, pd.DataFrame(rows)


@pytest.fixture(scope="session", autouse=True)
def _run_logging():
    setup_run_logging(run_name="pytest")
    yield
    # F-08 companion coverage (IP-11): the 2026-09-03 pytest exit lag and
    # the sync tail hang are one root-cause family.  Name any surviving
    # non-daemon thread (with its stack) in this session log — exported
    # right below — before interpreter teardown.  Detect-only: a forced
    # exit here would swallow the terminal report / CI result.
    guard_process_exit(timeout=5.0, force_exit=False)
    project_root = Path(__file__).resolve().parents[1]
    # Per-worker export target (PR3/B2+C2): each xdist worker exports its
    # own buffer to its own file; serial runs keep logs/pytest.txt.
    export_log_txt(path=project_root / "logs" / f"pytest{worker_log_suffix()}.txt")


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


@pytest.fixture
def bars_data():
    return make_bars()


def _populate_db(
    data_config: DataConfig,
    dates: list[str],
    ts_codes: list[str],
    bars,
    *,
    persist_manifest: bool,
    prelisted_open_sessions: list[str] | None = None,
) -> DataConfig:
    """Populate the shared synthetic database (single implementation).

    ``persist_manifest`` controls whether the composite snapshot generation
    is published: formal artifact flows (P8-05 + A02 p26 I2/I7) require a
    resolved dataset identity, while manifest-machinery tests need a
    database without one.  ``prelisted_open_sessions`` extends the trade
    calendar with future open sessions (no bars) so the pre-listed horizon
    covers the G8 evaluation day; the rows land before the publication so
    the generation stays authoritative for the full table.

    The v2 price source (p19 §2: the loader's formal price input) is
    populated next to the legacy database with neutral defaults
    (adj_factor=1.0, wide limit bands, no suspensions) so scenario tests
    re-populate it via :func:`populate_v2_price_tables`; the composite
    publication happens only after BOTH sides are populated.
    """

    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    data_config.parquet_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {
                    "ts_code": c,
                    "name": c,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for c in ts_codes
            ],
            data_config,
        )
        calendar_rows = [{"trade_date": d, "is_open": True} for d in dates]
        calendar_rows.extend(
            {"trade_date": d, "is_open": True}
            for d in prelisted_open_sessions or []
        )
        db.upsert_calendar(calendar_rows, data_config)
        db.upsert_constituents(
            [
                {
                    "index_code": "000300.SH",
                    "ts_code": c,
                    "in_date": f"2020010{i + 1}",
                    "out_date": "99991231",
                }
                for i, c in enumerate(ts_codes)
            ],
            data_config,
        )
    v2_path = Path(data_config.v2_duckdb_path)
    if not v2_path.is_absolute():
        v2_path = data_config.duckdb_path.parent / v2_path
    populate_v2_price_tables(v2_path, dates, ts_codes, bars)
    # D5: the loader's industry source lives in the same v2 database; the
    # shared fixture ships empty membership/taxonomy tables (all-NaN frame,
    # the same visible behavior as a no-membership database) so scenario
    # tests seed rows via :func:`populate_v2_industry_tables`.
    populate_v2_industry_tables(v2_path)
    if persist_manifest:
        # P8-05 + A02 (p26 I2/I7): formal artifacts require a resolved
        # dataset identity, and the formal path consumes the
        # content-confirmed COMPOSITE generation.  The publication happens
        # only after BOTH sides are populated — the group binds the
        # content a formal load will actually read.
        publish_manifest(data_config)
    return data_config


@pytest.fixture
def populated_db(data_config: DataConfig, bars_data) -> DataConfig:
    dates, ts_codes, bars = bars_data
    return _populate_db(data_config, dates, ts_codes, bars, persist_manifest=True)


@pytest.fixture
def fresh_data_config(tmp_path: Path) -> DataConfig:
    """Today-relative data window (t23): ``end_date`` is the real clock's
    date so the bar axis tracks the clock.  The G8 pre-listed horizon is
    kept ahead of the clock by :func:`_future_prelisted_sessions` in
    ``fresh_populated_db`` (weekend-safe; p16 contract, t13)."""

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=90)
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


@pytest.fixture
def fresh_bars_data():
    """Bars whose last session is today (or the last business day before
    it) — G8-fresh by construction together with the pre-listed calendar
    horizon of ``fresh_populated_db`` (weekend-safe)."""

    return make_bars(end=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))


def _future_prelisted_sessions(last_session: str, days: int = 7) -> list[str]:
    """Business days strictly after ``last_session`` through
    ``today + days``, in the bars-axis ``%Y%m%d`` format — the
    pre-listed open-session horizon G8 requires.

    G8 (p16) fails closed when the calendar's pre-listed horizon no
    longer covers the evaluation day.  The fresh fixtures end their bar
    axis at the last business day, so on a weekend the horizon stops at
    Friday while ``today`` is Saturday/Sunday and the gate refuses
    (t43-era full-suite rot: 4 CLI-subprocess tests failed identically
    on every branch).  Pre-listing future open sessions — exactly what a
    real trading calendar does — keeps the horizon ahead of the clock on
    every day of the week.  The sessions carry no bars: G8's lag counts
    only open sessions strictly before the evaluation day.  The
    ``%Y%m%d`` format matches ``make_bars``' axis: the gate compares
    dates as strings, and mixed formats would sort the extension rows
    below the ``%Y%m%d`` horizon (defeating the extension).
    """

    last = pd.Timestamp(last_session)
    horizon = pd.Timestamp.today().normalize() + pd.Timedelta(days=days)
    sessions: list[str] = []
    day = last + pd.Timedelta(days=1)
    while day <= horizon:
        if day.dayofweek < 5:  # Monday-Friday only
            sessions.append(day.strftime("%Y%m%d"))
        day += pd.Timedelta(days=1)
    return sessions


@pytest.fixture
def fresh_populated_db(
    fresh_data_config: DataConfig, fresh_bars_data
) -> tuple[DataConfig, list[str]]:
    """Today-relative variant of ``populated_db`` (t23); returns the data
    config plus the session axis so tests can derive fold dates from the
    same positions the historical 2024-01-01 axis used.  Used by tests
    whose formal entry runs in a SUBPROCESS (monkeypatch cannot cross the
    process boundary).  The trade calendar is pre-listed beyond the bar
    axis (see :func:`_future_prelisted_sessions`) so the G8 horizon never
    exhausts behind the clock."""

    dates, ts_codes, bars = fresh_bars_data
    cfg = _populate_db(
        fresh_data_config,
        dates,
        ts_codes,
        bars,
        persist_manifest=True,
        prelisted_open_sessions=_future_prelisted_sessions(dates[-1]),
    )
    return cfg, dates


@pytest.fixture
def production_gate_asof(monkeypatch, bars_data) -> str:
    """t23 option (a) for IN-PROCESS formal-entry tests: inject the G8
    evaluation day through the p16-reserved test-only seam
    (``ProductionGateRunner.require_production(*, today=...)`` —
    production callers never pass it and no config surface can override
    it).  The injected day is the fixture bars' max session, so the gates
    evaluate the synthetic era deterministically — no real wall clock
    anywhere.  P36: only the ONLINE basis is era-injected; historical
    basis calls keep their snapshot-derived day (their determinism comes
    from the frozen cutoff, and converting them to an era day would be a
    basis mix the real gate refuses).  Subprocess-entry tests must use
    ``fresh_populated_db`` instead (monkeypatch cannot cross process
    boundaries)."""

    from ashare_data.gates import (
        ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
        ELIGIBILITY_PURPOSE_ONLINE_OBSERVATION,
        ProductionGateRunner,
    )

    era = bars_data[0][-1]
    real = ProductionGateRunner.require_production

    def _asof(
        self,
        *,
        today: str | None = None,
        purpose: str = ELIGIBILITY_PURPOSE_ONLINE_OBSERVATION,
        snapshot_cutoff: str | None = None,
    ):
        if purpose == ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH:
            return real(
                self, purpose=purpose, snapshot_cutoff=snapshot_cutoff
            )
        return real(
            self,
            today=today or era,
            purpose=purpose,
            snapshot_cutoff=snapshot_cutoff,
        )

    monkeypatch.setattr(ProductionGateRunner, "require_production", _asof)
    return era


@pytest.fixture
def populated_db_without_manifest(data_config: DataConfig, bars_data) -> DataConfig:
    """The same synthetic database without a persisted dataset manifest."""
    dates, ts_codes, bars = bars_data
    return _populate_db(data_config, dates, ts_codes, bars, persist_manifest=False)


# --- P8-05: formal-artifact fixture helpers (single implementation) ----------


def make_test_spec(**overrides):
    """A valid frozen RunSpec for fixture writers.

    Provenance (git commit, dependency-lock hash) is captured from the real
    repository with the clean-tree gate off — test worktrees are routinely
    dirty; the SPEC_LOCKED clean-tree evidence gate is a lifecycle-stage
    concern (P8-06+), not exercised here.
    """

    from ashare_model.runspec import WindowSpec, build_runspec
    from ashare_portfolio.execution_spec import portfolio_config_provenance

    kwargs = dict(
        dataset_id="ds-test",
        data_cutoff="20241231",
        # P36: research fixtures default to the historical-research
        # purpose (the fixture window is historical by construction);
        # purpose-sensitive tests override it explicitly.
        purpose="historical_research",
        frequency="daily",
        target="forward_return",
        horizon=1,
        requested_budget=100,
        n_folds=1,
        dev_window=WindowSpec(start="20240101", end="20240201"),
        holdout_window=WindowSpec(start="20240201", end="20241231"),
        preregistered_contract_hash="a" * 64,
        resolved_thresholds={"min_eligible": 1.0},
        portfolio_config=portfolio_config_provenance(BacktestConfig()),
        seeds=(42,),
        searcher="random",
        repo_root=None,
        require_clean_tree=False,
    )
    kwargs.update(overrides)
    return build_runspec(**kwargs)


def open_test_handle(store_root, spec=None, **spec_overrides):
    """Open a fresh run in a test RunStore. The caller closes the handle."""

    from ashare_model.run_store import RunStore

    return RunStore(store_root).open_run(spec or make_test_spec(**spec_overrides))


def write_current_strategy(
    data_dir: Path,
    *,
    formula=(1,),
    formula_text: str = "RET_1",
    direction: int = 1,
    dataset_id: str,
    backtest_config: BacktestConfig | None = None,
    spec=None,
) -> dict:
    """Write a schema-v2 strategy artifact through the formal P8-05 path
    (RunStore-registered + convenience mirror) and return the persisted
    payload. This is the fixture replacement for hand-written legacy
    ``best_ashare_strategy.json`` files.
    """

    from ashare_model.alphagpt import MODEL_VERSION
    from ashare_model.artifact_schemas import StrategyArtifact
    from ashare_model.artifact_writer import write_boundary_artifact
    from ashare_model.identity import candidate_id
    from ashare_model.research_domain import RESEARCH_DOMAIN_VERSION
    from ashare_model.reward import REWARD_VERSION
    from ashare_model.search_contract import SEARCH_CONTRACT_VERSION
    from ashare_model.semantic_cache import SEMANTIC_CACHE_VERSION
    from ashare_model.vocab import FORMULA_VOCAB, GRAMMAR_VERSION
    from ashare_portfolio.execution_spec import execution_provenance

    from ashare_model.evaluation import PROTOCOL_VERSION
    from ashare_portfolio.execution_spec import portfolio_config_provenance

    backtest_config = backtest_config or BacktestConfig()
    if spec is None:
        spec = make_test_spec(
            dataset_id=dataset_id,
            frequency=backtest_config.rebalance_frequency,
            horizon=backtest_config.target_horizon,
            portfolio_config=portfolio_config_provenance(backtest_config),
        )
    cand = candidate_id(spec.spec_id, list(formula), direction)
    payload = {
        "formula": list(formula),
        "formula_text": formula_text,
        "source": "random",
        "direction": direction,
        "data_tier": None,
        "val_reward": 0.1,
        "val_icir": 0.2,
        "train_reward": 0.1,
        "train_icir": 0.2,
        "complexity_penalty": 0.0,
        "complexity_cost": 1.0,
        "active_ir": None,
        "risk_exposure": None,
        "average_turnover": None,
        "capacity_utilization": None,
        "eligible": True,
        "rejection_reasons": [],
        "history": [],
        "searcher": "random",
        "init_seed": 42,
        "device": "cpu",
        "model_version": MODEL_VERSION,
        # Real vocab identity: resolve_formula_tokens remaps by name, so the
        # fixture must carry the true vocabulary to round-trip losslessly.
        "feature_names": list(FORMULA_VOCAB.feature_names),
        "operator_names": list(FORMULA_VOCAB.operator_names),
        "feature_version": FORMULA_VOCAB.feature_version,
        "grammar_version": GRAMMAR_VERSION,
        "reward_version": REWARD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "research_domain": "unified",
        "research_domain_version": RESEARCH_DOMAIN_VERSION,
        **execution_provenance(backtest_config),
        "semantic_cache_version": SEMANTIC_CACHE_VERSION,
        "unique_semantic_evals": 1,
        "semantic_cache_stats": None,
        "search_contract_version": SEARCH_CONTRACT_VERSION,
        "search_result": None,
        "dataset_id": dataset_id,
        "searcher_candidate_label": f"random:{','.join(str(t) for t in formula)}",
    }
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open_test_handle(data_dir, spec=spec) as handle:
        write_boundary_artifact(
            handle,
            artifact_type="strategy",
            model_cls=StrategyArtifact,
            payload=payload,
            candidate_id=cand,
            convenience_path=data_dir / "best_ashare_strategy.json",
        )
    # Return the persisted mirror — exactly what formal consumers will read.
    return json.loads(
        (data_dir / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
