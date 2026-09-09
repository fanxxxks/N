"""Tests for the centralized ProductionGateRunner (T0-03)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import DataConfig
from ashare_data.db import AshareDB
from ashare_data.gates import (
    GateResult,
    ProductionGateRunner,
    _dev_membership_records,
)
from ashare_data.universe import (
    ResolvedUniverse,
    UniverseContractError,
    UniverseContractStatus,
    dev_fallback_membership_records,
    member_bar_coverage,
)
from tests.conftest import DEFAULT_CODES, make_bars, make_test_spec

# p16 §7: gate tests inject the evaluation day and never depend on the
# real wall clock.  The shared fixture's calendar == bars horizon (40
# business days from 2024-01-01), so "today" = the last fixture session
# makes the daily window fresh (reference = previous session, lag = 0).
TODAY_LAST_SESSION = make_bars()[0][-1]


def test_dev_membership_records_delegates_to_shared_definition(data_config):
    """F4: the gates' dev-fallback membership records come from the single
    definition in ashare_data.universe, byte-identical in content, key
    order, record order and string types."""
    status = UniverseContractStatus(
        mode="dev",
        strict=False,
        degraded=True,
        membership_source="development-fallback",
        session_source="trade_calendar",
        warnings=("dev fallback active",),
        constituent_rows=0,
        stock_rows=0,
        open_sessions=2,
    )
    contract = ResolvedUniverse(
        constituents=pd.DataFrame(),
        stocks=pd.DataFrame(),
        sessions=["20200102", "20200103"],
        codes=["600000.SH", "000001.SZ"],
        status=status,
    )
    records = _dev_membership_records(data_config, contract)
    shared = dev_fallback_membership_records(
        data_config.index_codes[0], contract
    )
    assert json.dumps(records) == json.dumps(shared)
    assert json.dumps(records) == (
        '[{"index_code": "000300.SH", "ts_code": "600000.SH", '
        '"in_date": "20200102", "out_date": "99991231"}, '
        '{"index_code": "000300.SH", "ts_code": "000001.SZ", '
        '"in_date": "20200102", "out_date": "99991231"}]'
    )


def _seed_zero_bar_member(data_config) -> None:
    """Add a historical member with an open interval overlapping the bar
    horizon but zero daily bars (the delisted/never-synced signature)."""

    with AshareDB(data_config.duckdb_path) as db:
        db.upsert_stocks(
            [
                {
                    "ts_code": "600999.SH",
                    "name": "600999.SH",
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
            ],
            data_config,
        )
        db.upsert_constituents(
            [
                {
                    "index_code": "000300.SH",
                    "ts_code": "600999.SH",
                    "in_date": "20240102",
                    "out_date": "20241231",
                }
            ],
            data_config,
        )


def test_formal_gate_passes_on_clean_db(populated_db):
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result = runner.run(mode="formal", today=TODAY_LAST_SESSION)
    assert result.ok and not result.degraded
    assert result.mode == "formal"
    assert {check.name for check in result.checks} >= {
        "G1 historical member intervals per configured index",
        "G2 valid list_date for all participating stocks",
        "G3 open calendar covers the daily-bar window",
        "G4 no duplicate/overlapping intervals",
        "G5 strict-mode PIT universe contract",
        "G6 >= 3 eligible stocks per major window",
        "G7 every PIT member interval has daily bars",
        "G8 daily bars fresh within the open-session tolerance",
    }
    assert all(check.ok for check in result.checks)
    assert result.status is not None and result.status.strict
    # require_production is the formal entry API and returns the status.
    status = runner.require_production(today=TODAY_LAST_SESSION)
    assert status.strict is True


def test_formal_gate_fails_on_zero_bar_historical_member(populated_db):
    _seed_zero_bar_member(populated_db)
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    with pytest.raises(UniverseContractError, match="G7"):
        runner.require_production(today=TODAY_LAST_SESSION)
    result = runner.run(mode="formal", today=TODAY_LAST_SESSION)
    assert not result.ok
    g7 = result.check("G7 every PIT member interval has daily bars")
    assert not g7.ok
    assert "600999.SH" in g7.detail


def test_formal_gate_fails_when_min_eligible_unreachable(populated_db):
    runner = ProductionGateRunner(populated_db, min_eligible=10_000)
    with pytest.raises(UniverseContractError, match="G6"):
        runner.require_production(today=TODAY_LAST_SESSION)
    result = runner.run(mode="formal", today=TODAY_LAST_SESSION)
    assert not result.check("G6 >= 10000 eligible stocks per major window").ok


def test_dev_mode_degrades_instead_of_raising(populated_db):
    _seed_zero_bar_member(populated_db)
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result: GateResult = runner.run(mode="dev", today=TODAY_LAST_SESSION)
    # Dev mode never raises and always reports degraded=true.
    assert result.mode == "dev"
    assert result.degraded is True
    assert not result.ok
    assert any("zero-bar" in warning for warning in result.warnings)
    # The failing check is visible, not swallowed.
    assert not result.check("G7 every PIT member interval has daily bars").ok


def test_dev_mode_clean_db_is_still_marked_degraded(populated_db):
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result = runner.run(mode="dev", today=TODAY_LAST_SESSION)
    assert result.ok
    # Dev mode is inherently non-production: the result always carries
    # degraded=true.
    assert result.degraded is True


def test_run_rejects_unknown_mode(populated_db):
    with pytest.raises(ValueError, match="unknown gate mode"):
        ProductionGateRunner(populated_db).run(mode="staging")


def test_member_bar_coverage_is_half_open_and_horizon_capped(data_config):
    """G7's audit uses the canonical half-open interval and caps sessions
    to the daily-bar horizon: a bar exactly on ``out_date`` never counts,
    and an interval entirely before the first bar is not auditable."""

    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_calendar(
            [{"trade_date": f"2024010{i}", "is_open": True} for i in range(1, 7)],
            data_config,
        )
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20240101", "out_date": "20240106"},
                {"index_code": "000300.SH", "ts_code": "600999.SH", "in_date": "20230101", "out_date": "20231231"},
            ],
            data_config,
        )
        db.upsert_daily(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"2024010{i}",
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
                for i in range(1, 6)
            ],
            data_config,
        )
        frame = member_bar_coverage(db, data_config)
    rows = frame.set_index("ts_code")
    # Half-open [20240101, 20240106) capped to the horizon (max bar
    # 20240105): the bar on 20240106 must NOT count (five bars exist;
    # four fall inside the audited span).
    assert rows.loc["000001.SZ", "bars"] == 4
    assert rows.loc["000001.SZ", "sessions"] == 4
    assert rows.loc["000001.SZ", "coverage"] == 1.0
    # An interval entirely before the daily horizon is not auditable with
    # the current data: no sessions, coverage undefined.
    assert rows.loc["600999.SH", "sessions"] == 0
    assert rows.loc["600999.SH", "bars"] == 0
    assert np.isnan(rows.loc["600999.SH", "coverage"])


def test_g7_zero_bar_requires_overlap_with_horizon(data_config):
    """A member interval entirely inside the pre-horizon past must not
    fail G7 (it is not auditable with the current data window)."""

    dates, codes, bars = make_bars(6, DEFAULT_CODES)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {
                    "ts_code": code,
                    "name": code,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for code in codes + ["600999.SH"]
            ],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in dates], data_config
        )
        # One member whose interval ends before the first bar, plus a
        # pre-horizon-only historical member with no bars at all.
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": code, "in_date": "20200101", "out_date": "99991231"}
                for code in codes
            ]
            + [
                {"index_code": "000300.SH", "ts_code": "600999.SH", "in_date": "20230101", "out_date": "20231231"},
            ],
            data_config,
        )
    runner = ProductionGateRunner(data_config, min_eligible=3)
    result = runner.run(mode="formal", today=dates[-1])
    assert result.ok
    assert result.check("G7 every PIT member interval has daily bars").ok


# ---------------------------------------------------------------------------
# G8 data-freshness gate (p16 contract, IP-05 / 03-F-03): daily-bar max
# trade date must sit within FRESHNESS_TOLERANCE_SESSIONS open sessions of
# the most recent completed open session (strictly before the evaluation
# day).  lag counts OPEN SESSIONS, never calendar days.
# ---------------------------------------------------------------------------


def _g8_check(result: GateResult):
    matches = [c for c in result.checks if c.name.startswith("G8")]
    assert matches, [c.name for c in result.checks]
    return matches[0]


def _stale_calendar(populated_db, *, extra_sessions: int = 9):
    """The p15 incident shape: the calendar pre-lists the future beyond
    the daily-bar horizon, so ``lag`` = extra_sessions and the reference
    is the last pre-listed session strictly before ``today``."""

    dates = make_bars()[0]
    extra = (
        pd.bdate_range("2024-01-01", periods=40 + extra_sessions)
        .strftime("%Y%m%d")
        .tolist()[40:]
    )
    with AshareDB(populated_db.duckdb_path) as db:
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in extra],
            populated_db,
        )
    return dates, extra


def test_g8_passes_when_daily_is_fresh(populated_db):
    from ashare_data.gates import FRESHNESS_TOLERANCE_SESSIONS

    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result = runner.run(mode="formal", today=TODAY_LAST_SESSION)
    g8 = _g8_check(result)
    assert g8.ok
    # p16 §2.2: the detail names every auditable quantity.
    assert f"today={TODAY_LAST_SESSION}" in g8.detail
    assert "lag=0" in g8.detail
    assert f"N={FRESHNESS_TOLERANCE_SESSIONS}" in g8.detail


def test_g8_fails_when_daily_stale_beyond_tolerance(populated_db):
    dates, extra = _stale_calendar(populated_db)  # lag = 8 > N = 3
    today = extra[-1]
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    with pytest.raises(UniverseContractError, match="G8"):
        runner.require_production(today=today)
    result = runner.run(mode="formal", today=today)
    assert not result.ok
    g8 = _g8_check(result)
    assert not g8.ok
    assert "lag=8" in g8.detail
    assert f"daily {dates[-1]}" in g8.detail


def test_g8_dev_mode_degrades_without_raising(populated_db):
    _stale_calendar(populated_db)
    today = make_bars(49)[0][-1]
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result = runner.run(mode="dev", today=today)
    assert result.degraded is True
    assert not result.ok
    assert not _g8_check(result).ok


def test_g8_tolerance_knob_flips_the_verdict(populated_db):
    _stale_calendar(populated_db)  # lag = 8
    today = make_bars(49)[0][-1]
    tight = ProductionGateRunner(populated_db, min_eligible=3, freshness_tolerance=3)
    loose = ProductionGateRunner(populated_db, min_eligible=3, freshness_tolerance=8)
    assert not _g8_check(tight.run(mode="formal", today=today)).ok
    assert _g8_check(loose.run(mode="formal", today=today)).ok


def test_g8_lag_counts_open_sessions_not_calendar_days(data_config):
    """A long holiday between the bar horizon and the reference must not
    inflate the lag: 3 open sessions + a 10-calendar-day holiday + 5 open
    sessions = lag 8, never 18 (p16 §2.1 hard rule)."""
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        base = pd.Timestamp("20240131")
        db.upsert_daily(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": base.strftime("%Y%m%d"),
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
            ],
            data_config,
        )
        rows = []
        for offset in range(1, 20):
            day = (base + pd.Timedelta(days=offset)).strftime("%Y%m%d")
            open_session = offset <= 3 or 14 <= offset <= 19
            rows.append({"trade_date": day, "is_open": open_session})
        db.upsert_calendar(rows, data_config)
    runner = ProductionGateRunner(data_config, min_eligible=3)
    result = runner.run(mode="formal", today="20240219")  # day 19
    g8 = _g8_check(result)
    assert "lag=8" in g8.detail  # 3 + 5 open sessions, holiday excluded
    assert not g8.ok  # 8 > N = 3


def test_g8_fail_closed_when_reference_unavailable(data_config):
    """p16 §2.2 fail-closed boundaries: no daily rows, no open session
    strictly before today, and an exhausted calendar horizon must all
    fail G8 — 'cannot compute' never degrades into a pass."""
    runner = ProductionGateRunner(data_config, min_eligible=3)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_calendar(
            [{"trade_date": f"2024010{i}", "is_open": True} for i in range(1, 7)],
            data_config,
        )
    # (a) daily table has no rows.
    result = runner.run(mode="formal", today="20240106")
    g8 = _g8_check(result)
    assert not g8.ok
    assert "daily" in g8.detail
    # (b) daily rows exist, but no open session strictly precedes the
    # evaluation day (today sits on the horizon's first session).
    with AshareDB(data_config.duckdb_path) as db:
        db.upsert_daily(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20240105",
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
            ],
            data_config,
        )
    result = runner.run(mode="formal", today="20240101")
    g8 = _g8_check(result)
    assert not g8.ok
    assert "reference" in g8.detail
    # (c) the pre-listed calendar horizon is exhausted (max session < today).
    result = runner.run(mode="formal", today="20240201")
    g8 = _g8_check(result)
    assert not g8.ok
    assert "horizon" in g8.detail


def test_g8_has_no_disable_surface():
    """p16 §2.2: no config/flag may turn G8 off (source-level check)."""
    source = (
        Path(__file__).resolve().parents[1] / "ashare_data" / "gates.py"
    ).read_text(encoding="utf-8")
    assert not re.search(
        r"(?i)(skip|disable|enable|bypass)[_\-]?(g8|freshness)", source
    ), "G8 must have no disable/enable switch surface"


# --- D7: per-formula-dependency data eligibility gate (p17 §12) ---------------
#
# The D7 gate (P8 FACTOR_SET_QUALIFIED's data check) resolves which p17
# §12 data sources a RunSpec's factor set actually depends on — via the
# feature registry's pit_level/family, the single per-feature source
# authority — and checks only those sources against the preregistered
# thresholds.  A global "all optional sources must be fresh" gate is a
# preregistered NON-goal (p17 §12): unrequired sources never affect a
# verdict, and a source that is not computable always fails (never
# passes).

D7_SESSIONS = pd.bdate_range("2024-01-01", periods=10).strftime("%Y%m%d").tolist()
D7_EVALUATION_DAY = "20240115"  # the next business day after the axis
D7_UNIVERSE = DEFAULT_CODES
D7_QUARTER_ENDS = ["20230331", "20230630", "20230930", "20231231"]

V2_REMAINING_DDL = {
    # p17 §5.2 table 18 — the D7 margin source (tolerance 2 sessions).
    "margin_detail_pit": """
        CREATE TABLE IF NOT EXISTS margin_detail_pit (
            ts_code VARCHAR, trade_date VARCHAR,
            rzye DOUBLE, rqye DOUBLE, rzmre DOUBLE, rqyl DOUBLE,
            rzche DOUBLE, rqchl DOUBLE, rqmcl DOUBLE, rzrqye DOUBLE,
            available_date VARCHAR, ingest_id VARCHAR
        )""",
    "index_weight_pit": """
        CREATE TABLE IF NOT EXISTS index_weight_pit (
            index_code VARCHAR, con_code VARCHAR, trade_date VARCHAR,
            weight DOUBLE, available_date VARCHAR, ingest_id VARCHAR
        )""",
    "financial_indicator_audit": """
        CREATE TABLE IF NOT EXISTS financial_indicator_audit (
            ts_code VARCHAR, end_date VARCHAR, ann_date VARCHAR,
            f_ann_date VARCHAR, update_flag VARCHAR, report_type VARCHAR,
            eps DOUBLE, roe DOUBLE, roe_waa DOUBLE,
            grossprofit_margin DOUBLE, netprofit_margin DOUBLE,
            debt_to_assets DOUBLE, ingest_id VARCHAR
        )""",
    "financial_disclosure_schedule": """
        CREATE TABLE IF NOT EXISTS financial_disclosure_schedule (
            ts_code VARCHAR, end_date VARCHAR, pre_date VARCHAR,
            actual_date VARCHAR, ingest_id VARCHAR
        )""",
    "earnings_forecast_pit": """
        CREATE TABLE IF NOT EXISTS earnings_forecast_pit (
            ts_code VARCHAR, ann_date VARCHAR, end_date VARCHAR,
            type VARCHAR, p_change_min DOUBLE, p_change_max DOUBLE,
            net_profit_min DOUBLE, net_profit_max DOUBLE,
            last_parent_net DOUBLE, summary VARCHAR, change_reason VARCHAR,
            available_date VARCHAR, ingest_id VARCHAR
        )""",
    "earnings_express_pit": """
        CREATE TABLE IF NOT EXISTS earnings_express_pit (
            ts_code VARCHAR, ann_date VARCHAR, end_date VARCHAR,
            revenue DOUBLE, operate_profit DOUBLE, total_profit DOUBLE,
            n_income DOUBLE, total_assets DOUBLE,
            available_date VARCHAR, ingest_id VARCHAR
        )""",
    "repurchase_event_pit": """
        CREATE TABLE IF NOT EXISTS repurchase_event_pit (
            ts_code VARCHAR, ann_date VARCHAR, end_date VARCHAR,
            proc VARCHAR, exp_date VARCHAR, vol DOUBLE, amount DOUBLE,
            high_limit DOUBLE, low_limit DOUBLE,
            available_date VARCHAR, ingest_id VARCHAR
        )""",
    "holder_number_pit": """
        CREATE TABLE IF NOT EXISTS holder_number_pit (
            ts_code VARCHAR, ann_date VARCHAR, end_date VARCHAR,
            holder_num BIGINT, available_date VARCHAR, ingest_id VARCHAR
        )""",
    "source_ingest_runs": """
        CREATE TABLE IF NOT EXISTS source_ingest_runs (
            ingest_id VARCHAR, provider VARCHAR, endpoint VARCHAR,
            parameters_json VARCHAR, fields_json VARCHAR,
            started_at TIMESTAMP, finished_at TIMESTAMP,
            status VARCHAR, row_count BIGINT,
            raw_payload_sha256 VARCHAR, raw_path VARCHAR,
            code_commit VARCHAR, sdk_version VARCHAR,
            hit_row_limit BOOLEAN, error_type VARCHAR, error_message VARCHAR
        )""",
}


def ensure_v2_store_tables(v2_path: Path, margin_rows: list[dict] | None = None):
    """Create the nine v2 tables the price/industry/statement helpers do
    not cover (p17 §5.2/§5.3/§5.4 loader-facing column subset), so a test
    store can carry the full 19-table dataset manifest.  ``margin_rows``
    seed the D7 margin source (``available_date`` = the §6.1 T+1 open
    session, writer-computed)."""

    v2_path = Path(v2_path)
    v2_path.parent.mkdir(parents=True, exist_ok=True)
    margin_columns = [
        "ts_code", "trade_date", "rzye", "rqye", "rzmre", "rqyl",
        "rzche", "rqchl", "rqmcl", "rzrqye", "available_date", "ingest_id",
    ]
    with AshareDB(v2_path) as db:
        for ddl in V2_REMAINING_DDL.values():
            db.execute(ddl)
        db.execute("DELETE FROM margin_detail_pit")
        rows = list(margin_rows or [])
        if rows:
            db.register(
                "_margin_rows", pd.DataFrame(rows).reindex(columns=margin_columns)
            )
            db.execute("INSERT INTO margin_detail_pit SELECT * FROM _margin_rows")
    return v2_path


def _d7_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path / "data",
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


def _d7_calendar(config: DataConfig) -> None:
    """Seed the unique production trade calendar with the D7 axis."""

    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in D7_SESSIONS],
            config,
        )


def _d7_statement_rows(table: str, *, skip_keys: set[tuple[str, str]] = ()):
    """Full-coverage version-table rows for the universe × quarter-ends
    (every p17 §5.3 core column non-null), minus ``skip_keys``."""

    from tests.conftest import V2_STATEMENT_COLUMNS

    identity = (
        "ts_code", "end_date", "ann_date", "f_ann_date", "available_date",
        "report_type", "comp_type", "update_flag", "end_type", "ingest_id",
    )
    value_columns = [c for c in V2_STATEMENT_COLUMNS[table] if c not in identity]
    rows = []
    for code in D7_UNIVERSE:
        for period in D7_QUARTER_ENDS:
            if (code, period) in skip_keys:
                continue
            row = {
                "ts_code": code,
                "end_date": period,
                "ann_date": period,
                "f_ann_date": period,
                "available_date": period,
                "report_type": "1",
                "comp_type": "1",
                "update_flag": "0",
                "end_type": "1",
                "ingest_id": f"ing-{table}-{code}-{period}",
            }
            for column in value_columns:
                row[column] = 1.0
            rows.append(row)
    return rows


def _d7_membership_rows(*, late_available_for: str | None = None,
                        overlap: bool = False):
    """Healthy membership (intervals start before the axis, available
    before it — coverage 1.0) with scenario overrides."""

    rows = []
    for code in D7_UNIVERSE:
        available = "20240108" if code == late_available_for else "20231221"
        rows.append(
            {
                "ts_code": code,
                "index_code": "801010",
                "level": 1,
                "in_date": "20231220",
                "out_date": None,
                "available_date": available,
            }
        )
    if overlap:
        rows.append(
            {
                "ts_code": D7_UNIVERSE[0],
                "index_code": "801020",
                "level": 1,
                "in_date": "20240103",
                "out_date": None,
                "available_date": "20240104",
            }
        )
    taxonomy = [
        {"index_code": "801010", "industry_name": "A", "level": "L1",
         "industry_code": "801010", "src": "sw2021"},
        {"index_code": "801020", "industry_name": "B", "level": "L1",
         "industry_code": "801020", "src": "sw2021"},
    ]
    return rows, taxonomy


def _d7_margin_rows(*, through_index: int):
    """Margin rows through ``axis[through_index]`` with the §6.1 T+1
    available_date mapping (writer-computed, consumed by the gate)."""

    rows = []
    for code in D7_UNIVERSE:
        for index, date in enumerate(D7_SESSIONS[: through_index + 1]):
            next_session = (
                D7_SESSIONS[index + 1]
                if index + 1 < len(D7_SESSIONS)
                else D7_EVALUATION_DAY
            )
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "rzye": 1.0,
                    "rqye": 1.0,
                    "rzmre": 1.0,
                    "rqyl": 1.0,
                    "rzche": 1.0,
                    "rqchl": 1.0,
                    "rqmcl": 1.0,
                    "rzrqye": 1.0,
                    "available_date": next_session,
                }
            )
    return rows


def _d7_world(
    tmp_path: Path,
    *,
    save_manifest: bool = True,
    bar_dates: list[str] | None = None,
    limit_through_index: int | None = None,
    drop_adj_from_index: int | None = None,
    margin_through_index: int | None = 9,
    membership_late_available_for: str | None = None,
    membership_overlap: bool = False,
    statement_skip_keys: set[tuple[str, str]] = frozenset(),
):
    """Build a v2 world (production calendar + v2 store) with per-scenario
    knobs; defaults pass every p17 §12 source check."""

    from tests.conftest import (
        populate_v2_for_config,
        populate_v2_industry_tables,
        populate_v2_statement_tables,
    )

    config = _d7_config(tmp_path)
    _d7_calendar(config)
    # The D5 industry-index closes live in the main database (t14) — seed
    # them through the axis end so the healthy world is index-fresh.
    with AshareDB(config.duckdb_path) as db:
        db.execute(
            f"CREATE TABLE IF NOT EXISTS {config.sw_index_table} "
            "(index_code VARCHAR, industry_name VARCHAR, trade_date VARCHAR, "
            "close DOUBLE, PRIMARY KEY (index_code, trade_date))"
        )
        rows = []
        for date in D7_SESSIONS:
            rows.append(
                {"index_code": "801010", "industry_name": "A",
                 "trade_date": date, "close": 100.0}
            )
            rows.append(
                {"index_code": "801020", "industry_name": "B",
                 "trade_date": date, "close": 100.0}
            )
        db.register("_sw_index_rows", pd.DataFrame(rows))
        db.execute(
            f"INSERT INTO {config.sw_index_table} "
            "SELECT index_code, industry_name, trade_date, close "
            "FROM _sw_index_rows"
        )
    bar_axis = bar_dates if bar_dates is not None else D7_SESSIONS
    axis_dates, _ts_codes, bars_frame = make_bars(n_dates=len(D7_SESSIONS))
    assert axis_dates == D7_SESSIONS  # the bars axis IS the calendar axis
    v2_path = populate_v2_for_config(
        config,
        bar_axis,
        D7_UNIVERSE,
        bars_frame,
        drop_limit_for=(
            {
                (code, date)
                for code in D7_UNIVERSE
                for date in D7_SESSIONS[limit_through_index + 1:]
            }
            if limit_through_index is not None
            else None
        ),
        drop_adj_for=(
            {
                (code, date)
                for code in D7_UNIVERSE
                for date in D7_SESSIONS[drop_adj_from_index:]
            }
            if drop_adj_from_index is not None
            else None
        ),
    )
    ensure_v2_store_tables(v2_path, margin_rows=_d7_margin_rows(
        through_index=9 if margin_through_index is None else margin_through_index
    ))
    membership_rows, taxonomy_rows = _d7_membership_rows(
        late_available_for=membership_late_available_for,
        overlap=membership_overlap,
    )
    populate_v2_industry_tables(
        v2_path, membership_rows=membership_rows, taxonomy_rows=taxonomy_rows
    )
    populate_v2_statement_tables(
        v2_path,
        income_rows=_d7_statement_rows(
            "income_statement_pit", skip_keys=set(statement_skip_keys)
        ),
        balance_rows=_d7_statement_rows(
            "balance_sheet_pit", skip_keys=set(statement_skip_keys)
        ),
        cashflow_rows=_d7_statement_rows(
            "cashflow_statement_pit", skip_keys=set(statement_skip_keys)
        ),
    )
    if save_manifest:
        from ashare_data.manifest import (
            TUSHARE_V2_CONFIG,
            build_v2_dataset_manifest,
            save_manifest,
        )

        with AshareDB(v2_path) as db:
            save_manifest(
                db,
                TUSHARE_V2_CONFIG,
                build_v2_dataset_manifest(db),
            )
    return config, v2_path


def test_d7_thresholds_are_the_p17_section12_values():
    """p17 §12 preregistered numbers live in the data-gate authority."""
    from ashare_data import gates

    assert gates.DAILY_ELIGIBILITY_MAX_LAG_SESSIONS == 3
    assert gates.PRICE_LIMIT_MAX_LAG_SESSIONS == 3
    assert gates.MARGIN_MAX_LAG_SESSIONS == 2
    assert gates.INDUSTRY_INDEX_MAX_LAG_SESSIONS == 3
    assert gates.INDUSTRY_MEMBERSHIP_MIN_COVERAGE == 0.98
    assert gates.FINANCIAL_REPORT_COVERAGE_MIN == 0.90


def test_d7_required_source_resolution(tmp_path: Path):
    """Feature-registry resolution: samples depending on daily bars,
    industry, financials and margin map to their p17 §12 sources; an
    unknown feature is a fail-closed contract violation."""
    from ashare_model.data_eligibility import (
        SOURCE_DAILY,
        SOURCE_FINANCIAL,
        SOURCE_INDUSTRY_INDEX,
        SOURCE_INDUSTRY_MEMBERSHIP,
        SOURCE_MARGIN,
        DataEligibilityError,
        required_source_families,
    )

    assert required_source_families(["RET_5", "TURNOVER"]) == frozenset(
        {SOURCE_DAILY}
    )
    assert required_source_families(
        ["CASHFLOW_QUALITY", "GROSS_MARGIN"]
    ) == frozenset({SOURCE_FINANCIAL})
    assert required_source_families(["MARGIN_BALANCE_CHG"]) == frozenset(
        {SOURCE_MARGIN}
    )
    assert required_source_families(
        ["IND_REL_RET_5", "INDUSTRY_MOMENTUM"]
    ) == frozenset({SOURCE_INDUSTRY_MEMBERSHIP, SOURCE_INDUSTRY_INDEX})
    assert required_source_families(["NORTHBOUND_CHG"]) == frozenset()
    with pytest.raises(DataEligibilityError):
        required_source_families(["NOT_A_FEATURE"])


def test_d7_healthy_world_passes(tmp_path: Path):
    """A healthy v2 store passes every required source check for a factor
    set that depends on all families."""
    from ashare_model.data_eligibility import (
        SOURCE_DAILY,
        SOURCE_FINANCIAL,
        SOURCE_INDUSTRY_INDEX,
        SOURCE_INDUSTRY_MEMBERSHIP,
        SOURCE_MARGIN,
        SOURCE_PRICE_LIMIT,
        SOURCE_DATASET_MANIFEST,
        check_runspec_data_eligibility,
    )

    config, _ = _d7_world(tmp_path)
    spec = make_test_spec()
    features = [
        "RET_5",
        "CASHFLOW_QUALITY",
        "MARGIN_BALANCE_CHG",
        "IND_REL_RET_5",
        "NORTHBOUND_CHG",
    ]
    verdict = check_runspec_data_eligibility(
        spec, features, D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert verdict.ok, verdict.reasons
    assert verdict.spec_id == spec.spec_id
    for source in (
        SOURCE_DAILY,
        SOURCE_PRICE_LIMIT,
        SOURCE_FINANCIAL,
        SOURCE_MARGIN,
        SOURCE_INDUSTRY_MEMBERSHIP,
        SOURCE_INDUSTRY_INDEX,
        SOURCE_DATASET_MANIFEST,
    ):
        assert verdict.checks[source].required
        assert verdict.checks[source].ok, verdict.checks[source].detail


def test_d7_rejects_when_required_source_fails_but_spares_unaffected(tmp_path):
    """A stale margin source rejects only RunSpecs whose factor set
    depends on margin; a daily-only factor set on the same store is
    unaffected (p17 §12: the gate is per-formula-dependency)."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    config, _ = _d7_world(tmp_path, margin_through_index=5)  # lag 3 > 2
    spec = make_test_spec()
    margin_verdict = check_runspec_data_eligibility(
        spec, ["MARGIN_BALANCE_CHG"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not margin_verdict.ok
    assert any(
        "margin_detail_pit" in reason for reason in margin_verdict.reasons
    ), margin_verdict.reasons

    daily_verdict = check_runspec_data_eligibility(
        spec, ["RET_5"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert daily_verdict.ok
    assert not daily_verdict.checks["margin_detail_pit"].required


def test_d7_no_global_all_sources_must_be_fresh_gate(tmp_path: Path):
    """p17 §12 non-goal: broken margin/industry/financial sources never
    affect a factor set that does not depend on them — no global G9."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    config, _ = _d7_world(
        tmp_path,
        margin_through_index=0,  # margin badly stale
        membership_late_available_for=D7_UNIVERSE[0],  # industry coverage < 98%
        statement_skip_keys={
            (code, period)
            for code in D7_UNIVERSE
            for period in D7_QUARTER_ENDS
        },  # financial coverage 0
    )
    verdict = check_runspec_data_eligibility(
        make_test_spec(), ["RET_5", "NORTHBOUND_CHG"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert verdict.ok, verdict.reasons
    for source in (
        "margin_detail_pit",
        "industry_membership_pit",
        "industry_index",
        "financial_version_tables",
    ):
        assert not verdict.checks[source].required
        assert verdict.checks[source].ok  # not evaluated, not a rejection


def test_d7_uncomputable_sources_fail_never_pass(tmp_path: Path):
    """p17 §12: 一切源 来源不可计算一律 fail 不得 pass — a missing v2
    store fails every required source with an explicit not-computable
    reason."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    config = _d7_config(tmp_path)  # no calendar db, no v2 store at all
    verdict = check_runspec_data_eligibility(
        make_test_spec(), ["RET_5"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict.ok
    for source in ("daily_bar_raw", "daily_price_limit", "dataset_manifest"):
        assert not verdict.checks[source].ok
        assert "not computable" in verdict.checks[source].detail


def test_d7_daily_family_worst_member_fails_and_boundary_passes(tmp_path):
    """The DAILY family (daily_bar_raw + daily_adj_factor +
    daily_basic_pit, the p17 §13 same-per-day pull) fails on its
    laggard; lag exactly at the preregistered tolerance passes."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    # adj_factor rows only through axis[4]: available_session = axis[5],
    # lag = 9 - 5 = 4 > 3 -> the family fails on its laggard member.
    config, _ = _d7_world(tmp_path, drop_adj_from_index=5)
    verdict = check_runspec_data_eligibility(
        make_test_spec(), ["RET_5"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict.ok
    assert any("daily_bar_raw" in reason for reason in verdict.reasons)

    # Boundary: adj_factor through axis[5] -> available axis[6], lag 3
    # <= 3 -> passes (the preregistered tolerance is inclusive).
    config_ok, _ = _d7_world(tmp_path / "boundary", drop_adj_from_index=6)
    verdict_ok = check_runspec_data_eligibility(
        make_test_spec(), ["RET_5"], D7_UNIVERSE, config_ok,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert verdict_ok.ok, verdict_ok.reasons


def test_d7_price_limit_and_margin_boundaries(tmp_path):
    """Per-source lag boundaries: price-limit tolerance 3 (same-session
    availability) and margin tolerance 2 (T+1 availability)."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    # Price-limit rows only through axis[5] -> same-session available
    # axis[5], lag 4 > 3 -> fails (the always-required execution source).
    config, _ = _d7_world(tmp_path, limit_through_index=5)
    verdict = check_runspec_data_eligibility(
        make_test_spec(), ["RET_5"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict.ok
    assert any("daily_price_limit" in reason for reason in verdict.reasons)

    # Margin rows through axis[6] -> available axis[7], lag 2 <= 2 -> the
    # margin source passes at the preregistered boundary.
    config_ok, _ = _d7_world(tmp_path / "boundary", margin_through_index=6)
    verdict_ok = check_runspec_data_eligibility(
        make_test_spec(), ["MARGIN_BALANCE_CHG"], D7_UNIVERSE, config_ok,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert verdict_ok.ok, verdict_ok.reasons


def test_d7_industry_membership_coverage_and_overlap(tmp_path):
    """p17 §12: 有效区间覆盖 ≥ 98% 且区间重叠 = 0 — availability lag
    depresses coverage and an overlapping interval is a red-line
    violation; both reject the industry-dependent RunSpec."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    config, _ = _d7_world(
        tmp_path, membership_late_available_for=D7_UNIVERSE[0]
    )
    verdict = check_runspec_data_eligibility(
        make_test_spec(), ["IND_REL_RET_5"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict.ok
    assert any(
        "industry_membership_pit" in reason for reason in verdict.reasons
    ), verdict.reasons

    config_overlap, _ = _d7_world(tmp_path / "overlap", membership_overlap=True)
    verdict_overlap = check_runspec_data_eligibility(
        make_test_spec(), ["IND_REL_RET_5"], D7_UNIVERSE, config_overlap,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict_overlap.ok


def test_d7_financial_report_period_coverage(tmp_path):
    """p17 §12: 财务关键字段 当前正式 scope 报告期覆盖 ≥ 90% — missing
    (code, report-period) keys push the coverage under the threshold and
    reject the financial-dependent RunSpec."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    # 3 codes × 4 quarter-ends = 12 expected keys; skipping 3 leaves
    # 9/12 = 0.75 < 0.90 -> fail.
    skip = {
        (D7_UNIVERSE[0], D7_QUARTER_ENDS[0]),
        (D7_UNIVERSE[1], D7_QUARTER_ENDS[1]),
        (D7_UNIVERSE[2], D7_QUARTER_ENDS[2]),
    }
    config, _ = _d7_world(tmp_path, statement_skip_keys=skip)
    verdict = check_runspec_data_eligibility(
        make_test_spec(), ["CASHFLOW_QUALITY"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict.ok
    assert any(
        "financial_version_tables" in reason for reason in verdict.reasons
    ), verdict.reasons


def test_d7_manifest_integrity_is_required(tmp_path):
    """A v2 store without a dataset manifest fails the identity check
    even when every source table is healthy (manifest 完整性)."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility

    config, _ = _d7_world(tmp_path, save_manifest=False)
    verdict = check_runspec_data_eligibility(
        make_test_spec(), ["RET_5"], D7_UNIVERSE, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict.ok
    assert any("dataset_manifest" in reason for reason in verdict.reasons)


def test_d7_financial_expected_excludes_pre_listing_quarters(tmp_path: Path):
    """t73 correctness fix (p17 section 12 measurement recipe): a mid-quarter-
    listed stock's PRE-LISTING quarters are a source gap - no reports
    exist before listing (t72 sampling: every sampled (code, period) gap
    was pre-listing) - so they must not count in the expected coverage
    denominator.  Post-listing gaps stay counted (delisting/suspension
    periods are honestly missing).  The 90 percent threshold is
    unchanged - a correctness fix, not a relaxation."""
    from ashare_model.data_eligibility import check_runspec_data_eligibility
    from tests.conftest import V2_STATEMENT_COLUMNS, populate_v2_statement_tables

    universe = ['000001.SZ', '600000.SH']
    listing_dates = {'000001.SZ': '20200101', '600000.SH': '20230701'}
    # STOCK_600000 lists mid-window (20230701): its 20230331/20230630
    # reports cannot exist (pre-listing source gap).
    periods_by_code = {
        '000001.SZ': D7_QUARTER_ENDS,
        '600000.SH': ['20230930', '20231231'],  # post-listing periods only
    }

    config, v2_path = _d7_world(tmp_path)

    def statement_rows(table):
        identity = (
            'ts_code', 'end_date', 'ann_date', 'f_ann_date',
            'available_date', 'report_type', 'comp_type',
            'update_flag', 'end_type', 'ingest_id',
        )
        value_columns = [
            c for c in V2_STATEMENT_COLUMNS[table] if c not in identity
        ]
        rows = []
        for code, periods in periods_by_code.items():
            for period in periods:
                row = {
                    'ts_code': code,
                    'end_date': period,
                    'ann_date': period,
                    'f_ann_date': period,
                    'available_date': period,
                    'report_type': '1',
                    'comp_type': '1',
                    'update_flag': '0',
                    'end_type': '1',
                    'ingest_id': 'ing-' + code + '-' + period,
                }
                for column in value_columns:
                    row[column] = 1.0
                rows.append(row)
        return rows

    # One populate call carries all three tables (the helper wipes and
    # re-inserts all of them per call).
    populate_v2_statement_tables(
        v2_path,
        income_rows=statement_rows('income_statement_pit'),
        balance_rows=statement_rows('balance_sheet_pit'),
        cashflow_rows=statement_rows('cashflow_statement_pit'),
    )

    # Listing dates ride the production stocks table (the single source).
    with AshareDB(config.duckdb_path) as db:
        db.execute('DELETE FROM stocks')
        db.register(
            '_listing_rows',
            pd.DataFrame([
                {'ts_code': code, 'name': code, 'industry': None,
                 'list_date': date, 'is_st': False}
                for code, date in listing_dates.items()
            ]),
        )
        db.execute(
            'INSERT INTO stocks SELECT ts_code, name, industry, list_date, '
            'is_st FROM _listing_rows'
        )

    # The store content changed after _d7_world's manifest: rebuild it so
    # the manifest integrity check reflects the actual store.
    from ashare_data.manifest import (
        TUSHARE_V2_CONFIG,
        build_v2_dataset_manifest,
        save_manifest,
    )
    with AshareDB(v2_path) as db:
        save_manifest(db, TUSHARE_V2_CONFIG, build_v2_dataset_manifest(db))

    verdict = check_runspec_data_eligibility(
        make_test_spec(resolved_thresholds={
            'min_eligible': 1.0, 'factor_min_coverage': 0.90,
        }),
        ['CASHFLOW_QUALITY'], universe, config,
        evaluation_day=D7_EVALUATION_DAY,
    )
    # B's pre-listing quarters are excluded from the expected denominator:
    # post-listing coverage is complete, so the gate passes.
    assert verdict.ok, verdict.reasons
    assert 'pre-listing' in verdict.checks['financial_version_tables'].detail

    # Non-relaxation proof: POST-listing missing periods still fail.
    config_b, v2_b = _d7_world(tmp_path / 'b', save_manifest=False)
    skipped = {
        ('000001.SZ', '20230630'),
        ('000001.SZ', '20230930'),
    }
    populate_v2_statement_tables(
        v2_b,
        income_rows=_d7_statement_rows(
            'income_statement_pit', skip_keys=skipped),
        balance_rows=_d7_statement_rows(
            'balance_sheet_pit', skip_keys=skipped),
        cashflow_rows=_d7_statement_rows(
            'cashflow_statement_pit', skip_keys=skipped),
    )
    verdict_b = check_runspec_data_eligibility(
        make_test_spec(), ['CASHFLOW_QUALITY'], D7_UNIVERSE, config_b,
        evaluation_day=D7_EVALUATION_DAY,
    )
    assert not verdict_b.ok
    assert any(
        'financial_version_tables' in reason for reason in verdict_b.reasons
    )
