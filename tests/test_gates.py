"""Tests for the centralized ProductionGateRunner (T0-03)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
from tests.conftest import DEFAULT_CODES, make_bars

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
