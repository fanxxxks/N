"""Tests for the centralized ProductionGateRunner (T0-03)."""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.db import AshareDB
from ashare_data.gates import GateResult, ProductionGateRunner
from ashare_data.universe import UniverseContractError, member_bar_coverage
from tests.conftest import DEFAULT_CODES, make_bars


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
    result = runner.run(mode="formal")
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
    }
    assert all(check.ok for check in result.checks)
    assert result.status is not None and result.status.strict
    # require_production is the formal entry API and returns the status.
    status = runner.require_production()
    assert status.strict is True


def test_formal_gate_fails_on_zero_bar_historical_member(populated_db):
    _seed_zero_bar_member(populated_db)
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    with pytest.raises(UniverseContractError, match="G7"):
        runner.require_production()
    result = runner.run(mode="formal")
    assert not result.ok
    g7 = result.check("G7 every PIT member interval has daily bars")
    assert not g7.ok
    assert "600999.SH" in g7.detail


def test_formal_gate_fails_when_min_eligible_unreachable(populated_db):
    runner = ProductionGateRunner(populated_db, min_eligible=10_000)
    with pytest.raises(UniverseContractError, match="G6"):
        runner.require_production()
    result = runner.run(mode="formal")
    assert not result.check("G6 >= 10000 eligible stocks per major window").ok


def test_dev_mode_degrades_instead_of_raising(populated_db):
    _seed_zero_bar_member(populated_db)
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result: GateResult = runner.run(mode="dev")
    # Dev mode never raises and always reports degraded=true.
    assert result.mode == "dev"
    assert result.degraded is True
    assert not result.ok
    assert any("zero-bar" in warning for warning in result.warnings)
    # The failing check is visible, not swallowed.
    assert not result.check("G7 every PIT member interval has daily bars").ok


def test_dev_mode_clean_db_is_still_marked_degraded(populated_db):
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result = runner.run(mode="dev")
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
    result = runner.run(mode="formal")
    assert result.ok
    assert result.check("G7 every PIT member interval has daily bars").ok
