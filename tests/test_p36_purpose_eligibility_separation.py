"""P36 (work package 2): historical-research vs online-observation
eligibility separation (docs/p36_purpose_eligibility_separation_draft.md).

Four frozen acceptance semantics from the user's direct instruction:

1. the purpose explicitly enters the run identity (RunSpec.purpose is a
   required, spec_id-bearing field — never defaulted in production);
2. the same historical snapshot earns the SAME historical-research
   eligibility on any run day (the historical basis provably never reads
   the wall clock — the evaluation day is snapshot_as_of_day(cutoff));
3. missing bars at the snapshot window end, identity drift, PIT and
   trading-constraint problems still reject (G1–G7, A02 confirm, the D7
   thresholds are untouched);
4. online observation on stale data still rejects (the default basis is
   the stricter online/wall-clock one; run_sim pins it).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pandas as pd
import pytest

from ashare_data.db import AshareDB
from ashare_data.gates import ProductionGateRunner
from ashare_data.universe import UniverseContractError
from tests.conftest import make_bars, make_test_spec

ROOT = Path(__file__).resolve().parents[1]


def _g8(result):
    matches = [c for c in result.checks if c.name.startswith("G8")]
    assert matches, [c.name for c in result.checks]
    return matches[0]


def _stale_calendar(config, *, extra_sessions: int = 9):
    """The p15 incident shape (shared with test_gates): the calendar
    pre-lists sessions beyond the daily-bar horizon, so an online-basis
    evaluation day at the far end measures lag = extra_sessions."""

    dates = make_bars()[0]
    extra = (
        pd.bdate_range("2024-01-01", periods=40 + extra_sessions)
        .strftime("%Y%m%d")
        .tolist()[40:]
    )
    with AshareDB(config.duckdb_path) as db:
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in extra],
            config,
        )
    return dates, extra


# -- R1: the purpose explicitly enters the run identity -----------------------


def _minimal_runspec_kwargs():
    """The complete build_runspec argument set (mirrors conftest's
    make_test_spec) — used to prove `purpose` is a required argument, so
    the omission TypeError cannot come from some other missing field."""

    from ashare_model.runspec import WindowSpec
    from ashare_portfolio.execution_spec import portfolio_config_provenance
    from ashare_data.config import BacktestConfig

    return dict(
        dataset_id="ds-p36",
        data_cutoff="20241231",
        frequency="daily",
        target="forward_return",
        horizon=1,
        requested_budget=1,
        n_folds=1,
        dev_window=WindowSpec(start="20240101", end="20240201"),
        preregistered_contract_hash="a" * 64,
        resolved_thresholds={"min_eligible": 1.0},
        portfolio_config=portfolio_config_provenance(BacktestConfig()),
        seeds=(42,),
        searcher="random",
        repo_root=None,
        require_clean_tree=False,
    )


def test_r1_purpose_is_required_and_identity_bearing():
    from ashare_model.runspec import RUNSPEC_SCHEMA_VERSION, build_runspec

    assert RUNSPEC_SCHEMA_VERSION == 3

    # No default: callers must declare the purpose (p36 §2.1 — a wrong
    # silent default IS a wrong eligibility basis).
    signature = inspect.signature(build_runspec)
    assert "purpose" in signature.parameters
    assert signature.parameters["purpose"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="purpose"):
        build_runspec(**_minimal_runspec_kwargs())

    historical = make_test_spec(purpose="historical_research")
    online = make_test_spec(purpose="online_observation")
    assert historical.purpose == "historical_research"
    assert historical.to_payload()["purpose"] == "historical_research"
    # The purpose differentiates the spec identity (p36 §2.1).
    assert historical.spec_id != online.spec_id


def test_r1_unknown_purpose_refused():
    from ashare_model.runspec import build_runspec

    with pytest.raises(ValueError, match="purpose"):
        build_runspec(purpose="bogus", **_minimal_runspec_kwargs())


def test_r7_v2_payload_without_purpose_refused_as_lineage():
    """Migration/rejection (p36 §4.3): a stored v2-shaped spec payload
    (no purpose) fails RunSpec validation — old runs stay audit-readable
    as raw JSON but can never attach as lineage sources."""

    from pydantic import ValidationError

    from ashare_model.runspec import RunSpec

    payload = make_test_spec().to_payload()
    del payload["purpose"]
    with pytest.raises(ValidationError):
        RunSpec.model_validate(payload)


# -- R2: the historical basis is a pure function of the snapshot --------------


def test_r2_historical_basis_never_reads_the_wall_clock(populated_db, monkeypatch):
    import ashare_data.gates as gates_mod

    cutoff = make_bars()[0][-1]  # the fixture calendar == the bar horizon

    def _boom():
        raise AssertionError(
            "the historical basis must not read the wall clock (p36 §2.2)"
        )

    monkeypatch.setattr(gates_mod, "_shanghai_today", _boom)
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    result = runner.run(
        mode="formal",
        purpose="historical_research",
        snapshot_cutoff=cutoff,
    )
    assert result.ok
    g8 = _g8(result)
    assert g8.ok
    # Auditable basis (p36 §2.6).
    assert "basis=historical_snapshot" in g8.detail
    assert f"data_cutoff={cutoff}" in g8.detail


def test_r8_online_detail_carries_the_basis(populated_db):
    result = ProductionGateRunner(populated_db, min_eligible=3).run(
        mode="formal", today=make_bars()[0][-1]
    )
    assert "basis=online_wall_clock" in _g8(result).detail


# -- R3: the historical basis keeps its teeth ---------------------------------


def test_r3_1_historical_basis_rejects_missing_window_end_bars(populated_db):
    """The snapshot claims data through a cutoff the primary bars do not
    reach (all 9 pre-listed sessions missing) — still a refusal, on any
    run day.  Note the historical reference is the cutoff session itself
    (eval day = cutoff + 1), so the lag counts every missing session up
    to the cutoff: 9, not the online case's 8 (whose reference sits one
    session earlier)."""

    dates, extra = _stale_calendar(populated_db)
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    with pytest.raises(UniverseContractError, match="G8"):
        runner.require_production(
            purpose="historical_research", snapshot_cutoff=extra[-1]
        )
    result = runner.run(
        mode="formal", purpose="historical_research", snapshot_cutoff=extra[-1]
    )
    g8 = _g8(result)
    assert not g8.ok
    assert "lag=9" in g8.detail


def test_r3_2_historical_basis_requires_a_valid_cutoff(populated_db):
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    with pytest.raises(ValueError, match="snapshot_cutoff"):
        runner.require_production(
            purpose="historical_research", snapshot_cutoff=None
        )
    with pytest.raises(ValueError, match="snapshot_cutoff"):
        runner.require_production(
            purpose="historical_research", snapshot_cutoff="20241399"
        )


def test_r3_3_basis_mixing_refused(populated_db):
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    cutoff = make_bars()[0][-1]
    with pytest.raises(ValueError, match="today"):
        runner.run(
            mode="formal",
            today=cutoff,
            purpose="historical_research",
            snapshot_cutoff=cutoff,
        )
    with pytest.raises(ValueError, match="snapshot_cutoff"):
        runner.run(
            mode="formal",
            purpose="online_observation",
            snapshot_cutoff=cutoff,
        )


def test_r3_4_horizon_exhaustion_guard_is_online_only(populated_db):
    """p36 §3.3: a cutoff at the calendar's last pre-listed session is a
    legitimate snapshot (reference = the cutoff session itself); the
    horizon guard stays an online-basis defense (p16 §2.2)."""

    cutoff = make_bars()[0][-1]  # fixture calendar max == cutoff
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    historical = runner.run(
        mode="formal", purpose="historical_research", snapshot_cutoff=cutoff
    )
    assert _g8(historical).ok
    future_today = (
        pd.bdate_range(cutoff, periods=3)[-1].strftime("%Y%m%d")
    )
    online = runner.run(mode="formal", today=future_today)
    assert not _g8(online).ok


# -- R4: online observation on stale data still rejects -----------------------


def test_r4_online_stale_still_rejects(populated_db):
    _stale_calendar(populated_db)
    today = make_bars(49)[0][-1]
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    with pytest.raises(UniverseContractError, match="G8"):
        runner.require_production(today=today)


def test_r9_default_purpose_is_the_stricter_online_basis(populated_db):
    """One fixture, two verdicts: the same live DB with 9 pre-listed
    sessions of primary-bar lag rejects under the (default) online basis
    and passes under the explicitly declared historical basis with the
    snapshot's own cutoff — the separation in a nutshell."""

    dates, extra = _stale_calendar(populated_db)
    today = make_bars(49)[0][-1]
    runner = ProductionGateRunner(populated_db, min_eligible=3)
    with pytest.raises(UniverseContractError, match="G8"):
        runner.require_production(today=today)
    historical = runner.run(
        mode="formal", purpose="historical_research", snapshot_cutoff=dates[-1]
    )
    assert historical.ok


def test_r4b_run_sim_pins_the_online_basis():
    """run_sim's formal paper path declares no purpose/snapshot_cutoff:
    the default online basis (wall clock) IS its semantics (p36 §2.4)."""

    source = (ROOT / "ashare_trading" / "run_sim.py").read_text(
        encoding="utf-8"
    )
    calls = re.findall(r"require_production\([^)]*\)", source)
    assert calls, "run_sim must keep gating through require_production"
    for call in calls:
        assert "purpose" not in call, call
        assert "snapshot_cutoff" not in call, call


# -- R5: the D7 default evaluation day follows the spec purpose ---------------


def test_r5_d7_default_day_follows_spec_purpose(tmp_path, monkeypatch):
    from test_gates import D7_SESSIONS, D7_UNIVERSE, _d7_world

    import ashare_model.data_eligibility as de

    config, _v2_path = _d7_world(tmp_path)
    # Pre-list sessions beyond the sources' last collected session: a
    # wall-clock day anywhere after them measures daily lag = 9 > 3.
    extra = (
        pd.bdate_range("2024-01-01", periods=19).strftime("%Y%m%d").tolist()[10:]
    )
    with AshareDB(config.duckdb_path) as db:
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in extra], config
        )

    historical = make_test_spec(
        purpose="historical_research", data_cutoff=D7_SESSIONS[-1]
    )
    online = make_test_spec(
        purpose="online_observation", data_cutoff=D7_SESSIONS[-1]
    )

    def _verdicts(spec):
        first = de.check_runspec_data_eligibility(
            spec, ["RET_1"], list(D7_UNIVERSE), config
        )
        second = de.check_runspec_data_eligibility(
            spec, ["RET_1"], list(D7_UNIVERSE), config
        )
        return first, second

    # Two different wall-clock days — the historical verdict must be an
    # identical, clock-free function of (spec, data).
    monkeypatch.setattr(de, "_shanghai_today", lambda: "20250101")
    hist_a, hist_repeat = _verdicts(historical)
    monkeypatch.setattr(de, "_shanghai_today", lambda: "20260917")
    hist_b, _ = _verdicts(historical)
    online_verdict, _ = _verdicts(online)

    def _shape(verdict):
        return (
            verdict.ok,
            tuple(verdict.reasons),
            tuple(
                (name, check.required, check.ok, check.detail)
                for name, check in sorted(verdict.checks.items())
            ),
        )

    assert hist_a.ok, hist_a.reasons
    assert _shape(hist_a) == _shape(hist_repeat)
    assert _shape(hist_a) == _shape(hist_b)
    # The online purpose consumes the wall clock: 9 sessions of lag.
    assert not online_verdict.ok
    assert any(
        "lag 9 > 3" in reason for reason in online_verdict.reasons
    ), online_verdict.reasons


# -- R6: the formal historical entries wire the snapshot basis ----------------


def test_r6_entries_wire_the_historical_basis():
    train_src = (ROOT / "ashare_model" / "train_entry.py").read_text(
        encoding="utf-8"
    )
    assert "purpose=ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH" in train_src
    assert "snapshot_cutoff=spec.data_cutoff" in train_src

    evaluation_src = (ROOT / "ashare_model" / "evaluation.py").read_text(
        encoding="utf-8"
    )
    assert "purpose=ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH" in evaluation_src
    assert "snapshot_cutoff=spec.data_cutoff" in evaluation_src

    backtest_src = (ROOT / "ashare_model" / "backtest.py").read_text(
        encoding="utf-8"
    )
    assert "purpose=ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH" in backtest_src
    assert "snapshot_cutoff=source_spec.data_cutoff" in backtest_src

    r01_src = (ROOT / "ashare_model" / "r01_baseline.py").read_text(
        encoding="utf-8"
    )
    assert "purpose=ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH" in r01_src
    assert "snapshot_cutoff=spec.data_cutoff" in r01_src


def test_r6b_every_spec_factory_declares_its_purpose():
    producers = (
        "ashare_model/campaign_executor.py",
        "ashare_model/evaluation.py",
        "ashare_model/r01_baseline.py",
        "ashare_model/train_artifacts.py",
        "ashare_model/train_entry.py",
    )
    for rel in producers:
        source = (ROOT / rel).read_text(encoding="utf-8")
        calls = source.count("resolve_runtime_runspec(")
        declared = source.count('purpose="historical_research"')
        assert calls > 0, rel
        assert declared >= calls, (rel, calls, declared)
