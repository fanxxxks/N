"""P56 WP18 signal-vs-account RED/GREEN tests (WP 18).

Frozen spec: ``reports/wp18/FROZEN_18_PLAN.md`` (pre-registered
2026-09-27, read-only for the implementer).  Every assertion below
cites the frozen plan section it pins.  Written BEFORE any WP18
implementation exists; the verbatim RED output is preserved in
``reports/p56_red_evidence.txt`` (append-only).

Groups:

* §1/§4/§10.2 — candidate registry, per-stage expected row-key sets
  (行集物化), the formal-row budget accounting (≤300) and the
  experiment_integrity additive registration (materialize + precheck);
* §3.2 — the hold mode: constructor opt-in ``not_due_forced_exit`` /
  ``hold_after_first_funding``, the ``HoldModeConstructor`` entry, the
  engine execution gate, and the default-path byte-identity red line;
* §2.4 — frozen-direction consumption: the account/stress stages read
  the stage-1 ``direction_frozen.json`` only; re-learning is ignored
  (stub proof) and conflicting re-learn artifacts are refused;
* §5/§6.2/§7 — the five-question math: quantile spread (Q1),
  cash-exposure decomposition on post-trade holdings (Q4), annualized
  dynamic-minus-hold and the mechanical screen, drop-max-year variant,
  and the two Q5 cost readings (fixed-path repricing vs re-execution).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_portfolio.constructor import PortfolioConstructor

# --- RED: the WP18 surface does not exist yet ----------------------------
from ashare_model.wp18_signal_vs_account import (  # noqa: E402
    FORMAL_EVALUATION_ROW_CAP,
    SLOW_COMBO_ID,
    SLOW_SINGLES,
    WP18_MODES,
    WP18_RULES,
    WP18_STAGES,
    FrozenDirectionBook,
    FrozenDirectionRefused,
    frozen_oriented_signal,
    load_frozen_directions,
    wp18_candidates,
    wp18_formal_row_budget,
    wp18_planned_arms,
    wp18_stage_row_keys,
    write_frozen_directions,
)
from ashare_model.wp18_five_questions import (  # noqa: E402
    annualized_from_cumulative,
    arithmetic_cumulative,
    cash_exposure_decomposition,
    dynamic_minus_hold_annualized,
    drop_max_contribution_year,
    geometric_cumulative_excess,
    hold_forced_exit_session_count,
    mechanical_screen,
    quantile_spread,
    reprice_fixed_path_fees,
    split_cost_q5_effects,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _tiny_constructor(top_n: int = 2) -> PortfolioConstructor:
    return PortfolioConstructor(
        BacktestConfig(
            top_n=top_n,
            buy_rank=top_n,
            sell_rank=top_n + 1,
            single_weight_cap=1.0 / top_n,
        )
    )


def _hold_entry(top_n: int = 2):
    from ashare_portfolio.constructor import HoldModeConstructor

    return HoldModeConstructor(
        BacktestConfig(
            top_n=top_n,
            buy_rank=top_n,
            sell_rank=top_n + 1,
            single_weight_cap=1.0 / top_n,
        )
    )


N = 6


def _base_arrays(prev_nonzero: int = 2):
    """A 6-name cross-section: names 0..prev_nonzero-1 held, all eligible."""
    rng = np.random.default_rng(7)
    signal = np.arange(N, dtype=np.float64)  # name 5 ranks highest
    prev = np.zeros(N)
    prev[:prev_nonzero] = 0.5
    eligible = np.ones(N, dtype=bool)
    buy_blocked = np.zeros(N, dtype=bool)
    sell_blocked = np.zeros(N, dtype=bool)
    keys = np.asarray([f"S{i:02d}" for i in range(N)])
    return signal, prev, eligible, buy_blocked, sell_blocked, keys, rng


def _engine_raw_cache(n_stocks: int, n_dates: int) -> dict:
    """A p19-shaped raw cache: raw open prices with mild growth, no limits,
    no suspensions, positive volume."""
    open_ = np.full((n_stocks, n_dates), 10.0)
    open_[1, :] = 20.0
    open_[2, :] = 30.0
    if n_stocks > 3:
        for i in range(3, n_stocks):
            open_[i, :] = 10.0 + i
    return {
        "open": open_,
        "adjusted_open": open_,
        "up_limit": np.full((n_stocks, n_dates), 1e9),
        "down_limit": np.full((n_stocks, n_dates), 1e-9),
        "suspended": np.zeros((n_stocks, n_dates), dtype=bool),
        "volume": np.full((n_stocks, n_dates), 1e6),
    }


# ==========================================================================
# Group A — frozen candidate registry, row sets, budget, registration §1/§4
# ==========================================================================


class TestFrozenRegistry:
    """§1: 8 singles = medium 3 + slow 5; 10 candidates; no new factors."""

    def test_medium_candidates_frozen(self):
        candidates = wp18_candidates("medium")
        assert [c[0] for c in candidates] == [
            "MOMENTUM_20",
            "RSQ_60",
            "REVERSAL_60",
            "combo:all3",
        ]
        combo = candidates[3]
        assert combo[1] == "combo"
        assert combo[2] == ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")

    def test_slow_candidates_frozen(self):
        candidates = wp18_candidates("slow")
        assert [c[0] for c in candidates] == [
            "ROE",
            "PE_TTM",
            "REVENUE_YOY",
            "DIVIDEND_YIELD",
            "MARKET_CAP",
            SLOW_COMBO_ID,
        ]
        combo = candidates[5]
        assert combo[1] == "combo"
        assert combo[2] == SLOW_SINGLES
        # §1.3: slow combo must NOT carry any medium member (the historical
        # r01_slow KeyError defect — precheck's documented case).
        assert not set(combo[2]) & {"MOMENTUM_20", "RSQ_60", "REVERSAL_60"}

    def test_rules_and_modes_frozen(self):
        assert dict(WP18_RULES) == {"A": "reference_return", "B": "label_ic"}
        assert WP18_MODES == ("dynamic", "hold")

    def test_unknown_domain_refused(self):
        with pytest.raises(ValueError):
            wp18_candidates("fast")
        with pytest.raises(ValueError):
            wp18_stage_row_keys("fast", "M1")


class TestStageRowSets:
    """§4: every stage's expected row-key set, count and uniqueness."""

    def test_stage_row_counts_match_frozen_budget_table(self):
        expected = {
            ("medium", "M1"): 30,
            ("medium", "M2"): 40,
            ("medium", "M3"): 90,
            ("medium", "C1"): 40,
            ("slow", "S1"): 50,
            ("slow", "S2"): 60,
            ("slow", "S3"): 130,
            ("slow", "C2"): 40,
        }
        assert WP18_STAGES == expected
        for (domain, stage), rows in expected.items():
            keys = wp18_stage_row_keys(domain, stage)
            assert len(keys) == rows, (domain, stage, len(keys))
            # uniqueness on the full §4 key tuple
            assert len(set(keys)) == rows
            for key in keys:
                assert key.domain == domain
                assert key.stage == stage

    def test_m1_rows_are_singles_x_rules_x_folds(self):
        keys = wp18_stage_row_keys("medium", "M1")
        singles = {"MOMENTUM_20", "RSQ_60", "REVERSAL_60"}
        assert {k.candidate for k in keys} == singles
        assert {k.rule for k in keys} == {"A", "B"}
        assert len({k.fold_test_end for k in keys}) == 5
        # combos never produce direction rows (§4.1)
        assert all("combo" not in k.candidate for k in keys)

    def test_m3_rows_have_dynamic_hold_and_controls(self):
        keys = wp18_stage_row_keys("medium", "M3")
        signal_keys = [k for k in keys if k.candidate.startswith("combo") or k.candidate in
                       {"MOMENTUM_20", "RSQ_60", "REVERSAL_60"}]
        controls = [k for k in keys if k.candidate.startswith("control:") or k.candidate.startswith("benchmark:")]
        assert len(signal_keys) == 80
        assert len(controls) == 10
        assert {k.mode for k in signal_keys} == {"dynamic", "hold"}
        assert {k.rule for k in signal_keys} == {"A", "B"}
        # every account row is 1x except the zero-cost benchmark (§3.5)
        assert all(
            k.cost_level == "1x"
            for k in keys
            if k.candidate != "benchmark:equal_weight"
        )
        # §3.5: controls are rule-agnostic and mode-agnostic
        assert all(k.rule is None and k.mode is None for k in controls)
        control_names = {k.candidate for k in controls}
        assert "benchmark:equal_weight" in control_names
        assert "control:cash" in control_names
        # benchmark at the zero-cost reference level, cash at 1x (§3.5)
        bench = [k for k in controls if k.candidate == "benchmark:equal_weight"]
        assert all(k.cost_level == "ref" for k in bench)
        cash = [k for k in controls if k.candidate == "control:cash"]
        assert all(k.cost_level == "1x" for k in cash)

    def test_c1_rows_are_zero_and_two_x_only(self):
        keys = wp18_stage_row_keys("medium", "C1")
        assert {k.cost_level for k in keys} == {"0x", "2x"}
        assert {k.candidate for k in keys} == {"combo:all3"}
        assert {k.rule for k in keys} == {"A", "B"}
        assert {k.mode for k in keys} == {"dynamic", "hold"}

    def test_formal_row_budget_accounting(self):
        # §4.6: formal rows = account (1x) + stress (0x/2x); measurement
        # rows (M1/M2/S1/S2) never enter the budget.
        medium = wp18_formal_row_budget(("medium",))
        slow = wp18_formal_row_budget(("slow",))
        both = wp18_formal_row_budget(("medium", "slow"))
        assert medium == 90 + 40
        assert slow == 130 + 40
        assert both == 300
        assert both <= FORMAL_EVALUATION_ROW_CAP == 300

    def test_measurement_stages_out_of_formal_budget(self):
        for domain, stage in (("medium", "M1"), ("medium", "M2"),
                              ("slow", "S1"), ("slow", "S2")):
            keys = wp18_stage_row_keys(domain, stage)
            formal = [k for k in keys if k.stage in ("M3", "S3", "C1", "C2")]
            assert formal == []


class TestIntegrityRegistration:
    """§10.2: additive experiment_integrity registration; the plan must
    pass WP14 precheck before any run."""

    def test_experiment_ids_registered(self):
        from ashare_model.experiment_integrity.types import EXPERIMENT_IDS

        assert "wp18_medium" in EXPERIMENT_IDS
        assert "wp18_slow" in EXPERIMENT_IDS

    def test_materialize_medium_plan(self, tmp_path):
        from ashare_model.experiment_integrity.plan import (
            materialize_plan,
            plan_from_json,
            plan_to_json,
        )

        plan = materialize_plan("wp18_medium")
        # measurement rows (M1 30 + M2 40) + formal rows (M3 90 + C1 40)
        assert plan.expected_row_count == 200
        assert plan.expected_row_count <= 300  # §10.2 mechanical checkpoint
        assert plan.research_domain == "medium_cross_section"
        arms = wp18_planned_arms("medium")
        assert [a.candidate for a in plan.arms] == [a.candidate for a in arms]
        # JSON round-trip preserves the row identity
        rebuilt = plan_from_json(plan_to_json(plan))
        assert rebuilt.expected_row_count == plan.expected_row_count

    def test_materialize_slow_plan(self):
        from ashare_model.experiment_integrity.plan import materialize_plan

        plan = materialize_plan("wp18_slow")
        assert plan.expected_row_count == 280  # 50 + 60 + 130 + 40
        assert plan.research_domain == "slow_fundamental"

    def test_precheck_passes_on_both_plans(self):
        from ashare_model.experiment_integrity.plan import materialize_plan
        from ashare_model.experiment_integrity.precheck import run_precheck

        for experiment in ("wp18_medium", "wp18_slow"):
            report = run_precheck(materialize_plan(experiment))
            assert report.ok, [f"{i.code}:{i.detail[:80]}" for i in report.issues]

    def test_precheck_bites_on_a_corrupted_wp18_plan(self):
        """The checks are not vacuous on WP18 plans: an out-of-domain
        member (the historical slow KeyError) is refused pre-load."""
        from ashare_model.experiment_integrity.plan import materialize_plan
        from ashare_model.experiment_integrity.precheck import run_precheck
        from ashare_model.experiment_integrity.types import PlannedArm

        plan = materialize_plan("wp18_slow")
        poisoned = list(plan.arms)
        poisoned[0] = PlannedArm(
            candidate="S1:A:MOMENTUM_20",
            kind="single",
            members=("MOMENTUM_20",),
            direction_rule=poisoned[0].direction_rule,
            cost_levels=poisoned[0].cost_levels,
            is_primary=True,
            members_source="hand-edit",
        )
        object.__setattr__(plan, "arms", tuple(poisoned))
        report = run_precheck(plan)
        assert not report.ok
        assert any(i.code == "member_outside_domain" for i in report.issues)


# ==========================================================================
# Group B — hold mode §3.2
# ==========================================================================


class TestNotDueForcedExit:
    """§3.2 implementation constraint 1: hoist forced_exit above the
    not_due early return, opt-in only, default byte-identical."""

    def test_default_not_due_path_is_byte_identical_today(self):
        """The defect, pinned: a held name that lost eligibility is NOT
        exited on a non-due session — reason 'not_due', prev carried
        verbatim, no forced-exit diagnostics (constructor.py:333-335 early
        return precedes the forced_exit computation at :357)."""
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        eligible = eligible.copy()
        eligible[0] = False  # held name 0 loses universe eligibility
        out = _tiny_constructor().construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=False,
        )
        assert out.reason == "not_due"
        np.testing.assert_array_equal(np.asarray(out.weights), prev)
        assert out.diagnostics["forced_exit_count"] == 0
        assert "hold_forced_exit" not in out.diagnostics
        assert float(np.asarray(out.sell_weights).sum()) == 0.0

    def test_optin_executes_forced_exit_leg_on_non_due_session(self):
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        eligible = eligible.copy()
        eligible[0] = False
        out = _tiny_constructor().construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=False, not_due_forced_exit=True,
        )
        assert out.reason == "not_due_forced_exit"
        assert float(np.asarray(out.weights)[0]) == 0.0
        assert float(np.asarray(out.weights)[1]) == pytest.approx(0.5)
        # sell leg priced, buy leg empty (§3.2: no buys, proceeds stay cash)
        assert float(np.asarray(out.sell_weights)[0]) == pytest.approx(0.5)
        assert float(np.asarray(out.buy_weights).sum()) == 0.0
        assert out.diagnostics["forced_exit_count"] == 1
        assert out.diagnostics["hold_forced_exit"] is True

    def test_optin_without_forced_exits_stays_plain_not_due(self):
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        out = _tiny_constructor().construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=False, not_due_forced_exit=True,
        )
        assert out.reason == "not_due"
        np.testing.assert_array_equal(np.asarray(out.weights), prev)

    def test_sell_blocked_ineligible_name_is_not_forced_out(self):
        """forced_exit = held & ~eligible & ~sell_blocked (§3.2, verbatim
        from constructor.py:357): a sell-blocked name keeps its weight."""
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        eligible = eligible.copy()
        eligible[0] = False
        sell_blocked = sell_blocked.copy()
        sell_blocked[0] = True
        out = _tiny_constructor().construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=False, not_due_forced_exit=True,
        )
        assert out.reason == "not_due"
        assert float(np.asarray(out.weights)[0]) == pytest.approx(0.5)


class TestHoldAfterFirstFunding:
    """§3.2 implementation constraint 3: post-funding due sessions close
    active re-selection; only the forced-exit leg trades."""

    def test_due_session_after_funding_holds_the_book(self):
        constructor = _tiny_constructor()
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        # signal ranks names 4,5 highest; held names are 0,1 — active
        # re-selection would rotate the whole book.
        out = constructor.construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=True,
            hold_after_first_funding=True,
        )
        assert out.reason in ("hold_forced_exit", "hold_no_trade")
        assert out.reason == "hold_no_trade"  # nothing ineligible
        np.testing.assert_array_equal(np.asarray(out.weights), prev)
        assert out.diagnostics["forced_exit_count"] == 0
        assert out.diagnostics["buffer_survivors"] == ()

    def test_due_session_after_funding_still_forces_risk_exits(self):
        constructor = _tiny_constructor()
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        eligible = eligible.copy()
        eligible[0] = False
        out = constructor.construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=True,
            hold_after_first_funding=True,
        )
        assert out.reason == "hold_forced_exit"
        assert float(np.asarray(out.weights)[0]) == 0.0
        assert float(np.asarray(out.weights)[1]) == pytest.approx(0.5)
        # no entry leg: names 4/5 (ranked top) are never bought
        assert float(np.asarray(out.weights)[4]) == 0.0
        assert float(np.asarray(out.weights)[5]) == 0.0
        assert out.diagnostics["forced_exit_count"] == 1

    def test_flag_off_keeps_normal_reselection(self):
        constructor = _tiny_constructor()
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        out = constructor.construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=True,
        )
        assert out.reason == "signal"
        # dynamic semantics unchanged: top-ranked names 5,4 enter
        assert float(np.asarray(out.weights)[5]) > 0.0
        assert float(np.asarray(out.weights)[4]) > 0.0


class TestHoldModeConstructor:
    """The hold strategy entry (§3.2): one stateful wrapper — first-funding
    session builds normally, every later session only runs risk exits.

    The wrapper owns the account lifecycle state, so post-funding tests
    FIRST drive a first-funding session (the engine always starts a run
    from an empty book — one instance per account run)."""

    def _first_funding(self, entry, **overrides):
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays(
            prev_nonzero=0
        )
        return entry.construct(
            signal, prev, capital=100000.0,
            eligible=overrides.get("eligible", eligible),
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=True,
        )

    def test_first_funding_session_builds_normally(self):
        entry = _hold_entry()
        out = self._first_funding(entry)
        assert out.reason == "signal"
        assert float(np.asarray(out.weights).sum()) > 0.0
        assert entry.first_funding_done is True

    def test_post_funding_due_session_does_not_rotate(self):
        entry = _hold_entry()
        first = self._first_funding(entry)
        # the funded book is the top-ranked names 5,4; the signal would
        # now rotate to any new ranking — the hold account must not.
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        prev = np.asarray(first.weights).copy()
        out = entry.construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=True,
        )
        assert out.reason == "hold_no_trade"
        np.testing.assert_array_equal(np.asarray(out.weights), prev)

    def test_post_funding_non_due_session_runs_exit_leg(self):
        entry = _hold_entry()
        first = self._first_funding(entry)  # book = names 5, 4
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        prev = np.asarray(first.weights).copy()
        eligible = eligible.copy()
        eligible[5] = False  # a HELD name loses eligibility
        out = entry.construct(
            signal, prev, capital=100000.0, eligible=eligible,
            buy_blocked=buy_blocked, sell_blocked=sell_blocked,
            stable_keys=keys, rebalance_due=False,
        )
        assert out.reason == "not_due_forced_exit"
        assert float(np.asarray(out.weights)[5]) == 0.0
        assert float(np.asarray(out.weights)[4]) == pytest.approx(0.5)

    def test_no_rebuy_after_full_liquidation(self):
        """§3.2: 'hold 模式首建仓后永不主动买入' — after every holding is
        forced out, a later due session stays in cash."""
        entry = _hold_entry()
        signal, prev, eligible, buy_blocked, sell_blocked, keys, _ = _base_arrays()
        # session 1: first funding on a due date
        first = entry.construct(
            signal, np.zeros(N), capital=100000.0, eligible=np.ones(N, dtype=bool),
            buy_blocked=np.zeros(N, dtype=bool),
            sell_blocked=np.zeros(N, dtype=bool),
            stable_keys=keys, rebalance_due=True,
        )
        assert first.reason == "signal"
        # session 2: everything forced out on a due date
        none_eligible = np.zeros(N, dtype=bool)
        second = entry.construct(
            signal, np.asarray(first.weights), capital=100000.0,
            eligible=none_eligible,
            buy_blocked=np.zeros(N, dtype=bool),
            sell_blocked=np.zeros(N, dtype=bool),
            stable_keys=keys, rebalance_due=True,
        )
        assert float(np.asarray(second.weights).sum()) == 0.0
        # session 3: eligibility restored — the hold account must NOT re-buy
        third = entry.construct(
            signal, np.zeros(N), capital=100000.0,
            eligible=np.ones(N, dtype=bool),
            buy_blocked=np.zeros(N, dtype=bool),
            sell_blocked=np.zeros(N, dtype=bool),
            stable_keys=keys, rebalance_due=True,
        )
        assert float(np.asarray(third.weights).sum()) == 0.0
        assert third.reason in ("hold_forced_exit", "hold_no_trade")


class TestEngineGate:
    """§3.2 implementation constraint 2: the engine execution gate admits
    the not_due_forced_exit reason so the sell leg + fees book."""

    def _run(self, constructor, rebalance_mask, universe_mask=None):
        from ashare_model.backtest import AshareBacktestEngine

        n_dates = 6
        raw = _engine_raw_cache(N, n_dates)
        signal = np.zeros((N, n_dates))
        signal[5, :] = 5.0
        signal[4, :] = 4.0
        signal[3, :] = 3.0
        signal[2, :] = 2.0
        signal[1, :] = 1.0
        signal[0, :] = 0.5
        cfg = BacktestConfig(
            top_n=2, buy_rank=2, sell_rank=3, single_weight_cap=0.5,
        )
        engine = (
            AshareBacktestEngine(cfg)
            if constructor is None
            else AshareBacktestEngine(cfg, portfolio_constructor=constructor)
        )
        return engine.run(
            signal,
            raw,
            [f"S{i:02d}" for i in range(N)],
            [f"2025-01-{d:02d}" for d in range(1, n_dates + 1)],
            universe_mask=(
                np.ones((N, n_dates), dtype=bool)
                if universe_mask is None
                else universe_mask
            ),
            rebalance_mask=np.asarray(rebalance_mask, dtype=bool),
            execution_delay=1,
        )

    def test_hold_entry_charges_fees_on_non_due_exit(self):
        """due build at t=0 (book = top-ranked names 5,4), non-due at
        t=1 with held name 4 ineligible on its entry day: the exit leg
        must trade and book fees (the historical early return suppressed
        it).  universe[4, 2:] keeps name 4 selectable at session 0
        (eligible needs columns 0 & 1) while failing it at session 1
        (columns 1 & 2)."""
        mask = np.zeros(6, dtype=bool)
        mask[0] = True  # due at session 0 only
        universe = np.ones((N, 6), dtype=bool)
        universe[4, 2:] = False  # held name 4 loses eligibility at t+1
        result = self._run(_hold_entry(), mask, universe)
        diag = result.construction_diagnostics
        assert diag[0].get("rebalance_due") is True
        assert diag[1].get("hold_forced_exit") is True
        assert diag[1]["forced_exit_count"] == 1
        # the sell executed: turnover and fees on session 1
        assert result.turnover[1] > 0.0
        assert result.cost_fractions[1] > 0.0
        assert result.buy_weights[1].sum() == 0.0
        # and later non-due sessions without forced exits stay free
        assert result.cost_fractions[2] == 0.0

    def test_default_engine_keeps_non_due_zero_cost(self):
        """Red line: without the opt-in the historical behavior is
        unchanged — a non-due session with an ineligible holding trades
        nothing and pays nothing."""
        mask = np.zeros(6, dtype=bool)
        mask[0] = True
        universe = np.ones((N, 6), dtype=bool)
        universe[4, 2:] = False
        result = self._run(None, mask, universe)
        assert result.cost_fractions[1] == 0.0
        assert result.turnover[1] == 0.0

    def test_hold_sessions_funnel_counter(self):
        diag = [
            {"rebalance_due": False, "hold_forced_exit": True},
            {"rebalance_due": True, "hold_forced_exit": True},
            {"rebalance_due": True},
            {"rebalance_due": False},
        ]
        assert hold_forced_exit_session_count(diag) == 2

    def test_evaluate_signal_accepts_constructor_kwarg(self):
        """The hold row entry: evaluate_signal passes the constructor
        through to the engine (§3.1 — the only row engine)."""
        import inspect

        from ashare_model.eval_metrics import evaluate_signal

        params = inspect.signature(evaluate_signal).parameters
        assert "portfolio_constructor" in params


# ==========================================================================
# Group C — frozen-direction consumption §2.4
# ==========================================================================


def _write_book(tmp_path: Path, *, frozen: bool = True, entries=None,
                domain: str = "medium") -> Path:
    if entries is None:
        entries = [
            {
                "candidate": "MOMENTUM_20",
                "rule": "A",
                "fold_test_end": "2021-12-31",
                "direction": -1,
                "provenance": {"rule": "reference_return"},
            },
            {
                "candidate": "MOMENTUM_20",
                "rule": "B",
                "fold_test_end": "2021-12-31",
                "direction": 1,
                "provenance": {"rule": "label_ic"},
            },
        ]
    payload = {
        "wp18_direction_schema_version": 1,
        "domain": domain,
        "rules": dict(WP18_RULES),
        "frozen": frozen,
        "entries": entries,
    }
    path = tmp_path / "direction_frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestFrozenDirectionBook:
    def test_round_trip_and_lookup(self, tmp_path):
        path = _write_book(tmp_path)
        book = load_frozen_directions(path)
        assert isinstance(book, FrozenDirectionBook)
        assert book.direction("MOMENTUM_20", "A", "2021-12-31") == -1
        assert book.direction("MOMENTUM_20", "B", "2021-12-31") == 1

    def test_missing_entry_refused_fail_closed(self, tmp_path):
        path = _write_book(tmp_path)
        book = load_frozen_directions(path)
        with pytest.raises(FrozenDirectionRefused):
            book.direction("RSQ_60", "A", "2021-12-31")
        with pytest.raises(FrozenDirectionRefused):
            book.direction("MOMENTUM_20", "A", "2022-12-31")

    def test_unfrozen_artifact_refused(self, tmp_path):
        path = _write_book(tmp_path, frozen=False)
        with pytest.raises(FrozenDirectionRefused):
            load_frozen_directions(path)

    def test_conflicting_relearn_entry_refused(self, tmp_path):
        """A stage-1 artifact carrying two directions for one
        (candidate, rule, fold) is a re-learn — refused, not averaged."""
        entries = [
            {"candidate": "MOMENTUM_20", "rule": "A",
             "fold_test_end": "2021-12-31", "direction": 1, "provenance": {}},
            {"candidate": "MOMENTUM_20", "rule": "A",
             "fold_test_end": "2021-12-31", "direction": -1, "provenance": {}},
        ]
        path = _write_book(tmp_path, entries=entries)
        with pytest.raises(FrozenDirectionRefused):
            load_frozen_directions(path)

    def test_completeness_against_stage_row_keys(self, tmp_path):
        """Loader + M1 expected keys: every single x rule x fold must be
        present before any account row may consume the book."""
        keys = wp18_stage_row_keys("medium", "M1")
        entries = []
        for key in keys:
            entries.append(
                {
                    "candidate": key.candidate,
                    "rule": key.rule,
                    "fold_test_end": key.fold_test_end,
                    "direction": 1,
                    "provenance": {},
                }
            )
        path = _write_book(tmp_path, entries=entries)
        book = load_frozen_directions(path, expected_keys=keys)
        assert book.size == len(keys)
        # drop one entry -> fail-closed refusal naming the gap
        entries.pop(0)
        path = _write_book(tmp_path, entries=entries)
        with pytest.raises(FrozenDirectionRefused) as excinfo:
            load_frozen_directions(path, expected_keys=keys)
        assert "missing" in str(excinfo.value).lower()

    def test_write_frozen_directions_round_trip(self, tmp_path):
        decisions = [
            {"candidate": "RSQ_60", "rule": "A", "fold_test_end": "2021-12-31",
             "direction": 1, "provenance": {"rule": "reference_return"}},
        ]
        path = write_frozen_directions(
            tmp_path / "out.json", domain="medium", decisions=decisions
        )
        book = load_frozen_directions(path)
        assert book.direction("RSQ_60", "A", "2021-12-31") == 1


class TestNoRelearn:
    """§2.4 红线: 方向每折学习一次，M1 产出后冻结；账户执行与成本压力
    一律消费同一份冻结方向，不得重学 —— stub proof."""

    def test_account_consumption_ignores_direction_learners(self, tmp_path,
                                                            monkeypatch):
        import ashare_model.direction as direction_module

        calls: list[str] = []

        def _sentinel(*args, **kwargs):
            calls.append(kwargs.get("rule", "unknown"))
            raise AssertionError(
                "re-learn attempted: the account stage must consume the "
                "frozen direction book only (FROZEN_18_PLAN §2.4)"
            )

        monkeypatch.setattr(
            direction_module, "reference_return_direction", _sentinel
        )
        monkeypatch.setattr(direction_module, "label_ic_direction", _sentinel)
        monkeypatch.setattr(direction_module, "decide_direction", _sentinel)

        path = _write_book(tmp_path)
        book = load_frozen_directions(path)
        raw = np.array([1.0, -2.0, 3.0])
        oriented = frozen_oriented_signal(
            raw,
            book=book,
            candidate="MOMENTUM_20",
            rule_letter="A",
            fold_test_end="2021-12-31",
        )
        np.testing.assert_allclose(oriented, -raw)  # frozen direction -1
        assert calls == []  # no learner touched

    def test_consumption_without_frozen_entry_refused(self, tmp_path):
        path = _write_book(tmp_path)
        book = load_frozen_directions(path)
        raw = np.array([1.0])
        with pytest.raises(FrozenDirectionRefused):
            frozen_oriented_signal(
                raw,
                book=book,
                candidate="RSQ_60",  # never frozen in this book
                rule_letter="A",
                fold_test_end="2021-12-31",
            )


# ==========================================================================
# Group D — five-question math §5/§6.2/§7
# ==========================================================================


class TestQuantileSpread:
    """§7.1 Q1: per-date cross-sectional quantile buckets of the oriented
    signal vs the fixed causal label; Q-high minus Q-low."""

    def _panel(self, n_dates: int = 12):
        rng = np.random.default_rng(11)
        n = 40
        target = rng.normal(size=(n, n_dates))
        signal = target + 0.1 * rng.normal(size=(n, n_dates))
        mask = np.ones((n, n_dates), dtype=bool)
        return signal, target, mask

    def test_monotone_signal_positive_spread(self):
        signal, target, mask = self._panel()
        out = quantile_spread(signal, target, universe_mask=mask, n_quantiles=5)
        assert out["n_dates"] == signal.shape[1]
        assert len(out["spread_series"]) == signal.shape[1]
        assert out["mean_spread"] > 0.0

    def test_sign_flip_negates_spread(self):
        signal, target, mask = self._panel()
        plus = quantile_spread(signal, target, universe_mask=mask, n_quantiles=5)
        minus = quantile_spread(-signal, target, universe_mask=mask, n_quantiles=5)
        assert plus["mean_spread"] == pytest.approx(-minus["mean_spread"], abs=1e-12)

    def test_universe_mask_gates_future_members(self):
        signal, target, mask = self._panel()
        poisoned = signal.copy()
        poisoned[0, :] *= 1000.0  # an extreme non-member value
        masked = mask.copy()
        masked[0, :] = False
        base = quantile_spread(poisoned, target, universe_mask=masked, n_quantiles=5)
        assert base["n_dates"] == signal.shape[1]
        # the same value WITH the mask on changes the result (the gate bites)
        ungated = quantile_spread(poisoned, target, universe_mask=mask, n_quantiles=5)
        assert base["mean_spread"] != pytest.approx(ungated["mean_spread"])

    def test_min_stocks_skips_thin_cross_sections(self):
        signal, target, mask = self._panel(4)
        thin = np.zeros_like(mask)
        thin[:5, 1:] = True  # only 5 eligible names after date 0
        out = quantile_spread(signal, target, universe_mask=thin, n_quantiles=5,
                              min_stocks=10)
        assert out["n_dates"] == 0
        assert out["mean_spread"] == 0.0


class TestCashExposureDecomposition:
    """§7.4 Q4: split net excess into exposure/cash vs selection using the
    ACTUAL post-trade holdings (not target weights)."""

    def test_fully_invested_equal_book_has_zero_both(self):
        # 4 names, book == equal-weight benchmark over the same names
        weights = np.full((4, 3), 0.25)
        rets = np.array(
            [[0.01, 0.02, -0.01], [0.02, -0.01, 0.0],
             [0.0, 0.01, 0.02], [-0.01, 0.0, 0.01]]
        )
        bench = rets.mean(axis=0)
        out = cash_exposure_decomposition(weights, rets, bench)
        np.testing.assert_allclose(out["exposure_effect"], 0.0, atol=1e-15)
        np.testing.assert_allclose(out["selection_effect"], 0.0, atol=1e-15)

    def test_pure_cash_book_is_all_exposure_effect(self):
        weights = np.zeros((4, 3))
        rets = np.array(
            [[0.01, 0.02, -0.01], [0.02, -0.01, 0.0],
             [0.0, 0.01, 0.02], [-0.01, 0.0, 0.01]]
        )
        bench = rets.mean(axis=0)
        out = cash_exposure_decomposition(weights, rets, bench)
        np.testing.assert_allclose(out["exposure_effect"], -bench)
        np.testing.assert_allclose(out["selection_effect"], 0.0, atol=1e-15)

    def test_half_invested_concentrated_book_splits_effects(self):
        # 50% cash + all weight on the best name: exposure drag AND
        # selection tilt both present, and they add to the total excess.
        weights = np.zeros((4, 2))
        weights[0, :] = 0.5
        rets = np.array(
            [[0.02, 0.04], [0.0, -0.02], [0.01, 0.0], [-0.01, 0.01]]
        )
        bench = rets.mean(axis=0)
        out = cash_exposure_decomposition(weights, rets, bench)
        actual = (weights * rets).sum(axis=0) - bench
        np.testing.assert_allclose(
            np.asarray(out["exposure_effect"]) + np.asarray(out["selection_effect"]),
            actual,
            atol=1e-15,
        )
        assert np.all(np.asarray(out["exposure_effect"]) < 0.0)   # under-invested drag
        assert np.all(np.asarray(out["selection_effect"]) > 0.0)  # best-name tilt

    def test_three_reported_scalars_stay_separate(self):
        """§7 通用口径: arithmetic cumulative, geometric cumulative excess
        and annualized are separate outputs, never merged."""
        weights = np.zeros((4, 5))
        weights[0, :] = 0.5
        rets = np.tile(np.array([[0.01], [0.0], [0.005], [-0.005]]), (1, 5))
        bench = np.full(5, 0.0025)
        out = cash_exposure_decomposition(weights, rets, bench)
        excess = (weights * rets).sum(axis=0) - bench
        assert out["arithmetic_cumulative_excess"] == pytest.approx(float(excess.sum()))
        assert out["geometric_cumulative_excess"] == pytest.approx(
            float(np.prod(1.0 + excess) - 1.0)
        )
        assert set(out) >= {
            "exposure_effect", "selection_effect",
            "arithmetic_cumulative_excess", "geometric_cumulative_excess",
            "annualized_excess",
        }


class TestMechanicalScreen:
    """§5/§6.2: the pre-registered resource-allocation gate, verbatim."""

    def test_all_three_conditions_pass(self):
        years = [0.02, 0.03, -0.01, 0.015, 0.025]  # 4/5 positive, median>0
        out = mechanical_screen(
            years, dynamic_cumulative=0.30, hold_cumulative=0.10, years=5
        )
        assert out["passed"] is True
        assert out["min_positive_years"] == 3
        assert out["n_positive_years"] == 4
        assert out["condition_ii_median_positive"] is True

    def test_two_positive_years_fails_condition_i(self):
        years = [0.02, 0.03, -0.01, -0.02, -0.025]
        out = mechanical_screen(years, 0.2, 0.1, years=5)
        assert out["passed"] is False
        assert out["condition_i_3_of_5_positive"] is False

    def test_negative_median_fails_condition_ii(self):
        years = [0.05, -0.01, -0.02, 0.06, -0.03]
        out = mechanical_screen(years, 0.2, 0.1, years=5)
        assert out["condition_ii_median_positive"] is False

    def test_dynamic_below_50bp_fails_condition_iii(self):
        years = [0.02, 0.03, -0.01, 0.015, 0.025]
        # both legs nearly identical -> annualized difference << 50bp
        out = mechanical_screen(years, 0.101, 0.100, years=5)
        assert out["condition_iii_dynamic_minus_hold_50bp"] is False
        assert out["passed"] is False
        assert out["dynamic_minus_hold_annualized"] == pytest.approx(
            dynamic_minus_hold_annualized(0.101, 0.100, years=5)
        )

    def test_annotation_is_the_frozen_disclaimer(self):
        out = mechanical_screen([0.01] * 5, 0.2, 0.1, years=5)
        assert "研发资源分配门槛" in out["annotation"]
        assert "非统计显著性证明" in out["annotation"]

    def test_drop_max_year_variant_reported(self):
        """§6.2: the gate is also reported without the largest
        contributing year — never only the one-line average."""
        years = [0.02, 0.03, -0.01, 0.015, 0.25]
        out = mechanical_screen(years, 0.30, 0.10, years=5)
        assert out["excluded_year_excess"] == pytest.approx(0.25)
        remaining = [y for y in years if y != 0.25]
        assert out["ex_minus_max_year"]["n_positive_years"] == sum(
            1 for y in remaining if y > 0
        )

    def test_drop_max_contribution_year_helper(self):
        years = [0.1, -0.05, 0.2, 0.0]
        remaining, dropped = drop_max_contribution_year(years)
        assert dropped == 0.2
        assert list(remaining) == [0.1, -0.05, 0.0]


class TestAnnualizationHelpers:
    def test_annualized_from_cumulative_matches_section_5_formula(self):
        # §5 (iii): 年化 = 各腿 (1+累计净超额)^(1/5)−1
        assert annualized_from_cumulative(0.10, years=5) == pytest.approx(
            1.10 ** 0.2 - 1.0
        )
        assert dynamic_minus_hold_annualized(0.10, 0.05, years=5) == pytest.approx(
            (1.10 ** 0.2 - 1.0) - (1.05 ** 0.2 - 1.0)
        )

    def test_arithmetic_and_geometric_kept_separate(self):
        series = [0.02, -0.01, 0.03]
        assert arithmetic_cumulative(series) == pytest.approx(0.04)
        assert geometric_cumulative_excess(series) == pytest.approx(
            1.02 * 0.99 * 1.03 - 1.0
        )
        assert arithmetic_cumulative(series) != geometric_cumulative_excess(series)


class TestCostQ5:
    """§7.5 Q5: fixed-path fee repricing vs re-execution stress, separate."""

    def test_zero_x_repricing_is_free(self):
        cfg = BacktestConfig(
            commission_rate=0.00025, min_commission=5.0,
            stamp_tax_rate=0.0005, transfer_fee_rate=0.00001,
            slippage_rate=0.0005,
        )
        fees = reprice_fixed_path_fees(
            buy_notionals=[10000.0], sell_notionals=[5000.0],
            backtest_config=cfg, level="0x",
        )
        assert fees == 0.0

    def test_two_x_costs_more_than_one_x(self):
        cfg = BacktestConfig(
            commission_rate=0.00025, min_commission=5.0,
            stamp_tax_rate=0.0005, transfer_fee_rate=0.00001,
            slippage_rate=0.0005,
        )
        common = dict(
            buy_notionals=[100000.0, 25000.0],
            sell_notionals=[40000.0],
            backtest_config=cfg,
        )
        one_x = reprice_fixed_path_fees(level="1x", **common)
        two_x = reprice_fixed_path_fees(level="2x", **common)
        assert one_x > 0.0
        assert two_x > one_x

    def test_min_commission_floor_unchanged_at_2x(self):
        """§3.3: 2x doubles the rates but the commission floor stays."""
        cfg = BacktestConfig(
            commission_rate=0.00025, min_commission=5.0,
            stamp_tax_rate=0.0005, transfer_fee_rate=0.00001,
            slippage_rate=0.0005,
        )
        common = dict(
            buy_notionals=[100.0],  # far below the floor
            sell_notionals=[100.0],
            backtest_config=cfg,
        )
        one_x = reprice_fixed_path_fees(level="1x", **common)
        two_x = reprice_fixed_path_fees(level="2x", **common)
        # buy side: commission floored at 5 both ways; the sell side carries
        # stamp/transfer/slippage which DO double, so 2x exceeds 1x but by
        # less than the full 2x of the floored leg.
        assert one_x > 0.0
        assert two_x > one_x

    def test_unknown_level_refused(self):
        cfg = BacktestConfig()
        with pytest.raises(ValueError):
            reprice_fixed_path_fees([1.0], [1.0], cfg, "3x")

    def test_q5_split_reports_direct_and_path_separately(self):
        out = split_cost_q5_effects(
            fixed_path_net_excess=0.10, reexecuted_net_excess=0.08
        )
        assert out["direct_fee_reading"] == 0.10
        assert out["reexecuted_reading"] == 0.08
        assert out["path_effect"] == pytest.approx(-0.02)
