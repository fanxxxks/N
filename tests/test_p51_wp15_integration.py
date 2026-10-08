"""P51 WP15 integration tests: the §1.1.4 execution layer and the wiring.

Every numeric assertion below is derived from the frozen contract
``reports/wp15/FROZEN_EXPECTATIONS.md`` (§ references in each docstring);
no assertion is written from implementation output (W5 rule, §7.1).

Covered surfaces (this file is the wiring/execution companion of
``test_p51_wp15_planner.py`` and ``test_p51_wp15_funding.py``):

* §1.1.4 / §1.8 — ``execute_trade_session``: settling a plan against a
  ledger with an INJECTABLE execution stub.  A failed buy keeps the real
  sells and the cash they raised (no fabricated rollback, no compensation
  trades) and is recorded in ``execution_failed_buys`` (sorted by
  stable_key); the §1.8 diagnostics surface the plan's reason codes.
* §1.5 — execution-time fees are the authority's totals on the merged
  (session, name, direction) notionals of the ACTUALLY FILLED orders,
  reconciling byte-for-byte with the plan when everything fills.
* §1.1.2 end-to-end — constructor (``trade_groups`` mode, unbudgeted
  path + label) -> pure planner -> executor, reconciled by actual fills
  and fees (§3.2 对账口径), never by target weights.
* §6.1/§6.2 (W4) — the fixed-target probe: the DEFAULT mode's priority
  allocator sells the book to empty in ten sessions (exits-first, zero
  buys) and the eleventh session then RE-GAINS the unbudgeted full-buy
  privilege (the frozen defect, pinned here as legacy behavior that must
  not change — red line 2); the run-level ``initial_funding_used`` flag
  and the ``trade_groups`` mode each close the loophole.
* §5 (W6) — the pure feasibility criterion function reproduces
  ``reports/wp15/config_feasibility.json`` for the slow reference config.
* Audit fix (2026-09-23, major) — the weight-space execution layers that
  do NOT consume the planner (backtest engine, paper sim, reward baskets)
  FAIL CLOSED on ``turnover_allocation="trade_groups"`` instead of
  silently executing the constructor's unbudgeted target weights — the
  mode must never be a budget bypass at a run entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ashare_data.config import BacktestConfig
from ashare_execution import ExecutionCostModel
from ashare_portfolio.constructor import PortfolioConstructor
from ashare_portfolio.trade_groups import plan_trade_session

CAPITAL = 100_000.0
ZERO_MODEL = ExecutionCostModel(
    commission_rate=0.0,
    min_commission=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_rate=0.0,
)
BASIS_POINT_MODEL = ExecutionCostModel(
    commission_rate=0.001,
    min_commission=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_rate=0.0,
)


def _execute(plan, keys, shares, cash, model, fill=None):
    """Lazy import so the executor tests genuinely FAIL before the
    execution layer exists (RED first), while the W4 probe pins below
    (which exercise already-landed constructor behavior) still run."""
    from ashare_portfolio.trade_groups import execute_trade_session

    return execute_trade_session(
        plan,
        stable_keys=np.asarray(keys, dtype=object),
        prev_shares=np.asarray(shares, dtype=np.int64),
        cash=float(cash),
        cost_model=model,
        fill=fill,
    )


def _plan(**kwargs):
    n = len(kwargs["keys"])
    defaults = dict(
        eligible=np.ones(n, dtype=bool),
        buy_blocked=np.zeros(n, dtype=bool),
        sell_blocked=np.zeros(n, dtype=bool),
        min_trade_amount=None,
        lot_size=100,
        initial_funding_used=None,
    )
    defaults.update(kwargs)
    return plan_trade_session(
        prev_shares=np.asarray(defaults.pop("shares"), dtype=np.int64),
        prices=np.asarray(defaults.pop("prices"), dtype=np.float64),
        cash=float(defaults.pop("cash")),
        target_weights=np.asarray(defaults.pop("weights"), dtype=np.float64),
        ranks=np.asarray(defaults.pop("ranks"), dtype=np.int64),
        stable_keys=np.asarray(defaults.pop("keys"), dtype=object),
        cost_model=defaults.pop("model"),
        turnover_budget=defaults.pop("budget"),
        **defaults,
    )


def _reject(*keys):
    """Execution stub that fails exactly the named names' orders."""

    def fill(key, side, shares, notional):  # noqa: ANN001
        return key not in set(keys)

    return fill


# --- §1.1.4/§1.5: the execution layer ---------------------------------------------


def test_executor_all_fills_reconcile_with_the_plan():
    """Frozen §3.2 + §1.5: with every order filled, execution must land on
    the frozen scenario-2 session-1 numbers — sell A 500 sh / 5,000 with
    fee 5.00, buy B 400 sh / 4,000 with fee 4.00, cash 991.00, equity
    99,991.00 — and the execution-time session fees must equal the plan's
    authority totals exactly (§1.5 telescoping identity: the sum of the
    incremental merged-notional authority calls IS the merged authority
    total, so planning affordability and execution accounting never
    diverge)."""
    plan = _plan(
        shares=[10_000, 0],
        prices=[10.0, 10.0],
        cash=0.0,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=["A", "B"],
        model=BASIS_POINT_MODEL,
        budget=0.10,
        min_trade_amount=0.0,
    )
    outcome = _execute(plan, ["A", "B"], [10_000, 0], 0.0, BASIS_POINT_MODEL)
    assert outcome.filled_orders == plan.executed_orders
    assert outcome.failed_buys == ()
    assert outcome.shares.tolist() == [9_500, 400]
    assert outcome.cash == pytest.approx(991.0)
    assert outcome.session_fees == {
        "sell": pytest.approx(5.0),
        "buy": pytest.approx(4.0),
        "total": pytest.approx(9.0),
    }
    assert outcome.session_fees["total"] == pytest.approx(plan.session_fees["total"])
    equity = outcome.cash + outcome.shares[0] * 10.0 + outcome.shares[1] * 10.0
    assert equity == pytest.approx(99_991.0)  # §3.2: 100,000 - 9 fees


def test_failed_buy_keeps_real_sell_and_cash_no_fabricated_rollback():
    """Frozen §1.1.4 (W2): when the buy fails at execution time the real
    sell and the cash it raised are KEPT — the A sell settles for real
    (shares reduced by 500, cash 5,000 - 5.00 fee = 4,995.00), B receives
    no shares, no compensation trade is issued, and the failure is
    recorded as execution_failed_buys == ("B",).  The executor must NOT
    restore the pre-session ledger (no fabricated rollback)."""
    plan = _plan(
        shares=[10_000, 0],
        prices=[10.0, 10.0],
        cash=0.0,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=["A", "B"],
        model=BASIS_POINT_MODEL,
        budget=0.10,
        min_trade_amount=0.0,
    )
    outcome = _execute(
        plan, ["A", "B"], [10_000, 0], 0.0, BASIS_POINT_MODEL, fill=_reject("B")
    )
    assert outcome.filled_orders == (("A", "sell", 500, 5000.0),)
    assert outcome.failed_buys == ("B",)
    assert outcome.shares.tolist() == [9_500, 0]  # the sell was real
    assert outcome.cash == pytest.approx(4_995.0)  # 5,000 - 5.00 sell fee
    assert outcome.session_fees == {
        "sell": pytest.approx(5.0),
        "buy": pytest.approx(0.0),
        "total": pytest.approx(5.0),
    }
    assert outcome.diagnostics["execution_failed_buys"] == ("B",)
    # No compensation trade: the filled ledger contains the sell only.
    assert [o for o in outcome.filled_orders if o[1] == "buy"] == []


def test_failed_buys_are_deduped_sorted_and_prior_sells_kept():
    """Frozen §1.1.4 + §1.5: two R-groups whose sells both settle but whose
    buys both fail — the sells stay (each group's sell precedes its buy in
    ledger order), the two failures are recorded once per name sorted by
    stable_key, and the merged A sell unit (4,000 + 1,000 = 5,000) is
    billed exactly once by the authority: fee 5.0, not the per-leg 4 + 2
    = 6.0 (§1.5 W3).  Session derivation: frozen §1.4 mechanics on the
    test_w3 market (E = 100,000 all in A; targets B1 0.03 / B2 0.97;
    commission 0.1% min 2.0; B_r = 10,000): group 1 = sell A 400/4,000 +
    buy B1 300/3,000, group 2 = sell A 100/1,000 + buy B2 100/1,000."""
    model = ExecutionCostModel(
        commission_rate=0.001,
        min_commission=2.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )
    plan = _plan(
        shares=[10_000, 0, 0],
        prices=[10.0, 10.0, 10.0],
        cash=0.0,
        weights=[0.0, 0.03, 0.97],
        ranks=[2, 0, 1],
        keys=["A", "B1", "B2"],
        model=model,
        budget=0.10,
    )
    assert plan.trade_groups[0]["sells"] == (("A", 400, 4000.0),)  # §1.4 sanity
    outcome = _execute(
        plan, ["A", "B1", "B2"], [10_000, 0, 0], 0.0, model, fill=_reject("B1", "B2")
    )
    assert outcome.failed_buys == ("B1", "B2")  # sorted by stable_key
    assert outcome.filled_orders == (
        ("A", "sell", 400, 4000.0),
        ("A", "sell", 100, 1000.0),
    )
    assert outcome.shares.tolist() == [9_500, 0, 0]
    # Real sells kept: cash carries BOTH net sell legs with merged billing.
    assert outcome.cash == pytest.approx((4000.0 - 4.0) + (1000.0 - 1.0))
    assert outcome.cash == pytest.approx(4_995.0)
    assert outcome.session_fees["sell"] == pytest.approx(5.0)
    assert outcome.session_fees["buy"] == pytest.approx(0.0)


def test_sell_fill_failure_is_fail_closed():
    """Frozen §1.1.4 models BUY failure only.  A stub that fails a SELL
    contradicts the planner's verified preconditions (§1.3.1: only names
    with a price and not sell-blocked are ever emitted; the engine's sell
    leg has no failure path, backtest.py:302-324), so the executor must
    fail closed (raise) rather than invent an unmodelled semantics."""
    plan = _plan(
        shares=[10_000, 0],
        prices=[10.0, 10.0],
        cash=0.0,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=["A", "B"],
        model=BASIS_POINT_MODEL,
        budget=0.10,
        min_trade_amount=0.0,
    )
    with pytest.raises(ValueError, match="1.1.4"):
        _execute(plan, ["A", "B"], [10_000, 0], 0.0, BASIS_POINT_MODEL, fill=_reject("A"))


def test_executor_validates_its_contract_inputs():
    """§1.1.4 surface hygiene: the executor takes exactly a
    TradeSessionPlan, an ExecutionCostModel fee authority (§0.3 red line
    3), unique stable_keys matching the ledger, and non-negative cash;
    an order naming an unknown key fails loudly instead of corrupting the
    ledger."""
    from ashare_portfolio.trade_groups import TradeSessionPlan

    plan = _plan(
        shares=[0],
        prices=[10.0],
        cash=1_000.0,
        weights=[1.0],
        ranks=[0],
        keys=["P"],
        model=ZERO_MODEL,
        budget=None,
    )
    empty = TradeSessionPlan()
    with pytest.raises(TypeError):
        _execute("not-a-plan", ["P"], [0], 0.0, ZERO_MODEL)
    with pytest.raises(TypeError):
        _execute(empty, ["P"], [0], 0.0, object())  # not the fee authority
    with pytest.raises(ValueError):
        _execute(empty, ["P", "P"], [0, 0], 0.0, ZERO_MODEL)  # duplicate keys
    with pytest.raises(ValueError):
        _execute(empty, ["P"], [0], -1.0, ZERO_MODEL)  # negative cash
    rigged = TradeSessionPlan(executed_orders=(("GHOST", "buy", 1, 10.0),))
    with pytest.raises(ValueError, match="GHOST"):
        _execute(rigged, ["P"], [0], 1_000.0, ZERO_MODEL)


def test_execution_diagnostics_surface_the_section_1_8_contract():
    """Frozen §1.8 (item: '预算未用满输出原因码到 diagnostics'): the
    execution outcome carries the full trade_groups diagnostics block with
    the §4.2 branch-1 values — forced exit executed budget-free, one
    pure-buy group consuming the whole 2,000 ordinary budget, session
    reason None (budget exactly exhausted, §1.7), leg code (Y, buy,
    budget) — plus the execution-layer key execution_failed_buys = ()."""
    plan = _plan(
        shares=[1000, 0],
        prices=[10.0, 10.0],
        cash=0.0,
        weights=[0.0, 1.0],
        ranks=[5, 0],
        keys=["X", "Y"],
        model=ZERO_MODEL,
        budget=0.2,
        min_trade_amount=0.0,
        eligible=[False, True],
    )
    outcome = _execute(plan, ["X", "Y"], [1000, 0], 0.0, ZERO_MODEL)
    assert dict(outcome.diagnostics) == {
        "turnover_budget_mode": "trade_groups",
        "forced_exit_count": 1,
        "forced_exit_executed": ("X",),
        "forced_exit_blocked": (),
        "deleveraging_sell_notional": 0.0,
        "deleveraging_skipped": (),
        "trade_groups": (
            {"sells": (), "buy": ("Y", 200, 2000.0), "budget_consumed": 2000.0},
        ),
        "budget_unspent_reason": None,
        "leg_skip_reasons": (("Y", "buy", "budget"),),
        "session_fees": {"sell": 0.0, "buy": 0.0, "total": 0.0},
        "turnover_budget_consumed": 2000.0,
        "execution_failed_buys": (),
        "executed_orders": (
            ("X", "sell", 1000, 10000.0),
            ("Y", "buy", 200, 2000.0),
        ),
    }
    assert outcome.shares.tolist() == [0, 200]
    assert outcome.cash == pytest.approx(8_000.0)


# --- §1.1.2 end-to-end: constructor -> planner -> executor ------------------------


def _constructor_ranks(signal, eligible, keys):
    """Mirror constructor.py:316-322 exactly — the caller-side rank the
    planner consumes must be the constructor's own ordering key (§1.2:
    'rank[i] 与 stable_keys[i]（构造器排序键）')."""
    signal = np.asarray(signal, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    keys = np.asarray(keys).astype(str)
    finite_eligible = np.isfinite(signal) & eligible
    rank = np.full(signal.shape, np.iinfo(np.int64).max, dtype=np.int64)
    if finite_eligible.any():
        candidates = np.flatnonzero(finite_eligible)
        order = np.lexsort((keys[candidates], -signal[candidates]))
        rank[candidates[order]] = np.arange(len(candidates), dtype=np.int64)
    return rank


def test_end_to_end_trade_groups_session_reconciles_by_actual_fills():
    """Frozen §1.1.1/§1.1.2 wiring + §3.2: one full session under the
    experimental mode — the constructor walks the unbudgeted path and only
    labels turnover_budget_mode = "trade_groups" (its own priority /
    proportional allocation never runs), the pure planner turns those
    target weights into the frozen financed group (sell A 500/5,000, buy
    B 400/4,000), and the executor settles it.  The reconciliation is by
    ACTUAL fills and fees (§3.2 对账口径): achieved B = 4,000 yuan on
    equity 99,991 — the 100% target weight must never be read as filled."""
    cfg = BacktestConfig(
        top_n=1,
        buy_rank=1,
        sell_rank=1,
        single_weight_cap=1.0,
        turnover_budget=0.10,
        min_trade_amount=1000.0,
        turnover_allocation="trade_groups",
    )
    keys = ["A", "B"]
    signal = np.asarray([1.0, 2.0])
    prev = np.asarray([1.0, 0.0])
    out = PortfolioConstructor(cfg).construct(
        signal,
        prev,
        capital=CAPITAL,
        eligible=np.ones(2, dtype=bool),
        buy_blocked=np.zeros(2, dtype=bool),
        sell_blocked=np.zeros(2, dtype=bool),
        stable_keys=np.asarray(keys, dtype=object),
    )
    # §1.1.2: label only — the constructor's own allocator never ran.
    assert out.diagnostics["turnover_budget_mode"] == "trade_groups"
    assert out.diagnostics["budget_exits_executed"] == ()
    assert out.diagnostics["turnover_budget_consumed"] == 0.0
    np.testing.assert_array_equal(out.weights, [0.0, 1.0])  # full swap target
    plan = _plan(
        shares=[10_000, 0],
        prices=[10.0, 10.0],
        cash=0.0,
        weights=out.weights,
        ranks=_constructor_ranks(signal, np.ones(2, dtype=bool), keys),
        keys=keys,
        model=BASIS_POINT_MODEL,
        budget=0.10,
        min_trade_amount=0.0,
    )
    assert plan.trade_groups == (
        {"sells": (("A", 500, 5000.0),), "buy": ("B", 400, 4000.0), "budget_consumed": 9000.0},
    )
    assert plan.budget_unspent_reason == "budget"
    outcome = _execute(plan, keys, [10_000, 0], 0.0, BASIS_POINT_MODEL)
    assert outcome.shares.tolist() == [9_500, 400]
    assert outcome.cash == pytest.approx(991.0)
    equity = outcome.cash + 9_500 * 10.0 + 400 * 10.0
    assert equity == pytest.approx(99_991.0)
    # §3.2: never claim the target is reached — B holds 4,000 of 99,991.
    assert outcome.shares[1] * 10.0 / equity == pytest.approx(4_000.0 / 99_991.0)
    assert outcome.diagnostics["turnover_budget_mode"] == "trade_groups"


# --- §6.1/§6.2 (W4): the fixed-target probe ---------------------------------------


def _probe_market():
    """20 names: OLD1..OLD10 held at 0.10 (signals 1.0..1.9) and
    NEW1..NEW10 entrants (signals 2.0..2.9).  Constructor ranks
    (constructor.py:316-322, signal DESC): NEW10 = 0 ... NEW1 = 9,
    OLD10 = 10 ... OLD1 = 19 — so the priority sell order (-rank) exits
    OLD1 first and the buy order (rank) enters NEW10 first."""
    keys = [f"OLD{i}" for i in range(1, 11)] + [f"NEW{i}" for i in range(1, 11)]
    signal = np.concatenate([np.linspace(1.0, 1.9, 10), np.linspace(2.0, 2.9, 10)])
    prev = np.zeros(20)
    prev[:10] = 0.10
    return keys, signal, prev


def _probe_cfg(allocation=None, budget=0.10) -> BacktestConfig:
    kwargs = {}
    if allocation is not None:
        kwargs["turnover_allocation"] = allocation
    return BacktestConfig(
        top_n=10,
        buy_rank=10,
        sell_rank=10,
        single_weight_cap=1.0 / 10,
        turnover_budget=budget,
        min_trade_amount=1000.0,
        **kwargs,
    )


def _construct_probe(cfg, signal, prev, **kwargs):
    keys, _, _ = _probe_market()
    return PortfolioConstructor(cfg).construct(
        signal,
        prev,
        capital=CAPITAL,
        eligible=np.ones(20, dtype=bool),
        buy_blocked=np.zeros(20, dtype=bool),
        sell_blocked=np.zeros(20, dtype=bool),
        stable_keys=np.asarray(keys, dtype=object),
        **kwargs,
    )


def _wind_down_to_empty(cfg, signal):
    """§6.1 probe driver: ten fixed-target sessions under the DEFAULT
    (priority) mode.  Exits-first consumes the whole 0.10 budget on one
    0.10-weight exit per session (each exit = 10,000 yuan >= the 1,000
    minimum), every entry leg is starved to zero, and the book is empty
    after session 10 (frozen §2.4 对照: '10 期清仓、期间零买入、现金累积')."""
    prev = np.zeros(20)
    prev[:10] = 0.10
    outs = []
    for session in range(1, 11):
        out = _construct_probe(cfg, signal, prev)
        assert out.diagnostics["turnover_budget_mode"] == "priority"
        assert out.diagnostics["budget_exits_executed"] == (session - 1,)
        assert out.diagnostics["budget_buys_executed"] == ()
        assert out.diagnostics["turnover_budget_consumed"] == pytest.approx(0.10)
        outs.append(out)
        prev = out.weights
    assert prev.sum() == pytest.approx(0.0)  # book cleared
    return outs, prev


def test_w4_probe_default_mode_re_gains_unbudgeted_after_wind_down():
    """Frozen §6.1 (the defect, pinned as legacy behavior): after the
    default mode's priority allocator sells the book to empty, the NEXT
    session's empty prev makes line-272's rule read initial_funding=True
    and constructor.py:411 grant the unbudgeted full-buy privilege again —
    turnover_budget_mode flips to "unbudgeted" and a FULL 1.0-turnover buy
    executes despite the 0.10 budget.  Red line 2 pins this default-path
    behavior byte-for-byte; the fixes are the §6.2 flag and the
    trade_groups mode, tested below."""
    keys, signal, prev = _probe_market()
    cfg = _probe_cfg()  # default allocation = priority
    _, empty = _wind_down_to_empty(cfg, signal)
    assert empty.sum() <= 1e-12
    out = _construct_probe(cfg, signal, empty)  # NO run-level flag: legacy call
    assert out.diagnostics["initial_funding"] is True  # §6.1: the defect
    assert out.diagnostics["turnover_budget_mode"] == "unbudgeted"
    assert float(np.asarray(out.buy_weights).sum()) == pytest.approx(1.0)
    assert out.diagnostics["budget_buys_executed"] == ()  # own allocator idle
    np.testing.assert_allclose(out.weights[10:], np.full(10, 0.10))
    np.testing.assert_array_equal(out.weights[:10], np.zeros(10))


def test_w4_probe_run_level_flag_keeps_session_eleven_budgeted():
    """Frozen §6.2 (the fix, caller-held state): the run started with a
    non-empty book, so the caller maintains used=True from session 1
    (initialization rule: 'empty since run start -> False, else True').
    Sessions 1-10 are bit-identical to the default probe (a non-empty
    book already reads initial_funding=False either way); session 11 now
    reads initial_funding := (not True) and ... = False -> budgeted, so
    the priority allocator funds exactly ONE 0.10 entry (NEW10, rank 0)
    and the echo key records the flag."""
    keys, signal, _ = _probe_market()
    cfg = _probe_cfg()
    default_outs, _ = _wind_down_to_empty(cfg, signal)  # baseline driver
    prev = np.zeros(20)
    prev[:10] = 0.10
    flagged_outs = []
    for session in range(1, 11):
        out = _construct_probe(cfg, signal, prev, initial_funding_used=True)
        assert out.diagnostics["initial_funding_used"] is True  # §6.2 echo
        assert out.diagnostics["initial_funding"] is False
        flagged_outs.append(out)
        prev = out.weights
    # While the book is non-empty the flag changes NOTHING: all ten
    # sessions' weights are bit-identical to the default probe (red line 2).
    for default, flagged in zip(default_outs, flagged_outs):
        np.testing.assert_array_equal(default.weights, flagged.weights)
    assert prev.sum() <= 1e-12
    out = _construct_probe(cfg, signal, prev, initial_funding_used=True)
    assert out.diagnostics["initial_funding"] is False  # no exemption refund
    assert out.diagnostics["turnover_budget_mode"] == "priority"  # budgeted
    assert out.diagnostics["budget_buys_executed"] == (19,)  # NEW10 only
    assert out.turnover == pytest.approx(0.10)  # one entry within budget
    assert out.weights[19] == pytest.approx(0.10)
    assert float(np.asarray(out.weights).sum()) == pytest.approx(0.10)


def test_w4_probe_trade_groups_mode_buys_within_budget_after_wind_down():
    """Frozen §6.1/§6.2 via the experimental mode: same wind-down, then
    session 11 under turnover_allocation="trade_groups" with the run-level
    flag.  The constructor still walks the unbudgeted path and only
    labels the mode (§1.1.2) — its weights are the FULL 10x0.10 target —
    but the planner (§1.2: exemption requires an empty book AND
    initial_funding_used is not True) binds B_r = 0.10 x 100,000 = 10,000,
    so exactly ONE pure-buy group of 10,000 (NEW10 1,000 sh, the rank-0
    buy, §1.4 step 3 need <= 0 against 100,000 cash) executes; the other
    nine entrants carry leg code "budget" (§1.7.6) and the session reason
    is None (budget exactly exhausted, §1.7).  Cash 90,000 is left — no
    forced full deployment."""
    keys, signal, prev = _probe_market()
    _, empty = _wind_down_to_empty(_probe_cfg(), signal)
    grouped_cfg = _probe_cfg(allocation="trade_groups")
    out = _construct_probe(grouped_cfg, signal, empty, initial_funding_used=True)
    assert out.diagnostics["turnover_budget_mode"] == "trade_groups"
    assert out.diagnostics["initial_funding"] is False
    np.testing.assert_allclose(out.weights[10:], np.full(10, 0.10))
    plan = _plan(
        shares=np.zeros(20, dtype=int),
        prices=[10.0] * 20,
        cash=CAPITAL,
        weights=out.weights,
        ranks=_constructor_ranks(signal, np.ones(20, dtype=bool), keys),
        keys=keys,
        model=ZERO_MODEL,
        budget=0.10,
        min_trade_amount=1000.0,
        initial_funding_used=True,  # §6.2: the burned exemption stays burned
    )
    assert plan.trade_groups == (
        {"sells": (), "buy": ("NEW10", 1000, 10000.0), "budget_consumed": 10000.0},
    )
    assert plan.turnover_budget_consumed == pytest.approx(10_000.0)
    assert plan.budget_unspent_reason is None
    assert plan.leg_skip_reasons == tuple(
        (f"NEW{i}", "buy", "budget") for i in range(1, 10)
    )
    outcome = _execute(plan, keys, np.zeros(20, dtype=int), CAPITAL, ZERO_MODEL)
    assert outcome.failed_buys == ()
    assert outcome.shares[19] == 1000
    assert outcome.cash == pytest.approx(90_000.0)
    assert outcome.diagnostics["budget_unspent_reason"] is None


# --- §5 (W6): the pure feasibility criterion -------------------------------------


def test_w6_pure_criterion_reproduces_the_slow_reference_json():
    """Frozen §5.1/§5.2 (W6): the pure criterion function
    ``ordinary_swap_session_feasibility`` — no IO, arithmetic only — must
    reproduce ``reports/wp15/config_feasibility.json`` field-for-field on
    the slow reference inputs (initial_capital 100,000 and
    min_trade_amount 5,000 from configs/r1_slow_ab.yaml; turnover_budget
    0.05 injected by the slow_fundamental domain): 2 x 5,000 = 10,000 >
    0.05 x 100,000 = 5,000 -> INFEASIBLE for one ordinary pair, while a
    single minimum leg (5,000 <= 5,000) still fits.  A feasible
    counterexample (2 x 2,500 = 5,000 <= 0.10 x 100,000 = 10,000) proves
    the function is a criterion, not a constant."""
    from ashare_domain import RESEARCH_DOMAINS

    from ashare_portfolio.trade_groups import ordinary_swap_session_feasibility

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "r1_slow_ab.yaml").read_text(encoding="utf-8"))
    capital = float(raw["backtest"]["initial_capital"])
    minimum = float(raw["backtest"]["min_trade_amount"])
    budget = float(RESEARCH_DOMAINS["slow_fundamental"].turnover_budget)
    result = ordinary_swap_session_feasibility(
        min_trade_amount=minimum,
        turnover_budget=budget,
        equity=capital,
    )
    recorded = json.loads(
        (root / "reports" / "wp15" / "config_feasibility.json").read_text(encoding="utf-8")
    )
    slow = recorded["slow_reference_config"]
    assert slow["initial_capital"] == result["equity"] == 100_000.0
    assert slow["min_trade_amount"] == result["min_trade_amount"] == 5_000.0
    assert slow["turnover_budget"] == result["turnover_budget"] == 0.05
    assert slow["pair_requirement_yuan"] == result["pair_requirement_yuan"] == 10_000.0
    assert slow["session_budget_yuan"] == result["session_budget_yuan"] == 5_000.0
    assert slow["substitution"] == result["substitution"]
    assert slow["substitution"] == "2 * 5000 = 10000 > 0.05 * 100000 = 5000"
    assert result["pair_feasible"] is False
    assert result["single_leg_feasible"] is True  # §5.2 boundary: 5,000 = 5,000
    assert slow["verdict"].startswith("INFEASIBLE")
    assert recorded["criterion"]["formula"] == result["formula"]
    assert recorded["criterion"]["pair_feasible_iff"] == result["pair_feasible_iff"]
    # Feasible counterexample (§5.1): 2 x 2,500 <= 0.10 x 100,000.
    ok = ordinary_swap_session_feasibility(
        min_trade_amount=2500.0,
        turnover_budget=0.10,
        equity=100_000.0,
    )
    assert ok["pair_feasible"] is True
    assert ok["substitution"] == "2 * 2500 = 5000 <= 0.1 * 100000 = 10000"
    # Unbudgeted sessions cannot bind (§1.2: B_r = inf).
    unbudgeted = ordinary_swap_session_feasibility(
        min_trade_amount=5000.0,
        turnover_budget=None,
        equity=100_000.0,
    )
    assert unbudgeted["pair_feasible"] is True
    assert unbudgeted["session_budget_yuan"] is None


# --- audit fix (2026-09-23, major): unwired weight-space run entries fail closed ----


def _trade_groups_cfg(**kwargs) -> BacktestConfig:
    values = dict(
        top_n=2,
        turnover_budget=0.10,
        turnover_allocation="trade_groups",
    )
    values.update(kwargs)
    return BacktestConfig(**values)


def test_unwired_backtest_engine_fails_closed_on_trade_groups():
    """Audit fix (major, option 2 — engine fail-closed): the engine's
    execution section runs its own weight-space loop over
    ``construct().weights`` (backtest.py:277-363) and never calls
    ``plan_trade_session``/``execute_trade_session``.  Under
    ``turnover_allocation="trade_groups"`` the constructor only LABELS the
    mode and delegates the budget to the planner (frozen §1.1.2), so an
    engine run would execute the FULL unbudgeted swap target — the budget
    silently dropped, the mode a budget bypass instead of an allocator.
    The engine must refuse the mode at initialization until the wiring
    lands (docs/p51_trade_groups_draft.md §3); the default and the two
    wired opt-in modes initialize exactly as before (red line 2)."""
    from ashare_model.backtest import AshareBacktestEngine

    with pytest.raises(ValueError, match="trade_groups"):
        AshareBacktestEngine(_trade_groups_cfg())
    # Red line 2: every wired mode (including the default) is untouched.
    AshareBacktestEngine(BacktestConfig(top_n=2, turnover_budget=0.10))
    AshareBacktestEngine(
        BacktestConfig(top_n=2, turnover_budget=0.10, turnover_allocation="priority")
    )
    AshareBacktestEngine(
        BacktestConfig(
            top_n=2, turnover_budget=0.10, turnover_allocation="proportional"
        )
    )


def test_unwired_paper_sim_fails_closed_on_trade_groups():
    """Audit fix (major, option 2 — paper sim fail-closed): the paper
    simulator consumes ``construct().weights`` through ``_make_orders``
    (run_sim.py, the same weight-space hole as the engine), so a run_sim
    config with ``turnover_allocation="trade_groups"`` would trade the
    unbudgeted target with no warning.  ``SimulationRunner`` must refuse
    the mode at initialization (before any portfolio state is created)
    until its main loop is wired to the planner (docs/
    p51_trade_groups_draft.md §3)."""
    from ashare_data.config import DataConfig, ModelConfig, SimConfig
    from ashare_trading.run_sim import SimulationRunner

    data_config = DataConfig()
    model_config = ModelConfig()
    sim_config = SimConfig(
        initial_capital=100_000.0,
        max_positions=10,
        single_weight_cap=0.1,
    )
    grouped = BacktestConfig(
        initial_capital=100_000.0,
        top_n=10,
        single_weight_cap=0.1,
        turnover_budget=0.05,
        turnover_allocation="trade_groups",
    )
    # validate_execution_config passes (no size drift) — the refusal is
    # the trade_groups wiring guard alone.
    with pytest.raises(ValueError, match="trade_groups"):
        SimulationRunner(
            data_config, model_config, grouped, sim_config, loader=None
        )


def test_unwired_reward_baskets_fail_closed_on_trade_groups():
    """Audit fix (major, option 2 — reward simulation fail-closed): the
    basket reward simulation books costs on the constructor's weight-space
    buy/sell matrices (reward_baskets.py), so a research config with
    ``turnover_allocation="trade_groups"`` would score rewards on
    unbudgeted books — the same silent budget bypass, one layer down.
    Both the scalar and the batched entry must refuse the mode; the
    default mode simulates exactly as before (red line 2)."""
    from ashare_model.reward_baskets import (
        simulate_basket_daily_returns,
        simulate_basket_daily_returns_batch,
    )

    n_stocks, n_dates = 4, 5
    signal = np.tile(np.arange(n_stocks, dtype=float)[:, None], (1, n_dates))
    target = np.zeros_like(signal)
    mask = np.ones_like(signal, dtype=bool)
    with pytest.raises(ValueError, match="trade_groups"):
        simulate_basket_daily_returns(signal, target, _trade_groups_cfg(), universe_mask=mask)
    with pytest.raises(ValueError, match="trade_groups"):
        simulate_basket_daily_returns_batch(
            signal[None, :, :].repeat(2, axis=0),
            target[None, :, :].repeat(2, axis=0),
            _trade_groups_cfg(),
            universe_mask=mask,
        )
    # Red line 2: the default path is unchanged — same unbudgeted fixture
    # as test_execution.py::test_exact_cost_path_covers_capital_and_topn_grid
    # (5 dates -> 3 scored periods; first-period turnover from an empty
    # book with single_weight_cap=1.0 is the full 1.0 deployment).
    wired = simulate_basket_daily_returns(
        signal,
        target,
        BacktestConfig(top_n=2, single_weight_cap=1.0),
        universe_mask=mask,
    )
    assert wired.daily_cost_fractions.shape == (n_dates - 2,)
    # p52 (WP16-A FROZEN_A §1 (C1)/§3.4/§6): with default fees and
    # single_weight_cap=1.0 the full 1.0 deployment must reserve its fee
    # cash, so the booked first-period turnover is 1/(1+c) ≈ 0.99924
    # (c ≈ 7.6e-4); the exact 1.0 required booking the buy unfunded
    # (cash = −c), the §3.4 infeasible state.  The p51 red line itself —
    # the two fail-closed raises above and the trade_groups semantics —
    # is untouched.
    assert wired.turnover[0] == pytest.approx(1.0, abs=5e-3)
