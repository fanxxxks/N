"""P51 WP15 trade-group planner RED/GREEN tests.

Every numeric assertion below is derived from the frozen contract
``reports/wp15/FROZEN_EXPECTATIONS.md`` (§ references in each docstring);
no assertion is written from implementation output (W5 rule, §7.1).

The planner under test is the pure function
``ashare_portfolio.trade_groups.plan_trade_session`` (frozen §1.1.3/§1.2):
explicit shares/prices/cash/target/budget/fee-model inputs in, trade
groups plus closed reason codes out, no IO and no global state.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_execution import ExecutionCostModel
from ashare_portfolio.trade_groups import plan_trade_session


def _model(
    *,
    commission_rate: float = 0.0,
    min_commission: float = 0.0,
) -> ExecutionCostModel:
    return ExecutionCostModel(
        commission_rate=commission_rate,
        min_commission=min_commission,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )


def _plan(
    *,
    shares,
    prices,
    cash: float,
    weights,
    ranks,
    keys,
    model: ExecutionCostModel,
    budget=None,
    min_trade=None,
    lot_size: int = 100,
    eligible=None,
    buy_blocked=None,
    sell_blocked=None,
    initial_funding_used=None,
):
    n = len(keys)
    return plan_trade_session(
        prev_shares=np.asarray(shares, dtype=np.int64),
        prices=np.asarray(prices, dtype=np.float64),
        cash=float(cash),
        target_weights=np.asarray(weights, dtype=np.float64),
        ranks=np.asarray(ranks, dtype=np.int64),
        stable_keys=np.asarray(keys, dtype=object),
        cost_model=model,
        eligible=np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool),
        buy_blocked=(
            np.zeros(n, dtype=bool) if buy_blocked is None else np.asarray(buy_blocked, dtype=bool)
        ),
        sell_blocked=(
            np.zeros(n, dtype=bool) if sell_blocked is None else np.asarray(sell_blocked, dtype=bool)
        ),
        turnover_budget=budget,
        min_trade_amount=min_trade,
        lot_size=lot_size,
        initial_funding_used=initial_funding_used,
    )


def _ledger_after(plan, shares, cash, index):
    """Apply executed orders, booking fees from the plan's authority totals.

    Engine semantics (frozen §1.2 basis, backtest.py:322): sell proceeds net
    of fees land in cash the same day; buys pay notional + fees.
    """
    shares = list(shares)
    sell_notional = 0.0
    buy_notional = 0.0
    for key, side, qty, notional in plan.executed_orders:
        if side == "sell":
            shares[index[key]] -= qty
            sell_notional += notional
        else:
            shares[index[key]] += qty
            buy_notional += notional
    cash = cash + sell_notional - buy_notional - plan.session_fees["total"]
    return shares, cash


# --- §1.1.3: purity -------------------------------------------------------------


def test_planner_is_pure_and_does_not_mutate_inputs():
    """Frozen §1.1.3: the planner is a pure function — no IO, no internal
    state, no randomness; identical inputs must yield identical plans, and
    the caller's arrays must pass through unmutated."""
    shares = np.asarray([1000, 0], dtype=np.int64)
    prices = np.asarray([10.0, 10.0])
    weights = np.asarray([0.0, 1.0])
    ranks = np.asarray([1, 0], dtype=np.int64)
    keys = np.asarray(["A", "B"], dtype=object)
    snapshot = (shares.copy(), prices.copy(), weights.copy(), ranks.copy(), keys.copy())
    one = _plan(
        shares=shares,
        prices=prices,
        cash=0.0,
        weights=weights,
        ranks=ranks,
        keys=keys,
        model=_model(),
        budget=0.5,
    )
    two = _plan(
        shares=shares,
        prices=prices,
        cash=0.0,
        weights=weights,
        ranks=ranks,
        keys=keys,
        model=_model(),
        budget=0.5,
    )
    assert one == two
    assert np.array_equal(shares, snapshot[0])
    assert np.array_equal(prices, snapshot[1])
    assert np.array_equal(weights, snapshot[2])
    assert np.array_equal(ranks, snapshot[3])
    assert np.array_equal(keys, snapshot[4])


# --- §1.5: same-day same-name same-direction merged billing ---------------------


def test_w3_fees_merged_once_per_name_direction():
    """Frozen §1.5 (W3): the billing unit is (session, name, direction);
    every fill of that unit is merged into ONE notional sent to the fee
    authority exactly once — splitting the calls would re-charge the
    per-order minimum commission (§1.5 rationale, ashare_execution.py:116-120).

    Derivation of the session (all steps follow §1.4 mechanically):
    E = 100,000 all in A (10,000 sh @ 10); targets w = (A: 0, B1: 0.03,
    B2: 0.97); B_r = 0.10 x 100,000 = 10,000; commission 0.1% with
    min_commission 2.0; m = 0; L = 100; C_g = 0.

    * No M (all eligible) and no D: S = 100,000 = B_dem -> R = 0 (§1.3.2).
    * B1 (rank 0) first. n_max = floor(3,000/1,000) = 3 (B1's own demand
      caps n; the scan starts there and takes the FIRST feasible n).
      n = 3: y = 3,000, Fbuy = max(3, 2) = 3 -> need 3,003; sell cap
      10,000 - 3,000 = 7,000; smallest lot-grid f with f - max(0.001f, 2)
      >= 3,003 is f = 4,000 (net 3,996; f = 3,000 nets only 2,997);
      x + y = 7,000 <= 10,000 -> viable at the first probed n.
      Group 1: sell A 400 sh / 4,000, buy B1 300 sh / 3,000,
      consumed 7,000; cash = 4,000 - 4 - 3,000 - 3 = 993.
    * B1 filled; B2 next (demand 97,000). (a)+(b) imply y <= 1,995 -> the
      only grid point is y = 1,000: Fbuy(1,000) = max(1, 2) = 2 -> need
      1,002 - 993 = 9.02; attach the smallest lot to A: f = 1,000 with
      INCREMENTAL fee Fsell(4,000+1,000) - Fsell(4,000) = 5 - 4 = 1
      (merged unit, §1.5) -> net 999 >= 9.02; x + y = 2,000 <= 3,000
      remaining. Group 2: sell A 100 sh / 1,000, buy B2 100 sh / 1,000,
      consumed 2,000; cash = 993 + 999 - 1,002 = 990.
    * Remaining budget 1,000: every further y = 1,000 needs cap 0 for the
      sell leg -> infeasible -> session ends with reason "budget" (§1.7.6).

    Merged-fee expectation (§1.5: exactly one authority call per unit):
    A's sell unit notional = 4,000 + 1,000 = 5,000 -> sell fee =
    sell_cost(5,000).total = max(5, 2) = 5.0.  The per-leg counterfactual
    sell_cost(4,000).total + sell_cost(1,000).total = 4 + 2 = 6.0 differs,
    so this assertion can distinguish merging from double-billing.
    Buy fees: B1 unit 3,000 -> 3.0; B2 unit 1,000 -> max(1, 2) = 2.0.
    """
    index = {"A": 0, "B1": 1, "B2": 2}
    model = _model(commission_rate=0.001, min_commission=2.0)
    plan = _plan(
        shares=[10000, 0, 0],
        prices=[10.0, 10.0, 10.0],
        cash=0.0,
        weights=[0.0, 0.03, 0.97],
        ranks=[2, 0, 1],
        keys=["A", "B1", "B2"],
        model=model,
        budget=0.10,
    )
    assert plan.trade_groups == (
        {
            "sells": (("A", 400, 4000.0),),
            "buy": ("B1", 300, 3000.0),
            "budget_consumed": 7000.0,
        },
        {
            "sells": (("A", 100, 1000.0),),
            "buy": ("B2", 100, 1000.0),
            "budget_consumed": 2000.0,
        },
    )
    merged_sell_fee = model.sell_cost(5000.0).total
    per_leg_sell_fee = model.sell_cost(4000.0).total + model.sell_cost(1000.0).total
    assert merged_sell_fee != per_leg_sell_fee  # 5.0 vs 6.0: the test discriminates
    assert plan.session_fees["sell"] == pytest.approx(merged_sell_fee)
    assert plan.session_fees["buy"] == pytest.approx(
        model.buy_cost(3000.0).total + model.buy_cost(1000.0).total
    )
    assert plan.session_fees["total"] == pytest.approx(10.0)
    assert plan.turnover_budget_consumed == pytest.approx(9000.0)
    assert plan.budget_unspent_reason == "budget"
    shares, cash = _ledger_after(plan, [10000, 0, 0], 0.0, index)
    assert shares == [9500, 300, 100]
    assert cash == pytest.approx(990.0)
    equity = cash + 9500 * 10.0 + 300 * 10.0 + 100 * 10.0
    assert equity == pytest.approx(100000.0 - 10.0)


# --- §1.7: closed skip-code set --------------------------------------------------


def test_skip_code_lot_when_demand_below_one_lot():
    """Frozen §1.7.1: a buy candidate whose remaining demand is below one
    lot notional L*p_j is skipped with code "lot". Demand = 0.05*10,000 =
    500 < 100*10 = 1,000; no budget is set (B_r = inf, §1.2), m = None,
    zero fees, so the ONLY binding constraint is the lot."""
    index = {"P": 0}
    plan = _plan(
        shares=[0],
        prices=[10.0],
        cash=10_000.0,
        weights=[0.05],
        ranks=[0],
        keys=["P"],
        model=_model(),
        budget=None,
        min_trade=None,
    )
    assert plan.trade_groups == ()
    assert plan.budget_unspent_reason == "lot"
    assert plan.leg_skip_reasons == (("P", "buy", "lot"),)
    assert plan.turnover_budget_consumed == 0.0


def test_skip_code_min_amount_when_max_mergeable_below_minimum():
    """Frozen §1.7.2: demand forms whole lots (2,400 >= one lot 1,000) but
    the largest mergeable lot-grid notional floor(2,400/1,000)*1,000 =
    2,000 < m = 2,500 -> "min_amount". Budget unset, zero fees, so the
    minimum-order rule is the only binder."""
    index = {"P": 0}
    plan = _plan(
        shares=[0],
        prices=[10.0],
        cash=10_000.0,
        weights=[0.24],
        ranks=[0],
        keys=["P"],
        model=_model(),
        budget=None,
        min_trade=2500.0,
    )
    assert plan.trade_groups == ()
    assert plan.budget_unspent_reason == "min_amount"
    assert plan.leg_skip_reasons == (("P", "buy", "min_amount"),)


def test_skip_code_cash_when_sell_capacity_insufficient_zero_fee():
    """Frozen §1.7.4: acceptable sell legs EXIST (A, discretionary, 3,000
    yuan of sellable demand) but even the zero-fee relaxation (all fee
    rates 0, budget = inf) cannot fund y_min: cash 0 + capacity 3,000 <
    y_min = 100*100 = 10,000 -> "cash". X is a blocked forced exit
    (held, ~eligible, sell_blocked -> §1.3.1 M-class, cause
    "sell_blocked") and contributes no sellable capacity."""
    index = {"X": 0, "A": 1, "B": 2}
    plan = _plan(
        shares=[1000, 300, 0],
        prices=[10.0, 10.0, 100.0],
        cash=0.0,
        weights=[0.0, 0.0, 1.0],
        ranks=[2, 1, 0],
        keys=["X", "A", "B"],
        model=_model(),
        budget=None,
        min_trade=None,
        eligible=[False, True, True],
        sell_blocked=[True, False, False],
    )
    assert plan.forced_exit_blocked == (("X", "sell_blocked"),)
    assert plan.trade_groups == ()
    assert plan.budget_unspent_reason == "cash"
    assert plan.leg_skip_reasons == (("B", "buy", "cash"),)


def test_skip_code_fees_when_real_fees_break_zero_fee_feasibility():
    """Frozen §1.7.5: the zero-fee relaxation IS feasible (cash 0 + full
    sell capacity 10,000 >= y_min 10,000) but real fees are not: net real
    proceeds 0.999*10,000 = 9,990 < y_min + Fbuy(y_min) = 10,000 + 10 =
    10,010 -> "fees". Commission 0.1%, min 0, budget unset, m = 0."""
    index = {"A": 0, "B": 1}
    model = _model(commission_rate=0.001)
    plan = _plan(
        shares=[1000, 0],
        prices=[10.0, 100.0],
        cash=0.0,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=["A", "B"],
        model=model,
        budget=None,
        min_trade=None,
    )
    assert plan.trade_groups == ()
    assert plan.budget_unspent_reason == "fees"
    assert plan.leg_skip_reasons == (("B", "buy", "fees"),)


# --- §1.4 step 3 / §1.6: liquidation legs and pure-buy groups -------------------


def test_liquidation_sell_leg_includes_odd_lot():
    """Frozen §1.4 step 3 liquidation clause + §1.6: when the attached
    sell candidate has target weight 0 and u_a equals its whole remaining
    holding, the full position (odd lot included) may be attached.

    Derivation: A = 150 sh @ 10 (1,500); B @ 10, w_B = 1.0 -> demand
    1,500 -> n_max = 1 -> y = 1,000; commission 0.1% -> need =
    1,000 + 1 = 1,001. u_A = min(1,500, B_r - y) with B_r = 2.0*1,500 =
    3,000 -> u_A = 1,500 = full holding -> liquidation allowed. Lot grid
    f = 1,000 nets 1,000 - 1 = 999 < 1,001; the full-position f = 1,500
    nets 1,500 - 1.5 = 1,498.5 >= 1,001 -> attach ALL 150 shares (the
    50-share odd lot included, §1.6). Budget 1,500 + 1,000 = 2,500 <=
    3,000. Cash = (1,500 - 1.5) - (1,000 + 1) = 497.5."""
    index = {"A": 0, "B": 1}
    model = _model(commission_rate=0.001)
    plan = _plan(
        shares=[150, 0],
        prices=[10.0, 10.0],
        cash=0.0,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=["A", "B"],
        model=model,
        budget=2.0,
        min_trade=None,
    )
    assert plan.trade_groups == (
        {
            "sells": (("A", 150, 1500.0),),
            "buy": ("B", 100, 1000.0),
            "budget_consumed": 2500.0,
        },
    )
    shares, cash = _ledger_after(plan, [150, 0], 0.0, index)
    assert shares == [0, 100]
    assert cash == pytest.approx(497.5)


def test_pure_buy_group_when_cash_sufficient():
    """Frozen §1.4 step 3: need = y + Fbuy(y) - C_g <= 0 -> a PURE BUY
    group with no sell leg (W1 "existing cash can fund a buy alone");
    §1.4 compliance note: a sell leg is NEVER driven by the budget alone.

    Derivation: C_g = 8,500; A held 1,500 with w_A = 0 (sell demand
    1,500); B @ 10, w_B = 1.0 -> demand 10,000; B_r = 0.5*10,000 = 5,000;
    zero fees. Scanning n from 10 down: every y > 5,000 breaks (a);
    y = 5,000 has need = 5,000 - 8,500 < 0 -> pure buy, 5,000 <= 5,000.
    So y* = 5,000 and the group carries sells = () — A is NOT sold."""
    index = {"A": 0, "B": 1}
    plan = _plan(
        shares=[150, 0],
        prices=[10.0, 10.0],
        cash=8_500.0,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=["A", "B"],
        model=_model(),
        budget=0.5,
        min_trade=None,
    )
    assert plan.trade_groups == (
        {"sells": (), "buy": ("B", 500, 5000.0), "budget_consumed": 5000.0},
    )
    sells = [o for o in plan.executed_orders if o[1] == "sell"]
    assert sells == []


# --- §1.2/§6: the exemption input ------------------------------------------------


def test_exemption_input_controls_budget_relief():
    """Frozen §1.2: B_r = inf when the exemption applies (§6.2 state
    machine: exemption = book empty AND initial_funding_used is not
    True); with an empty book and used=False the whole 100,000 target is
    buyable in one unbudgeted group, while used=True (the exemption was
    burned in an earlier session — §6.1, "clearing the book does not
    restore the exemption") binds B_r = 0.10*100,000 = 10,000 and caps
    the buy at 10,000 yuan, the unfilled remainder carrying code
    "budget" (§1.7.6: zero-fee and real-fee relaxations are feasible,
    the actual remaining budget 0 is not)."""
    index = {"P": 0}
    common = dict(
        shares=[0],
        prices=[10.0],
        cash=100_000.0,
        weights=[1.0],
        ranks=[0],
        keys=["P"],
        model=_model(),
        budget=0.10,
        min_trade=None,
    )
    relieved = _plan(initial_funding_used=False, **common)
    assert relieved.trade_groups == (
        {"sells": (), "buy": ("P", 10_000, 100_000.0), "budget_consumed": 100_000.0},
    )
    burned = _plan(initial_funding_used=True, **common)
    assert burned.trade_groups == (
        {"sells": (), "buy": ("P", 1_000, 10_000.0), "budget_consumed": 10_000.0},
    )
    assert burned.leg_skip_reasons == (("P", "buy", "budget"),)


# --- §1.3.2: D-class single-side deleveraging sells ------------------------------


def test_deleveraging_sell_is_budget_exempt_and_not_a_trade_group():
    """Frozen §1.3.2 (W1: the target itself demanding a lower book allows
    a one-sided sell): A held 10,000 (E = 10,000), w_A = 0.5 -> S = 5,000,
    B_dem = 0 -> R = 5,000. The D-class sell fills min(s_A, R) = 5,000
    (exactly 5 lots), passes m = 2,500, consumes NO budget, and is NOT
    recorded as a trade group (§1.8: trade_groups carries R-class groups
    only; turnover_budget_consumed sums R-class consumption only)."""
    index = {"A": 0}
    plan = _plan(
        shares=[1000],
        prices=[10.0],
        cash=0.0,
        weights=[0.5],
        ranks=[0],
        keys=["A"],
        model=_model(),
        budget=0.10,
        min_trade=2500.0,
    )
    assert plan.deleveraging_sell_notional == pytest.approx(5000.0)
    assert plan.deleveraging_skipped == ()
    assert plan.trade_groups == ()
    assert plan.turnover_budget_consumed == 0.0
    assert plan.executed_orders == (("A", "sell", 500, 5000.0),)


def test_deleveraging_sell_skipped_below_minimum_order():
    """Frozen §1.3.2: a D-class fill that cannot reach the minimum merged
    notional is skipped and recorded in deleveraging_skipped. Sell demand
    = (1.0 - 0.8)*10,000 = 2,000; lot floor = 2,000 < m = 2,500 -> skip;
    nothing trades."""
    index = {"A": 0}
    plan = _plan(
        shares=[1000],
        prices=[10.0],
        cash=0.0,
        weights=[0.8],
        ranks=[0],
        keys=["A"],
        model=_model(),
        budget=0.10,
        min_trade=2500.0,
    )
    assert plan.deleveraging_sell_notional == 0.0
    assert plan.deleveraging_skipped == (("A", "min_amount"),)
    assert plan.executed_orders == ()
