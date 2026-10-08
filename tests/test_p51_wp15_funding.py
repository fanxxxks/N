"""P51 WP15 funding tests: the three frozen scenarios plus the W4 wiring.

Every numeric assertion is a frozen value of
``reports/wp15/FROZEN_EXPECTATIONS.md`` (§ references in each docstring);
assertions are never rewritten from implementation output (W5, §7.1).

* §2 — scenario 1: zero-fee multi-session replacement (20 sessions).
* §3 — scenario 2: financing with fees (two sessions, 991/99,991 then
  983/99,983).
* §4 — scenario 3: forced exit, three branches.
* §6 — W4 first-funding exemption state machine in the constructor.
* §1.1 — the ``trade_groups`` constructor mode wiring (§1.1.1/§1.1.2)
  and the §1.1.5 anti-drift requirement as adjudicated 2026-09-22
  (forward-facing rule in docs/p51_trade_groups_draft.md §2; the
  anchored reference config stays byte-original).
* §5 — W6 feasibility criterion applied to the slow reference config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from ashare_data.config import BacktestConfig
from ashare_execution import ExecutionCostModel
from ashare_portfolio.constructor import PortfolioConstructor
from ashare_portfolio.execution_spec import (
    portfolio_config_provenance,
    validate_portfolio_config_provenance,
)

CAPITAL = 100_000.0
ZERO_MODEL = ExecutionCostModel(
    commission_rate=0.0,
    min_commission=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_rate=0.0,
)


def _plan(**kwargs):
    # Imported lazily so the W4/mode wiring tests below can run (and
    # genuinely fail) before the planner module exists.
    from ashare_portfolio.trade_groups import plan_trade_session

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


def _apply(plan, index, shares, cash, price=10.0):
    """Ledger step: engine semantics — sell proceeds net of fees and buy
    outlays incl. fees move cash the same day (frozen §1.2 basis)."""
    shares = list(shares)
    sell = 0.0
    buy = 0.0
    for key, side, qty, notional in plan.executed_orders:
        if side == "sell":
            shares[index[key]] -= qty
            sell += notional
        else:
            shares[index[key]] += qty
            buy += notional
    return shares, cash + sell - buy - plan.session_fees["total"]


# --- §2: scenario 1 — zero-fee multi-session replacement ------------------------


def test_s1_twenty_session_replacement_trajectory():
    """Frozen §2: 100,000 yuan, ten old names -> ten new names, prices
    frozen at 10, budget 10% (B_r = 10,000 per session, §2.1), minimum
    order 2,500, zero fees.  §2.2: y* = 5,000 with minimal financing
    x = 5,000 (y <= x and x + y <= 10,000), one group per session, both
    legs 500 shares / 5,000 yuan >= m.  §2.2 queue advance: sessions
    2j-1 and 2j work on O(11-j) and N(j).  §2.3/§2.4: cash 0, equity
    100,000, invested 100,000 every session; the two-sided notional gap
    G(k) = 200,000 - 10,000*k reaches 0 at k = 20."""
    old = [f"O{i}" for i in range(1, 11)]
    new = [f"N{i}" for i in range(1, 11)]
    keys = old + new
    index = {k: i for i, k in enumerate(keys)}
    prices = [10.0] * 20
    ranks = list(range(10, 20)) + list(range(0, 10))  # §2.1: N1=0..N10=9, O1=10..O10=19
    weights = [0.0] * 10 + [0.10] * 10
    shares = [1000] * 10 + [0] * 10
    cash = 0.0

    for session in range(1, 21):
        plan = _plan(
            shares=shares,
            prices=prices,
            cash=cash,
            weights=weights,
            ranks=ranks,
            keys=keys,
            model=ZERO_MODEL,
            budget=0.10,
            min_trade_amount=2500.0,
        )
        j = (session + 1) // 2  # §2.2: sessions 2j-1/2j pair on O(11-j)/N(j)
        assert len(plan.trade_groups) == 1
        group = plan.trade_groups[0]
        assert group["sells"] == ((f"O{11 - j}", 500, 5000.0),)
        assert group["buy"] == (f"N{j}", 500, 5000.0)
        assert group["budget_consumed"] == 10000.0
        assert plan.turnover_budget_consumed == 10000.0
        shares, cash = _apply(plan, index, shares, cash)
        invested = sum(shares[i] * 10.0 for i in range(20))
        equity = cash + invested
        # §2.4 invariants: equity and invested never fall; cash cycles 0.
        assert cash == pytest.approx(0.0)
        assert invested == pytest.approx(100_000.0)
        assert equity == pytest.approx(100_000.0)
        # §2.4: G(k) = 200,000 - 10,000*k (sum of both-sided residual demand).
        gap = sum(shares[i] * 10.0 for i in range(10))
        gap += sum(max(10_000.0 - shares[10 + i] * 10.0, 0.0) for i in range(10))
        assert gap == pytest.approx(200_000.0 - 10_000.0 * session)

    # §2.4: completion takes exactly 20 sessions; the book is fully turned.
    assert shares[:10] == [0] * 10
    assert shares[10:] == [1000] * 10


# --- §3: scenario 2 — financing with fees ----------------------------------------


def test_s2_financing_with_fees_two_sessions():
    """Frozen §3: 100,000 all in A @ 10, rotate into B @ 10, lot 100,
    0.1% commission both sides, budget 10%.

    Session 1 (§3.2): y* = 4,000 (400 sh B) with x = 5,000 (500 sh A) —
    y = 5,000 would need x >= 5,010.01 -> x = 6,000 -> sum 11,000 >
    B_r = 10,000.  Fees sell 5.00 / buy 4.00 (§3.2); cash 991.00 and
    equity 99,991.00 (= 100,000 - 9 fees); budget consumed 9,000,
    reason "budget" (second group infeasible: paired needs 2,000 >
    1,000 remaining; pure buy 1,001 > 991 cash — and the same fails in
    the zero-fee relaxation, so not cash/fees).  End-of-session B
    weight = 4,000/99,991 (~4.00%) — the reconciliation never claims
    the target is reached (§3.2 对账口径).

    Session 2 (§3.3): B_r = 9,999.10; y = 5,000 needs x = 5,000 -> sum
    10,000 > 9,999.10; y* = 4,000 with x = 4,000 -> sell 400 sh A, buy
    400 sh B; fees 4.00/4.00; cash 983.00; equity 99,983.00; consumed
    8,000; reason "budget"."""
    keys = ["A", "B"]
    index = {"A": 0, "B": 1}
    model = ExecutionCostModel(
        commission_rate=0.001,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )
    shares = [10_000, 0]
    cash = 0.0

    plan = _plan(
        shares=shares,
        prices=[10.0, 10.0],
        cash=cash,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=keys,
        model=model,
        budget=0.10,
        min_trade_amount=0.0,
    )
    assert plan.trade_groups == (
        {
            "sells": (("A", 500, 5000.0),),
            "buy": ("B", 400, 4000.0),
            "budget_consumed": 9000.0,
        },
    )
    assert plan.session_fees == {
        "sell": pytest.approx(5.0),
        "buy": pytest.approx(4.0),
        "total": pytest.approx(9.0),
    }
    assert plan.turnover_budget_consumed == pytest.approx(9000.0)
    assert plan.budget_unspent_reason == "budget"
    shares, cash = _apply(plan, index, shares, cash)
    assert cash == pytest.approx(991.0)
    equity = cash + shares[0] * 10.0 + shares[1] * 10.0
    assert equity == pytest.approx(99_991.0)
    assert shares[1] * 10.0 / equity == pytest.approx(4_000.0 / 99_991.0)

    plan = _plan(
        shares=shares,
        prices=[10.0, 10.0],
        cash=cash,
        weights=[0.0, 1.0],
        ranks=[1, 0],
        keys=keys,
        model=model,
        budget=0.10,
        min_trade_amount=0.0,
    )
    assert plan.trade_groups == (
        {
            "sells": (("A", 400, 4000.0),),
            "buy": ("B", 400, 4000.0),
            "budget_consumed": 8000.0,
        },
    )
    assert plan.session_fees == {
        "sell": pytest.approx(4.0),
        "buy": pytest.approx(4.0),
        "total": pytest.approx(8.0),
    }
    assert plan.turnover_budget_consumed == pytest.approx(8000.0)
    assert plan.budget_unspent_reason == "budget"
    shares, cash = _apply(plan, index, shares, cash)
    assert cash == pytest.approx(983.0)
    equity = cash + shares[0] * 10.0 + shares[1] * 10.0
    assert equity == pytest.approx(99_983.0)


# --- §4: scenario 3 — forced exit, three branches --------------------------------


def _s3_base(**overrides):
    """§4.1 canonical instance: E = 10,000; X 1,000 sh @ 10, ineligible;
    ordinary budget 2,000/session (turnover_budget = 0.2); zero fees;
    m = 0; L = 100.  Y (price 10, target 10,000 yuan) exists in branches
    1 and 2 only."""
    base = dict(
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
        buy_blocked=[False, False],
        sell_blocked=[False, False],
    )
    base.update(overrides)
    return _plan(**base)


def test_s3_branch1_sellable_forced_exit_ignores_ordinary_budget():
    """Frozen §4.2: X sells its WHOLE 10,000 position on the risk-exit
    channel with budget consumption 0 (10,000 > 2,000 proves the
    ordinary budget cannot squeeze the risk exit); the R channel then
    buys y* = 2,000 (200 sh) of Y as a pure-buy group (budget-capped);
    remaining buy demand 8,000 carries leg code "budget" while the
    session reason is None (budget exactly exhausted, §1.7).  End
    state: cash 8,000, Y 2,000, equity 10,000."""
    plan = _s3_base()
    assert plan.turnover_budget_mode == "trade_groups"
    assert plan.forced_exit_count == 1
    assert plan.forced_exit_executed == ("X",)
    assert plan.forced_exit_blocked == ()
    assert plan.trade_groups == (
        {"sells": (), "buy": ("Y", 200, 2000.0), "budget_consumed": 2000.0},
    )
    assert plan.turnover_budget_consumed == 2000.0
    assert plan.budget_unspent_reason is None
    assert plan.leg_skip_reasons == (("Y", "buy", "budget"),)
    assert plan.session_fees == {"sell": pytest.approx(0.0), "buy": pytest.approx(0.0), "total": pytest.approx(0.0)}
    index = {"X": 0, "Y": 1}
    shares, cash = _apply(plan, index, [1000, 0], 0.0)
    assert shares == [0, 200]
    assert cash == pytest.approx(8_000.0)
    assert cash + shares[1] * 10.0 == pytest.approx(10_000.0)


def test_s3_branch2_sell_blocked_forced_exit_is_recorded():
    """Frozen §4.3: X sell_blocked -> the position is KEPT and the block
    recorded; no trade can happen (Y has no acceptable counterpart and
    zero cash), so the session reason is "no_viable_counterpart" with
    leg code (Y, buy, no_viable_counterpart)."""
    plan = _s3_base(sell_blocked=[True, False])
    assert plan.forced_exit_count == 1
    assert plan.forced_exit_executed == ()
    assert plan.forced_exit_blocked == (("X", "sell_blocked"),)
    assert plan.trade_groups == ()
    assert plan.turnover_budget_consumed == 0.0
    assert plan.budget_unspent_reason == "no_viable_counterpart"
    assert plan.leg_skip_reasons == (("Y", "buy", "no_viable_counterpart"),)


def test_s3_branch2_missing_price_variant_records_missing_price():
    """Frozen §4.3.4 parenthetical: the missing-price variant of the
    blocked forced exit records cause "missing_price" (§1.2: price <= 0
    or non-finite forbids ordering the name)."""
    plan = _s3_base(prices=[float("nan"), 10.0])
    assert plan.forced_exit_blocked == (("X", "missing_price"),)


def test_s3_branch3_sellable_but_no_replacement_keeps_cash():
    """Frozen §4.4: X sells on the risk-exit channel (budget 0); with no
    qualified new target the buy candidate set is empty — cash stays
    (no forced reinvestment / minimum position), the session reason is
    "no_viable_counterpart" and no leg codes are recorded."""
    plan = _plan(
        shares=[1000],
        prices=[10.0],
        cash=0.0,
        weights=[0.0],
        ranks=[0],
        keys=["X"],
        model=ZERO_MODEL,
        budget=0.2,
        min_trade_amount=0.0,
        eligible=[False],
    )
    assert plan.forced_exit_executed == ("X",)
    assert plan.forced_exit_blocked == ()
    assert plan.trade_groups == ()
    assert plan.turnover_budget_consumed == 0.0
    assert plan.budget_unspent_reason == "no_viable_counterpart"
    assert plan.leg_skip_reasons == ()
    index = {"X": 0}
    shares, cash = _apply(plan, index, [1000], 0.0)
    assert shares == [0]
    assert cash == pytest.approx(10_000.0)


# --- §6: W4 first-funding exemption state machine --------------------------------


def _cfg(allocation=None, budget=0.10, min_trade=1000.0) -> BacktestConfig:
    kwargs = {}
    if allocation is not None:
        kwargs["turnover_allocation"] = allocation
    return BacktestConfig(
        top_n=10,
        buy_rank=10,
        sell_rank=10,
        single_weight_cap=1.0 / 10,
        turnover_budget=budget,
        min_trade_amount=min_trade,
        **kwargs,
    )


def _construct(cfg, signal, prev, **kwargs):
    n = len(signal)
    return PortfolioConstructor(cfg).construct(
        signal,
        prev,
        capital=CAPITAL,
        eligible=np.ones(n, dtype=bool),
        buy_blocked=np.zeros(n, dtype=bool),
        sell_blocked=np.zeros(n, dtype=bool),
        stable_keys=np.asarray([f"{i:06d}.SZ" for i in range(n)], dtype=object),
        **kwargs,
    )


def _replacement_market(n: int = 30):
    signal = np.full(n, np.nan)
    signal[10:20] = np.linspace(2.0, 2.9, 10)  # entrants rank 0..9
    signal[0:10] = np.linspace(1.0, 1.9, 10)  # held names rank 10..19
    prev = np.zeros(n)
    prev[0:10] = 0.10
    return signal, prev


def test_w4_default_kwarg_keeps_legacy_semantics_byte_for_byte():
    """Frozen §6.2: ``initial_funding_used=None`` (the default) keeps the
    line-272 semantics verbatim — an empty prev book still counts as
    first funding (unbudgeted), and NO ``initial_funding_used`` key is
    added to diagnostics."""
    signal, _ = _replacement_market()
    prev = np.zeros(30)  # an EMPTY book: the legacy rule reads this as first funding
    out = _construct(_cfg(), signal, prev)
    assert out.diagnostics["initial_funding"] is True
    assert out.diagnostics["turnover_budget_mode"] == "unbudgeted"
    assert "initial_funding_used" not in out.diagnostics


def test_w4_used_flag_blocks_the_exemption_refund_on_empty_book():
    """Frozen §6.1/§6.2: the legacy defect is that a fully cleared book
    RE-GAINS the unbudgeted full-buy privilege.  With the run-level flag
    passed (used=True — the exemption was burned or a position was held
    earlier in this run), an empty book is budgeted:
    ``initial_funding := (not b) and prev.sum() <= eps`` -> False, and
    the default priority allocator executes (mode "priority"), the echo
    key recording True."""
    signal, _ = _replacement_market()
    prev = np.zeros(30)  # the book is empty — only the run-level flag distinguishes
    out = _construct(_cfg(), signal, prev, initial_funding_used=True)
    assert out.diagnostics["initial_funding"] is False
    assert out.diagnostics["turnover_budget_mode"] == "priority"
    assert out.diagnostics["initial_funding_used"] is True
    # The explicit False flag on an empty book keeps the exemption.
    out = _construct(_cfg(), signal, prev, initial_funding_used=False)
    assert out.diagnostics["initial_funding"] is True
    assert out.diagnostics["turnover_budget_mode"] == "unbudgeted"
    assert out.diagnostics["initial_funding_used"] is False


def test_w4_flag_on_nonempty_book_matches_default_weights():
    """Frozen §6.2: for a non-empty prev book the legacy rule already
    yields initial_funding=False, so the explicit flag must not change
    the constructed weights — only the echo key is added."""
    signal, _ = _replacement_market()
    prev = np.zeros(30)
    prev[0:10] = 0.10
    legacy = _construct(_cfg(), signal, prev)
    flagged = _construct(_cfg(), signal, prev, initial_funding_used=True)
    assert legacy.diagnostics["initial_funding"] is False
    assert flagged.diagnostics["initial_funding_used"] is True
    np.testing.assert_array_equal(legacy.weights, flagged.weights)
    assert legacy.diagnostics["turnover_budget_mode"] == "priority"
    assert flagged.diagnostics["turnover_budget_mode"] == "priority"
    assert "initial_funding_used" not in legacy.diagnostics


# --- §1.1: the trade_groups constructor mode --------------------------------------


def test_trade_groups_mode_takes_unbudgeted_path_and_labels_itself():
    """Frozen §1.1.2: under ``turnover_allocation="trade_groups"`` the
    constructor walks EXACTLY the unbudgeted code path (threshold / min
    trade filters and the cash cap unchanged), only labels
    diagnostics["turnover_budget_mode"] = "trade_groups"; its own
    priority/proportional allocation must NOT execute (allocation keys
    untouched, consumed 0) and the weights equal the same config with
    no budget at all."""
    signal, prev = _replacement_market()
    prev = np.zeros(30)
    prev[0:10] = 0.10
    grouped = _construct(_cfg(allocation="trade_groups"), signal, prev)
    unbudgeted = _construct(_cfg(allocation="priority", budget=None), signal, prev)
    assert grouped.diagnostics["turnover_budget_mode"] == "trade_groups"
    np.testing.assert_array_equal(grouped.weights, unbudgeted.weights)
    assert grouped.diagnostics["budget_exits_executed"] == ()
    assert grouped.diagnostics["budget_buys_executed"] == ()
    assert grouped.diagnostics["budget_skipped_no_viable"] == ()
    assert grouped.diagnostics["turnover_budget_consumed"] == 0.0


def test_trade_groups_mode_passes_validation_and_provenance():
    """Frozen §1.1.1: "trade_groups" joins the turnover_allocation enum
    (validate_portfolio_config accepts it) and, because the key is
    reused, provenance carries the mode with no new field."""
    cfg = _cfg(allocation="trade_groups")
    PortfolioConstructor(cfg)  # validate_portfolio_config runs at init
    record = portfolio_config_provenance(cfg)
    assert record["turnover_allocation"] == "trade_groups"
    assert validate_portfolio_config_provenance(record) == record


def test_allocation_mode_mechanism_no_default_drift_and_documented_rule():
    """Frozen §1.1.5 as adjudicated (2026-09-22 ruling, recorded in
    docs/p51_trade_groups_draft.md §4): the anti-drift intent of
    "the mode must be explicit" is carried FORWARD-FACING, not by
    retro-editing the anchored slow reference config — WP14 (p50)
    byte-pins its sha256 against frozen real-run reports, and any edit
    (even a semantic no-op) refuses reconciliation.  This test pins the
    three adjudicated facts:
    (a) the opt-in mechanism exists and the default does NOT drift: the
        dataclass default is "priority", the loader resolves an omitted
        key to "priority", and the anchored r1_slow_ab.yaml — which
        deliberately omits the key — loads to exactly that value;
    (b) the future-facing rule (new configs must write the mode key
        explicitly) is documented in docs/p51_trade_groups_draft.md §2;
    (c) the anchored reference config remains byte-original (its
        backtest section carries no turnover_allocation key, per the
        ruling's "historical immutability first")."""
    from ashare_data.config import BacktestConfig, make_backtest_config

    assert BacktestConfig().turnover_allocation == "priority"
    path = Path(__file__).resolve().parents[1] / "configs" / "r1_slow_ab.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "turnover_allocation" not in raw["backtest"]
    # The loader resolves the omitted key to the same non-drifted default.
    resolved = make_backtest_config(raw)
    assert resolved.turnover_allocation == "priority"
    doc = (
        Path(__file__).resolve().parents[1] / "docs" / "p51_trade_groups_draft.md"
    ).read_text(encoding="utf-8")
    assert "turnover_allocation" in doc
    assert "新配置" in doc or "新的实验计划" in doc


# --- §5: W6 feasibility of the slow reference config ------------------------------


def test_w6_slow_reference_config_cannot_pair_one_ordinary_round_trip():
    """Frozen §5.1/§5.2 (W6): one ordinary replacement pair needs
    2 x min_trade_amount <= turnover_budget x E.  The slow reference
    config has initial_capital = 100,000, min_trade_amount = 5,000
    (configs/r1_slow_ab.yaml backtest section) and the slow_fundamental
    domain injects turnover_budget = 0.05 (ashare_domain.py:271):
    2 x 5,000 = 10,000 > 0.05 x 100,000 = 5,000 -> a single session can
    NEVER complete one ordinary buy+sell pair under this config."""
    from ashare_domain import RESEARCH_DOMAINS

    path = Path(__file__).resolve().parents[1] / "configs" / "r1_slow_ab.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    capital = float(raw["backtest"]["initial_capital"])
    min_trade = float(raw["backtest"]["min_trade_amount"])
    budget = float(RESEARCH_DOMAINS["slow_fundamental"].turnover_budget)
    assert capital == 100_000.0
    assert min_trade == 5_000.0
    assert budget == 0.05
    assert 2 * min_trade > budget * capital
