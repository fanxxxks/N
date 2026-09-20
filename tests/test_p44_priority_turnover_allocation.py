"""P44 priority turnover-allocation RED/GREEN tests (WP 8).

Each test asserts an independent invariant of the p44 preregistration
contract (``docs/p44_priority_turnover_allocation_draft.md`` §2/§5):

* the reproduced counterexample — twenty 500-yuan legs all dying at every
  min-trade level while a full exit leg fits the budget — is closed under
  the new default ``turnover_allocation="priority"`` and stays
  reproducible under the versioned ``"proportional"`` comparison (P1);
* the allocation is concentrated in the pre-fixed exit/entry priority —
  worst-ranked exits first, then best-ranked entries, budget one leg at
  a time, skipped legs never dilute executed ones (P2);
* the "no feasible trade under constraints" state is recorded with an
  explicit reason instead of silently trading nothing (P3);
* budget, self-financing and post-fee-cash invariants hold (P4);
* forced exits are exempt from the budget, sell-blocked names pin at
  prev (P5/P6);
* the allocation is permutation-invariant and tie-stable by
  ``stable_keys`` (P7);
* the rank buffer survives the budget path unchanged (P8);
* a book with no legs trades nothing — fills are never fabricated (P9);
* config legality and provenance record the new mode (P10) and the
  constructor version is 2 (P11).
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_portfolio.constructor import (
    PORTFOLIO_CONSTRUCTOR_VERSION,
    PortfolioConstructor,
)
from ashare_portfolio.execution_spec import (
    portfolio_config_provenance,
    validate_portfolio_config_provenance,
)

CAPITAL = 100_000.0


def _keys(n: int) -> np.ndarray:
    return np.asarray([f"{i:06d}.SZ" for i in range(n)])


def _construct(
    cfg: BacktestConfig,
    signal: np.ndarray,
    prev: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
    buy_blocked: np.ndarray | None = None,
    sell_blocked: np.ndarray | None = None,
    capital: float = CAPITAL,
):
    n = len(signal)
    return PortfolioConstructor(cfg).construct(
        signal,
        prev,
        capital=capital,
        eligible=np.ones(n, dtype=bool) if eligible is None else eligible,
        buy_blocked=(
            np.zeros(n, dtype=bool) if buy_blocked is None else buy_blocked
        ),
        sell_blocked=(
            np.zeros(n, dtype=bool) if sell_blocked is None else sell_blocked
        ),
        stable_keys=_keys(n),
    )


def _replacement_market(n: int = 30):
    """Ten fully-held names to be replaced by ten better-ranked entrants.

    Entrants (indices 10..19) rank 0..9; the held names (0..9) rank
    10..19 — below ``sell_rank=10`` they are all discretionary exits.
    """
    signal = np.full(n, np.nan)
    for r, i in enumerate(range(10, 20)):
        signal[i] = 100.0 - r  # ranks 0..9 -> the new book
    for r, i in enumerate(range(0, 10)):
        signal[i] = 85.0 - r  # ranks 10..19 -> below the buffer
    prev = np.zeros(n)
    prev[:10] = 0.10
    return signal, prev


def _cfg(
    *,
    budget=0.10,
    min_trade=1000.0,
    allocation=None,
    buy_rank=10,
    sell_rank=10,
) -> BacktestConfig:
    kwargs = {}
    if allocation is not None:
        kwargs["turnover_allocation"] = allocation
    return BacktestConfig(
        top_n=buy_rank,
        buy_rank=buy_rank,
        sell_rank=sell_rank,
        single_weight_cap=1.0 / buy_rank,
        turnover_budget=budget,
        min_trade_amount=min_trade,
        **kwargs,
    )


# --- P1: the counterexample is closed; the comparison mode reproduces it ------


@pytest.mark.parametrize("min_trade", [1000.0, 2500.0, 5000.0])
def test_p1_counterexample_produces_executable_orders_under_priority(min_trade):
    signal, prev = _replacement_market()
    out = _construct(_cfg(min_trade=min_trade), signal, prev)
    # A2: the budget concentrates on the pre-fixed priority — the
    # worst-ranked exit (index 9, rank 19) takes the whole budget as one
    # full 0.10-weight (10k yuan) leg, well above every min-trade level.
    assert out.diagnostics["turnover_budget_mode"] == "priority"
    assert out.diagnostics["budget_exits_executed"] == (9,)
    assert out.order_count >= 1
    assert out.turnover == pytest.approx(0.10)
    assert out.weights[9] == pytest.approx(0.0)
    # The nine remaining exits and ten entries cannot form a minimum
    # order out of a zero remaining budget: skipped with the reason.
    skipped = set(out.diagnostics["budget_skipped_no_viable"])
    assert len(skipped) == 19
    assert 9 not in skipped
    assert out.diagnostics["suppressed_trade_count"] == 19


def test_p1_proportional_comparison_mode_reproduces_the_elimination():
    signal, prev = _replacement_market()
    out = _construct(
        _cfg(allocation="proportional"), signal, prev
    )
    # v1 behaviour, kept verbatim as the versioned comparison: the budget
    # is spread into twenty 500-yuan legs and every one dies.
    assert out.order_count == 0
    assert out.turnover == pytest.approx(0.0)
    assert out.diagnostics["turnover_budget_mode"] == "proportional"
    assert out.diagnostics["turnover_budget_scale"] == pytest.approx(0.05)
    assert len(out.diagnostics["min_trade_dropped"]) == 20


# --- P2: concentration in the pre-fixed priority -------------------------------


def test_p2_budget_covers_k_legs_in_priority_order():
    signal, prev = _replacement_market()
    out = _construct(_cfg(budget=0.30), signal, prev)
    # Exits by rank DESC: 9, 8, 7 each take a full 0.10 leg; the rest is
    # below one minimum order of remaining budget.
    assert out.diagnostics["budget_exits_executed"] == (9, 8, 7)
    assert out.turnover == pytest.approx(0.30)
    for i in (7, 8, 9):
        assert out.weights[i] == pytest.approx(0.0)
    for i in range(0, 7):
        assert out.weights[i] == pytest.approx(0.10)  # skipped, untraded


def test_p2_exits_release_cash_before_entries_rank_order_within_class():
    # Five held survivors top up to the unit and five entrants want in —
    # all buys; the rank-ASC order funds the best-ranked legs first.
    n = 20
    signal = np.full(n, np.nan)
    for r, i in enumerate(range(0, 5)):
        signal[i] = 100.0 - r  # held survivors, ranks 0..4
    for r, i in enumerate(range(5, 20)):
        signal[i] = 90.0 - r  # candidates, ranks 5..19
    prev = np.zeros(n)
    prev[:5] = 0.05  # half-unit survivors -> +0.05 top-up legs
    cfg = _cfg(budget=0.15, min_trade=1000.0, buy_rank=10, sell_rank=15)
    out = _construct(cfg, signal, prev)
    assert out.diagnostics["budget_buys_executed"] == (0, 1, 2)
    assert out.weights[0] == pytest.approx(0.10)
    assert out.weights[1] == pytest.approx(0.10)
    assert out.weights[2] == pytest.approx(0.10)
    assert out.weights[3] == pytest.approx(0.05)  # skipped, untraded
    assert out.weights[4] == pytest.approx(0.05)
    assert out.weights[5] == pytest.approx(0.0)  # entrants unfunded
    assert out.turnover == pytest.approx(0.15)


# --- P3: the no-feasible-trade state records its reason ------------------------


def test_p3_no_feasible_trade_records_the_reason():
    signal, prev = _replacement_market()
    # 0.005 weight = 500 yuan < 1000 minimum on every leg: nothing is
    # feasible, and the diagnostics say so instead of silently idling.
    out = _construct(_cfg(budget=0.005), signal, prev)
    assert out.order_count == 0
    assert out.turnover == pytest.approx(0.0)
    assert out.diagnostics["turnover_budget_mode"] == "priority"
    assert set(out.diagnostics["budget_skipped_no_viable"]) == set(range(20))
    assert out.diagnostics["turnover_budget_consumed"] == pytest.approx(0.0)
    assert out.weights == pytest.approx(prev)  # nobody traded


# --- P4: budget, self-financing and post-fee cash invariants --------------------


def test_p4_random_scenario_respects_budget_and_self_financing():
    rng = np.random.default_rng(44)
    n = 25
    signal = rng.normal(0.0, 1.0, size=n)
    prev = np.zeros(n)
    prev[rng.choice(n, size=8, replace=False)] = rng.dirichlet(
        np.ones(8)
    )
    sell_blocked = np.zeros(n, dtype=bool)
    sell_blocked[3] = True
    out = _construct(
        _cfg(budget=0.07, min_trade=700.0), signal, prev,
        sell_blocked=sell_blocked,
    )
    executed = (
        out.diagnostics["budget_exits_executed"]
        + out.diagnostics["budget_buys_executed"]
    )
    consumed = sum(abs(out.weights[i] - prev[i]) for i in executed)
    assert consumed <= 0.07 + 1e-9
    assert out.diagnostics["turnover_budget_consumed"] == pytest.approx(
        consumed
    )
    assert float(out.weights.sum()) <= 1.0 + 1e-9
    assert (out.weights >= 0.0).all()


def test_p4_engine_end_to_end_keeps_post_fee_cash_non_negative():
    from ashare_model.backtest import AshareBacktestEngine
    from ashare_portfolio.rebalance import RebalancePolicy
    from tests.conftest import with_price_semantics

    n_stocks, n_dates = 35, 18
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = [f"202401{i + 2:02d}" for i in range(n_dates)]
    base = 10.0 + np.arange(n_stocks, dtype=np.float64)[:, None] * 0.01
    drift = 1.0 + np.arange(n_dates, dtype=np.float64)[None, :] * (
        0.001 + np.arange(n_stocks, dtype=np.float64)[:, None] * 0.00001
    )
    open_ = base * drift
    raw = with_price_semantics({
        "open": open_,
        "high": open_ * 1.01,
        "low": open_ * 0.99,
        "pre_close": np.column_stack([open_[:, 0], open_[:, :-1]]),
        "volume": np.full_like(open_, 1_000_000.0),
    })
    signal = np.empty((n_stocks, n_dates), dtype=np.float64)
    for t in range(n_dates):
        order = np.roll(np.arange(n_stocks), t % 8)
        signal[order, t] = np.arange(n_stocks, 0, -1, dtype=np.float64)
    mask = np.ones(signal.shape, dtype=bool)

    cfg = BacktestConfig(
        rebalance_frequency="every_5_days",
        target_horizon=5,
        initial_capital=CAPITAL,
        top_n=10,
        buy_rank=10,
        sell_rank=15,
        single_weight_cap=0.10,
        turnover_budget=0.10,
        min_trade_amount=1000.0,
    )
    rebalance_mask = RebalancePolicy.from_config(cfg).rebalance_mask(dates)
    result = AshareBacktestEngine(cfg).run(
        signal, raw, codes, dates, universe_mask=mask,
        rebalance_mask=rebalance_mask,
    )
    # Post-fee cash never goes negative (engine fail-closed + the
    # constructor's self-financing targets).
    assert result.final_cash >= 0.0
    # Non-due sessions carry mode None (no allocation happens); due ones
    # are unbudgeted (initial funding) or priority.
    modes = {
        d["turnover_budget_mode"]
        for d in result.construction_diagnostics
    }
    assert modes <= {None, "unbudgeted", "priority"}
    assert "priority" in modes  # the allocator actually ran
    # Every budgeted session's constructor turnover (its own recorded
    # field, mandatory legs exempt by design) stays within the budget.
    for diag in result.construction_diagnostics:
        if diag["turnover_budget_mode"] != "priority":
            continue
        assert diag["turnover_budget_consumed"] <= 0.10 + 1e-9


# --- P5/P6: forced exits are exempt; sell-blocked names pin at prev ------------


def test_p5_forced_exit_bypasses_the_budget():
    n = 30
    signal, prev = _replacement_market(n)
    eligible = np.isfinite(signal)
    eligible[0] = False  # held name 0 leaves the universe
    out = _construct(_cfg(budget=0.001), signal, prev, eligible=eligible)
    # The mandatory exit executes in full no matter how small the budget.
    assert out.weights[0] == pytest.approx(0.0)
    assert out.sell_weights[0] == pytest.approx(0.10)
    assert 0 not in set(out.diagnostics["budget_skipped_no_viable"])
    # Discretionary legs cannot form minimum orders from 100 yuan.
    assert out.order_count >= 1


def test_p6_sell_blocked_name_pins_at_prev_with_no_order():
    signal, prev = _replacement_market()
    sell_blocked = np.zeros(len(signal), dtype=bool)
    sell_blocked[9] = True  # the worst-ranked exit is blocked
    out = _construct(_cfg(budget=0.10), signal, prev, sell_blocked=sell_blocked)
    assert out.weights[9] == pytest.approx(0.10)  # pinned
    assert out.sell_weights[9] == pytest.approx(0.0)
    assert 9 not in set(out.diagnostics["budget_skipped_no_viable"])
    # The budget moves to the next-worst exit instead.
    assert out.diagnostics["budget_exits_executed"] == (8,)


# --- P7: permutation invariance and stable tie-break ---------------------------


def test_p7_allocation_is_permutation_invariant():
    rng = np.random.default_rng(7)
    n = 25
    signal = np.where(rng.random(n) < 0.8, rng.normal(0, 1, n), np.nan)
    prev = np.zeros(n)
    prev[:8] = rng.dirichlet(np.ones(8)) * 0.8
    eligible = np.isfinite(signal)
    buy_blocked = rng.random(n) < 0.1
    sell_blocked = rng.random(n) < 0.1
    cfg = _cfg(budget=0.09, min_trade=900.0, buy_rank=8, sell_rank=12)

    out = _construct(
        cfg, signal, prev, eligible=eligible,
        buy_blocked=buy_blocked, sell_blocked=sell_blocked,
    )
    perm = rng.permutation(n)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(n)
    # stable_keys travel with their rows: per-name results must be
    # identical after mapping through the permutation.
    out_perm = PortfolioConstructor(cfg).construct(
        signal[perm],
        prev[perm],
        capital=CAPITAL,
        eligible=eligible[perm],
        buy_blocked=buy_blocked[perm],
        sell_blocked=sell_blocked[perm],
        stable_keys=_keys(n)[perm],
    )
    np.testing.assert_allclose(
        np.asarray(out_perm.weights)[inverse],
        np.asarray(out.weights),
        rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(out_perm.sell_weights)[inverse],
        np.asarray(out.sell_weights),
        rtol=0.0, atol=1e-12,
    )


def test_p7_ties_break_by_stable_keys():
    n = 16
    signal = np.full(n, np.nan)
    # Four candidates tied at the top signal; keys decide the entrants.
    for i in (10, 11, 12, 13):
        signal[i] = 5.0
    signal[0] = 1.0  # one held name to exit (rank 4)
    prev = np.zeros(n)
    prev[0] = 1.0
    cfg = _cfg(budget=0.10, min_trade=1000.0, buy_rank=2, sell_rank=2)
    out = _construct(cfg, signal, prev)
    # The two smallest keys among the tied candidates enter the book.
    assert out.selected == (10, 11)
    # The exit's 1.0 want is partially funded to the budget (10k yuan).
    assert out.diagnostics["budget_exits_executed"] == (0,)
    assert out.weights[0] == pytest.approx(0.90)


# --- P8: the rank buffer survives the budget path ------------------------------


def test_p8_rank_buffer_survivors_keep_their_weights():
    n = 30
    signal = np.full(n, np.nan)
    for r, i in enumerate(range(10, 20)):
        signal[i] = 100.0 - r  # entrants ranks 0..9
    for r, i in enumerate(range(0, 10)):
        signal[i] = 85.0 - r  # held ranks 10..19
    prev = np.zeros(n)
    prev[:10] = 0.10
    cfg = _cfg(budget=0.10, min_trade=1000.0, buy_rank=10, sell_rank=15)
    out = _construct(cfg, signal, prev)
    # Held ranks 10..14 sit inside the sell buffer: untouched.
    assert out.diagnostics["buffer_survivors"] == (0, 1, 2, 3, 4)
    for i in range(0, 5):
        assert out.weights[i] == pytest.approx(0.10)
    # The budget funds the worst exit (rank 19 = index 9), not a survivor.
    assert out.diagnostics["budget_exits_executed"] == (9,)
    assert out.weights[9] == pytest.approx(0.0)


# --- P9: no legs -> zero action (no fabricated fills) --------------------------


def test_p9_stable_book_trades_nothing_under_the_budget():
    signal, prev = _replacement_market()
    # Reach the target book with an UNBUDGETED config (no min either) so
    # the achieved book equals the signal's target.
    settle_cfg = BacktestConfig(
        top_n=10, buy_rank=10, sell_rank=10, single_weight_cap=0.10,
    )
    settled = _construct(settle_cfg, signal, prev)
    # Re-run against the achieved book with the same signal and a binding
    # budget: the target equals the book, there are no legs, and the
    # allocator must not fabricate any activity.
    cfg = _cfg()
    again = _construct(cfg, signal, np.asarray(settled.weights))
    assert again.order_count == 0
    assert again.turnover == pytest.approx(0.0)
    assert again.diagnostics["budget_exits_executed"] == ()
    assert again.diagnostics["budget_buys_executed"] == ()
    assert again.diagnostics["budget_skipped_no_viable"] == ()


# --- P10/P11: config legality, provenance and version --------------------------


def test_p10_turnover_allocation_is_validated():
    # The legality check lives in validate_portfolio_config, which the
    # constructor runs at init (fixture correction: the RED version
    # asserted at the dataclass layer, below the validation seam).
    signal, prev = _replacement_market()
    with pytest.raises(ValueError, match="turnover_allocation"):
        _construct(_cfg(allocation="sometimes"), signal, prev)


def test_p10_provenance_records_the_mode():
    cfg = _cfg()
    record = portfolio_config_provenance(cfg)
    assert record["turnover_allocation"] == "priority"
    assert validate_portfolio_config_provenance(record) == record


def test_p11_constructor_version_is_2():
    assert PORTFOLIO_CONSTRUCTOR_VERSION == 2
