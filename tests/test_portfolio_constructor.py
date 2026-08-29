"""P3-03/P3-04/P3-05 contract tests.

Assertion source: ``docs/p3_portfolio_contract.md`` sections 2, 3 and 5.
The expected ranks, weights and constraint effects below are calculated from
that contract, not copied from an implementation run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, load_config, make_backtest_config
from ashare_portfolio.constructor import (
    PORTFOLIO_CONSTRUCTOR_VERSION,
    PortfolioConstructor,
)
from ashare_portfolio.optimizer import (
    PortfolioConstraints,
    PortfolioObjective,
    PortfolioOptimizer,
)


def _signal_from_order(order: list[int]) -> np.ndarray:
    signal = np.empty(len(order), dtype=np.float64)
    for rank, index in enumerate(order):
        signal[index] = float(len(order) - rank)
    return signal


def _construct(
    cfg: BacktestConfig,
    signal: np.ndarray,
    prev: np.ndarray | None = None,
    *,
    eligible: np.ndarray | None = None,
    buy_blocked: np.ndarray | None = None,
    sell_blocked: np.ndarray | None = None,
    capital: float = 100_000.0,
    optimizer: PortfolioOptimizer | None = None,
):
    n = len(signal)
    return PortfolioConstructor(cfg, optimizer=optimizer).construct(
        signal,
        np.zeros(n, dtype=np.float64) if prev is None else prev,
        capital=capital,
        eligible=np.ones(n, dtype=bool) if eligible is None else eligible,
        buy_blocked=(
            np.zeros(n, dtype=bool) if buy_blocked is None else buy_blocked
        ),
        sell_blocked=(
            np.zeros(n, dtype=bool) if sell_blocked is None else sell_blocked
        ),
        stable_keys=np.asarray([f"{i:06d}.SZ" for i in range(n)]),
    )


def test_constructor_version_and_equal_weight_top20_from_cash():
    assert PORTFOLIO_CONSTRUCTOR_VERSION == 1
    cfg = BacktestConfig(
        top_n=20,
        buy_rank=20,
        sell_rank=30,
        single_weight_cap=0.05,
    )
    out = _construct(cfg, _signal_from_order(list(range(35))))
    assert out.rebalanced is True
    assert out.reason == "signal"
    assert out.selected == tuple(range(20))
    assert out.order_count == 20
    assert out.turnover == pytest.approx(1.0)
    assert out.weights[:20] == pytest.approx(np.full(20, 0.05))
    assert out.weights[20:].sum() == 0.0
    assert out.diagnostics["initial_funding"] is True


def test_rank_buffer_keeps_top30_incumbent_until_it_falls_out():
    cfg = BacktestConfig(
        top_n=20,
        buy_rank=20,
        sell_rank=30,
        single_weight_cap=0.05,
    )
    first = _construct(cfg, _signal_from_order(list(range(35))))

    # Incumbent 19 falls to rank 25. Newcomer 20 reaches rank 0, but there
    # is no vacancy: the incumbent remains and no trade is generated.
    within_buffer = [20, *range(0, 19), *range(21, 26), 19, *range(26, 35)]
    held = _construct(cfg, _signal_from_order(within_buffer), first.weights)
    assert held.weights == pytest.approx(first.weights)
    assert held.order_count == 0
    assert 19 in held.diagnostics["buffer_survivors"]

    # Rank 30 is outside Top-30 (zero-based rank >= 30): now 19 is sold and
    # the best Top-20 newcomer fills the one vacant slot.
    outside_buffer = [20, *range(0, 19), *range(21, 31), 19, *range(31, 35)]
    rotated = _construct(cfg, _signal_from_order(outside_buffer), first.weights)
    assert rotated.weights[19] == 0.0
    assert rotated.weights[20] == pytest.approx(0.05)
    assert rotated.order_count == 2
    assert rotated.turnover == pytest.approx(0.10)


def test_legacy_top_n_only_keeps_unbuffered_rotation_semantics():
    cfg = BacktestConfig(top_n=1, single_weight_cap=1.0)
    first = _construct(cfg, np.asarray([3.0, 2.0, 1.0]))
    rotated = _construct(cfg, np.asarray([2.0, 3.0, 1.0]), first.weights)
    assert first.selected == (0,)
    assert rotated.selected == (1,)
    assert rotated.weights.tolist() == [0.0, 1.0, 0.0]
    assert rotated.order_count == 2


def test_legacy_overcapacity_book_winds_down_deterministically():
    cfg = BacktestConfig(
        top_n=2, buy_rank=2, sell_rank=3, single_weight_cap=0.5
    )
    prev = np.asarray([0.3, 0.3, 0.3, 0.0])
    out = _construct(cfg, np.asarray([4.0, 3.0, 2.0, 1.0]), prev)
    assert out.weights.tolist() == [0.5, 0.5, 0.0, 0.0]
    assert out.diagnostics["legacy_position_wind_down"] == (2,)


def test_portfolio_output_weight_arrays_are_read_only():
    cfg = BacktestConfig(top_n=1, single_weight_cap=1.0)
    out = _construct(cfg, np.asarray([2.0, 1.0]))
    with pytest.raises(ValueError):
        out.weights[0] = 0.0


def test_exact_ties_use_stable_keys_not_row_order():
    cfg = BacktestConfig(top_n=2, buy_rank=2, sell_rank=2, single_weight_cap=0.5)
    signal = np.asarray([1.0, 1.0, 0.0])
    constructor = PortfolioConstructor(cfg)
    base = constructor.construct(
        signal,
        np.zeros(3),
        capital=100_000.0,
        eligible=np.ones(3, dtype=bool),
        buy_blocked=np.zeros(3, dtype=bool),
        sell_blocked=np.zeros(3, dtype=bool),
        stable_keys=np.asarray(["600000.SH", "000001.SZ", "300001.SZ"]),
    )
    assert base.selected == (1, 0)


def test_no_dispersion_holds_book_but_forced_universe_exit_still_sells():
    cfg = BacktestConfig(
        top_n=2,
        buy_rank=2,
        sell_rank=3,
        single_weight_cap=0.5,
        min_trade_amount=100_000.0,
        target_weight_change_threshold=0.9,
    )
    prev = np.asarray([0.5, 0.5, 0.0])
    eligible = np.asarray([False, True, True])
    out = _construct(cfg, np.ones(3), prev, eligible=eligible)
    assert out.weights.tolist() == [0.0, 0.5, 0.0]
    assert out.sell_weights.tolist() == [0.5, 0.0, 0.0]
    assert out.diagnostics["forced_exit_count"] == 1
    assert out.diagnostics["threshold_dropped"] == ()
    assert out.diagnostics["min_trade_dropped"] == ()


def test_sell_blocked_reduction_is_force_held_and_buys_only_scale_down():
    cfg = BacktestConfig(top_n=2, buy_rank=2, sell_rank=2, single_weight_cap=0.5)
    prev = np.asarray([0.5, 0.5, 0.0])
    signal = _signal_from_order([1, 2, 0])
    out = _construct(
        cfg,
        signal,
        prev,
        sell_blocked=np.asarray([True, False, False]),
    )
    assert out.weights[0] == pytest.approx(0.5)
    assert out.weights[1] == pytest.approx(0.5)
    assert out.weights[2] == 0.0
    assert out.weights.sum() <= 1.0
    assert out.order_count == 0


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("target_weight_change_threshold", 0.06, "threshold_dropped"),
        ("min_trade_amount", 6_000.0, "min_trade_dropped"),
    ],
)
def test_small_rotation_is_suppressed_by_shared_trade_filters(
    field: str, value: float, diagnostic: str
):
    kwargs = {
        "top_n": 20,
        "buy_rank": 20,
        "sell_rank": 20,
        "single_weight_cap": 0.05,
        field: value,
    }
    cfg = BacktestConfig(**kwargs)
    prior_cfg = BacktestConfig(
        top_n=20, buy_rank=20, sell_rank=20, single_weight_cap=0.05
    )
    first = _construct(prior_cfg, _signal_from_order(list(range(21))))
    rotated_order = [20, *range(19), 19]
    out = _construct(cfg, _signal_from_order(rotated_order), first.weights)
    assert out.weights == pytest.approx(first.weights)
    assert out.order_count == 0
    assert set(out.diagnostics[diagnostic]) == {19, 20}


def test_turnover_budget_scales_existing_book_but_not_initial_funding():
    cfg = BacktestConfig(
        top_n=20,
        buy_rank=20,
        sell_rank=20,
        single_weight_cap=0.05,
        turnover_budget=0.20,
    )
    first = _construct(cfg, _signal_from_order(list(range(25))))
    assert first.turnover == pytest.approx(1.0)  # initial funding is exempt

    rotated_order = [20, 21, 22, 23, 24, *range(15), *range(15, 20)]
    out = _construct(cfg, _signal_from_order(rotated_order), first.weights)
    # Raw rotation is 5 sells + 5 buys at 5% = 0.50 L1; scale is 0.20/0.50.
    assert out.diagnostics["turnover_budget_scale"] == pytest.approx(0.4)
    assert out.turnover == pytest.approx(0.20)
    assert out.weights[15:20] == pytest.approx(np.full(5, 0.03))
    assert out.weights[20:25] == pytest.approx(np.full(5, 0.02))


def test_minimum_amount_is_reapplied_after_turnover_scaling():
    cfg = BacktestConfig(
        top_n=20,
        buy_rank=20,
        sell_rank=20,
        single_weight_cap=0.05,
        turnover_budget=0.20,
        min_trade_amount=3_000.0,
    )
    # Build the prior book without the minimum; otherwise initial 5k orders
    # still pass, but keeping setup explicit makes the tested stage unambiguous.
    prior_cfg = BacktestConfig(
        top_n=20, buy_rank=20, sell_rank=20, single_weight_cap=0.05
    )
    prior = _construct(prior_cfg, _signal_from_order(list(range(25))))
    rotated_order = [20, 21, 22, 23, 24, *range(15), *range(15, 20)]
    out = _construct(cfg, _signal_from_order(rotated_order), prior.weights)
    # Budget scaling makes every 5k raw order a 2k order; the second minimum
    # amount pass removes all ten without weakening the 3k contract.
    assert out.weights == pytest.approx(prior.weights)
    assert out.order_count == 0
    assert set(out.diagnostics["min_trade_dropped"]) == set(range(15, 25))


def test_non_rebalance_day_never_generates_drift_alignment_orders():
    cfg = BacktestConfig(top_n=2, buy_rank=2, sell_rank=2, single_weight_cap=0.5)
    prev = np.asarray([0.6, 0.4, 0.0])
    out = PortfolioConstructor(cfg).construct(
        _signal_from_order([2, 1, 0]),
        prev,
        capital=100_000.0,
        eligible=np.ones(3, dtype=bool),
        buy_blocked=np.zeros(3, dtype=bool),
        sell_blocked=np.zeros(3, dtype=bool),
        stable_keys=np.asarray(["0", "1", "2"]),
        rebalance_due=False,
    )
    assert out.weights.tolist() == prev.tolist()
    assert out.rebalanced is False
    assert out.reason == "not_due"
    assert out.order_count == 0
    assert out.turnover == 0.0


def test_optimizer_uses_same_membership_and_output_contract():
    cfg = BacktestConfig(
        portfolio_method="optimizer",
        top_n=2,
        buy_rank=2,
        sell_rank=2,
        single_weight_cap=0.6,
    )
    optimizer = PortfolioOptimizer(
        PortfolioConstraints(single_weight_cap=0.6),
        PortfolioObjective(risk_aversion=0.0),
    )
    out = _construct(
        cfg,
        np.asarray([3.0, 2.0, 1.0]),
        optimizer=optimizer,
        capital=1_000_000.0,
    )
    assert out.diagnostics["method"] == "optimizer"
    assert out.weights[2] == 0.0
    assert out.weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert out.weights.max() <= 0.6 + 1e-5


def test_optimizer_cannot_sell_an_incumbent_still_inside_buffer():
    cfg = BacktestConfig(
        portfolio_method="optimizer",
        top_n=2,
        buy_rank=2,
        sell_rank=3,
        single_weight_cap=0.5,
    )
    optimizer = PortfolioOptimizer(
        PortfolioConstraints(single_weight_cap=0.5),
        PortfolioObjective(risk_aversion=0.0),
    )
    prev = np.asarray([0.5, 0.5, 0.0])
    out = _construct(
        cfg,
        np.asarray([3.0, 1.0, 2.0]),
        prev,
        optimizer=optimizer,
        capital=1_000_000.0,
    )
    assert out.weights == pytest.approx(prev, abs=1e-6)
    assert out.sell_weights.sum() == 0.0


def test_production_defaults_prevent_thirty_micro_orders_at_100k():
    root = Path(__file__).resolve().parents[1]
    cfg = make_backtest_config(load_config(project_root=root))
    assert cfg.buy_rank == 20
    assert cfg.sell_rank == 30
    assert cfg.min_trade_amount == 5_000.0
    assert cfg.target_weight_change_threshold == 0.01
    assert cfg.turnover_budget == 0.20

    out = _construct(cfg, _signal_from_order(list(range(30))))
    assert out.order_count == 20
    assert np.count_nonzero(out.buy_weights) == 20
    assert out.buy_weights[out.buy_weights > 0] * 100_000.0 == pytest.approx(
        np.full(20, 5_000.0)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"portfolio_method": "hand_rolled"},
        {"buy_rank": 0},
        {"buy_rank": 20, "sell_rank": 19},
        {"min_trade_amount": 0.0},
        {"turnover_budget": -0.1},
        {"target_weight_change_threshold": 1.1},
    ],
)
def test_invalid_portfolio_config_fails_at_load_boundary(overrides: dict):
    with pytest.raises(ValueError):
        make_backtest_config({"backtest": overrides})
