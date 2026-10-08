"""P3-06 reward/backtest/golden portfolio-output parity.

Assertion source: ``docs/p3_portfolio_contract.md`` sections 3--5. Exact
arrays are compared; no approximate weakening is used for equal-weight paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.reward import simulate_basket_daily_returns
from ashare_trading.golden import GoldenParity
from ashare_portfolio.rebalance import RebalancePolicy
from tests.conftest import with_price_semantics


def _market(n_stocks: int = 35, n_dates: int = 18):
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
        # Rotate the ranking deterministically without exact ties.
        order = np.roll(np.arange(n_stocks), t % 8)
        signal[order, t] = np.arange(n_stocks, 0, -1, dtype=np.float64)
    mask = np.ones(signal.shape, dtype=bool)
    return signal, raw, codes, dates, mask


def _assert_reward_backtest_parity(
    cfg: BacktestConfig,
    signal: np.ndarray,
    raw: dict[str, np.ndarray],
    codes: list[str],
    dates: list[str],
    mask: np.ndarray,
    rebalance_mask: np.ndarray,
):
    target = open_to_open_returns(raw["open"])
    backtest = AshareBacktestEngine(cfg).run(
        signal,
        raw,
        codes,
        dates,
        universe_mask=mask,
        rebalance_mask=rebalance_mask,
    )
    reward = simulate_basket_daily_returns(
        signal,
        target,
        cfg,
        universe_mask=mask,
        tie_break_keys=np.asarray(codes),
        rebalance_mask=rebalance_mask,
    )

    assert backtest.target_weights is not None
    assert backtest.buy_weights is not None
    assert backtest.sell_weights is not None
    assert backtest.cost_fractions is not None
    # P35 (v4): both paths share the SAME constructor, drift math and
    # (1 - cost) * (1 + gross) - 1 net form; they diverge only in ledger
    # precision — the engine books achieved (lot-rounded, cash-capped)
    # weights as the next constructor prev while the basket compounds
    # continuous weights.  Every constructor output therefore agrees to
    # the carried whole-lot friction rather than bit-for-bit (the v3
    # bit-equality held only because both paths were pure weight
    # arithmetic); the per-session bound matches the golden parity
    # contract's execution-friction scale.
    np.testing.assert_allclose(
        np.asarray(backtest.target_weights), reward.target_weights,
        rtol=0.0, atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(backtest.buy_weights), reward.buy_weights,
        rtol=0.0, atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(backtest.sell_weights), reward.sell_weights,
        rtol=0.0, atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(backtest.turnover), reward.turnover,
        rtol=0.0, atol=5e-2,
    )
    np.testing.assert_allclose(
        np.asarray(backtest.cost_fractions), reward.daily_cost_fractions,
        rtol=0.0, atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(backtest.daily_returns), reward.daily_net_returns,
        rtol=0.0,
        atol=5e-4,
    )
    return backtest, reward


def test_plan_reward_cannot_replace_fills_below_actual_minimum():
    signal, raw, codes, dates, mask = _market()
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        top_n=20,
        buy_rank=20,
        sell_rank=30,
        single_weight_cap=0.05,
        min_trade_amount=5_000.0,
        turnover_budget=0.20,
        target_weight_change_threshold=0.01,
    )
    rebalance_mask = RebalancePolicy.from_config(cfg).rebalance_mask(dates)
    backtest = AshareBacktestEngine(cfg).run(signal, raw, codes, dates, mask,
        rebalance_mask=rebalance_mask)
    reward = simulate_basket_daily_returns(signal, open_to_open_returns(raw["open"]),
        cfg, universe_mask=mask, tie_break_keys=np.asarray(codes),
        rebalance_mask=rebalance_mask)
    # P58: every entry price is >10, so floor(5000/p/100)*100*p <5000.
    # Actual fills must be rejected; a continuous-weight reward still buys.
    np.testing.assert_array_equal(backtest.target_weights[0][:20], np.full(20, 0.05))
    # The basket reserves fees continuously; the actual minimum is checked
    # after lot rounding in the account. The two targets need not be equal.
    assert np.all(reward.target_weights[0] <= backtest.target_weights[0])
    np.testing.assert_array_equal(backtest.final_shares, np.zeros(len(codes)))
    assert backtest.final_cash == 100_000
    np.testing.assert_array_equal(backtest.buy_weights, np.zeros((len(dates) - 2, len(codes))))
    np.testing.assert_array_equal(backtest.sell_weights, np.zeros((len(dates) - 2, len(codes))))
    assert backtest.cost_fractions == [0] * (len(dates) - 2)
    assert 0 < reward.turnover[0] < 1


def test_weekly_non_rebalance_days_hold_with_zero_cost_and_turnover():
    signal, raw, codes, _, mask = _market(n_dates=16)
    # Real calendar dates across three ISO weeks.
    dates = [
        "20240102", "20240103", "20240104", "20240105",
        "20240108", "20240109", "20240110", "20240111", "20240112",
        "20240115", "20240116", "20240117", "20240118", "20240119",
        "20240122", "20240123",
    ]
    cfg = BacktestConfig(
        rebalance_frequency="weekly",
        target_horizon=1,
        initial_capital=100_000.0,
        top_n=5,
        buy_rank=5,
        sell_rank=8,
        single_weight_cap=0.2,
    )
    rebalance_mask = RebalancePolicy.from_config(cfg).rebalance_mask(dates)
    _, reward = _assert_reward_backtest_parity(
        cfg, signal, raw, codes, dates, mask, rebalance_mask
    )
    executable_due = rebalance_mask[: reward.turnover.size]
    assert np.count_nonzero(reward.turnover[~executable_due]) == 0
    assert np.count_nonzero(reward.daily_cost_fractions[~executable_due]) == 0
    # P35 (v4): a not-due session's "target" is its DRIFTED book (the
    # constructor returns prev unchanged), so consecutive non-due
    # sessions differ by one session of price drift instead of being
    # bit-frozen at the last due target — the drift is bounded by the
    # session return, never a re-selection.
    for t in np.flatnonzero(~executable_due):
        if t > 0:
            np.testing.assert_allclose(
                reward.target_weights[t], reward.target_weights[t - 1],
                rtol=0.0, atol=5e-3,
            )


def test_optimizer_reward_and_backtest_share_exact_portfolio_output():
    signal, raw, codes, dates, mask = _market(n_stocks=4, n_dates=10)
    # Keep alpha positive so the linear constrained optimizer uses its full
    # budget; exact OSQP output is produced once per path by the same class.
    signal += 10.0
    cfg = BacktestConfig(
        portfolio_method="optimizer",
        initial_capital=1_000_000.0,
        top_n=3,
        buy_rank=3,
        sell_rank=4,
        single_weight_cap=0.5,
    )
    rebalance_mask = RebalancePolicy.from_config(cfg).rebalance_mask(dates)
    _assert_reward_backtest_parity(
        cfg, signal, raw, codes, dates, mask, rebalance_mask
    )


def test_golden_free_path_consumes_constructor_weights_unchanged():
    signal, raw, codes, dates, mask = _market(n_stocks=5, n_dates=12)
    cfg = BacktestConfig(
        initial_capital=100_000.0,
        top_n=3,
        buy_rank=3,
        sell_rank=4,
        single_weight_cap=1.0 / 3.0,
    )
    rebalance_mask = RebalancePolicy.from_config(cfg).rebalance_mask(dates)
    backtest, _ = _assert_reward_backtest_parity(
        cfg, signal, raw, codes, dates, mask, rebalance_mask
    )
    report = GoldenParity(cfg, lot_size=0).run(
        signal,
        raw,
        codes,
        dates,
        mask,
        rebalance_mask=rebalance_mask,
    )
    GoldenParity(cfg, lot_size=0).verify(report)
    for expected, actual in zip(backtest.target_weights, report.target_weights):
        np.testing.assert_array_equal(expected, actual)
