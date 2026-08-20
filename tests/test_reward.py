from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.reward import (
    annualized_turnover_cost,
    batched_basket_rewards,
    formula_reward,
    icir_from_series,
    per_turnover_cost_rate,
    rank_ic_series,
    simulate_basket_daily_returns,
    simulate_basket_daily_returns_batch,
    sortino_ratio,
    trading_cost_fraction,
)


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=100000.0,
        top_n=2,
        single_weight_cap=0.5,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _reward_cfg(**kwargs) -> RewardConfig:
    defaults = dict(ic_min_stocks=2)
    defaults.update(kwargs)
    return RewardConfig(**defaults)


# --- trading_cost_fraction -------------------------------------------------


def test_cost_min_commission_floor_binds_for_small_positions():
    # 30 positions of 1/30 each on a 100k account: the proportional
    # commission (0.83 yuan) is below the 5 yuan floor, so the floor binds.
    cfg = _cfg(top_n=30, single_weight_cap=1.0)
    buy = np.full(30, 1.0 / 30)
    sell = np.full(30, 1.0 / 30)
    min_fee = cfg.min_commission / cfg.initial_capital
    expected = 30 * (
        min_fee + (cfg.transfer_fee_rate + cfg.slippage_rate) / 30
        + min_fee
        + (cfg.stamp_tax_rate + cfg.transfer_fee_rate + cfg.slippage_rate) / 30
    )
    assert trading_cost_fraction(buy, sell, cfg) == pytest.approx(expected, rel=1e-9)


def test_cost_is_purely_proportional_when_floor_does_not_bind():
    cfg = _cfg(initial_capital=10_000_000.0)
    buy = np.array([0.5, 0.0])
    sell = np.array([0.0, 0.5])
    expected_buy = (
        cfg.commission_rate * 0.5 + (cfg.transfer_fee_rate + cfg.slippage_rate) * 0.5
    )
    expected_sell = (
        cfg.commission_rate * 0.5
        + (cfg.stamp_tax_rate + cfg.transfer_fee_rate + cfg.slippage_rate) * 0.5
    )
    assert trading_cost_fraction(buy, sell, cfg) == pytest.approx(
        expected_buy + expected_sell, rel=1e-12
    )


def test_cost_zero_when_no_trades():
    # The minimum commission is a per-trade floor: zero trades -> zero cost.
    cfg = _cfg()
    assert trading_cost_fraction(np.zeros(3), np.zeros(3), cfg) == 0.0


def test_cost_buy_sell_asymmetry_is_stamp_tax_only():
    cfg = _cfg(initial_capital=10_000_000.0)
    w = np.array([1.0])
    buy = trading_cost_fraction(w, np.zeros(1), cfg)
    sell = trading_cost_fraction(np.zeros(1), w, cfg)
    assert sell - buy == pytest.approx(cfg.stamp_tax_rate, rel=1e-12)


# --- sortino_ratio ---------------------------------------------------------


def test_sortino_empty_returns_zero():
    assert sortino_ratio(np.array([])) == 0.0


def test_sortino_uses_downside_deviation_with_enough_negative_days():
    daily = np.array([0.02, -0.01, 0.02, -0.02, 0.02, -0.015])
    ann_mean = daily.mean() * 252
    downside = daily[daily < 0]
    expected = ann_mean / (downside.std(ddof=1) * math.sqrt(252) + 1e-9)
    assert sortino_ratio(daily) == pytest.approx(expected, rel=1e-12)


def test_sortino_falls_back_to_full_volatility_with_few_negative_days():
    daily = np.array([0.01, -0.005, 0.01, 0.01, 0.01, 0.01])
    ann_mean = daily.mean() * 252
    expected = ann_mean / (daily.std(ddof=1) * math.sqrt(252) + 1e-6 + 1e-9)
    assert sortino_ratio(daily) == pytest.approx(expected, rel=1e-12)


# --- rank IC / ICIR --------------------------------------------------------


def test_rank_ic_series_perfect_positive_correlation():
    rng = np.random.default_rng(0)
    n_stocks, n_dates = 12, 20
    target = rng.normal(size=(n_stocks, n_dates))
    signals = np.stack([target, -target]).astype(float)
    ic = rank_ic_series(signals, target, min_stocks=5)
    assert np.allclose(ic[0], 1.0)
    assert np.allclose(ic[1], -1.0)
    assert icir_from_series(ic)[0] > 100.0  # near-perfect: huge positive ICIR
    assert icir_from_series(ic)[1] < -100.0


def test_rank_ic_series_average_ranks_tie_handling():
    # A constant signal cross-section has zero variance: the correlation is
    # undefined, so those dates contribute NaN and the row scores ICIR 0.
    target = np.array(
        [[0.01, 0.02], [0.02, 0.01], [-0.01, 0.03], [0.005, 0.001]]
    )
    constant = np.ones_like(target)
    ic = rank_ic_series(constant[None], target, min_stocks=2)
    assert np.isnan(ic).all()
    assert icir_from_series(ic)[0] == 0.0


def test_rank_ic_series_skips_days_below_min_stocks():
    # Three valid pairs on day 0, only one on days 1-2: they must be NaN.
    target = np.array(
        [[0.01, 0.02, np.nan], [0.02, np.nan, np.nan], [0.03, np.nan, 0.02]]
    )
    signal = target.copy()
    ic = rank_ic_series(signal[None], target, min_stocks=2)
    assert ic[0, 0] == pytest.approx(1.0)
    assert np.isnan(ic[0, 1])
    assert np.isnan(ic[0, 2])


def test_rank_ic_series_excludes_nonfinite_signal_cells_per_row():
    # A NaN signal cell must be excluded (per row): the surviving pairs keep
    # the exact monotonic correlation.  A finite but extreme replacement
    # would have polluted the rank and moved the IC away from 1.
    target = np.tile(np.array([0.01, 0.02, 0.03, 0.04])[:, None], (1, 5))
    signal = target.copy()
    signal[0, :] = np.nan
    ic = rank_ic_series(signal[None], target, min_stocks=2)
    assert np.allclose(ic[0], 1.0)
    extreme = target.copy()
    extreme[0, :] = 1e9
    ic_extreme = rank_ic_series(extreme[None], target, min_stocks=2)
    assert not np.allclose(ic_extreme[0], 1.0)


def test_icir_zero_with_fewer_than_two_observations():
    assert icir_from_series(np.array([np.nan, 0.5])).tolist() == [0.0]
    # Constant positive IC -> mean/std explodes; mean-zero IC -> exact 0.
    icir = icir_from_series(np.array([[1.0, 1.0], [1.0, -1.0]]))
    assert icir[0] > 1e8
    assert icir[1] == 0.0


# --- turnover cost (continuous, rate-proportional) --------------------------


def test_per_turnover_cost_rate_formula():
    cfg = _cfg()
    assert per_turnover_cost_rate(cfg) == pytest.approx(
        2 * cfg.commission_rate
        + cfg.stamp_tax_rate
        + 2 * cfg.transfer_fee_rate
        + 2 * cfg.slippage_rate,
        rel=1e-12,
    )


def test_annualized_turnover_cost_is_linear_in_turnover():
    cfg = _cfg()
    rate = per_turnover_cost_rate(cfg)
    assert annualized_turnover_cost(np.array([0.0, 0.5]), cfg).tolist() == [
        pytest.approx(0.0),
        pytest.approx(0.5 * rate * 252, rel=1e-12),
    ]


# --- basket simulation vs the full backtest engine -------------------------


def _synthetic_market(n_stocks=5, n_dates=6):
    ts_codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    dates = [f"2024{i:02d}01" for i in range(1, n_dates + 1)]
    rng = np.random.default_rng(7)
    open_ = np.cumprod(1.0 + rng.normal(0.0005, 0.005, (n_stocks, n_dates)), axis=1) * 10.0
    pre_close = np.roll(open_, 1, axis=1)
    pre_close[:, 0] = open_[:, 0]
    raw = {
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": pre_close,
        "volume": np.full((n_stocks, n_dates), 1_000_000.0),
    }
    signal = rng.normal(size=(n_stocks, n_dates))
    target = open_to_open_returns(open_)
    return ts_codes, dates, raw, signal, target


def test_simulate_basket_matches_backtest_engine_daily_returns():
    ts_codes, dates, raw, signal, target = _synthetic_market()
    cfg = _cfg()
    result = AshareBacktestEngine(cfg).run(
        signal, raw, ts_codes, dates, stock_names={}
    )
    daily, avg_turnover = simulate_basket_daily_returns(signal, target, cfg)
    assert daily == pytest.approx(np.asarray(result.daily_returns), abs=1e-12)
    assert avg_turnover == pytest.approx(
        float(np.mean(result.turnover)), abs=1e-12
    )


def test_nonfinite_signal_rows_are_excluded_from_selection():
    cfg = _cfg(top_n=3, commission_rate=0.0, min_commission=0.0,
               stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_rate=0.0)
    # Row 0 is NaN everywhere but has a huge target: it must never be
    # selected even though NaNs sort unpredictably under argpartition.
    signal = np.array(
        [
            [np.nan] * 4,
            [3.0] * 4,
            [2.0] * 4,
            [1.0] * 4,
        ]
    )
    target = np.array(
        [
            [1.0] * 4,
            [0.001] * 4,
            [0.001] * 4,
            [0.001] * 4,
        ]
    )
    daily, _ = simulate_basket_daily_returns(signal, target, cfg)
    assert daily == pytest.approx(np.full(3, 0.001), abs=1e-12)


# --- v3 reward semantics ----------------------------------------------------


def test_formula_reward_is_icir_minus_continuous_cost():
    cfg = _cfg()
    rc = _reward_cfg()
    rng = np.random.default_rng(3)
    n_stocks, n_dates = 10, 30
    target = rng.normal(size=(n_stocks, n_dates))
    signal = target.copy()
    # Near-perfect IC -> the ICIR term saturates the clip band.
    reward = formula_reward(signal, target, cfg, rc)
    assert reward == pytest.approx(rc.reward_clip_high, abs=1e-12)


def test_reward_penalizes_churn_continuously():
    # The cost drag is exactly cost_weight * annualized turnover cost: with
    # cost_weight 1 vs 0 the reward difference equals the proportional cost,
    # inside the clip band (no threshold jumps).
    cfg = _cfg()
    n_stocks, n_dates = 6, 12
    rng = np.random.default_rng(11)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    static = np.tile(np.arange(n_stocks, dtype=float)[::-1][:, None], (1, n_dates))
    churn = static.copy()
    churn[0, 1::2] = 1.0  # odd days: stock 0 drops out of the top-2 basket
    _, to_static = simulate_basket_daily_returns(static, target, cfg)
    _, to_churn = simulate_basket_daily_returns(churn, target, cfg)
    assert to_churn > to_static

    for signal in (static, churn):
        _, to = simulate_basket_daily_returns(signal, target, cfg)
        expected_cost = annualized_turnover_cost(np.asarray([to]), cfg)[0]
        r_free = formula_reward(signal, target, cfg, _reward_cfg(cost_weight=0.0))
        r_honest = formula_reward(signal, target, cfg, _reward_cfg(cost_weight=1.0))
        assert -1.0 < r_honest < 1.0  # inside the band: the diff is exact
        assert r_free - r_honest == pytest.approx(expected_cost, abs=1e-12)


def test_degenerate_input_has_zero_icir_and_no_cost():
    # One date: no IC observations and no daily return -> reward 0.0
    # (the trainer reserves clip_low/bad_reward for invalid formulas).
    cfg = _cfg()
    rc = _reward_cfg()
    reward = formula_reward(np.zeros((3, 1)), np.zeros((3, 1)), cfg, rc)
    assert reward == 0.0


def test_formula_reward_clips_to_configured_band():
    cfg = _cfg()
    rc = _reward_cfg(reward_clip_low=-0.25, reward_clip_high=0.5)
    rng = np.random.default_rng(4)
    target = rng.normal(size=(10, 40))
    signal = -target  # ICIR strongly negative
    assert formula_reward(signal, target, cfg, rc) == -0.25


# --- batched basket simulation: equivalence with the scalar path ------------


def _random_batch(rng, b, n_stocks, n_dates, nan_frac=0.1):
    signals = rng.normal(size=(b, n_stocks, n_dates))
    signals[rng.random((b, n_stocks, n_dates)) < nan_frac] = np.nan
    target = rng.normal(0.0, 0.01, size=(n_stocks, n_dates))
    target[rng.random((n_stocks, n_dates)) < 0.05] = np.nan
    return signals, target


def test_batch_sim_matches_scalar_reference():
    rng = np.random.default_rng(5)
    cfg = _cfg(top_n=3, single_weight_cap=1.0)
    for b, n_stocks, n_dates in ((1, 6, 30), (4, 12, 45), (3, 5, 8)):
        signals, target = _random_batch(rng, b, n_stocks, n_dates)
        df, tf = simulate_basket_daily_returns_batch(signals, target, cfg)
        for i in range(b):
            ref_f, ref_tf = simulate_basket_daily_returns(signals[i], target, cfg)
            assert df[i] == pytest.approx(ref_f, rel=1e-9, abs=1e-11)
            assert tf[i] == pytest.approx(ref_tf, rel=1e-9, abs=1e-11)


def test_batch_sim_restarting_a_window_matches_scalar():
    # A window simulated on its own columns equals the scalar path on the
    # same slice: the window basket starts from zero weights, exactly like
    # a fresh out-of-sample deployment.
    rng = np.random.default_rng(6)
    signals, target = _random_batch(rng, 2, 6, 20, nan_frac=0.0)
    cfg = _cfg(top_n=2)
    start, end = 8, 16
    df, _ = simulate_basket_daily_returns_batch(
        signals[:, :, start:end], target[:, start:end], cfg
    )
    for i in range(2):
        ref_v, _ = simulate_basket_daily_returns(
            signals[i, :, start:end], target[:, start:end], cfg
        )
        assert df[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-11)


def test_batch_sim_degenerate_shapes_match_scalar():
    cfg = _cfg()
    # Fewer than two dates: empty daily series.
    df, tf = simulate_basket_daily_returns_batch(
        np.zeros((2, 3, 1)), np.zeros((3, 1)), cfg
    )
    assert df.shape == (2, 0)
    assert list(tf) == [0.0, 0.0]
    ref_daily, ref_to = simulate_basket_daily_returns(
        np.zeros((3, 1)), np.zeros((3, 1)), cfg
    )
    assert ref_daily.size == 0 and ref_to == 0.0
    # top_n <= 0: every day skipped, zero daily and zero turnover.
    df, tf = simulate_basket_daily_returns_batch(
        np.zeros((2, 3, 6)), np.zeros((3, 6)), _cfg(top_n=0)
    )
    assert np.all(df == 0.0)
    assert list(tf) == [0.0, 0.0]
    # Not enough finite rows on any day: same zero behavior.
    sig = np.full((1, 3, 6), np.nan)
    df, tf = simulate_basket_daily_returns_batch(sig, np.zeros((3, 6)), cfg)
    assert np.all(df == 0.0)
    assert list(tf) == [0.0]


def test_batched_rewards_match_scalar_rewards():
    rng = np.random.default_rng(7)
    cfg = _cfg(top_n=2, single_weight_cap=1.0)
    reward_cfg = _reward_cfg()
    for _ in range(8):
        b = int(rng.integers(1, 6))
        n_stocks = int(rng.integers(3, 15))
        n_dates = int(rng.integers(10, 60))
        signals, target = _random_batch(rng, b, n_stocks, n_dates)
        val_windows = [(n_dates // 2, n_dates)]
        rewards, val_rewards = batched_basket_rewards(
            signals, target, cfg, reward_cfg, val_windows
        )
        for i in range(b):
            ref_r = formula_reward(signals[i], target, cfg, reward_cfg)
            ref_v = formula_reward(
                signals[i, :, val_windows[0][0] :], target[:, val_windows[0][0] :],
                cfg, reward_cfg,
            )
            assert rewards[i] == pytest.approx(ref_r, rel=1e-9, abs=1e-10)
            assert val_rewards[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-10)


def test_val_reward_is_median_over_windows():
    rng = np.random.default_rng(9)
    cfg = _cfg(top_n=2, single_weight_cap=1.0)
    reward_cfg = _reward_cfg()
    n_stocks, n_dates = 8, 30
    target = rng.normal(0.0, 0.01, size=(n_stocks, n_dates))
    signals = rng.normal(size=(2, n_stocks, n_dates))
    # Aligned in window 1, inverted in window 2, noise in window 3.
    aligned = np.stack([target, -target, rng.normal(size=target.shape)])
    windows = [(0, 10), (10, 20), (20, 30)]
    rewards, val_rewards = batched_basket_rewards(
        aligned, target, cfg, reward_cfg, windows
    )
    per_window = []
    for start, end in windows:
        per_window.append(
            np.asarray(
                [
                    formula_reward(
                        aligned[i, :, start:end], target[:, start:end], cfg, reward_cfg
                    )
                    for i in range(aligned.shape[0])
                ]
            )
        )
    expected = np.median(np.stack(per_window, axis=1), axis=1)
    assert val_rewards == pytest.approx(expected, abs=1e-12)
    # The perfect-alignment window dominates the median for row 0.
    assert val_rewards[0] == pytest.approx(reward_cfg.reward_clip_high, abs=1e-12)
    assert rewards[0] == pytest.approx(
        formula_reward(aligned[0], target, cfg, reward_cfg), abs=1e-12
    )


def test_batched_rewards_without_val_windows_returns_none():
    rng = np.random.default_rng(10)
    signals, target = _random_batch(rng, 3, 8, 25)
    rewards, val_rewards = batched_basket_rewards(
        signals, target, _cfg(), _reward_cfg()
    )
    assert rewards.shape == (3,)
    assert val_rewards is None


def test_batched_rewards_are_deterministic():
    rng = np.random.default_rng(8)
    signals, target = _random_batch(rng, 4, 8, 25)
    cfg = _cfg()
    reward_cfg = _reward_cfg()
    windows = [(0, 8), (8, 16)]
    first = batched_basket_rewards(signals, target, cfg, reward_cfg, windows)
    second = batched_basket_rewards(signals, target, cfg, reward_cfg, windows)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    # Rewards stay inside the configured clip band.
    assert np.all(first[0] >= reward_cfg.reward_clip_low)
    assert np.all(first[0] <= reward_cfg.reward_clip_high)
    assert np.all(first[1] >= reward_cfg.reward_clip_low)
    assert np.all(first[1] <= reward_cfg.reward_clip_high)
