from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.reward import (
    batched_basket_rewards,
    fast_basket_reward,
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


# --- fast_basket_reward ----------------------------------------------------


def test_churn_is_charged_more_than_a_static_basket():
    cfg = _cfg()
    target = np.full((3, 4), 0.002)
    static_signal = np.array(
        [[3.0, 3.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]
    )
    churn_signal = np.array(
        [[3.0, 1.0, 3.0, 1.0], [2.0, 2.0, 2.0, 2.0], [1.0, 3.0, 1.0, 3.0]]
    )
    _, mean_static = fast_basket_reward(static_signal, target, cfg, RewardConfig())
    _, mean_churn = fast_basket_reward(churn_signal, target, cfg, RewardConfig())
    assert mean_static > mean_churn
    # The two flip days each pay one full buy + sell round trip on the
    # swapped positions; the entry day cost is identical for both signals.
    buy_cost = cfg.commission_rate * 0.5 + (cfg.transfer_fee_rate + cfg.slippage_rate) * 0.5
    sell_cost = buy_cost + cfg.stamp_tax_rate * 0.5
    assert mean_static - mean_churn == pytest.approx(
        2 * (buy_cost + sell_cost) / 3, abs=1e-12
    )


def test_fast_reward_survives_nonfinite_targets():
    cfg = _cfg()
    signal = np.array([[3.0] * 4, [2.0] * 4, [1.0] * 4])
    target = np.array(
        [
            [0.01, np.inf, 0.01, np.nan],
            [0.02, 0.02, 0.02, 0.02],
            [0.03, 0.03, 0.03, 0.03],
        ]
    )
    reward, mean = fast_basket_reward(signal, target, cfg, RewardConfig())
    assert np.isfinite(reward)
    assert np.isfinite(mean)


def test_losing_basket_collapses_to_bad_reward():
    cfg = _cfg(commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0,
               transfer_fee_rate=0.0, slippage_rate=0.0)
    signal = np.array([[3.0] * 4, [2.0] * 4, [1.0] * 4])
    target = np.full((3, 4), -0.001)
    reward, _ = fast_basket_reward(signal, target, cfg, RewardConfig())
    assert reward == -2.0


def test_turnover_penalty_subtracts_when_above_threshold():
    cfg = _cfg(commission_rate=0.0, min_commission=0.0, stamp_tax_rate=0.0,
               transfer_fee_rate=0.0, slippage_rate=0.0)
    # Identical targets per stock make the top-2 basket return equal to the
    # daily pattern below; a static basket keeps turnover at the entry-day
    # value only (1/6), between the two thresholds.
    daily = [0.01, -0.01, 0.005, -0.005, 0.01, -0.008]
    signal = np.array([[3.0] * 7, [2.0] * 7, [1.0] * 7])
    target = np.tile(np.asarray(daily), (3, 1)).astype(float)
    target = np.column_stack([target, np.zeros(3)])
    penalized, _ = fast_basket_reward(
        signal, target, cfg,
        RewardConfig(turnover_threshold=0.1, turnover_penalty=2.5),
    )
    unpenalized, _ = fast_basket_reward(
        signal, target, cfg,
        RewardConfig(turnover_threshold=1.0, turnover_penalty=2.5),
    )
    sortino = sortino_ratio(np.asarray(daily, dtype=np.float64))
    assert 0.0 < sortino < RewardConfig().reward_clip_high
    assert unpenalized == pytest.approx(sortino, rel=1e-12)
    assert penalized == pytest.approx(sortino - 2.5, rel=1e-12)


def test_degenerate_input_returns_clip_low():
    cfg = _cfg()
    reward, mean = fast_basket_reward(
        np.zeros((3, 1)), np.zeros((3, 1)), cfg, RewardConfig()
    )
    assert reward == RewardConfig().reward_clip_low
    assert mean == 0.0


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
    _, mean = fast_basket_reward(signal, target, cfg, RewardConfig())
    assert mean == pytest.approx(0.001, abs=1e-12)


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
        val_start = n_dates // 2 + 1
        df, tf, dv, tv = simulate_basket_daily_returns_batch(
            signals, target, cfg, val_start=val_start
        )
        for i in range(b):
            ref_f, ref_tf = simulate_basket_daily_returns(signals[i], target, cfg)
            ref_v, ref_tv = simulate_basket_daily_returns(
                signals[i, :, val_start:], target[:, val_start:], cfg
            )
            assert df[i] == pytest.approx(ref_f, rel=1e-9, abs=1e-11)
            assert tf[i] == pytest.approx(ref_tf, rel=1e-9, abs=1e-11)
            assert dv[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-11)
            assert tv[i] == pytest.approx(ref_tv, rel=1e-9, abs=1e-11)


def test_batch_sim_val_basket_restarts_from_zero():
    # The validation basket must behave exactly like the scalar path on the
    # sliced signal: fresh zero weights at val_start, no carry-in from the
    # full-window basket even though both share one pass over the dates.
    rng = np.random.default_rng(6)
    signals, target = _random_batch(rng, 2, 6, 20, nan_frac=0.0)
    val_start = 12
    cfg = _cfg(top_n=2)
    df, _, dv, _ = simulate_basket_daily_returns_batch(
        signals, target, cfg, val_start=val_start
    )
    for i in range(2):
        ref_v, _ = simulate_basket_daily_returns(
            signals[i, :, val_start:], target[:, val_start:], cfg
        )
        assert dv[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-11)


def test_batch_sim_degenerate_shapes_match_scalar():
    cfg = _cfg()
    # Fewer than two dates: empty daily series.
    df, tf, dv, tv = simulate_basket_daily_returns_batch(
        np.zeros((2, 3, 1)), np.zeros((3, 1)), cfg, val_start=1
    )
    assert df.shape == (2, 0) and dv.shape == (2, 0)
    assert list(tf) == [0.0, 0.0] and list(tv) == [0.0, 0.0]
    ref_daily, ref_to = simulate_basket_daily_returns(
        np.zeros((3, 1)), np.zeros((3, 1)), cfg
    )
    assert ref_daily.size == 0 and ref_to == 0.0
    # top_n <= 0: every day skipped, zero daily and zero turnover.
    df, tf, dv, tv = simulate_basket_daily_returns_batch(
        np.zeros((2, 3, 6)), np.zeros((3, 6)), _cfg(top_n=0), val_start=3
    )
    assert np.all(df == 0.0) and np.all(dv == 0.0)
    assert list(tf) == [0.0, 0.0] and list(tv) == [0.0, 0.0]
    # Not enough finite rows on any day: same zero behavior.
    sig = np.full((1, 3, 6), np.nan)
    df, tf, dv, tv = simulate_basket_daily_returns_batch(sig, np.zeros((3, 6)), cfg, val_start=3)
    assert np.all(df == 0.0) and np.all(dv == 0.0)


def test_batched_rewards_match_scalar_rewards():
    rng = np.random.default_rng(7)
    cfg = _cfg(top_n=2, single_weight_cap=1.0)
    reward_cfg = RewardConfig(turnover_threshold=0.4, turnover_penalty=1.5)
    for _ in range(8):
        b = int(rng.integers(1, 6))
        n_stocks = int(rng.integers(3, 15))
        n_dates = int(rng.integers(10, 60))
        signals, target = _random_batch(rng, b, n_stocks, n_dates)
        val_start = max(1, n_dates // 2)
        rewards, val_rewards = batched_basket_rewards(
            signals, target, cfg, reward_cfg, val_start
        )
        for i in range(b):
            ref_r, _ = fast_basket_reward(signals[i], target, cfg, reward_cfg)
            ref_v, _ = fast_basket_reward(
                signals[i, :, val_start:], target[:, val_start:], cfg, reward_cfg
            )
            assert rewards[i] == pytest.approx(ref_r, rel=1e-9, abs=1e-10)
            assert val_rewards[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-10)


def test_batched_rewards_are_deterministic():
    rng = np.random.default_rng(8)
    signals, target = _random_batch(rng, 4, 8, 25)
    cfg = _cfg()
    reward_cfg = RewardConfig()
    first = batched_basket_rewards(signals, target, cfg, reward_cfg, 12)
    second = batched_basket_rewards(signals, target, cfg, reward_cfg, 12)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    # Rewards stay inside the configured clip band.
    assert np.all(first[0] >= reward_cfg.reward_clip_low)
    assert np.all(first[0] <= reward_cfg.reward_clip_high)
