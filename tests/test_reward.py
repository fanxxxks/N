from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_data.processor import open_to_open_returns, tradability_blocked_matrix
from ashare_execution import ExecutionCostModel
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.reward import (
    batched_basket_rewards,
    formula_reward,
    icir_from_series,
    rank_ic_series,
    signal_direction,
    simulate_basket_daily_returns,
    simulate_basket_daily_returns_batch,
    sortino_ratio,
)
from ashare_model.targets import causal_target_returns
from ashare_portfolio.rebalance import RebalancePolicy


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


def test_multi_period_research_target_never_replaces_daily_portfolio_returns():
    n_stocks, n_dates = 4, 18
    dates = [f"202401{i + 1:02d}" for i in range(n_dates)]
    open_ = (
        10.0
        + np.arange(n_stocks, dtype=np.float64)[:, None]
        + np.arange(n_dates, dtype=np.float64)[None, :]
        * np.asarray([0.05, 0.10, 0.20, 0.30])[:, None]
    )
    signal = np.tile(np.arange(n_stocks, dtype=np.float64)[:, None], (1, n_dates))
    cfg = _cfg(
        rebalance_frequency="every_5_days",
        target_horizon=5,
        top_n=2,
        buy_rank=2,
        sell_rank=2,
    )
    policy = RebalancePolicy.from_config(cfg)
    rebalance_mask = policy.rebalance_mask(dates)
    target = causal_target_returns(open_, dates, policy)
    realized = open_to_open_returns(open_)
    mask = np.ones_like(signal, dtype=bool)

    first = batched_basket_rewards(
        signal[None],
        target,
        cfg,
        _reward_cfg(),
        train_signal_range=(0, n_dates - policy.exit_offset),
        universe_mask=mask,
        realized_ret=realized,
        rebalance_mask=rebalance_mask,
    )
    second = batched_basket_rewards(
        signal[None],
        -target,
        cfg,
        _reward_cfg(),
        train_signal_range=(0, n_dates - policy.exit_offset),
        universe_mask=mask,
        realized_ret=realized,
        rebalance_mask=rebalance_mask,
    )
    # Portfolio reward/cost uses the same daily path; only research IC sees
    # the changed five-day label orientation.
    np.testing.assert_array_equal(first[0], second[0])
    assert first[2][0] != second[2][0]


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
    model = ExecutionCostModel.from_config(cfg)
    assert model.rebalance_cost_fraction(buy, sell, cfg.initial_capital) == pytest.approx(expected, rel=1e-9)


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
    assert ExecutionCostModel.from_config(cfg).rebalance_cost_fraction(
        buy, sell, cfg.initial_capital
    ) == pytest.approx(
        expected_buy + expected_sell, rel=1e-12
    )


def test_cost_zero_when_no_trades():
    # The minimum commission is a per-trade floor: zero trades -> zero cost.
    cfg = _cfg()
    assert ExecutionCostModel.from_config(cfg).rebalance_cost_fraction(
        np.zeros(3), np.zeros(3), cfg.initial_capital
    ) == 0.0


def test_cost_buy_sell_asymmetry_is_stamp_tax_only():
    cfg = _cfg(initial_capital=10_000_000.0)
    w = np.array([1.0])
    model = ExecutionCostModel.from_config(cfg)
    buy = model.rebalance_cost_fraction(w, np.zeros(1), cfg.initial_capital)
    sell = model.rebalance_cost_fraction(np.zeros(1), w, cfg.initial_capital)
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
    # One epsilon guards the denominator on both branches; the fallback
    # branch previously added a stray second epsilon.
    expected = ann_mean / (daily.std(ddof=1) * math.sqrt(252) + 1e-9)
    assert sortino_ratio(daily) == pytest.approx(expected, rel=1e-12)


# --- rank IC / ICIR --------------------------------------------------------


def test_rank_ic_series_perfect_positive_correlation():
    rng = np.random.default_rng(0)
    n_stocks, n_dates = 12, 20
    target = rng.normal(size=(n_stocks, n_dates))
    signals = np.stack([target, -target]).astype(float)
    ic = rank_ic_series(signals, target, min_stocks=5, universe_mask=np.ones(target.shape, dtype=bool))
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
    ic = rank_ic_series(constant[None], target, min_stocks=2, universe_mask=np.ones(target.shape, dtype=bool))
    assert np.isnan(ic).all()
    assert icir_from_series(ic)[0] == 0.0


def test_rank_ic_series_skips_days_below_min_stocks():
    # Three valid pairs on day 0, only one on days 1-2: they must be NaN.
    target = np.array(
        [[0.01, 0.02, np.nan], [0.02, np.nan, np.nan], [0.03, np.nan, 0.02]]
    )
    signal = target.copy()
    ic = rank_ic_series(signal[None], target, min_stocks=2, universe_mask=np.ones(target.shape, dtype=bool))
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
    ic = rank_ic_series(signal[None], target, min_stocks=2, universe_mask=np.ones(target.shape, dtype=bool))
    assert np.allclose(ic[0], 1.0)
    extreme = target.copy()
    extreme[0, :] = 1e9
    ic_extreme = rank_ic_series(extreme[None], target, min_stocks=2, universe_mask=np.ones(target.shape, dtype=bool))
    assert not np.allclose(ic_extreme[0], 1.0)


def test_icir_zero_with_fewer_than_two_observations():
    assert icir_from_series(np.array([np.nan, 0.5])).tolist() == [0.0]
    # Constant positive IC -> mean/std explodes; mean-zero IC -> exact 0.
    icir = icir_from_series(np.array([[1.0, 1.0], [1.0, -1.0]]))
    assert icir[0] > 1e8
    assert icir[1] == 0.0


# --- turnover cost (continuous, rate-proportional) --------------------------


def test_exact_round_trip_cost_formula_above_commission_floor():
    cfg = _cfg()
    model = ExecutionCostModel.from_config(cfg)
    exact = float(model.buy_cost(cfg.initial_capital).total) + float(
        model.sell_cost(cfg.initial_capital).total
    )
    assert exact / cfg.initial_capital == pytest.approx(
        2 * cfg.commission_rate
        + cfg.stamp_tax_rate
        + 2 * cfg.transfer_fee_rate
        + 2 * cfg.slippage_rate,
        rel=1e-12,
    )


def test_annualized_reward_cost_uses_exact_daily_fraction():
    cfg = _cfg()
    target = np.zeros((4, 6))
    signal = np.tile(np.arange(4, dtype=float)[:, None], (1, 6))
    simulation = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=np.ones(signal.shape, dtype=bool)
    )
    free = formula_reward(signal, target, cfg, _reward_cfg(cost_weight=0.0), universe_mask=np.ones(signal.shape, dtype=bool))
    honest = formula_reward(signal, target, cfg, _reward_cfg(cost_weight=1.0), universe_mask=np.ones(signal.shape, dtype=bool))
    assert free - honest == pytest.approx(
        simulation.daily_cost_fractions.mean() * 252, abs=1e-12
    )


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
        signal, raw, ts_codes, dates,
        universe_mask=np.ones(signal.shape, dtype=bool),
    )
    daily, avg_turnover = simulate_basket_daily_returns(signal, target, cfg, universe_mask=np.ones(signal.shape, dtype=bool))
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
    daily, _ = simulate_basket_daily_returns(signal, target, cfg, universe_mask=np.ones(signal.shape, dtype=bool))
    assert daily == pytest.approx(np.full(2, 0.001), abs=1e-12)


# --- v3 reward semantics ----------------------------------------------------


def test_formula_reward_is_icir_minus_continuous_cost():
    cfg = _cfg()
    rc = _reward_cfg()
    rng = np.random.default_rng(3)
    n_stocks, n_dates = 10, 30
    target = rng.normal(size=(n_stocks, n_dates))
    signal = target.copy()
    # Near-perfect IC -> the ICIR term saturates the clip band.
    reward = formula_reward(signal, target, cfg, rc, universe_mask=np.ones(signal.shape, dtype=bool))
    assert reward == pytest.approx(rc.reward_clip_high, abs=1e-12)


def test_reward_penalizes_churn_continuously():
    # The cost drag is exactly cost_weight * annualized turnover cost: with
    # cost_weight 1 vs 0 the reward difference equals the proportional cost
    # (the active-IR term is identical in both, so the difference isolates
    # the cost decomposition).
    cfg = _cfg()
    n_stocks, n_dates = 6, 12
    rng = np.random.default_rng(11)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    static = np.tile(np.arange(n_stocks, dtype=float)[::-1][:, None], (1, n_dates))
    churn = static.copy()
    churn[0, 1::2] = 1.0  # odd days: stock 0 drops out of the top-2 basket
    _, to_static = simulate_basket_daily_returns(static, target, cfg, universe_mask=np.ones(static.shape, dtype=bool))
    _, to_churn = simulate_basket_daily_returns(churn, target, cfg, universe_mask=np.ones(churn.shape, dtype=bool))
    assert to_churn > to_static

    for signal in (static, churn):
        simulation = simulate_basket_daily_returns(signal, target, cfg, universe_mask=np.ones(signal.shape, dtype=bool))
        expected_cost = simulation.daily_cost_fractions.mean() * 252
        # A wide clip band keeps the raw decomposition visible: the active
        # IR of these signals would saturate the default +/-1 band.
        r_free = formula_reward(
            signal, target, cfg,
            _reward_cfg(cost_weight=0.0, reward_clip_low=-10.0, reward_clip_high=10.0),
            universe_mask=np.ones(signal.shape, dtype=bool),
        )
        r_honest = formula_reward(
            signal, target, cfg,
            _reward_cfg(cost_weight=1.0, reward_clip_low=-10.0, reward_clip_high=10.0),
            universe_mask=np.ones(signal.shape, dtype=bool),
        )
        assert r_free - r_honest == pytest.approx(expected_cost, abs=1e-12)


def test_degenerate_input_has_zero_icir_and_no_cost():
    # One date: no IC observations and no daily return -> reward 0.0
    # (the trainer reserves clip_low/bad_reward for invalid formulas).
    cfg = _cfg()
    rc = _reward_cfg()
    reward = formula_reward(np.zeros((3, 1)), np.zeros((3, 1)), cfg, rc, universe_mask=np.ones((3, 1), dtype=bool))
    assert reward == 0.0


def test_formula_reward_clips_to_configured_band():
    cfg = _cfg()
    rc = _reward_cfg(reward_clip_low=-0.25, reward_clip_high=0.5)
    rng = np.random.default_rng(4)
    target = rng.normal(size=(10, 40))
    signal = -target  # ICIR strongly negative
    assert formula_reward(signal, target, cfg, rc, universe_mask=np.ones(signal.shape, dtype=bool)) == -0.25


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
        df, tf = simulate_basket_daily_returns_batch(signals, target, cfg, universe_mask=np.ones(target.shape, dtype=bool))
        for i in range(b):
            ref_f, ref_tf = simulate_basket_daily_returns(signals[i], target, cfg, universe_mask=np.ones(target.shape, dtype=bool))
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
        signals[:, :, start:end], target[:, start:end], cfg,
        universe_mask=np.ones(target[:, start:end].shape, dtype=bool),
    )
    for i in range(2):
        ref_v, _ = simulate_basket_daily_returns(
            signals[i, :, start:end], target[:, start:end], cfg,
            universe_mask=np.ones(target[:, start:end].shape, dtype=bool),
        )
        assert df[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-11)


def test_batch_sim_degenerate_shapes_match_scalar():
    cfg = _cfg()
    # Fewer than two dates: empty daily series.
    df, tf = simulate_basket_daily_returns_batch(
        np.zeros((2, 3, 1)), np.zeros((3, 1)), cfg,
        universe_mask=np.ones((3, 1), dtype=bool),
    )
    assert df.shape == (2, 0)
    assert list(tf) == [0.0, 0.0]
    ref_daily, ref_to = simulate_basket_daily_returns(
        np.zeros((3, 1)), np.zeros((3, 1)), cfg,
        universe_mask=np.ones((3, 1), dtype=bool),
    )
    assert ref_daily.size == 0 and ref_to == 0.0
    # P3 contract section 2: ranks are positive; an invalid zero rank is
    # rejected instead of being interpreted as a silent empty strategy.
    with pytest.raises(ValueError, match="buy_rank must be >= 1"):
        simulate_basket_daily_returns_batch(
            np.zeros((2, 3, 6)), np.zeros((3, 6)), _cfg(top_n=0),
            universe_mask=np.ones((3, 6), dtype=bool),
        )
    # Not enough finite rows on any day: same zero behavior.
    sig = np.full((1, 3, 6), np.nan)
    df, tf = simulate_basket_daily_returns_batch(
        sig, np.zeros((3, 6)), cfg, universe_mask=np.ones((3, 6), dtype=bool)
    )
    assert np.all(df == 0.0)
    assert list(tf) == [0.0]


# --- v5 tradability masks ----------------------------------------------------


def test_blocked_buy_excludes_candidates_from_selection():
    cfg = _cfg(top_n=1, single_weight_cap=1.0, commission_rate=0.0,
               min_commission=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0,
               slippage_rate=0.0)
    n_stocks, n_dates = 4, 5
    target = np.zeros((n_stocks, n_dates))
    # Stock 0 is always the top signal but is blocked at the day-1 execution
    # open; the next-best stock must be selected instead.
    signal = np.tile(np.array([4.0, 3.0, 2.0, 1.0])[:, None], (1, n_dates))
    blocked_buy = np.zeros((n_stocks, n_dates), dtype=bool)
    blocked_buy[0, 1] = True
    # Zero targets and zero costs: returns are 0 everywhere, so the mask's
    # effect shows up purely in the turnover of the forced selection switch.
    _, to_free = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=np.ones(signal.shape, dtype=bool)
    )
    _, to_blocked = simulate_basket_daily_returns(
        signal, target, cfg, blocked_buy=blocked_buy,
        universe_mask=np.ones(signal.shape, dtype=bool),
    )
    # Free path: the single entry into stock 0 (1 unit over 3 complete t+2 periods).
    assert to_free == pytest.approx(1.0 / 3.0, abs=1e-12)
    # Blocked path: stock 0 skipped at the day-1 execution and re-selected
    # afterwards, adding two forced switches (2 extra units over 4 days).
    assert to_blocked == pytest.approx(to_free + 2.0 / 3.0, abs=1e-12)


def test_blocked_sell_force_holds_positions():
    cfg = _cfg(top_n=1, single_weight_cap=1.0, commission_rate=0.0,
               min_commission=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0,
               slippage_rate=0.0)
    n_stocks, n_dates = 4, 5
    target = np.zeros((n_stocks, n_dates))
    # Stock 0 leads every signal column except day 2, where stock 1 leads:
    # the natural rebalance would sell stock 0 at the day-3 execution open.
    # With stock 0 sell-blocked there, the position must be held.
    signal = np.tile(np.array([4.0, 3.0, 2.0, 1.0])[:, None], (1, n_dates))
    signal[0, 2] = 0.5
    blocked_sell = np.zeros((n_stocks, n_dates), dtype=bool)
    blocked_sell[0, 3] = True
    _, to_free = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=np.ones(signal.shape, dtype=bool)
    )
    _, to_held = simulate_basket_daily_returns(
        signal, target, cfg, blocked_sell=blocked_sell,
        universe_mask=np.ones(signal.shape, dtype=bool),
    )
    # The forced hold suppresses the day-3 sell exactly like the engine.
    assert to_held < to_free


def test_blocked_batch_matches_scalar_reference():
    rng = np.random.default_rng(12)
    cfg = _cfg(top_n=3, single_weight_cap=1.0)
    n_stocks, n_dates = 8, 20
    signals, target = _random_batch(rng, 3, n_stocks, n_dates)
    blocked_buy = rng.random((n_stocks, n_dates)) < 0.1
    blocked_sell = rng.random((n_stocks, n_dates)) < 0.1
    df, tf = simulate_basket_daily_returns_batch(
        signals, target, cfg, blocked_buy, blocked_sell,
        universe_mask=np.ones(target.shape, dtype=bool),
    )
    for i in range(signals.shape[0]):
        ref_f, ref_tf = simulate_basket_daily_returns(
            signals[i], target, cfg, blocked_buy, blocked_sell,
            universe_mask=np.ones(target.shape, dtype=bool),
        )
        assert df[i] == pytest.approx(ref_f, rel=1e-9, abs=1e-11)
        assert tf[i] == pytest.approx(ref_tf, rel=1e-9, abs=1e-11)


def test_blocked_mask_shape_validation():
    cfg = _cfg()
    mask = np.ones((3, 6), dtype=bool)
    with pytest.raises(ValueError, match="blocked_buy shape"):
        simulate_basket_daily_returns_batch(
            np.zeros((2, 3, 6)), np.zeros((3, 6)), cfg,
            blocked_buy=np.zeros((3, 5), dtype=bool),
            universe_mask=mask,
        )
    with pytest.raises(ValueError, match="blocked_sell shape"):
        simulate_basket_daily_returns_batch(
            np.zeros((2, 3, 6)), np.zeros((3, 6)), cfg,
            blocked_sell=np.zeros((2, 6), dtype=bool),
            universe_mask=mask,
        )


def test_blocked_basket_matches_backtest_engine_with_limits():
    # The strongest consistency check: the masked basket simulation must
    # reproduce the engine's daily returns on a market where limit moves
    # actually occur, day for day.
    ts_codes, dates, raw, signal, _ = _synthetic_market(n_stocks=6, n_dates=8)
    # Day 3: stock 0 opens at a one-word limit-up (cannot buy).
    raw["high"][0, 3] = raw["open"][0, 3] = raw["pre_close"][0, 3] * 1.10
    raw["low"][0, 3] = raw["open"][0, 3]
    # Day 5: stock 1 opens at a one-word limit-down (cannot sell).
    raw["high"][1, 5] = raw["open"][1, 5] = raw["pre_close"][1, 5] * 0.90
    raw["low"][1, 5] = raw["open"][1, 5]
    target = open_to_open_returns(raw["open"])
    cfg = _cfg(top_n=3, single_weight_cap=1.0)
    result = AshareBacktestEngine(cfg).run(
        signal, raw, ts_codes, dates,
        universe_mask=np.ones(signal.shape, dtype=bool),
    )
    blocked_buy = tradability_blocked_matrix(
        raw["open"], raw["high"], raw["low"], raw["pre_close"], raw["volume"],
        ts_codes, "buy",
    )
    blocked_sell = tradability_blocked_matrix(
        raw["open"], raw["high"], raw["low"], raw["pre_close"], raw["volume"],
        ts_codes, "sell",
    )
    daily, avg_turnover = simulate_basket_daily_returns(
        signal, target, cfg, blocked_buy, blocked_sell,
        universe_mask=np.ones(signal.shape, dtype=bool),
    )
    assert daily == pytest.approx(np.asarray(result.daily_returns), abs=1e-12)
    assert avg_turnover == pytest.approx(float(np.mean(result.turnover)), abs=1e-12)


def test_batched_rewards_match_scalar_rewards():
    rng = np.random.default_rng(7)
    cfg = _cfg(top_n=2, single_weight_cap=1.0)
    reward_cfg = _reward_cfg()
    for _ in range(8):
        b = int(rng.integers(1, 6))
        n_stocks = int(rng.integers(3, 15))
        n_dates = int(rng.integers(10, 60))
        signals, target = _random_batch(rng, b, n_stocks, n_dates)
        val_windows = [(n_dates // 2, n_dates - 2)]
        rewards, val_rewards, icir, val_icir, _objectives = batched_basket_rewards(
            signals, target, cfg, reward_cfg, val_windows,
            universe_mask=np.ones(target.shape, dtype=bool),
        )
        for i in range(b):
            ref_r = formula_reward(
                signals[i], target, cfg, reward_cfg,
                universe_mask=np.ones(target.shape, dtype=bool),
            )
            ref_v = formula_reward(
                signals[i],
                target,
                cfg,
                reward_cfg,
                signal_range=val_windows[0],
                universe_mask=np.ones(target.shape, dtype=bool),
            )
            assert rewards[i] == pytest.approx(ref_r, rel=1e-9, abs=1e-10)
            assert val_rewards[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-10)
        # The exposed raw ICIR must match the scalar decomposition (v12:
        # the robust, effective-n shrunk ICIR).
        from ashare_model.signal_quality import robust_icir

        ref_icir = np.asarray(
            [
                robust_icir(row, reward_cfg.ic_hac_max_lags)
                for row in rank_ic_series(
                    signals[:, :, : n_dates - 2],
                    target[:, : n_dates - 2],
                    reward_cfg.ic_min_stocks,
                    universe_mask=np.ones(
                        target[:, : n_dates - 2].shape, dtype=bool
                    ),
                )
            ]
        )
        assert icir == pytest.approx(ref_icir, rel=1e-9, abs=1e-10)
        ref_val_icir = np.asarray(
            [
                robust_icir(row, reward_cfg.ic_hac_max_lags)
                for row in rank_ic_series(
                    signals[:, :, val_windows[0][0] : val_windows[0][1]],
                    target[:, val_windows[0][0] : val_windows[0][1]],
                    reward_cfg.ic_min_stocks,
                    universe_mask=np.ones(
                        target[:, val_windows[0][0] : val_windows[0][1]].shape,
                        dtype=bool,
                    ),
                )
            ]
        )
        assert val_icir == pytest.approx(ref_val_icir, rel=1e-9, abs=1e-10)


def test_val_reward_is_median_over_windows():
    rng = np.random.default_rng(9)
    cfg = _cfg(top_n=2, single_weight_cap=1.0)
    reward_cfg = _reward_cfg()
    n_stocks, n_dates = 8, 30
    target = rng.normal(0.0, 0.01, size=(n_stocks, n_dates))
    signals = rng.normal(size=(2, n_stocks, n_dates))
    # Aligned in window 1, inverted in window 2, noise in window 3.
    aligned = np.stack([target, -target, rng.normal(size=target.shape)])
    windows = [(0, 9), (9, 18), (18, 28)]
    rewards, val_rewards, _, val_icir, _objectives = batched_basket_rewards(
        aligned, target, cfg, reward_cfg, windows,
        universe_mask=np.ones(target.shape, dtype=bool),
    )
    per_window = []
    for start, end in windows:
        per_window.append(
            np.asarray(
                [
                    formula_reward(
                        aligned[i],
                        target,
                        cfg,
                        reward_cfg,
                        signal_range=(start, end),
                        universe_mask=np.ones(target.shape, dtype=bool),
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
        formula_reward(
            aligned[0], target, cfg, reward_cfg,
            universe_mask=np.ones(target.shape, dtype=bool),
        ),
        abs=1e-12,
    )


def test_batched_rewards_without_val_windows_returns_none():
    rng = np.random.default_rng(10)
    signals, target = _random_batch(rng, 3, 8, 25)
    rewards, val_rewards, icir, val_icir, _objectives = batched_basket_rewards(
        signals, target, _cfg(), _reward_cfg(),
        universe_mask=np.ones(target.shape, dtype=bool),
    )
    assert rewards.shape == (3,)
    assert val_rewards is None
    assert icir.shape == (3,)
    assert val_icir is None


def test_batched_rewards_are_deterministic():
    rng = np.random.default_rng(8)
    signals, target = _random_batch(rng, 4, 8, 25)
    cfg = _cfg()
    reward_cfg = _reward_cfg()
    windows = [(0, 8), (8, 16)]
    first = batched_basket_rewards(
        signals, target, cfg, reward_cfg, windows,
        universe_mask=np.ones(target.shape, dtype=bool),
    )
    second = batched_basket_rewards(
        signals, target, cfg, reward_cfg, windows,
        universe_mask=np.ones(target.shape, dtype=bool),
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert np.array_equal(first[3], second[3])
    # Rewards stay inside the configured clip band.
    assert np.all(first[0] >= reward_cfg.reward_clip_low)
    assert np.all(first[0] <= reward_cfg.reward_clip_high)
    assert np.all(first[1] >= reward_cfg.reward_clip_low)
    assert np.all(first[1] <= reward_cfg.reward_clip_high)


# --- signal direction -------------------------------------------------------


def test_signal_direction_flips_negative_ic_signals():
    rng = np.random.default_rng(21)
    target = rng.normal(size=(20, 40))
    mask = np.ones(target.shape, dtype=bool)
    assert signal_direction(target, target, min_stocks=5, universe_mask=mask) == 1
    assert signal_direction(-target, target, min_stocks=5, universe_mask=mask) == -1


def test_signal_direction_defaults_to_positive_without_observations():
    rng = np.random.default_rng(22)
    # Constant signal: undefined correlation -> no finite IC -> neutral +1.
    assert (
        signal_direction(
            np.ones((20, 40)), rng.normal(size=(20, 40)),
            universe_mask=np.ones((20, 40), dtype=bool),
        )
        == 1
    )
    # Below the minimum cross-section: every date is skipped -> neutral +1.
    assert (
        signal_direction(
            np.ones((3, 10)), rng.normal(size=(3, 10)), min_stocks=5,
            universe_mask=np.ones((3, 10), dtype=bool),
        )
        == 1
    )


# --- v7 signal-date universe eligibility -------------------------------------


def _universe_mask(n_stocks, n_dates, future_row=-1, join_day=None):
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    if join_day is not None:
        mask[future_row, :join_day] = False
    return mask


def test_rank_ic_series_ignores_ineligible_extreme_values():
    rng = np.random.default_rng(41)
    n_stocks, n_dates = 6, 20
    join_day = 8
    target = rng.normal(size=(n_stocks, n_dates))
    signal = target.copy()  # perfect alignment over every row
    extreme = signal.copy()
    extreme[-1, :join_day] = 1e9  # future member's extreme pre-join values
    mask = _universe_mask(n_stocks, n_dates, join_day=join_day)
    ic_masked = rank_ic_series(extreme[None], target, min_stocks=3, universe_mask=mask)
    # The unmasked reference is now an all-eligible mask: identical to the
    # PIT mask from the join day on, polluted by the extreme values before.
    ic_open = rank_ic_series(
        extreme[None], target, min_stocks=3,
        universe_mask=np.ones(target.shape, dtype=bool),
    )
    # Pre-join the masked IC keeps the perfect eligible-only correlation...
    assert np.allclose(ic_masked[0, :join_day], 1.0)
    # ...while the unmasked path is polluted by the extreme finite values.
    assert not np.allclose(ic_open[0, :join_day], 1.0)
    # From the join day on the mask includes the future member: identical.
    assert np.allclose(ic_masked[0, join_day:], ic_open[0, join_day:])


def test_rank_ic_series_rejects_mask_shape_mismatch():
    with pytest.raises(ValueError, match="universe_mask shape"):
        rank_ic_series(
            np.zeros((2, 3, 4)),
            np.zeros((3, 4)),
            universe_mask=np.ones((3, 3), dtype=bool),
        )


def test_signal_direction_ignores_ineligible_extreme_values():
    # Deterministic three-stock cross-section: rows 0-1 are perfectly
    # anti-aligned, the (ineligible) future member perfectly aligned.  The
    # masked direction is decided on the eligible rows only (-1); the
    # unmasked direction is flipped by the future member's alignment (+1).
    target = np.tile(np.array([0.01, 0.02, 0.03])[:, None], (1, 10))
    signal = -target.copy()
    signal[2] = target[2]
    mask = _universe_mask(3, 10, join_day=10)  # row 2 never eligible
    assert signal_direction(signal, target, min_stocks=2, universe_mask=mask) == -1
    assert (
        signal_direction(
            signal, target, min_stocks=2,
            universe_mask=np.ones(target.shape, dtype=bool),
        )
        == 1
    )


def test_basket_excludes_ineligible_and_switches_on_join_day():
    cfg = _cfg(top_n=1, single_weight_cap=1.0, commission_rate=0.0,
               min_commission=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0,
               slippage_rate=0.0)
    n_stocks, n_dates = 3, 8
    join_day = 3
    # The future member always has the top signal but is ineligible before
    # the join day; the eligible runner-up is selected until then.
    signal = np.tile(np.array([2.0, 1.0, 9.0])[:, None], (1, n_dates))
    target = np.zeros((n_stocks, n_dates))
    mask = _universe_mask(n_stocks, n_dates, join_day=join_day)
    masked = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    # The unmasked reference is an all-eligible mask: it buys the future
    # member on day 0 because every cell is eligible.
    free = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=np.ones(signal.shape, dtype=bool)
    )
    # Unmasked: the future member is bought on day 0 and held throughout.
    assert free.turnover[0] == pytest.approx(1.0)
    assert np.allclose(free.turnover[1:], 0.0)
    # Masked: the runner-up is bought on day 0; the join day swaps the
    # basket into the future member (sell + buy) and nothing churns before.
    assert masked.turnover[0] == pytest.approx(1.0)
    assert np.allclose(masked.turnover[1:join_day], 0.0)
    assert masked.turnover[join_day] == pytest.approx(2.0)
    assert np.allclose(masked.turnover[join_day + 1 :], 0.0)


def test_basket_pre_join_extreme_values_do_not_move_returns_or_costs():
    rng = np.random.default_rng(45)
    cfg = _cfg(top_n=1, single_weight_cap=1.0)
    n_stocks, n_dates = 3, 10
    join_day = 4
    target = rng.normal(0.0, 0.01, size=(n_stocks, n_dates))
    base = np.tile(np.array([2.0, 1.0, 9.0])[:, None], (1, n_dates))
    extreme = base.copy()
    extreme[2, :join_day] = 1e9
    mask = _universe_mask(n_stocks, n_dates, join_day=join_day)
    mild_sim = simulate_basket_daily_returns(base, target, cfg, universe_mask=mask)
    extreme_sim = simulate_basket_daily_returns(extreme, target, cfg, universe_mask=mask)
    # Selection, daily returns and daily costs are bit-identical: the
    # future member's extreme finite pre-join values never reach the basket.
    assert np.array_equal(extreme_sim.daily_gross_returns, mild_sim.daily_gross_returns)
    assert np.array_equal(extreme_sim.daily_net_returns, mild_sim.daily_net_returns)
    assert np.array_equal(
        extreme_sim.daily_cost_fractions, mild_sim.daily_cost_fractions
    )
    assert np.array_equal(extreme_sim.turnover, mild_sim.turnover)
    # Without the mask the extreme values change the outcome (sanity: the
    # perturbation is real and the mask is what neutralizes it).
    open_sim = simulate_basket_daily_returns(
        extreme, target, cfg, universe_mask=np.ones(extreme.shape, dtype=bool)
    )
    assert not np.array_equal(open_sim.daily_net_returns, extreme_sim.daily_net_returns)


def test_basket_exit_is_sold_through_cost_path_or_force_held():
    cfg = _cfg(top_n=1, single_weight_cap=1.0)
    n_stocks, n_dates = 3, 8
    exit_day = 3
    # Stock 0 is the top signal every day but exits the member pool on
    # ``exit_day``: with the T1-04 entry-day alignment its last buyable
    # signal day is ``exit_day - 1`` (the entry at ``exit_day`` is already
    # outside the universe), so the sell happens at that execution — the
    # ordinary sell path (paying the sell leg), never silently.
    signal = np.tile(np.array([9.0, 2.0, 1.0])[:, None], (1, n_dates))
    target = np.zeros((n_stocks, n_dates))
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[0, exit_day:] = False
    sim = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    # Entry on day 0 pays costs; the exit-day execution pays the sell leg
    # on top of the buy leg (a silently vanished position would pay none).
    assert sim.daily_cost_fractions[0] > 0
    assert sim.daily_cost_fractions[1] == pytest.approx(0.0, abs=1e-15)
    assert sim.daily_cost_fractions[exit_day - 1] > sim.daily_cost_fractions[1]
    assert sim.turnover[exit_day - 1] == pytest.approx(2.0)
    # When the exit execution day is sell-blocked the position is
    # force-held: the sell is deferred and consumes the budget, so nothing
    # else can be bought either — zero turnover, zero cost (T1-02
    # no-renormalization: the book is [1.0, 0, 0], never 0.5/0.5).
    blocked_sell = np.zeros((n_stocks, n_dates), dtype=bool)
    blocked_sell[0, exit_day] = True  # the exit execution day itself
    held = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask, blocked_sell=blocked_sell
    )
    assert held.turnover[exit_day - 1] == pytest.approx(0.0, abs=1e-15)
    assert held.daily_cost_fractions[exit_day - 1] == pytest.approx(0.0, abs=1e-15)
    assert (
        held.daily_cost_fractions[exit_day - 1]
        < sim.daily_cost_fractions[exit_day - 1]
    )


def test_basket_underfilled_keeps_cash_like_backtest_engine():
    cfg = _cfg(top_n=3, single_weight_cap=1.0, commission_rate=0.0,
               min_commission=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0,
               slippage_rate=0.0)
    n_stocks, n_dates = 4, 6
    # Only two eligible stocks: fewer than top_n=3.  Both are selected at
    # min(1/3, 1.0) each and the remainder stays in cash — the weights are
    # never renormalized upward (T1-02).
    signal = np.tile(np.array([4.0, 3.0, 2.0, 1.0])[:, None], (1, n_dates))
    target = np.zeros((n_stocks, n_dates))
    target[0, 0] = 0.01
    target[1, 0] = 0.02
    mask = _universe_mask(n_stocks, n_dates, join_day=n_dates)
    mask[2:] = False
    sim = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    assert sim.daily_net_returns[0] == pytest.approx(
        (1.0 / 3.0) * 0.01 + (1.0 / 3.0) * 0.02, abs=1e-12
    )
    assert sim.turnover[0] == pytest.approx(2.0 / 3.0, abs=1e-12)
    # Engine parity: same universe mask, same per-day turnover path.
    ts_codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = [f"202401{i:02d}" for i in range(n_dates)]
    open_ = np.full((n_stocks, n_dates), 10.0)
    raw = {
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": open_.copy(),
        "volume": np.full_like(open_, 1_000_000.0),
    }
    engine = AshareBacktestEngine(cfg).run(
        signal, raw, ts_codes, dates, universe_mask=mask
    )
    assert sim.turnover == pytest.approx(np.asarray(engine.turnover), abs=1e-12)


def test_masked_basket_matches_backtest_engine_with_universe():
    # Day-by-day parity with the engine when both consume the same PIT
    # universe mask: a permanently ineligible stock with extreme finite
    # signals is excluded by both paths.
    ts_codes, dates, raw, signal, _ = _synthetic_market(n_stocks=6, n_dates=8)
    target = open_to_open_returns(raw["open"])
    cfg = _cfg(top_n=3, single_weight_cap=1.0)
    mask = np.ones((6, 8), dtype=bool)
    mask[5] = False
    signal[5] = 1e9
    result = AshareBacktestEngine(cfg).run(
        signal, raw, ts_codes, dates, universe_mask=mask
    )
    daily, avg_turnover = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask
    )
    assert daily == pytest.approx(np.asarray(result.daily_returns), abs=1e-12)
    assert avg_turnover == pytest.approx(float(np.mean(result.turnover)), abs=1e-12)


def test_masked_scalar_and_batched_rewards_agree():
    rng = np.random.default_rng(43)
    cfg = _cfg(top_n=2, single_weight_cap=1.0)
    reward_cfg = _reward_cfg()
    n_stocks, n_dates = 7, 25
    join_day = 10
    signals, target = _random_batch(rng, 3, n_stocks, n_dates, nan_frac=0.05)
    mask = _universe_mask(n_stocks, n_dates, join_day=join_day)
    windows = [(8, 16)]
    rewards, val_rewards, icir, val_icir, _objectives = batched_basket_rewards(
        signals, target, cfg, reward_cfg, windows, universe_mask=mask
    )
    df, tf = simulate_basket_daily_returns_batch(
        signals, target, cfg, universe_mask=mask
    )
    for i in range(3):
        ref_r = formula_reward(signals[i], target, cfg, reward_cfg, universe_mask=mask)
        ref_v = formula_reward(
            signals[i],
            target,
            cfg,
            reward_cfg,
            signal_range=windows[0],
            universe_mask=mask,
        )
        assert rewards[i] == pytest.approx(ref_r, rel=1e-9, abs=1e-10)
        assert val_rewards[i] == pytest.approx(ref_v, rel=1e-9, abs=1e-10)
        from ashare_model.signal_quality import robust_icir

        ref_icir = robust_icir(
            rank_ic_series(
                signals[i][None, :, : n_dates - 2],
                target[:, : n_dates - 2],
                reward_cfg.ic_min_stocks,
                universe_mask=mask[:, : n_dates - 2],
            )[0],
            reward_cfg.ic_hac_max_lags,
        )
        assert icir[i] == pytest.approx(ref_icir, rel=1e-9, abs=1e-10)
        ref_df, ref_tf = simulate_basket_daily_returns(
            signals[i], target, cfg, universe_mask=mask
        )
        assert df[i] == pytest.approx(ref_df, rel=1e-9, abs=1e-11)
        assert tf[i] == pytest.approx(ref_tf, rel=1e-9, abs=1e-11)


def test_mask_slices_align_exactly_without_off_by_one():
    # The eligibility flip must land exactly on the join column, both on
    # the full range and through validation windows: rows 0-2 are perfectly
    # anti-aligned, the future member perfectly aligned.
    n_stocks, n_dates = 4, 20
    join_day = 9
    target = np.tile(
        np.array([0.01, 0.02, 0.03, 0.04])[:, None], (1, n_dates)
    )
    signal = -target.copy()
    signal[3] = target[3] * 10.0
    mask = _universe_mask(n_stocks, n_dates, join_day=join_day)
    ic = rank_ic_series(signal[None], target, min_stocks=3, universe_mask=mask)[0]
    # Pre-join: rank correlation over the three eligible rows is exactly
    # -1; from the join day the aligned row enters and it becomes +0.2.
    # The flip at column ``join_day`` itself proves no off-by-one.
    assert np.allclose(ic[:join_day], -1.0)
    assert np.allclose(ic[join_day:], 0.2)
    assert np.isnan(ic).sum() == 0

    # The same slice alignment holds through the validation-window path.
    cfg = _cfg(top_n=2, single_weight_cap=1.0)
    reward_cfg = _reward_cfg()
    window = (join_day - 3, join_day + 3)
    rewards, val_rewards, _, val_icir, _objectives = batched_basket_rewards(
        signal[None], target, cfg, reward_cfg, [window], universe_mask=mask
    )
    ref_window = formula_reward(
        signal, target, cfg, reward_cfg, signal_range=window, universe_mask=mask
    )
    assert val_rewards[0] == pytest.approx(ref_window, rel=1e-9, abs=1e-10)
    from ashare_model.signal_quality import robust_icir

    ref_val_icir = robust_icir(
        rank_ic_series(
            signal[None, :, window[0] : window[1]],
            target[:, window[0] : window[1]],
            reward_cfg.ic_min_stocks,
            universe_mask=mask[:, window[0] : window[1]],
        )[0],
        reward_cfg.ic_hac_max_lags,
    )
    assert val_icir[0] == pytest.approx(ref_val_icir, rel=1e-9, abs=1e-10)
    # The window itself straddles the flip: -1 before, +0.2 at/after.
    window_ic = rank_ic_series(signal[None], target, min_stocks=3, universe_mask=mask)[0]
    assert window_ic[window[0]] == pytest.approx(-1.0)
    assert window_ic[join_day] == pytest.approx(0.2)
    assert rewards[0] == pytest.approx(
        formula_reward(signal, target, cfg, reward_cfg, universe_mask=mask),
        rel=1e-9,
        abs=1e-10,
    )


def test_reward_paths_reject_mask_shape_mismatch():
    cfg = _cfg()
    reward_cfg = _reward_cfg()
    with pytest.raises(ValueError, match="universe_mask shape"):
        formula_reward(
            np.zeros((3, 6)), np.zeros((3, 6)), cfg, reward_cfg,
            universe_mask=np.ones((3, 5), dtype=bool),
        )
    with pytest.raises(ValueError, match="universe_mask shape"):
        batched_basket_rewards(
            np.zeros((2, 3, 6)), np.zeros((3, 6)), cfg, reward_cfg,
            universe_mask=np.ones((2, 6), dtype=bool),
        )
    with pytest.raises(ValueError, match="universe_mask shape"):
        simulate_basket_daily_returns(
            np.zeros((3, 6)), np.zeros((3, 6)), cfg,
            universe_mask=np.ones((4, 6), dtype=bool),
        )
