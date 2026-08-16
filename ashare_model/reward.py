"""Unified reward scoring for formula training.

This module is the single source of truth for three pieces of logic that
used to drift in three separate copies:

* the trading-cost model (identical per-position semantics to the backtest
  engine, including the per-trade minimum-commission floor),
* the annualized Sortino ratio,
* the cheap long-only top-n basket simulation used inside the RL loop.

:data:`REWARD_VERSION` is bumped on every semantic change to this module's
scoring behavior and is recorded in training artifacts and experiment
archives, so ``best_reward`` values from different reward generations can
never be compared silently.

Known remaining gaps versus the full backtest engine (planned for a later
phase): the basket simulation does not apply the blocked mask (limit-up /
suspension) and does not hold limit-down positions instead of selling them.
"""

from __future__ import annotations

import math

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig

REWARD_VERSION = "1"

_ANNUALIZATION = 252


def trading_cost_fraction(
    buy_weights: np.ndarray,
    sell_weights: np.ndarray,
    cfg: BacktestConfig,
) -> float:
    """Trading cost of one rebalance day, as a fraction of capital.

    Commission is floored at ``min_commission`` yuan per trade (scaled by
    ``initial_capital``); sells additionally pay stamp tax; both sides pay
    transfer fee and slippage.  Vectorized over positions.
    """

    buy = np.asarray(buy_weights, dtype=np.float64)
    sell = np.asarray(sell_weights, dtype=np.float64)
    min_fee_fraction = cfg.min_commission / cfg.initial_capital
    buy_cost = np.where(
        buy > 0,
        np.maximum(min_fee_fraction, cfg.commission_rate * buy)
        + (cfg.transfer_fee_rate + cfg.slippage_rate) * buy,
        0.0,
    )
    sell_cost = np.where(
        sell > 0,
        np.maximum(min_fee_fraction, cfg.commission_rate * sell)
        + (cfg.stamp_tax_rate + cfg.transfer_fee_rate + cfg.slippage_rate) * sell,
        0.0,
    )
    return float(buy_cost.sum() + sell_cost.sum())


def sortino_ratio(
    daily_returns: np.ndarray,
    min_downside_obs: int = 3,
) -> float:
    """Annualized Sortino ratio of a daily net-return series.

    With fewer than ``min_downside_obs`` negative days the downside
    deviation is under-identified, so the full-sample volatility
    substitutes.  Single code path for the fast reward and the backtest
    metrics.
    """

    daily = np.asarray(daily_returns, dtype=np.float64)
    if daily.size == 0:
        return 0.0
    ann_mean = float(daily.mean()) * _ANNUALIZATION
    downside = daily[daily < 0]
    if downside.size >= min_downside_obs:
        downside_std = float(downside.std(ddof=1)) * math.sqrt(_ANNUALIZATION)
    else:
        daily_std = float(daily.std(ddof=1)) if daily.size > 1 else 0.0
        downside_std = daily_std * math.sqrt(_ANNUALIZATION) + 1e-6
    return ann_mean / (downside_std + 1e-9)


def simulate_basket_daily_returns(
    signal: np.ndarray,
    target_ret: np.ndarray,
    cfg: BacktestConfig,
) -> tuple[np.ndarray, float]:
    """Daily net returns of the long-only top-n basket, plus its average
    turnover.

    Mirrors the backtest engine's target-weight construction (position
    weight ``min(1/top_n, single_weight_cap)``, renormalized) and charges
    the same trading-cost model.  Non-finite target cells contribute a zero
    return; days with fewer than ``top_n`` finite signals contribute a zero
    return and keep the previous basket (as in the historical fast reward).
    """

    signal = np.asarray(signal, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    n_stocks, n_dates = signal.shape
    if n_dates < 2 or n_stocks == 0:
        return np.zeros(0, dtype=np.float64), 0.0

    top_n = min(int(cfg.top_n), n_stocks)
    daily = np.zeros(n_dates - 1, dtype=np.float64)
    turnover_total = 0.0
    prev = np.zeros(n_stocks, dtype=np.float64)

    for t in range(n_dates - 1):
        column = signal[:, t]
        valid_idx = np.flatnonzero(np.isfinite(column))
        if top_n <= 0 or valid_idx.size < top_n:
            continue
        top_idx = valid_idx[np.argpartition(column[valid_idx], -top_n)[-top_n:]]
        top_idx = top_idx[np.argsort(column[top_idx])[::-1]]

        target = np.zeros(n_stocks, dtype=np.float64)
        weight = 1.0 / top_n
        target[top_idx] = min(weight, float(cfg.single_weight_cap))
        total = target.sum()
        if total > 0:
            target /= total

        buy = np.maximum(target - prev, 0.0)
        sell = np.maximum(prev - target, 0.0)
        turnover_total += float(np.abs(target - prev).sum())

        rets = np.where(np.isfinite(target_ret[:, t]), target_ret[:, t], 0.0)
        gross = float(np.dot(target, rets))
        daily[t] = gross - trading_cost_fraction(buy, sell, cfg)
        prev = target

    avg_turnover = turnover_total / max(n_dates - 1, 1)
    return daily, avg_turnover


def fast_basket_reward(
    signal: np.ndarray,
    target_ret: np.ndarray,
    bt_cfg: BacktestConfig,
    reward_cfg: RewardConfig,
) -> tuple[float, float]:
    """Cheap long-only Sortino reward used inside the RL loop.

    Simulates the same top-n basket and the same trading-cost model as the
    backtest engine, then scores the daily net returns with the shared
    Sortino ratio plus a soft turnover penalty.  Losing baskets collapse to
    ``bad_reward`` so they cannot hide behind a low downside deviation.
    """

    daily, avg_turnover = simulate_basket_daily_returns(signal, target_ret, bt_cfg)
    if daily.size == 0:
        return float(reward_cfg.reward_clip_low), 0.0

    mean_daily = float(daily.mean())
    reward = sortino_ratio(daily, min_downside_obs=reward_cfg.downside_min_obs)
    if avg_turnover > reward_cfg.turnover_threshold:
        reward -= reward_cfg.turnover_penalty
    if mean_daily < 0:
        reward = reward_cfg.bad_reward
    if float(np.std(daily)) < 1e-6 and mean_daily <= 0:
        reward = reward_cfg.bad_reward
    return (
        float(
            np.clip(
                reward,
                reward_cfg.reward_clip_low,
                reward_cfg.reward_clip_high,
            )
        ),
        mean_daily,
    )
