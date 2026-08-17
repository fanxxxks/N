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

Version history
---------------
* v2: the RL loop scores formulas through the vectorized batch path
  (:func:`batched_basket_rewards`).  Scoring semantics are unchanged except
  that, when several stocks tie on the exact same signal value, the
  top-n selection may pick a different (equally valued) tie set than the
  scalar path did; with no ties the two paths produce the same daily
  returns and turnover up to floating-point summation order.
"""

from __future__ import annotations

import math

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig

REWARD_VERSION = "2"

_ANNUALIZATION = 252


def _trading_cost_columns(
    buy_weights: np.ndarray,
    sell_weights: np.ndarray,
    cfg: BacktestConfig,
) -> np.ndarray:
    """Trading cost of one rebalance, summed over the last axis.

    One code path for the scalar per-position cost and the batched per-day
    cost inside the RL loop: commission is floored at ``min_commission``
    yuan per trade (scaled by ``initial_capital``); sells additionally pay
    stamp tax; both sides pay transfer fee and slippage.
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
    return buy_cost.sum(axis=-1) + sell_cost.sum(axis=-1)


def trading_cost_fraction(
    buy_weights: np.ndarray,
    sell_weights: np.ndarray,
    cfg: BacktestConfig,
) -> float:
    """Trading cost of one rebalance day, as a fraction of capital."""

    return float(_trading_cost_columns(buy_weights, sell_weights, cfg))


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


def _daily_to_rewards(
    daily: np.ndarray,
    avg_turnover: np.ndarray,
    reward_cfg: RewardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sortino-based reward for one or many daily-return rows.

    Rows are independent baskets; every step mirrors the scalar
    :func:`fast_basket_reward` scoring (turnover penalty, losing-basket
    collapse to ``bad_reward``, constant-zero collapse, clipping).  A
    zero-length row scores ``reward_clip_low``.
    """

    daily = np.atleast_2d(np.asarray(daily, dtype=np.float64))
    b, n = daily.shape
    avg_to = np.broadcast_to(
        np.asarray(avg_turnover, dtype=np.float64).reshape(-1), (b,)
    )
    if n == 0:
        return (
            np.full(b, float(reward_cfg.reward_clip_low)),
            np.zeros(b),
        )

    means = daily.mean(axis=1)
    ann_mean = means * _ANNUALIZATION
    neg = daily < 0
    n_neg = neg.sum(axis=1)
    neg_sum = (daily * neg).sum(axis=1)
    neg_mean = np.where(n_neg > 0, neg_sum / np.maximum(n_neg, 1), 0.0)
    dev = np.where(neg, daily - neg_mean[:, None], 0.0)
    neg_var = np.where(
        n_neg > 1,
        (dev**2).sum(axis=1) / np.maximum(n_neg - 1, 1),
        0.0,
    )
    down_std_neg = np.sqrt(np.clip(neg_var, 0.0, None)) * math.sqrt(_ANNUALIZATION)
    # Full-sample variance with a safe denominator: rows with a single
    # observation fall back to zero variance without a divide-by-zero
    # warning (the scalar path guards ``size > 1`` the same way).
    mu = daily.mean(axis=1, keepdims=True)
    full_var = np.where(
        n > 1, ((daily - mu) ** 2).sum(axis=1) / np.maximum(n - 1, 1), 0.0
    )
    down_std_fb = (
        np.sqrt(np.clip(full_var, 0.0, None)) * math.sqrt(_ANNUALIZATION) + 1e-6
    )
    down_std = np.where(
        n_neg >= reward_cfg.downside_min_obs, down_std_neg, down_std_fb
    )
    rewards = ann_mean / (down_std + 1e-9)
    rewards = np.where(
        avg_to > reward_cfg.turnover_threshold,
        rewards - reward_cfg.turnover_penalty,
        rewards,
    )
    rewards = np.where(means < 0, float(reward_cfg.bad_reward), rewards)
    rewards = np.where(
        (daily.std(axis=1) < 1e-6) & (means <= 0),
        float(reward_cfg.bad_reward),
        rewards,
    )
    return (
        np.clip(rewards, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high),
        means,
    )


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

    daily_full, avg_to_full, _, _ = simulate_basket_daily_returns_batch(
        np.asarray(signal, dtype=np.float64)[None],
        np.asarray(target_ret, dtype=np.float64),
        cfg,
    )
    return daily_full[0], float(avg_to_full[0])


def simulate_basket_daily_returns_batch(
    signals: np.ndarray,
    target_ret: np.ndarray,
    cfg: BacktestConfig,
    val_start: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Batched basket simulation: one pass over dates, ``B`` formulas at once.

    ``signals`` is ``[B, stocks, dates]``; the result is
    ``(daily_full [B, dates-1], avg_turnover_full [B],
    daily_val [B, dates-val_start-1] | None, avg_turnover_val [B] | None)``.
    With ``val_start`` the validation basket restarts from zero weights at
    that column, exactly like the scalar implementation applied to
    ``signal[:, val_start:]``.

    Semantics are the scalar loop's, verified by the equivalence tests:
    per date the top-n finite signals are selected (a tie between exactly
    equal values may resolve to a different equally valued set than the
    scalar path), weights renormalize to ``1/top_n`` each, invalid days
    earn zero and keep the previous basket, and both baskets share one
    trading-cost model.
    """

    signals = np.asarray(signals, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    b, n_stocks, n_dates = signals.shape
    daily_full = np.zeros((b, max(n_dates - 1, 0)))
    avg_to_full = np.zeros(b)
    daily_val: np.ndarray | None = None
    avg_to_val: np.ndarray | None = None
    if val_start is not None:
        daily_val = np.zeros((b, max(n_dates - val_start - 1, 0)))
        avg_to_val = np.zeros(b)
    if n_dates < 2 or n_stocks == 0:
        return daily_full, avg_to_full, daily_val, avg_to_val

    top_n = min(int(cfg.top_n), n_stocks)
    if top_n <= 0:
        # Every day is skipped: all-zero daily series (scoring collapses
        # them to ``bad_reward`` exactly like the scalar path).
        return daily_full, avg_to_full, daily_val, avg_to_val

    weight = min(1.0 / top_n, float(cfg.single_weight_cap))
    min_fee = cfg.min_commission / cfg.initial_capital
    b_idx = np.arange(b)[:, None]
    prev = np.zeros((b, n_stocks))
    prev_val = np.zeros((b, n_stocks))
    turn_acc = np.zeros(b)
    turn_val_acc = np.zeros(b)
    # Non-finite target cells contribute zero; cleaned once outside the loop.
    target_clean = np.where(np.isfinite(target_ret), target_ret, 0.0)

    for t in range(n_dates - 1):
        column = signals[:, :, t]  # [B, stocks]
        finite = np.isfinite(column)
        day_ok = finite.sum(axis=1) >= top_n
        fill = np.where(finite, column, -np.inf)
        top_idx = np.argpartition(fill, kth=-top_n, axis=1)[:, -top_n:]  # [B, top_n]
        fresh = np.zeros((b, n_stocks))
        fresh[b_idx, top_idx] = weight
        totals = fresh.sum(axis=1, keepdims=True)
        fresh = np.where(totals > 0, fresh / totals, 0.0)

        target = np.where(day_ok[:, None], fresh, prev)
        buy = np.maximum(target - prev, 0.0)
        sell = np.maximum(prev - target, 0.0)
        cost = _trading_cost_columns(buy, sell, cfg)
        rets = target_clean[:, t][None, :]
        gross = (target * rets).sum(axis=1)
        daily_full[:, t] = np.where(day_ok, gross - cost, 0.0)
        turn_acc += np.where(
            day_ok, np.abs(target - prev).sum(axis=1), 0.0
        )
        prev = target

        if val_start is not None and t >= val_start:
            target_val = np.where(day_ok[:, None], fresh, prev_val)
            buy_val = np.maximum(target_val - prev_val, 0.0)
            sell_val = np.maximum(prev_val - target_val, 0.0)
            cost_val = _trading_cost_columns(buy_val, sell_val, cfg)
            gross_val = (target_val * rets).sum(axis=1)
            daily_val[:, t - val_start] = np.where(day_ok, gross_val - cost_val, 0.0)
            turn_val_acc += np.where(
                day_ok, np.abs(target_val - prev_val).sum(axis=1), 0.0
            )
            prev_val = target_val

    avg_to_full = turn_acc / max(n_dates - 1, 1)
    if val_start is not None:
        avg_to_val = turn_val_acc / max(n_dates - val_start - 1, 1)
    return daily_full, avg_to_full, daily_val, avg_to_val


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
    reward, mean_daily = _daily_to_rewards(daily, [avg_turnover], reward_cfg)
    return float(reward[0]), float(mean_daily[0])


def batched_basket_rewards(
    signals: np.ndarray,
    target_ret: np.ndarray,
    bt_cfg: BacktestConfig,
    reward_cfg: RewardConfig,
    val_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-window and validation rewards for a batch of signals, one pass.

    ``signals`` is ``[B, stocks, dates]`` and the training window ends at
    the last column; ``val_start`` anchors the out-of-sample tail used for
    best-formula selection.  Returns ``(rewards, val_rewards)`` of length
    ``B``; both are clipped exactly like the scalar path.
    """

    daily_full, avg_to_full, daily_val, avg_to_val = (
        simulate_basket_daily_returns_batch(signals, target_ret, bt_cfg, val_start)
    )
    rewards, _ = _daily_to_rewards(daily_full, avg_to_full, reward_cfg)
    val_rewards, _ = _daily_to_rewards(daily_val, avg_to_val, reward_cfg)
    return rewards, val_rewards
