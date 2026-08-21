"""Unified reward scoring for formula training.

This module is the single source of truth for three pieces of logic that
used to drift in three separate copies:

* the trading-cost model (identical per-position semantics to the backtest
  engine, including the per-trade minimum-commission floor),
* the annualized Sortino ratio (used by the backtest engine metrics),
* the cheap long-only top-n basket simulation used inside the RL loop.

:data:`REWARD_VERSION` is bumped on every semantic change to this module's
scoring behavior and is recorded in training artifacts and experiment
archives, so ``best_reward`` values from different reward generations can
never be compared silently.

Version history
---------------
* v5: two alignments to the backtest engine's deployment semantics.
  (a) The VM now returns every formula signal cross-sectionally z-scored
  per date (terminal standardization), so stacked arithmetic no longer
  drifts the scale that GATE/JUMP thresholds compare against; the rank-IC
  term and the top-n selection are monotone-invariant, so the change is a
  semantic-stability one.  (b) The basket simulation consumes the engine's
  tradability masks: buy-blocked stocks (suspended / one-word limit-up
  opens) can no longer be selected, and sell-blocked positions (suspended /
  one-word limit-down opens) are force-held and the remaining weights
  renormalized, exactly like the backtest engine, so the training reward
  can no longer buy stocks the engine could not have bought.
* v4: the scoring quantity is unchanged (rank-ICIR minus the continuous
  turnover cost), but the batched path now also returns the raw full-window
  ICIR and the median validation-window ICIR so the trainer can gate
  artifact saving on signal quality (``min_val_icir``), not only on the
  cost-adjusted reward.  The cost model keeps v3 semantics: in the
  20260820 screening the v3 reward's ordering of the single-factor
  baselines (ILLIQ_20 > RSQ_60 > ROE ~ TURNOVER ~ MOMENTUM_20 >
  REVERSAL_5) reproduced their out-of-sample Sharpe ordering, so the
  cost_weight=1.0 calibration is empirically anchored and locked by tests.
  ``signal_direction`` is added: the learned trade direction of a signal
  (+1/-1) from its forward rank-IC mean, so negative-IC signals are not
  mechanically traded backwards by the top-n long-only engine.
* v3: the reward is a **rank-ICIR minus a continuous turnover cost**.  The
  daily cross-sectional rank IC of the signal versus the forward target is
  summarized as an IC information ratio; turnover cost is charged
  proportionally to the simulated basket's average daily turnover at the
  configured per-turnover fee rate (no threshold jumps).  Sortino, the
  losing-basket collapse and the threshold turnover penalty are gone, so
  weak formulas keep a continuous, informative reward instead of a
  ``bad_reward`` cliff.  Validation scoring aggregates several independent
  sub-windows of the training tail by median.
* v2: the RL loop scored formulas through the vectorized batch path with a
  Sortino reward, a threshold turnover penalty and a losing-basket
  collapse (see git history for the exact semantics).
"""

from __future__ import annotations

import math

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig

REWARD_VERSION = "5"

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
    substitutes.  Single code path for the backtest metrics.
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


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks of ``x`` (ties share the mean of their ranks)."""

    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    obs = np.empty(n, dtype=bool)
    obs[0] = True
    np.not_equal(x_sorted[1:], x_sorted[:-1], out=obs[1:])
    dense_sorted = np.cumsum(obs) - 1
    dense = np.empty_like(dense_sorted)
    dense[order] = dense_sorted
    counts = np.bincount(dense)
    cum = np.cumsum(counts)
    avg = (cum - counts + 1 + cum) / 2.0
    return avg[dense]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; degenerate variance yields NaN."""

    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a @ a) * (b @ b)))
    if not math.isfinite(den) or den <= 1e-12:
        return math.nan
    return float(a @ b) / den


def rank_ic_series(
    signals: np.ndarray,
    target_ret: np.ndarray,
    min_stocks: int = 10,
) -> np.ndarray:
    """Per-date cross-sectional rank IC of each signal vs the forward target.

    ``signals`` is ``[B, stocks, dates]`` and ``target_ret`` is
    ``[stocks, dates]``; the result is ``[B, dates]`` of Spearman rank IC
    (NaN on days without at least ``min_stocks`` finite pairs).  Non-finite
    signal cells are excluded per row, so a formula that is invalid on a
    subset of stocks never fabricates correlation.
    """

    signals = np.asarray(signals, dtype=np.float64)
    target = np.asarray(target_ret, dtype=np.float64)
    if signals.ndim == 2:
        signals = signals[None]
    b, _, t = signals.shape
    out = np.full((b, t), np.nan)
    for day in range(t):
        tgt = target[:, day]
        base_valid = np.isfinite(tgt)
        if int(base_valid.sum()) < min_stocks:
            continue
        for i in range(b):
            col = signals[i, :, day]
            valid = base_valid & np.isfinite(col)
            if int(valid.sum()) < min_stocks:
                continue
            out[i, day] = _pearson(_rankdata(col[valid]), _rankdata(tgt[valid]))
    return out


def icir_from_series(ic: np.ndarray) -> np.ndarray:
    """IC information ratio (mean / std over dates) per row.

    Rows with fewer than two finite IC observations score 0.0 (an
    under-identified ratio is not evidence of predictive power).
    """

    ic = np.asarray(ic, dtype=np.float64)
    if ic.ndim == 1:
        ic = ic[None]
    b, t = ic.shape
    finite = np.isfinite(ic)
    n = finite.sum(axis=1)
    with np.errstate(all="ignore"):
        mean = np.where(
            n > 0, np.nansum(np.where(finite, ic, 0.0), axis=1) / np.maximum(n, 1), 0.0
        )
        dev = np.where(finite, ic - mean[:, None], 0.0)
        var = np.where(
            n > 1, (dev**2).sum(axis=1) / np.maximum(n - 1, 1), 0.0
        )
        std = np.sqrt(np.clip(var, 0.0, None))
        icir = mean / (std + 1e-9)
    return np.where(n >= 2, icir, 0.0)


def per_turnover_cost_rate(cfg: BacktestConfig) -> float:
    """Fee fraction paid per unit of two-sided turnover.

    Buying weight ``w`` and selling weight ``w`` together pay
    ``2*commission + stamp + 2*transfer + 2*slippage`` per unit of ``w``;
    the per-trade minimum-commission floor is ignored (a basket of ~30
    positions trades far above the floor on average).
    """

    return (
        2.0 * cfg.commission_rate
        + cfg.stamp_tax_rate
        + 2.0 * cfg.transfer_fee_rate
        + 2.0 * cfg.slippage_rate
    )


def annualized_turnover_cost(avg_turnover: np.ndarray, cfg: BacktestConfig) -> np.ndarray:
    """Annualized cost drag of a daily average turnover, in ICIR units."""

    avg_turnover = np.asarray(avg_turnover, dtype=np.float64)
    return avg_turnover * per_turnover_cost_rate(cfg) * _ANNUALIZATION


def simulate_basket_daily_returns(
    signal: np.ndarray,
    target_ret: np.ndarray,
    cfg: BacktestConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Daily net returns of the long-only top-n basket, plus its average
    turnover.

    Mirrors the backtest engine's target-weight construction (position
    weight ``min(1/top_n, single_weight_cap)``, renormalized) and charges
    the same trading-cost model.  Non-finite target cells contribute a zero
    return; days with fewer than ``top_n`` finite signals contribute a zero
    return and keep the previous basket.  ``blocked_buy`` / ``blocked_sell``
    carry the engine's tradability masks (``[stock, date]`` bool); see
    :func:`simulate_basket_daily_returns_batch` for their semantics.
    """

    daily_full, avg_to_full = simulate_basket_daily_returns_batch(
        np.asarray(signal, dtype=np.float64)[None],
        np.asarray(target_ret, dtype=np.float64),
        cfg,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
    )
    return daily_full[0], float(avg_to_full[0])


def simulate_basket_daily_returns_batch(
    signals: np.ndarray,
    target_ret: np.ndarray,
    cfg: BacktestConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched basket simulation: one pass over dates, ``B`` formulas at once.

    ``signals`` is ``[B, stocks, dates]``; the result is
    ``(daily_full [B, dates-1], avg_turnover_full [B])``.  Per date the
    top-n finite signals are selected (a tie between exactly equal values
    may resolve to a different equally valued set than the scalar path),
    weights renormalize to ``1/top_n`` each, invalid days earn zero and
    keep the previous basket, and both baskets share one trading-cost
    model.  The RL reward only consumes the average turnover (for the
    continuous cost drag); the daily series stays available for diagnostics
    and tests.

    ``blocked_buy`` / ``blocked_sell`` (``[stocks, dates]`` bool, both
    optional) align the simulation with the backtest engine's execution
    rules: buy-blocked stocks (suspended or opening at a one-word limit-up)
    are excluded from the top-n selection, and sell-blocked positions
    (suspended or opening at a one-word limit-down) are force-held with the
    remaining weights renormalized, so the reward can never credit a trade
    the engine could not have made.  Selection at signal day ``t`` executes
    at the ``t+1`` open, so the masks are consumed at column ``t+1`` —
    the same execution-day alignment as the engine.  Without the masks the
    simulation keeps the unconstrained semantics (rewards stay comparable
    across ``REWARD_VERSION`` for callers that never had masks).
    """

    signals = np.asarray(signals, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    b, n_stocks, n_dates = signals.shape
    daily_full = np.zeros((b, max(n_dates - 1, 0)))
    avg_to_full = np.zeros(b)
    if n_dates < 2 or n_stocks == 0:
        return daily_full, avg_to_full

    if blocked_buy is not None:
        blocked_buy = np.asarray(blocked_buy, dtype=bool)
        if blocked_buy.shape != (n_stocks, n_dates):
            raise ValueError(
                f"blocked_buy shape {blocked_buy.shape} does not match "
                f"({n_stocks}, {n_dates})"
            )
    if blocked_sell is not None:
        blocked_sell = np.asarray(blocked_sell, dtype=bool)
        if blocked_sell.shape != (n_stocks, n_dates):
            raise ValueError(
                f"blocked_sell shape {blocked_sell.shape} does not match "
                f"({n_stocks}, {n_dates})"
            )

    top_n = min(int(cfg.top_n), n_stocks)
    if top_n <= 0:
        return daily_full, avg_to_full

    weight = min(1.0 / top_n, float(cfg.single_weight_cap))
    b_idx = np.arange(b)[:, None]
    prev = np.zeros((b, n_stocks))
    turn_acc = np.zeros(b)
    # Non-finite target cells contribute zero; cleaned once outside the loop.
    target_clean = np.where(np.isfinite(target_ret), target_ret, 0.0)

    for t in range(n_dates - 1):
        # Selection uses the signal column t but executes at the t+1 open
        # (the target return is open[t+1] -> open[t+2]); the tradability
        # masks therefore use the t+1 columns, exactly like the engine.
        exec_col = t + 1
        column = signals[:, :, t]  # [B, stocks]
        finite = np.isfinite(column)
        if blocked_buy is not None:
            finite &= ~blocked_buy[None, :, exec_col]
        day_ok = finite.sum(axis=1) >= top_n
        fill = np.where(finite, column, -np.inf)
        top_idx = np.argpartition(fill, kth=-top_n, axis=1)[:, -top_n:]  # [B, top_n]
        fresh = np.zeros((b, n_stocks))
        fresh[b_idx, top_idx] = weight
        totals = fresh.sum(axis=1, keepdims=True)
        fresh = np.where(totals > 0, fresh / totals, 0.0)

        target = np.where(day_ok[:, None], fresh, prev)
        if blocked_sell is not None:
            # A position that must be reduced but cannot be sold at the
            # execution open (suspended or one-word limit-down) is held:
            # the remaining weights renormalize so cash stays consistent,
            # exactly like the engine.
            hold = blocked_sell[None, :, exec_col] & (target < prev)
            target = np.where(hold, prev, target)
        totals = target.sum(axis=1, keepdims=True)
        target = np.where(totals > 0, target / np.maximum(totals, 1e-12), 0.0)

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

    avg_to_full = turn_acc / max(n_dates - 1, 1)
    return daily_full, avg_to_full


def formula_reward(
    signal: np.ndarray,
    target_ret: np.ndarray,
    bt_cfg: BacktestConfig,
    reward_cfg: RewardConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
) -> float:
    """Scalar v5 reward of one signal: clipped ICIR minus turnover cost.

    Reference path for the batched implementation: the IC is computed over
    the full window and the cost drag over the simulated basket's average
    daily turnover.  ``blocked_buy`` / ``blocked_sell`` are the engine's
    tradability masks (see :func:`simulate_basket_daily_returns_batch`).
    """

    ic = icir_from_series(
        rank_ic_series(
            np.asarray(signal, dtype=np.float64)[None],
            target_ret,
            reward_cfg.ic_min_stocks,
        )
    )[0]
    _, avg_to = simulate_basket_daily_returns(
        signal, target_ret, bt_cfg, blocked_buy, blocked_sell
    )
    raw = ic - reward_cfg.cost_weight * annualized_turnover_cost(avg_to, bt_cfg)
    return float(np.clip(raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high))


def batched_basket_rewards(
    signals: np.ndarray,
    target_ret: np.ndarray,
    bt_cfg: BacktestConfig,
    reward_cfg: RewardConfig,
    val_windows: list[tuple[int, int]] | None = None,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """v5 rewards for a batch of signals: ICIR minus turnover cost, with the
    raw ICIR values exposed for the trainer's signal-quality gate.

    ``signals`` is ``[B, stocks, dates]``.  Returns
    ``(rewards [B], val_rewards [B] | None, icir [B], val_icir [B] | None)``:
    the full-window reward is clipped ICIR minus the proportional
    turnover-cost drag; with ``val_windows`` (column index pairs, half-open)
    the validation reward is the **median** over the windows of the same
    quantity computed on each window independently (each window's basket
    restarts from zero weights, mirroring a fresh out-of-sample deployment),
    and ``val_icir`` is the median window ICIR.  Without ``val_windows``
    the second and fourth results are ``None``.  ``blocked_buy`` /
    ``blocked_sell`` are ``[stocks, dates]`` tradability masks shared by all
    rows (per window they are sliced exactly like the signals).
    """

    signals = np.asarray(signals, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    icir = icir_from_series(
        rank_ic_series(signals, target_ret, reward_cfg.ic_min_stocks)
    )
    _, avg_to = simulate_basket_daily_returns_batch(
        signals, target_ret, bt_cfg, blocked_buy, blocked_sell
    )
    raw = icir - reward_cfg.cost_weight * annualized_turnover_cost(avg_to, bt_cfg)
    rewards = np.clip(raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high)

    val_rewards: np.ndarray | None = None
    val_icir: np.ndarray | None = None
    if val_windows:
        per_window = []
        per_window_icir = []
        for start, end in val_windows:
            win_signals = signals[:, :, start:end]
            win_target = target_ret[:, start:end]
            win_buy = blocked_buy[:, start:end] if blocked_buy is not None else None
            win_sell = blocked_sell[:, start:end] if blocked_sell is not None else None
            win_icir = icir_from_series(
                rank_ic_series(win_signals, win_target, reward_cfg.ic_min_stocks)
            )
            _, win_to = simulate_basket_daily_returns_batch(
                win_signals, win_target, bt_cfg, win_buy, win_sell
            )
            win_raw = win_icir - reward_cfg.cost_weight * annualized_turnover_cost(
                win_to, bt_cfg
            )
            per_window.append(
                np.clip(win_raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high)
            )
            per_window_icir.append(win_icir)
        val_rewards = np.median(np.stack(per_window, axis=1), axis=1)
        val_icir = np.median(np.stack(per_window_icir, axis=1), axis=1)
    return rewards, val_rewards, icir, val_icir


def signal_direction(
    signal: np.ndarray,
    target_ret: np.ndarray,
    min_stocks: int = 10,
) -> int:
    """Learned trade direction of one signal: +1 or -1.

    The direction is the sign of the mean forward rank IC (a negative-IC
    signal is traded by flipping it, so the top-n long-only engine never
    mechanically buys the wrong side).  Without enough finite IC
    observations the direction defaults to +1 (neutral).
    """

    ic = rank_ic_series(
        np.asarray(signal, dtype=np.float64)[None],
        np.asarray(target_ret, dtype=np.float64),
        min_stocks,
    )[0]
    finite = ic[np.isfinite(ic)]
    if finite.size == 0:
        return 1
    return 1 if float(finite.mean()) >= 0.0 else -1
