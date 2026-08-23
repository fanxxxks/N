"""Unified reward scoring for formula training.

This module is the single source of truth for reward statistics and the
continuous top-n basket used by candidate scoring. Execution fees themselves
live in :mod:`ashare_execution` and are shared with backtests and paper trades.

* the annualized Sortino ratio (used by the backtest engine metrics),
* the cheap long-only top-n basket simulation used inside the RL loop.

:data:`REWARD_VERSION` is bumped on every semantic change to this module's
scoring behavior and is recorded in training artifacts and experiment
archives, so ``best_reward`` values from different reward generations can
never be compared silently.

Version history
---------------
* v7: signal-date universe eligibility in every candidate-quality path.
  ``rank_ic_series``/``signal_direction``/``formula_reward``/
  ``batched_basket_rewards``/``simulate_basket_daily_returns(_batch)`` and
  the candidate scorer all take the PIT ``universe_mask``
  (``[stock, date]``): a signal column ``t`` contributes only its
  signal-date eligible cells, the basket's top-n selection uses
  ``universe_mask[:, t]`` (execution tradability still uses ``t+1``), and
  the scorer's near-constant check and direction tie-break scan eligible
  observations only.  A stock that leaves the member pool has its target
  weight go to zero through the ordinary sell path (force-held when the
  execution day is sell-blocked) — positions never vanish silently.
* v6: direction-symmetric candidate scoring and exact daily execution-cost
  fractions. Continuous weights are converted to yuan using each path's
  current capital; every non-zero order receives the same minimum-commission,
  tax, transfer-fee and slippage treatment as the backtest and paper matcher.
  Only complete signal[t] -> open[t+1] -> open[t+2] periods are simulated.
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

from dataclasses import dataclass
import math

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_execution import ExecutionCostModel

REWARD_VERSION = "7"

_ANNUALIZATION = 252


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
    *,
    universe_mask: np.ndarray,
) -> np.ndarray:
    """Per-date cross-sectional rank IC of each signal vs the forward target.

    ``signals`` is ``[B, stocks, dates]`` and ``target_ret`` is
    ``[stocks, dates]``; the result is ``[B, dates]`` of Spearman rank IC
    (NaN on days without at least ``min_stocks`` finite pairs).  Non-finite
    signal cells are excluded per row, so a formula that is invalid on a
    subset of stocks never fabricates correlation.  ``universe_mask`` is the
    mandatory ``[stocks, dates]`` bool PIT eligibility mask: only signal-date
    eligible cells (``universe_mask[:, day]``) may enter a day's
    correlation, so a future member's finite values can never move the IC
    before it joins.
    """

    signals = np.asarray(signals, dtype=np.float64)
    target = np.asarray(target_ret, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if universe_mask.shape != target.shape:
        raise ValueError(
            f"universe_mask shape {universe_mask.shape} does not match "
            f"target shape {target.shape}"
        )
    if signals.ndim == 2:
        signals = signals[None]
    b, _, t = signals.shape
    out = np.full((b, t), np.nan)
    for day in range(t):
        tgt = target[:, day]
        base_valid = np.isfinite(tgt) & universe_mask[:, day]
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


@dataclass(frozen=True)
class BasketSimulation:
    """Exact scalar basket path for every complete t+2 signal period."""

    daily_gross_returns: np.ndarray
    daily_cost_fractions: np.ndarray
    daily_net_returns: np.ndarray
    turnover: np.ndarray

    @property
    def average_turnover(self) -> float:
        return float(self.turnover.mean()) if self.turnover.size else 0.0

    def __iter__(self):
        """Legacy two-value unpacking: net returns and average turnover."""

        yield self.daily_net_returns
        yield self.average_turnover


@dataclass(frozen=True)
class BatchBasketSimulation:
    """Batched counterpart of :class:`BasketSimulation`."""

    daily_gross_returns: np.ndarray
    daily_cost_fractions: np.ndarray
    daily_net_returns: np.ndarray
    turnover: np.ndarray

    @property
    def average_turnover(self) -> np.ndarray:
        if self.turnover.shape[1] == 0:
            return np.zeros(self.turnover.shape[0], dtype=np.float64)
        return self.turnover.mean(axis=1)

    def __iter__(self):
        """Legacy two-value unpacking: net returns and average turnover."""

        yield self.daily_net_returns
        yield self.average_turnover


def simulate_basket_daily_returns(
    signal: np.ndarray,
    target_ret: np.ndarray,
    cfg: BacktestConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    signal_range: tuple[int, int] | None = None,
    *,
    universe_mask: np.ndarray,
) -> BasketSimulation:
    """Exact daily top-n basket path, restarted from cash at range start.

    Mirrors the backtest engine's target-weight construction (position
    weight ``min(1/top_n, single_weight_cap)``, renormalized) and charges
    the same trading-cost model.  Non-finite target cells contribute a zero
    return; days with fewer than ``top_n`` finite signals contribute a zero
    return and keep the previous basket.  ``blocked_buy`` / ``blocked_sell``
    carry the engine's tradability masks (``[stock, date]`` bool);
    ``universe_mask`` is the mandatory ``[stock, date]`` bool PIT eligibility
    mask consumed at the signal date; see
    :func:`simulate_basket_daily_returns_batch` for their semantics.
    """

    batch = simulate_basket_daily_returns_batch(
        np.asarray(signal, dtype=np.float64)[None],
        np.asarray(target_ret, dtype=np.float64),
        cfg,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        signal_range=signal_range,
        universe_mask=universe_mask,
    )
    return BasketSimulation(
        daily_gross_returns=batch.daily_gross_returns[0],
        daily_cost_fractions=batch.daily_cost_fractions[0],
        daily_net_returns=batch.daily_net_returns[0],
        turnover=batch.turnover[0],
    )


def simulate_basket_daily_returns_batch(
    signals: np.ndarray,
    target_ret: np.ndarray,
    cfg: BacktestConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    signal_range: tuple[int, int] | None = None,
    *,
    universe_mask: np.ndarray,
) -> BatchBasketSimulation:
    """Batched exact basket simulation over an explicit signal range.

    ``signals`` is ``[B, stocks, price_dates]``. By default only
    ``range(price_dates - 2)`` is executable; an explicit half-open
    ``signal_range`` is used for independently restarted validation windows.
    Per date the
    top-n finite signals are selected (a tie between exactly equal values
    may resolve to a different equally valued set than the scalar path),
    weights renormalize to ``1/top_n`` each, invalid days earn zero and
    keep the previous basket, and both baskets share one trading-cost
    model. The returned gross, cost, net and turnover arrays make the exact
    daily cost path auditable.

    ``blocked_buy`` / ``blocked_sell`` (``[stocks, dates]`` bool, both
    optional) align the simulation with the backtest engine's execution
    rules: buy-blocked stocks (suspended or opening at a one-word limit-up)
    are excluded from the top-n selection, and sell-blocked positions
    (suspended or opening at a one-word limit-down) are force-held with the
    remaining weights renormalized, so the reward can never credit a trade
    the engine could not have made.  Selection at signal day ``t`` executes
    at the ``t+1`` open, so the tradability masks are consumed at column
    ``t+1`` — the same execution-day alignment as the engine.

    ``universe_mask`` (``[stocks, dates]`` bool, mandatory) is the PIT
    eligibility mask consumed at the **signal date**: only
    ``universe_mask[:, t]`` cells may enter the top-n selection at ``t``, so
    a future member can never join the basket before its join day, and a
    stock that exits the member pool has its target weight go to zero
    through the ordinary sell path (counted at full cost, or force-held
    when the execution day is sell-blocked) — the mask never makes an
    existing position vanish silently.  When fewer than ``top_n`` cells are
    selectable the selected set is renormalized, the same under-filled
    semantics as the backtest engine.
    """

    signals = np.asarray(signals, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    if signals.ndim != 3:
        raise ValueError("signals must be [batch, stock, date]")
    b, n_stocks, n_dates = signals.shape
    if target_ret.shape != (n_stocks, n_dates):
        raise ValueError(
            f"target_ret shape {target_ret.shape} does not match "
            f"({n_stocks}, {n_dates})"
        )
    if signal_range is None:
        signal_range = (0, max(n_dates - 2, 0))
    signal_start, signal_end = (int(signal_range[0]), int(signal_range[1]))
    max_signal_end = max(n_dates - 2, 0)
    if not 0 <= signal_start <= signal_end <= max_signal_end:
        raise ValueError(
            f"signal_range {signal_range} is outside [0, {max_signal_end})"
        )
    n_periods = signal_end - signal_start
    gross_full = np.zeros((b, n_periods), dtype=np.float64)
    cost_full = np.zeros((b, n_periods), dtype=np.float64)
    net_full = np.zeros((b, n_periods), dtype=np.float64)
    turnover_full = np.zeros((b, n_periods), dtype=np.float64)

    def empty_result() -> BatchBasketSimulation:
        return BatchBasketSimulation(gross_full, cost_full, net_full, turnover_full)

    if n_periods == 0 or n_stocks == 0:
        return empty_result()

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
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if universe_mask.shape != (n_stocks, n_dates):
        raise ValueError(
            f"universe_mask shape {universe_mask.shape} does not match "
            f"({n_stocks}, {n_dates})"
        )

    top_n = min(int(cfg.top_n), n_stocks)
    if top_n <= 0:
        return empty_result()

    weight = min(1.0 / top_n, float(cfg.single_weight_cap))
    b_idx = np.arange(b)[:, None]
    prev = np.zeros((b, n_stocks))
    capital = np.full(b, float(cfg.initial_capital), dtype=np.float64)
    cost_model = ExecutionCostModel.from_config(cfg)
    # Non-finite target cells contribute zero; cleaned once outside the loop.
    target_clean = np.where(np.isfinite(target_ret), target_ret, 0.0)

    for output_col, t in enumerate(range(signal_start, signal_end)):
        # Selection uses the signal column t but executes at the t+1 open
        # (the target return is open[t+1] -> open[t+2]); the tradability
        # masks therefore use the t+1 columns, exactly like the engine,
        # while the PIT eligibility mask uses the signal-date column t.
        exec_col = t + 1
        column = signals[:, :, t]  # [B, stocks]
        finite = np.isfinite(column)
        if blocked_buy is not None:
            finite &= ~blocked_buy[None, :, exec_col]
        finite &= universe_mask[None, :, t]
        fill = np.where(finite, column, -np.inf)
        top_idx = np.argpartition(fill, kth=-top_n, axis=1)[:, -top_n:]  # [B, top_n]
        fresh = np.zeros((b, n_stocks))
        selected_is_finite = np.take_along_axis(finite, top_idx, axis=1)
        fresh[b_idx, top_idx] = selected_is_finite * weight
        totals = fresh.sum(axis=1, keepdims=True)
        fresh = np.divide(
            fresh,
            totals,
            out=np.zeros_like(fresh),
            where=totals > 0,
        )

        target = fresh
        if blocked_sell is not None:
            # A position that must be reduced but cannot be sold at the
            # execution open (suspended or one-word limit-down) is held:
            # the remaining weights renormalize so cash stays consistent,
            # exactly like the engine.
            hold = blocked_sell[None, :, exec_col] & (target < prev)
            target = np.where(hold, prev, target)
        totals = target.sum(axis=1, keepdims=True)
        target = np.divide(
            target,
            totals,
            out=np.zeros_like(target),
            where=totals > 0,
        )

        buy = np.maximum(target - prev, 0.0)
        sell = np.maximum(prev - target, 0.0)
        cost = np.asarray(
            cost_model.rebalance_cost_fraction(buy, sell, capital),
            dtype=np.float64,
        )
        rets = target_clean[:, t][None, :]
        gross = (target * rets).sum(axis=1)
        net = gross - cost
        turnover = np.abs(target - prev).sum(axis=1)
        gross_full[:, output_col] = gross
        cost_full[:, output_col] = cost
        net_full[:, output_col] = net
        turnover_full[:, output_col] = turnover
        capital *= 1.0 + net
        prev = target

    return BatchBasketSimulation(gross_full, cost_full, net_full, turnover_full)


def formula_reward(
    signal: np.ndarray,
    target_ret: np.ndarray,
    bt_cfg: BacktestConfig,
    reward_cfg: RewardConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    signal_range: tuple[int, int] | None = None,
    *,
    universe_mask: np.ndarray,
) -> float:
    """Scalar v7 reward: clipped ICIR minus exact annualized daily cost.

    Reference path for the batched implementation: the IC is computed over
    the full window and the cost drag over the simulated basket's average
    daily turnover.  ``blocked_buy`` / ``blocked_sell`` are the engine's
    tradability masks (see :func:`simulate_basket_daily_returns_batch`);
    ``universe_mask`` is the mandatory ``[stock, date]`` PIT eligibility
    mask used at the signal date by both the IC and the basket.
    """

    signal = np.asarray(signal, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if universe_mask.shape != signal.shape:
        raise ValueError(
            f"universe_mask shape {universe_mask.shape} does not match "
            f"signal shape {signal.shape}"
        )
    if signal_range is None:
        signal_range = (0, max(signal.shape[1] - 2, 0))
    start, end = signal_range
    ic = icir_from_series(
        rank_ic_series(
            signal[None, :, start:end],
            target_ret[:, start:end],
            reward_cfg.ic_min_stocks,
            universe_mask=universe_mask[:, start:end],
        )
    )[0]
    simulation = simulate_basket_daily_returns(
        signal,
        target_ret,
        bt_cfg,
        blocked_buy,
        blocked_sell,
        signal_range,
        universe_mask=universe_mask,
    )
    annualized_cost = (
        float(simulation.daily_cost_fractions.mean()) * _ANNUALIZATION
        if simulation.daily_cost_fractions.size
        else 0.0
    )
    raw = ic - reward_cfg.cost_weight * annualized_cost
    return float(np.clip(raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high))


def batched_basket_rewards(
    signals: np.ndarray,
    target_ret: np.ndarray,
    bt_cfg: BacktestConfig,
    reward_cfg: RewardConfig,
    val_windows: list[tuple[int, int]] | None = None,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    full_signal_range: tuple[int, int] | None = None,
    *,
    universe_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    """v7 rewards for a batch: ICIR minus exact annualized daily costs, with
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
    ``universe_mask`` is the mandatory ``[stocks, dates]`` PIT eligibility
    mask, sliced per window exactly like the signals, so each window's IC and
    basket see only its signal-date eligible cells.
    """

    signals = np.asarray(signals, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if universe_mask.shape != target_ret.shape:
        raise ValueError(
            f"universe_mask shape {universe_mask.shape} does not match "
            f"target_ret shape {target_ret.shape}"
        )
    if full_signal_range is None:
        full_signal_range = (0, max(signals.shape[2] - 2, 0))
    full_start, full_end = full_signal_range
    icir = icir_from_series(
        rank_ic_series(
            signals[:, :, full_start:full_end],
            target_ret[:, full_start:full_end],
            reward_cfg.ic_min_stocks,
            universe_mask=universe_mask[:, full_start:full_end],
        )
    )
    simulation = simulate_basket_daily_returns_batch(
        signals,
        target_ret,
        bt_cfg,
        blocked_buy,
        blocked_sell,
        full_signal_range,
        universe_mask=universe_mask,
    )
    mean_cost = (
        simulation.daily_cost_fractions.mean(axis=1)
        if simulation.daily_cost_fractions.shape[1]
        else np.zeros(signals.shape[0], dtype=np.float64)
    )
    raw = icir - reward_cfg.cost_weight * mean_cost * _ANNUALIZATION
    rewards = np.clip(raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high)

    val_rewards: np.ndarray | None = None
    val_icir: np.ndarray | None = None
    if val_windows:
        per_window = []
        per_window_icir = []
        for start, end in val_windows:
            win_icir = icir_from_series(
                rank_ic_series(
                    signals[:, :, start:end],
                    target_ret[:, start:end],
                    reward_cfg.ic_min_stocks,
                    universe_mask=universe_mask[:, start:end],
                )
            )
            win_simulation = simulate_basket_daily_returns_batch(
                signals,
                target_ret,
                bt_cfg,
                blocked_buy,
                blocked_sell,
                (start, end),
                universe_mask=universe_mask,
            )
            win_mean_cost = (
                win_simulation.daily_cost_fractions.mean(axis=1)
                if win_simulation.daily_cost_fractions.shape[1]
                else np.zeros(signals.shape[0], dtype=np.float64)
            )
            win_raw = (
                win_icir
                - reward_cfg.cost_weight * win_mean_cost * _ANNUALIZATION
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
    *,
    universe_mask: np.ndarray,
) -> int:
    """Learned trade direction of one signal: +1 or -1.

    The direction is the sign of the mean forward rank IC (a negative-IC
    signal is traded by flipping it, so the top-n long-only engine never
    mechanically buys the wrong side).  Without enough finite IC
    observations the direction defaults to +1 (neutral).
    ``universe_mask`` is the mandatory ``[stock, date]`` PIT eligibility
    mask: the direction is decided on signal-date eligible observations
    only, so a future member cannot flip it before its join day.
    """

    ic = rank_ic_series(
        np.asarray(signal, dtype=np.float64)[None],
        np.asarray(target_ret, dtype=np.float64),
        min_stocks,
        universe_mask=universe_mask,
    )[0]
    finite = ic[np.isfinite(ic)]
    if finite.size == 0:
        return 1
    return 1 if float(finite.mean()) >= 0.0 else -1
