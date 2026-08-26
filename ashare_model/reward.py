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
* v13: the reward's primary term is the portfolio **active IR** — annualized
  effective-n shrunk information ratio of the gross basket returns versus
  the equal-weight universe benchmark — minus the exact execution-cost
  drag; IC becomes the auxiliary reported/gated metric (T1-04).  The
  basket selects on signal-date AND entry-date PIT eligibility, exactly
  like the engine, and portfolio objectives (active IR, risk exposure,
  turnover, capacity utilization) are returned per candidate for
  constrained/Pareto selection; ``adv`` enables the capacity audit and
  ``capacity_above_maximum`` gates illiquid positions.  Rewards/artifacts
  recorded with earlier versions are not comparable.
* v12: the IC term of every reward path is the **robust ICIR** — the naive
  mean/std ratio shrunk by the effective sample size under autocorrelation
  (Newey-West HAC variance, ``ic_hac_max_lags``), so serial-correlated IC
  series are discounted (T1-03).  Complexity is billed from the AST
  (``complexity_penalty * complexity_bill`` combining node count, depth,
  longest operator window and operation cost) for every formula, not only
  bare factors, and formulas whose bill exceeds ``max_complexity`` are
  rejected.  Rewards/artifacts recorded with earlier versions are not
  comparable.
* v11: the basket weight construction aligns with the T1-02 no-signal
  contract: a selectable cross-section without dispersion (fewer than two
  distinct values) is never rebalanced (previous basket held, zero
  turnover, zero cost); under-filled days keep the remainder in cash —
  weights are never renormalized upward; force-held (sell-blocked)
  positions consume the budget and fresh buys scale down to it, so
  ``single_weight_cap`` is a hard per-name ceiling on every path; exact
  selection ties resolve by the new ``tie_break_keys`` (stable per-stock
  identifiers), making the basket invariant under stock-row permutation.
  Rewards/artifacts recorded with earlier versions are not comparable.
* v10: the policy gradient is isolated from the validation tail.  The
  trainer's primary scoring window (``train_signal_range``) is now the
  in-sample head that ends where the validation tail begins, instead of
  the full training window: REINFORCE rewards never read the selection
  data, closing the optimistic in-training bias where formulas were
  pushed toward whatever overfit the validation tail.  CandidateScore
  renames ``full_window_*`` to ``train_*`` (same window semantics change,
  recorded in artifacts); near-constant rejection scans the learning
  window.  Artifact schema changes with the rename.
* v9: the rolling CAPM factors (BETA_60/IVOL_60/RSQ_60) align the market
  window to each stock's own valid sessions.  The market prefix sums used
  to accumulate over every calendar session in the window while the
  observation count only counted the stock's trading days, so stocks with
  suspension gaps regressed on market returns from sessions they did not
  trade (biased cov/var_m).  Factor values of suspended stocks change;
  artifacts recorded with earlier versions are not comparable for
  CAPM-derived signals.
* v8: the JUMP operator is causal.  It used to standardize by the
  full-timeline mean/std of the signal — a look-ahead (position ``t`` saw
  the future of the window), which leaked post-test-window data into
  walk-forward evaluation whenever a JUMP formula was executed on the full
  tensor and sliced to the test window.  It now measures the value against
  its trailing 60-session baseline (expanding before that, population
  std), matching the windowed-operator conventions of the rest of the
  registry.  Signals and rewards of v7 artifacts containing JUMP are not
  comparable with v8.
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

REWARD_VERSION = "13"

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
        downside_std = daily_std * math.sqrt(_ANNUALIZATION)
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


def _robust_icir_batch(ic: np.ndarray, max_lags: int | None) -> np.ndarray:
    """Effective-n shrunk (HAC) ICIR per row (T1-03); degenerate rows 0.0.

    A perfectly stable IC series has an unbounded ratio; the reward path
    caps it at a large finite value so artifacts and the eligibility
    checks never see a non-finite ICIR (the reward itself saturates the
    clip band either way).
    """

    from .signal_quality import robust_icir

    ic = np.asarray(ic, dtype=np.float64)
    if ic.ndim == 1:
        ic = ic[None]
    out = np.asarray(
        [robust_icir(row, max_lags) for row in ic], dtype=np.float64
    )
    return np.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9)


@dataclass(frozen=True)
class BasketSimulation:
    """Exact scalar basket path for every complete t+2 signal period."""

    daily_gross_returns: np.ndarray
    daily_cost_fractions: np.ndarray
    daily_net_returns: np.ndarray
    turnover: np.ndarray
    # T1-04: per-period capacity utilization (max over held names of
    # position value / dollar volume); None when no adv was provided.
    capacity_utilization: np.ndarray | None = None

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
    capacity_utilization: np.ndarray | None = None

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
    tie_break_keys: np.ndarray | None = None,
    adv: np.ndarray | None = None,
) -> BasketSimulation:
    """Exact daily top-n basket path, restarted from cash at range start.

    Mirrors the backtest engine's target-weight construction (position
    weight ``min(1/top_n, single_weight_cap)``, **never renormalized
    upward**: under-filled days keep cash and force-held positions consume
    the budget, so ``single_weight_cap`` is a hard per-name ceiling) and
    charges the same trading-cost model.  A selectable cross-section
    without dispersion (fewer than two distinct values) triggers no
    rebalance at all: the previous basket is held with zero turnover and
    zero cost.  ``blocked_buy`` / ``blocked_sell`` carry the engine's
    tradability masks (``[stock, date]`` bool); ``universe_mask`` is the
    mandatory ``[stock, date]`` bool PIT eligibility mask consumed at the
    signal date **and** the entry date (T1-04 alignment);
    ``tie_break_keys`` (per-stock stable identifiers, e.g. ``ts_codes``)
    resolve exact selection ties deterministically so the path is
    invariant under stock-row permutation; ``adv`` (``[stock, date]``
    dollar volume, optional) enables the capacity-utilization audit.  See
    :func:`simulate_basket_daily_returns_batch` for the full semantics.
    """

    batch = simulate_basket_daily_returns_batch(
        np.asarray(signal, dtype=np.float64)[None],
        np.asarray(target_ret, dtype=np.float64),
        cfg,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        signal_range=signal_range,
        universe_mask=universe_mask,
        tie_break_keys=tie_break_keys,
        adv=adv,
    )
    return BasketSimulation(
        daily_gross_returns=batch.daily_gross_returns[0],
        daily_cost_fractions=batch.daily_cost_fractions[0],
        daily_net_returns=batch.daily_net_returns[0],
        turnover=batch.turnover[0],
        capacity_utilization=(
            batch.capacity_utilization[0]
            if batch.capacity_utilization is not None
            else None
        ),
    )


def _batch_scale_to_budget(
    raw: np.ndarray, held: np.ndarray
) -> np.ndarray:
    """Scale fresh buys down to the cash budget left by force-held
    positions (the no-renormalization contract, T1-02).  ``raw``/``held``
    are ``[batch, stocks]``; per-name weights only shrink, never grow."""

    budget = np.maximum(1.0 - held.sum(axis=1, keepdims=True), 0.0)
    raw_sum = raw.sum(axis=1, keepdims=True)
    scale = np.minimum(1.0, budget / np.maximum(raw_sum, 1e-12))
    return raw * scale


def _batch_distinct_count(masked: np.ndarray) -> np.ndarray:
    """Number of distinct finite values per row of ``masked`` (NaN cells
    sort last and never count).  Vectorized distinct-count for the
    dispersion gate."""

    sorted_vals = np.sort(masked, axis=1)
    with np.errstate(invalid="ignore"):
        diff = np.diff(sorted_vals, axis=1)
        transitions = np.count_nonzero(
            np.isfinite(diff) & (diff > 0), axis=1
        )
    first_finite = np.isfinite(sorted_vals[:, 0]).astype(np.int64)
    return first_finite + transitions


def simulate_basket_daily_returns_batch(
    signals: np.ndarray,
    target_ret: np.ndarray,
    cfg: BacktestConfig,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    signal_range: tuple[int, int] | None = None,
    *,
    universe_mask: np.ndarray,
    tie_break_keys: np.ndarray | None = None,
    adv: np.ndarray | None = None,
) -> BatchBasketSimulation:
    """Batched exact basket simulation over an explicit signal range.

    ``signals`` is ``[B, stocks, price_dates]``. By default only
    ``range(price_dates - 2)`` is executable; an explicit half-open
    ``signal_range`` is used for independently restarted validation windows.
    Weight-construction contract (T1-02, shared with the backtest engine):

    * per-date top-n selection over the selectable set (signal-date AND
      entry-date PIT eligible, entry-day buyable, finite);
    * a selectable cross-section with fewer than two distinct values is
      **not rebalanced**: the previous basket is held with zero turnover
      and zero cost (an all-neutral day never churns the book);
    * every selected name gets ``min(1/top_n, single_weight_cap)`` and the
      weights are **never renormalized upward** — under-filled days keep
      the remainder in cash, and force-held (sell-blocked) positions
      consume the budget, so no path can ever push a name above
      ``single_weight_cap`` or the book above full investment;
    * exact selection ties resolve by ``tie_break_keys`` (per-stock stable
      identifiers), so the path is invariant under stock-row permutation.

    ``blocked_buy`` / ``blocked_sell`` (``[stocks, dates]`` bool, both
    optional) align the simulation with the backtest engine's execution
    rules: buy-blocked stocks (suspended or opening at a one-word limit-up)
    are excluded from the top-n selection, and sell-blocked positions
    (suspended or opening at a one-word limit-down) are force-held with
    the remaining weights scaled to the freed budget, exactly like the
    backtest engine.  Selection at signal day ``t`` executes at the
    ``t+1`` open, so the tradability masks are consumed at column ``t+1``
    — the same execution-day alignment as the engine.

    ``universe_mask`` (``[stocks, dates]`` bool, mandatory) is the PIT
    eligibility mask consumed at the **signal date AND the entry date**
    (T1-04 alignment, exactly like the engine): only
    ``universe_mask[:, t] & universe_mask[:, t+1]`` cells may enter the
    top-n selection at ``t``, so a future member can never join the basket
    before its join day and a stock that leaves the member pool between
    signal and execution can never be bought.  A stock that exits the
    member pool has its target weight go to zero through the ordinary sell
    path (counted at full cost, or force-held when the execution day is
    sell-blocked) — the mask never makes an existing position vanish
    silently.

    ``adv`` (``[stocks, dates]`` dollar volume, optional) enables the
    capacity audit: per period, the utilization of each held name is its
    position value divided by the execution-day dollar volume, and the
    reported capacity is the per-period maximum over held names (days
    without volume data contribute NaN and are skipped).
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
    capacity_full = np.full((b, n_periods), np.nan, dtype=np.float64)

    def empty_result() -> BatchBasketSimulation:
        return BatchBasketSimulation(gross_full, cost_full, net_full, turnover_full,
                                     capacity_utilization=(
                                         capacity_full if adv is not None else None
                                     ))

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
    if tie_break_keys is not None:
        tie_break_keys = np.asarray(tie_break_keys)
        if tie_break_keys.shape != (n_stocks,):
            raise ValueError(
                f"tie_break_keys shape {tie_break_keys.shape} does not match "
                f"({n_stocks},)"
            )
    if adv is not None:
        adv = np.asarray(adv, dtype=np.float64)
        if adv.shape != (n_stocks, n_dates):
            raise ValueError(
                f"adv shape {adv.shape} does not match ({n_stocks}, {n_dates})"
            )

    top_n = min(int(cfg.top_n), n_stocks)
    if top_n <= 0:
        return empty_result()

    weight = min(1.0 / top_n, float(cfg.single_weight_cap))
    prev = np.zeros((b, n_stocks))
    capital = np.full(b, float(cfg.initial_capital), dtype=np.float64)
    cost_model = ExecutionCostModel.from_config(cfg)
    # Non-finite target cells contribute zero; cleaned once outside the loop.
    target_clean = np.where(np.isfinite(target_ret), target_ret, 0.0)
    keys_tiled = (
        np.tile(tie_break_keys, (b, 1)) if tie_break_keys is not None else None
    )

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
        # T1-04 alignment: signal-date AND entry-date PIT eligibility,
        # exactly like the engine's ``eligible`` vector.
        finite &= universe_mask[None, :, t] & universe_mask[None, :, exec_col]

        # No-signal semantics: a selectable cross-section without
        # dispersion is not signal-rebalanced (previous basket held, zero
        # turnover, zero cost).  The mask-exit contract still applies:
        # positions that lost universe eligibility are liquidated through
        # the ordinary sell path (force-held when sell-blocked), exactly
        # like the engine.
        masked = np.where(finite, column, np.nan)
        selectable_count = finite.sum(axis=1)
        has_dispersion = _batch_distinct_count(masked) >= 2
        hold_rows = (selectable_count > 0) & ~has_dispersion
        hold_target = prev
        if hold_rows.any():
            ineligible = ~(
                universe_mask[None, :, t] & universe_mask[None, :, exec_col]
            )
            liquidate = ineligible & (prev > 0.0)
            hold_target = np.where(liquidate, 0.0, prev)
            if blocked_sell is not None:
                force_hold = liquidate & blocked_sell[None, :, exec_col]
                hold_target = np.where(force_hold, prev, hold_target)

        if not hold_rows.all():
            if tie_break_keys is not None:
                # Deterministic top-n by (value desc, key asc): rank_key =
                # -value sorts ascending with the highest values first;
                # excluded cells carry -inf in the rank key and sort last,
                # so they can never displace a selectable value.
                rank_key = np.where(finite, -column, np.inf)
                top_idx = np.empty((b, top_n), dtype=np.int64)
                for i in range(b):
                    top_idx[i] = np.lexsort((keys_tiled[i], rank_key[i]))[:top_n]
            else:
                fill = np.where(finite, column, -np.inf)
                top_idx = np.argpartition(fill, kth=-top_n, axis=1)[:, -top_n:]
            fresh = np.zeros((b, n_stocks))
            selected_is_finite = np.take_along_axis(finite, top_idx, axis=1)
            fresh[np.arange(b)[:, None], top_idx] = selected_is_finite * weight

            # Force-held reductions (sell-blocked) keep their previous
            # weight and consume the budget; fresh buys scale down, never
            # up.
            held = np.zeros((b, n_stocks))
            raw = fresh
            if blocked_sell is not None:
                sell_blocked = blocked_sell[None, :, exec_col]
                for _ in range(n_stocks + 1):
                    tentative = held + raw
                    new_holds = sell_blocked & (prev > tentative) & (held == 0.0)
                    if not new_holds.any():
                        break
                    held = np.where(new_holds, prev, held)
                    raw = np.where(new_holds, 0.0, raw)
                    raw = _batch_scale_to_budget(raw, held)
            target = held + _batch_scale_to_budget(raw, held)
            if hold_rows.any():
                # Held rows keep the hold-branch target (previous basket
                # with universe-exit liquidations); traded rows use the
                # freshly constructed target.
                target = np.where(hold_rows[:, None], hold_target, target)
        else:
            target = hold_target
        # Quantize weights to 1e-12: the scale-to-budget arithmetic leaves
        # ~1e-16 summation noise (row-order dependent), which would
        # otherwise fabricate phantom micro-orders that pay the full
        # minimum commission and break permutation invariance.
        target = np.round(target, 12)

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
        if adv is not None:
            # Capacity audit (T1-04): per held name, position value
            # divided by the execution-day dollar volume; the day's
            # utilization is the maximum over held names.  Days without
            # volume data (adv == 0) contribute NaN and are skipped.
            with np.errstate(divide="ignore", invalid="ignore"):
                util = (
                    capital[:, None] * target
                ) / adv[:, exec_col][None, :]
            util = np.where(adv[None, :, exec_col] > 0, util, np.nan)
            util = np.where(target > 0, util, np.nan)
            with np.errstate(invalid="ignore"):
                capacity_full[:, output_col] = np.nanmax(util, axis=1)
        capital *= 1.0 + net
        prev = target

    return BatchBasketSimulation(
        gross_full,
        cost_full,
        net_full,
        turnover_full,
        capacity_utilization=(
            capacity_full if adv is not None else None
        ),
    )


def _annualized_active_ir(
    active: np.ndarray, max_lags: int | None
) -> np.ndarray:
    """Annualized, effective-n shrunk information ratio of active returns.

    ``robust_icir(active) * sqrt(252)`` per row; a perfectly constant
    active series has an unbounded ratio, capped at a large finite value
    (the reward saturates the clip band either way).
    """

    from .signal_quality import robust_icir

    active = np.asarray(active, dtype=np.float64)
    if active.ndim == 1:
        active = active[None]
    out = np.asarray(
        [robust_icir(row, max_lags) for row in active], dtype=np.float64
    )
    return np.nan_to_num(out, nan=0.0, posinf=1e9, neginf=-1e9) * np.sqrt(
        _ANNUALIZATION
    )


def _window_benchmark(
    target_ret: np.ndarray,
    window: tuple[int, int],
    universe_mask: np.ndarray,
) -> np.ndarray:
    """Equal-weight universe benchmark over one signal window (engine path)."""

    from .backtest import equal_weight_benchmark_returns

    start, end = window
    return np.asarray(
        equal_weight_benchmark_returns(
            target_ret, list(range(start, end)), universe_mask
        ),
        dtype=np.float64,
    )


def _portfolio_objectives(
    simulation: BatchBasketSimulation,
    benchmark: np.ndarray,
    max_lags: int | None,
) -> np.ndarray:
    """Per-row ``[active_ir, exposure, turnover, capacity]`` of one
    window's simulation against its benchmark (T1-04).

    Accepts both the batched (``[B, periods]``) and the scalar
    (``[periods]``) simulation arrays.
    """

    gross = np.atleast_2d(simulation.daily_gross_returns)
    net = np.atleast_2d(simulation.daily_net_returns)
    turnover = np.atleast_2d(simulation.turnover)
    capacity = simulation.capacity_utilization
    if capacity is not None:
        capacity = np.atleast_2d(capacity)
    benchmark = np.asarray(benchmark, dtype=np.float64)
    active = gross - benchmark[None, :]
    active_ir = _annualized_active_ir(active, max_lags)
    exposure = np.asarray(
        [
            float(np.std(row, ddof=1)) * np.sqrt(_ANNUALIZATION)
            if row.size > 1
            else 0.0
            for row in net
        ],
        dtype=np.float64,
    )
    with np.errstate(invalid="ignore"):
        avg_turnover = (
            turnover.mean(axis=1)
            if turnover.shape[1] > 0
            else np.zeros(turnover.shape[0], dtype=np.float64)
        )
    avg_capacity = (
        np.nanmean(capacity, axis=1)
        if capacity is not None
        else np.full(net.shape[0], np.nan, dtype=np.float64)
    )
    return np.column_stack([active_ir, exposure, avg_turnover, avg_capacity])


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
    tie_break_keys: np.ndarray | None = None,
    adv: np.ndarray | None = None,
) -> float:
    """Scalar v13 reward: active IR minus exact annualized daily cost.

    The primary term is the portfolio **active IR** (gross basket returns
    minus the equal-weight universe benchmark, effective-n shrunk,
    annualized); the exact execution-cost drag and the (scorer-side)
    complexity bill are subtracted, and IC is the auxiliary reported/gated
    metric.  Reference path for the batched implementation.
    ``blocked_buy`` / ``blocked_sell`` are the engine's tradability masks
    (see :func:`simulate_basket_daily_returns_batch`); ``universe_mask``
    is the mandatory ``[stock, date]`` PIT eligibility mask consumed at
    the signal date AND the entry date (T1-04); ``tie_break_keys``
    resolve exact selection ties deterministically (T1-02); ``adv``
    enables the capacity audit.
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
    simulation = simulate_basket_daily_returns(
        signal,
        target_ret,
        bt_cfg,
        blocked_buy,
        blocked_sell,
        signal_range,
        universe_mask=universe_mask,
        tie_break_keys=tie_break_keys,
        adv=adv,
    )
    benchmark = _window_benchmark(target_ret, signal_range, universe_mask)
    objectives = _portfolio_objectives(
        simulation, benchmark, reward_cfg.ic_hac_max_lags
    )
    annualized_cost = (
        float(simulation.daily_cost_fractions.mean()) * _ANNUALIZATION
        if simulation.daily_cost_fractions.size
        else 0.0
    )
    raw = float(objectives[0, 0]) - reward_cfg.cost_weight * annualized_cost
    return float(np.clip(raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high))


def batched_basket_rewards(
    signals: np.ndarray,
    target_ret: np.ndarray,
    bt_cfg: BacktestConfig,
    reward_cfg: RewardConfig,
    val_windows: list[tuple[int, int]] | None = None,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    train_signal_range: tuple[int, int] | None = None,
    *,
    universe_mask: np.ndarray,
    tie_break_keys: np.ndarray | None = None,
    adv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None,
           np.ndarray | None]:
    """v13 rewards for a batch: active IR minus exact annualized daily
    costs, with raw ICIR values and portfolio objectives exposed.

    ``signals`` is ``[B, stocks, dates]``.  Returns
    ``(rewards [B], val_rewards [B] | None, icir [B], val_icir [B] | None,
    objectives [B, 8] | None)``: the training-window reward is clipped
    annualized active IR (gross basket returns minus the equal-weight
    universe benchmark, effective-n shrunk) minus the proportional
    turnover-cost drag, computed on ``train_signal_range`` — the caller's
    *learning* window, which for the trainer is the in-sample window
    ending where the validation tail begins (the policy gradient must
    never score on the selection data); with ``val_windows`` (column index
    pairs, half-open) the validation reward is the **median** over the
    windows of the same quantity computed on each window independently
    (each window's basket restarts from zero weights, mirroring a fresh
    out-of-sample deployment), and ``val_icir`` is the median window ICIR
    (the auxiliary metric).  Without ``val_windows`` the second and fourth
    results are ``None``.

    ``objectives`` columns (T1-04) are the portfolio objectives of the
    train window followed by the validation-window medians:
    ``[train_active_ir, train_exposure, train_turnover, train_capacity,
    val_active_ir, val_exposure, val_turnover, val_capacity]``; capacity
    is NaN when ``adv`` (``[stocks, dates]`` dollar volume) is not
    provided.  ``blocked_buy`` / ``blocked_sell`` are ``[stocks, dates]``
    tradability masks shared by all rows (per window they are sliced
    exactly like the signals).  ``universe_mask`` is the mandatory
    ``[stocks, dates]`` PIT eligibility mask, sliced per window exactly
    like the signals, consumed at the signal date AND the entry date by
    the basket (T1-04 alignment).
    """

    signals = np.asarray(signals, dtype=np.float64)
    target_ret = np.asarray(target_ret, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if universe_mask.shape != target_ret.shape:
        raise ValueError(
            f"universe_mask shape {universe_mask.shape} does not match "
            f"target_ret shape {target_ret.shape}"
        )
    if train_signal_range is None:
        train_signal_range = (0, max(signals.shape[2] - 2, 0))
    train_start, train_end = train_signal_range
    icir = _robust_icir_batch(
        rank_ic_series(
            signals[:, :, train_start:train_end],
            target_ret[:, train_start:train_end],
            reward_cfg.ic_min_stocks,
            universe_mask=universe_mask[:, train_start:train_end],
        ),
        reward_cfg.ic_hac_max_lags,
    )
    simulation = simulate_basket_daily_returns_batch(
        signals,
        target_ret,
        bt_cfg,
        blocked_buy,
        blocked_sell,
        train_signal_range,
        universe_mask=universe_mask,
        tie_break_keys=tie_break_keys,
        adv=adv,
    )
    benchmark = _window_benchmark(target_ret, train_signal_range, universe_mask)
    objectives = _portfolio_objectives(
        simulation, benchmark, reward_cfg.ic_hac_max_lags
    )
    mean_cost = (
        simulation.daily_cost_fractions.mean(axis=1)
        if simulation.daily_cost_fractions.shape[1]
        else np.zeros(signals.shape[0], dtype=np.float64)
    )
    raw = objectives[:, 0] - reward_cfg.cost_weight * mean_cost * _ANNUALIZATION
    rewards = np.clip(raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high)

    val_rewards: np.ndarray | None = None
    val_icir: np.ndarray | None = None
    val_objectives: np.ndarray | None = None
    if val_windows:
        per_window = []
        per_window_icir = []
        per_window_objectives = []
        for start, end in val_windows:
            win_icir = _robust_icir_batch(
                rank_ic_series(
                    signals[:, :, start:end],
                    target_ret[:, start:end],
                    reward_cfg.ic_min_stocks,
                    universe_mask=universe_mask[:, start:end],
                ),
                reward_cfg.ic_hac_max_lags,
            )
            win_simulation = simulate_basket_daily_returns_batch(
                signals,
                target_ret,
                bt_cfg,
                blocked_buy,
                blocked_sell,
                (start, end),
                universe_mask=universe_mask,
                tie_break_keys=tie_break_keys,
                adv=adv,
            )
            win_objectives = _portfolio_objectives(
                win_simulation,
                _window_benchmark(target_ret, (start, end), universe_mask),
                reward_cfg.ic_hac_max_lags,
            )
            win_mean_cost = (
                win_simulation.daily_cost_fractions.mean(axis=1)
                if win_simulation.daily_cost_fractions.shape[1]
                else np.zeros(signals.shape[0], dtype=np.float64)
            )
            win_raw = (
                win_objectives[:, 0]
                - reward_cfg.cost_weight * win_mean_cost * _ANNUALIZATION
            )
            per_window.append(
                np.clip(win_raw, reward_cfg.reward_clip_low, reward_cfg.reward_clip_high)
            )
            per_window_icir.append(win_icir)
            per_window_objectives.append(win_objectives)
        val_rewards = np.median(np.stack(per_window, axis=1), axis=1)
        val_icir = np.median(np.stack(per_window_icir, axis=1), axis=1)
        val_objectives = np.median(
            np.stack(per_window_objectives, axis=1), axis=1
        )
    if val_objectives is None:
        objectives = np.concatenate(
            [objectives, np.full((signals.shape[0], 4), np.nan)], axis=1
        )
    else:
        objectives = np.concatenate([objectives, val_objectives], axis=1)
    return rewards, val_rewards, icir, val_icir, objectives


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
