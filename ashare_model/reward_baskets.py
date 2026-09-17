"""Basket simulation layer (mechanical split, t79): the long-only
top-n basket simulators (``BasketSimulation``, ``BatchBasketSimulation``,
``simulate_basket_daily_returns[_batch]``) — moved byte-identical from
``reward.py``, which keeps the metrics/reward assembly and a re-export
facade (single semantic path, AGENTS §9).  Zero semantic change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_execution import ExecutionCostModel
from ashare_portfolio.constructor import (
    PortfolioConstructor,
    canonical_weights_to_output_grid,
)

# P35: floor for the (1 - cost) drift divisor — the weight-space analogue
# of the engine's structural cost < 1 (its cash cap).  Only synthetic
# collapse fixtures ever reach it.
_P35_DRIFT_DIVISOR_FLOOR = 1e-12

@dataclass(frozen=True)
class BasketSimulation:
    """Exact scalar basket path for every complete t+2 signal period."""

    daily_gross_returns: np.ndarray
    daily_cost_fractions: np.ndarray
    daily_net_returns: np.ndarray
    turnover: np.ndarray
    target_weights: np.ndarray
    buy_weights: np.ndarray
    sell_weights: np.ndarray
    construction_diagnostics: tuple[dict, ...]
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
    target_weights: np.ndarray
    buy_weights: np.ndarray
    sell_weights: np.ndarray
    construction_diagnostics: tuple[tuple[dict, ...], ...]
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
    rebalance_mask: np.ndarray | None = None,
) -> BasketSimulation:
    """Exact basket path through the unified portfolio constructor.

    Target, buy and sell weights are the immutable outputs of
    :class:`PortfolioConstructor`, shared with backtest, golden parity and
    paper simulation.  The configured ranking buffer, construction method,
    trade filters and turnover budget therefore have one implementation.
    ``blocked_buy`` / ``blocked_sell`` carry the engine's
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
        rebalance_mask=rebalance_mask,
    )
    return BasketSimulation(
        daily_gross_returns=batch.daily_gross_returns[0],
        daily_cost_fractions=batch.daily_cost_fractions[0],
        daily_net_returns=batch.daily_net_returns[0],
        turnover=batch.turnover[0],
        target_weights=batch.target_weights[0],
        buy_weights=batch.buy_weights[0],
        sell_weights=batch.sell_weights[0],
        construction_diagnostics=batch.construction_diagnostics[0],
        capacity_utilization=(
            batch.capacity_utilization[0]
            if batch.capacity_utilization is not None
            else None
        ),
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
    tie_break_keys: np.ndarray | None = None,
    adv: np.ndarray | None = None,
    rebalance_mask: np.ndarray | None = None,
) -> BatchBasketSimulation:
    """Batched exact basket simulation over an explicit signal range.

    ``signals`` is ``[B, stocks, price_dates]``. By default only
    ``range(price_dates - 2)`` is executable; an explicit half-open
    ``signal_range`` is used for independently restarted validation windows.
    Weight construction is delegated to :class:`PortfolioConstructor`.
    This includes PIT eligibility, stable ranking, no-signal holds, ranking
    buffers, equal-weight or optimizer targets, blocking, minimum order,
    weight-change threshold and turnover-budget post-processing.

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
    target_weights_full = np.zeros((b, n_periods, n_stocks), dtype=np.float64)
    buy_weights_full = np.zeros_like(target_weights_full)
    sell_weights_full = np.zeros_like(target_weights_full)
    capacity_full = np.full((b, n_periods), np.nan, dtype=np.float64)
    diagnostics_full: list[list[dict]] = [[] for _ in range(b)]

    def empty_result() -> BatchBasketSimulation:
        return BatchBasketSimulation(
            gross_full,
            cost_full,
            net_full,
            turnover_full,
            target_weights_full,
            buy_weights_full,
            sell_weights_full,
            tuple(tuple(row) for row in diagnostics_full),
            capacity_utilization=(capacity_full if adv is not None else None),
        )

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
    if rebalance_mask is None:
        if cfg.rebalance_frequency != "daily":
            raise ValueError(
                "rebalance_mask is required for non-daily reward simulation"
            )
        rebalance_mask = np.ones(n_dates, dtype=bool)
    rebalance_mask = np.asarray(rebalance_mask, dtype=bool)
    if rebalance_mask.shape != (n_dates,):
        raise ValueError(
            f"rebalance_mask shape {rebalance_mask.shape} does not match "
            f"date axis ({n_dates},)"
        )

    prev = np.zeros((b, n_stocks))
    capital = np.full(b, float(cfg.initial_capital), dtype=np.float64)
    cost_model = ExecutionCostModel.from_config(cfg)
    constructor = PortfolioConstructor(cfg)
    # Non-finite target cells contribute zero; cleaned once outside the loop.
    target_clean = np.where(np.isfinite(target_ret), target_ret, 0.0)
    if tie_break_keys is None:
        tie_break_keys = np.asarray([f"row:{i:09d}" for i in range(n_stocks)])

    for output_col, t in enumerate(range(signal_start, signal_end)):
        exec_col = t + 1
        eligible = universe_mask[:, t] & universe_mask[:, exec_col]
        buy_block = (
            blocked_buy[:, exec_col]
            if blocked_buy is not None
            else np.zeros(n_stocks, dtype=bool)
        )
        sell_block = (
            blocked_sell[:, exec_col]
            if blocked_sell is not None
            else np.zeros(n_stocks, dtype=bool)
        )
        target = np.zeros_like(prev)
        buy = np.zeros_like(prev)
        sell = np.zeros_like(prev)
        for row in range(b):
            output = constructor.construct(
                signals[row, :, t],
                prev[row],
                capital=float(capital[row]),
                eligible=eligible,
                buy_blocked=buy_block,
                sell_blocked=sell_block,
                stable_keys=tie_break_keys,
                rebalance_due=bool(rebalance_mask[t]),
                adv=(adv[:, exec_col] if adv is not None else None),
            )
            target[row] = output.weights
            buy[row] = output.buy_weights
            sell[row] = output.sell_weights
            diagnostics_full[row].append(dict(output.diagnostics))
        cost = np.asarray(
            cost_model.rebalance_cost_fraction(buy, sell, capital),
            dtype=np.float64,
        )
        rets = target_clean[:, t][None, :]
        # P35: `target` is the post-trade book on due sessions and the
        # drifted book on non-due sessions (the constructor returns prev
        # unchanged there) — either way it is the book the session's
        # return applies to.  The session net is the exact equity ratio:
        # the fees debit at execution, then the post-fee book compounds
        # by the gross return ((1 - cost) * (1 + gross) - 1) — the same
        # exact form the p35 engine books.
        gross = (target * rets).sum(axis=1)
        net = (1.0 - cost) * (1.0 + gross) - 1.0
        turnover = buy.sum(axis=1) + sell.sum(axis=1)
        gross_full[:, output_col] = gross
        cost_full[:, output_col] = cost
        net_full[:, output_col] = net
        turnover_full[:, output_col] = turnover
        target_weights_full[:, output_col] = target
        buy_weights_full[:, output_col] = buy
        sell_weights_full[:, output_col] = sell
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
            # IP-10 01 (04-TC-07, itemized in
            # docs/test_runtime_measurement_log.md): days whose utilization
            # is all-NaN contribute NaN directly -- nanmax would emit
            # "All-NaN slice encountered" for exactly those rows while
            # returning the identical NaN, so the mask removes the warning
            # source without touching any value (fail-closed NaN stays
            # NaN; golden parity guards value identity).
            has_finite_util = np.isfinite(util).any(axis=1)
            day_capacity = np.full(util.shape[0], np.nan, dtype=np.float64)
            if has_finite_util.any():
                day_capacity[has_finite_util] = np.nanmax(
                    util[has_finite_util], axis=1
                )
            capacity_full[:, output_col] = day_capacity
        capital *= 1.0 + net
        # P35: drift the book to the next session (share-conserving state
        # evolution).  The engine debits the fees from CASH, so the stock
        # weights inflate by 1/(1 - cost) on due sessions (the book is
        # more fully invested after the fee drag); the drifted book then
        # moves by the per-stock adjusted returns.  On non-due sessions
        # target == prev and cost == 0, so the same formula is the drift.
        # cost >= 1 (fees at or beyond the whole book) is unreachable in
        # the engine, whose cash cap structurally keeps cost < 1; in the
        # weight-space basket it only occurs in synthetic collapse
        # fixtures (normal(0,1) targets crushing capital into the
        # min-commission floor).  Clamping the divisor to eps keeps the
        # drift non-negative there; the >1 scale-back below then restores
        # a valid long-only book (physical-region values are unchanged:
        # 1 - cost sits orders of magnitude above eps).
        with np.errstate(divide="ignore", invalid="ignore"):
            drifted = (
                target
                / np.maximum(1.0 - cost, _P35_DRIFT_DIVISOR_FLOOR)[:, None]
                * np.clip(1.0 + rets, 0.0, None)
                / np.clip(1.0 + gross, 0.0, None)[:, None]
            )
        # For physical returns the drifted sum stays bounded; unphysical
        # synthetic extremes (r < -100%) are scaled back to a valid
        # long-only book.
        total = drifted.sum(axis=1)
        scale = np.where(total > 1.0, 1.0 / np.where(total > 0, total, 1.0), 1.0)
        drifted = drifted * scale[:, None]
        # P35: hand the constructor its prev on the constructor output
        # grid (round-12 + sub-eps zeroing), or a not-due session
        # re-rounds into ~1e-13 phantom turnover dust — the same
        # projection the engine applies to its drifted book.
        prev = canonical_weights_to_output_grid(
            np.where(np.isfinite(drifted), drifted, 0.0)
        )

    return BatchBasketSimulation(
        gross_full,
        cost_full,
        net_full,
        turnover_full,
        target_weights_full,
        buy_weights_full,
        sell_weights_full,
        tuple(tuple(row) for row in diagnostics_full),
        capacity_utilization=(
            capacity_full if adv is not None else None
        ),
    )


