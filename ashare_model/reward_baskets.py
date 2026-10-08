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
    PortfolioOutput,
    canonical_weights_to_output_grid,
)
from ashare_portfolio.trade_groups import refuse_unwired_trade_groups

# P35: floor for the drift divisor — the weight-space analogue of the
# engine's structural cost < 1 (its cash cap).  Only synthetic collapse
# fixtures ever reach it.  p52 (WP16-A FROZEN §6): the divisor is now the
# post-trade equity growth 1 + g − c (§1 (3)); the floor keeps the same
# non-physical-guard role.
_P35_DRIFT_DIVISOR_FLOOR = 1e-12
# p52 (WP16-A, reports/wp16/FROZEN_A.md): the additive cash-fees identity
# §1 (C1): the post-trade book must satisfy Σa + c ≤ 1 + tol (fees are
# payable from the zero-yield cash account; a violating session is an
# unfundable fee-financing state that must be reserved away or refused,
# never washed by normalizing Σw' back to 1 — §3.4).  The tolerance is
# aligned with the canonical round-12 output grid (§6).
_P52_C1_TOLERANCE = 1e-12
# FROZEN_A §1 (C1) REFUSAL gate, errata-adjusted 2026-09-27 per FROZEN_A
# §8.3 (adjudicated WP18 contract ruling, append-only): the frozen 1e-12
# refusal threshold sat AT the noise floor of the canonical round-12
# weight grid it was meant to absorb (n=20 names x up-to-5e-13 rounding
# each => observed shortfalls of 1-2e-12 on the first slow-domain x guard
# encounter), while any REAL financing shortfall is bounded below by
# min_commission/capital = 5e-5 and the 2026-09-23 audit blocker was
# +9.93e-5.  The split is deliberate and minimal: _P52_C1_TOLERANCE above
# remains the BOOKING precision (fixed-point landing, every acceptance
# gate — all previously booked sessions stay byte-identical); ONLY the
# final (C1) refusal comparison uses this 1e-9 gate, admitting exactly
# the float-noise band (three orders above the grid noise, four-plus
# orders below the smallest physical fee fraction).  §4.1's relative
# 1e-12 identity-verification tolerance (channel 1) is a DIFFERENT
# constant and is untouched by this erratum.
_P52_C1_REFUSAL_TOLERANCE = 1e-9
# Fixed-point iterations for the constructor-side fee reservation (§6):
# each pass sizes buys against 1 − Σbase − reserve and re-measures the
# exact ExecutionCostModel fee on the resulting legs; the shortfall
# (Σa + c − 1) contracts geometrically (ratio ≈ the fee rate), so a
# handful of iterations reach the round-12 grid noise.
_P52_FEE_RESERVE_ITERATIONS = 8
# Grid-sticky nudge for the rare session whose shortfall oscillates inside
# the round-12 sum noise: one final over-reserving pass (the cash-positive
# direction is always safe).  Sized above n×5e-13 grid noise yet five
# orders below any physical fee fraction.
_P52_FEE_RESERVE_NUDGE = 1e-11

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
    # p52 (WP16-A FROZEN §1 (4)): the cash component carried into the
    # next session, w'_cash = (1 − Σa − c)/(1 + g − c), grid-consistent
    # with the drifted prev book (1 − Σ prev).
    daily_cash_weights: np.ndarray | None = None

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
    # p52 (WP16-A FROZEN §1 (4)): per-row cash component carried into the
    # next session (see BasketSimulation.daily_cash_weights).
    daily_cash_weights: np.ndarray | None = None

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
        daily_cash_weights=(
            batch.daily_cash_weights[0]
            if batch.daily_cash_weights is not None
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
    # Audit fix (2026-09-23, major): this reward simulation books costs on
    # the constructor's weight-space buy/sell matrices and does not consume
    # plan_trade_session/execute_trade_session, so the experimental
    # trade_groups mode (whose budget lives in the planner, frozen §1.1.2)
    # must REFUSE to run here instead of scoring rewards on unbudgeted
    # books.  The scalar entry delegates here, so both surfaces are
    # covered.  No-op for every other allocation mode (red line 2).
    refuse_unwired_trade_groups(
        cfg, "simulate_basket_daily_returns (reward baskets)"
    )
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
    cash_full = np.zeros((b, n_periods), dtype=np.float64)
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
            daily_cash_weights=cash_full,
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
    # p52 (WP16-A FROZEN §6): per-row physical-ledger flag — stays True
    # while the equity ratio 1+g−c > 0, which (C1) and r >= −1 guarantee
    # on physical inputs; it only trips on synthetic collapse fixtures
    # (returns beyond −100% wiping the equity), which keep the legacy
    # clamp/scale-back guards instead of the (C1) fail-closed refusal.
    ledger_physical = np.ones(b, dtype=bool)
    cost_model = ExecutionCostModel.from_config(cfg)
    constructor = PortfolioConstructor(cfg)
    # Non-finite target cells contribute zero; cleaned once outside the loop.
    target_clean = np.where(np.isfinite(target_ret), target_ret, 0.0)
    if tie_break_keys is None:
        tie_break_keys = np.asarray([f"row:{i:09d}" for i in range(n_stocks)])

    def _legs_cost_fraction(
        buy_weights: np.ndarray, sell_weights: np.ndarray, capital_row: float
    ) -> float:
        """Exact fee fraction of explicit leg arrays (ExecutionCostModel is
        the sole fee authority, red line 3)."""

        arr = np.asarray(
            cost_model.rebalance_cost_fraction(
                np.asarray(buy_weights, dtype=np.float64)[None, :],
                np.asarray(sell_weights, dtype=np.float64)[None, :],
                np.asarray([capital_row], dtype=np.float64),
            ),
            dtype=np.float64,
        )
        return float(arr.reshape(-1)[0])

    def _session_cost_fraction(output, capital_row: float) -> float:
        """Exact fee fraction of one row's legs (ExecutionCostModel is the
        sole fee authority, red line 3)."""

        return _legs_cost_fraction(
            output.buy_weights, output.sell_weights, capital_row
        )

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
        cost = np.zeros(b, dtype=np.float64)
        for row in range(b):
            # p52 (WP16-A FROZEN §1 (C1)/§3.4/§6): the session's buys are
            # sized only against cash left after the fees — the constructor
            # reserves a fee fraction out of the investable cash, and the
            # reserve is refined toward the fixed point Σa + c = 1 (Newton
            # in the reserve; the shortfall shrinks geometrically with the
            # fee rate).  A session that is feasible on the first pass
            # (partial book / zero fees / not due) breaks immediately with
            # reserve 0.0, byte-identical to the legacy construction.  The
            # per-order minimum-commission floor makes the fee a step
            # function of the reserve (a leg shrinking under the floor
            # stops responding), so the refinement can enter a limit cycle
            # around the feasible point; the tightest feasible construction
            # seen is then kept — the weight-space analogue of the engine's
            # affordability loop skipping an unfundable buy.  A session
            # that stays infeasible on a PHYSICAL ledger (equity never
            # wiped: with (C1) and r >= -1 the equity ratio 1+g-c >= 0
            # always) is REFUSED fail-closed — no daily values are
            # produced, and the negative cash is never washed away by
            # normalizing Σw' back to 1 (§3.4).  Synthetic collapse
            # fixtures (returns beyond -100% wiping the equity) keep the
            # legacy clamp/scale-back guards (§6, "非物理合成极端").
            reserve = 0.0
            shortfall = 0.0
            output = None
            cost_row = 0.0
            feasible_out = None
            feasible_cost = 0.0
            feasible_shortfall = -np.inf
            broke = False
            for attempt in range(_P52_FEE_RESERVE_ITERATIONS):
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
                    fee_reserve=reserve,
                )
                cost_row = _session_cost_fraction(output, float(capital[row]))
                shortfall = float(np.sum(output.weights)) + cost_row - 1.0
                if shortfall <= _P52_C1_TOLERANCE:
                    if shortfall > feasible_shortfall:
                        feasible_out = output
                        feasible_cost = cost_row
                        feasible_shortfall = shortfall
                    if attempt == 0 or abs(shortfall) <= _P52_C1_TOLERANCE:
                        broke = True
                        break
                next_reserve = max(reserve + shortfall, 0.0)
                if next_reserve == reserve:
                    broke = True
                    break
                reserve = next_reserve
            if not broke:
                # Grid-sticky or floor-cycled without landing inside the
                # tolerance window: one over-reserving nudge (the
                # cash-positive direction; sized above the round-12 sum
                # noise, far below any physical fee fraction).
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
                    fee_reserve=reserve + max(shortfall, 0.0)
                    + _P52_FEE_RESERVE_NUDGE,
                )
                cost_row = _session_cost_fraction(output, float(capital[row]))
                shortfall = float(np.sum(output.weights)) + cost_row - 1.0
            if shortfall > _P52_C1_TOLERANCE and feasible_out is not None:
                # Limit cycle around the floor discontinuity: book the
                # tightest feasible construction (buys skipped to fund the
                # fees, exactly like the engine's affordability loop).
                output = feasible_out
                cost_row = feasible_cost
                shortfall = feasible_shortfall
            booked = output.weights
            booked_diag: dict = dict(output.diagnostics)
            if shortfall > _P52_C1_TOLERANCE and ledger_physical[row]:
                # Audit fix (2026-09-23, blocker), route 2 of §3.4 for the
                # SELL side: the old code exempted buyless sessions from
                # the (C1) refusal on the claim "sells add cash net of
                # their fees" — false under the per-order minimum
                # commission floor (ashare_execution.py:116-120,
                # max(rate·notional, min_commission)): a dust sell whose
                # notional is below min_commission/rate frees less cash
                # than its own floor fee, so a fully invested book with
                # ZERO buys can end the session at Σa + c > 1.  The
                # constructor's fee reserve only shrinks INCREASES
                # (constructor.py _cap_increases_to_available_cash), so it
                # cannot fund a sell-side shortfall.  Remediation, applied
                # before any refusal: DEFER the floor-dominated sell legs
                # (book them back at their prev weight; their exit retries
                # next session) smallest notional first, keeping a drop
                # only while it actually reduces the shortfall, until the
                # book funds its own fees.  This is the weight-space
                # analogue of the engine's buy-affordability loop skipping
                # an unfundable order (backtest.py:356-368 "fund nothing
                # rather than go negative"), on the sell side; the engine
                # itself executes the dust exit and books the negative
                # cash exactly in its own ledger, which the basket must
                # NOT mirror (FROZEN_A §1 (C1) bars silently booked
                # negative cash).
                amended = np.array(output.weights, dtype=np.float64)
                prev_row = prev[row]
                dropped: list[int] = []
                while shortfall > _P52_C1_TOLERANCE:
                    sells_now = np.maximum(prev_row - amended, 0.0)
                    candidates = np.flatnonzero(sells_now > 0.0)
                    if candidates.size == 0:
                        break
                    leg = int(candidates[int(np.argmin(sells_now[candidates]))])
                    before = shortfall
                    amended[leg] = float(prev_row[leg])
                    sells_now = np.maximum(prev_row - amended, 0.0)
                    buys_now = np.maximum(amended - prev_row, 0.0)
                    cost_row = _legs_cost_fraction(
                        buys_now, sells_now, float(capital[row])
                    )
                    shortfall = float(amended.sum()) + cost_row - 1.0
                    if shortfall >= before:
                        # Dropping the smallest sell no longer helps (the
                        # remaining legs are not floor-dominated): revert
                        # and fall through to the fail-closed refusal.
                        amended[leg] = float(output.weights[leg])
                        sells_now = np.maximum(prev_row - amended, 0.0)
                        buys_now = np.maximum(amended - prev_row, 0.0)
                        cost_row = _legs_cost_fraction(
                            buys_now, sells_now, float(capital[row])
                        )
                        shortfall = float(amended.sum()) + cost_row - 1.0
                        break
                    dropped.append(leg)
                if shortfall <= _P52_C1_TOLERANCE:
                    booked = canonical_weights_to_output_grid(amended)
                    booked_diag["sell_fee_funding_dropped"] = tuple(
                        sorted(dropped)
                    )
                    booked_diag["sell_fee_funding"] = (
                        "deferred floor-dominated sell legs so the session "
                        "funds its execution fees from cash (FROZEN_A "
                        "§1 (C1)/§3.4 route 2; engine analogue: the "
                        "buy-affordability loop of backtest.py:356-368)"
                    )
                    output = PortfolioOutput(
                        weights=booked,
                        buy_weights=np.maximum(booked - prev_row, 0.0),
                        sell_weights=np.maximum(prev_row - booked, 0.0),
                        selected=output.selected,
                        rebalanced=output.rebalanced,
                        reason=output.reason,
                        diagnostics=output.diagnostics,
                    )
            if (
                shortfall > _P52_C1_REFUSAL_TOLERANCE
                and ledger_physical[row]
            ):
                # Audit fix (2026-09-23, blocker): the (C1) refusal must
                # NOT be gated on the presence of buys — a buyless session
                # can violate (C1) through floor-dominated dust sells, and
                # after the sell-fee funding above any REMAINING violation
                # on a physical ledger is an unfundable fee-financing
                # state (§3.4 route 1: refuse, no daily values; never wash
                # it by normalizing Σw' back to 1).  Errata 2026-09-27
                # (FROZEN_A §8.3): the refusal GATE is
                # _P52_C1_REFUSAL_TOLERANCE (1e-9) — the 1-2e-12 round-12
                # grid-noise band books instead of refusing, while every
                # physical shortfall (>= 5e-5 scale) still refuses.
                # p57 (errata E8 / FROZEN_A §8.4, 2026-09-27): the printed
                # threshold below discloses the OPERATIVE refusal gate —
                # 1 + _P52_C1_REFUSAL_TOLERANCE — not the booking-precision
                # constant; wording-only, the comparison above is untouched.
                raise ValueError(
                    "simulate_basket_daily_returns (reward baskets): "
                    f"session t={t} row={row} cannot fund its execution "
                    "fees from cash: sum(target)+cost="
                    f"{float(np.sum(output.weights)) + cost_row!r} > "
                    f"{1.0 + _P52_C1_REFUSAL_TOLERANCE!r} (WP16-A FROZEN_A "
                    "§1 (C1)/§3.4 fee-financing shortfall; refusing to "
                    "book, no daily values are produced)"
                )
            target[row] = output.weights
            buy[row] = output.buy_weights
            sell[row] = output.sell_weights
            cost[row] = cost_row
            diagnostics_full[row].append(booked_diag)
        rets = target_clean[:, t][None, :]
        # p52 (WP16-A FROZEN §1 (2)/§2.2): the session net is the additive
        # cash-fees identity R_net = g − c — fees debit from the zero-yield
        # cash account at execution and the stock notionals are fixed at
        # a_i·E at the fill, so E' = E·(1+g−c) exactly.  The previous
        # multiplicative form (1−c)(1+g)−1 booked the fees as if they
        # financed proportionally from the whole book (§2.1's −c·g cross
        # term) and is superseded; the engine's own exact booking is §2.3
        # (its fees debit from cash, its achieved weights inflate by
        # 1/(1−c), netting to the same g − c).
        gross = (target * rets).sum(axis=1)
        net = gross - cost
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
        ledger_physical &= np.isfinite(1.0 + net) & ((1.0 + net) > 0.0)
        # p52 (WP16-A FROZEN §1 (3)/(4)): the book drifts on the post-trade
        # equity E·(1+g−c): w'_i = a_i(1+r_i)/(1+g−c), and the cash
        # component w'_cash = (1−Σa−c)/(1+g−c) rides along as the zero
        # residual (exposed as daily_cash_weights below, grid-consistent
        # with the gridded prev book).  With the fee reservation the book
        # already carries its fee in cash, so — unlike the engine, whose
        # achieved weights inflate by 1/(1−c) because its fees debit
        # inside the share-space ledger (§2.3) — there is no 1/(1−c)
        # factor here.  The divisor floor only guards synthetic collapse
        # fixtures (1+g−c at or below zero; physical fee schedules keep it
        # orders of magnitude above), and the >1 scale-back below only
        # restores a valid long-only book for unphysical synthetic
        # extremes (r < −100%): (C1) is already enforced upstream, so
        # neither can mask fee financing (§3.4/§6).
        with np.errstate(divide="ignore", invalid="ignore"):
            drifted = (
                target
                * np.clip(1.0 + rets, 0.0, None)
                / np.maximum(
                    1.0 + gross - cost, _P35_DRIFT_DIVISOR_FLOOR
                )[:, None]
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
        # §1 (4): the cash weight carried into the next session, kept
        # consistent with the gridded prev book (1 − Σ prev).
        cash_full[:, output_col] = 1.0 - prev.sum(axis=1)

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
        daily_cash_weights=cash_full,
    )


