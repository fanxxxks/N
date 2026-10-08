"""Unified signal-to-target portfolio construction (P3-03/P3-04/P3-05)."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ashare_data.processor import has_cross_sectional_dispersion

from .optimizer import (
    PortfolioConstraints,
    PortfolioObjective,
    PortfolioOptimizer,
)

if TYPE_CHECKING:
    from ashare_data.config import BacktestConfig


# p44: bumped 1 -> 2 — a binding turnover_budget is allocated by
# the deterministic priority rule (exits by signal rank DESC,
# then entries by rank ASC, one leg at a time, skipped legs never
# dilute executed ones) instead of proportional scaling that
# spread the budget into sub-minimum fragments; v1 behaviour is
# kept verbatim as the versioned "proportional" comparison.
PORTFOLIO_CONSTRUCTOR_VERSION = 2
_WEIGHT_EPSILON = 1e-12


def canonical_weights_to_output_grid(values: np.ndarray) -> np.ndarray:
    """Project book weights onto the constructor's output grid.

    ``_output`` rounds its target to 12 decimals and zeroes sub-eps
    components; a caller that passes a drifted/achieved book back as the
    next ``prev`` must hand over the SAME projection, or a not-due
    session (target == prev) re-rounds into ~1e-13 phantom sell dust and
    reports nonzero turnover for a session that traded nothing.  Pure
    projection helper — zero constructor semantics change (p35 §3 keeps
    the constructor version frozen; consumers of the prev source are
    engine/basket side).
    """

    out = np.round(np.maximum(np.asarray(values, dtype=np.float64), 0.0), 12)
    out[np.abs(out) <= _WEIGHT_EPSILON] = 0.0
    return out


def _readonly(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out.setflags(write=False)
    return out


def _cap_increases_to_available_cash(
    target: np.ndarray,
    prev: np.ndarray,
    fee_reserve: float = 0.0,
) -> np.ndarray:
    """Shrink only fresh increases until the long-only book is funded.

    p52 (WP16-A, ``reports/wp16/FROZEN_A.md`` §1 (C1)/§6): ``fee_reserve``
    (a fraction of pre-trade equity) carves the session's execution fees
    out of the investable cash BEFORE the buys are sized, so the post-trade
    book satisfies ``Σa + c ≤ 1`` — fees are payable from the zero-yield
    cash account, never financed by a negative cash balance.  The engine
    analogue is its buy-affordability loop (``backtest.py:356-368``), which
    executes with its own cash ledger and therefore keeps the default 0.0:
    every legacy caller is byte-identical, and only the weight-space basket
    reward simulation (which has no execution layer) passes a reserve.
    """

    increase = np.maximum(target - prev, 0.0)
    base = target - increase
    available = max(
        1.0 - float(base.sum()) - max(float(fee_reserve), 0.0), 0.0
    )
    increase_sum = float(increase.sum())
    # Do not turn an exact minimum-sized order into a sub-minimum order
    # merely because summing twenty 0.05 weights yields 1.0000000000000002.
    if increase_sum > available + _WEIGHT_EPSILON and increase_sum > 0.0:
        return base + increase * (available / increase_sum)
    return target


def effective_ranks(config: "BacktestConfig") -> tuple[int, int]:
    """Resolve P3 ranks while preserving legacy ``top_n`` callers."""

    buy_rank = int(config.buy_rank if config.buy_rank is not None else config.top_n)
    sell_rank = int(
        config.sell_rank if config.sell_rank is not None else buy_rank
    )
    return buy_rank, sell_rank


def validate_portfolio_config(config: "BacktestConfig") -> None:
    method = str(config.portfolio_method)
    if method not in ("equal_weight", "optimizer"):
        raise ValueError(
            "portfolio_method must be 'equal_weight' or 'optimizer', "
            f"got {method!r}"
        )
    buy_rank, sell_rank = effective_ranks(config)
    if buy_rank < 1:
        raise ValueError(f"buy_rank must be >= 1, got {buy_rank}")
    if sell_rank < buy_rank:
        raise ValueError(
            f"sell_rank must be >= buy_rank, got {sell_rank} < {buy_rank}"
        )
    if not 0.0 < float(config.single_weight_cap) <= 1.0:
        raise ValueError("single_weight_cap must be in (0, 1]")
    if config.min_trade_amount is not None and float(config.min_trade_amount) <= 0.0:
        raise ValueError("min_trade_amount must be > 0")
    if config.turnover_budget is not None and float(config.turnover_budget) < 0.0:
        raise ValueError("turnover_budget must be >= 0")
    allocation = str(getattr(config, "turnover_allocation", "priority"))
    if allocation not in ("proportional", "priority", "trade_groups"):
        raise ValueError(
            "turnover_allocation must be 'proportional', 'priority' or "
            f"'trade_groups', got {allocation!r}"
        )
    threshold = float(config.target_weight_change_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("target_weight_change_threshold must be in [0, 1]")
    lot = getattr(config, "lot_size", 100)
    if isinstance(lot, bool) or not isinstance(lot, (int, np.integer)) or lot < 1:
        raise ValueError("lot_size must be a positive integer")
    if int(getattr(config, "lot_size", 100)) < 1:
        raise ValueError("lot_size must be >= 1")


@dataclass(frozen=True)
class PortfolioOutput:
    """Immutable target weights and auditable construction diagnostics."""

    weights: np.ndarray
    buy_weights: np.ndarray
    sell_weights: np.ndarray
    selected: tuple[int, ...]
    rebalanced: bool
    reason: str
    diagnostics: Mapping[str, Any]

    @property
    def turnover(self) -> float:
        return float(np.asarray(self.buy_weights).sum() + np.asarray(self.sell_weights).sum())

    @property
    def order_count(self) -> int:
        traded = np.asarray(self.buy_weights) + np.asarray(self.sell_weights)
        return int(np.count_nonzero(traded > _WEIGHT_EPSILON))


class PortfolioConstructor:
    """Construct one target book for either equal-weight or optimizer mode."""

    def __init__(
        self,
        config: "BacktestConfig",
        *,
        optimizer: PortfolioOptimizer | None = None,
    ) -> None:
        validate_portfolio_config(config)
        self.config = config
        self.buy_rank, self.sell_rank = effective_ranks(config)
        self.method = str(config.portfolio_method)
        if optimizer is not None and self.method != "optimizer":
            raise ValueError("optimizer may only be supplied for portfolio_method='optimizer'")
        self.optimizer = optimizer
        if self.method == "optimizer" and self.optimizer is None:
            self.optimizer = PortfolioOptimizer(
                PortfolioConstraints(single_weight_cap=float(config.single_weight_cap)),
                PortfolioObjective(risk_aversion=0.0),
            )

    @staticmethod
    def _validate_inputs(
        signal: np.ndarray,
        prev_weights: np.ndarray,
        eligible: np.ndarray,
        buy_blocked: np.ndarray,
        sell_blocked: np.ndarray,
        stable_keys: np.ndarray,
        capital: float,
    ) -> None:
        if signal.ndim != 1:
            raise ValueError("signal must be 1-D")
        shape = signal.shape
        for name, values in (
            ("prev_weights", prev_weights),
            ("eligible", eligible),
            ("buy_blocked", buy_blocked),
            ("sell_blocked", sell_blocked),
            ("stable_keys", stable_keys),
        ):
            if values.shape != shape:
                raise ValueError(f"{name} shape {values.shape} does not match signal {shape}")
        if not np.isfinite(prev_weights).all() or np.any(prev_weights < 0.0):
            raise ValueError("prev_weights must be finite and non-negative")
        if float(prev_weights.sum()) > 1.0 + 1e-9:
            raise ValueError("prev_weights must not exceed full investment")
        if not np.isfinite(float(capital)):
            raise ValueError("capital must be finite")
        if len(set(stable_keys.tolist())) != len(stable_keys):
            raise ValueError("stable_keys must be unique")

    def _output(
        self,
        target: np.ndarray,
        prev: np.ndarray,
        selected: tuple[int, ...],
        reason: str,
        diagnostics: dict[str, Any],
    ) -> PortfolioOutput:
        target = np.round(np.maximum(target, 0.0), 12)
        target[np.abs(target) <= _WEIGHT_EPSILON] = 0.0
        buy = np.maximum(target - prev, 0.0)
        sell = np.maximum(prev - target, 0.0)
        order_count = int(np.count_nonzero((buy + sell) > _WEIGHT_EPSILON))
        diagnostics.update(
            rebalance_executed=order_count > 0,
            order_count=order_count,
            turnover=float(buy.sum() + sell.sum()),
        )
        return PortfolioOutput(
            weights=_readonly(target),
            buy_weights=_readonly(buy),
            sell_weights=_readonly(sell),
            selected=selected,
            rebalanced=order_count > 0,
            reason=reason,
            diagnostics=MappingProxyType(dict(diagnostics)),
        )

    def construct(
        self,
        signal,
        prev_weights,
        *,
        capital: float,
        eligible,
        buy_blocked,
        sell_blocked,
        stable_keys,
        rebalance_due: bool = True,
        cov: np.ndarray | None = None,
        industries: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        size: np.ndarray | None = None,
        adv: np.ndarray | None = None,
        initial_funding_used: bool | None = None,
        fee_reserve: float = 0.0,
        not_due_forced_exit: bool = False,
        hold_after_first_funding: bool = False,
    ) -> PortfolioOutput:
        """Construct one target from only signal-date and entry-date inputs.

        ``initial_funding_used`` (p51 W4, frozen §6.2) is the run-level
        first-funding state maintained by the CALLER (engine / run_sim /
        reward baskets — one flag per account ledger).  ``None`` (the
        default) keeps the legacy semantics verbatim: an empty prev book
        counts as first funding.  An explicit bool opts into the state
        machine ``initial_funding := (not b) and prev.sum() <= eps`` —
        a burned exemption (or any earlier holding) never re-gains the
        unbudgeted privilege after the book is cleared.

        ``fee_reserve`` (p52, WP16-A FROZEN §1 (C1)/§6) is a fraction of
        pre-trade equity withheld from the cash cap so the session's
        execution fees stay payable from cash (``Σa + c ≤ 1``).  Default
        0.0 keeps the legacy cap byte-identical — the engine/run_sim pay
        fees inside their own share-space cash ledgers, so only the
        weight-space basket simulator passes a reserve.

        ``not_due_forced_exit`` / ``hold_after_first_funding`` (p56, WP18
        FROZEN_18_PLAN §3.2) are the opt-in hold-mode legs, both default
        False so every legacy path stays byte-identical:

        * ``not_due_forced_exit=True`` lets a NON-due session run its
          forced-risk-exit leg only (``target=prev`` with
          ``forced_exit`` names zeroed, reason
          ``"not_due_forced_exit"``).  Historically the not_due early
          return (below) preceded the ``forced_exit`` computation, so an
          ineligible holding was never exited on a non-due session —
          the WP18 hold mode closes that gap per the frozen semantics
          "保留必要的强制风险退出，仅关闭主动换股"; with the flag off
          the early return is unchanged.
        * ``hold_after_first_funding=True`` marks a DUE session of an
          already-funded hold account: active re-selection is OFF — the
          target is ``prev`` with ``forced_exit`` names zeroed (reason
          ``"hold_forced_exit"`` / ``"hold_no_trade"``); no buys are
          ever planned on this leg.  The run-level "already funded"
          state is the CALLER's (see :class:`HoldModeConstructor`).
        """

        if not isinstance(fee_reserve, (int, float, np.integer, np.floating)):
            raise TypeError("fee_reserve must be a real fraction of equity")
        if not np.isfinite(float(fee_reserve)) or float(fee_reserve) < 0.0:
            raise ValueError("fee_reserve must be finite and non-negative")

        signal = np.asarray(signal, dtype=np.float64)
        prev = np.asarray(prev_weights, dtype=np.float64)
        eligible = np.asarray(eligible, dtype=bool)
        buy_blocked = np.asarray(buy_blocked, dtype=bool)
        sell_blocked = np.asarray(sell_blocked, dtype=bool)
        stable_keys = np.asarray(stable_keys).astype(str)
        self._validate_inputs(
            signal,
            prev,
            eligible,
            buy_blocked,
            sell_blocked,
            stable_keys,
            capital,
        )
        if initial_funding_used is not None and not isinstance(
            initial_funding_used, (bool, np.bool_)
        ):
            raise TypeError(
                "initial_funding_used must be a bool or None (frozen §6.2)"
            )
        # p51 W4 (frozen §6.2): default None keeps the legacy line-272
        # rule byte-for-byte; an explicit bool switches to the run-level
        # state machine and echoes itself into diagnostics.
        if initial_funding_used is None:
            initial_funding = bool(prev.sum() <= _WEIGHT_EPSILON)
        else:
            initial_funding = (not bool(initial_funding_used)) and bool(
                prev.sum() <= _WEIGHT_EPSILON
            )
        diagnostics: dict[str, Any] = {
            "version": PORTFOLIO_CONSTRUCTOR_VERSION,
            "method": self.method,
            "rebalance_due": bool(rebalance_due),
            "buffer_survivors": (),
            "threshold_dropped": (),
            "min_trade_dropped": (),
            "turnover_budget_scale": 1.0,
            # p44 A3: the allocation diagnostics.  mode is set
            # below ("unbudgeted" | "proportional" | "priority", plus
            # the p51 experimental "trade_groups" delegation label);
            # the executed tuples keep their PRIORITY order (the
            # allocation order is the auditable fact).
            "turnover_budget_mode": None,
            "budget_exits_executed": (),
            "budget_buys_executed": (),
            "budget_skipped_no_viable": (),
            "budget_cash_capped": (),
            "turnover_budget_consumed": 0.0,
            "forced_exit_count": 0,
            "legacy_position_wind_down": (),
            "initial_funding": initial_funding,
        }
        if initial_funding_used is not None:
            # §6.2 diagnostics contract: echo the run-level state only
            # when the caller actually passed it (no new key otherwise).
            diagnostics["initial_funding_used"] = bool(initial_funding_used)
        notional_capital = max(float(capital), 0.0)
        # p56 (WP18 FROZEN §3.2): forced_exit is computed BEFORE the
        # not_due early return so the opt-in not_due_forced_exit /
        # hold_after_first_funding legs can act on it.  The formula is
        # verbatim the due-path one (below); with both flags off this
        # hoist changes no observable value (the due path recomputes the
        # identical mask and the default not_due return never reads it).
        held = prev > _WEIGHT_EPSILON
        forced_exit = held & ~eligible & ~sell_blocked
        if not rebalance_due:
            if not_due_forced_exit and bool(forced_exit.any()):
                # §3.2 row 7: non-due session forced-exit leg — sell the
                # disqualified holdings, no buys, proceeds stay cash.
                diagnostics["forced_exit_count"] = int(
                    np.count_nonzero(forced_exit)
                )
                diagnostics["hold_forced_exit"] = True
                target = prev.copy()
                target[forced_exit] = 0.0
                selected = tuple(
                    np.flatnonzero(target > _WEIGHT_EPSILON).astype(int)
                )
                return self._output(
                    target, prev, selected, "not_due_forced_exit", diagnostics
                )
            selected = tuple(np.flatnonzero(prev > _WEIGHT_EPSILON).astype(int))
            return self._output(prev, prev, selected, "not_due", diagnostics)

        if hold_after_first_funding:
            # §3.2 row 5: a due session of an already-funded hold account
            # closes active re-selection — target = prev with only the
            # forced-risk-exit names zeroed; no buy leg is ever planned.
            reason = (
                "hold_forced_exit" if bool(forced_exit.any()) else "hold_no_trade"
            )
            diagnostics["forced_exit_count"] = int(np.count_nonzero(forced_exit))
            if bool(forced_exit.any()):
                diagnostics["hold_forced_exit"] = True
            target = prev.copy()
            target[forced_exit] = 0.0
            selected = tuple(
                np.flatnonzero(target > _WEIGHT_EPSILON).astype(int)
            )
            return self._output(target, prev, selected, reason, diagnostics)

        finite_eligible = np.isfinite(signal) & eligible
        selectable = finite_eligible & ~buy_blocked
        selectable_values = signal[selectable]
        has_signal = has_cross_sectional_dispersion(selectable_values)

        rank = np.full(signal.shape, np.iinfo(np.int64).max, dtype=np.int64)
        ranked = np.asarray([], dtype=np.int64)
        if finite_eligible.any():
            candidates = np.flatnonzero(finite_eligible)
            order = np.lexsort((stable_keys[candidates], -signal[candidates]))
            ranked = candidates[order]
            rank[ranked] = np.arange(len(ranked), dtype=np.int64)

        ranked_selectable = np.asarray([], dtype=np.int64)
        if selectable.any():
            candidates = np.flatnonzero(selectable)
            order = np.lexsort((stable_keys[candidates], -signal[candidates]))
            ranked_selectable = candidates[order]

        forced_exit = held & ~eligible & ~sell_blocked
        mandatory = forced_exit.copy()
        # (forced_exit recomputed verbatim above the not_due early return
        # since p56/WP18 §3.2 — the mask is identical, one formula hoisted;
        # the assignment below is kept for the due path's readability)
        diagnostics["forced_exit_count"] = int(np.count_nonzero(forced_exit))

        survivors: list[int] = []
        selected_list: list[int] = []
        legacy_wind_down: tuple[int, ...] = ()
        if not has_signal:
            raw_target = prev.copy()
            raw_target[forced_exit] = 0.0
            selected = tuple(np.flatnonzero(raw_target > _WEIGHT_EPSILON).astype(int))
            reason = "no_signal"
        else:
            survivors = [
                int(index)
                for index in ranked
                if held[index] and rank[index] < self.sell_rank
            ]
            if len(survivors) > self.buy_rank:
                dropped = survivors[self.buy_rank :]
                survivors = survivors[: self.buy_rank]
                legacy_wind_down = tuple(dropped)
                mandatory[np.asarray(dropped, dtype=np.int64)] = True
            survivor_set = set(survivors)
            slots = self.buy_rank - len(survivors)
            entrants = [
                int(index)
                for index in ranked_selectable[: self.buy_rank]
                if slots > 0
                and index not in survivor_set
            ][:slots]
            selected_list = survivors + entrants
            selected = tuple(selected_list)
            diagnostics["buffer_survivors"] = tuple(survivors)
            diagnostics["legacy_position_wind_down"] = legacy_wind_down

            if self.method == "equal_weight":
                raw_target = np.zeros_like(prev)
                unit = min(
                    1.0 / float(self.buy_rank),
                    float(self.config.single_weight_cap),
                )
                if selected_list:
                    raw_target[np.asarray(selected_list, dtype=np.int64)] = unit
            else:
                alpha = np.full(signal.shape, np.nan, dtype=np.float64)
                if selected_list:
                    alpha[np.asarray(selected_list, dtype=np.int64)] = signal[
                        np.asarray(selected_list, dtype=np.int64)
                    ]
                lower = np.zeros_like(prev)
                if survivors:
                    survivor_indices = np.asarray(survivors, dtype=np.int64)
                    # A survivor's floor is its current weight, capped at
                    # the constraint: a DRIFTED prev can ride above the
                    # cap between executions (p35 §2), and the floor only
                    # ever protects held value, never a cap breach.  For
                    # constructor-produced prev (always <= cap) this clip
                    # is a no-op — zero semantic change.
                    lower[survivor_indices] = np.minimum(
                        prev[survivor_indices],
                        float(self.config.single_weight_cap),
                    )
                solution = self.optimizer.solve(
                    alpha,
                    prev,
                    capital=notional_capital,
                    cov=cov,
                    industries=industries,
                    beta=beta,
                    size=size,
                    adv=adv,
                    min_weights=lower,
                )
                raw_target = np.asarray(solution.weights, dtype=np.float64).copy()
                diagnostics["optimizer_status"] = solution.status
                diagnostics["optimizer_diagnostics"] = dict(solution.diagnostics)
            reason = "signal"

        # A blocked reduction remains at its previous weight. If this makes
        # the book too large, only fresh increases shrink; no name is ever
        # renormalized upward.
        blocked_reduction = sell_blocked & (raw_target < prev)
        raw_target[blocked_reduction] = prev[blocked_reduction]
        raw_target = _cap_increases_to_available_cash(raw_target, prev)

        target = raw_target.copy()
        discretionary = ~mandatory
        threshold_dropped: set[int] = set()
        min_trade_dropped: set[int] = set()

        threshold = float(self.config.target_weight_change_threshold)
        delta = target - prev
        if threshold > 0.0:
            below = discretionary & (np.abs(delta) > 0.0) & (np.abs(delta) < threshold)
            if below.any():
                threshold_dropped.update(np.flatnonzero(below).astype(int).tolist())
                target[below] = prev[below]

        minimum = self.config.min_trade_amount
        delta = target - prev
        if minimum is not None:
            below = (
                discretionary
                & (np.abs(delta) > 0.0)
                & (np.abs(delta) * notional_capital < float(minimum))
            )
            if below.any():
                min_trade_dropped.update(np.flatnonzero(below).astype(int).tolist())
                target[below] = prev[below]

        budget = self.config.turnover_budget
        allocation = str(getattr(self.config, "turnover_allocation", "priority"))
        budgeted = budget is not None and not diagnostics["initial_funding"]
        if allocation == "trade_groups":
            # p51 (frozen §1.1.2): the experimental trade-groups mode.
            # The constructor walks EXACTLY the unbudgeted code path
            # (threshold/min-trade filters and the cash cap unchanged)
            # and only labels the mode; its own priority/proportional
            # allocation never executes — the budget is delegated to the
            # pure planner (ashare_portfolio/trade_groups.py, §1.4),
            # which sees shares/prices/cash the weight-space constructor
            # does not.
            diagnostics["turnover_budget_mode"] = (
                "trade_groups" if budget is not None else "unbudgeted"
            )
        elif not budgeted:
            diagnostics["turnover_budget_mode"] = "unbudgeted"
        elif allocation == "proportional":
            # v1 comparison path (p44 A1): scale every discretionary
            # leg by budget / discretionary turnover, then re-apply
            # the minimum filter.
            diagnostics["turnover_budget_mode"] = "proportional"
            delta = target - prev
            discretionary_turnover = float(np.abs(delta[discretionary]).sum())
            if discretionary_turnover > float(budget) and discretionary_turnover > 0.0:
                scale = float(budget) / discretionary_turnover
                target[discretionary] = (
                    prev[discretionary] + delta[discretionary] * scale
                )
                diagnostics["turnover_budget_scale"] = scale
            # Scaling can turn a previously valid order into a sub-minimum one.
            delta = target - prev
            if minimum is not None:
                below = (
                    discretionary
                    & (np.abs(delta) > 0.0)
                    & (np.abs(delta) * notional_capital < float(minimum))
                )
                if below.any():
                    min_trade_dropped.update(np.flatnonzero(below).astype(int).tolist())
                    target[below] = prev[below]
        else:
            # p44 A2: the deterministic priority allocation.  The
            # budget funds the pre-fixed leg order one leg at a
            # time — exits first (signal rank DESC: the least
            # wanted leaves first and frees cash), then entries
            # (rank ASC: the most wanted enters first), ties by
            # stable_keys.  A leg that cannot form a minimum
            # order from the remaining budget is skipped WITHOUT
            # consuming budget or diluting the other legs; the
            # allocation only ever shrinks an existing planned
            # delta, so every fill traces back to the signal's
            # own target and the pre-fixed priority.
            diagnostics["turnover_budget_mode"] = "priority"
            delta = target - prev
            leg_mask = discretionary & (np.abs(delta) > 0.0)
            exits = sorted(
                np.flatnonzero(leg_mask & (delta < 0.0)).tolist(),
                key=lambda i: (-int(rank[i]), str(stable_keys[i])),
            )
            entries = sorted(
                np.flatnonzero(leg_mask & (delta > 0.0)).tolist(),
                key=lambda i: (int(rank[i]), str(stable_keys[i])),
            )
            remaining = float(budget)
            consumed = 0.0
            exits_executed: list[int] = []
            buys_executed: list[int] = []
            skipped_no_viable: list[int] = []
            cash_capped: list[int] = []
            # The allocator re-derives every leg from prev: legs are reset
            # to prev first (mandatory and non-leg names keep their
            # targets), so an entry's cash headroom reflects only the
            # exits already executed above it in the priority order.
            working = target.copy()
            working[leg_mask] = prev[leg_mask]
            for i in exits:
                want = float(prev[i] - target[i])
                alloc = min(want, remaining)
                if (
                    minimum is not None
                    and alloc * notional_capital < float(minimum)
                ):
                    skipped_no_viable.append(int(i))
                    working[i] = prev[i]
                    continue
                working[i] = prev[i] - alloc
                remaining -= alloc
                consumed += alloc
                exits_executed.append(int(i))
            for i in entries:
                want = float(target[i] - prev[i])
                headroom = max(1.0 - float(working.sum()), 0.0)
                alloc = min(want, remaining, headroom)
                if (
                    minimum is not None
                    and alloc * notional_capital < float(minimum)
                ):
                    skipped_no_viable.append(int(i))
                    working[i] = prev[i]
                    continue
                if headroom < want and headroom <= remaining:
                    cash_capped.append(int(i))
                working[i] = prev[i] + alloc
                remaining -= alloc
                consumed += alloc
                buys_executed.append(int(i))
            target = working
            diagnostics["budget_exits_executed"] = tuple(exits_executed)
            diagnostics["budget_buys_executed"] = tuple(buys_executed)
            diagnostics["budget_skipped_no_viable"] = tuple(
                sorted(skipped_no_viable)
            )
            diagnostics["budget_cash_capped"] = tuple(sorted(cash_capped))
            diagnostics["turnover_budget_consumed"] = float(consumed)

        # Both suppression passes can restore a planned reduction after fresh
        # buys were sized.  Re-apply the same cash boundary so the resulting
        # book remains self-financing; only increases may shrink.  p52
        # (WP16-A FROZEN §1 (C1)/§6): the fee reserve applies ONLY at this
        # final funding boundary — after the threshold / min-trade /
        # budget passes have sized the PLAN — mirroring the engine's
        # execution order (it plans at full size, then its affordability
        # loop funds the buys against cash-after-fees).  Reserving before
        # the min-trade filter would shrink every planned leg below the
        # minimum order and suppress the whole book instead of funding it.
        target = _cap_increases_to_available_cash(
            target, prev, fee_reserve=float(fee_reserve)
        )

        diagnostics["threshold_dropped"] = tuple(sorted(threshold_dropped))
        diagnostics["min_trade_dropped"] = tuple(sorted(min_trade_dropped))
        # p44 A3: budget skips are suppressed trades too (the
        # three-state funnel counts them with the same field).
        diagnostics["suppressed_trade_count"] = len(
            threshold_dropped
            | min_trade_dropped
            | set(diagnostics["budget_skipped_no_viable"])
        )
        return self._output(target, prev, selected, reason, diagnostics)


class HoldModeConstructor(PortfolioConstructor):
    """The WP18 hold-mode strategy entry (p56, FROZEN_18_PLAN §3.2).

    "首次建仓后持有" with the necessary forced risk exits kept: the
    wrapper owns the ONE piece of run-level state the weight-space
    constructor cannot see — whether the account's first funding has
    already happened — and routes every session accordingly:

    * before first funding: sessions are the base constructor verbatim
      (the first DUE session with an empty book builds normally, §3.2
      row 4; non-due pre-funding sessions are plain ``not_due`` — an
      empty book has nothing to exit);
    * after first funding: ``hold_after_first_funding=True`` on due
      sessions (active re-selection OFF, only the forced-exit leg) and
      ``not_due_forced_exit=True`` on non-due sessions (the exit-only
      leg) — so a hold account NEVER buys again, even after its book is
      fully liquidated by forced exits (§3.2: "hold 模式首建仓后永不
      主动买入", proceeds stay cash).

    One instance per account run (the engine holds its constructor for
    the whole ``run``): the first-funding flag is sequential session
    state, exactly like the engine's own weights/cash ledger.  The
    dynamic mode is the unwrapped base class — the two modes differ
    ONLY in the active-reselection leg and share the session grid,
    signal input, cost ladder and fee accounting (§3.2 header table).
    """

    def __init__(
        self,
        config: "BacktestConfig",
        *,
        optimizer: PortfolioOptimizer | None = None,
    ) -> None:
        super().__init__(config, optimizer=optimizer)
        self._first_funding_done: bool = False

    @property
    def first_funding_done(self) -> bool:
        return self._first_funding_done

    def construct(
        self,
        signal,
        prev_weights,
        *,
        capital: float,
        eligible,
        buy_blocked,
        sell_blocked,
        stable_keys,
        rebalance_due: bool = True,
        cov: np.ndarray | None = None,
        industries: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        size: np.ndarray | None = None,
        adv: np.ndarray | None = None,
        initial_funding_used: bool | None = None,
        fee_reserve: float = 0.0,
    ) -> PortfolioOutput:
        prev = np.asarray(prev_weights, dtype=np.float64)
        if not self._first_funding_done:
            output = super().construct(
                signal,
                prev,
                capital=capital,
                eligible=eligible,
                buy_blocked=buy_blocked,
                sell_blocked=sell_blocked,
                stable_keys=stable_keys,
                rebalance_due=rebalance_due,
                cov=cov,
                industries=industries,
                beta=beta,
                size=size,
                adv=adv,
                initial_funding_used=initial_funding_used,
                fee_reserve=fee_reserve,
            )
            if rebalance_due and float(np.asarray(output.weights).sum()) > (
                _WEIGHT_EPSILON
            ):
                self._first_funding_done = True
            return output
        return super().construct(
            signal,
            prev,
            capital=capital,
            eligible=eligible,
            buy_blocked=buy_blocked,
            sell_blocked=sell_blocked,
            stable_keys=stable_keys,
            rebalance_due=rebalance_due,
            cov=cov,
            industries=industries,
            beta=beta,
            size=size,
            adv=adv,
            initial_funding_used=initial_funding_used,
            fee_reserve=fee_reserve,
            not_due_forced_exit=True,
            hold_after_first_funding=True,
        )
