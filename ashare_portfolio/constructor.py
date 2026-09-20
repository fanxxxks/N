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
) -> np.ndarray:
    """Shrink only fresh increases until the long-only book is funded."""

    increase = np.maximum(target - prev, 0.0)
    base = target - increase
    available = max(1.0 - float(base.sum()), 0.0)
    increase_sum = float(increase.sum())
    if increase_sum > available and increase_sum > 0.0:
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
    if allocation not in ("proportional", "priority"):
        raise ValueError(
            "turnover_allocation must be 'proportional' or 'priority', "
            f"got {allocation!r}"
        )
    threshold = float(config.target_weight_change_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("target_weight_change_threshold must be in [0, 1]")
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
    ) -> PortfolioOutput:
        """Construct one target from only signal-date and entry-date inputs."""

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
        diagnostics: dict[str, Any] = {
            "version": PORTFOLIO_CONSTRUCTOR_VERSION,
            "method": self.method,
            "rebalance_due": bool(rebalance_due),
            "buffer_survivors": (),
            "threshold_dropped": (),
            "min_trade_dropped": (),
            "turnover_budget_scale": 1.0,
            # p44 A3: the allocation diagnostics.  mode is set
            # below ("unbudgeted" | "proportional" | "priority");
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
            "initial_funding": bool(prev.sum() <= _WEIGHT_EPSILON),
        }
        notional_capital = max(float(capital), 0.0)
        if not rebalance_due:
            selected = tuple(np.flatnonzero(prev > _WEIGHT_EPSILON).astype(int))
            return self._output(prev, prev, selected, "not_due", diagnostics)

        held = prev > _WEIGHT_EPSILON
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
        budgeted = budget is not None and not diagnostics["initial_funding"]
        if not budgeted:
            diagnostics["turnover_budget_mode"] = "unbudgeted"
        elif (
            str(getattr(self.config, "turnover_allocation", "priority"))
            == "proportional"
        ):
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
        # book remains self-financing; only increases may shrink.
        target = _cap_increases_to_available_cash(target, prev)

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
