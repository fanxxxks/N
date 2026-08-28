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


PORTFOLIO_CONSTRUCTOR_VERSION = 1
_WEIGHT_EPSILON = 1e-12


def _readonly(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out.setflags(write=False)
    return out


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
    threshold = float(config.target_weight_change_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("target_weight_change_threshold must be in [0, 1]")


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
        if not np.isfinite(float(capital)) or float(capital) <= 0.0:
            raise ValueError("capital must be a finite positive number")
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
            "forced_exit_count": 0,
            "legacy_position_wind_down": (),
            "initial_funding": bool(prev.sum() <= _WEIGHT_EPSILON),
        }
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
                for index in ranked
                if slots > 0
                and rank[index] < self.buy_rank
                and selectable[index]
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
                    lower[survivor_indices] = prev[survivor_indices]
                solution = self.optimizer.solve(
                    alpha,
                    prev,
                    capital=float(capital),
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
        increase = np.maximum(raw_target - prev, 0.0)
        base = raw_target - increase
        available = max(1.0 - float(base.sum()), 0.0)
        increase_sum = float(increase.sum())
        if increase_sum > available and increase_sum > 0.0:
            raw_target = base + increase * (available / increase_sum)

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
                & (np.abs(delta) * float(capital) < float(minimum))
            )
            if below.any():
                min_trade_dropped.update(np.flatnonzero(below).astype(int).tolist())
                target[below] = prev[below]

        budget = self.config.turnover_budget
        if budget is not None and not diagnostics["initial_funding"]:
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
                & (np.abs(delta) * float(capital) < float(minimum))
            )
            if below.any():
                min_trade_dropped.update(np.flatnonzero(below).astype(int).tolist())
                target[below] = prev[below]

        diagnostics["threshold_dropped"] = tuple(sorted(threshold_dropped))
        diagnostics["min_trade_dropped"] = tuple(sorted(min_trade_dropped))
        diagnostics["suppressed_trade_count"] = len(
            threshold_dropped | min_trade_dropped
        )
        return self._output(target, prev, selected, reason, diagnostics)
