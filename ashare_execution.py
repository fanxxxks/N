"""Shared A-share execution costs and deployment-config validation.

This module is deliberately outside the model and trading packages: reward
scoring, continuous-weight backtests and whole-lot paper matching all depend
on the same domain rules and must not grow private fee implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

from ashare_data.config import BacktestConfig, SimConfig


Number = float | np.ndarray


def _scalar_or_array(value: np.ndarray) -> Number:
    return float(value) if value.ndim == 0 else value


@dataclass(frozen=True)
class ExecutionCosts:
    """Fee components for one trade or a collection of trades.

    Fields are floats for scalar input and numpy arrays for vectorized input.
    ``total`` is stored explicitly so callers cannot accidentally omit a
    component when converting costs into cash or return fractions.
    """

    commission: Number
    stamp_tax: Number
    transfer_fee: Number
    slippage: Number
    total: Number

    def as_dict(self) -> dict[str, Number]:
        return {
            "commission": self.commission,
            "stamp_tax": self.stamp_tax,
            "transfer_fee": self.transfer_fee,
            "slippage": self.slippage,
            "total": self.total,
        }


class ExecutionCostModel:
    """Exact fee model shared by every execution path.

    Commission is floored independently for every non-zero order. Stamp tax
    applies to sells only; transfer fees and slippage apply to both sides.
    """

    def __init__(
        self,
        *,
        commission_rate: float,
        min_commission: float,
        stamp_tax_rate: float,
        transfer_fee_rate: float,
        slippage_rate: float,
    ) -> None:
        values = {
            "commission_rate": commission_rate,
            "min_commission": min_commission,
            "stamp_tax_rate": stamp_tax_rate,
            "transfer_fee_rate": transfer_fee_rate,
            "slippage_rate": slippage_rate,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        self.commission_rate = float(commission_rate)
        self.min_commission = float(min_commission)
        self.stamp_tax_rate = float(stamp_tax_rate)
        self.transfer_fee_rate = float(transfer_fee_rate)
        self.slippage_rate = float(slippage_rate)

    @classmethod
    def from_config(cls, config: BacktestConfig) -> "ExecutionCostModel":
        return cls(
            commission_rate=config.commission_rate,
            min_commission=config.min_commission,
            stamp_tax_rate=config.stamp_tax_rate,
            transfer_fee_rate=config.transfer_fee_rate,
            slippage_rate=config.slippage_rate,
        )

    @staticmethod
    def weights_to_notional(weights: Any, capital: Any) -> np.ndarray:
        """Convert position weights to yuan notionals with batch broadcasting."""

        weight_arr = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
        capital_arr = np.maximum(np.asarray(capital, dtype=np.float64), 0.0)
        if weight_arr.ndim > capital_arr.ndim:
            capital_arr = np.expand_dims(capital_arr, axis=-1)
        return weight_arr * capital_arr

    def trade_cost(
        self,
        side: Literal["buy", "sell"],
        notional: Any,
    ) -> ExecutionCosts:
        """Return exact components for scalar or vectorized yuan notionals."""

        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        amount = np.asarray(notional, dtype=np.float64)
        if np.any(~np.isfinite(amount)) or np.any(amount < 0):
            raise ValueError("notional must contain finite non-negative values")
        nonzero = amount > 0.0
        commission = np.where(
            nonzero,
            np.maximum(self.commission_rate * amount, self.min_commission),
            0.0,
        )
        stamp = (
            np.where(nonzero, self.stamp_tax_rate * amount, 0.0)
            if side == "sell"
            else np.zeros_like(amount)
        )
        transfer = np.where(nonzero, self.transfer_fee_rate * amount, 0.0)
        slippage = np.where(nonzero, self.slippage_rate * amount, 0.0)
        total = commission + stamp + transfer + slippage
        return ExecutionCosts(
            _scalar_or_array(commission),
            _scalar_or_array(stamp),
            _scalar_or_array(transfer),
            _scalar_or_array(slippage),
            _scalar_or_array(total),
        )

    def buy_cost(self, notional: Any) -> ExecutionCosts:
        return self.trade_cost("buy", notional)

    def sell_cost(self, notional: Any) -> ExecutionCosts:
        return self.trade_cost("sell", notional)

    def rebalance_cost(
        self,
        buy_weights: Any,
        sell_weights: Any,
        capital: Any,
    ) -> ExecutionCosts:
        """Aggregate exact costs of a continuous-weight rebalance.

        The last axis represents independently charged orders. Each weight is
        converted with that path's actual portfolio capital before applying
        the per-order minimum commission.
        """

        buy = np.asarray(buy_weights, dtype=np.float64)
        sell = np.asarray(sell_weights, dtype=np.float64)
        if buy.shape != sell.shape:
            raise ValueError("buy_weights and sell_weights must share one shape")
        buy_cost = self.buy_cost(self.weights_to_notional(buy, capital))
        sell_cost = self.sell_cost(self.weights_to_notional(sell, capital))

        def combined(name: str) -> Number:
            value = np.asarray(getattr(buy_cost, name), dtype=np.float64) + np.asarray(
                getattr(sell_cost, name), dtype=np.float64
            )
            if value.ndim:
                value = value.sum(axis=-1)
            return _scalar_or_array(np.asarray(value))

        return ExecutionCosts(
            commission=combined("commission"),
            stamp_tax=combined("stamp_tax"),
            transfer_fee=combined("transfer_fee"),
            slippage=combined("slippage"),
            total=combined("total"),
        )

    def rebalance_cost_fraction(
        self,
        buy_weights: Any,
        sell_weights: Any,
        capital: Any,
    ) -> Number:
        costs = self.rebalance_cost(buy_weights, sell_weights, capital)
        total = np.asarray(costs.total, dtype=np.float64)
        capital_arr = np.asarray(capital, dtype=np.float64)
        fraction = np.divide(
            total,
            capital_arr,
            out=np.zeros_like(total, dtype=np.float64),
            where=capital_arr > 0,
        )
        return _scalar_or_array(fraction)

    def affordable_shares(
        self,
        cash: float,
        price: float,
        requested: int | None = None,
        *,
        lot_size: int = 100,
    ) -> int:
        """Largest affordable whole-lot buy, including every buy-side fee."""

        cash = float(cash)
        price = float(price)
        if not math.isfinite(cash) or not math.isfinite(price):
            return 0
        if cash <= 0.0 or price <= 0.0 or lot_size <= 0:
            return 0
        max_lots = int(cash // (price * lot_size))
        if requested is not None:
            max_lots = min(max_lots, max(0, int(requested)) // lot_size)
        low, high = 0, max_lots
        while low < high:
            mid = (low + high + 1) // 2
            shares = mid * lot_size
            notional = shares * price
            payable = notional + float(self.buy_cost(notional).total)
            if payable <= cash + 1e-9:
                low = mid
            else:
                high = mid - 1
        return low * lot_size


def execution_config_mismatches(
    backtest: BacktestConfig,
    sim: SimConfig,
) -> dict[str, tuple[float | int, float | int]]:
    """Return deployment fields whose backtest and sim values disagree."""

    pairs = {
        "initial_capital": (backtest.initial_capital, sim.initial_capital),
        "top_n/max_positions": (backtest.top_n, sim.max_positions),
        "single_weight_cap": (
            backtest.single_weight_cap,
            sim.single_weight_cap,
        ),
    }
    mismatches: dict[str, tuple[float | int, float | int]] = {}
    for name, (backtest_value, sim_value) in pairs.items():
        if isinstance(backtest_value, float) or isinstance(sim_value, float):
            equal = math.isclose(
                float(backtest_value), float(sim_value), rel_tol=0.0, abs_tol=1e-12
            )
        else:
            equal = backtest_value == sim_value
        if not equal:
            mismatches[name] = (backtest_value, sim_value)
    return mismatches


def validate_execution_config(
    backtest: BacktestConfig,
    sim: SimConfig,
) -> None:
    """Reject a deployment whose backtest and paper-execution sizes drift."""

    mismatches = execution_config_mismatches(backtest, sim)
    if not mismatches:
        return
    details = ", ".join(
        f"{name}: backtest={left}, sim={right}"
        for name, (left, right) in mismatches.items()
    )
    raise ValueError(f"execution config mismatch: {details}")
