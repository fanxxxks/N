"""Shared dataclasses used across the A-share modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DailyBar:
    """A single trading day for one stock."""

    ts_code: str
    trade_date: str
    open: float
    high: float
    low: float
    close: float
    pre_close: float
    volume: float
    amount: float
    turnover_rate: float | None = None
    adj_factor: float | None = None


@dataclass
class StockMeta:
    ts_code: str
    name: str
    industry: str | None = None
    list_date: str | None = None
    is_st: bool = False


@dataclass
class FactorFrame:
    """Cross-sectional factor tensor represented as a NumPy array."""

    values: Any
    dates: list[str]
    ts_codes: list[str]
    feature_names: list[str] = field(default_factory=list)

    @property
    def shape(self):
        return self.values.shape


@dataclass
class BacktestResult:
    equity_curve: list[float]
    dates: list[str]
    daily_returns: list[float]
    turnover: list[float]
    benchmark_equity: list[float] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    positions: list[dict[str, Any]] = field(default_factory=list)
    trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SimOrder:
    order_id: str
    ts_code: str
    trade_date: str
    side: str
    quantity: int
    price: float
    status: str = "pending"
    reason: str = ""


@dataclass
class SimTrade:
    trade_id: str
    order_id: str
    ts_code: str
    trade_date: str
    side: str
    quantity: int
    price: float
    amount: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    slippage: float


@dataclass
class PortfolioPosition:
    ts_code: str
    name: str
    quantity: int
    available_quantity: int
    avg_cost: float
    last_price: float
    last_date: str
