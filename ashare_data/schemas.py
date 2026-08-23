"""Shared dataclasses used across the A-share modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
