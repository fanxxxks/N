"""Shared dataclasses used across the A-share modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
    # T3-02: exact per-signal-period target weight vectors (never rounded
    # to the 6-decimal positions record).  The golden execution spec
    # replays these through the whole-lot matcher; None for legacy
    # producers that did not record them.
    target_weights: list[np.ndarray] | None = None
    # P3-06: exact constructor output and cost trail. Optional only for
    # one-way compatibility with legacy producers/readers.
    buy_weights: list[np.ndarray] | None = None
    sell_weights: list[np.ndarray] | None = None
    cost_fractions: list[float] | None = None
    construction_diagnostics: list[dict[str, Any]] | None = None


@dataclass
class SimOrder:
    order_id: str
    ts_code: str
    trade_date: str
    side: str
    quantity: float
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
    quantity: float
    price: float
    amount: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    slippage: float


@dataclass(frozen=True)
class TushareSnapshotRecord:
    """L0 raw-snapshot provenance for one ingest batch (p17 §4.2).

    ``raw_path`` is relative to the store's raw root
    (``data/raw/tushare/<endpoint>/<ingest_id>/part-*.parquet``);
    ``raw_payload_sha256`` is the SHA-256 of the snapshot file bytes.
    """

    ingest_id: str
    endpoint: str
    raw_path: str
    raw_payload_sha256: str
    row_count: int
    hit_row_limit: bool


@dataclass(frozen=True)
class TushareIngestResult:
    """Typed outcome of one L1 ingest batch (mirrors source_ingest_runs)."""

    ingest_id: str
    endpoint: str
    status: str  # 'succeeded' | 'failed'
    row_count: int
    raw_path: str = ""
    raw_payload_sha256: str = ""
    hit_row_limit: bool = False
    error_type: str | None = None
    error_message: str | None = None
