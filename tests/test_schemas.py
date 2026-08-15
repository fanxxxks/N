from __future__ import annotations

import numpy as np

from ashare_data.schemas import (
    BacktestResult,
    DailyBar,
    FactorFrame,
    PortfolioPosition,
    SimOrder,
    SimTrade,
    StockMeta,
)


def test_schema_defaults():
    bar = DailyBar(
        ts_code="000001.SZ",
        trade_date="20240101",
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.05,
        pre_close=1.0,
        volume=100.0,
        amount=105.0,
    )
    assert bar.turnover_rate is None
    assert bar.adj_factor is None

    stock = StockMeta(ts_code="000001.SZ", name="平安银行")
    assert stock.is_st is False

    order = SimOrder(order_id="o1", ts_code="000001.SZ", trade_date="20240101", side="buy", quantity=1, price=1.0)
    assert order.status == "pending"
    assert order.reason == ""

    result = BacktestResult(equity_curve=[1.0], dates=[], daily_returns=[], turnover=[])
    assert result.benchmark_equity is None
    assert result.metrics == {}

    frame = FactorFrame(values=np.array([[1.0]]), dates=["d"], ts_codes=["A"], feature_names=["f"])
    assert frame.shape == (1, 1)

    pos = PortfolioPosition(ts_code="A", name="A", quantity=1, available_quantity=0, avg_cost=1.0, last_price=1.0, last_date="d")
    assert pos.available_quantity == 0
