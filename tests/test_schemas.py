from __future__ import annotations

from ashare_data.schemas import (
    BacktestResult,
    SimOrder,
    SimTrade,
)


def test_schema_defaults():
    order = SimOrder(order_id="o1", ts_code="000001.SZ", trade_date="20240101", side="buy", quantity=1, price=1.0)
    assert order.status == "pending"
    assert order.reason == ""

    result = BacktestResult(equity_curve=[1.0], dates=[], daily_returns=[], turnover=[])
    assert result.benchmark_equity is None
    assert result.metrics == {}

    trade = SimTrade(
        trade_id="t1",
        order_id="o1",
        ts_code="000001.SZ",
        trade_date="20240101",
        side="buy",
        quantity=1,
        price=1.0,
        amount=1.0,
        commission=0.0,
        stamp_tax=0.0,
        transfer_fee=0.0,
        slippage=0.0,
    )
    assert trade.order_id == "o1"
