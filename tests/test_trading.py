from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ashare_data.config import BacktestConfig, SimConfig
from ashare_data.schemas import SimOrder
from ashare_trading.matching import SimBroker
from ashare_trading.portfolio import SimulationPortfolio
from ashare_trading.risk import (
    is_one_word_limit_down,
    is_one_word_limit_up,
    is_suspended,
    limit_rate,
)


def test_risk_helpers():
    assert limit_rate("000001.SZ") == 0.10
    assert limit_rate("300001.SZ") == 0.20
    assert limit_rate("000001.SZ", "ST 风险") == 0.05
    assert is_suspended(0.0, 100.0)
    assert is_suspended(10.0, 0.0)
    assert not is_suspended(10.0, 100.0)
    assert is_one_word_limit_up(1.1, 1.1, 1.1, 1.0, "000001.SZ")
    assert is_one_word_limit_down(0.9, 0.9, 0.9, 1.0, "000001.SZ")


def _sim_config(tmp_path: Path) -> SimConfig:
    return SimConfig(
        initial_capital=100000.0,
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
    )


def test_portfolio_buy_sell_and_mark_new_day(tmp_path: Path):
    portfolio = SimulationPortfolio(100000.0, tmp_path / "state.json")
    portfolio.add_buy("000001.SZ", "平安银行", 100, 10.0, "20240101")
    assert portfolio.cash == 99000.0
    assert portfolio.positions["000001.SZ"].quantity == 100
    assert portfolio.positions["000001.SZ"].available_quantity == 0

    portfolio.mark_new_day()
    portfolio.add_sell("000001.SZ", 30, 11.0, "20240102")
    assert portfolio.cash == 99000.0 + 330.0
    assert portfolio.positions["000001.SZ"].quantity == 70
    assert portfolio.positions["000001.SZ"].available_quantity == 70


def test_portfolio_save_load_and_market_value(tmp_path: Path):
    path = tmp_path / "state.json"
    portfolio = SimulationPortfolio(100000.0, path)
    portfolio.add_buy("000001.SZ", "平安银行", 10, 10.0, "20240101")
    portfolio.record_equity("20240101", {"000001.SZ": 10.5})
    portfolio.save()

    loaded = SimulationPortfolio(100000.0, path)
    assert loaded.positions["000001.SZ"].quantity == 10
    assert loaded.equity_history[-1]["equity"] == pytest.approx(100005.0)
    assert loaded.market_value({"000001.SZ": 11.0}) == 110.0


def _bars_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20240102", "open": 10.0, "high": 10.2, "low": 9.9, "pre_close": 9.9, "volume": 100000.0},
            {"ts_code": "600000.SH", "trade_date": "20240102", "open": 8.0, "high": 8.1, "low": 7.9, "pre_close": 8.0, "volume": 100000.0},
        ]
    )


def _broker_and_portfolio(tmp_path: Path):
    config = _sim_config(tmp_path)
    return SimBroker(config), SimulationPortfolio(config.initial_capital, config.state_path), config


def test_broker_no_orders_and_missing_bar(tmp_path: Path):
    broker, portfolio, config = _broker_and_portfolio(tmp_path)
    assert broker.execute_orders([], _bars_frame(), "20240102", portfolio, {}, BacktestConfig()) == []

    order = SimOrder(order_id="o1", ts_code="missing", trade_date="20240102", side="buy", quantity=100, price=10.0)
    trades = broker.execute_orders([order], _bars_frame(), "20240102", portfolio, {}, BacktestConfig())
    assert trades == []
    assert order.status == "skipped"
    assert order.reason == "missing_bar"


def test_broker_buy_fill_and_fees(tmp_path: Path):
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    bars = _bars_frame()
    order = SimOrder(order_id="o1", ts_code="000001.SZ", trade_date="20240102", side="buy", quantity=100, price=10.0)
    trades = broker.execute_orders([order], bars, "20240102", portfolio, {"000001.SZ": "平安银行"}, BacktestConfig())
    assert len(trades) == 1
    assert order.status == "filled"
    assert portfolio.positions["000001.SZ"].quantity > 0


def test_broker_sell_requires_available_position(tmp_path: Path):
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    portfolio.add_buy("000001.SZ", "平安银行", 100, 10.0, "20240101")
    bars = _bars_frame()
    sell = SimOrder(order_id="o2", ts_code="000001.SZ", trade_date="20240102", side="sell", quantity=50, price=10.0)
    trades = broker.execute_orders([sell], bars, "20240102", portfolio, {"000001.SZ": "平安银行"}, BacktestConfig())
    assert len(trades) == 1
    assert portfolio.positions["000001.SZ"].quantity == 50


def test_broker_limit_up_skip(tmp_path: Path):
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    bars = pd.DataFrame(
        [{"ts_code": "000001.SZ", "trade_date": "20240102", "open": 1.1, "high": 1.1, "low": 1.1, "pre_close": 1.0, "volume": 100000.0}]
    )
    order = SimOrder(order_id="o1", ts_code="000001.SZ", trade_date="20240102", side="buy", quantity=100, price=1.1)
    broker.execute_orders([order], bars, "20240102", portfolio, {"000001.SZ": "平安银行"}, BacktestConfig())
    assert order.status == "skipped"
    assert order.reason == "limit_up"


def test_broker_buy_uses_whole_lots_and_min_commission(tmp_path: Path):
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    bars = _bars_frame()
    # Enough cash for ~19 lots at price 10 with the 5-yuan minimum commission.
    portfolio.cash = 20000.0
    order = SimOrder(order_id="o1", ts_code="000001.SZ", trade_date="20240102", side="buy", quantity=2000, price=10.0)
    trades = broker.execute_orders([order], bars, "20240102", portfolio, {"000001.SZ": "平安银行"}, BacktestConfig())
    assert len(trades) == 1
    assert trades[0].quantity % 100 == 0
    assert trades[0].commission == pytest.approx(5.0)
    assert portfolio.cash >= 0.0


def test_broker_buy_odd_lot_skipped(tmp_path: Path):
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    bars = _bars_frame()
    order = SimOrder(order_id="o1", ts_code="000001.SZ", trade_date="20240102", side="buy", quantity=50, price=10.0)
    trades = broker.execute_orders([order], bars, "20240102", portfolio, {"000001.SZ": "平安银行"}, BacktestConfig())
    assert trades == []
    assert order.status == "skipped"


def test_broker_cash_never_goes_negative_with_min_fee(tmp_path: Path):
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    bars = _bars_frame()
    # 1005 yuan buys 100 shares at 10 but cannot cover the 5-yuan minimum
    # commission: affordability must include the fee floor and skip the order
    # instead of driving cash negative.
    portfolio.cash = 1005.0
    order = SimOrder(order_id="o1", ts_code="000001.SZ", trade_date="20240102", side="buy", quantity=1000, price=10.0)
    trades = broker.execute_orders([order], bars, "20240102", portfolio, {"000001.SZ": "平安银行"}, BacktestConfig())
    assert trades == []
    assert portfolio.cash >= 0.0


def test_portfolio_save_is_atomic(tmp_path: Path):
    portfolio = SimulationPortfolio(100000.0, tmp_path / "state.json")
    portfolio.add_buy("000001.SZ", "平安银行", 100, 10.0, "20240101")
    portfolio.save()
    assert (tmp_path / "state.json").exists()
    assert not (tmp_path / "state.tmp.json").exists()
    loaded = SimulationPortfolio(100000.0, tmp_path / "state.json")
    assert loaded.positions["000001.SZ"].quantity == 100
