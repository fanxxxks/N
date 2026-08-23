from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_data.config import BacktestConfig, SimConfig
from ashare_data.schemas import SimOrder
from ashare_trading.matching import SimBroker
from ashare_trading.portfolio import SimulationPortfolio


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
    return SimBroker(), SimulationPortfolio(config.initial_capital, config.state_path), config


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


def test_stop_signal_helpers_protocol(tmp_path: Path):
    from ashare_trading.signals import (
        clear_stop_signal,
        request_stop,
        stop_requested,
    )

    path = tmp_path / "STOP"
    assert stop_requested(path) is False  # absent -> keep running
    request_stop(path)
    assert path.read_text(encoding="utf-8") == "STOP"
    assert stop_requested(path) is True
    assert path.read_text(encoding="utf-8") == "STOPPED"  # acknowledged
    clear_stop_signal(path)
    assert not path.exists()
    # Garbage content is not a stop request.
    path.write_text("garbage", encoding="utf-8")
    assert stop_requested(path) is False


def test_portfolio_last_exec_date_roundtrip(tmp_path: Path):
    path = tmp_path / "state.json"
    portfolio = SimulationPortfolio(100000.0, path)
    portfolio.last_exec_date = "20240105"
    portfolio.record_equity("20240105", {})
    portfolio.save()

    loaded = SimulationPortfolio(100000.0, path)
    assert loaded.last_exec_date == "20240105"
    assert loaded.has_history


def test_portfolio_legacy_state_derives_last_exec_date(tmp_path: Path):
    # State files written before last_exec_date existed carry no watermark;
    # the equity history tail is the best available resume point.
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "initial_capital": 100000.0,
                "cash": 100000.0,
                "trade_count": 0,
                "positions": {},
                "equity_history": [{"trade_date": "20240103", "equity": 100000.0}],
            }
        ),
        encoding="utf-8",
    )
    loaded = SimulationPortfolio(100000.0, path)
    assert loaded.last_exec_date == "20240103"
    assert loaded.has_history


def test_portfolio_fresh_state_has_no_history(tmp_path: Path):
    portfolio = SimulationPortfolio(100000.0, tmp_path / "state.json")
    assert not portfolio.has_history
    portfolio.reset()
    assert not portfolio.has_history
    assert portfolio.last_exec_date is None


# --- ST status: display names never drive limits; st_codes is as-of only ---


def test_broker_limit_down_sell_skipped_keeps_position(tmp_path: Path):
    # A position whose exit day opens at the one-word limit-down board
    # cannot be sold: the order is skipped and the position is kept for
    # the next attempt (existing matching rule).
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    portfolio.add_buy("000001.SZ", "平安银行", 100, 10.0, "20240101")
    bars = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 0.9,
                "high": 0.9,
                "low": 0.9,
                "pre_close": 1.0,
                "volume": 100000.0,
            }
        ]
    )
    sell = SimOrder(
        order_id="o2",
        ts_code="000001.SZ",
        trade_date="20240102",
        side="sell",
        quantity=100,
        price=0.9,
    )
    trades = broker.execute_orders(
        [sell], bars, "20240102", portfolio, {"000001.SZ": "平安银行"}, BacktestConfig()
    )
    assert trades == []
    assert sell.status == "skipped"
    assert sell.reason == "limit_down"
    assert portfolio.positions["000001.SZ"].quantity == 100


def test_broker_display_name_does_not_apply_st_band(tmp_path: Path):
    # A current "*ST" name is display-only: without st_codes the 10% main
    # board band applies, so a +5.2% one-word open is buyable.
    broker, portfolio, _ = _broker_and_portfolio(tmp_path)
    bars = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 1.052,
                "high": 1.052,
                "low": 1.052,
                "pre_close": 1.0,
                "volume": 100000.0,
            }
        ]
    )
    order = SimOrder(
        order_id="o1",
        ts_code="000001.SZ",
        trade_date="20240102",
        side="buy",
        quantity=100,
        price=1.052,
    )
    trades = broker.execute_orders(
        [order],
        bars,
        "20240102",
        portfolio,
        {"000001.SZ": "*ST 星源"},
        BacktestConfig(),
    )
    assert len(trades) == 1
    assert order.status == "filled"
    assert portfolio.positions["000001.SZ"].name == "*ST 星源"


def test_broker_st_codes_apply_5pct_band_as_of_date(tmp_path: Path):
    # st_codes carries the current snapshot for same-day execution: the 5%
    # band then blocks a +5.2% one-word buy as limit_up and a -5.2%
    # one-word sell as limit_down.
    limit_up_bars = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 1.052,
                "high": 1.052,
                "low": 1.052,
                "pre_close": 1.0,
                "volume": 100000.0,
            }
        ]
    )
    limit_down_bars = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240102",
                "open": 0.948,
                "high": 0.948,
                "low": 0.948,
                "pre_close": 1.0,
                "volume": 100000.0,
            }
        ]
    )
    names = {"000001.SZ": "*ST 星源"}

    buy_broker, buy_portfolio, _ = _broker_and_portfolio(tmp_path / "buy")
    buy = SimOrder(
        order_id="o1",
        ts_code="000001.SZ",
        trade_date="20240102",
        side="buy",
        quantity=100,
        price=1.052,
    )
    buy_broker.execute_orders(
        [buy],
        limit_up_bars,
        "20240102",
        buy_portfolio,
        names,
        BacktestConfig(),
        st_codes={"000001.SZ"},
    )
    assert buy.status == "skipped"
    assert buy.reason == "limit_up"
    assert "000001.SZ" not in buy_portfolio.positions

    sell_broker, sell_portfolio, _ = _broker_and_portfolio(tmp_path / "sell")
    sell_portfolio.add_buy("000001.SZ", "*ST 星源", 100, 1.0, "20240101")
    sell = SimOrder(
        order_id="o2",
        ts_code="000001.SZ",
        trade_date="20240102",
        side="sell",
        quantity=100,
        price=0.948,
    )
    sell_broker.execute_orders(
        [sell],
        limit_down_bars,
        "20240102",
        sell_portfolio,
        names,
        BacktestConfig(),
        st_codes={"000001.SZ"},
    )
    assert sell.status == "skipped"
    assert sell.reason == "limit_down"
    assert sell_portfolio.positions["000001.SZ"].quantity == 100
