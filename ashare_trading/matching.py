"""Deterministic paper-trading order matcher."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from ashare_data.config import SimConfig
from ashare_data.schemas import SimOrder, SimTrade
from ashare_execution import ExecutionCostModel

from .portfolio import SimulationPortfolio
from .risk import is_one_word_limit_down, is_one_word_limit_up, is_suspended


class SimBroker:
    def __init__(self, config: SimConfig):
        self.config = config

    def execute_orders(
        self,
        orders: list[SimOrder],
        bars: pd.DataFrame,
        date: str,
        portfolio: SimulationPortfolio,
        stock_names: dict[str, str],
        backtest_config,
    ) -> list[SimTrade]:
        """Execute orders for one trading day at the day's open price."""

        if not orders:
            return []
        day_bars = bars[bars["trade_date"] == date]
        if day_bars.empty:
            return []

        bar_lookup = {
            row.ts_code: row for row in day_bars.itertuples(index=False)
        }
        trades: list[SimTrade] = []
        portfolio.mark_new_day()
        cost_model = ExecutionCostModel.from_config(backtest_config)

        for order in orders:
            if order.trade_date != date:
                continue
            bar = bar_lookup.get(order.ts_code)
            if bar is None:
                order.status = "skipped"
                order.reason = "missing_bar"
                continue
            if is_suspended(bar.open, bar.volume):
                order.status = "skipped"
                order.reason = "suspended"
                continue

            name = stock_names.get(order.ts_code, order.ts_code)
            price = float(bar.open)
            if order.side == "buy":
                if is_one_word_limit_up(
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.pre_close,
                    order.ts_code,
                    name,
                ):
                    order.status = "skipped"
                    order.reason = "limit_up"
                    continue
                shares = self._affordable_shares(
                    portfolio.cash,
                    price,
                    order.quantity,
                    cost_model,
                )
                if shares <= 0:
                    order.status = "skipped"
                    order.reason = "insufficient_cash"
                    continue
                amount = shares * price
                costs = cost_model.buy_cost(amount)
                commission = float(costs.commission)
                transfer = float(costs.transfer_fee)
                slippage = float(costs.slippage)
                fees = float(costs.total)
                portfolio.add_buy(order.ts_code, name, shares, price, date)
                portfolio.cash -= fees
                trade = self._make_trade(
                    order, shares, price, amount, commission, 0.0, transfer, slippage
                )
                order.quantity = shares
                order.status = "filled"
            else:
                pos = portfolio.positions.get(order.ts_code)
                if pos is None or pos.available_quantity <= 0:
                    order.status = "skipped"
                    order.reason = "no_position"
                    continue
                if is_one_word_limit_down(
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.pre_close,
                    order.ts_code,
                    name,
                ):
                    order.status = "skipped"
                    order.reason = "limit_down"
                    continue
                shares = min(order.quantity, pos.available_quantity)
                if shares <= 0:
                    order.status = "skipped"
                    order.reason = "no_available_position"
                    continue
                amount = shares * price
                costs = cost_model.sell_cost(amount)
                commission = float(costs.commission)
                stamp = float(costs.stamp_tax)
                transfer = float(costs.transfer_fee)
                slippage = float(costs.slippage)
                fees = float(costs.total)
                portfolio.add_sell(order.ts_code, shares, price, date)
                portfolio.cash -= fees
                trade = self._make_trade(
                    order,
                    shares,
                    price,
                    amount,
                    commission,
                    stamp,
                    transfer,
                    slippage,
                )
                order.quantity = shares
                order.status = "filled"

            trades.append(trade)

        # No save here: the runner persists the portfolio once per completed
        # day (orders/trades files + equity + last_exec_date in one atomic
        # snapshot). Saving mid-day would leave a half-processed day in the
        # state file that --resume cannot safely roll back.
        return trades

    def _affordable_shares(
        self,
        cash: float,
        price: float,
        requested: int,
        cost_model: ExecutionCostModel,
    ) -> int:
        return cost_model.affordable_shares(cash, price, requested, lot_size=100)

    def _make_trade(
        self,
        order: SimOrder,
        shares: int,
        price: float,
        amount: float,
        commission: float,
        stamp: float,
        transfer: float,
        slippage: float,
    ) -> SimTrade:
        return SimTrade(
            trade_id=f"{order.trade_date}-{order.order_id}-{order.side}",
            order_id=order.order_id,
            ts_code=order.ts_code,
            trade_date=order.trade_date,
            side=order.side,
            quantity=shares,
            price=price,
            amount=amount,
            commission=commission,
            stamp_tax=stamp,
            transfer_fee=transfer,
            slippage=slippage,
        )
