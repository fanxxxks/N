"""Deterministic paper-trading order matcher."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_data.processor import blocked_components
from ashare_data.schemas import SimOrder, SimTrade
from ashare_execution import ExecutionCostModel

from .portfolio import SimulationPortfolio


class SimBroker:
    def execute_orders(
        self,
        orders: list[SimOrder],
        bars: pd.DataFrame,
        date: str,
        portfolio: SimulationPortfolio,
        stock_names: dict[str, str],
        backtest_config,
        *,
        st_codes: set[str] | None = None,
    ) -> list[SimTrade]:
        """Execute orders for one trading day at the day's open price.

        ``stock_names`` is display-only (persisted as the position name).
        Limit detection is the shared processor rule with board rates plus
        ``st_codes`` — the stocks whose current ``stocks.is_st`` snapshot
        is valid as of ``date``, i.e. same-day execution only; historical
        replay passes None so current names never rewrite the past.
        """

        if not orders:
            return []
        day_bars = bars[bars["trade_date"] == date]
        if day_bars.empty:
            return []

        bar_lookup = {
            row.ts_code: row for row in day_bars.itertuples(index=False)
        }
        rows = list(day_bars.itertuples(index=False))
        codes = [row.ts_code for row in rows]
        columns = {
            name: np.asarray(
                [float(getattr(row, name)) for row in rows], dtype=np.float64
            )
            for name in ("open", "high", "low", "pre_close", "volume")
        }
        st_mask = (
            np.asarray([code in st_codes for code in codes], dtype=bool)
            if st_codes is not None
            else None
        )
        suspended, limit_buy = blocked_components(
            columns["open"],
            columns["high"],
            columns["low"],
            columns["pre_close"],
            columns["volume"],
            codes,
            "buy",
            st_mask=st_mask,
        )
        _, limit_sell = blocked_components(
            columns["open"],
            columns["high"],
            columns["low"],
            columns["pre_close"],
            columns["volume"],
            codes,
            "sell",
            st_mask=st_mask,
        )
        blocked = {
            code: (
                bool(suspended[i]),
                bool(limit_buy[i]),
                bool(limit_sell[i]),
            )
            for i, code in enumerate(codes)
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
            is_suspended_flag, limit_up_flag, limit_down_flag = blocked[
                order.ts_code
            ]
            if is_suspended_flag:
                order.status = "skipped"
                order.reason = "suspended"
                continue

            name = stock_names.get(order.ts_code, order.ts_code)
            price = float(bar.open)
            if order.side == "buy":
                if limit_up_flag:
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
                if limit_down_flag:
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
