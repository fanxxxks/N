"""Deterministic paper-trading order matcher."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from ashare_data.config import SimConfig
from ashare_data.schemas import SimOrder, SimTrade

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
                    backtest_config,
                )
                if shares <= 0:
                    order.status = "skipped"
                    order.reason = "insufficient_cash"
                    continue
                amount = shares * price
                commission = max(
                    backtest_config.min_commission,
                    backtest_config.commission_rate * amount,
                )
                transfer = backtest_config.transfer_fee_rate * amount
                slippage = backtest_config.slippage_rate * amount
                fees = commission + transfer + slippage
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
                commission = max(
                    backtest_config.min_commission,
                    backtest_config.commission_rate * amount,
                )
                stamp = backtest_config.stamp_tax_rate * amount
                transfer = backtest_config.transfer_fee_rate * amount
                slippage = backtest_config.slippage_rate * amount
                fees = commission + stamp + transfer + slippage
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

        portfolio.save()
        return trades

    def _affordable_shares(
        self,
        cash: float,
        price: float,
        requested: int,
        backtest_config,
    ) -> int:
        if price <= 0:
            return 0
        bc = backtest_config
        # Solve for the share count whose cost plus commission (with the
        # minimum fee floor), transfer fee and slippage fits the cash.
        per_share_flat = price * (1.0 + bc.slippage_rate + bc.transfer_fee_rate)
        per_share_full = price * (
            1.0 + bc.slippage_rate + bc.commission_rate + bc.transfer_fee_rate
        )
        max_shares = int(
            (cash - bc.min_commission) / (per_share_flat + 1e-9)
        )
        max_shares = min(max_shares, int(cash / (per_share_full + 1e-9)))
        # A-share buys must be whole lots of 100 shares.
        max_shares = (max(0, max_shares) // 100) * 100
        shares = (min(requested, max_shares) // 100) * 100
        return max(0, shares)

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
