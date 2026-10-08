"""Deterministic paper-trading order matcher."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_data.processor import exact_blocked_components
from ashare_data.schemas import SimOrder, SimTrade
from ashare_execution import ExecutionCostModel

from .portfolio import SimulationPortfolio


class SimBroker:
    _CONSTRAINT_COLUMNS = ("up_limit", "down_limit", "suspended")

    def execute_orders(
        self,
        orders: list[SimOrder],
        bars: pd.DataFrame,
        date: str,
        portfolio: SimulationPortfolio,
        stock_names: dict[str, str],
        backtest_config,
        *,
        lot_size: int = 100,
    ) -> list[SimTrade]:
        """Execute orders for one trading day at the day's open price.

        ``stock_names`` is display-only (persisted as the position name).
        Fill gating consumes the exact constraint columns the loader joins
        onto the bar frame (p19 §2.4): ``up_limit`` / ``down_limit`` from
        ``daily_price_limit`` and the ``suspended`` flag from
        ``daily_suspension`` — buy blocked ⟺ open >= up_limit, sell
        blocked ⟺ open <= down_limit, a suspension record or a missing
        limit record blocks the order (skip reasons ``suspended`` /
        ``constraint_missing``).  There is no approximate board-rate
        fallback and no use of current stock names.

        ``lot_size`` is the whole-lot size in shares.  ``lot_size <= 0``
        switches to the golden-spec continuous-share mode (T3-02): shares
        fill at the requested quantity without whole-lot flooring and
        without the cash-affordability cap — the vectorized engine's
        funding convention, where target weights are fully invested and
        fees are charged against returns.  This mode exists so the
        authoritative matcher can be run lot-free against the research
        engine; the paper-trading default (100) keeps real cash
        accounting.
        """

        if not orders:
            return []
        missing = [
            column
            for column in self._CONSTRAINT_COLUMNS
            if column not in bars.columns
        ]
        if missing:
            raise ValueError(
                "bars frame is missing the p19 exact-constraint columns "
                f"{missing}: the matcher gates fills on the "
                "daily_price_limit/daily_suspension tables (p19 §2.4); "
                "build the frame from AshareDataLoader.bars"
            )
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
        suspended_flags = np.asarray(
            [bool(getattr(row, "suspended")) for row in rows], dtype=bool
        )
        up_limits = np.asarray(
            [float(getattr(row, "up_limit")) for row in rows], dtype=np.float64
        )
        down_limits = np.asarray(
            [float(getattr(row, "down_limit")) for row in rows], dtype=np.float64
        )
        blocked_buy, missing_buy = exact_blocked_components(
            columns["open"], up_limits, down_limits, suspended_flags, "buy"
        )
        blocked_sell, missing_sell = exact_blocked_components(
            columns["open"], up_limits, down_limits, suspended_flags, "sell"
        )
        blocked = {
            code: (
                bool(blocked_buy[i]),
                bool(blocked_sell[i]),
                bool(missing_buy[i] or missing_sell[i]),
            )
            for i, code in enumerate(codes)
        }

        trades: list[SimTrade] = []
        portfolio.mark_new_day()
        cost_model = ExecutionCostModel.from_config(backtest_config)

        # Same deterministic cash order as the ordinary backtest engine.
        for order in sorted(orders, key=lambda item: (item.side == "buy", item.ts_code)):
            if order.trade_date != date:
                continue
            bar = bar_lookup.get(order.ts_code)
            if bar is None:
                order.status = "skipped"
                order.reason = "missing_bar"
                continue
            is_buy_blocked, is_sell_blocked, constraint_missing = blocked[
                order.ts_code
            ]
            if constraint_missing:
                # Fail-closed: no exact limit record for that stock/day
                # (p19 §2.4 — degraded, never an approximate fallback).
                order.status = "skipped"
                order.reason = "constraint_missing"
                continue
            if is_buy_blocked and is_sell_blocked:
                # A suspension record blocks both sides.
                order.status = "skipped"
                order.reason = "suspended"
                continue

            name = stock_names.get(order.ts_code, order.ts_code)
            price = float(bar.open)
            if order.side == "buy":
                if is_buy_blocked:
                    order.status = "skipped"
                    order.reason = "limit_up"
                    continue
                if lot_size > 0:
                    shares = self._affordable_shares(
                        portfolio.cash,
                        price,
                        order.quantity,
                        cost_model,
                        lot_size=lot_size,
                    )
                else:
                    # Golden-spec continuous mode: engine-aligned funding,
                    # no cash cap (fees are charged against returns).
                    shares = max(float(order.quantity), 0.0)
                if shares <= 0:
                    order.status = "skipped"
                    order.reason = "insufficient_cash"
                    continue
                amount = shares * price
                if (lot_size > 0 and backtest_config.min_trade_amount is not None
                        and amount < backtest_config.min_trade_amount):
                    order.status = "skipped"
                    order.reason = "below_min_trade_amount"
                    continue
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
                if is_sell_blocked:
                    order.status = "skipped"
                    order.reason = "limit_down"
                    continue
                shares = min(order.quantity, pos.available_quantity)
                if shares <= 0:
                    order.status = "skipped"
                    order.reason = "no_available_position"
                    continue
                amount = shares * price
                if (lot_size > 0 and shares < pos.quantity
                        and backtest_config.min_trade_amount is not None
                        and amount < backtest_config.min_trade_amount):
                    order.status = "skipped"
                    order.reason = "below_min_trade_amount"
                    continue
                costs = cost_model.sell_cost(amount)
                commission = float(costs.commission)
                stamp = float(costs.stamp_tax)
                transfer = float(costs.transfer_fee)
                slippage = float(costs.slippage)
                fees = float(costs.total)
                if lot_size > 0 and portfolio.cash + amount < fees:
                    order.status = "skipped"
                    order.reason = "insufficient_cash_for_fees"
                    continue
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
        requested: float,
        cost_model: ExecutionCostModel,
        *,
        lot_size: int = 100,
    ) -> int:
        if lot_size <= 0:
            raise ValueError("_affordable_shares requires lot_size > 0")
        return cost_model.affordable_shares(cash, price, requested, lot_size=lot_size)

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
