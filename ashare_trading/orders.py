"""Order construction from target weights (T3-02 unified execution spec).

One implementation of the weight -> order rule shared by the paper-trading
runner (``ashare_trading.run_sim``) and the golden execution spec
(``ashare_portfolio.golden``): whole-lot buys, sells first so their
proceeds fund the day's buys, full-exit sells for held names outside the
target set, and a deterministic order-id scheme.

``lot_size > 0`` floors buys to whole lots (A-share rule); ``lot_size <= 0``
keeps continuous quantities for the golden spec's lot-free mode.
"""

from __future__ import annotations

import numpy as np

from ashare_data.schemas import SimOrder


def target_shares_from_weights(
    weights: np.ndarray,
    equity: float,
    prices: np.ndarray,
    *,
    lot_size: int = 100,
) -> dict[int, float]:
    """Target share count per name from a weight vector.

    ``weights`` is the ``[n]`` target weight vector, ``equity`` the account
    equity the weights are fractions of, ``prices`` the entry open prices.
    Names with weight <= 0 or a non-positive/non-finite price get no
    target.  With ``lot_size > 0`` the share count is floored to whole
    lots; a floor that reaches zero produces no target.
    """

    weights = np.asarray(weights, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    if weights.shape != prices.shape:
        raise ValueError("weights and prices must share one shape")
    out: dict[int, float] = {}
    for i in range(weights.shape[0]):
        weight = float(weights[i])
        price = float(prices[i])
        if weight <= 0.0 or not np.isfinite(price) or price <= 0.0:
            continue
        shares = float(equity) * weight / price
        if lot_size > 0:
            shares = (int(shares) // lot_size) * lot_size
        if shares > 0.0:
            out[i] = shares
    return out


def build_orders(
    exec_date: str,
    ts_codes: list[str],
    open_prices: np.ndarray,
    target_shares: dict[int, float],
    selected: list[int],
    current_quantities: dict[str, float],
    *,
    lot_size: int = 100,
) -> list[SimOrder]:
    """Build the day's order list from target share counts.

    Sells execute before buys (A-share sell proceeds are immediately
    reusable for buying); buys are floored to whole lots (``lot_size > 0``)
    and sub-lot adjustments are skipped; every held name outside the
    target set is sold in full.  ``selected`` supplies the deterministic
    iteration order of the target names (the runner passes the signal
    ranking, the golden spec passes index order).  Order ids embed a
    monotonically increasing counter in construction order.
    """

    open_prices = np.asarray(open_prices, dtype=np.float64)
    selected_codes = {ts_codes[i] for i in selected}
    counter = 0

    def _order(side: str, code: str, quantity: float, price: float) -> SimOrder:
        nonlocal counter
        order = SimOrder(
            order_id=f"{exec_date}-{code}-{side}-{counter}",
            ts_code=code,
            trade_date=exec_date,
            side=side,
            quantity=quantity,
            price=float(price),
        )
        counter += 1
        return order

    sells: list[SimOrder] = []
    buys: list[SimOrder] = []
    for i in selected:
        code = ts_codes[i]
        current = current_quantities.get(code, 0.0)
        target = target_shares.get(i, 0.0)
        delta = target - current
        if delta == 0.0:
            continue
        if delta > 0.0:
            buy_qty = (int(delta) // lot_size) * lot_size if lot_size > 0 else delta
            if buy_qty <= 0.0:
                continue
            buys.append(_order("buy", code, buy_qty, open_prices[i]))
        else:
            sells.append(_order("sell", code, abs(delta), open_prices[i]))

    for code, quantity in current_quantities.items():
        if code in selected_codes or code not in ts_codes:
            continue
        i = ts_codes.index(code)
        # Full-position sell target: the execution day is already T+1
        # relative to any previous-day buy, and the broker still caps the
        # fill by the T+1-available quantity.
        sells.append(_order("sell", code, quantity, open_prices[i]))

    return sells + buys
