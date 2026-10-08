"""WP15 pure trade-group session planner (frozen §1 of the contract
``reports/wp15/FROZEN_EXPECTATIONS.md``).

``plan_trade_session`` is the deterministic allocator behind the
experimental ``turnover_allocation = "trade_groups"`` mode: it turns the
constructor's target weights into whole-lot, self-financing trade groups
under a session turnover budget, with forced risk exits handled on their
own budget-free channel (§1.3.1) and every fee delegated to the single
authority :class:`ExecutionCostModel` — one merged notionional per
(session, name, direction) unit sent to the authority exactly once,
never per leg or per lot (§1.5).

Purity (§1.1.3): no IO, no globals, no randomness; the same inputs
always yield the same plan.  Everything the planner needs — shares,
prices, cash, target weights, ranks, masks, budget, fee model — is an
explicit argument (§1.2).

The allocation rule (§1.3/§1.4, no degrees of freedom):
* M — forced risk exits: whole-position sells (odd lots included),
  budget-exempt; blocked or priceless names keep their position and are
  recorded with a cause (§1.3.1).
* D — target-deleveraging single-side sells capped by
  ``R = max(S - B_dem, 0)``: budget-exempt one-sided reductions when the
  target itself wants a smaller book (§1.3.2).
* R — ordinary replacement groups: the buy side takes the largest
  feasible whole-lot notional y* (budget + financing + demand + minimum
  constraints, §1.4 step 2) and the sell side attaches the MINIMAL
  whole-lot legs that finance it (§1.4 step 3); a cash-rich session
  forms pure-buy groups and a sell leg is never driven by the budget
  alone (§1.4 compliance note).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ashare_execution import ExecutionCostModel

# §1 (总则): cash-style comparisons carry an absolute tolerance aligned
# with affordable_shares (ashare_execution.py:221); budget comparisons
# are EXACT.  §1.7 reason codes are frozen string literals.
_CASH_TOL = 1e-9
_WEIGHT_EPS = 1e-12

SKIP_CODES = (
    "budget",
    "lot",
    "min_amount",
    "cash",
    "fees",
    "no_viable_counterpart",
)


@dataclass(frozen=True)
class TradeSessionPlan:
    """Auditable plan for one rebalance session (§1.8 diagnostics keys).

    ``trade_groups`` carries ONLY the R-class groups (§1.8); M/D trades
    are visible through ``forced_exit_executed`` / ``executed_orders``.
    ``turnover_budget_consumed`` sums R-class consumption only (§1.8).
    ``execution_failed_buys`` is always empty from the planner — it is
    the execution layer's contract surface (§1.1.4: a failed buy keeps
    the real sells and cash; no fabricated rollbacks, no compensation).
    """

    turnover_budget_mode: str = "trade_groups"
    forced_exit_count: int = 0
    forced_exit_executed: tuple[str, ...] = ()
    forced_exit_blocked: tuple[tuple[str, str], ...] = ()
    deleveraging_sell_notional: float = 0.0
    deleveraging_skipped: tuple[tuple[str, str], ...] = ()
    trade_groups: tuple[dict[str, Any], ...] = ()
    budget_unspent_reason: str | None = None
    leg_skip_reasons: tuple[tuple[str, str, str], ...] = ()
    session_fees: dict[str, float] = field(
        default_factory=lambda: {"sell": 0.0, "buy": 0.0, "total": 0.0}
    )
    turnover_budget_consumed: float = 0.0
    execution_failed_buys: tuple[str, ...] = ()
    # (key, side, shares, notional) in ledger order: M sells (sell
    # order), D sells (sell order), then per group its sells and buy.
    executed_orders: tuple[tuple[str, str, int, float], ...] = ()


class _Session:
    """Mutable working state of one planning session (planner-internal)."""

    def __init__(
        self,
        *,
        keys: list[str],
        shares: np.ndarray,
        prices: np.ndarray,
        cash: float,
        weights: np.ndarray,
        ranks: np.ndarray,
        eligible: np.ndarray,
        buy_blocked: np.ndarray,
        sell_blocked: np.ndarray,
        mandatory_extra: np.ndarray,
        budget: float,
        min_trade: float,
        lot_size: int,
        cost_model: ExecutionCostModel,
    ) -> None:
        self.keys = keys
        self.shares = shares.astype(np.int64).copy()
        self.prices = prices.astype(np.float64).copy()
        self.weights = weights.astype(np.float64).copy()
        self.ranks = ranks
        self.eligible = eligible
        self.buy_blocked = buy_blocked
        self.sell_blocked = sell_blocked
        self.mandatory_extra = mandatory_extra
        self.cash = float(cash)
        self.budget_remaining = float(budget)
        self.min_trade = float(min_trade)
        self.lot_size = int(lot_size)
        self.cost_model = cost_model
        self.n = len(keys)
        # §1.2: a missing price (<= 0 or non-finite) forbids ordering the
        # name; such holdings contribute nothing to session equity.
        self.valid_price = np.isfinite(self.prices) & (self.prices > 0.0)
        valued = np.where(self.valid_price, self.shares.astype(np.float64) * self.prices, 0.0)
        self.equity = float(cash) + float(valued.sum())
        # The M-class universe (§1.3.1): held names that are ineligible
        # (constructor.py:299 risk exits) or flagged mandatory by the
        # caller (legacy wind-down, constructor.py:317-321 semantics).
        held = self.shares > 0
        self.mandatory_mask = held & (~self.eligible | self.mandatory_extra)
        self.discretionary = [
            i for i in range(self.n) if not bool(self.mandatory_mask[i])
        ]
        # (session, name, direction) merged notionals — the §1.5 billing
        # units; every fee below is an incremental authority call on them.
        self.sell_notional = np.zeros(self.n, dtype=np.float64)
        self.buy_notional = np.zeros(self.n, dtype=np.float64)
        self.forced_exit_count = 0
        self.m_executed: list[str] = []
        self.m_blocked: list[tuple[str, str]] = []
        self.d_skipped: list[tuple[str, str]] = []
        self.d_notional = 0.0
        self.groups: list[dict[str, Any]] = []
        self.orders: list[tuple[str, str, int, float]] = []
        self.consumed = 0.0

    # -- fee authority wrappers (§1.5: merged notionals only) -------------

    def sell_fee_increment(self, i: int, add_notional: float) -> float:
        base = float(self.sell_notional[i])
        merged = base + float(add_notional)
        cost = self.cost_model.sell_cost(merged).total
        return float(cost) - (float(self.cost_model.sell_cost(base).total) if base > 0.0 else 0.0)

    def buy_fee_increment(self, i: int, add_notional: float) -> float:
        base = float(self.buy_notional[i])
        merged = base + float(add_notional)
        cost = self.cost_model.buy_cost(merged).total
        return float(cost) - (float(self.cost_model.buy_cost(base).total) if base > 0.0 else 0.0)

    # -- orderings (§1.3/§1.4: sell = (-rank, key), buy = (rank, key)) -----

    def sell_order(self, indices: Sequence[int]) -> list[int]:
        return sorted(indices, key=lambda i: (-int(self.ranks[i]), self.keys[i]))

    def buy_order(self, indices: Sequence[int]) -> list[int]:
        return sorted(indices, key=lambda i: (int(self.ranks[i]), self.keys[i]))

    # -- demand bookkeeping (§1.2) ------------------------------------------

    def position_notional(self, i: int) -> float:
        return float(self.shares[i]) * float(self.prices[i])

    def sell_demand(self, i: int) -> float:
        """Discretionary sell demand in yuan: max(v_i - w_i, 0) * E."""
        if not self.valid_price[i] or self.equity <= 0.0:
            return 0.0
        return max(self.position_notional(i) - float(self.weights[i]) * self.equity, 0.0)

    def buy_demand(self, i: int) -> float:
        """Discretionary buy demand in yuan: max(w_i - v_i, 0) * E."""
        if not self.valid_price[i] or self.equity <= 0.0:
            return 0.0
        return max(float(self.weights[i]) * self.equity - self.position_notional(i), 0.0)

    def usable_sell(self, i: int) -> bool:
        return (
            self.valid_price[i]
            and not bool(self.sell_blocked[i])
            and self.shares[i] > 0
            and self.sell_demand(i) > 0.0
        )

    def usable_sell_capacity(self) -> float:
        return float(sum(self.sell_demand(i) for i in self.discretionary if self.usable_sell(i)))

    def buy_candidates(self) -> list[int]:
        return self.buy_order(
            [
                i
                for i in self.discretionary
                if self.buy_demand(i) > 0.0
                and bool(self.eligible[i])
                and not bool(self.buy_blocked[i])
                and bool(self.valid_price[i])
            ]
        )

    # -- execution helpers ----------------------------------------------------

    def execute_sell(self, i: int, shares: int, notional: float) -> None:
        fee = self.sell_fee_increment(i, notional)
        self.shares[i] -= int(shares)
        self.sell_notional[i] += float(notional)
        self.cash += float(notional) - fee
        self.orders.append((self.keys[i], "sell", int(shares), float(notional)))

    def execute_buy(self, i: int, shares: int, notional: float) -> None:
        fee = self.buy_fee_increment(i, notional)
        self.shares[i] += int(shares)
        self.buy_notional[i] += float(notional)
        self.cash -= float(notional) + fee
        self.orders.append((self.keys[i], "buy", int(shares), float(notional)))

    def session_fee_totals(self) -> dict[str, float]:
        """§1.5: session fees always come from the authority's ``total``
        on the merged per-unit notionals."""
        sell_units = np.asarray([v for v in self.sell_notional.tolist() if v > 0.0])
        buy_units = np.asarray([v for v in self.buy_notional.tolist() if v > 0.0])
        sell_total = (
            float(np.sum(self.cost_model.sell_cost(sell_units).total)) if sell_units.size else 0.0
        )
        buy_total = (
            float(np.sum(self.cost_model.buy_cost(buy_units).total)) if buy_units.size else 0.0
        )
        return {"sell": sell_total, "buy": buy_total, "total": sell_total + buy_total}


def _validate_inputs(
    shares: np.ndarray,
    prices: np.ndarray,
    cash: float,
    weights: np.ndarray,
    ranks: np.ndarray,
    stable_keys: np.ndarray,
    eligible: np.ndarray,
    buy_blocked: np.ndarray,
    sell_blocked: np.ndarray,
    lot_size: int,
    min_trade_amount: float | None,
    turnover_budget: float | None,
) -> None:
    n = shares.shape[0]
    for name, values in (
        ("prices", prices),
        ("target_weights", weights),
        ("ranks", ranks),
        ("stable_keys", stable_keys),
        ("eligible", eligible),
        ("buy_blocked", buy_blocked),
        ("sell_blocked", sell_blocked),
    ):
        if values.shape != (n,):
            raise ValueError(f"{name} shape {values.shape} does not match prev_shares {(n,)}")
    if np.any(shares < 0):
        raise ValueError("prev_shares must be non-negative")
    if not math.isfinite(float(cash)) or float(cash) < 0.0:
        raise ValueError("cash must be a finite non-negative number")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("target_weights must be finite and non-negative")
    if len(set(stable_keys.tolist())) != len(stable_keys):
        raise ValueError("stable_keys must be unique")
    if int(lot_size) < 1:
        raise ValueError("lot_size must be >= 1")
    if min_trade_amount is not None and float(min_trade_amount) < 0.0:
        raise ValueError("min_trade_amount must be >= 0")
    if turnover_budget is not None and float(turnover_budget) < 0.0:
        raise ValueError("turnover_budget must be >= 0")


def _run_m_class(session: _Session) -> None:
    """§1.3.1 M — forced risk exits on their own budget-free channel.

    Sell order (-rank, key); a name with a price and not blocked sells
    its WHOLE remaining position (odd lots included — engine semantics,
    backtest.py:310-318) with budget consumption 0; otherwise the
    position is kept and the block recorded with cause "sell_blocked"
    or "missing_price".  M is exempt from the minimum-order filter.
    """
    session.forced_exit_count = int(np.count_nonzero(session.mandatory_mask))
    for i in session.sell_order(np.flatnonzero(session.mandatory_mask).tolist()):
        blocked = bool(session.sell_blocked[i])
        priceless = not bool(session.valid_price[i])
        if blocked or priceless:
            cause = "sell_blocked" if blocked else "missing_price"
            session.m_blocked.append((session.keys[i], cause))
            continue
        shares = int(session.shares[i])
        notional = shares * float(session.prices[i])
        session.execute_sell(i, shares, notional)
        session.m_executed.append(session.keys[i])


def _run_d_class(session: _Session) -> None:
    """§1.3.2 D — one-sided target-deleveraging sells, budget-exempt,
    capped by R = max(S - B_dem, 0) over the remaining discretionary
    demand (after M).  Partial sells floor to whole lots; a liquidation
    leg (target weight 0, fill = the whole remaining holding) sells
    everything incl. odd lots (§1.6).  The minimum merged-notional rule
    applies (skip + record when unreachable)."""
    total_sell = float(sum(session.sell_demand(i) for i in session.discretionary))
    total_buy = float(sum(session.buy_demand(i) for i in session.discretionary))
    relief = max(total_sell - total_buy, 0.0)
    if relief <= 0.0:
        return
    lot = session.lot_size
    for i in session.sell_order(
        [i for i in session.discretionary if session.usable_sell(i)]
    ):
        if relief <= 0.0:
            break
        want = min(session.sell_demand(i), relief)
        price = float(session.prices[i])
        lot_notional = lot * price
        position_value = session.position_notional(i)
        liquidation = (
            float(session.weights[i]) <= _WEIGHT_EPS
            and want >= position_value - _CASH_TOL
            and position_value > 0.0
        )
        if liquidation:
            shares = int(session.shares[i])
        else:
            shares = int(want // lot_notional) * lot if lot_notional > 0.0 else 0
        notional = shares * price
        if shares <= 0:
            session.d_skipped.append((session.keys[i], "lot"))
            continue
        if session.min_trade > 0.0 and session.sell_notional[i] + notional < session.min_trade:
            session.d_skipped.append((session.keys[i], "min_amount"))
            continue
        session.execute_sell(i, shares, notional)
        session.d_notional += notional
        relief -= notional


def _attach_sells(session: _Session, y: float, need: float):
    """§1.4 step 3 — minimal-financing sell attachment for a fixed buy
    notional ``y`` with financing gap ``need`` > 0.

    Walks the sell candidates in sell order; each candidate may attach
    the SMALLEST allowed notional f (whole-lot grid, or the full
    remaining position for a liquidation leg — §1.6) that lifts the
    accumulated net proceeds to ``need``; when no allowed f reaches it,
    the LARGEST allowed f is attached and the next candidate tops up.
    Attachments are budget-capped (u_a = min(demand, B_r - y - X)).
    Returns [(index, shares, notional), ...] or None when the acceptable
    sell legs cannot finance ``need``.  Nothing is committed here.
    """
    attachments: list[tuple[int, int, float]] = []
    attached = 0.0
    net = 0.0
    lot = session.lot_size
    minimum = session.min_trade
    for i in session.sell_order(
        [i for i in session.discretionary if session.usable_sell(i)]
    ):
        if net >= need - _CASH_TOL:
            break
        budget_cap = session.budget_remaining - y - attached
        if budget_cap <= 0.0:
            continue
        cap = min(session.sell_demand(i), budget_cap)
        price = float(session.prices[i])
        lot_notional = lot * price
        if lot_notional <= 0.0 or cap < lot_notional:
            continue
        position_value = session.position_notional(i)
        liquidation_allowed = (
            float(session.weights[i]) <= _WEIGHT_EPS
            and cap >= position_value - _CASH_TOL
            and position_value > 0.0
        )
        lots = int(cap // lot_notional)
        grid_max = lots * lot_notional

        def meets_min(f: float) -> bool:
            return minimum <= 0.0 or session.sell_notional[i] + f >= minimum

        def net_after(f: float) -> float:
            return net + (f - session.sell_fee_increment(i, f))

        # Smallest allowed f that satisfies the financing need.
        chosen: tuple[int, float] | None = None
        if lots >= 1 and meets_min(grid_max):
            floor_notional = max(lot_notional, minimum - float(session.sell_notional[i]))
            lo_lots = max(1, int(math.ceil(floor_notional / lot_notional - 1e-12)))
            for k in range(lo_lots, lots + 1):
                f = k * lot_notional
                if net_after(f) >= need - _CASH_TOL:
                    chosen = (k * lot, f)
                    break
        if chosen is None and liquidation_allowed and meets_min(position_value):
            if net_after(position_value) >= need - _CASH_TOL:
                chosen = (int(session.shares[i]), position_value)
        if chosen is not None:
            shares, f = chosen
            attachments.append((i, int(shares), float(f)))
            attached += f
            net = net_after(f)
            continue
        # No allowed f meets the need: attach the largest allowed f and
        # let the next candidate top up (§1.4 step 3).
        fallback: tuple[int, float] | None = None
        if lots >= 1 and meets_min(grid_max):
            fallback = (lots * lot, grid_max)
        elif liquidation_allowed and meets_min(position_value):
            fallback = (int(session.shares[i]), position_value)
        if fallback is None:
            continue
        shares, f = fallback
        attachments.append((i, int(shares), float(f)))
        attached += f
        net = net_after(f)
    if net >= need - _CASH_TOL:
        return attachments
    return None


def _buy_skip_code(session: _Session, j: int) -> str:
    """§1.7 — the frozen six-step adjudication for one buy candidate,
    evaluated against the CURRENT session state."""
    price = float(session.prices[j])
    lot_notional = session.lot_size * price
    demand = session.buy_demand(j)
    if demand < lot_notional:
        return "lot"
    max_grid = float(int(demand // lot_notional)) * lot_notional
    if (
        session.min_trade > 0.0
        and float(session.buy_notional[j]) + max_grid < session.min_trade
    ):
        return "min_amount"
    y_min = max(session.min_trade, lot_notional)
    payable_min = y_min + float(session.cost_model.buy_cost(y_min).total)
    capacity = session.usable_sell_capacity()
    if capacity <= 0.0 and session.cash + _CASH_TOL < payable_min:
        return "no_viable_counterpart"
    if session.cash + capacity < y_min:
        return "cash"
    real_relief = session.cash
    for i in session.discretionary:
        if not session.usable_sell(i):
            continue
        cap_i = session.sell_demand(i)
        base = float(session.sell_notional[i])
        fee_full = float(session.cost_model.sell_cost(base + cap_i).total)
        fee_base = float(session.cost_model.sell_cost(base).total) if base > 0.0 else 0.0
        real_relief += cap_i - (fee_full - fee_base)
    if real_relief < payable_min:
        return "fees"
    return "budget"


def _run_r_class(session: _Session) -> tuple[list[int], dict[int, str]]:
    """§1.4 — the trade-group loop.

    Each pass scans the buy candidates in buy order for the FIRST one
    that can form a viable group (largest feasible whole-lot y* with the
    minimal financing sells), commits it, and restarts from the queue
    head (the same candidate may group repeatedly).  A candidate that
    cannot form any group is recorded with its §1.7 code and removed for
    the rest of the session; when no candidate can group, the session
    ends.  Every step either forms a group or removes a candidate, so
    the loop terminates (§1.4 step 5).
    """
    candidates = session.buy_candidates()
    removed: dict[int, str] = {}
    lot = session.lot_size
    while True:
        formed = False
        for j in candidates:
            if j in removed or session.buy_demand(j) <= 0.0:
                continue
            price = float(session.prices[j])
            lot_notional = lot * price
            demand = session.buy_demand(j)
            n_max = int(demand // lot_notional)
            floor_notional = max(
                lot_notional,
                session.min_trade - float(session.buy_notional[j]),
            )
            n_min = max(1, int(math.ceil(floor_notional / lot_notional - 1e-12)))
            group_sells: tuple[tuple[int, int, float], ...] | None = None
            group_buy: tuple[int, int, float] | None = None
            if n_max >= 1 and n_max >= n_min:
                for n in range(n_max, n_min - 1, -1):
                    y = n * lot_notional
                    buy_fee = session.buy_fee_increment(j, y)
                    need = y + buy_fee - session.cash
                    if need <= _CASH_TOL:
                        # §1.4 step 3: current cash funds the buy — a
                        # pure-buy group with no sell leg.
                        if y <= session.budget_remaining:
                            group_sells = ()
                            group_buy = (j, n * lot, y)
                            break
                        continue  # smaller n relaxes (a): keep scanning
                    attachments = _attach_sells(session, y, need)
                    if attachments is None:
                        continue
                    x = sum(f for _, _, f in attachments)
                    if x + y <= session.budget_remaining:
                        group_sells = tuple(attachments)
                        group_buy = (j, n * lot, y)
                        break
                    continue  # attachments are budget-capped; smaller n may fit
            if group_buy is None or group_sells is None:
                removed[j] = _buy_skip_code(session, j)
                continue
            # Commit the group (ledger order: sells, then the buy).
            sells_record: list[tuple[str, int, float]] = []
            for i, shares, f in group_sells:
                session.execute_sell(i, shares, f)
                sells_record.append((session.keys[i], int(shares), float(f)))
            buy_index, buy_shares, y = group_buy
            session.execute_buy(buy_index, buy_shares, y)
            consumed = sum(f for _, _, f in group_sells) + y
            session.budget_remaining -= consumed
            session.consumed += consumed
            session.groups.append(
                {
                    "sells": tuple(sells_record),
                    "buy": (session.keys[buy_index], int(buy_shares), float(y)),
                    "budget_consumed": float(consumed),
                }
            )
            formed = True
            break
        if not formed:
            return candidates, removed


def _session_reason(
    session: _Session,
    candidates: list[int],
    removed: dict[int, str],
) -> tuple[str | None, tuple[tuple[str, str, str], ...]]:
    """§1.7 session-level reason + per-leg codes (end-of-session state)."""
    unfilled = [j for j in candidates if session.buy_demand(j) > 0.0]
    legs = [
        (session.keys[j], "buy", removed.get(j) or _buy_skip_code(session, j))
        for j in sorted(unfilled, key=lambda i: (int(session.ranks[i]), session.keys[i]))
    ]
    budget_exhausted = (
        math.isfinite(session.budget_remaining)
        and session.budget_remaining <= _CASH_TOL
    )
    if budget_exhausted:
        # §1.7: a budget EXACTLY used up reports None even while demand
        # remains (frozen §4.2.4 "budget_unspent_reason = None（用尽）").
        return None, tuple(sorted(legs))
    if unfilled:
        first = min(unfilled, key=lambda i: (int(session.ranks[i]), session.keys[i]))
        code = removed.get(first) or _buy_skip_code(session, first)
        return code, tuple(sorted(legs))
    if not candidates:
        # No buy candidate ever existed: when anything else happened (or
        # any sell demand sits unfilled) the budget simply had no
        # counterpart to deploy into (§4.4.4); a session with nothing to
        # do reports None (§1.7 "无需求时为 None").
        anything = (
            bool(session.m_executed)
            or session.d_notional > 0.0
            or any(session.usable_sell(i) for i in session.discretionary)
        )
        return ("no_viable_counterpart" if anything else None), ()
    return None, tuple(sorted(legs))


def plan_trade_session(
    *,
    prev_shares: Sequence[int],
    prices: Sequence[float],
    cash: float,
    target_weights: Sequence[float],
    ranks: Sequence[int],
    stable_keys: Sequence[str],
    cost_model: ExecutionCostModel,
    eligible: Sequence[bool],
    buy_blocked: Sequence[bool],
    sell_blocked: Sequence[bool],
    turnover_budget: float | None,
    min_trade_amount: float | None = None,
    lot_size: int = 100,
    initial_funding_used: bool | None = None,
    mandatory: Sequence[bool] | None = None,
) -> TradeSessionPlan:
    """Plan one rebalance session as whole-lot trade groups (frozen §1).

    Parameters follow §1.2: ``prev_shares`` (>= 0), ``prices`` (entry
    opens; <= 0 or non-finite = missing price — the name cannot be
    ordered and contributes nothing to equity), ``cash`` (>= 0), session
    equity ``E = cash + Σ shares·price``, ``target_weights`` (the
    constructor's output), ``ranks`` (0 = best) with unique
    ``stable_keys`` tie-breaks, the three masks, the budget / minimum /
    lot configuration and the fee authority.  ``initial_funding_used``
    is the §6.2 run-level exemption state: the budget relief applies
    only when the book is empty AND the flag is not True (``None`` keeps
    the legacy "empty book" reading).  ``mandatory`` optionally merges
    extra held names (e.g. legacy wind-down, constructor.py:317-321
    semantics) into the budget-exempt M class.
    """
    shares = np.asarray(prev_shares, dtype=np.int64)
    prices = np.asarray(prices, dtype=np.float64)
    weights = np.asarray(target_weights, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.int64)
    keys_arr = np.asarray(stable_keys).astype(str)
    eligible = np.asarray(eligible, dtype=bool)
    buy_blocked = np.asarray(buy_blocked, dtype=bool)
    sell_blocked = np.asarray(sell_blocked, dtype=bool)
    mandatory_extra = (
        np.zeros(shares.shape, dtype=bool)
        if mandatory is None
        else np.asarray(mandatory, dtype=bool)
    )
    _validate_inputs(
        shares,
        prices,
        cash,
        weights,
        ranks,
        keys_arr,
        eligible,
        buy_blocked,
        sell_blocked,
        lot_size,
        min_trade_amount,
        turnover_budget,
    )
    if not isinstance(cost_model, ExecutionCostModel):
        raise TypeError("cost_model must be an ExecutionCostModel (fee authority, §0.3)")
    if initial_funding_used is not None and not isinstance(
        initial_funding_used, (bool, np.bool_)
    ):
        raise TypeError("initial_funding_used must be a bool or None (§6.2)")

    session = _Session(
        keys=[str(k) for k in keys_arr.tolist()],
        shares=shares,
        prices=prices,
        cash=float(cash),
        weights=weights,
        ranks=ranks,
        eligible=eligible,
        buy_blocked=buy_blocked,
        sell_blocked=sell_blocked,
        mandatory_extra=mandatory_extra,
        budget=math.inf,
        min_trade=0.0 if min_trade_amount is None else float(min_trade_amount),
        lot_size=int(lot_size),
        cost_model=cost_model,
    )
    # §1.2: B_r = inf when the budget is None or the §6 exemption applies
    # (empty book AND the run-level flag does not say the exemption was
    # already used); otherwise turnover_budget * E.
    book_empty = bool(np.all(shares == 0))
    exempt = book_empty and initial_funding_used is not True
    if turnover_budget is None or exempt:
        session.budget_remaining = math.inf
    else:
        session.budget_remaining = float(turnover_budget) * max(session.equity, 0.0)

    _run_m_class(session)
    _run_d_class(session)
    candidates, removed = _run_r_class(session)
    reason, legs = _session_reason(session, candidates, removed)

    return TradeSessionPlan(
        turnover_budget_mode="trade_groups",
        forced_exit_count=session.forced_exit_count,
        forced_exit_executed=tuple(session.m_executed),
        forced_exit_blocked=tuple(sorted(session.m_blocked)),
        deleveraging_sell_notional=float(session.d_notional),
        deleveraging_skipped=tuple(sorted(session.d_skipped)),
        trade_groups=tuple(session.groups),
        budget_unspent_reason=reason,
        leg_skip_reasons=tuple(sorted(legs)),
        session_fees=session.session_fee_totals(),
        turnover_budget_consumed=float(session.consumed),
        execution_failed_buys=(),
        executed_orders=tuple(session.orders),
    )


# --- §1.1.4: the execution-layer contract ----------------------------------------

# The injectable fill stub: (key, side, shares, notional) -> filled?
FillCallback = Callable[[str, str, int, float], bool]


@dataclass(frozen=True)
class SessionExecution:
    """Outcome of settling one :class:`TradeSessionPlan` against a ledger.

    Frozen §1.1.4 (W2): the planner only produces a plan; when actual
    execution fails a BUY, the real sells and the cash they raised are
    KEPT (no fabricated rollback, no compensation trades) and the failed
    names are recorded in ``failed_buys`` (deduplicated, sorted by
    stable_key).  ``diagnostics`` is the §1.8 trade_groups block as
    ACTUALLY executed — the run-level surface for the mode's reason codes
    (§1.7), fees (§1.5) and the execution-failure record.
    """

    shares: np.ndarray
    cash: float
    filled_orders: tuple[tuple[str, str, int, float], ...]
    failed_buys: tuple[str, ...]
    session_fees: dict[str, float]
    diagnostics: Mapping[str, Any]


def _merged_fee_increment(
    cost: Callable[[float], Any], base: float, add: float
) -> float:
    """§1.5 increment: fee(base + add) - fee(base), both calls on merged
    notionals through the authority (the telescoping identity — the sum of
    the increments equals the merged authority total exactly)."""

    total = float(cost(base + add).total)
    return total - (float(cost(base).total) if base > 0.0 else 0.0)


def _authority_fee_totals(
    cost_model: ExecutionCostModel,
    sell_merged: dict[str, float],
    buy_merged: dict[str, float],
) -> dict[str, float]:
    """§1.5: session fees always come from the authority's ``total`` on the
    merged per-(session, name, direction) notionals of the FILLED orders."""

    sell_units = np.asarray([v for v in sell_merged.values() if v > 0.0])
    buy_units = np.asarray([v for v in buy_merged.values() if v > 0.0])
    sell_total = (
        float(np.sum(cost_model.sell_cost(sell_units).total))
        if sell_units.size
        else 0.0
    )
    buy_total = (
        float(np.sum(cost_model.buy_cost(buy_units).total)) if buy_units.size else 0.0
    )
    return {"sell": sell_total, "buy": buy_total, "total": sell_total + buy_total}


def execute_trade_session(
    plan: TradeSessionPlan,
    *,
    stable_keys: Sequence[str],
    prev_shares: Sequence[int],
    cash: float,
    cost_model: ExecutionCostModel,
    fill: FillCallback | None = None,
) -> SessionExecution:
    """Settle one planned session against a share/cash ledger (frozen §1.1.4).

    Walks ``plan.executed_orders`` in ledger order (M sells, D sells, then
    per group its sells and buy).  ``fill`` is the injectable execution
    stub ``(key, side, shares, notional) -> bool``; ``None`` (the default)
    fills everything — the planner has already verified feasibility.
    Semantics (§1.1.4, no degrees of freedom):

    * a FAILED BUY keeps every already-executed sell and the cash it
      raised, records the name in ``execution_failed_buys`` (sorted by
      stable_key), and issues no compensation trade;
    * a failed SELL fill is a contract violation, not a modelled state —
      the planner only emits sells whose price/blocked preconditions it
      verified (§1.3.1) and the engine sell leg has no failure path, so
      the executor fails closed (raises);
    * fees are the authority's, per merged (session, name, direction) unit
      of the FILLED orders (§1.5) — one incremental merged-notional call
      per leg, reconciling with the plan when everything fills;
    * cash may never go negative (planner-consistent input cannot reach
      it; a stub that fails buys only ever leaves MORE cash).
    """

    if not isinstance(plan, TradeSessionPlan):
        raise TypeError("plan must be a TradeSessionPlan (§1.1.4)")
    if not isinstance(cost_model, ExecutionCostModel):
        raise TypeError("cost_model must be an ExecutionCostModel (fee authority, §0.3)")
    if fill is not None and not callable(fill):
        raise TypeError("fill must be callable or None")
    keys = [str(k) for k in np.asarray(stable_keys).astype(str).tolist()]
    shares = np.asarray(prev_shares, dtype=np.int64).copy()
    if shares.ndim != 1:
        raise ValueError("prev_shares must be 1-D")
    if np.any(shares < 0):
        raise ValueError("prev_shares must be non-negative")
    if len(set(keys)) != len(keys) or len(keys) != int(shares.shape[0]):
        raise ValueError("stable_keys must be unique and match prev_shares")
    if not math.isfinite(float(cash)) or float(cash) < 0.0:
        raise ValueError("cash must be a finite non-negative number")
    cash = float(cash)
    index = {key: i for i, key in enumerate(keys)}
    sell_merged: dict[str, float] = {}
    buy_merged: dict[str, float] = {}
    failed: set[str] = set()
    filled: list[tuple[str, str, int, float]] = []
    for key, side, qty, notional in plan.executed_orders:
        if key not in index:
            raise ValueError(f"order references unknown stable_key {key!r}")
        qty = int(qty)
        notional = float(notional)
        if qty <= 0 or notional < 0.0:
            raise ValueError("orders must carry positive shares and a non-negative notional")
        i = index[key]
        settled = True if fill is None else bool(fill(key, side, qty, notional))
        if side == "sell":
            if not settled:
                raise ValueError(
                    "a sell fill cannot fail under frozen §1.1.4 — the "
                    "planner only emits sells with a verified price and "
                    "unblocked status, and the engine sell leg has no "
                    "failure path; refusing to invent an unmodelled semantics"
                )
            fee = _merged_fee_increment(cost_model.sell_cost, sell_merged.get(key, 0.0), notional)
            if qty > int(shares[i]):
                raise ValueError(f"sell of {key!r} exceeds the ledger position")
            shares[i] -= qty
            sell_merged[key] = sell_merged.get(key, 0.0) + notional
            cash += notional - fee
            filled.append((key, "sell", qty, notional))
        elif side == "buy":
            if not settled:
                # §1.1.4: keep the real sells and the cash; no rollback,
                # no compensation trade — just record the failure.
                failed.add(key)
                continue
            fee = _merged_fee_increment(cost_model.buy_cost, buy_merged.get(key, 0.0), notional)
            shares[i] += qty
            buy_merged[key] = buy_merged.get(key, 0.0) + notional
            cash -= notional + fee
            filled.append((key, "buy", qty, notional))
        else:
            raise ValueError(f"order side must be 'buy' or 'sell', got {side!r}")
        if cash < -_CASH_TOL:
            raise RuntimeError("cash went negative — the settled plan was not self-financing")
    session_fees = _authority_fee_totals(cost_model, sell_merged, buy_merged)
    failed_buys = tuple(sorted(failed))
    book = shares.copy()
    book.setflags(write=False)
    return SessionExecution(
        shares=book,
        cash=float(cash),
        filled_orders=tuple(filled),
        failed_buys=failed_buys,
        session_fees=dict(session_fees),
        diagnostics=MappingProxyType(
            {
                "turnover_budget_mode": plan.turnover_budget_mode,
                "forced_exit_count": plan.forced_exit_count,
                "forced_exit_executed": plan.forced_exit_executed,
                "forced_exit_blocked": plan.forced_exit_blocked,
                "deleveraging_sell_notional": plan.deleveraging_sell_notional,
                "deleveraging_skipped": plan.deleveraging_skipped,
                "trade_groups": plan.trade_groups,
                "budget_unspent_reason": plan.budget_unspent_reason,
                "leg_skip_reasons": plan.leg_skip_reasons,
                "session_fees": dict(session_fees),
                "turnover_budget_consumed": plan.turnover_budget_consumed,
                "execution_failed_buys": failed_buys,
                "executed_orders": tuple(filled),
            }
        ),
    )


# --- audit fix 2026-09-23: fail-closed guard for unwired run entries (draft §3) ----


def refuse_unwired_trade_groups(config: Any, consumer: str) -> None:
    """Refuse ``turnover_allocation = "trade_groups"`` at a run entry that
    does not consume this module's planner (audit fix, major).

    Under the experimental mode the constructor walks the unbudgeted path
    and only labels ``diagnostics["turnover_budget_mode"]`` (frozen §1.1.2)
    — the session budget is delegated to :func:`plan_trade_session`.  A
    consumer that then executes the constructor's output WEIGHTS directly
    (the engine's weight-space loop, the paper sim's order maker, the
    basket reward simulation) silently drops that budget: the run becomes
    a budget bypass instead of a budget allocator.  Until those consumers
    are wired to ``plan_trade_session``/``execute_trade_session``
    (docs/p51_trade_groups_draft.md §3), they call this guard at
    initialization and refuse the mode instead of running it unbudgeted.

    The library-level composition — constructor -> ``plan_trade_session``
    -> ``execute_trade_session`` — is the supported surface of the mode
    (pinned end-to-end in tests/test_p51_wp15_integration.py).  Configs
    without the key, and the wired modes ``priority``/``proportional``,
    pass through untouched (red line 2: default behavior unchanged).
    """

    allocation = str(getattr(config, "turnover_allocation", "priority"))
    if allocation == "trade_groups":
        raise ValueError(
            f"{consumer} does not consume the trade_groups planner: "
            "turnover_allocation='trade_groups' delegates the turnover "
            "budget to ashare_portfolio.trade_groups.plan_trade_session "
            "(frozen §1.1.2), and executing the constructor's unbudgeted "
            "target weights here would silently drop that budget. "
            "Refusing to run (fail-closed) until the wiring lands; use a "
            "wired allocation mode ('priority'/'proportional') or the "
            "library-level composition constructor -> plan_trade_session "
            "-> execute_trade_session (docs/p51_trade_groups_draft.md §3)."
        )


# --- §5.1 (W6): the pure configuration-feasibility criterion ----------------------

_FEASIBILITY_FORMULA = "sell_min + buy_min <= turnover_budget * equity"
_FEASIBILITY_PAIR = "2 * min_trade_amount <= turnover_budget * E"
_FEASIBILITY_LEG = "min_trade_amount <= turnover_budget * E"
_FEASIBILITY_SCOPE = (
    "upper-bound NECESSARY condition ignoring lot granularity and fees; "
    "lots and fees can only tighten it (§5.1)"
)


def ordinary_swap_session_feasibility(
    *,
    min_trade_amount: float | None,
    turnover_budget: float | None,
    equity: float,
) -> dict[str, Any]:
    """Can one ordinary replacement pair complete in a single session?

    Pure arithmetic, no IO (frozen §5.1, W6): a pair of minimum-sized legs
    is completable iff ``2 x min_trade_amount <= turnover_budget x E`` and
    a single minimum leg iff ``min_trade_amount <= turnover_budget x E``.
    This is the upper-bound NECESSARY condition — lot granularity and fees
    can only tighten it.  ``turnover_budget=None`` means unbudgeted
    (``B_r = inf``, §1.2): the criterion cannot bind.  The slow reference
    verdict and wording this function reproduces live in
    ``reports/wp15/config_feasibility.json`` (frozen §5.2/§5.3).
    """

    equity = float(equity)
    if not math.isfinite(equity) or equity < 0.0:
        raise ValueError("equity must be a finite non-negative number")
    if min_trade_amount is not None and float(min_trade_amount) < 0.0:
        raise ValueError("min_trade_amount must be >= 0")
    if turnover_budget is not None and float(turnover_budget) < 0.0:
        raise ValueError("turnover_budget must be >= 0")
    minimum = 0.0 if min_trade_amount is None else float(min_trade_amount)
    pair_requirement = 2.0 * minimum
    common = {
        "formula": _FEASIBILITY_FORMULA,
        "pair_feasible_iff": _FEASIBILITY_PAIR,
        "single_leg_feasible_iff": _FEASIBILITY_LEG,
        "scope": _FEASIBILITY_SCOPE,
        "min_trade_amount": minimum,
        "equity": equity,
        "pair_requirement_yuan": pair_requirement,
    }
    if turnover_budget is None:
        return {
            **common,
            "turnover_budget": None,
            "session_budget_yuan": None,
            "substitution": (
                "turnover_budget is None -> B_r = inf (unbudgeted): the "
                "criterion cannot bind"
            ),
            "pair_feasible": True,
            "single_leg_feasible": True,
            "verdict": (
                "UNBUDGETED: turnover_budget is None -> B_r = inf; the "
                "§5.1 condition cannot bind"
            ),
        }
    budget = float(turnover_budget)
    session_budget = budget * equity
    pair_feasible = pair_requirement <= session_budget
    single_feasible = minimum <= session_budget
    comparator = "<=" if pair_feasible else ">"
    substitution = (
        f"2 * {minimum:g} = {pair_requirement:g} {comparator} "
        f"{budget:g} * {equity:g} = {session_budget:g}"
    )
    if pair_feasible:
        verdict = (
            "FEASIBLE: the upper-bound necessary condition (§5.1) admits "
            "one ordinary buy+sell pair in a single session (lots and fees "
            "can only tighten it)"
        )
    else:
        verdict = (
            "INFEASIBLE: a single session can never complete one ordinary "
            "buy+sell pair under this configuration"
        )
    return {
        **common,
        "turnover_budget": budget,
        "session_budget_yuan": session_budget,
        "substitution": substitution,
        "pair_feasible": pair_feasible,
        "single_leg_feasible": single_feasible,
        "verdict": verdict,
    }
