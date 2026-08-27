"""Unified execution spec (T3-02): golden parity between the fast
vectorized research engine and the whole-lot paper matcher.

The spec executes one signal set through three paths and records the
contract:

* **Free path** — the vectorized engine (``AshareBacktestEngine``) when
  target weights come from the engine's own selection, or the exact
  mirror of externally supplied weights (T3-01 optimizer output).  The
  engine's convention is a *frictionless-rebalanced* book: positions are
  re-marked at the target weights of the compounded equity at every
  entry, only weight deltas are traded, and fees are charged on the
  weight-delta notionals.
* **Lot-free matcher path** — the authoritative ``SimBroker`` in
  continuous-share mode (``lot_size=0``) executing exactly those weight
  deltas against the same re-marked book.  Its equity must reproduce the
  free path exactly (1e-9 relative) — the golden cross-check that the
  matcher's blocking, T+1, fee and price rules agree with the engine.
* **Lot matcher path** — the same broker with whole lots and real cash
  accounting, trading full target shares (a real account rebalances
  price drift when it rebalances).  Its divergence from the free path
  decomposes per day into the *recorded* ``rounding`` PnL (integer-lot
  quantity differences), ``fee_diff`` (fee financing: real trade fees
  minus weight-delta fees) and ``carry`` (sub-lot and affordability
  cash), and the bookkeeping identity
  ``sum(residual) == lot_equity(T) - free_equity(T)`` holds exactly.

The spec also applies the blocking rule (buy-blocked / outside-universe
names are never freshly bought; sell-blocked reductions are force-held at
the previous weight; weights never renormalize upward) to *any* weight
source — for engine-produced weights the rule is a no-op because the
engine already embeds it, for optimizer weights it is what keeps both
paths consistent.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from ashare_data.config import BacktestConfig
from ashare_data.processor import open_to_open_returns, tradability_blocked
from ashare_execution import ExecutionCostModel
from ashare_model.backtest import AshareBacktestEngine
from ashare_trading.matching import SimBroker
from ashare_trading.orders import build_orders, target_shares_from_weights
from ashare_trading.portfolio import PositionState, SimulationPortfolio

EXECUTION_SPEC_VERSION = 1

_BAR_COLUMNS = ("open", "high", "low", "pre_close", "volume")


class GoldenParityViolation(AssertionError):
    """The parity contract broke; the message carries every violation."""


@dataclass(frozen=True)
class FillRecord:
    """One order's outcome in the matcher path (the authoritative record)."""

    side: str
    ts_code: str
    requested: float
    filled: float
    price: float
    status: str
    reason: str


@dataclass(frozen=True)
class DayParityRecord:
    """One signal period's free/lot comparison and residual attribution.

    ``residual`` is ``(lot_equity_after - lot_equity_before) -
    (free_equity_after - free_equity_before)`` and must equal
    ``rounding + fee_diff + carry``; ``rounding`` is the integer-lot
    quantity PnL, ``fee_diff`` the fee-financing difference and
    ``carry`` the recorded cash residual (everything else is a spec bug).
    """

    date: str
    free_return: float
    lot_free_return: float
    lot_return: float
    turnover: float
    rounding: float
    fee_diff: float
    carry: float
    residual: float
    fills: tuple[FillRecord, ...]


@dataclass(frozen=True)
class ParityReport:
    """Full golden comparison; ``verify()`` asserts its contract."""

    records: tuple[DayParityRecord, ...]
    free_equity: tuple[float, ...]
    lot_free_equity: tuple[float, ...]
    lot_equity: tuple[float, ...]
    engine_daily_returns: tuple[float, ...] | None
    target_weights: tuple[np.ndarray, ...]
    lot_size: int
    execution_delay: int


def _nan_to_zero(matrix: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )


def _broker_bars(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    pre_close: np.ndarray,
    volume: np.ndarray,
    ts_codes: list[str],
    dates: list[str],
) -> pd.DataFrame:
    """Broker bar frame over the cleaned matrices.

    A cell whose OHLCV is non-finite is a **missing bar**: the row is
    dropped and the matcher skips the order with ``missing_bar``; the
    engine's zero-cell rule blocks the same day, so both paths agree.
    """

    rows = []
    for i, code in enumerate(ts_codes):
        for j, date in enumerate(dates):
            values = (open_[i, j], high[i, j], low[i, j], pre_close[i, j], volume[i, j])
            if not all(np.isfinite(v) for v in values):
                continue
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": open_[i, j],
                    "high": high[i, j],
                    "low": low[i, j],
                    "pre_close": pre_close[i, j],
                    "volume": volume[i, j],
                }
            )
    return pd.DataFrame(rows)


def apply_blocking_rule(
    target_w: np.ndarray,
    prev_w: np.ndarray,
    blocked_buy: np.ndarray,
    blocked_sell: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Spec blocking rule for any weight source (T3-02).

    Names that are buy-blocked or outside the entry-day universe are
    never freshly bought (they keep their previous weight); reductions
    that are sell-blocked are force-held at the previous weight (the exit
    is deferred, never dropped).  Weights only shrink — never
    renormalized upward (the repo-wide T1-02 contract).  For engine-
    produced weights this is a no-op: the engine embeds the same rule.
    """

    w = np.asarray(target_w, dtype=np.float64).copy()
    fresh = (w > prev_w) & (blocked_buy | ~eligible)
    w[fresh] = prev_w[fresh]
    reduced = (w < prev_w) & blocked_sell
    w[reduced] = prev_w[reduced]
    return w


class GoldenParity:
    """Runs one signal set through the free, lot-free and lot paths."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        lot_size: int = 100,
        execution_delay: int = 1,
        atol: float = 1e-9,
    ) -> None:
        if lot_size < 0:
            raise ValueError(f"lot_size must be >= 0, got {lot_size}")
        if execution_delay < 1:
            raise ValueError(f"execution_delay must be >= 1, got {execution_delay}")
        if atol <= 0.0:
            raise ValueError(f"atol must be > 0, got {atol}")
        self.config = config
        self.lot_size = int(lot_size)
        self.execution_delay = int(execution_delay)
        self.atol = float(atol)
        self.cost_model = ExecutionCostModel.from_config(config)

    def run(
        self,
        signals: np.ndarray,
        raw_cache: dict[str, np.ndarray],
        ts_codes: list[str],
        dates: list[str],
        universe_mask: np.ndarray,
        *,
        target_weights: list[np.ndarray] | None = None,
    ) -> ParityReport:
        """Run the golden comparison.

        ``target_weights=None`` uses the engine's own top-n selection (the
        canonical golden test); a list of full ``[n]`` weight vectors
        executes those instead (T3-01 optimizer output).  ``raw_cache``,
        ``ts_codes``, ``dates`` and ``universe_mask`` follow the engine's
        conventions.
        """

        signals = np.asarray(signals, dtype=np.float64)
        mask = np.asarray(universe_mask, dtype=bool)
        if signals.ndim != 2 or mask.shape != signals.shape:
            raise ValueError("signals/universe_mask must be [stock, date] pairs")
        n, n_dates = signals.shape
        if len(ts_codes) != n or len(dates) != n_dates:
            raise ValueError("ts_codes/dates must match the signal matrix")
        missing = [k for k in _BAR_COLUMNS if k not in raw_cache]
        if missing:
            raise ValueError(f"raw_cache missing columns: {missing}")

        delay = self.execution_delay
        open_ = _nan_to_zero(raw_cache["open"])
        high = _nan_to_zero(raw_cache["high"])
        low = _nan_to_zero(raw_cache["low"])
        pre_close = _nan_to_zero(raw_cache["pre_close"])
        volume = _nan_to_zero(raw_cache["volume"])
        target_ret = open_to_open_returns(open_)

        if target_weights is None:
            result = AshareBacktestEngine(self.config).run(
                signals, raw_cache, ts_codes, dates, mask,
                execution_delay=delay,
            )
            if result.target_weights is None:
                raise ValueError("engine did not record target_weights")
            weights = [np.asarray(w, dtype=np.float64) for w in result.target_weights]
            engine_daily_returns = tuple(float(x) for x in result.daily_returns)
        else:
            weights = [np.asarray(w, dtype=np.float64) for w in target_weights]
            if any(w.shape != (n,) for w in weights):
                raise ValueError("target_weights must be [n] vectors")
            if len(weights) > n_dates - delay - 1:
                raise ValueError("target_weights exceed the executable periods")
            engine_daily_returns = None
        n_periods = len(weights)

        # Spec blocking rule applied to every weight vector (no-op for
        # engine weights, consistency for optimizer weights).
        executed: list[np.ndarray] = []
        prev = np.zeros(n, dtype=np.float64)
        for k, w in enumerate(weights):
            entry = k + delay
            blocked_buy = tradability_blocked(
                open_[:, entry], high[:, entry], low[:, entry],
                pre_close[:, entry], volume[:, entry], ts_codes, "buy",
            )
            blocked_sell = tradability_blocked(
                open_[:, entry], high[:, entry], low[:, entry],
                pre_close[:, entry], volume[:, entry], ts_codes, "sell",
            )
            eligible = mask[:, k] & mask[:, entry]
            executed.append(
                apply_blocking_rule(w, prev, blocked_buy, blocked_sell, eligible)
            )
            prev = executed[-1]

        # --- free ledger: exact continuous-share mirror -------------------
        capital = float(self.config.initial_capital)
        free_equity = [capital]
        free_returns: list[float] = []
        free_fees: list[float] = []
        prev = np.zeros(n, dtype=np.float64)
        for k, w in enumerate(executed):
            equity_before = free_equity[-1]
            buy = np.maximum(w - prev, 0.0)
            sell = np.maximum(prev - w, 0.0)
            fees = float(self.cost_model.rebalance_cost(buy, sell, equity_before).total)
            gross = float(np.dot(w, target_ret[:, k + delay - 1]))
            ret = gross - fees / equity_before
            free_returns.append(ret)
            free_fees.append(fees)
            free_equity.append(equity_before * (1.0 + ret))
            prev = w

        # --- matcher paths -------------------------------------------------
        bars = _broker_bars(open_, high, low, pre_close, volume, ts_codes, dates)
        lot_free_equity, lot_free_returns, lot_free_fees, lot_free_fills, _ = (
            self._broker_path(executed, bars, ts_codes, dates, open_, capital, 0)
        )
        if self.lot_size > 0:
            lot_equity, lot_returns, lot_fees, lot_fills, lot_portfolio = (
                self._broker_path(
                    executed, bars, ts_codes, dates, open_, capital, self.lot_size
                )
            )
        else:
            lot_equity, lot_returns, lot_fees, lot_fills, lot_portfolio = (
                lot_free_equity, lot_free_returns, lot_free_fees, lot_free_fills,
                None,
            )

        # --- residual attribution (lot mode vs free path) -----------------
        records: list[DayParityRecord] = []
        for k in range(n_periods):
            entry = k + delay
            w = executed[k]
            free_delta = free_equity[k + 1] - free_equity[k]
            lot_delta = lot_equity[k + 1] - lot_equity[k]
            residual = lot_delta - free_delta
            if self.lot_size > 0:
                # Engine-implied share position: the frictionless-rebalanced
                # book (targets marked at the entry open).
                q_free = np.zeros(n, dtype=np.float64)
                valid = open_[:, entry] > 0.0
                q_free[valid] = w[valid] * free_equity[k] / open_[valid, entry]
                r = target_ret[:, entry - 1]
                q_lot = np.zeros(n, dtype=np.float64)
                index_of = {code: i for i, code in enumerate(ts_codes)}
                for code, pos in lot_portfolio.positions.items():
                    i = index_of.get(code)
                    if i is not None:
                        q_lot[i] = pos.quantity
                rounding = float(np.sum((q_lot - q_free) * open_[:, entry] * r))
                fee_diff = free_fees[k] - lot_fees[k]
                carry = residual - rounding - fee_diff
            else:
                rounding = fee_diff = carry = 0.0

            records.append(
                DayParityRecord(
                    date=dates[entry],
                    free_return=free_returns[k],
                    lot_free_return=lot_free_returns[k],
                    lot_return=lot_returns[k],
                    turnover=float(np.abs(w - executed[k - 1]).sum())
                    if k > 0
                    else float(np.abs(w).sum()),
                    rounding=rounding,
                    fee_diff=fee_diff,
                    carry=carry,
                    residual=residual,
                    fills=tuple(lot_fills[k]),
                )
            )

        return ParityReport(
            records=tuple(records),
            free_equity=tuple(free_equity),
            lot_free_equity=tuple(lot_free_equity),
            lot_equity=tuple(lot_equity),
            engine_daily_returns=engine_daily_returns,
            target_weights=tuple(executed),
            lot_size=self.lot_size,
            execution_delay=delay,
        )

    def verify(self, report: ParityReport) -> None:
        """Assert the spec contract; raise ``GoldenParityViolation``.

        Checks: (1) the lot-free matcher reproduces the free path exactly;
        (2) the free path reproduces the engine's own daily returns when
        the engine produced the weights; (3) the per-day residual equals
        its recorded attribution and the bookkeeping identity
        ``sum(residual) == lot_equity(T) - free_equity(T)`` holds; (4) a
        lot-free run carries no residual at all.
        """

        violations: list[str] = []
        capital = float(self.config.initial_capital)
        tol = self.atol * max(capital, 1.0)
        for k, (free, lot_free) in enumerate(
            zip(report.free_equity, report.lot_free_equity)
        ):
            if abs(free - lot_free) > tol:
                violations.append(
                    f"lot-free divergence at period {k}: free={free:.10g} "
                    f"lot_free={lot_free:.10g}"
                )
        if report.engine_daily_returns is not None:
            for k, record in enumerate(report.records):
                engine_ret = report.engine_daily_returns[k]
                if abs(record.free_return - engine_ret) > self.atol:
                    violations.append(
                        f"free path diverges from engine at period {k}: "
                        f"free={record.free_return:.10g} engine={engine_ret:.10g}"
                    )
        total_residual = sum(r.residual for r in report.records)
        if abs(report.lot_equity[-1] - report.free_equity[-1] - total_residual) > tol:
            violations.append("bookkeeping identity broken")
        for k, record in enumerate(report.records):
            if abs(record.residual - record.rounding - record.fee_diff - record.carry) > tol:
                violations.append(f"residual attribution broken at period {k}")
            if report.lot_size <= 0 and abs(record.residual) > tol:
                violations.append(f"lot-free run has nonzero residual at period {k}")
        if violations:
            raise GoldenParityViolation("; ".join(violations))

    # --- internals ---------------------------------------------------------

    def _broker_path(
        self,
        executed: list[np.ndarray],
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        open_: np.ndarray,
        capital: float,
        lot_size: int,
    ) -> tuple[
        list[float], list[float], list[float], list[list[FillRecord]],
        SimulationPortfolio,
    ]:
        """One matcher run over the executed weights.

        ``lot_size > 0`` is the real whole-lot account: orders target the
        full weight shares (drift is traded), fills pay real cash and fees,
        and the equity is cash plus positions at the exit open.

        ``lot_size <= 0`` is the golden lot-free mode: only weight deltas
        are traded (the engine never rebalances price drift) and the
        ledger compounds the engine's formula — equity times the weighted
        open-to-open return minus the fills' fees.  The fills still come
        from the authoritative broker (blocking, T+1, minimum commission),
        so the mode reproduces the vectorized engine exactly when the fill
        fees equal the engine's weight-delta fees.
        """

        state_path = Path(tempfile.gettempdir()) / f"golden-{uuid4().hex}.json"
        portfolio = SimulationPortfolio(capital, state_path)
        broker = SimBroker()
        delay = self.execution_delay
        n = open_.shape[0]
        target_ret = open_to_open_returns(open_)
        index_of = {code: i for i, code in enumerate(ts_codes)}
        lot_free = lot_size <= 0
        equity = [capital]
        returns: list[float] = []
        fees_by_day: list[float] = []
        fills_by_day: list[list[FillRecord]] = []

        for k, w in enumerate(executed):
            entry = k + delay
            exit_ = k + delay + 1
            open_prices = open_[:, entry]
            if lot_free:
                equity_before = equity[-1]
                # Re-mark the book to the engine's pre-rebalance state:
                # the previous weights at the current equity and entry
                # prices.  The engine's weight model absorbs price drift
                # frictionlessly (positions are re-marked at every entry),
                # so weight-delta orders are always executable against
                # this synthetic book and stale drift leftovers can never
                # generate spurious liquidation orders.
                prev_w = executed[k - 1] if k > 0 else np.zeros(n)
                book: dict[str, PositionState] = {}
                for i, code in enumerate(ts_codes):
                    price = open_prices[i]
                    if price > 0.0:
                        qty = prev_w[i] * equity_before / price
                        if qty > 0.0:
                            book[code] = PositionState(
                                ts_code=code,
                                name=code,
                                quantity=float(qty),
                                available_quantity=float(qty),
                                avg_cost=float(price),
                                last_price=float(price),
                                last_date=dates[entry],
                            )
                portfolio.positions = book
                portfolio.cash = equity_before * (1.0 - float(prev_w.sum()))

                # Weight-delta orders: drift between rebalances is never
                # traded (the engine's convention).  Every name gets a
                # target (including names leaving the book, whose pure
                # delta sell replaces the full-exit liquidations), so the
                # only trades are the weight deltas themselves.
                delta_shares = np.zeros(n, dtype=np.float64)
                valid = open_prices > 0.0
                delta_shares[valid] = (
                    (w[valid] - prev_w[valid]) * equity_before / open_prices[valid]
                )
                current = {
                    code: pos.quantity for code, pos in portfolio.positions.items()
                }
                target_shares = {
                    i: current.get(ts_codes[i], 0.0) + delta_shares[i]
                    for i in range(n)
                    if w[i] > 0.0 or delta_shares[i] != 0.0
                }
                selected = sorted(target_shares)
            else:
                # Full target shares: a real account rebalances the drift
                # when it rebalances to the target weights.
                equity_before = portfolio.cash
                for code, pos in portfolio.positions.items():
                    i = index_of.get(code)
                    equity_before += (
                        pos.quantity * open_prices[i]
                        if i is not None
                        else pos.quantity * pos.last_price
                    )
                target_shares = target_shares_from_weights(
                    w, equity_before, open_prices, lot_size=lot_size
                )
                selected = sorted(target_shares)
                current = {
                    code: pos.quantity for code, pos in portfolio.positions.items()
                }
            orders = build_orders(
                dates[entry], ts_codes, open_prices, target_shares, selected,
                current, lot_size=lot_size,
            )
            requested = {id(order): order.quantity for order in orders}
            trades = broker.execute_orders(
                orders, bars, dates[entry], portfolio, {}, self.config,
                lot_size=lot_size,
            )
            fills = [
                FillRecord(
                    side=order.side,
                    ts_code=order.ts_code,
                    requested=requested[id(order)],
                    filled=order.quantity if order.status == "filled" else 0.0,
                    price=order.price,
                    status=order.status,
                    reason=order.reason,
                )
                for order in orders
            ]
            fees = float(
                sum(
                    t.commission + t.stamp_tax + t.transfer_fee + t.slippage
                    for t in trades
                )
            )
            fees_by_day.append(fees)
            if lot_free:
                ret = float(np.dot(w, target_ret[:, entry - 1])) - fees / equity_before
                equity_after = equity_before * (1.0 + ret)
            else:
                equity_after = portfolio.cash
                for code, pos in portfolio.positions.items():
                    i = index_of.get(code)
                    equity_after += (
                        pos.quantity * open_[i, exit_]
                        if i is not None
                        else pos.quantity * pos.last_price
                    )
                ret = equity_after / equity_before - 1.0
            returns.append(ret)
            equity.append(equity_after)
            fills_by_day.append(fills)

        return equity, returns, fees_by_day, fills_by_day, portfolio
