"""Analyze AlphaGPT sim trades: fee drag, gross P&L, and anomalies.

Every figure derives from the per-day trade files and the portfolio state
file; paths and the initial-capital fallback come from the sim config, so
nothing (initial capital, final cash) is hardcoded.  The cash-accounting
reconstruction (initial - buys + sells - fees) is cross-checked against the
cash recorded in the state file and a mismatch is reported as a warning.

Run from anywhere: python scripts/analyze_sim.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ashare_data.config import load_config, make_sim_config


@dataclass
class TradeSummary:
    """Aggregates over all non-empty per-day trade files."""

    days: int
    trades: int
    buy_amount: float
    sell_amount: float
    total_amount: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    slippage: float
    fees_by_year: dict[str, float]

    @property
    def total_fees(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee + self.slippage


@dataclass
class StateSnapshot:
    """Facts read from the portfolio state file."""

    initial_capital: float
    final_cash: float
    final_equity: float | None


def load_trades(trades_dir: Path) -> list[tuple[str, list[dict]]]:
    """Return (day_prefix, rows) pairs for every non-empty trade file."""
    days: list[tuple[str, list[dict]]] = []
    files = sorted(glob.glob(str(trades_dir / "*.json")))
    for filename in files:
        path = Path(filename)
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Skipping unreadable trade file {path}: {exc}")
            continue
        if not rows:
            continue
        days.append((path.stem[:8], rows))
    return days


def summarize_trades(days: list[tuple[str, list[dict]]]) -> TradeSummary:
    """Aggregate trade rows; missing or null fee fields count as zero."""
    summary = TradeSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    for day, rows in days:
        for trade in rows:
            summary.trades += 1
            amount = float(trade.get("amount") or 0.0)
            commission = float(trade.get("commission") or 0.0)
            stamp = float(trade.get("stamp_tax") or 0.0)
            transfer = float(trade.get("transfer_fee") or 0.0)
            slippage = float(trade.get("slippage") or 0.0)
            summary.total_amount += amount
            summary.commission += commission
            summary.stamp_tax += stamp
            summary.transfer_fee += transfer
            summary.slippage += slippage
            year = day[:4]
            summary.fees_by_year[year] = (
                summary.fees_by_year.get(year, 0.0)
                + commission
                + stamp
                + transfer
                + slippage
            )
            if trade.get("side") == "buy":
                summary.buy_amount += amount
            else:
                summary.sell_amount += amount
        summary.days += 1
    return summary


def load_state_snapshot(
    state_path: Path, fallback_initial_capital: float
) -> StateSnapshot | None:
    """Read initial capital / final cash / final equity; None if unreadable."""
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        initial = float(payload.get("initial_capital") or fallback_initial_capital)
        cash = float(payload.get("cash") or initial)
        history = payload.get("equity_history") or []
        equity = float(history[-1]["equity"]) if history else None
        return StateSnapshot(
            initial_capital=initial, final_cash=cash, final_equity=equity
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read portfolio state {state_path}: {exc}")
        return None


def render_report(
    summary: TradeSummary,
    state: StateSnapshot | None,
    fallback_initial_capital: float,
    state_path: Path,
) -> str:
    """Build the human-readable report; every number derives from the inputs."""
    initial = state.initial_capital if state else fallback_initial_capital
    lines: list[str] = []
    avg = summary.trades / summary.days if summary.days else 0.0
    lines.append(f"days with trades: {summary.days}")
    lines.append(
        f"total trades: {summary.trades}  avg trades/trading-day: {round(avg, 1)}"
    )
    lines.append(f"buy amount:  {summary.buy_amount:>15,.0f}")
    lines.append(f"sell amount: {summary.sell_amount:>15,.0f}")
    lines.append(f"total amount:{summary.total_amount:>15,.0f}")
    lines.append(
        f"fees: commission={summary.commission:,.0f} stamp={summary.stamp_tax:,.0f} "
        f"transfer={summary.transfer_fee:,.0f} slippage={summary.slippage:,.0f} "
        f"TOTAL={summary.total_fees:,.0f}"
    )
    implied_gross = initial + summary.sell_amount - summary.buy_amount
    implied_net = implied_gross - summary.total_fees
    if state:
        lines.append(
            f"initial {initial:,.0f} -> final cash {state.final_cash:,.2f} "
            f"(state file); implied gross pnl (no fees) = {implied_gross:,.0f}; "
            f"with fees = {implied_net:,.2f}"
        )
        if state.final_equity is not None:
            lines.append(
                f"final equity (state equity_history tail): {state.final_equity:,.2f}"
            )
    else:
        lines.append(
            f"state file not found at {state_path}; final cash unknown. "
            f"implied gross pnl (no fees) = {implied_gross:,.0f}; "
            f"with fees = {implied_net:,.2f}"
        )
    drag = summary.total_fees / initial * 100 if initial else 0.0
    lines.append(f"cumulative fee drag = {drag:.1f}% of initial capital")
    lines.append("fees by year:")
    for year in sorted(summary.fees_by_year):
        lines.append(f"  {year}: {summary.fees_by_year[year]:>10,.0f}")
    lines.append("")
    reconstructed = (
        initial - summary.buy_amount + summary.sell_amount - summary.total_fees
    )
    lines.append(
        "cash accounting: initial - buy_amount + sell_amount - fees = "
        f"{reconstructed:,.2f}"
    )
    if state:
        drift = abs(reconstructed - state.final_cash)
        if drift > 0.01:
            lines.append(
                "WARNING: trade-file reconstruction differs from state cash "
                f"by {drift:,.2f}"
            )
        else:
            lines.append(f"matches state cash {state.final_cash:,.2f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze AlphaGPT sim trades: fees, gross P&L, cash accounting."
    )
    parser.add_argument(
        "--trades-dir",
        type=Path,
        default=None,
        help="Per-day trade JSON directory (default: config sim.trades_dir)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Portfolio state JSON (default: config sim.state_path)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw = load_config(project_root=REPO_ROOT)
    sim_config = make_sim_config(raw, REPO_ROOT)
    trades_dir = args.trades_dir or sim_config.trades_dir
    state_path = args.state or sim_config.state_path
    days = load_trades(trades_dir)
    summary = summarize_trades(days)
    state = load_state_snapshot(state_path, sim_config.initial_capital)
    print(render_report(summary, state, sim_config.initial_capital, state_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
