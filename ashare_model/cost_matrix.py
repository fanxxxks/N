"""Capital x positions x turnover fee-matrix report (P1-02).

A pure cost-diagnostic report over the single fee schedule the whole
project shares (``BacktestConfig`` — the same source the backtest engine
and the simulator consume).  It answers the P1 acceptance question "how
many positions can a 100k / 200k / 500k account afford" as a *capacity*
number under a pre-registered annual cost budget, and lists every
(capital, positions, turnover) structure whose annual fee drag stays
within that budget.  It makes no return or alpha claim of any kind.

Contract (see docs/phase5_measurement_log.md §2, FEE_MATRIX_VERSION 1):

* One round trip of one position costs: buy commission + sell commission +
  stamp tax (sell side) + 2 x transfer fee + 2 x slippage; each commission
  leg is ``max(per_order * commission_rate, min_commission)`` — the
  minimum-commission floor applies per order.  The arithmetic is NOT
  re-derived here: ``round_trip_cost`` delegates to the project's single
  fee authority, ``ashare_execution.ExecutionCostModel``
  (``buy_cost(n).total + sell_cost(n).total``).  A zero-notional position
  (outside the report domain — ``build_fee_matrix`` rejects capital <= 0)
  therefore costs zero: no order, no minimum commission.
* T is the annual round trips per position, so the total annual traded
  notional is ``T * capital``.
* ``annual_drag`` returns (annual yuan, percent of capital).
* Acceptability (pre-registered): ``drag_pct <= budget_pct`` (default
  1.5%/year).  ``capacity`` is the largest N in the grid whose drag is
  within the budget at a given turnover (0 when none — a 0-cell means
  "no position count in the grid is affordable at this turnover");
  ``recommended`` is the capacity cell at ``default_turnover`` (None when
  capacity is 0); ``feasible_structures`` lists every within-budget cell.

The CLI writes ``data/fee_matrix.json`` and prints the capacity table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from ashare_data.config import BacktestConfig, load_config, make_backtest_config
from ashare_execution import ExecutionCostModel
from ashare_logging import export_log_txt, setup_run_logging

FEE_MATRIX_VERSION = 1

# Default grids of the P1 diagnostic (the acceptance capitals are 10万 /
# 20万 / 50万).  Turnover = annual round trips per position.
DEFAULT_CAPITALS = [100_000, 200_000, 500_000]
DEFAULT_POSITIONS = [5, 10, 15, 20, 30, 50]
DEFAULT_TURNOVERS = [1, 2, 4, 6, 12, 26]
DEFAULT_TURNOVER = 6  # ~one rebalance every two months
DEFAULT_BUDGET_PCT = 1.5  # pre-registered annual cost-drag budget (%)


def round_trip_cost(
    capital: float, positions: int, bt_cfg: BacktestConfig
) -> float:
    """Yuan cost of one full buy-and-sell round trip of one position.

    Delegates to the single fee authority ``ExecutionCostModel``: the cost
    is ``buy_cost(per_order).total + sell_cost(per_order).total`` — per-leg
    commission floor, sell-side stamp tax, two-sided transfer fee and
    slippage.  A zero-notional position costs zero (no order is placed).
    """

    per_order = capital / positions
    model = ExecutionCostModel.from_config(bt_cfg)
    return float(model.buy_cost(per_order).total) + float(
        model.sell_cost(per_order).total
    )


def annual_drag(
    capital: float, positions: int, turnover: int, bt_cfg: BacktestConfig
) -> tuple[float, float]:
    """(annual cost in yuan, annual cost as % of capital) for the
    structure (positions, turnover)."""

    annual = round_trip_cost(capital, positions, bt_cfg) * positions * turnover
    return annual, annual / capital * 100.0


def build_fee_matrix(
    capitals: list[float],
    positions: list[int],
    turnovers: list[int],
    bt_cfg: BacktestConfig,
    budget_pct: float = DEFAULT_BUDGET_PCT,
    default_turnover: int = DEFAULT_TURNOVER,
) -> dict:
    """Build the full fee-matrix report payload (JSON-serializable)."""

    if budget_pct <= 0:
        raise ValueError("budget_pct must be positive")
    if any(c <= 0 for c in capitals):
        raise ValueError("capitals must be positive")
    if any(p <= 0 for p in positions):
        raise ValueError("positions must be positive")
    if any(t <= 0 for t in turnovers):
        raise ValueError("turnovers must be positive")

    cells: dict[str, dict[str, dict[str, dict]]] = {}
    capacity: dict[str, dict[str, int]] = {}
    feasible: list[dict] = []
    for capital in capitals:
        cap_key = str(capital)
        cells[cap_key] = {str(n): {} for n in positions}
        capacity[cap_key] = {}
        for turnover in turnovers:
            turn_key = str(turnover)
            last_feasible = 0
            for n_positions in positions:
                annual, drag_pct = annual_drag(capital, n_positions, turnover, bt_cfg)
                cells[cap_key][str(n_positions)][turn_key] = {
                    "round_trip_yuan": round_trip_cost(
                        capital, n_positions, bt_cfg
                    ),
                    "annual_yuan": annual,
                    "drag_pct": drag_pct,
                }
                if drag_pct <= budget_pct:
                    last_feasible = n_positions
                    feasible.append(
                        {
                            "capital": capital,
                            "positions": n_positions,
                            "turnover": turnover,
                            "annual_yuan": annual,
                            "drag_pct": drag_pct,
                        }
                    )
            capacity[cap_key][turn_key] = last_feasible

    recommended: dict[str, dict | None] = {}
    for capital in capitals:
        cap_key = str(capital)
        n_positions = capacity[cap_key][str(default_turnover)]
        if n_positions == 0:
            recommended[cap_key] = None
            continue
        annual, drag_pct = annual_drag(capital, n_positions, default_turnover, bt_cfg)
        recommended[cap_key] = {
            "positions": n_positions,
            "turnover": default_turnover,
            "annual_yuan": annual,
            "drag_pct": drag_pct,
        }

    return {
        "version": FEE_MATRIX_VERSION,
        "fee_params": {
            "commission_rate": bt_cfg.commission_rate,
            "min_commission": bt_cfg.min_commission,
            "stamp_tax_rate": bt_cfg.stamp_tax_rate,
            "transfer_fee_rate": bt_cfg.transfer_fee_rate,
            "slippage_rate": bt_cfg.slippage_rate,
        },
        "grid": {
            "capitals": list(capitals),
            "positions": list(positions),
            "turnovers": list(turnovers),
            "budget_pct": budget_pct,
            "default_turnover": default_turnover,
        },
        "cells": cells,
        "capacity": capacity,
        "recommended": recommended,
        "feasible_structures": feasible,
    }


def main(argv=None) -> int:
    setup_run_logging(run_name="cost_matrix")
    parser = argparse.ArgumentParser(
        description="capital x positions x turnover fee-matrix report (P1-02)"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data/fee_matrix.json")
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        bt_cfg = make_backtest_config(raw)
        report = build_fee_matrix(
            DEFAULT_CAPITALS, DEFAULT_POSITIONS, DEFAULT_TURNOVERS, bt_cfg
        )
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Fee matrix written to {out_path}")
        for capital in DEFAULT_CAPITALS:
            cap_key = str(capital)
            rec = report["recommended"][cap_key]
            if rec is None:
                print(
                    f"capital={capital:>9,}  no affordable structure at "
                    f"turnover={DEFAULT_TURNOVER} within budget "
                    f"{DEFAULT_BUDGET_PCT}%/yr"
                )
            else:
                print(
                    f"capital={capital:>9,}  capacity positions="
                    f"{rec['positions']:>2}  turnover={rec['turnover']:>2}  "
                    f"drag={rec['drag_pct']:.3f}%/yr"
                )
        print(
            f"feasible_structures: {len(report['feasible_structures'])} "
            f"cells within budget"
        )
    finally:
        export_log_txt(run_name="cost_matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
