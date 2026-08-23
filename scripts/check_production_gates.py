"""Phase-9 production gate checks over the local DuckDB.

Every formal entry (train / protocol / diagnostics / backtest / sim /
archive) is gated on the same point-in-time universe contract.  This
script verifies the six gates explicitly and exits non-zero when any of
them fails:

    G1  every configured index has historical membership intervals
        (not a current snapshot stretched over history)
    G2  every participating stock has a valid listing date
    G3  the open-session calendar covers the daily-bar window
    G4  membership intervals are non-overlapping and non-duplicate
    G5  the loader builds the PIT mask in strict mode (no dev fallback)
    G6  every major backtest window has enough eligible stocks

Usage:
    python scripts/check_production_gates.py [--min-eligible N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Scripts live one level below the package root: make `ashare_*` importable
# regardless of the caller's sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from ashare_data.config import load_config, make_data_config
from ashare_data.db import AshareDB
from ashare_data.universe import (
    UniverseContractError,
    membership_interval_issues,
    require_production_universe,
)
from ashare_model.data_loader import AshareDataLoader


def _root() -> Path:
    return _ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-eligible", type=int, default=100)
    args = parser.parse_args()

    config = make_data_config(load_config(None, project_root=_root()), _root())
    failures: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    with AshareDB(config.duckdb_path, read_only=True) as db:
        constituents = db.query(f"SELECT * FROM {config.constituents_table}")
        stocks = db.query(f"SELECT * FROM {config.stocks_table}")
        calendar = db.query(
            f"SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx "
            f"FROM {config.calendar_table} WHERE is_open=true"
        ).iloc[0]
        daily = db.query(
            f"SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx "
            f"FROM {config.daily_table}"
        ).iloc[0]

    # G1: historical intervals per configured index, not one stretched snapshot.
    in_dates = constituents["in_date"].astype(str).str.replace("-", "")
    out_dates = constituents["out_date"].astype(str).str.replace("-", "")
    snapshot_shaped = (
        len(constituents) >= 2
        and in_dates.nunique() == 1
        and out_dates.nunique() == 1
        and out_dates.iloc[0] == "99991231"
    )
    per_index = {
        code: int((constituents["index_code"].astype(str) == code).sum())
        for code in config.index_codes
    }
    g1_ok = not snapshot_shaped and all(per_index.get(c, 0) > 0 for c in config.index_codes)
    checks.append((
        "G1 historical member intervals per configured index",
        g1_ok,
        f"rows per index: {per_index}; snapshot-shaped={snapshot_shaped}",
    ))

    # G2: valid list dates for every participating stock.
    member_codes = set(constituents["ts_code"].astype(str))
    stock_table = stocks.copy()
    stock_table["ts_code"] = stock_table["ts_code"].astype(str)
    missing_rows = sorted(member_codes - set(stock_table["ts_code"]))
    null_dates = sorted(
        stock_table.loc[
            stock_table["ts_code"].isin(member_codes) & stock_table["list_date"].isna(),
            "ts_code",
        ].tolist()
    )
    g2_ok = not missing_rows and not null_dates
    checks.append((
        "G2 valid list_date for all participating stocks",
        g2_ok,
        f"missing stock rows: {len(missing_rows)}, null list_date: {len(null_dates)}",
    ))

    # G3: calendar covers the bar window.
    g3_ok = str(calendar["mn"]) <= str(daily["mn"]) and str(calendar["mx"]) >= str(daily["mx"])
    checks.append((
        "G3 open calendar covers the daily-bar window",
        g3_ok,
        f"calendar {calendar['mn']}..{calendar['mx']} vs daily {daily['mn']}..{daily['mx']}",
    ))

    # G4: interval structure.
    issues = membership_interval_issues(constituents)
    checks.append((
        "G4 no duplicate/overlapping intervals",
        not issues,
        "; ".join(issues[:2]) if issues else "clean",
    ))

    # G5: strict-mode loader builds the mask (the authoritative contract gate).
    try:
        status = require_production_universe(config)
        loader = AshareDataLoader(config)
        loader.load_data()
        mask = loader.universe_mask
        assert mask is not None
        eligible_per_day = mask.sum(axis=0)
        checks.append((
            "G5 strict-mode loader builds the PIT mask",
            True,
            f"mode={status.mode}, dates={mask.shape[1]}, "
            f"mean eligible/day={float(eligible_per_day.mean()):.1f}",
        ))
    except (UniverseContractError, AssertionError) as exc:
        checks.append(("G5 strict-mode loader builds the PIT mask", False, str(exc)))
        eligible_per_day = None

    # G6: enough eligible stocks in every major backtest window (yearly slices).
    if eligible_per_day is not None:
        dates = loader.dates
        year_min = {}
        for i, day in enumerate(eligible_per_day):
            year_min.setdefault(dates[i][:4], []).append(int(day))
        minima = {year: min(values) for year, values in sorted(year_min.items())}
        g6_ok = all(v >= args.min_eligible for v in minima.values())
        checks.append((
            f"G6 >= {args.min_eligible} eligible stocks per major window",
            g6_ok,
            "min eligible per year: " + ", ".join(
                f"{y}:{v}" for y, v in minima.items()
            ),
        ))
    else:
        checks.append(("G6 eligible-stock minimum", False, "mask unavailable (G5 failed)"))

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        print(f"      {detail}")
    for name, ok, _ in checks:
        if not ok:
            failures.append(name)
    if failures:
        print("\nGATES FAILED: " + ", ".join(failures))
        print(
            "No representative run, archive or v0.29 tag may be produced "
            "until every gate passes."
        )
        return 1
    print("\nALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
