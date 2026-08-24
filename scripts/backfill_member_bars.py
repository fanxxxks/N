"""Backfill daily bars for historical PIT members with zero-bar intervals.

Delisted, merged and otherwise never-synced historical constituents leave
zero-bar membership intervals, which the production gate (G7 in
``ashare_data.gates``) rejects.  This script audits those intervals and
fetches the missing history through the repo's own AkShare client (the
Tencent fallback serves delisted stocks) into both the DuckDB daily table
and the parquet cache.

Usage:
    python scripts/backfill_member_bars.py [--config config/ashare_config.yaml]

Idempotent: only members whose intervals overlap the data horizon with
zero bars are fetched; a re-run audits the current state.  Exits non-zero
when any zero-bar interval remains unfetchable.
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

from ashare_data.config import load_config, make_data_config
from ashare_data.sync import backfill_member_bars
from ashare_logging import export_log_txt, setup_run_logging


def _root() -> Path:
    return _ROOT


def main() -> int:
    setup_run_logging(run_name="backfill_member_bars")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--provider",
        choices=["auto", "eastmoney", "sina"],
        default=None,
        help="daily-bar provider chain override (default: config "
        "daily_provider; the Tencent fallback always runs last)",
    )
    args = parser.parse_args()
    try:
        root = _root()
        config = make_data_config(load_config(args.config, project_root=root), root)
        if args.provider:
            config.daily_provider = args.provider
        result = backfill_member_bars(config)
        print(
            f"zero-bar member codes audited: {result['audited_zero_bar_codes']}, "
            f"fetched: {result['fetched_codes']}, "
            f"daily rows: {result['daily_rows']}, "
            f"failures: {len(result['failures'])}"
        )
        if result["failures"]:
            print("unfetchable codes: " + ", ".join(result["failures"][:20]))
        print(f"remaining zero-bar intervals: {result['remaining_zero_bar_intervals']}")
        if result["remaining_zero_bar_intervals"]:
            print("run scripts/check_production_gates.py to confirm the audit")
            return 1
        print("member bar audit clean")
        return 0
    finally:
        export_log_txt(run_name="backfill_member_bars")


if __name__ == "__main__":
    sys.exit(main())
