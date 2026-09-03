"""Production gate checks over the local DuckDB (thin CLI).

Delegates every check to the centralized :class:`ProductionGateRunner`
(``ashare_data.gates``), which is the same gate every formal entry
(train / protocol / backtest / sim / archive / web API) runs, so this
script reports exactly what a formal run would enforce:

    G1  every configured index has historical membership intervals
        (not a current snapshot stretched over history)
    G2  every participating stock has a valid listing date
    G3  the open-session calendar covers the daily-bar window
    G4  membership intervals are non-overlapping and non-duplicate
    G5  the strict PIT universe contract resolves (dev mode may fall back)
    G6  every major backtest window has enough eligible stocks
    G7  every PIT membership interval that overlaps the daily-bar horizon
        has daily bars (no zero-bar member intervals — survivorship audit)
    G8  daily bars are fresh: the bar window ends within the tolerated
        open-session lag of the most recent completed open session
        (data-freshness contract docs/p16_data_freshness_gate_contract.md)

Usage:
    python scripts/check_production_gates.py [--min-eligible N] [--dev]
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
from ashare_data.gates import ProductionGateRunner


def _root() -> Path:
    return _ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-eligible", type=int, default=100)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="run in development mode: failures degrade the result "
        "(degraded=true) instead of exiting non-zero",
    )
    args = parser.parse_args()

    config = make_data_config(load_config(None, project_root=_root()), _root())
    result = ProductionGateRunner(config, min_eligible=args.min_eligible).run(
        mode="dev" if args.dev else "formal"
    )

    for check in result.checks:
        print(f"{'PASS' if check.ok else 'FAIL'}  {check.name}")
        print(f"      {check.detail}")
    failures = [check.name for check in result.checks if not check.ok]
    if result.degraded:
        print(
            f"\nDEGRADED (mode={result.mode}, degraded=true): "
            + (", ".join(failures) if failures else "all checks pass")
        )
    elif failures:
        print("\nGATES FAILED: " + ", ".join(failures))
        print(
            "No representative run, archive or v0.29 tag may be produced "
            "until every gate passes."
        )
        return 1
    else:
        print("\nALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
