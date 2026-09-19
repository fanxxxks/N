"""P37 R02 round-1 six-cell portfolio/cost matrix CLI.

The formal run is driven by
:func:`ashare_model.r02_matrix.run_r02_matrix` — see
``docs/p37_r02_portfolio_cost_matrix_draft.md``.  Exit codes follow
the campaign convention: 0 completed, 1 failure, 2 fail-closed refusal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_logging import setup_run_logging
from ashare_model.r02_matrix import R02Refusal, run_r02_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P37 R02 round-1 six-cell portfolio/cost matrix "
        "(holdings 10/20 x min-trade 1000/2500/5000 at the frozen "
        "every_10_days/horizon-10 cadence; "
        "docs/p37_r02_portfolio_cost_matrix_draft.md)"
    )
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "r1_medium_a.yaml"),
        help="the frozen round-1 configuration (default: r1_medium_a)",
    )
    parser.add_argument(
        "--contract-sha256",
        required=True,
        help="sha256 of the approved p37 preregistration contract (stamped "
        "into the IdeaRecord and the report provenance)",
    )
    parser.add_argument(
        "--out-root",
        default=str(_ROOT / "data" / "r1" / "r02_matrix"),
        help="measurement directory for the report copies",
    )
    parser.add_argument(
        "--wall-clock-budget-s",
        type=float,
        default=6 * 3600,
        help="p37 §8 frozen wall-clock budget (default 6h)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="ProductionGateRunner deployment tuning knob (default: the "
        "gates' own MIN_ELIGIBLE_DEFAULT)",
    )
    args = parser.parse_args(argv)
    setup_run_logging(run_name="r02_matrix")
    try:
        summary = run_r02_matrix(
            args.config,
            args.contract_sha256,
            args.out_root,
            wall_clock_budget_s=args.wall_clock_budget_s,
            min_eligible=args.min_eligible,
        )
    except R02Refusal as exc:
        print(f"r02_matrix: refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
