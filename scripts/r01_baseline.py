"""P34 R01 round-1 fixed-baseline diagnostic CLI.

The formal run is driven by
:func:`ashare_model.r01_baseline.run_r01_baseline` — see
``docs/p34_r01_fixed_baseline_diagnostic_draft.md``.  Exit codes follow
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
from ashare_model.r01_baseline import R01Refusal, run_r01_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P34 R01 round-1 fixed-baseline diagnostic (baselines "
        "before search; docs/p34_r01_fixed_baseline_diagnostic_draft.md)"
    )
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "r1_medium_a.yaml"),
        help="the frozen round-1 configuration (default: r1_medium_a)",
    )
    parser.add_argument(
        "--contract-sha256",
        required=True,
        help="sha256 of the approved p34 preregistration contract (stamped "
        "into the IdeaRecord and the report provenance)",
    )
    parser.add_argument(
        "--out-root",
        default=str(_ROOT / "data" / "r1" / "r01_baseline"),
        help="measurement directory for the report copies",
    )
    parser.add_argument(
        "--wall-clock-budget-s",
        type=float,
        default=6 * 3600,
        help="p34 §3.6 frozen wall-clock budget (default 6h)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="ProductionGateRunner deployment tuning knob (default: the "
        "gates' own MIN_ELIGIBLE_DEFAULT)",
    )
    args = parser.parse_args(argv)
    setup_run_logging(run_name="r01_baseline")
    try:
        summary = run_r01_baseline(
            args.config,
            args.contract_sha256,
            args.out_root,
            wall_clock_budget_s=args.wall_clock_budget_s,
            min_eligible=args.min_eligible,
        )
    except R01Refusal as exc:
        print(f"r01_baseline: refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
