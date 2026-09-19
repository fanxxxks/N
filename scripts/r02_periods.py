"""P39 R02 round-2 legal-period comparison CLI.

The formal run is driven by
:func:`ashare_model.r02_matrix.run_r02_periods` — see
``docs/p39_r02_periods_draft.md``.  Exit codes follow the campaign
convention: 0 completed, 1 failure, 2 fail-closed refusal.
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
from ashare_model.r02_matrix import R02Refusal, run_r02_periods


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="P39 R02 round-2 legal-period comparison "
        "(every_5_days/5 vs every_10_days/10 at the frozen reference "
        "cell; docs/p39_r02_periods_draft.md)"
    )
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "r1_medium_a.yaml"),
        help="the frozen round-2 configuration (default: r1_medium_a)",
    )
    parser.add_argument(
        "--contract-sha256",
        required=True,
        help="sha256 of the approved p39 preregistration contract",
    )
    parser.add_argument(
        "--out-root",
        default=str(_ROOT / "data" / "r1" / "r02_periods"),
        help="measurement directory for the report copies",
    )
    parser.add_argument(
        "--wall-clock-budget-s",
        type=float,
        default=4 * 3600,
        help="p39 §6 frozen wall-clock budget (default 4h)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="ProductionGateRunner deployment tuning knob",
    )
    args = parser.parse_args(argv)
    setup_run_logging(run_name="r02_periods")
    try:
        summary = run_r02_periods(
            args.config,
            args.contract_sha256,
            args.out_root,
            wall_clock_budget_s=args.wall_clock_budget_s,
            min_eligible=args.min_eligible,
        )
    except R02Refusal as exc:
        print(f"r02_periods: refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
