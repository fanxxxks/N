"""CLI shim for the p42 R03 combination-weight run (exit-code contract
mirrors scripts/r02_matrix.py: 0 = verdict produced, 1 = refusal, 2 =
usage)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_logging import setup_run_logging
from ashare_model.r03_combos import R03Refusal, run_r03_matrix


def main(argv: list[str] | None = None) -> int:
    setup_run_logging(run_name="r03_combos")

    parser = argparse.ArgumentParser(
        description=(
            "R03 combination-weight panel (p42): fitted shrinkage weights "
            "vs the equal-weight combo on the p37 n10 stable region"
        )
    )
    parser.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "r1_medium_a.yaml"),
        help="the frozen round-1 configuration (default: r1_medium_a)",
    )
    parser.add_argument(
        "--contract-sha256",
        required=True,
        help="sha256 of docs/p42_r03_combo_weights_draft.md (self-verified)",
    )
    parser.add_argument(
        "--out-root",
        default=str(_ROOT / "data" / "r1" / "r03_combos"),
    )
    parser.add_argument("--min-eligible", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        summary = run_r03_matrix(
            args.config,
            contract_sha256=args.contract_sha256,
            out_root=args.out_root,
            min_eligible=args.min_eligible,
        )
    except R03Refusal as exc:
        print(f"r03_combos: refused: {exc}", file=sys.stderr)
        return 1
    print(
        f"verdict={summary['verdict']} state={summary['state']} "
        f"rows={summary['row_count']} wall={summary['wall_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
