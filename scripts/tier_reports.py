"""Tier-grouped diagnostics and ablation reports (P2-05).

Runs the diagnostics chain for each tier set (A / A+B / all) and the
formula-generator ablation over the same three sets, then writes one
versioned report to ``data/tier_report.json``.  Every ablation formula
carries its data-tier trace, so any formula in the report can be traced
back to the data it depends on.

Usage:
    python scripts/tier_reports.py --steps 50 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# When invoked as `python scripts/tier_reports.py`, sys.path[0] is the
# scripts dir; make the package roots importable first.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
)
from ashare_logging import export_log_txt, setup_run_logging
from ashare_model.data_loader import AshareDataLoader
from ashare_model.tier_reports import (
    TIER_SETS,
    assemble_tier_report,
    run_tier_ablation,
    run_tier_diagnostics,
)
from loguru import logger


def main() -> None:
    setup_run_logging(run_name="tier_reports")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="diagnostics only (no training runs)",
    )
    parser.add_argument("--output", default="data/tier_report.json")
    args = parser.parse_args()
    try:
        raw = load_config(args.config, project_root=_ROOT)
        data_config = make_data_config(raw, _ROOT)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()

        diagnostics = {}
        for tier_set in TIER_SETS:
            label = "".join(tier_set) if tier_set != ("A", "B", "C") else "all"
            diagnostics[label] = run_tier_diagnostics(
                loader, backtest_config.train_end_date, tier_set
            )
            logger.info(f"diagnostics({label}): {diagnostics[label]['feature_count']} features")

        ablation = {}
        if not args.skip_ablation:
            ablation = run_tier_ablation(
                loader,
                data_config,
                model_config,
                backtest_config,
                steps=args.steps,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            for label, run in ablation.items():
                logger.info(
                    f"ablation({label}): reward={run['best_reward']:.3f} "
                    f"formula={run['best_formula']} "
                    f"tier={run['formula_data_tier']}"
                )

        payload = assemble_tier_report(
            diagnostics=diagnostics,
            ablation=ablation,
            dataset_id=loader.dataset_id,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        out_path = _ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Tier report written to {out_path}")

        print(f"\nTier report written to {out_path}:")
        for label, report in diagnostics.items():
            print(f"  diagnostics[{label}]: {report['feature_count']} features")
        if ablation:
            baseline = ablation["all"]["best_reward"]
            for label in ("AB", "A"):
                run = ablation[label]
                print(
                    f"  ablation[{label}]: reward={run['best_reward']:8.3f} "
                    f"delta={run['delta_vs_baseline']:+8.3f} "
                    f"tier={run['formula_data_tier']['max_tier'] if run['formula_data_tier'] else None}"
                )
    finally:
        export_log_txt(run_name="tier_reports")


if __name__ == "__main__":
    main()
