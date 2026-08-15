"""Family-level ablation experiments.

For each factor family the script retrains the formula generator with that
family neutralized (same seed, steps and batch as the baseline), measuring
how much the best validation reward depends on the family.  This is the
objective gate for "is this factor family pulling its weight".

Usage:
    python scripts/ablate_families.py --steps 50 --batch-size 256
    python scripts/ablate_families.py --families fundamental,technical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# When invoked as `python scripts/ablate_families.py`, sys.path[0] is the
# scripts dir; make the package roots importable first.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
)
from ashare_logging import export_log_txt, setup_run_logging
from ashare_model.data_loader import AshareDataLoader
from ashare_model.diagnostics import feature_family
from ashare_model.factors import ablate_factors
from ashare_model.train import AshareTrainer
from ashare_model.vocab import FEATURE_NAMES
from loguru import logger


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _families_present() -> list[str]:
    families = []
    for name in FEATURE_NAMES:
        family = feature_family(name)
        if family not in families:
            families.append(family)
    return families


def _train_one(
    loader: AshareDataLoader,
    tensor: np.ndarray,
    data_config,
    model_config,
    backtest_config,
    steps: int,
    batch_size: int,
    seed: int,
) -> tuple[float, str]:
    loader.factor_tensor = torch.tensor(tensor, dtype=torch.float32)
    trainer = AshareTrainer(
        data_config, model_config, backtest_config, loader=loader
    )
    trainer.train(
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        save_artifacts=False,
    )
    return float(trainer.best_reward), trainer.best_formula


def main() -> None:
    setup_run_logging(run_name="ablation")
    parser = argparse.ArgumentParser(description="Factor family ablation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated families to ablate (default: all present)",
    )
    parser.add_argument("--output", default="data/ablation_results.json")
    args = parser.parse_args()

    try:
        root = _project_root()
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        full = loader.factor_tensor.numpy()

        families = (
            [f.strip() for f in args.families.split(",") if f.strip()]
            if args.families
            else _families_present()
        )

        results: dict[str, dict[str, object]] = {}
        baseline_reward, baseline_formula = _train_one(
            loader, full, data_config, model_config, backtest_config,
            args.steps, args.batch_size, args.seed,
        )
        results["baseline"] = {
            "best_reward": baseline_reward,
            "best_formula": baseline_formula,
        }
        logger.info(f"baseline: reward={baseline_reward:.3f} formula={baseline_formula}")

        for family in families:
            excluded = [
                name for name in FEATURE_NAMES if feature_family(name) == family
            ]
            tensor = ablate_factors(full, excluded)
            reward, formula = _train_one(
                loader, tensor, data_config, model_config, backtest_config,
                args.steps, args.batch_size, args.seed,
            )
            results[f"ablate_{family}"] = {
                "family": family,
                "excluded": excluded,
                "best_reward": reward,
                "best_formula": formula,
                "delta_vs_baseline": round(reward - baseline_reward, 4),
            }
            logger.info(
                f"ablate {family}: reward={reward:.3f} "
                f"(delta={reward - baseline_reward:+.3f}) formula={formula}"
            )

        payload = {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "baseline_reward": baseline_reward,
            "results": results,
        }
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Ablation results written to {out_path}")

        print("\nFamily ablation (validation reward; baseline "
              f"{baseline_reward:.3f}):")
        for family in families:
            row = results[f"ablate_{family}"]
            print(
                f"  -{family:16s} reward={row['best_reward']:8.3f} "
                f"delta={row['delta_vs_baseline']:+8.3f}"
            )
    finally:
        export_log_txt(run_name="ablation")


if __name__ == "__main__":
    main()
