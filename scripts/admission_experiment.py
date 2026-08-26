"""RL admission experiment (T2-03): is RL worth keeping as the default
searcher?

For each of >= 5 independent policy initializations (``INIT_SEEDS``):

1. train RL on the admission tier (40 steps x 128) with the fixed
   sampling seed and record its best-so-far curve and its **actual**
   unique-semantic-evaluation budget (the v18 ledger);
2. run random / GP / TPE with exactly that budget per seed;
3. measure the normalized best-so-far area and the OOS active IR of each
   searcher's selection on the fold's test window.

The pre-registered rule (ashare_model/admission.py, also in
docs/phase2_measurement_log.md) decides whether RL keeps the default
searcher slot; the verdict and the raw per-seed measurements are written
to ``experiments/admission_<date>.json`` and printed.

Usage:
    python scripts/admission_experiment.py [--config config/ashare_config.yaml]
        [--fold 0] [--output experiments/admission_YYYYMMDD.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_reward_config,
)
from ashare_data.gates import ProductionGateRunner
from ashare_logging import export_log_txt, setup_run_logging

from ashare_model.admission import (
    ADMISSION_BATCH,
    ADMISSION_STEPS,
    INIT_SEEDS,
    best_so_far_area,
    decide_admission,
)
from ashare_model.baseline_harness import oos_active_ir
from ashare_model.data_loader import AshareDataLoader
from ashare_model.evaluation import (
    evaluate_formula,
    resolve_folds,
    run_gp_search,
    run_random_search,
    run_tpe_search,
)
from ashare_model.train import AshareTrainer


def main(argv=None) -> int:
    setup_run_logging(run_name="admission_experiment")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--output",
        default="experiments/admission_experiment.json",
        help="JSON output path (relative to the repo root)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window",
    )
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(
            data_config, min_eligible=args.min_eligible
        ).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        proto_cfg = make_protocol_config(raw)
        fold_cfg = proto_cfg.folds[args.fold]

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        fold = resolve_folds([fold_cfg], loader.dates)[0]

        rl_areas: list[float] = []
        rl_irs: list[float] = []
        baseline_areas: dict[str, list[float]] = {"random": [], "gp": [], "tpe": []}
        baseline_irs: dict[str, list[float]] = {"random": [], "gp": [], "tpe": []}
        per_seed: list[dict] = []

        for init_seed in INIT_SEEDS:
            print(f"=== init_seed={init_seed} ===")
            trainer = AshareTrainer(
                data_config,
                model_config,
                backtest_config,
                loader,
                reward_config,
                init_seed=init_seed,
            )
            tokens = trainer.train(
                steps=ADMISSION_STEPS,
                batch_size=ADMISSION_BATCH,
                seed=42,
                save_artifacts=False,
                train_end_date=fold.train_end,
            )
            rl_budget = int(trainer.semantic_cache.budget_used)
            rl_curve = [
                (float(h["unique_semantic_evals"]), float(h["best_val_reward"]))
                for h in trainer.history
            ]
            rl_area = best_so_far_area(rl_curve, rl_budget)
            rl_irs.append(_oos_ir(tokens, trainer, loader, fold, backtest_config))
            rl_areas.append(rl_area)

            seed_rows: dict[str, dict] = {}
            for name, runner in (
                ("random", lambda: run_random_search(
                    loader, model_config, backtest_config, reward_config, fold,
                    n_samples=rl_budget, seed=int(proto_cfg.random_seed),
                    budget=rl_budget,
                )),
                ("gp", lambda: run_gp_search(
                    loader, model_config, backtest_config, reward_config, fold,
                    seed=int(proto_cfg.gp_seed), budget=rl_budget,
                )),
                ("tpe", lambda: run_tpe_search(
                    loader, model_config, backtest_config, reward_config, fold,
                    seed=int(proto_cfg.tpe_seed), budget=rl_budget,
                )),
            ):
                row = runner()
                curve = row.get("best_so_far") or []
                area = best_so_far_area(curve, rl_budget)
                baseline_areas[name].append(area)
                ir = _row_oos_ir(row, loader, fold, backtest_config)
                baseline_irs[name].append(ir)
                seed_rows[name] = {
                    "failed": row.get("failed", False),
                    "budget_used": row.get("unique_semantic_evals"),
                    "area": area,
                    "oos_ir": ir,
                }
                print(
                    f"  {name:6s} budget={rl_budget} "
                    f"area={area:+.3f} oos_ir={ir:+.3f}"
                )

            per_seed.append(
                {
                    "init_seed": int(init_seed),
                    "rl_budget": rl_budget,
                    "rl_area": rl_area,
                    "rl_oos_ir": rl_irs[-1],
                    "rl_formula": (trainer.best_formula or ""),
                    "baselines": seed_rows,
                }
            )
            print(
                f"  rl     budget={rl_budget} area={rl_area:+.3f} "
                f"oos_ir={rl_irs[-1]:+.3f}"
            )

        verdict = decide_admission(
            rl_areas=rl_areas,
            rl_oos_irs=rl_irs,
            baseline_areas=baseline_areas,
            baseline_oos_irs=baseline_irs,
        )
        payload = {
            "experiment": "t2-03-rl-admission",
            "fold": args.fold,
            "fold_train_end": fold.train_end,
            "fold_test_end": fold.test_end,
            "admission_tier": {"steps": ADMISSION_STEPS, "batch_size": ADMISSION_BATCH},
            "init_seeds": list(INIT_SEEDS),
            "sampling_seed": 42,
            "dataset_id": loader.dataset_id,
            "protocol_version": "19",
            "seeds": per_seed,
            "metrics": {
                "rl_areas": rl_areas,
                "rl_oos_irs": rl_irs,
                "baseline_areas": baseline_areas,
                "baseline_oos_irs": baseline_irs,
            },
            "verdict": verdict,
        }
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        print(
            "verdict: RL {} the default searcher; default -> {}".format(
                "KEEPS" if verdict["admitted"] else "LOSES",
                verdict["default_searcher"],
            )
        )
        return 0
    finally:
        export_log_txt(run_name="admission_experiment")


def _oos_ir(tokens, trainer, loader, fold, backtest_config) -> float:
    if tokens is None:
        return float("nan")
    direction = int(getattr(trainer, "best_direction", 1))
    metrics = evaluate_formula(
        tokens, loader, fold, backtest_config, direction=direction
    )
    if metrics is None:
        return float("nan")
    return oos_active_ir(metrics["daily_returns"], metrics["benchmark_daily_returns"])


def _row_oos_ir(row: dict, loader, fold, backtest_config) -> float:
    if row.get("failed"):
        return float("nan")
    tokens = row.get("formula")
    if not tokens:
        return float("nan")
    metrics = evaluate_formula(
        tokens, loader, fold, backtest_config, direction=int(row.get("direction", 1))
    )
    if metrics is None:
        return float("nan")
    return oos_active_ir(metrics["daily_returns"], metrics["benchmark_daily_returns"])


if __name__ == "__main__":
    raise SystemExit(main())
