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
    ADMISSION_WINDOW,
    INIT_SEEDS,
    best_so_far_area,
    decide_admission,
)
from ashare_model.baseline_harness import (
    SemanticBudgetEvaluator,
    canonical_form_pool,
    oos_active_ir,
)
from ashare_model.data_loader import AshareDataLoader
from ashare_model.evaluation import (
    PROTOCOL_VERSION,
    evaluate_formula,
    resolve_folds,
)
from ashare_model.gp_search import run_gp_baseline
from ashare_model.train import AshareTrainer, resolve_device
from ashare_model.tpe_search import run_tpe_baseline


def _build_evaluator(
    *,
    trainer: AshareTrainer,
    loader: AshareDataLoader,
    backtest_config,
    reward_config,
    fold,
    seed: int,
    budget: int,
    source: str,
    candidate_prefix: str,
    window_cap: tuple[int, int],
) -> tuple[SemanticBudgetEvaluator, object]:
    """One search evaluator over the capped admission window: a fresh
    semantic-budget ledger per searcher, sharing the trainer's VM and
    calibration executor bindings."""

    import numpy as np

    vm_device = resolve_device("auto")
    window = trainer.prepare_window(fold.train_end, vm_device, window_cap)
    stocks = int(window_cap[0])

    def execute(tokens):
        signal = trainer.vm.execute(tokens, window.factor_tensor)
        if signal is None:
            return None
        return signal.detach().cpu().numpy()

    evaluator = SemanticBudgetEvaluator(
        target=window.target_ret,
        universe_mask=window.train_universe_mask,
        backtest_config=backtest_config,
        reward_config=reward_config,
        val_windows=window.val_windows,
        train_signal_range=window.train_signal_range,
        budget=budget,
        execute=execute,
        fingerprint_execute=trainer._calibration_execute,
        dataset_id=loader.dataset_id,
        protocol_version=PROTOCOL_VERSION,
        window_id=(
            f"admission:fold:{fold.train_end}:{fold.test_end}:"
            f"seed:{seed}:cap:{stocks}x{window.factor_tensor.shape[2]}:{source}"
        ),
        tie_break_keys=np.asarray(loader.ts_codes[:stocks]),
        adv=np.asarray(loader.dollar_volume())[:stocks, : window.factor_tensor.shape[2]],
        blocked_buy=window.blocked_buy,
        blocked_sell=window.blocked_sell,
        source=source,
        candidate_prefix=candidate_prefix,
        chunk=window.reward_chunk,
    )
    return evaluator, window


def main(argv=None) -> int:
    setup_run_logging(run_name="admission_experiment")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--steps", type=int, default=ADMISSION_STEPS, help="RL training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=ADMISSION_BATCH, help="RL batch size"
    )
    parser.add_argument(
        "--window",
        default=f"{ADMISSION_WINDOW[0]},{ADMISSION_WINDOW[1]}",
        help="window cap 'stocks,dates' (head slice of the fold window)",
    )
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
        stocks_s, dates_s = args.window.split(",")
        window_cap = (int(stocks_s), int(dates_s))

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
                steps=args.steps,
                batch_size=args.batch_size,
                seed=42,
                save_artifacts=False,
                train_end_date=fold.train_end,
                window_cap=window_cap,
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
                ("random", lambda ev: _run_random(ev, model_config, seed=int(proto_cfg.random_seed))),
                ("gp", lambda ev: run_gp_baseline(
                    seed=int(proto_cfg.gp_seed),
                    evaluator=ev,
                    max_formula_len=model_config.max_formula_len,
                )),
                ("tpe", lambda ev: run_tpe_baseline(
                    seed=int(proto_cfg.tpe_seed),
                    evaluator=ev,
                    max_formula_len=model_config.max_formula_len,
                )),
            ):
                evaluator, _ = _build_evaluator(
                    trainer=trainer,
                    loader=loader,
                    backtest_config=backtest_config,
                    reward_config=reward_config,
                    fold=fold,
                    seed=init_seed,
                    budget=rl_budget,
                    source=name,
                    candidate_prefix=name,
                    window_cap=window_cap,
                )
                result = runner(evaluator)
                curve = list(result.best_so_far)
                area = best_so_far_area(curve, rl_budget)
                baseline_areas[name].append(area)
                ir = _result_oos_ir(result, loader, fold, backtest_config)
                baseline_irs[name].append(ir)
                seed_rows[name] = {
                    "budget_used": result.n_evaluated,
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
            "admission_tier": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "window_cap": list(window_cap),
            },
            "init_seeds": list(INIT_SEEDS),
            "sampling_seed": 42,
            "dataset_id": loader.dataset_id,
            "protocol_version": PROTOCOL_VERSION,
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


def _run_random(evaluator, model_config, seed: int):
    from ashare_model.vocab import FORMULA_VOCAB

    for key in canonical_form_pool(
        seed,
        FORMULA_VOCAB,
        model_config.max_formula_len,
        evaluator.budget,
    ):
        evaluator.propose(key)
        if evaluator.budget_used >= evaluator.budget:
            break
    return evaluator.finish()


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


def _result_oos_ir(result, loader, fold, backtest_config) -> float:
    selected = result.selected
    if selected is None or selected.tokens is None:
        return float("nan")
    metrics = evaluate_formula(
        list(selected.tokens),
        loader,
        fold,
        backtest_config,
        direction=selected.direction,
    )
    if metrics is None:
        return float("nan")
    return oos_active_ir(metrics["daily_returns"], metrics["benchmark_daily_returns"])


if __name__ == "__main__":
    raise SystemExit(main())
