"""P4 paired admission: random RL vs imitation RL vs GP/TPE/Random.

Every one of the five pre-registered pairs uses one independent seed for all
arms, one requested unique-semantic-evaluation budget and one data/scoring
window. Baseline elites are merged before the imitation arm. A failed arm is
retained as evidence and never replaced by another seed's result.

Usage:
    python scripts/admission_experiment.py [--config config/ashare_config.yaml]
        [--fold 0] [--output experiments/p4_admission_experiment.json]
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

# ``python scripts/admission_experiment.py`` sets sys.path[0] to scripts/.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from loguru import logger

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
    ADMISSION_RULE_VERSION,
    ADMISSION_STEPS,
    ADMISSION_WINDOW,
    PAIR_SEEDS,
    apply_p4_tier_gate,
    best_so_far_area,
    decide_p4_admission,
    paired_seed_plan,
)
from ashare_model.alphagpt import MODEL_VERSION
from ashare_model.baseline_harness import SemanticBudgetEvaluator, oos_active_ir
from ashare_model.data_loader import AshareDataLoader
from ashare_model.elite_archive import merge_elite_archives
from ashare_model.evaluation import PROTOCOL_VERSION, evaluate_formula, resolve_folds
from ashare_model.reward import REWARD_VERSION
from ashare_model.search_backends import get_search_backend
from ashare_model.search_contract import (
    SEARCH_CONTRACT_VERSION,
    SearchRequest,
    SearchResult,
)
from ashare_model.train import AshareTrainer, resolve_device
from ashare_model.vocab import GRAMMAR_VERSION
from ashare_portfolio.execution_spec import execution_provenance


def _build_evaluator(
    *,
    trainer: AshareTrainer,
    loader: AshareDataLoader,
    backtest_config,
    reward_config,
    fold,
    pair_seed: int,
    budget: int,
    backend: str,
    window_cap: tuple[int, int],
) -> SemanticBudgetEvaluator:
    """Fresh ledger over the exact paired window and candidate scorer."""

    vm_device = resolve_device("auto")
    window = trainer.prepare_window(fold.train_end, vm_device, window_cap)
    stocks = window.factor_tensor.shape[1]

    def execute(tokens):
        signal = trainer.vm.execute(tokens, window.factor_tensor)
        return None if signal is None else signal.detach().cpu().numpy()

    return SemanticBudgetEvaluator(
        target=window.target_ret,
        realized_ret=window.realized_ret,
        rebalance_mask=window.rebalance_mask,
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
            f"p4-admission:fold:{fold.train_end}:{fold.test_end}:"
            f"pair_seed:{pair_seed}:cap:{stocks}x{window.factor_tensor.shape[2]}:"
            f"backend:{backend}"
        ),
        tie_break_keys=np.asarray(loader.ts_codes[:stocks]),
        adv=np.asarray(loader.dollar_volume())[
            :stocks, : window.factor_tensor.shape[2]
        ],
        blocked_buy=window.blocked_buy,
        blocked_sell=window.blocked_sell,
        source=backend,
        candidate_prefix=backend,
        chunk=window.reward_chunk,
    )


def _run_baseline(
    *,
    name: str,
    request: SearchRequest,
    trainer: AshareTrainer,
    loader: AshareDataLoader,
    backtest_config,
    reward_config,
    fold,
    window_cap: tuple[int, int],
) -> tuple[SearchResult | None, str | None]:
    try:
        evaluator = _build_evaluator(
            trainer=trainer,
            loader=loader,
            backtest_config=backtest_config,
            reward_config=reward_config,
            fold=fold,
            pair_seed=request.seed,
            budget=request.budget,
            backend=name,
            window_cap=window_cap,
        )
        result = get_search_backend(name).search(
            request, evaluator, vocab=trainer.vocab
        )
        return result, None
    except Exception as exc:  # a failed comparison row is still evidence
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("paired admission backend={} failed: {}", name, error)
        return None, error


def _run_rl(
    *,
    initialization: str,
    archive,
    request: SearchRequest,
    data_config,
    model_config,
    backtest_config,
    reward_config,
    loader,
    fold,
    window_cap: tuple[int, int],
) -> tuple[SearchResult | None, str | None]:
    try:
        trainer = AshareTrainer(
            data_config,
            model_config,
            backtest_config,
            loader,
            reward_config,
            init_seed=request.seed,
        )
        result = trainer.search(
            searcher="rl",
            steps=request.steps,
            batch_size=request.batch_size,
            seed=request.seed,
            save_artifacts=False,
            train_end_date=fold.train_end,
            window_cap=window_cap,
            rl_initialization=initialization,
            elite_archive=archive,
        )
        return result, None
    except Exception as exc:  # RL failure must not erase the successful GP row
        error = f"{type(exc).__name__}: {exc}"
        logger.exception(
            "paired admission backend=rl_{} failed: {}", initialization, error
        )
        return None, error


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _result_row(
    result: SearchResult | None,
    error: str | None,
    *,
    requested_budget: int,
    loader,
    fold,
    backtest_config,
) -> dict[str, object]:
    if result is None:
        return {
            "error": error,
            "requested_budget": requested_budget,
            "consumed_budget": 0,
            "area": None,
            "oos_ir": None,
            "search_result": None,
        }
    area = _finite(best_so_far_area(list(result.best_so_far), requested_budget))
    row_error = error
    try:
        oos_ir = _finite(_result_oos_ir(result, loader, fold, backtest_config))
    except Exception as exc:  # retain the search result and mark OOS evidence absent
        row_error = f"{type(exc).__name__}: {exc}"
        logger.exception("paired admission OOS evaluation failed: {}", row_error)
        oos_ir = None
    return {
        "error": row_error,
        "requested_budget": requested_budget,
        "consumed_budget": result.consumed_budget,
        "area": area,
        "oos_ir": oos_ir,
        "search_result": result.to_dict(),
    }


def _result_oos_ir(result, loader, fold, backtest_config) -> float | None:
    selected = result.selected
    if selected is None or selected.tokens is None:
        return None
    metrics = evaluate_formula(
        list(selected.tokens),
        loader,
        fold,
        backtest_config,
        direction=selected.direction,
    )
    if metrics is None:
        return None
    return oos_active_ir(metrics["daily_returns"], metrics["benchmark_daily_returns"])


def _json_safe(value):
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(argv=None) -> int:
    setup_run_logging(run_name="admission_experiment")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--steps", type=int, default=ADMISSION_STEPS)
    parser.add_argument("--batch-size", type=int, default=ADMISSION_BATCH)
    parser.add_argument(
        "--window",
        default=f"{ADMISSION_WINDOW[0]},{ADMISSION_WINDOW[1]}",
        help="paired window cap 'stocks,dates'",
    )
    parser.add_argument(
        "--output",
        default="experiments/p4_admission_experiment.json",
    )
    parser.add_argument("--min-eligible", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        root = _ROOT
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(
            data_config, min_eligible=args.min_eligible
        ).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        protocol_config = make_protocol_config(raw)
        fold_config = protocol_config.folds[int(args.fold)]
        stocks, dates = (int(value) for value in args.window.split(","))
        window_cap = (stocks, dates)

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        fold = resolve_folds([fold_config], loader.dates)[0]
        requested_budget = int(args.steps) * int(args.batch_size)

        imitation_areas: list[float | None] = []
        imitation_irs: list[float | None] = []
        random_rl_areas: list[float | None] = []
        random_rl_irs: list[float | None] = []
        baseline_areas: dict[str, list[float | None]] = {
            "gp": [],
            "tpe": [],
            "random": [],
        }
        baseline_irs: dict[str, list[float | None]] = {
            "gp": [],
            "tpe": [],
            "random": [],
        }
        pair_rows: list[dict[str, object]] = []

        for seed_row in paired_seed_plan(PAIR_SEEDS):
            pair_seed = int(seed_row["pair_seed"])
            request = SearchRequest(
                seed=pair_seed,
                budget=requested_budget,
                max_formula_len=int(model_config.max_formula_len),
                steps=int(args.steps),
                batch_size=int(args.batch_size),
            )
            print(f"=== pair_seed={pair_seed} ===")
            baseline_trainer = AshareTrainer(
                data_config,
                model_config,
                backtest_config,
                loader,
                reward_config,
                init_seed=pair_seed,
            )
            results: dict[str, SearchResult | None] = {}
            rows: dict[str, dict[str, object]] = {}
            for name in ("gp", "tpe", "random"):
                result, error = _run_baseline(
                    name=name,
                    request=request,
                    trainer=baseline_trainer,
                    loader=loader,
                    backtest_config=backtest_config,
                    reward_config=reward_config,
                    fold=fold,
                    window_cap=window_cap,
                )
                results[name] = result
                row = _result_row(
                    result,
                    error,
                    requested_budget=requested_budget,
                    loader=loader,
                    fold=fold,
                    backtest_config=backtest_config,
                )
                rows[name] = row
                baseline_areas[name].append(row["area"])
                baseline_irs[name].append(row["oos_ir"])

            archives = [
                result.elite_archive
                for result in results.values()
                if result is not None and result.elite_archive is not None
            ]
            merged_archive = merge_elite_archives(archives)

            random_result, random_error = _run_rl(
                initialization="random",
                archive=None,
                request=request,
                data_config=data_config,
                model_config=model_config,
                backtest_config=backtest_config,
                reward_config=reward_config,
                loader=loader,
                fold=fold,
                window_cap=window_cap,
            )
            random_row = _result_row(
                random_result,
                random_error,
                requested_budget=requested_budget,
                loader=loader,
                fold=fold,
                backtest_config=backtest_config,
            )
            rows["rl_random"] = random_row
            random_rl_areas.append(random_row["area"])
            random_rl_irs.append(random_row["oos_ir"])

            imitation_result, imitation_error = _run_rl(
                initialization="imitation",
                archive=merged_archive,
                request=request,
                data_config=data_config,
                model_config=model_config,
                backtest_config=backtest_config,
                reward_config=reward_config,
                loader=loader,
                fold=fold,
                window_cap=window_cap,
            )
            imitation_row = _result_row(
                imitation_result,
                imitation_error,
                requested_budget=requested_budget,
                loader=loader,
                fold=fold,
                backtest_config=backtest_config,
            )
            rows["rl_imitation"] = imitation_row
            imitation_areas.append(imitation_row["area"])
            imitation_irs.append(imitation_row["oos_ir"])

            pair_rows.append(
                {
                    "pair_seed": pair_seed,
                    "backend_seeds": seed_row["backend_seeds"],
                    "requested_budget": requested_budget,
                    "archive": merged_archive.to_dict(),
                    "archive_generation_consumed_budget": sum(
                        result.consumed_budget
                        for result in results.values()
                        if result is not None
                    ),
                    "arms": rows,
                }
            )

        verdict = apply_p4_tier_gate(
            decide_p4_admission(
                imitation_areas=imitation_areas,
                imitation_oos_irs=imitation_irs,
                random_rl_areas=random_rl_areas,
                random_rl_oos_irs=random_rl_irs,
                gp_areas=baseline_areas["gp"],
                gp_oos_irs=baseline_irs["gp"],
            ),
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            window=window_cap,
        )
        payload = _json_safe(
            {
                "experiment": "p4-paired-rl-admission",
                "rule_version": ADMISSION_RULE_VERSION,
                "search_contract_version": SEARCH_CONTRACT_VERSION,
                "model_version": MODEL_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "reward_version": REWARD_VERSION,
                "dataset_id": loader.dataset_id,
                "grammar_version": GRAMMAR_VERSION,
                "max_formula_len": int(model_config.max_formula_len),
                "reward_config": asdict(reward_config),
                "execution": execution_provenance(backtest_config),
                "fold": int(args.fold),
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "requested_budget": requested_budget,
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "window_cap": list(window_cap),
                "registered_admission_tier": {
                    "steps": ADMISSION_STEPS,
                    "batch_size": ADMISSION_BATCH,
                    "window_cap": list(ADMISSION_WINDOW),
                },
                "pair_seed_plan": paired_seed_plan(PAIR_SEEDS),
                "pairs": pair_rows,
                "metrics": {
                    "imitation_areas": imitation_areas,
                    "imitation_oos_irs": imitation_irs,
                    "random_rl_areas": random_rl_areas,
                    "random_rl_oos_irs": random_rl_irs,
                    "baseline_areas": baseline_areas,
                    "baseline_oos_irs": baseline_irs,
                },
                "verdict": verdict,
            }
        )
        output = root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
        return 0
    finally:
        export_log_txt(run_name="admission_experiment")


if __name__ == "__main__":
    raise SystemExit(main())
