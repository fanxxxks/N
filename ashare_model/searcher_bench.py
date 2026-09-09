"""Searcher cost benchmark: time & peak RSS per unique semantic
evaluation (P1-04) and the small-budget smoke test (P1-05), extended in
P10 with the pre-registered multi-seed matched campaign runner.

The four searchers (gp / tpe / random / rl) each run to completion on
the SAME capped window (``prepare_window`` window_cap — by default the
300x400 head of fold 0's training window), the SAME nominal budget and
the SAME seed.  The budget unit is the unique semantic formula
evaluation (T2-01 ledger; duplicates, degenerate and invalid formulas
never bill).  Non-RL searchers run through ``train_search`` with
(steps=budget, batch_size=1); RL runs ``train`` with a fixed step/batch
split (steps x batch = budget), so all four share the same nominal
budget.

Per searcher the report records the actual ``unique_semantic_evals``,
wall-clock seconds, wall seconds **per 1000 evaluations**, peak RSS in
MB (stdlib polling), the completion flag, the selected validation
reward, the termination/stagnation reason and the evaluated-formula
length histogram (content tokens + EOS).  The wall time includes the
identical ``prepare_window`` cost every searcher pays, so per-searcher
comparisons are fair by construction.

P10 campaign mode (``benchmark_campaign``; SEARCHER_BENCH_VERSION 3):
seeds x searchers rows with a per-row fresh trainer/cache, an
append-only ``ExperimentLedger`` trial per row attempt, atomic per-row
campaign JSON persistence with resume support, and the fail-closed
circuit breakers of docs/p10_searcher_fairness_contract.md §4.5/§6
(wall cap checked between rows, post-seed-block calibration-deviation
stop, identity-drift refusal).  Campaign identity (budget/seeds/window/
fold/rl-steps/dataset_id) must match on resume; failed rows are retried
as new ledger trials while every prior entry stays in the chain.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from loguru import logger

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_reward_config,
    make_sim_config,
)
from ashare_data.gates import ProductionGateRunner
from ashare_execution import validate_execution_config
from ashare_logging import export_log_txt, setup_run_logging

from .admission import ADMISSION_WINDOW
from .data_loader import AshareDataLoader
from .data_tier import (
    DATA_TIER_VERSION,
    feature_tier,
    formula_data_tier_report,
)
from .ir import canonical_tokens
from .ledger import ExperimentLedger
from .runspec import new_run_id
from .search_length_prior import LENGTH_PRIOR_PROFILE as P14_LENGTH_PRIOR_PROFILE
from .semantic_cache import SEMANTIC_CACHE_VERSION
from .train import AshareTrainer, resolve_device
from .vocab import FORMULA_VOCAB


# --- Mechanical split (t79) ------------------------------------------------
# The campaign layer (constants, row bench, campaign runner) moved to
# ``searcher_campaign.py``; this module keeps the CLI entry and re-exports
# the moved names so every existing importer keeps working unchanged
# (import-surface compatibility, guarded by tests/test_searcher_bench.py).
from .searcher_campaign import (  # noqa: F401
    DEFAULT_WINDOW_CAP,
    MIN_BUDGET,
    P10_CALIBRATION_S_PER_EVAL,
    P10_COMPARE_BUDGET,
    P10_COMPARE_FOLD,
    P10_COMPARE_SEEDS,
    P10_COMPARE_WINDOW,
    P10_DEVIATION_STOP_RATIO,
    P10_RL_STEPS,
    P10_ROW_ORDER,
    P10_STAGE_A_WALL_CAP_S,
    P14_LENGTH_PRIOR_PROFILE,
    P14_PROMOTION_BUDGET,
    P14_RESEARCH_BUDGET,
    P14_TRACKS,
    RL_STEPS,
    SEARCHER_BENCH_VERSION,
    SEARCHERS,
    _atomic_write_json,
    _campaign_identity,
    _formula_len_histogram,
    _git_commit,
    _run_row,
    _validate_promotion_tier_purity,
    benchmark_campaign,
    benchmark_searchers,
    current_rss_mb,
    measure_peak_rss,
    rl_split,
    tier_a_feature_ids,
)


def _parse_window_cap(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"window-cap must be '<stocks>x<dates>', got {text!r}")
    stocks, dates = int(parts[0]), int(parts[1])
    if stocks <= 0 or dates <= 0:
        raise ValueError("window-cap must be positive (stocks, dates)")
    return stocks, dates


def _config_hash(config_arg: str | None, root: Path) -> str | None:
    """SHA-256 over the effective config inputs (YAML file bytes plus the
    runtime-overrides file when present).  ``config_arg=None`` resolves
    to :func:`ashare_data.config.load_config`'s default path.  ``None``
    is returned only when the config file cannot be read — the campaign
    still runs, provenance records the gap."""

    from ashare_data.config import RUNTIME_OVERRIDES_FILENAME

    try:
        path = Path(config_arg) if config_arg is not None else None
        if path is None:
            path = root / "config" / "ashare_config.yaml"
        elif not path.is_absolute():
            path = root / path
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        overrides = path.parent / RUNTIME_OVERRIDES_FILENAME
        if overrides.exists():
            digest.update(b"\x00--runtime-overrides--\x00")
            digest.update(overrides.read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def main(argv=None) -> int:
    setup_run_logging(run_name="searcher_bench")
    parser = argparse.ArgumentParser(
        description="Searcher cost benchmark: time & peak RSS per 1000 "
        "unique semantic evaluations (P1-04) and the small-budget smoke "
        "test (P1-05); P10 campaign mode with --run-dir"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data/searcher_bench.json")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="requested unique semantic evaluations per row "
        "(default: 1000 legacy bench / P10_COMPARE_BUDGET campaign)",
    )
    parser.add_argument(
        "--research-budget",
        type=int,
        default=None,
        help="campaign mode: research-track budget (default: "
        "P14_RESEARCH_BUDGET; must sum with --promotion-budget to "
        "--budget — p14 §7.3)",
    )
    parser.add_argument(
        "--promotion-budget",
        type=int,
        default=None,
        help="campaign mode: promotion_tier_a-track budget (default: "
        "P14_PROMOTION_BUDGET; must sum with --research-budget to "
        "--budget — p14 §7.3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="enable P10 campaign mode: seeds x searchers rows with "
        "ledger, resume and circuit breakers under this directory",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="campaign mode: comma-separated paired seeds "
        "(default: P10_COMPARE_SEEDS)",
    )
    parser.add_argument(
        "--rl-steps",
        type=int,
        default=None,
        help="campaign mode: RL optimizer steps (default: P10_RL_STEPS=8)",
    )
    parser.add_argument(
        "--wall-cap-hours",
        type=float,
        default=None,
        help="campaign mode: hard Stage-A wall cap in hours "
        "(default: P10_STAGE_A_WALL_CAP_S)",
    )
    parser.add_argument(
        "--fold", type=int, default=0, help="protocol fold index (default 0)"
    )
    parser.add_argument(
        "--window-cap",
        default="300x400",
        help="capped window '<stocks>x<dates>' (default 300x400)",
    )
    parser.add_argument(
        "--searchers",
        default=",".join(SEARCHERS),
        help="comma-separated searchers (default gp,tpe,random,rl)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
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
        sim_config = make_sim_config(raw, root)
        validate_execution_config(backtest_config, sim_config)
        proto_cfg = make_protocol_config(raw)
        fold_index = int(args.fold)
        if not 0 <= fold_index < len(proto_cfg.folds):
            raise ValueError(
                f"fold index out of range: {fold_index} "
                f"(config has {len(proto_cfg.folds)} folds)"
            )
        train_end_date = proto_cfg.folds[fold_index].train_end
        window_cap = _parse_window_cap(args.window_cap)
        searchers = tuple(s.strip() for s in args.searchers.split(",") if s.strip())

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()

        if args.run_dir is not None:
            seeds = (
                tuple(int(s) for s in args.seeds.split(",") if s.strip())
                if args.seeds
                else P10_COMPARE_SEEDS
            )
            payload = benchmark_campaign(
                data_config,
                model_config,
                backtest_config,
                reward_config,
                loader,
                seeds=seeds,
                budget=args.budget or P10_COMPARE_BUDGET,
                train_end_date=train_end_date,
                window_cap=window_cap,
                run_dir=root / args.run_dir,
                research_budget=args.research_budget,
                promotion_budget=args.promotion_budget,
                fold_index=fold_index,
                rl_steps=args.rl_steps or P10_RL_STEPS,
                device=None,
                wall_cap_s=(
                    args.wall_cap_hours * 3600.0
                    if args.wall_cap_hours is not None
                    else P10_STAGE_A_WALL_CAP_S
                ),
                config_hash=_config_hash(args.config, root),
                searchers=searchers,
            )
            print(
                f"campaign run_id={payload['run_id']} "
                f"status={payload['campaign_status']} "
                f"rows={len(payload['rows'])} "
                f"not_run={len(payload['not_run'])}"
            )
            print(
                f"{'seed':>6} {'searcher':<8} {'evals':>6} {'req':>6} "
                f"{'wall_s':>9} {'per_1000':>9} {'rss_mb':>9} {'done':>6}"
            )
            for row in payload["rows"]:
                per_1000 = (
                    f"{row['wall_per_1000_evals']:.1f}"
                    if row["wall_per_1000_evals"] is not None
                    else "-"
                )
                rss = f"{row['peak_rss_mb']:.0f}" if row["peak_rss_mb"] else "-"
                print(
                    f"{row['seed']:>6} {row['searcher']:<8} "
                    f"{row['unique_semantic_evals']:>6} "
                    f"{row['requested_budget']:>6} "
                    f"{row['wall_seconds']:>9.1f} {per_1000:>9} {rss:>9} "
                    f"{str(row['completed']):>6}"
                )
            for row in payload["not_run"]:
                print(
                    f"{row['seed']:>6} {row['searcher']:<8} "
                    f"{'-':>6} {'-':>6} {'-':>9} {'-':>9} {'-':>9} "
                    f"not_run({row['reason']})"
                )
            return 0

        payload = benchmark_searchers(
            data_config,
            model_config,
            backtest_config,
            reward_config,
            loader,
            searchers=searchers,
            budget=args.budget if args.budget is not None else 1000,
            seed=args.seed,
            train_end_date=train_end_date,
            window_cap=window_cap,
        )
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Searcher bench written to {out_path}")
        print(
            f"{'searcher':<8} {'evals':>6} {'wall_s':>9} "
            f"{'per_1000':>9} {'rss_mb':>9} {'done':>6}"
        )
        for searcher in searchers:
            row = payload["rows"][searcher]
            per_1000 = (
                f"{row['wall_per_1000_evals']:.1f}"
                if row["wall_per_1000_evals"] is not None
                else "-"
            )
            rss = f"{row['peak_rss_mb']:.0f}" if row["peak_rss_mb"] else "-"
            print(
                f"{searcher:<8} {row['unique_semantic_evals']:>6} "
                f"{row['wall_seconds']:>9.1f} {per_1000:>9} {rss:>9} "
                f"{str(row['completed']):>6}"
            )
    finally:
        export_log_txt(run_name="searcher_bench")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
