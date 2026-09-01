"""P10 Stage B/C adjudication for the four-searcher matched comparison.

Contract: docs/p10_searcher_fairness_contract.md (APPROVED 2026-09-01) —
§5 (single evaluation path, hard gates), §5.3 (arm aggregates),
§8 (pre-registered decision mapping).

Single semantic path: every selected formula is scored by
:func:`ashare_model.evaluation.evaluate_formula` (full-history execution →
fold test-window slice → ``evaluate_signal`` → ``AshareBacktestEngine``
with the shared constructor/cost semantics).  The gates read ONLY the
engine metric fields; nothing is recomputed and the direction is the
search-time ``selected.direction`` — it is never re-fit on the test
window.

OOS sign metric note: the contract's ``oos_positive`` row is executed as
``icir > 0`` on the fold-0 test window — the ``icir`` field of the
single-path payload (universe-masked rank-IC IR).  A separate "active
IR" recomputation would be a second IR semantics path; the search-time
``active_ir`` values stay recorded in the campaign artifact as context.

Artifact: ``adjudication.json`` (``P10_ADJUDICATION_VERSION = 1``), a
measurement-report artifact of the P10 campaign — like the searcher_bench
v3 payload it is not wired into the RunSpec registry, which pins semantic
research versions (contract §9).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from loguru import logger

P10_ADJUDICATION_VERSION = 1

# §5.2 hard gates (engine ``AshareBacktestEngine._metrics`` definitions:
# 252-day annualization, sharpe carries the 2% risk-free offset, calmar
# uses max_drawdown + 1e-9).
GATE_THRESHOLDS = {
    "annual_return_gt": 0.10,
    "max_drawdown_lt": 0.15,
    "sharpe_ge": 1.0,
    "calmar_ge": 1.0,
}
# §5.3 arm aggregates require >= 2 of the 3 paired seeds.
ARM_MAJORITY = 2
# §8 "全臂 consumed≪requested 且停滞" — a stagnant arm: every seed ended in
# proposal_stagnation below half the requested budget.
STAGNANT_CONSUMED_FRACTION = 0.5


def best_so_far_area(best_so_far, requested_budget: int) -> float:
    """Area under the best-so-far curve over the REQUESTED budget.

    P4 §5 convention: the curve's reward is held between budget
    coordinates and the last best is held to the requested end when the
    search stopped early (stagnation/exhausted steps).  The first
    coordinate's reward covers ``[0, x_1)`` so every consumed budget
    counts.  Returns 0.0 for an empty curve (zero consumption).
    """

    budget = float(requested_budget)
    if not best_so_far:
        return 0.0
    area = 0.0
    previous_x = 0.0
    previous_r = float(best_so_far[0][1])
    for x, reward in best_so_far:
        x = float(x)
        if x > budget:  # defensive: curves must not extend past B
            break
        area += previous_r * (x - previous_x)
        previous_x = x
        previous_r = float(reward)
    area += previous_r * (budget - previous_x)
    return area


def gates_passed(metrics: dict) -> dict[str, bool]:
    """Evaluate §5.2 against the engine metric fields, verbatim.

    Reads exactly ``annual_return``, ``max_drawdown``, ``sharpe`` and
    ``calmar``; a missing field raises (fail closed) instead of scoring.
    """

    annual = metrics["annual_return"]
    drawdown = metrics["max_drawdown"]
    sharpe = metrics["sharpe"]
    calmar = metrics["calmar"]
    annual_ok = annual > GATE_THRESHOLDS["annual_return_gt"]
    drawdown_ok = drawdown < GATE_THRESHOLDS["max_drawdown_lt"]
    sharpe_ok = sharpe >= GATE_THRESHOLDS["sharpe_ge"]
    calmar_ok = calmar >= GATE_THRESHOLDS["calmar_ge"]
    return {
        "annual_return_gt_0.10": annual_ok,
        "max_drawdown_lt_0.15": drawdown_ok,
        "sharpe_ge_1.0": sharpe_ok,
        "calmar_ge_1.0": calmar_ok,
        "all": annual_ok and drawdown_ok and sharpe_ok and calmar_ok,
    }


def evaluate_selected_rows(
    stage_a_rows,
    *,
    loader,
    fold,
    bt_cfg,
    evaluate_fn=None,
):
    """Stage B: score every completed Stage-A row's selected formula on
    the fold test window via the single evaluation path.

    ``evaluate_fn`` defaults to :func:`ashare_model.evaluation.evaluate_formula`
    and must accept ``(tokens, loader, fold, bt_cfg, vm=..., direction=...)``.
    The direction passed is the search-time ``selected.direction`` —
    never re-fit.  Rows without a selection and formula failures are
    recorded failure rows (``failed`` True), never dropped.
    """

    if evaluate_fn is None:
        from .eval_metrics import evaluate_formula  # noqa: PLC0415

        evaluate_fn = evaluate_formula
    out: list[dict] = []
    for row in stage_a_rows:
        seed = row.get("seed")
        searcher = row.get("searcher")
        record: dict = {
            "seed": seed,
            "searcher": searcher,
            "requested_budget": row.get("requested_budget"),
            "consumed_budget": row.get("consumed_budget"),
            "termination_reason": row.get("termination_reason"),
            "stagnation_reason": row.get("stagnation_reason"),
            "failed": True,
            "error": None,
            "direction_used": None,
            "metrics": None,
            "gates": None,
        }
        selected = (row.get("search_result") or {}).get("selected") if isinstance(
            row.get("search_result"), dict
        ) else None
        if not row.get("completed") or selected is None:
            record["error"] = (
                "stage-a row not completed or no selected formula"
            )
            out.append(record)
            continue
        direction = int(selected["direction"])
        record["direction_used"] = direction
        record["selected"] = {
            "formula_text": selected.get("formula_text"),
            "tokens": list(selected.get("tokens") or []),
            "direction": direction,
            "val_reward": selected.get("val_reward"),
            "val_icir": selected.get("val_icir"),
        }
        try:
            metrics = evaluate_fn(
                list(selected["tokens"]),
                loader,
                fold,
                bt_cfg,
                direction=direction,
            )
        except Exception as exc:  # a failed evaluation is a recorded row
            record["error"] = f"{type(exc).__name__}: {exc}"
            out.append(record)
            continue
        if metrics is None:
            record["error"] = "evaluate_formula returned None (invalid formula)"
            out.append(record)
            continue
        record["metrics"] = metrics
        record["gates"] = gates_passed(metrics)
        record["failed"] = False
        record["error"] = None
        out.append(record)
    return out


def arm_aggregates(stage_a_rows, stage_b_rows, areas) -> dict[str, dict]:
    """§5.3 arm aggregates over the three paired seeds.

    ``areas`` maps ``(seed, searcher)`` to the row's best-so-far area
    (``best_so_far_area`` over the requested budget).  ``admissible`` /
    ``oos_positive`` need the majority of the arm's seeds;
    ``stagnant`` means every seed ended in ``proposal_stagnation`` below
    half the requested budget (contract §8 row 4).
    """

    arms: dict[str, dict] = {}
    for row in stage_a_rows:
        arm = row["searcher"]
        entry = arms.setdefault(arm, {"seeds": 0, "areas": [], "gates": [], "icir": [], "stagnant_seeds": 0})
        entry["seeds"] += 1
        key = (row["seed"], arm)
        if key in areas:
            entry["areas"].append(float(areas[key]))
        if row.get("termination_reason") == "proposal_stagnation" and float(
            row.get("consumed_budget") or 0.0
        ) < STAGNANT_CONSUMED_FRACTION * float(row.get("requested_budget") or 0.0):
            entry["stagnant_seeds"] += 1
    for record in stage_b_rows:
        arm = record["searcher"]
        entry = arms.setdefault(arm, {"seeds": 0, "areas": [], "gates": [], "icir": [], "stagnant_seeds": 0})
        if record.get("failed") or record.get("gates") is None:
            continue
        entry["gates"].append(bool(record["gates"]["all"]))
        metrics = record.get("metrics") or {}
        if "icir" in metrics:
            entry["icir"].append(float(metrics["icir"]))
    for arm, entry in arms.items():
        n = max(entry["seeds"], 1)
        entry["area_median"] = (
            statistics.median(entry["areas"]) if entry["areas"] else None
        )
        entry["admissible"] = (
            sum(entry["gates"]) >= ARM_MAJORITY and len(entry["gates"]) >= ARM_MAJORITY
        )
        entry["oos_positive"] = (
            sum(1 for value in entry["icir"] if value > 0) >= ARM_MAJORITY
            and len(entry["icir"]) >= ARM_MAJORITY
        )
        entry["stagnant"] = entry["stagnant_seeds"] == n
    return arms


def adjudicate(stage_a_rows, stage_b_rows, areas) -> dict:
    """§8 pre-registered decision mapping, executed verbatim.

    ``headline`` is the first matching row of the mapping table
    (investment / scaling / negative); ``additional_findings`` collects
    the diagnosis rows (all-arms stagnation, efficient-but-not-OOS-
    positive).  With n=3 paired seeds the verdicts are directional
    statements only — no statistical-significance claim (§5.3).
    """

    arms = arm_aggregates(stage_a_rows, stage_b_rows, areas)
    admissible = sorted(arm for arm, e in arms.items() if e["admissible"])
    efficient = [
        arm
        for arm, e in arms.items()
        if e["area_median"] is not None
        and e["area_median"] == max(
            (x["area_median"] for x in arms.values() if x["area_median"] is not None),
            default=None,
        )
    ]
    all_stagnant = bool(arms) and all(e["stagnant"] for e in arms.values())
    matched: list[str] = []
    additional: list[str] = []
    headline = "negative_no_admissible_formula"
    arm = None
    if len(admissible) == 1 and efficient == admissible:
        matched.append("exactly_one_admissible_and_efficient")
        headline = "invest_in_searcher"
        arm = admissible[0]
    elif len(admissible) >= 2:
        matched.append("multiple_admissible_budget_scaling_study")
        headline = "budget_scaling_study"
        arm = efficient[0] if efficient else None
    else:
        matched.append("no_arm_admissible_negative_result")
        headline = "negative_no_admissible_formula"
    if all_stagnant:
        additional.append("all_arms_stagnant_proposal_diversity")
    if len(efficient) == 1 and not arms[efficient[0]]["oos_positive"]:
        additional.append("efficient_arm_not_oos_positive_reward_audit")
    return {
        "version": P10_ADJUDICATION_VERSION,
        "headline": headline,
        "arm": arm,
        "matched_rows": matched,
        "additional_findings": additional,
        "arms": arms,
        "admissible_arms": admissible,
        "efficient_arms": efficient,
        "significance_note": (
            "n=3 paired seeds: directional statements only, no "
            "statistical-significance claim (contract §5.3)"
        ),
    }


def main(argv=None) -> int:
    """Run Stage B/C for one campaign directory and write
    ``adjudication.json`` next to the campaign artifact."""

    from ashare_data.config import (
        load_config,
        make_backtest_config,
        make_data_config,
        make_model_config,
        make_protocol_config,
    )
    from ashare_data.gates import ProductionGateRunner
    from .data_loader import AshareDataLoader
    from .eval_folds import resolve_folds
    from .eval_metrics import benchmark_row, evaluate_formula

    parser = argparse.ArgumentParser(
        description="P10 Stage B/C: OOS evaluation, hard backtest gates "
        "and the pre-registered decision mapping for one campaign"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    campaign_path = run_dir / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("campaign_status") != "completed":
        raise RuntimeError(
            f"campaign_status={campaign.get('campaign_status')!r}; a "
            "matched adjudication requires a completed campaign "
            "(fail-closed, contract §6)"
        )

    started = time.perf_counter()
    raw = load_config(args.config, project_root=root)
    data_config = make_data_config(raw, root)
    ProductionGateRunner(data_config).require_production()
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    proto_cfg = make_protocol_config(raw)
    fold_cfg = proto_cfg.folds[int(args.fold)]
    loader = AshareDataLoader(data_config, model_config)
    loader.load_data()
    folds = resolve_folds(
        proto_cfg.folds,
        loader.dates,
        frequency=backtest_config.rebalance_frequency,
        horizon=backtest_config.target_horizon,
    )
    fold = folds[int(args.fold)]

    rows = campaign["rows"]
    stage_b = evaluate_selected_rows(
        rows,
        loader=loader,
        fold=fold,
        bt_cfg=backtest_config,
        evaluate_fn=evaluate_formula,
    )
    bench = benchmark_row(loader, fold)
    areas = {
        (row["seed"], row["searcher"]): best_so_far_area(
            row.get("best_so_far") or [], row["requested_budget"]
        )
        for row in rows
    }
    verdict = adjudicate(rows, stage_b, areas)
    payload = {
        "version": P10_ADJUDICATION_VERSION,
        "run_id": campaign["run_id"],
        "fold_train_end": fold_cfg.train_end,
        "fold_test_end": fold_cfg.test_end,
        "dataset_id": loader.dataset_id,
        "git_commit": campaign.get("campaign", {}).get("git_commit"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds_total": time.perf_counter() - started,
        "benchmark_row": bench,
        "stage_b_rows": stage_b,
        "areas": {f"{seed}:{arm}": value for (seed, arm), value in areas.items()},
        "verdict": verdict,
    }
    out_path = run_dir / "adjudication.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, out_path)
    logger.success("adjudication written to {}", out_path)
    print(f"headline={verdict['headline']} arm={verdict['arm']}")
    print(
        f"{'seed':>6} {'searcher':<8} {'ann_ret':>9} {'max_dd':>8} "
        f"{'sharpe':>8} {'calmar':>8} {'turnover':>9} {'icir':>8} {'gates':>6}"
    )
    for record in stage_b:
        m = record.get("metrics") or {}
        gates = record.get("gates") or {}
        print(
            f"{record['seed']:>6} {record['searcher']:<8} "
            f"{m.get('annual_return', float('nan')):>9.4f} "
            f"{m.get('max_drawdown', float('nan')):>8.4f} "
            f"{m.get('sharpe', float('nan')):>8.3f} "
            f"{m.get('calmar', float('nan')):>8.3f} "
            f"{m.get('average_turnover', float('nan')):>9.4f} "
            f"{m.get('icir', float('nan')):>8.3f} "
            f"{str(gates.get('all')):>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
