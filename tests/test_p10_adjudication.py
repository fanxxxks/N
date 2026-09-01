"""P10 Stage B/C adjudication tests (contract §10-6 + §5/§8).

Contract source: docs/p10_searcher_fairness_contract.md —

* §5.2: the four hard backtest gates read ONLY the engine metric fields
  produced by the single evaluation path (``evaluation.evaluate_formula``
  → ``evaluate_signal`` → ``AshareBacktestEngine._metrics``); boundaries
  are strict for annual_return (>) and max_drawdown (<), inclusive for
  sharpe (>=) and calmar (>=).  A missing metric field fails closed.
* §2 不变量 4 / §10-6: the evaluation direction is the search-time
  ``selected.direction`` — the adjudicator never re-fits direction on
  the test window.
* §5.3: best-so-far area integrates over the REQUESTED budget with the
  last best held to the end (P4 §5 convention); arm aggregates are
  paired medians over the three seeds.
* §8: the pre-registered decision mapping (five observation rows) is
  executed verbatim; negative results are legal outputs.
"""
from __future__ import annotations

import pytest

from ashare_model.searcher_adjudication import (
    P10_ADJUDICATION_VERSION,
    adjudicate,
    arm_aggregates,
    best_so_far_area,
    gates_passed,
)


# ---------------------------------------------------------------------------
# §5.2: hard gates read only the engine metrics
# ---------------------------------------------------------------------------


def test_p10_adjudication_version_pin():
    assert P10_ADJUDICATION_VERSION == 1


def test_p10_gates_boundaries_and_strictness():
    all_pass = {
        "annual_return": 0.101,
        "max_drawdown": 0.149,
        "sharpe": 1.0,
        "calmar": 1.0,
    }
    verdict = gates_passed(all_pass)
    assert verdict == {
        "annual_return_gt_0.10": True,
        "max_drawdown_lt_0.15": True,
        "sharpe_ge_1.0": True,
        "calmar_ge_1.0": True,
        "all": True,
    }
    # Exact thresholds: annual_return is strict >, max_drawdown strict <,
    # sharpe/calmar inclusive >= (engine _metrics definitions).
    boundary = gates_passed(
        {
            "annual_return": 0.10,
            "max_drawdown": 0.15,
            "sharpe": 1.0,
            "calmar": 1.0,
        }
    )
    assert boundary["annual_return_gt_0.10"] is False
    assert boundary["max_drawdown_lt_0.15"] is False
    assert boundary["sharpe_ge_1.0"] is True
    assert boundary["calmar_ge_1.0"] is True
    assert boundary["all"] is False
    # Fail closed: a missing engine metric raises instead of scoring.
    with pytest.raises(KeyError):
        gates_passed({"annual_return": 0.2, "max_drawdown": 0.1, "sharpe": 2.0})


# ---------------------------------------------------------------------------
# §5.3: best-so-far area over the requested budget
# ---------------------------------------------------------------------------


def test_p10_best_so_far_area_holds_last_best_to_budget():
    # Step-function convention (P4 §5): the reward achieved at consumed
    # x_i stands on [x_i, x_{i+1}); the head [0, x_1) holds r_1 back; the
    # last best is held to the requested end when the search stopped
    # early.  Curve [(1, 0.2), (3, 0.5)] over B=10: r1 covers [0,3),
    # r2 covers [3,10].
    area = best_so_far_area([(1, 0.2), (3, 0.5)], 10)
    assert area == pytest.approx(0.2 * 3.0 + 0.5 * 7.0, rel=1e-12)
    # A budget-exhausted curve ends exactly at B: r1 stands on [0,2),
    # r2 becomes effective only at consumed 2 and carries no interval.
    assert best_so_far_area([(1, 0.5), (2, 1.0)], 2) == pytest.approx(
        0.5 * 2.0, rel=1e-12
    )
    # Zero consumption (no curve) contributes nothing.
    assert best_so_far_area([], 2000) == 0.0


# ---------------------------------------------------------------------------
# §10-6: single evaluation path, selected direction, verbatim engine metrics
# ---------------------------------------------------------------------------


def _stage_row(seed, searcher, direction, consumed=2000, stagnation=None):
    # The fixture mirrors the real campaign.json row shape: the selected
    # candidate lives at ``search_result.selected``.
    selected = {
        "tokens": [55, 41, 110, 77],
        "formula_text": "(TURNOVER_STD20 DIV DOWNVOL60(IVOL_60))",
        "direction": direction,
        "val_reward": 0.9,
        "val_icir": 0.3,
    }
    return {
        "seed": seed,
        "searcher": searcher,
        "completed": True,
        "requested_budget": 2000,
        "consumed_budget": consumed,
        "termination_reason": (
            "proposal_stagnation" if stagnation else "budget_exhausted"
        ),
        "stagnation_reason": stagnation,
        "best_so_far": [(1, 0.8)],
        "search_result": {"selected": selected},
    }


def test_p10_stage_b_uses_selected_direction_and_verbatim_metrics(monkeypatch):
    from ashare_model import searcher_adjudication as adj

    captured = {}

    def fake_evaluate(tokens, loader, fold, bt_cfg, vm=None, direction=1):
        captured["tokens"] = list(tokens)
        captured["direction"] = direction
        return {
            "annual_return": 0.2,
            "max_drawdown": 0.1,
            "sharpe": 1.5,
            "calmar": 2.0,
            "average_turnover": 0.05,
            "icir": 0.4,
        }

    rows = [_stage_row(42, "gp", direction=-1)]
    stage_b = adj.evaluate_selected_rows(
        rows,
        loader=object(),
        fold=object(),
        bt_cfg=object(),
        evaluate_fn=fake_evaluate,
    )
    assert captured["direction"] == -1  # the search-time direction, verbatim
    assert captured["tokens"] == [55, 41, 110, 77]
    row = stage_b[0]
    assert row["seed"] == 42 and row["searcher"] == "gp"
    # Engine metrics are recorded verbatim — never recomputed.
    assert row["metrics"]["calmar"] == 2.0
    assert row["gates"]["all"] is True
    # A selected-less or failed Stage-A row is a recorded failure row, not
    # a silent gap.
    broken = dict(_stage_row(7, "gp", 1))
    broken["search_result"] = {"selected": None}
    stage_b_broken = adj.evaluate_selected_rows(
        [broken],
        loader=object(),
        fold=object(),
        bt_cfg=object(),
        evaluate_fn=fake_evaluate,
    )
    assert stage_b_broken[0]["failed"] is True
    assert stage_b_broken[0]["gates"] is None

    def boom(*args, **kwargs):
        raise RuntimeError("formula invalid on the test window")

    stage_b_boom = adj.evaluate_selected_rows(
        [_stage_row(2024, "tpe", 1)],
        loader=object(),
        fold=object(),
        bt_cfg=object(),
        evaluate_fn=boom,
    )
    assert stage_b_boom[0]["failed"] is True
    assert "formula invalid" in stage_b_boom[0]["error"]


# ---------------------------------------------------------------------------
# §5.3/§8: arm aggregates and the pre-registered decision mapping
# ---------------------------------------------------------------------------


def _metrics_all_pass(icir=0.4):
    return {
        "annual_return": 0.2,
        "max_drawdown": 0.1,
        "sharpe": 1.5,
        "calmar": 2.0,
        "icir": icir,
    }


def _metrics_all_fail(icir=0.4):
    return {
        "annual_return": 0.01,
        "max_drawdown": 0.4,
        "sharpe": 0.1,
        "calmar": 0.02,
        "icir": icir,
    }


def test_p10_arm_aggregates_paired_median_and_majority():
    rows = [
        _stage_row(42, "gp", 1),
        _stage_row(7, "gp", 1),
        _stage_row(2024, "gp", 1),
    ]
    areas = {(42, "gp"): 100.0, (7, "gp"): 300.0, (2024, "gp"): 200.0}
    stage_b = [
        {"seed": 42, "searcher": "gp", "failed": False, "gates": gates_passed(_metrics_all_pass()), "metrics": _metrics_all_pass(0.3)},
        {"seed": 7, "searcher": "gp", "failed": False, "gates": gates_passed(_metrics_all_pass()), "metrics": _metrics_all_pass(0.1)},
        {"seed": 2024, "searcher": "gp", "failed": False, "gates": gates_passed(_metrics_all_pass()), "metrics": _metrics_all_pass(0.2)},
    ]
    agg = arm_aggregates(rows, stage_b, areas)
    assert agg["gp"]["area_median"] == pytest.approx(200.0)
    assert agg["gp"]["admissible"] is True  # 3/3 seeds pass all gates
    assert agg["gp"]["oos_positive"] is True  # 3/3 seeds icir > 0
    assert agg["gp"]["stagnant"] is False


def test_p10_decision_mapping_pre_registered_rows():
    # --- case 1: exactly one admissible arm and it is the efficient one.
    rows = []
    stage_b = []
    areas = {}
    for arm, admissible in (("gp", True), ("tpe", False), ("random", False), ("rl", False)):
        for i, seed in enumerate((42, 7, 2024)):
            rows.append(_stage_row(seed, arm, 1))
            areas[(seed, arm)] = 300.0 + i if arm == "gp" else 100.0 + i
            metrics = _metrics_all_pass() if admissible else _metrics_all_fail()
            stage_b.append(
                {
                    "seed": seed,
                    "searcher": arm,
                    "failed": False,
                    "gates": gates_passed(metrics),
                    "metrics": metrics,
                }
            )
    verdict = adjudicate(rows, stage_b, areas)
    assert verdict["headline"] == "invest_in_searcher"
    assert verdict["arm"] == "gp"
    assert verdict["matched_rows"][0] == "exactly_one_admissible_and_efficient"

    # --- case 2: two admissible arms -> budget-scaling study on the
    # efficient arm; no single-winner claim.
    rows, stage_b, areas = [], [], {}
    for arm in ("gp", "tpe", "random", "rl"):
        admissible = arm in ("gp", "tpe")
        for i, seed in enumerate((42, 7, 2024)):
            rows.append(_stage_row(seed, arm, 1))
            areas[(seed, arm)] = 400.0 + i if arm == "tpe" else 200.0 + i
            metrics = _metrics_all_pass() if admissible else _metrics_all_fail()
            stage_b.append(
                {
                    "seed": seed,
                    "searcher": arm,
                    "failed": False,
                    "gates": gates_passed(metrics),
                    "metrics": metrics,
                }
            )
    verdict = adjudicate(rows, stage_b, areas)
    assert verdict["headline"] == "budget_scaling_study"
    assert verdict["arm"] == "tpe"

    # --- case 3: no arm admissible -> legal negative result.
    rows, stage_b, areas = [], [], {}
    for arm in ("gp", "tpe", "random", "rl"):
        for seed in (42, 7, 2024):
            rows.append(_stage_row(seed, arm, 1))
            areas[(seed, arm)] = 100.0
            stage_b.append(
                {
                    "seed": seed,
                    "searcher": arm,
                    "failed": False,
                    "gates": gates_passed(_metrics_all_fail()),
                    "metrics": _metrics_all_fail(),
                }
            )
    verdict = adjudicate(rows, stage_b, areas)
    assert verdict["headline"] == "negative_no_admissible_formula"
    assert verdict["arm"] is None

    # --- case 4 (additional finding): every arm stagnated well below the
    # requested budget -> proposal-diversity finding recorded on top.
    rows, stage_b, areas = [], [], {}
    for arm in ("gp", "tpe", "random", "rl"):
        for seed in (42, 7, 2024):
            row = _stage_row(seed, arm, 1, consumed=90, stagnation="x")
            rows.append(row)
            areas[(seed, arm)] = 10.0
            stage_b.append(
                {
                    "seed": seed,
                    "searcher": arm,
                    "failed": False,
                    "gates": gates_passed(_metrics_all_fail()),
                    "metrics": _metrics_all_fail(),
                }
            )
    verdict = adjudicate(rows, stage_b, areas)
    assert verdict["headline"] == "negative_no_admissible_formula"
    assert "all_arms_stagnant_proposal_diversity" in verdict["additional_findings"]

    # --- case 5 (additional finding): the efficient arm is not OOS
    # positive (icir <= 0 on the test window for >= 2/3 seeds).
    rows, stage_b, areas = [], [], {}
    for arm in ("gp", "tpe", "random", "rl"):
        for seed in (42, 7, 2024):
            rows.append(_stage_row(seed, arm, 1))
            areas[(seed, arm)] = 400.0 if arm == "gp" else 100.0
            metrics = _metrics_all_fail(icir=-0.2 if arm == "gp" else 0.4)
            stage_b.append(
                {
                    "seed": seed,
                    "searcher": arm,
                    "failed": False,
                    "gates": gates_passed(metrics),
                    "metrics": metrics,
                }
            )
    verdict = adjudicate(rows, stage_b, areas)
    assert "efficient_arm_not_oos_positive_reward_audit" in verdict["additional_findings"]
