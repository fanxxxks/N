"""P50 WP14 plan-stage RED/GREEN tests: run-plan parsing and
materialization (R1).

Contract: ``docs/p50_experiment_integrity_draft.md`` §3.1/§8 (the WP14
plan stage; the contract file lands with the WP14 contract-writing
stage per DESIGN §1 — every clause asserted here is pinned to a DESIGN
§8 row and a measured fact F2/F8/F9/F10/F11/F19/F22/F23).

Invariants (plan stage — the DESIGN §9 INV numbering belongs to the
full-suite file; this module uses PL1-PL6 to avoid collision):

* PL1 — all four experiments materialize a NON-EMPTY arm list that
  explicitly contains the benchmark comparison arm (and the cash
  control where the experiment runs one);
* PL2 — R1 explicitness: every arm carries candidate / members /
  direction rule / cost ladder / primary flag, folds are explicit
  (train_end, test_end), and the comparison arms are listed as their
  own kinds, never mixed among signal arms;
* PL3 — the slow plan's materialized expected row set matches the REAL
  ``r01_baseline_report.json`` (125 rows, sha256-anchored, set-equal
  keys: missing=0, unexpected=0) — the recon-measured scale, not a
  narrative;
* PL4 — expected key sets never fold economic objects: distinct key
  counts 125/310/120/140 with the period axis carried on every
  r02_periods key;
* PL5 — the plan JSON round-trips losslessly (CLI offline audit path);
* PL6 — constructible violations refuse locatably BEFORE any data
  load: unknown experiment id, config missing folds/baseline_signals
  (PlanValidationError with (experiment, locator) messages).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ashare_model.experiment_integrity.plan import (
    VERDICT_MAPS,
    expected_keys,
    materialize_plan,
    plan_from_json,
    plan_to_json,
)
from ashare_model.experiment_integrity.types import (
    ARM_KINDS,
    EXPERIMENT_IDS,
    PlanValidationError,
    RowKey,
)

#: The frozen r1 artifact root, resolved from the package (never from
#: the process cwd) — read-only access, the WP14 red line.
_ROOT = Path(__file__).resolve().parents[1]
_R1_SLOW_REPORT = _ROOT / "data" / "r1" / "r01_slow_baseline" / "r01_baseline_report.json"

#: Recon-measured facts this suite anchors on (reports/wp14/_recon):
#: F1 the slow report digest, F2/F8/F9/F10 the four row counts.
_SLOW_REPORT_SHA256 = (
    "95fac73249c0ee93f86e0f385ce25ddf5ae7e7a92bc830db0706710bc05f550b"
)
_VERIFIED_ROW_COUNTS = {
    "r01_slow_baseline": 125,
    "r02_matrix": 310,
    "r02_periods": 120,
    "r03_combos": 140,
    # p56 (WP18 FROZEN_18_PLAN §4): measurement rows (M1 30 + M2 40 /
    # S1 50 + S2 60) + formal rows (M3 90 + C1 40 / S3 130 + C2 40) per
    # domain; the FORMAL sub-counts sum to exactly 300 across domains.
    "wp18_medium": 200,
    "wp18_slow": 280,
}
#: DESIGN §8 — verified expanded-arm counts (r02_matrix == its 32
#: arm_summaries keys F8, r03 == its 16 keys F10).
_VERIFIED_ARM_COUNTS = {
    "r01_slow_baseline": 9,
    "r02_matrix": 32,
    "r02_periods": 14,
    "r03_combos": 16,
    # p56 (WP18 §4 via wp18_planned_arms): 6/10 direction + 8/12 summary
    # + 18/26 account (16/24 signal + 2 controls) + 8 stress arms.
    "wp18_medium": 40,
    "wp18_slow": 56,
}


def _row_key(row: dict) -> RowKey:
    """The plan-side key projection of one report row (reconcile §3.3
    builds the same five-axis key; here it only feeds PL3/PL4)."""

    return RowKey(
        period=row.get("period"),
        cell=row.get("cell"),
        candidate=row.get("candidate", ""),
        cost_level=row.get("cost_level", ""),
        fold_test_end=row.get("fold_test_end", ""),
    )


# --- PL1: four plans parse, non-empty, comparison arms in list --------


@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_p50_pl1_plan_parses_nonempty_arms_with_benchmark(experiment_id):
    plan = materialize_plan(experiment_id)
    assert plan.arms, f"{experiment_id}: arm list must not be empty"
    assert len(plan.arms) == _VERIFIED_ARM_COUNTS[experiment_id]
    kinds = {arm.kind for arm in plan.arms}
    assert kinds <= set(ARM_KINDS)
    benchmarks = [a for a in plan.arms if a.kind == "benchmark"]
    assert [a.candidate for a in benchmarks] == ["benchmark:equal_weight"] * len(
        benchmarks
    )
    # The benchmark arm is comparison-only: no members, no direction
    # rule, the zero-cost 'ref' ladder, never primary (R1 listing).
    for arm in benchmarks:
        assert arm.members == ()
        assert arm.direction_rule is None
        assert arm.cost_levels == ("ref",)
        assert arm.is_primary is False
    # Cash control runs in every experiment except r03 (recon §4).
    cash = [a for a in plan.arms if a.kind == "control"]
    if experiment_id == "r03_combos":
        assert cash == []
    else:
        assert [a.candidate for a in cash] == ["control:cash"] * len(cash)


def test_p50_pl1_r02_periods_expands_control_arms_over_periods():
    # F19: under r02_periods BOTH comparison arms expand over the two
    # periods (cash 2x5 rows, benchmark 2x5 rows) — the plan must list
    # them per period, not once.
    plan = materialize_plan("r02_periods")
    bench_periods = sorted(
        a.period for a in plan.arms if a.kind == "benchmark"
    )
    cash_periods = sorted(a.period for a in plan.arms if a.kind == "control")
    assert bench_periods == ["every_10_days", "every_5_days"]
    assert cash_periods == ["every_10_days", "every_5_days"]
    for arm in plan.arms:
        if arm.kind in ("benchmark", "control"):
            # benchmark carries NO cell; cash carries the reference cell
            # while its candidate string keeps the bare 'control:cash'.
            if arm.kind == "benchmark":
                assert arm.cell is None
            else:
                assert arm.cell == "n20_m5000"
                assert arm.candidate == "control:cash"


# --- PL2: R1 — the run plan lists every element explicitly -------------


@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_p50_pl2_run_plan_lists_all_r1_elements(experiment_id):
    plan = materialize_plan(experiment_id)

    # Folds: the L2 config's five folds, explicit on both ends.
    assert [f.test_end for f in plan.folds] == [
        "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31",
    ]
    assert [f.train_end for f in plan.folds] == [
        "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31",
    ]

    signal_arms = [a for a in plan.arms if a.kind in ("single", "combo")]
    assert signal_arms, "an experiment plan without signal arms is empty"
    for arm in signal_arms:
        # candidate, members, direction rule, cost ladder: all explicit,
        # all non-default (a missing member or empty ladder is an
        # under-specified plan, not a plan).
        assert arm.candidate and arm.kind in ("single", "combo")
        assert arm.members and all(arm.members)
        assert len(arm.cost_levels) >= 1 and all(arm.cost_levels)
        assert arm.members_source in (
            "config.protocol.baseline_signals",
            "code.R01_COMBO_ARMS",
            "code.R03_MEMBERS",
            # p56 (WP18 §1): the preregistered candidate registry of the
            # frozen plan is itself the members source.
            "wp18_frozen_plan§1(FROZEN_18_PLAN candidates)",
        )
        rule = arm.direction_rule
        assert rule is not None
        # p56 (WP18 §2): the WP18 plans carry BOTH frozen rules A/B — the
        # historical four run rule A only, unchanged.
        if experiment_id.startswith("wp18_"):
            assert rule.rule in ("reference_return", "label_ic")
        else:
            assert rule.rule == "reference_return"
        assert rule.rule_version == 1  # direction.py:38 at run time (§8)
        assert rule.scope in ("per_fold", "per_(period,fold)")
        assert rule.learned_at  # non-empty provenance string


def test_p50_pl2_cost_ladders_are_per_arm_not_global():
    # S §2.2: the ladder is asymmetric — the plan must carry it per arm,
    # never assume one uniform ladder.
    r01 = materialize_plan("r01_slow_baseline")
    by_candidate = {a.candidate: a for a in r01.arms}
    assert by_candidate["baseline:ROE"].cost_levels == ("0x", "1x", "2x")
    assert by_candidate["combo:all3"].cost_levels == ("0x", "1x", "2x")
    assert by_candidate["control:cash"].cost_levels == ("0x", "1x", "2x")
    assert by_candidate["benchmark:equal_weight"].cost_levels == ("ref",)

    r02m = materialize_plan("r02_matrix")
    cash = next(a for a in r02m.arms if a.kind == "control")
    assert cash.cost_levels == ("1x",)
    signal = next(a for a in r02m.arms if a.kind == "single")
    assert signal.cost_levels == ("0x", "1x")

    r03 = materialize_plan("r03_combos")
    ref_single = next(
        a for a in r03.arms if a.kind == "single" and a.cell == "n10_m2500"
    )
    assert ref_single.cost_levels == ("1x",)
    combo = next(a for a in r03.arms if a.kind == "combo")
    assert combo.cost_levels == ("0x", "1x")


def test_p50_pl2_direction_rules_match_design_table():
    rules = {}
    for experiment_id in EXPERIMENT_IDS:
        plan = materialize_plan(experiment_id)
        arm = next(
            a for a in plan.arms if a.kind in ("single", "combo")
        )
        rules[experiment_id] = arm.direction_rule
    assert rules["r01_slow_baseline"].scope == "per_fold"
    assert rules["r01_slow_baseline"].learned_at == (
        "none(reference=own train window)"
    )
    assert rules["r02_matrix"].scope == "per_fold"
    assert rules["r02_matrix"].learned_at == "reference_cell=n20_m5000"
    # p39 §3: the ONLY per-(period, fold) relearn in the family.
    assert rules["r02_periods"].scope == "per_(period,fold)"
    assert rules["r02_periods"].learned_at == "reference_cell=n20_m5000"
    assert rules["r03_combos"].scope == "per_fold"
    assert rules["r03_combos"].learned_at == "reference_cell=n10_m2500"


def test_p50_pl2_primary_arms_frozen_per_experiment():
    # §8: primary = the preregistered arms the adjudicator consumes.
    r01 = materialize_plan("r01_slow_baseline")
    assert sorted(a.candidate for a in r01.primary_arms) == sorted(
        [
            "baseline:ROE", "baseline:PE_TTM", "baseline:REVENUE_YOY",
            "baseline:DIVIDEND_YIELD", "baseline:MARKET_CAP",
            "combo:all3", "combo:momentum_reversal",
        ]
    )
    assert len(materialize_plan("r02_matrix").primary_arms) == 30
    assert len(materialize_plan("r02_periods").primary_arms) == 10
    # r03: only the two adjudicated arms per cell are primary — the
    # descriptive lambdas and the reference singles never are (p42 §3).
    r03 = materialize_plan("r03_combos")
    assert sorted({a.candidate for a in r03.primary_arms}) == sorted(
        {f"{cell}::{arm}" for cell in ("n10_m1000", "n10_m2500", "n10_m5000")
         for arm in ("combo:all3", "combo:fitted_lam050")}
    )
    assert not any(a.is_primary for a in r03.arms if a.kind == "single")


def test_p50_pl2_combo_members_frozen_verbatim():
    # INV3 flavor at plan level: the plan carries R01_COMBO_ARMS
    # verbatim — it must never re-derive or re-order members.
    plan = materialize_plan("r01_slow_baseline")
    by_candidate = {a.candidate: a for a in plan.arms}
    assert by_candidate["combo:all3"].members == (
        "MOMENTUM_20", "RSQ_60", "REVERSAL_60",
    )
    assert by_candidate["combo:momentum_reversal"].members == (
        "MOMENTUM_20", "REVERSAL_60",
    )


def test_p50_pl2_sources_snapshot_and_verdict_map_are_plan_data():
    for experiment_id in EXPERIMENT_IDS:
        plan = materialize_plan(experiment_id)
        sources = plan.sources
        # L1/L2 digests computed from the current files (64 hex chars).
        assert len(sources["contract"]["sha256"]) == 64
        assert len(sources["config"]["sha256"]) == 64
        assert sources["code_constants"], "L3 names must be recorded"
        assert sources["product"]["report_path"].startswith("data/r1/")
        # F22 adjudication outcome-word location declared per experiment.
        assert sources["adjudication_path"]
        # F20: only r02_periods carries the period-prefiltered summary
        # layer with the reference-cell-signal arm domain.
        if experiment_id == "r02_periods":
            assert sources["summary_layer_period_prefilter"] is True
            assert sources["summary_layer_arms"] == (
                "reference_cell_signal_arms"
            )
        else:
            assert sources["summary_layer_period_prefilter"] is False
            assert sources["summary_layer_arms"] == "all_arms"
        assert plan.verdict_map == VERDICT_MAPS[experiment_id]


def test_p50_pl2_verdict_map_same_word_different_experiments():
    # F23: 'panel_no_reliable_info' is NEGATIVE in r02_matrix (no
    # viable cell, no gross info — the bottleneck is the signal) but
    # MIXED in r03 (equal-weight survives 2/3 cells) — same word,
    # different economic object, never a cross-experiment wildcard.
    assert VERDICT_MAPS["r02_matrix"]["panel_no_reliable_info"] == (
        "economic_negative"
    )
    assert VERDICT_MAPS["r03_combos"]["panel_no_reliable_info"] == (
        "inconclusive_mixed"
    )
    # UNSTABLE / execution words are never negative (§5.5).
    for word in (
        "isolated_single_winner",
        "multiple_disjoint_viable",
        "gross_info_execution_blocked",
        "isolated_period_winner",
        "gross_info_execution_blocked_period_panel",
    ):
        for table in VERDICT_MAPS.values():
            assert table.get(word) != "economic_negative"


# --- PL3: the slow plan reproduces the REAL artifact's row set ---------


def test_p50_pl3_slow_expected_rows_match_real_artifact():
    assert _R1_SLOW_REPORT.is_file(), "the frozen slow artifact must exist"
    digest = hashlib.sha256(_R1_SLOW_REPORT.read_bytes()).hexdigest()
    assert digest == _SLOW_REPORT_SHA256  # F1: the audited bytes

    plan = materialize_plan("r01_slow_baseline")
    assert plan.expected_row_count == 125  # F2, recon §1
    report = json.loads(_R1_SLOW_REPORT.read_text(encoding="utf-8"))
    assert len(report["rows"]) == 125

    expected = set(expected_keys(plan))
    actual = {_row_key(row) for row in report["rows"]}
    assert len(expected) == 125
    assert expected - actual == set(), "missing rows vs the real artifact"
    assert actual - expected == set(), "unexpected rows vs the real artifact"


@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_p50_pl4_expected_keys_never_fold(experiment_id):
    plan = materialize_plan(experiment_id)
    keys = expected_keys(plan)
    expected_n = _VERIFIED_ROW_COUNTS[experiment_id]
    assert len(keys) == expected_n
    assert len(set(keys)) == expected_n  # no key collision
    if experiment_id == "r02_periods":
        # F9: every key carries the period axis — dropping it would fold
        # 120 rows onto 60 keys of multiplicity 2 (two periods share
        # fold names); the plan side keeps the axis load-bearing.
        assert all(key.period in ("every_5_days", "every_10_days") for key in keys)
        folded = {
            (key.cell, key.candidate, key.cost_level, key.fold_test_end)
            for key in keys
        }
        assert len(folded) == 60  # the counterfactual collapse, measured
    if experiment_id == "r01_slow_baseline":
        # r01 declares no cell/period axes — keys stay neutral there.
        assert all(key.cell is None and key.period is None for key in keys)


# --- PL5: JSON round trip (CLI offline audit path) ----------------------


@pytest.mark.parametrize("experiment_id", EXPERIMENT_IDS)
def test_p50_pl5_plan_json_round_trip_is_lossless(experiment_id):
    plan = materialize_plan(experiment_id)
    payload = plan_to_json(plan)
    # The dict itself must be jsonable with the repo's write convention.
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    assert json.loads(encoded) == payload
    restored = plan_from_json(json.loads(encoded))
    assert restored == plan
    assert restored.expected_row_count == plan.expected_row_count
    assert [a.candidate for a in restored.arms] == [
        a.candidate for a in plan.arms
    ]


# --- PL6: constructible violations refuse locatably --------------------


def test_p50_pl6_unknown_experiment_refuses_with_locator():
    with pytest.raises(PlanValidationError, match=r"locator=experiment_id"):
        materialize_plan("r04_does_not_exist")


def test_p50_pl6_config_missing_folds_refuses_before_data_load(tmp_path):
    config = tmp_path / "broken.yaml"
    config.write_text(
        "protocol:\n  domain: slow_fundamental\n  baseline_signals: [ROE]\n",
        encoding="utf-8",
    )
    with pytest.raises(PlanValidationError, match=r"protocol\.folds"):
        materialize_plan("r01_slow_baseline", config)


def test_p50_pl6_config_missing_baseline_signals_refuses(tmp_path):
    config = tmp_path / "broken.yaml"
    config.write_text(
        "protocol:\n"
        "  domain: slow_fundamental\n"
        "  folds:\n"
        '    - {train_end: "2020-12-31", test_end: "2021-12-31"}\n',
        encoding="utf-8",
    )
    with pytest.raises(PlanValidationError, match=r"protocol\.baseline_signals"):
        materialize_plan("r01_slow_baseline", config)


def test_p50_pl6_missing_config_file_refuses_with_config_locator(tmp_path):
    absent = tmp_path / "not_there.yaml"
    with pytest.raises(PlanValidationError, match=r"locator=config:"):
        materialize_plan("r01_slow_baseline", absent)


def test_p50_pl6_plan_from_json_rejects_wrong_schema_version():
    plan = materialize_plan("r01_slow_baseline")
    payload = plan_to_json(plan)
    payload["plan_schema_version"] = 99
    with pytest.raises(PlanValidationError, match=r"plan_schema_version"):
        plan_from_json(payload)
