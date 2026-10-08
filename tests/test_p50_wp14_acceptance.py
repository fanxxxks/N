"""P50 WP14 acceptance RED/GREEN tests: the four R7 counterexamples,
the real-artifact read-only pipeline, and the happy path — every test
END-TO-END through the landed stages (``materialize_plan`` ->
``run_precheck`` / ``reconcile_report`` -> ``classify_status``), never
through a hand-built Reconciliation.

Contract: ``docs/p50_experiment_integrity_draft.md`` §6 (the contract
file lands with the WP14 contract-writing stage per DESIGN §1; the
counterexample matrix itself is ``reports/wp14/DESIGN.md`` §6 — every
scenario asserted here is that matrix's row, and every number is a
measured fact F1-F4/F7 from reports/wp14/_recon/report_schemas.md).

Invariants (acceptance stage; the stage files use PL/PC/RC/ST/ER —
this module uses the DESIGN §6 names R7-S1..S4 plus ACC1/ACC2):

* R7-S1 — wrong-domain member: the REAL slow plan's combo members
  (MOMENTUM_20/RSQ_60/REVERSAL_60, medium_cross_section features) are
  rejected by the pre-load precheck with locatable errors on exactly
  the KeyError root-cause members, while the five in-domain singles
  and both comparison arms stay clean — and the frozen artifact's own
  reconciliation still delivers the five singles citable (reject the
  wrong verdict, preserve the usable part);
* R7-S2 — one primary arm missing: a synthetic r01-shaped report
  minus all 15 combo:all3 rows reconciles to exactly those 15 missing
  keys and classifies measurement_incomplete — NOT economic_negative,
  even though the same complete report with worth_search=false IS a
  clean negative (the incompleteness, and only it, blocks the verdict)
  — while every other arm's measurement is preserved untouched;
* R7-S3 — one duplicated fold: a synthetic report with one row run
  twice reports that key with multiplicity 2 (126 rows over 125 keys,
  never silently deduplicated) and classifies
  measurement_incomplete — a duplicate shadows even a positive frozen
  verdict — with zero false missing-row reports and every arm still
  citable;
* R7-S4 — all combos failed, singles succeed: a synthetic replica of
  the real slow form (30 failed combo rows, single variant
  "KeyError: 'MOMENTUM_20'", completed=true, worth_search=false)
  classifies measurement_incomplete with the adjudication quarantined
  — never economic_negative — while the five singles, cash and
  benchmark come out citable and both combos program_error with the
  verbatim reason;
* ACC1 — the REAL artifact, read-only, end-to-end: sha256-anchored,
  the pipeline lands on measurement_incomplete, the combo failure is
  attributed to KeyError: 'MOMENTUM_20', and the five slow singles are
  citable with zero failed rows — the live regression guard for the
  landed errata facts (data/r1/r01_slow_baseline/errata.json);
* ACC2 — the happy path: a synthetic complete error-free report makes
  all three layers true (report_generated / primary_arms_measured /
  verdict_valid), the overall outcome is an economic label with the
  adjudication read rather than quarantined — worth_search:false is
  economic_negative and worth_search:true is economic_positive (M3: a
  true negative is judged, never swallowed).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import pytest

from ashare_model.experiment_integrity.plan import (
    expected_keys,
    materialize_plan,
)
from ashare_model.experiment_integrity.precheck import run_precheck
from ashare_model.experiment_integrity.reconcile import (
    reconcile_report,
    row_key_of,
)
from ashare_model.experiment_integrity.status import classify_status

#: The frozen r1 artifact root, resolved from the package (never from
#: the process cwd) — read-only access, the WP14 red line.
_ROOT = Path(__file__).resolve().parents[1]
_SLOW_DIR = _ROOT / "data" / "r1" / "r01_slow_baseline"
_SLOW_REPORT = _SLOW_DIR / "r01_baseline_report.json"
_SLOW_RUN_META = _SLOW_DIR / "r01_baseline_run_meta.json"
_SLOW_ERRATA = _SLOW_DIR / "errata.json"

#: F1 (DESIGN §0): the audited slow-report bytes — the anchor every
#: ACC1 measurement below is taken on, and the digest the landed
#: errata target block pins.
_SLOW_REPORT_SHA256 = (
    "95fac73249c0ee93f86e0f385ce25ddf5ae7e7a92bc830db0706710bc05f550b"
)
_SLOW_REPORT_BYTES = 2049495

#: F4: the measured failed-row layer — 30 = 15 + 15 on the full
#: {0x,1x,2x} x 5-fold grid, one verbatim error variant.
_KEYERROR_REASON = "KeyError: 'MOMENTUM_20'"
_COMBO_ARMS = ("combo:all3", "combo:momentum_reversal")
_CITABLE_SINGLES = (
    "baseline:ROE",
    "baseline:PE_TTM",
    "baseline:REVENUE_YOY",
    "baseline:DIVIDEND_YIELD",
    "baseline:MARKET_CAP",
)
#: The five real folds of configs/r1_slow_ab.yaml protocol.folds (the
#: plan materializes them; the synthetic rows below reuse the dates).
_REAL_FOLD_ENDS = (
    "2021-12-31",
    "2022-12-31",
    "2023-12-31",
    "2024-12-31",
    "2025-12-31",
)

#: A deliberately tiny synthetic test calendar (nothing like the real
#: artifacts' 221-237-day calendars): reconcile checks coverage WITHIN
#: one report only (S §6 — cross-experiment date comparison is
#: forbidden), so a 3-day calendar is fully legal for a synthetic
#: r01-shaped report.
_SYNTH_CALENDAR = ["20210104", "20210105", "20210106"]


def _real_slow_artifact():
    """(plan, report, run_meta) of the frozen slow artifact, read-only."""

    plan = materialize_plan("r01_slow_baseline")
    if not _SLOW_REPORT.is_file():
        pytest.skip("optional frozen historical run archive is not shipped with source")
    report = json.loads(_SLOW_REPORT.read_text(encoding="utf-8"))
    run_meta = json.loads(_SLOW_RUN_META.read_text(encoding="utf-8"))
    return plan, report, run_meta


def _train_end_by_test_end(plan) -> dict[str, str]:
    return {fold.test_end: fold.train_end for fold in plan.folds}


def _success_row(key, train_end: str) -> dict:
    """One synthetic r01-shaped successful row on a 3-day calendar."""

    return {
        "candidate": key.candidate,
        "formula_text": "synthetic",
        "formula": "synthetic",
        "fold_train_end": train_end,
        "fold_test_end": key.fold_test_end,
        "seed": None,
        "cost_level": key.cost_level,
        "failed": False,
        "n_dates": len(_SYNTH_CALENDAR),
        "daily_dates": list(_SYNTH_CALENDAR),
        "daily_returns": [0.01] * len(_SYNTH_CALENDAR),
        "benchmark_daily_returns": [0.01] * len(_SYNTH_CALENDAR),
        "total_return": 0.0,
        "sharpe": 0.0,
        "excess_return": 0.0,
        "val_reward": 0.0,
    }


def _failed_row(key, train_end: str, reason: str) -> dict:
    """One synthetic r01-shaped failed stub (the S §2.3 minimal shape:
    identity plus failure, no metrics)."""

    return {
        "candidate": key.candidate,
        "formula_text": "synthetic",
        "fold_train_end": train_end,
        "fold_test_end": key.fold_test_end,
        "seed": None,
        "cost_level": key.cost_level,
        "failed": True,
        "failure_reason": reason,
    }


def _synth_summaries(plan, failed_folds_by_arm=None) -> dict:
    """An arm_summaries layer consistent with a synthetic rows layer
    (failed_folds per arm; 0 for every arm unless overridden)."""

    overrides = failed_folds_by_arm or {}
    return {
        arm.candidate: {
            "failed_folds": overrides.get(arm.candidate, 0),
            "n_folds": len(plan.folds),
        }
        for arm in plan.arms
    }


def _slow_report(plan, rows, *, worth_search, failed_folds_by_arm=None):
    """A synthetic r01-shaped report around the given rows: schema key,
    completed stamp, the rows layer, a summary layer consistent with
    it, a frozen adjudication block, and the provenance anchors the
    real reports carry (taken from the plan's own L1/L2 snapshot, so
    the synthetic report passes the anchoring comparison like the real
    ones do)."""

    return {
        "r01_report_schema_version": 3,
        "completed": True,
        "rows": rows,
        "arm_summaries": _synth_summaries(plan, failed_folds_by_arm),
        "provenance": {
            "config_sha256": plan.sources["config"]["sha256"],
            "contract_sha256": plan.sources["contract"]["sha256"],
        },
        "adjudication": {
            "worth_search": worth_search,
            "bottleneck": "synthetic: no_qualified_arm",
        },
    }


def _clean_slow_report(plan, *, worth_search=False):
    """The complete error-free synthetic r01-shaped report: one
    successful row per expected key (125 rows), all arms citable."""

    train = _train_end_by_test_end(plan)
    rows = [
        _success_row(key, train[key.fold_test_end])
        for key in expected_keys(plan)
    ]
    assert len(rows) == 125  # 5 singles x3x5 + 2 combos x3x5 + cash x3x5 + bench x1x5
    return _slow_report(plan, rows, worth_search=worth_search)


def _status_by_candidate(status):
    return {arm.candidate: arm for arm in status.arm_status}


# --- R7-S1: wrong-domain member — precheck rejects, singles stay citable --


def test_p50_r7_s1_wrong_domain_member_precheck_rejects_and_singles_stay_citable():
    # DESIGN §6 row 1.  RED: the REAL materialized slow plan carries
    # combo members that live in medium_cross_section, not
    # slow_fundamental AB — the exact load-time image of the 30
    # measured KeyError: 'MOMENTUM_20' failed rows (F4).  The precheck
    # must refuse the plan before any data load, with a failure
    # locatable to (experiment, arm, member).
    plan, report, run_meta = _real_slow_artifact()
    pre = run_precheck(plan)

    assert pre.ok is False
    assert pre.checks_run == ("P-DOM", "P-DEP", "P-REF")
    domain_issues = [
        i for i in pre.issues if i.code == "member_outside_domain"
    ]
    # 3 members in combo:all3 + 2 in combo:momentum_reversal, plan order.
    assert [(i.arm, i.member) for i in domain_issues] == [
        ("combo:all3", "MOMENTUM_20"),
        ("combo:all3", "RSQ_60"),
        ("combo:all3", "REVERSAL_60"),
        ("combo:momentum_reversal", "MOMENTUM_20"),
        ("combo:momentum_reversal", "REVERSAL_60"),
    ]
    # Locatable: the first issue names experiment, arm and member.
    assert pre.issues[0].locator == (
        "(experiment=r01_slow_baseline, arm=combo:all3, member=MOMENTUM_20)"
    )
    assert pre.issues[0].experiment_id == "r01_slow_baseline"
    assert "KeyError" in pre.issues[0].detail  # points at what it prevents
    # The same defect under the reference lens (§3.2 bullet 3).
    codes = Counter(i.code for i in pre.issues)
    assert codes["combo_member_not_in_plan_singles"] == 5

    # GREEN (preserve the usable part): the five in-domain single arms
    # and both comparison arms produce no issue of any code — the
    # rejection never smears the legal arms.
    single_names = set(_CITABLE_SINGLES)
    comparison_names = {"benchmark:equal_weight", "control:cash"}
    assert not [i for i in pre.issues if i.arm in single_names]
    assert not [i for i in pre.issues if i.arm in comparison_names]

    # GREEN (the artifact's own measurement survives): reconciling the
    # frozen report still delivers the five slow singles citable — the
    # precheck rejection of the PLAN never deletes the artifact's
    # measured rows (post-hoc audit, read-only).
    recon = reconcile_report(plan, report, run_meta)
    by_candidate = {arm.candidate: arm for arm in recon.arm_status}
    for candidate in _CITABLE_SINGLES:
        assert by_candidate[candidate].status == "citable", candidate
        assert by_candidate[candidate].n_failed_rows == 0
    for candidate in _COMBO_ARMS:
        assert by_candidate[candidate].status == "program_error", candidate


# --- R7-S2: one primary arm missing — incomplete, others kept ------------


def test_p50_r7_s2_missing_primary_arm_incomplete_but_other_rows_kept():
    # DESIGN §6 row 2: a synthetic r01-shaped report with every
    # combo:all3 row deleted (110 rows left, no failed rows anywhere,
    # a negative-looking frozen adjudication).
    plan, _report, _meta = _real_slow_artifact()
    complete = _clean_slow_report(plan, worth_search=False)
    rows = [
        row for row in complete["rows"] if row["candidate"] != "combo:all3"
    ]
    assert len(rows) == 110  # 125 - 15 deleted rows
    tampered = dict(complete, rows=rows)

    # Control first: the COMPLETE synthetic report with worth_search
    # false is a clean negative (M3) — so if the 110-row version came
    # out economic_negative, absence would have been read as refutation.
    full_recon = reconcile_report(plan, complete)
    full_status = classify_status(plan, full_recon, complete)
    assert full_status.label == "economic_negative"

    recon = reconcile_report(plan, tampered)
    status = classify_status(plan, recon, tampered)

    # RED: exactly the 15 deleted keys are missing — the complete
    # {0x,1x,2x} x 5-fold grid of ONE primary arm, fully locatable.
    assert len(recon.missing) == 15
    assert {key.candidate for key in recon.missing} == {"combo:all3"}
    assert {(key.cost_level, key.fold_test_end) for key in recon.missing} == {
        (level, fold)
        for level in ("0x", "1x", "2x")
        for fold in _REAL_FOLD_ENDS
    }
    assert Counter(i.code for i in recon.issues)["missing_row"] == 15
    # RED: measurement_incomplete — explicitly NOT an economic negative.
    assert status.report_generated is True
    assert status.primary_arms_measured is False
    assert status.verdict_valid is False
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_negative"
    assert status.adjudication_quarantined is True
    assert len(status.primary_arms_evidence["m1_primary_missing_keys"]) == 15

    # GREEN: every OTHER arm's measurement is preserved untouched —
    # the four kept singles, the other combo, cash and benchmark all
    # reconcile citable, and no failure is smeared onto the kept rows.
    assert recon.duplicates == ()
    assert recon.unexpected == ()
    assert recon.failed_rows == ()
    by_candidate = _status_by_candidate(status)
    assert by_candidate["combo:all3"].status == "missing"
    assert by_candidate["combo:all3"].n_present_rows == 0
    assert by_candidate["combo:all3"].n_expected_rows == 15
    for candidate in (
        "baseline:ROE",
        "baseline:PE_TTM",
        "baseline:REVENUE_YOY",
        "baseline:DIVIDEND_YIELD",
        "baseline:MARKET_CAP",
        "combo:momentum_reversal",
        "control:cash",
        "benchmark:equal_weight",
    ):
        assert by_candidate[candidate].status == "citable", candidate
        assert by_candidate[candidate].n_failed_rows == 0


# --- R7-S3: one duplicated fold — reported, never overwritten ------------


def test_p50_r7_s3_duplicate_fold_incomplete_but_unique_rows_kept():
    # DESIGN §6 row 3: a synthetic report with one row run twice —
    # 126 physical rows over 125 distinct keys.
    plan, _report, _meta = _real_slow_artifact()
    report = _clean_slow_report(plan, worth_search=True)  # a POSITIVE word
    victim = next(
        row
        for row in report["rows"]
        if row["candidate"] == "baseline:ROE"
        and row["cost_level"] == "1x"
        and row["fold_test_end"] == "2023-12-31"
    )
    report["rows"] = report["rows"] + [dict(victim)]
    victim_key = row_key_of(victim, plan)
    recon = reconcile_report(plan, report)
    status = classify_status(plan, recon, report)

    # RED: the duplicate is pinpointed — multiplicity 2, exactly one
    # duplicate_row finding naming the key — and it forces
    # measurement_incomplete even against a positive frozen verdict.
    assert recon.n_rows_observed == 126  # counted, never silently deduped
    assert recon.duplicates == ((victim_key, 2),)
    dup_issues = [i for i in recon.issues if i.code == "duplicate_row"]
    assert len(dup_issues) == 1
    assert dup_issues[0].key == victim_key
    assert "2 rows share one key" in dup_issues[0].detail
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_positive"
    assert status.primary_arms_measured is False
    assert status.verdict_valid is False
    assert status.primary_arms_evidence["m2_duplicate_keys"] == [
        (
            "period=None;cell=None;candidate=baseline:ROE;"
            "cost_level=1x;fold_test_end=2023-12-31",
            2,
        )
    ]

    # GREEN: no collateral damage — all 125 unique keys stay visible
    # to the audit, nothing goes missing or unexpected, and every arm
    # (including the duplicated one) still reconciles citable.
    assert recon.missing == ()
    assert recon.unexpected == ()
    assert len(recon.actual_keys) == 125
    assert victim_key in recon.actual_keys
    assert all(arm.status == "citable" for arm in status.arm_status)


# --- R7-S4: all combos failed, singles succeed — incomplete, not negative -


def test_p50_r7_s4_all_combos_failed_is_incomplete_singles_citable_not_economic_negative():
    # DESIGN §6 row 4: a synthetic replica of the real slow form —
    # 125 rows, 30 failed combo rows (15 + 15 on the full 3-level x
    # 5-fold grid) all carrying the one verbatim error string, the
    # five singles at zero failures, completed=true and a frozen
    # worth_search=false.
    plan, _report, _meta = _real_slow_artifact()
    train = _train_end_by_test_end(plan)
    rows = []
    for key in expected_keys(plan):
        if key.candidate in _COMBO_ARMS:
            rows.append(_failed_row(key, train[key.fold_test_end], _KEYERROR_REASON))
        else:
            rows.append(_success_row(key, train[key.fold_test_end]))
    assert len(rows) == 125
    assert len([row for row in rows if row["failed"]]) == 30
    report = _slow_report(
        plan,
        rows,
        worth_search=False,
        failed_folds_by_arm={arm: 5 for arm in _COMBO_ARMS},
    )

    recon = reconcile_report(plan, report)
    status = classify_status(plan, recon, report)

    # RED: the overall result is measurement_incomplete — NOT an
    # economic negative.  "No qualified arm" under 30 primary program
    # errors is undetermined, not refuted (R5/M3); the frozen
    # negative-looking adjudication is quarantined, preserved verbatim
    # but unreadable as economic evidence.
    assert status.report_generated is True  # the true stamp stops at layer 1
    assert status.primary_arms_measured is False
    assert status.verdict_valid is False
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_negative"
    assert status.adjudication_quarantined is True
    assert status.adjudication_as_recorded["worth_search"] is False
    assert status.primary_arms_evidence["m3_primary_failed_row_count"] == 30
    assert status.primary_arms_evidence["m3_primary_failure_reasons"] == (
        _KEYERROR_REASON,
    )
    # The failure layer is measured, not narrated: 30 = 15 + 15, all
    # primary, and the row set itself is intact (incompleteness comes
    # from the program errors alone).
    assert Counter(key.candidate for key, _r in recon.failed_rows) == {
        "combo:all3": 15,
        "combo:momentum_reversal": 15,
    }
    assert len(recon.primary_failed_rows) == 30
    assert recon.missing == ()
    assert recon.duplicates == ()
    assert recon.unexpected == ()

    # GREEN: partial results preserved — the five single-factor arms
    # (and both comparison arms) are citable with complete grids; the
    # two combos are program_error with the verbatim reason kept.
    by_candidate = _status_by_candidate(status)
    for candidate in _CITABLE_SINGLES:
        arm = by_candidate[candidate]
        assert arm.status == "citable", candidate
        assert arm.n_failed_rows == 0
        assert arm.n_present_rows == arm.n_expected_rows == 15
    for candidate in _COMBO_ARMS:
        arm = by_candidate[candidate]
        assert arm.status == "program_error", candidate
        assert arm.n_failed_rows == 15
        assert arm.failure_reasons == (_KEYERROR_REASON,)
        assert arm.is_primary is True
    assert by_candidate["control:cash"].status == "citable"
    assert by_candidate["benchmark:equal_weight"].status == "citable"


# --- ACC1: the REAL slow artifact, read-only, end-to-end ------------------


@pytest.mark.skipif(not _SLOW_REPORT.is_file(), reason="optional frozen historical run archive absent")
def test_p50_acc1_real_slow_artifact_pipeline_incomplete_and_errata_facts():
    # The frozen bytes first (F1): every measurement below is taken on
    # exactly this file, and the digest is the one the landed errata
    # target block pins — this test is the errata facts' regression
    # guard (any drift in the artifact OR in the classifier goes red).
    report_bytes = _SLOW_REPORT.read_bytes()
    assert len(report_bytes) == _SLOW_REPORT_BYTES
    digest = hashlib.sha256(report_bytes).hexdigest()
    assert digest == _SLOW_REPORT_SHA256
    errata = json.loads(_SLOW_ERRATA.read_text(encoding="utf-8"))
    assert errata["target"]["report_sha256"] == digest
    assert errata["classification"]["overall_label"] == "measurement_incomplete"

    plan, report, run_meta = _real_slow_artifact()
    recon = reconcile_report(plan, report, run_meta)
    status = classify_status(plan, recon, report)

    # The classification: measurement_incomplete — completed=true buys
    # only layer 1; the frozen worth_search=false is quarantined.
    assert status.experiment_id == "r01_slow_baseline"
    assert status.report_generated is True
    assert status.primary_arms_measured is False
    assert status.verdict_valid is False
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_negative"
    assert status.adjudication_quarantined is True
    assert status.adjudication_as_recorded["worth_search"] is False

    # The failure layer, measured from the rows authority (F4): 30 =
    # 15 + 15 on the full grid, every failed row primary, one verbatim
    # variant — the numbers the errata facts record, re-derived live.
    assert recon.n_rows_observed == 125
    assert len(recon.failed_rows) == 30
    assert Counter(key.candidate for key, _r in recon.failed_rows) == {
        "combo:all3": 15,
        "combo:momentum_reversal": 15,
    }
    assert {(key.cost_level, key.fold_test_end) for key, _r in recon.failed_rows} == {
        (level, fold)
        for level in ("0x", "1x", "2x")
        for fold in _REAL_FOLD_ENDS
    }
    assert {reason for _k, reason in recon.failed_rows} == {_KEYERROR_REASON}
    assert len(recon.primary_failed_rows) == 30
    assert recon.benchmark_failed_rows == ()
    # The set layer is intact and the run_meta count agrees — the
    # incompleteness is the program errors, nothing else.
    assert recon.missing == ()
    assert recon.duplicates == ()
    assert recon.unexpected == ()
    assert not [
        i for i in recon.issues if i.code == "meta_row_count_mismatch"
    ]

    # R6 citability, live: the five slow singles are citable (zero
    # failed rows, full 15-row grids) and their metrics are intact on
    # the rows layer; the two combos are program_error attributed to
    # KeyError: 'MOMENTUM_20' verbatim; cash and benchmark citable.
    by_candidate = _status_by_candidate(status)
    assert len(by_candidate) == 9
    for candidate in _CITABLE_SINGLES:
        arm = by_candidate[candidate]
        assert arm.status == "citable", candidate
        assert arm.n_failed_rows == 0
        assert arm.n_present_rows == arm.n_expected_rows == 15
        # citable means the measurements are there to cite: every row
        # of the arm carries the metric fields common to all 75
        # successful single rows (measured: sharpe/total_return/ic_mean
        # are universal; val_reward exists only on 1x reward rows).
        arm_rows = [
            row
            for row in report["rows"]
            if row["candidate"] == candidate and not row.get("failed")
        ]
        assert len(arm_rows) == 15
        assert all(
            "sharpe" in row and "total_return" in row and "ic_mean" in row
            for row in arm_rows
        )
    for candidate in _COMBO_ARMS:
        arm = by_candidate[candidate]
        assert arm.status == "program_error", candidate
        assert arm.failure_reasons == (_KEYERROR_REASON,)
    assert by_candidate["control:cash"].status == "citable"
    assert by_candidate["benchmark:equal_weight"].status == "citable"


# --- ACC2: the happy path — complete, error-free, all three layers true ---


def test_p50_acc2_happy_path_clean_synthetic_three_layers_true_overall_ok():
    # A synthetic complete error-free r01-shaped report: every expected
    # row present exactly once, no failures, consistent summary layer,
    # completed stamp, frozen adjudication.  The pipeline must deliver
    # all three layers true and read the verdict as an economic label.
    plan, _report, _meta = _real_slow_artifact()
    report = _clean_slow_report(plan, worth_search=False)

    recon = reconcile_report(plan, report)
    status = classify_status(plan, recon, report)

    # Nothing at all on the findings layer.
    assert recon.issues == ()
    assert recon.missing == () and recon.duplicates == ()
    assert recon.unexpected == ()
    assert recon.failed_rows == ()
    assert recon.n_rows_observed == 125
    assert len(recon.expected_keys) == len(recon.actual_keys) == 125

    # Three states all true; every M/V gate reads clean.
    assert status.report_generated is True
    assert status.primary_arms_measured is True
    assert status.verdict_valid is True
    evidence = status.primary_arms_evidence
    assert evidence["m1_primary_missing_keys"] == []
    assert evidence["m2_duplicate_keys"] == []
    assert evidence["m3_primary_failed_row_count"] == 0
    assert evidence["m4_benchmark_failed_row_count"] == 0
    assert evidence["m5_coverage_issue_codes"] == []
    assert status.verdict_valid_evidence["v0_missing_key_count"] == 0
    assert status.verdict_valid_evidence["v1_unexpected_key_count"] == 0

    # Overall ok: the adjudication is READ, not quarantined — and a
    # true negative stays a negative (M3: negatives are allowed, never
    # swallowed into incompleteness by over-conservatism).
    assert status.adjudication_quarantined is False
    assert status.label == "economic_negative"
    assert status.verdict_valid_evidence["verdict_word"] == "worth_search:false"
    assert all(arm.status == "citable" for arm in status.arm_status)

    # The same clean measurement under the positive outcome word reads
    # positive — the label follows the frozen table, both directions.
    positive_report = _clean_slow_report(plan, worth_search=True)
    positive_status = classify_status(
        plan, reconcile_report(plan, positive_report), positive_report
    )
    assert positive_status.report_generated is True
    assert positive_status.primary_arms_measured is True
    assert positive_status.verdict_valid is True
    assert positive_status.label == "economic_positive"
    assert positive_status.adjudication_quarantined is False
