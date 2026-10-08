"""P50 WP14 status-stage RED/GREEN tests: the R4 three-layer split and
the R5 terminal-label discipline.

Contract: ``docs/p50_experiment_integrity_draft.md`` §3.4/§5 (the WP14
status stage; the contract file lands with the WP14 contract-writing
stage per DESIGN §1 — every clause asserted here is pinned to a DESIGN
§5 rule and a measured fact F1-F4/F7/F23).

Invariants (status stage — the DESIGN §9 INV numbering belongs to the
full-suite file; this module uses ST1-ST11 to avoid collision, the
same convention test_p50_wp14_plan.py uses with PL1-PL6):

* ST1 — a TRUE negative survives: program-clean, arms complete,
  frozen verdict word mapping to ``economic_negative`` is judged
  negative, never swallowed into measurement_incomplete (M3
  anti-over-conservatism);
* ST2 — the REAL slow artifact (sha256-anchored): 30 primary program
  errors + a true completion stamp + a negative-looking frozen
  adjudication -> label measurement_incomplete, adjudication
  quarantined, five singles citable, two combos program_error (F4/F7);
* ST3 — the REQUIRED single-point counterexample: one primary arm's
  rows entirely absent -> measurement_incomplete, explicitly NOT
  economic_negative, every other arm's row-level outcome preserved;
* ST4 — non-primary absence (all benchmark rows deleted) passes the
  M layer but is blocked by V0 — no verdict-layer shortcut;
* ST5 — one duplicated row -> measurement_incomplete with no false
  missing-row report;
* ST6 — summary/meta layer disagreement ONLY -> the distinct
  ``summary_layer_inconsistent`` label (V3/V4 reachable, Rev2 #5);
* ST7 — a report that never declared itself generated (stamp false /
  schema-version key absent) -> ``not_generated``; the conjunction
  chain never upgrades lower layers;
* ST8 — a verdict word outside the frozen map is an IntegrityRefusal
  (fail-closed), never a silent inconclusive_mixed (§5.4 #4);
* ST9 — verdict_map is plan DATA: remapping changes the label,
  worth_search:true is economic_positive, and UNSTABLE /
  execution-blocked words never come out negative (F23);
* ST10 — an orphan or mismatched Reconciliation is refused before any
  judgment (§3.4 rule 3 — the empty-true path to an economic label is
  physically closed);
* ST11 — source discipline: the completion-stamp literal is read
  exactly once in status.py and no verdict-word literal appears in it
  (the lookup table is the only path);
* ST12 — end-to-end on the REAL slow artifact: the landed
  ``reconcile_report`` feeds ``classify_status`` and the full pipeline
  lands on measurement_incomplete with the five singles citable (the
  R4/R5 reading of F2/F4/F7 at the integrated scale).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from ashare_model.experiment_integrity.plan import (
    BENCHMARK_CANDIDATE,
    expected_keys,
    materialize_plan,
)
from ashare_model.experiment_integrity.reconcile import reconcile_report
from ashare_model.experiment_integrity.status import classify_status
from ashare_model.experiment_integrity.types import (
    ArmMeasurementStatus,
    IntegrityRefusal,
    Reconciliation,
    ReconcileIssue,
    RowKey,
)

#: The frozen r1 artifact root, resolved from the package (never from
#: the process cwd) — read-only access, the WP14 red line.
_ROOT = Path(__file__).resolve().parents[1]
_R1_SLOW_REPORT = _ROOT / "data" / "r1" / "r01_slow_baseline" / "r01_baseline_report.json"
_STATUS_SOURCE = _ROOT / "ashare_model" / "experiment_integrity" / "status.py"

#: F1: the audited slow-report digest (bytes measured 2049495).
_SLOW_REPORT_SHA256 = (
    "95fac73249c0ee93f86e0f385ce25ddf5ae7e7a92bc830db0706710bc05f550b"
)

#: F4: the measured failed-row breakdown on the slow artifact's rows
#: layer — the tests below re-derive it, never trust the narrative.
_EXPECTED_COMBO_FAILURES = {"combo:all3": 15, "combo:momentum_reversal": 15}
_KEYERROR_REASON = "KeyError: 'MOMENTUM_20'"

#: F7: the frozen bottleneck text, preserved verbatim when quarantined.
_SLOW_BOTTLENECK = (
    "no_qualified_arm: no cost-bearing arm shows durable net-of-cost "
    "space under the frozen rules (low_activity / thin_book / "
    "unreliable / single_year_dependence / non-positive net excess)"
)


def _row_key(row: dict) -> RowKey:
    """The §3.3 five-axis projection (identical to the one reconcile
    materializes; test-side copy, same convention as the plan file)."""

    return RowKey(
        period=row.get("period"),
        cell=row.get("cell"),
        candidate=row.get("candidate", ""),
        cost_level=row.get("cost_level", ""),
        fold_test_end=row.get("fold_test_end", ""),
    )


def _key_sort(key: RowKey) -> tuple[str, ...]:
    return (
        str(key.period),
        str(key.cell),
        key.candidate,
        key.cost_level,
        key.fold_test_end,
    )


def _synth_row(key: RowKey, *, failed: bool = False, reason: str | None = None):
    return {
        "period": key.period,
        "cell": key.cell,
        "candidate": key.candidate,
        "cost_level": key.cost_level,
        "fold_test_end": key.fold_test_end,
        "failed": failed,
        "failure_reason": reason,
    }


def _clean_rows(plan) -> list[dict]:
    """One successful row per expected key — the measurement-complete
    synthetic baseline (synthetic payloads only, no engine, C8)."""

    return [_synth_row(key) for key in expected_keys(plan)]


def _arm_statuses(plan, rows) -> tuple[ArmMeasurementStatus, ...]:
    """Arm-level measurement outcomes rebuilt from the rows layer
    (types.ArmMeasurementStatus semantics: citable = all present & ok;
    program_error = >=1 failed row; missing = none present; partial =
    some present)."""

    present = {_row_key(row) for row in rows}
    failed: dict[RowKey, str] = {}
    for row in rows:
        if row.get("failed"):
            failed[_row_key(row)] = str(row.get("failure_reason"))
    out = []
    for arm in plan.arms:
        keys = [
            RowKey(
                period=arm.period,
                cell=arm.cell,
                candidate=arm.candidate,
                cost_level=cost_level,
                fold_test_end=fold.test_end,
            )
            for cost_level in arm.cost_levels
            for fold in plan.folds
        ]
        n_present = sum(1 for key in keys if key in present)
        n_failed = sum(1 for key in keys if key in failed)
        reasons = tuple(sorted({failed[key] for key in keys if key in failed}))
        if n_failed:
            status = "program_error"
        elif n_present == len(keys):
            status = "citable"
        elif n_present == 0:
            status = "missing"
        else:
            status = "partial"
        out.append(
            ArmMeasurementStatus(
                candidate=arm.candidate,
                kind=arm.kind,
                is_primary=arm.is_primary,
                n_expected_rows=len(keys),
                n_present_rows=n_present,
                n_failed_rows=n_failed,
                failure_reasons=reasons,
                status=status,
            )
        )
    return tuple(out)


def _recon(plan, rows, *, issues=()) -> Reconciliation:
    """A test-side Reconciliation faithful to the types.py field
    semantics: expected = plan keys, actual = row-key multiset, failed
    rows from the authoritative row layer split by primary/benchmark,
    arm statuses rebuilt from rows (the real reconcile.py lands with
    DESIGN §3.3 and must agree with this construction)."""

    expected = frozenset(expected_keys(plan))
    counts = Counter(_row_key(row) for row in rows)
    actual = frozenset(counts)
    primary_ids = {(arm.candidate, arm.period) for arm in plan.primary_arms}
    failed_rows = tuple(
        sorted(
            (
                (_row_key(row), str(row.get("failure_reason")))
                for row in rows
                if row.get("failed")
            ),
            key=lambda pair: _key_sort(pair[0]),
        )
    )
    return Reconciliation(
        experiment_id=plan.experiment_id,
        expected_keys=expected,
        actual_keys=actual,
        n_rows_observed=len(rows),
        missing=tuple(sorted(expected - actual, key=_key_sort)),
        duplicates=tuple(
            (key, multiplicity)
            for key, multiplicity in sorted(counts.items(), key=lambda kv: _key_sort(kv[0]))
            if multiplicity > 1
        ),
        unexpected=tuple(sorted(actual - expected, key=_key_sort)),
        failed_rows=failed_rows,
        primary_failed_rows=tuple(
            (key, reason)
            for key, reason in failed_rows
            if (key.candidate, key.period) in primary_ids
        ),
        benchmark_failed_rows=tuple(
            (key, reason)
            for key, reason in failed_rows
            if key.candidate == BENCHMARK_CANDIDATE
        ),
        coverage={},
        arm_status=_arm_statuses(plan, rows),
        issues=tuple(issues),
    )


def _slow_report(rows, *, worth_search=False, with_schema=True, stamp=True):
    """A synthetic r01-shaped report around the given rows."""

    report = {
        "completed": stamp,
        "rows": rows,
        "adjudication": {
            "worth_search": worth_search,
            "bottleneck": _SLOW_BOTTLENECK,
        },
    }
    if with_schema:
        report["r01_report_schema_version"] = 3
    return report


def _status_by_candidate(status):
    return {arm.candidate: arm for arm in status.arm_status}


# --- ST1: a clean, complete, below-bar panel is a TRUE negative --------


def test_p50_st1_clean_slow_report_true_negative_is_economic_negative():
    # GREEN (M3 discipline, the anti-over-conservatism case): no failed
    # rows, every expected row present, completed stamp true, and the
    # frozen adjudication outcome maps to economic_negative — the
    # classifier must say negative, not hide behind incomplete.
    plan = materialize_plan("r01_slow_baseline")
    rows = _clean_rows(plan)
    report = _slow_report(rows, worth_search=False)
    status = classify_status(plan, _recon(plan, rows), report)

    assert status.report_generated is True
    assert status.primary_arms_measured is True
    assert status.verdict_valid is True
    assert status.label == "economic_negative"
    assert status.adjudication_quarantined is False
    assert status.verdict_valid_evidence["verdict_word"] == "worth_search:false"
    # The negative verdict is READ, not quarantined.
    assert status.adjudication_as_recorded["worth_search"] is False
    assert all(arm.status == "citable" for arm in status.arm_status)


# --- ST2: the REAL slow artifact — completed stamp must not upgrade ----


def test_p50_st2_real_slow_artifact_incomplete_not_negative():
    # RED (F7 + F4, measured): completed=true, worth_search=false, and
    # 30 primary program-error rows.  The overall result must be
    # measurement_incomplete; the frozen negative-looking adjudication
    # is quarantined, never read as the economic result.
    assert _R1_SLOW_REPORT.is_file(), "the frozen slow artifact must exist"
    digest = hashlib.sha256(_R1_SLOW_REPORT.read_bytes()).hexdigest()
    assert digest == _SLOW_REPORT_SHA256  # F1: the audited bytes

    plan = materialize_plan("r01_slow_baseline")
    report = json.loads(_R1_SLOW_REPORT.read_text(encoding="utf-8"))
    rows = report["rows"]
    # F4 re-derived from the rows layer (never from the narrative).
    combo_failures = Counter(
        row["candidate"] for row in rows if row.get("failed")
    )
    assert dict(combo_failures) == _EXPECTED_COMBO_FAILURES

    status = classify_status(plan, _recon(plan, rows), report)
    assert status.report_generated is True  # the stamp is true — and stops here
    assert status.primary_arms_measured is False
    assert status.primary_arms_evidence["m3_primary_failed_row_count"] == 30
    assert status.verdict_valid is False
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_negative"  # the explicit R5 refusal
    assert status.adjudication_quarantined is True
    # The frozen adjudication output survives verbatim, quarantined.
    assert status.adjudication_as_recorded["worth_search"] is False
    assert status.adjudication_as_recorded["bottleneck"] == _SLOW_BOTTLENECK
    # Row-set integrity itself is intact: the incompleteness comes from
    # the 30 program errors alone, not from missing/duplicate drift.
    assert status.verdict_valid_evidence["v0_missing_key_count"] == 0
    assert status.verdict_valid_evidence["v1_unexpected_key_count"] == 0

    # GREEN (R6): partial results preserved — five singles citable with
    # complete annual tables, two combos program_error with the verbatim
    # failure reason, benchmark and cash citable.
    by_candidate = _status_by_candidate(status)
    singles = [c for c in by_candidate if c.startswith("baseline:")]
    assert sorted(singles) == [
        "baseline:DIVIDEND_YIELD",
        "baseline:MARKET_CAP",
        "baseline:PE_TTM",
        "baseline:REVENUE_YOY",
        "baseline:ROE",
    ]
    for candidate in singles:
        arm = by_candidate[candidate]
        assert arm.status == "citable"
        assert arm.n_failed_rows == 0
        assert arm.n_present_rows == arm.n_expected_rows == 15
    for candidate in ("combo:all3", "combo:momentum_reversal"):
        arm = by_candidate[candidate]
        assert arm.status == "program_error"
        assert arm.n_failed_rows == 15
        assert arm.failure_reasons == (_KEYERROR_REASON,)
    assert by_candidate["benchmark:equal_weight"].status == "citable"
    assert by_candidate["control:cash"].status == "citable"


# --- ST3: REQUIRED single-point counterexample — one primary arm gone --


def test_p50_st3_missing_one_primary_arm_single_point_incomplete():
    # RED (the task's required counterexample): every combo:all3 row
    # deleted (110 rows left), no failed rows anywhere, a negative-
    # looking frozen adjudication.  The overall label must be
    # measurement_incomplete — NOT economic_negative — because a
    # missing primary arm makes "no qualified arm" undetermined, not
    # refuted (§5.5).
    plan = materialize_plan("r01_slow_baseline")
    rows = [
        row
        for row in _clean_rows(plan)
        if not (
            row["candidate"] == "combo:all3"
            and row["cost_level"] in ("0x", "1x", "2x")
        )
    ]
    assert len(rows) == 110  # 125 - 15 deleted rows
    report = _slow_report(rows, worth_search=False)
    recon = _recon(plan, rows)
    assert recon.primary_failed_rows == ()  # incompleteness is absence, not error

    status = classify_status(plan, recon, report)
    assert status.report_generated is True
    assert status.primary_arms_measured is False
    assert status.verdict_valid is False
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_negative"
    assert status.adjudication_quarantined is True
    # The missing list names exactly the deleted arm's 15 keys.
    assert len(recon.missing) == 15
    assert {key.candidate for key in recon.missing} == {"combo:all3"}
    assert len(status.primary_arms_evidence["m1_primary_missing_keys"]) == 15

    # GREEN: every OTHER arm's row-level outcome is still delivered.
    by_candidate = _status_by_candidate(status)
    assert by_candidate["combo:all3"].status == "missing"
    assert by_candidate["combo:all3"].n_present_rows == 0
    assert by_candidate["combo:all3"].n_expected_rows == 15
    assert by_candidate["combo:momentum_reversal"].status == "citable"
    assert by_candidate["baseline:ROE"].status == "citable"
    assert by_candidate["benchmark:equal_weight"].status == "citable"
    assert by_candidate["control:cash"].status == "citable"


# --- ST4: non-primary absence passes M but is blocked by V0 ------------


def test_p50_st4_missing_benchmark_rows_blocked_by_v0():
    # RED (Rev2 #9 backstop): deleting all five benchmark rows creates
    # no failed row and no primary-arm absence — the M layer passes —
    # yet the verdict layer must refuse via V0 (a comparison arm's
    # absence poisons every excess-return reading, not a verdict).
    plan = materialize_plan("r01_slow_baseline")
    rows = [
        row
        for row in _clean_rows(plan)
        if row["candidate"] != BENCHMARK_CANDIDATE
    ]
    assert len(rows) == 120
    report = _slow_report(rows, worth_search=False)
    recon = _recon(plan, rows)

    status = classify_status(plan, recon, report)
    assert status.report_generated is True
    assert status.primary_arms_measured is True  # M genuinely passes
    assert status.verdict_valid is False  # V0 refuses
    assert status.label == "measurement_incomplete"  # §5.4 #2 includes V0
    assert status.label != "economic_negative"
    assert status.verdict_valid_evidence["v0_missing_key_count"] == 5
    assert {key.candidate for key in recon.missing} == {BENCHMARK_CANDIDATE}
    # GREEN: all measured arms stay citable and reportable.
    assert all(
        arm.status == "citable" for arm in status.arm_status
        if arm.candidate != BENCHMARK_CANDIDATE
    )
    assert _status_by_candidate(status)[BENCHMARK_CANDIDATE].status == "missing"


# --- ST5: one duplicated row — no false missing report ------------------


def test_p50_st5_duplicate_row_incomplete_no_missing_false_positive():
    # RED (M2): a duplicated fold row is a measurement defect that
    # forces measurement_incomplete, and the duplicate is pinpointed.
    plan = materialize_plan("r01_slow_baseline")
    rows = _clean_rows(plan)
    victim = next(
        row for row in rows if row["candidate"] == "baseline:ROE"
        and row["cost_level"] == "1x" and row["fold_test_end"] == "2023-12-31"
    )
    rows = rows + [dict(victim)]  # 126 rows, one key of multiplicity 2
    report = _slow_report(rows, worth_search=True)  # even a positive word is shadowed
    recon = _recon(plan, rows)

    status = classify_status(plan, recon, report)
    assert status.report_generated is True
    assert status.primary_arms_measured is False
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_positive"  # duplicates shadow positives too
    assert recon.duplicates == (( _row_key(victim), 2 ),)
    assert status.primary_arms_evidence["m2_duplicate_keys"] == [
        (
            f"period=None;cell=None;candidate=baseline:ROE;"
            f"cost_level=1x;fold_test_end=2023-12-31",
            2,
        )
    ]
    # GREEN: no false missing-row report and every unique key stays
    # visible to the audit.
    assert recon.missing == ()
    assert len(recon.actual_keys) == 125
    assert all(arm.status == "citable" for arm in status.arm_status)


# --- ST6: summary/meta disagreement ONLY -> its own label --------------


@pytest.mark.parametrize(
    "issue",
    [
        ReconcileIssue(
            code="summary_layer_mismatch",
            key=None,
            detail="synthetic: arm_summaries.failed_folds tampered",
        ),
        ReconcileIssue(
            code="meta_row_count_mismatch",
            key=None,
            detail="synthetic: run_meta.row_count disagrees with rows",
        ),
    ],
    ids=["v3_summary_layer", "v4_meta_row_count"],
)
def test_p50_st6_summary_layer_only_inconsistent_label(issue):
    # RED (Rev2 #5 — V3/V4 must be reachable): rows complete, no
    # failed rows, only the report's own summary/meta layer disagrees.
    # The label is summary_layer_inconsistent — measurement exists,
    # the summarized evidence earns no verdict — never an economic
    # label and never a silent pass.
    plan = materialize_plan("r01_slow_baseline")
    rows = _clean_rows(plan)
    report = _slow_report(rows, worth_search=True)
    recon = _recon(plan, rows, issues=[issue])

    status = classify_status(plan, recon, report)
    assert status.report_generated is True
    assert status.primary_arms_measured is True
    assert status.verdict_valid is False
    assert status.label == "summary_layer_inconsistent"
    assert status.adjudication_quarantined is True
    assert status.label not in ("economic_positive", "economic_negative")
    # GREEN: the measurement itself is preserved and reportable.
    assert all(arm.status == "citable" for arm in status.arm_status)
    assert status.verdict_valid_evidence["v0_missing_key_count"] == 0


# --- ST7: a report that never declared itself generated -----------------


def test_p50_st7_not_generated_when_stamp_or_schema_fails():
    plan = materialize_plan("r01_slow_baseline")
    rows = _clean_rows(plan)
    recon = _recon(plan, rows)

    # A false completion stamp: not_generated, and the clean
    # reconciliation does NOT upgrade any higher layer.
    unfinished = _slow_report(rows, worth_search=False, stamp=False)
    status = classify_status(plan, recon, unfinished)
    assert status.report_generated is False
    assert status.report_generated_evidence["g3_completion_stamp_is_true"] is False
    assert status.primary_arms_measured is False  # conjunction chain
    assert status.verdict_valid is False
    assert status.label == "not_generated"
    assert status.adjudication_quarantined is True

    # An absent schema-version key: the report was written by code the
    # auditor has no contract for -> not_generated.
    unknown_schema = _slow_report(rows, worth_search=False, with_schema=False)
    status = classify_status(plan, recon, unknown_schema)
    assert status.report_generated is False
    assert status.report_generated_evidence["g2_schema_version_observed"] is None
    assert status.label == "not_generated"
    # GREEN: the rows were still reconciled and the frozen adjudication
    # block is preserved verbatim for the record.
    assert status.adjudication_as_recorded["worth_search"] is False
    assert len(status.arm_status) == 9


# --- ST8: unmapped verdict word — fail-closed refusal --------------------


def test_p50_st8_unmapped_verdict_word_refuses_fail_closed():
    # RED (§5.4 #4): a verdict word outside the frozen map is word-table
    # drift — an IntegrityRefusal naming the word, the map keys and the
    # experiment.  It must NOT silently become inconclusive_mixed (that
    # would disguise drift as a legitimate mixed outcome, P45 lesson).
    plan = materialize_plan("r01_slow_baseline")
    rows = _clean_rows(plan)
    report = _slow_report(rows, worth_search=False)
    drifted = replace(
        plan,
        verdict_map={"worth_search:true": "economic_positive"},  # false-key dropped
    )
    with pytest.raises(IntegrityRefusal, match=r"word-table drift") as excinfo:
        classify_status(drifted, _recon(plan, rows), report)
    message = str(excinfo.value)
    assert "word='worth_search:false'" in message
    assert "map_keys=['worth_search:true']" in message
    assert "r01_slow_baseline" in message

    # A map value that is not an economic label is corrupt plan data —
    # also refused, never emitted as a label.
    corrupt = replace(plan, verdict_map={"worth_search:false": "ok"})
    with pytest.raises(IntegrityRefusal, match=r"non-economic label"):
        classify_status(corrupt, _recon(plan, rows), report)


# --- ST9: verdict_map is plan DATA; positives and UNSTABLE words --------


def test_p50_st9_verdict_map_is_data_positive_and_unstable_words():
    plan = materialize_plan("r01_slow_baseline")
    rows = _clean_rows(plan)
    recon = _recon(plan, rows)

    # GREEN (M3 "all bars met" case): the positive outcome word.
    positive = classify_status(
        plan, recon, _slow_report(rows, worth_search=True)
    )
    assert positive.label == "economic_positive"
    assert positive.verdict_valid_evidence["verdict_word"] == "worth_search:true"

    # The label follows the TABLE, not any logic in status.py: remap
    # the same word to inconclusive_mixed and the label follows.
    remapped = replace(
        plan, verdict_map=dict(plan.verdict_map, **{"worth_search:false": "inconclusive_mixed"})
    )
    assert classify_status(
        plan, recon, _slow_report(rows, worth_search=False)
    ).label == "economic_negative"
    assert classify_status(
        remapped, recon, _slow_report(rows, worth_search=False)
    ).label == "inconclusive_mixed"

    # F23 at the classifier level: UNSTABLE / execution words are never
    # negative, and the same word may map differently per experiment —
    # r02_matrix panel_no_reliable_info is a signal negative while the
    # r03 word of the same spelling is a mixed panel.
    matrix_plan = materialize_plan("r02_matrix")
    matrix_rows = _clean_rows(matrix_plan)
    matrix_recon = _recon(matrix_plan, matrix_rows)

    def _matrix_report(verdict: str) -> dict:
        return {
            "r02_report_schema_version": 1,
            "completed": True,
            "rows": matrix_rows,
            "grid_adjudication": {"verdict": verdict},
        }

    assert classify_status(
        matrix_plan, matrix_recon, _matrix_report("isolated_single_winner")
    ).label == "inconclusive_mixed"  # UNSTABLE: never negative
    assert classify_status(
        matrix_plan, matrix_recon, _matrix_report("panel_no_reliable_info")
    ).label == "economic_negative"  # clean signal negative (M3)

    r03_plan = materialize_plan("r03_combos")
    r03_rows = _clean_rows(r03_plan)
    r03_report = {
        "r03_report_schema_version": 1,
        "completed": True,
        "rows": r03_rows,
        "adjudication": {"verdict": "panel_no_reliable_info"},
    }
    assert classify_status(
        r03_plan, _recon(r03_plan, r03_rows), r03_report
    ).label == "inconclusive_mixed"  # same word, different economic object


# --- ST10: orphan / mismatched reconciliation refused -------------------


def _empty_recon(experiment_id: str) -> Reconciliation:
    return Reconciliation(
        experiment_id=experiment_id,
        expected_keys=frozenset(),
        actual_keys=frozenset(),
        n_rows_observed=0,
        missing=(),
        duplicates=(),
        unexpected=(),
        failed_rows=(),
        primary_failed_rows=(),
        benchmark_failed_rows=(),
        coverage={},
        arm_status=(),
        issues=(),
    )


def test_p50_st10_orphan_or_mismatched_reconciliation_refused():
    # RED (§3.4 rule 3): a hand-built all-clean Reconciliation paired
    # with a real populated report is refused BEFORE any judgment —
    # the empty-true shortcut to an economic label is physically closed.
    plan = materialize_plan("r01_slow_baseline")
    rows = _clean_rows(plan)
    report = _slow_report(rows, worth_search=False)

    with pytest.raises(IntegrityRefusal, match=r"row count") as excinfo:
        classify_status(plan, _empty_recon(plan.experiment_id), report)
    assert "recon.n_rows_observed=0" in str(excinfo.value)
    assert "len(report rows)=125" in str(excinfo.value)

    # Wrong experiment id: refused first, with both ids in the message.
    with pytest.raises(IntegrityRefusal, match=r"experiment_id") as excinfo:
        classify_status(plan, _empty_recon("r02_matrix"), report)
    assert "r02_matrix" in str(excinfo.value)
    assert "r01_slow_baseline" in str(excinfo.value)

    # A tampered row count on an otherwise honest reconciliation.
    honest = _recon(plan, rows)
    with pytest.raises(IntegrityRefusal, match=r"row count"):
        classify_status(plan, replace(honest, n_rows_observed=126), report)

    # A tampered key set: same count, different keys — still refused.
    with pytest.raises(IntegrityRefusal, match=r"actual key set"):
        classify_status(plan, replace(honest, actual_keys=frozenset()), report)

    # A report with no auditable rows list is refused at the boundary
    # (locator=rows), never classified from an absent measurement.
    with pytest.raises(IntegrityRefusal, match=r"locator=rows"):
        classify_status(
            plan,
            _empty_recon(plan.experiment_id),
            {"completed": True, "adjudication": {"worth_search": False}},
        )


# --- ST11: source discipline — one stamp read, no verdict literals ------


def test_p50_st11_source_discipline_single_stamp_read_no_verdict_literals():
    # INV4 flavor at the source level: the completion-stamp literal is
    # read exactly once in status.py (its only legal consumer is the
    # report_generated layer), and no frozen verdict word appears in
    # the module — the lookup table is the only path to an economic
    # label (§5.4 #4 / §8 n2).
    source = _STATUS_SOURCE.read_text(encoding="utf-8")
    assert source.count('"completed"') == 1
    for word in (
        "worth_search",
        "stable_region_found",
        "isolated_single_winner",
        "multiple_disjoint_viable",
        "gross_info_execution_blocked",
        "stable_period_region",
        "isolated_period_winner",
        "gross_info_execution_blocked_period_panel",
        "period_panel_no_reliable_info",
        "panel_no_reliable_info",
        "fitted_increment_region",
        "equal_weight_stands",
        "combo_no_net_space",
    ):
        assert word not in source, f"verdict-word literal {word!r} leaked into status.py"


# --- ST12: end-to-end — real plan, real reconcile, real report ----------


def test_p50_st12_end_to_end_real_reconcile_feeds_real_classification():
    # The integrated R3->R4/R5 pipeline on the frozen slow artifact:
    # the landed reconcile_report (no synthetic construction) feeds
    # classify_status.  The binding check accepts it (same rows, same
    # keys), and the verdict is the WP14 headline: measurement_incomplete
    # — the completed stamp and the frozen negative-looking adjudication
    # never survive contact with 30 primary program errors.
    plan = materialize_plan("r01_slow_baseline")
    report = json.loads(_R1_SLOW_REPORT.read_text(encoding="utf-8"))
    recon = reconcile_report(plan, report)

    status = classify_status(plan, recon, report)
    assert status.report_generated is True
    assert status.primary_arms_measured is False
    assert status.verdict_valid is False
    assert status.label == "measurement_incomplete"
    assert status.label != "economic_negative"
    assert status.adjudication_quarantined is True
    assert status.adjudication_as_recorded["worth_search"] is False
    # R6 made concrete: the five citable slow singles, delivered by the
    # full pipeline (arm layer rebuilt from rows by the real reconcile).
    by_candidate = _status_by_candidate(status)
    for candidate in (
        "baseline:ROE",
        "baseline:PE_TTM",
        "baseline:REVENUE_YOY",
        "baseline:DIVIDEND_YIELD",
        "baseline:MARKET_CAP",
    ):
        assert by_candidate[candidate].status == "citable"
        assert by_candidate[candidate].is_primary is True
    for candidate in ("combo:all3", "combo:momentum_reversal"):
        assert by_candidate[candidate].status == "program_error"
        assert by_candidate[candidate].failure_reasons == (_KEYERROR_REASON,)
