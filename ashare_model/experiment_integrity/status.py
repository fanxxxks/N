"""WP14 (p50) R4/R5: the three-layer status classification of a report.

Contract: ``docs/p50_experiment_integrity_draft.md`` §3.4/§5.  Pure
function of (plan, reconciliation, report) — no IO, no re-run, no
engine import.  The three layers are a CONJUNCTION CHAIN, never a
default-true shortcut (§5 anti-shortcut):

1. ``report_generated``  — the report self-describes as complete: a
   known report-schema-version key at its frozen value plus a true
   completion stamp.  The stamp is read EXACTLY ONCE in this module
   and feeds ONLY this layer (INV4: a completed stamp never upgrades
   measurement or verdict validity — the slow artifact carries a true
   completion stamp, a frozen negative-looking adjudication output
   and 30 failed combo rows, F4/F7).
2. ``primary_arms_measured`` — the reconciliation says every primary
   preregistered arm has all its rows, no duplicates anywhere, no
   program-failed primary/benchmark rows, no coverage defects.
3. ``verdict_valid`` — additionally no missing/unexpected rows at all
   (V0 covers non-primary and comparison arms), no inactive-axis
   values, a summary layer consistent with the rows and a consistent
   run-meta row count.

The terminal label follows the fixed order of §5.4:
``measurement_incomplete`` SHADOWS every economic label (M3
discipline): primary program errors / missing rows / duplicate rows
must never be read as an economic negative.  A TRUE negative —
program-clean, arms complete, frozen verdict word mapping to
``economic_negative`` — is still judged negative (never swallowed).
The economic step only ever LOOKS UP ``plan.verdict_map`` (frozen
plan data, F23); no verdict-word literal is compared anywhere in
this module, and a word absent from the map is a fail-closed
``IntegrityRefusal`` — never a silent ``inconclusive_mixed``.

What this module deliberately does not do: it does not re-run, patch
or re-adjudicate any historical artifact; it does not rebuild the
reconciliation (a ``Reconciliation`` that does not bind to the passed
report — wrong experiment id, wrong row count, divergent key set —
is refused before any judgment, §3.4 rule 3, so no orphan
all-clean recon can smuggle a verdict); it does not read the
completion stamp more than once or let it stand in for measurement
or adjudication evidence.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .reconcile import row_key_of
from .types import (
    STATUS_LABELS,
    IntegrityRefusal,
    Reconciliation,
    RowKey,
    RunPlan,
    StatusReport,
)

#: p50 §5.1 G2 (F11/C2): each report's self-declared schema-version
#: key and the one known value.  A report whose key is absent or at a
#: different value was written by code the auditor has no contract for
#: -> report_generated stays False (label ``not_generated``).
_REPORT_SCHEMA_VERSIONS: dict[str, tuple[str, int]] = {
    "r01_slow_baseline": ("r01_report_schema_version", 3),
    "r02_matrix": ("r02_report_schema_version", 1),
    "r02_periods": ("r02_periods_report_schema_version", 1),
    "r03_combos": ("r03_report_schema_version", 1),
    # WP18 (p56): one key per experiment, matching reconcile's
    # _REPORT_SCHEMA_KEYS (a wp18_medium report never classifies against a
    # wp18_slow plan).
    "wp18_medium": ("wp18_medium_report_schema_version", 1),
    "wp18_slow": ("wp18_slow_report_schema_version", 1),
}

#: p50 §5.2 M5: coverage-integrity issue codes (date coverage, returns
#: length, fold-window overlap) — measurement defects, not economics.
_COVERAGE_ISSUE_CODES = frozenset(
    ("date_coverage_mismatch", "returns_length_mismatch", "fold_window_overlap")
)

#: p50 §5.3: the single-code V-layer conditions (V2/V3/V4).
_UNEXPECTED_AXIS_CODE = "unexpected_axis_value"
_SUMMARY_LAYER_CODE = "summary_layer_mismatch"
_META_COUNT_CODE = "meta_row_count_mismatch"

#: p50 §5.4 #4: the only labels the economic step may produce — the
#: three non-measurement members of STATUS_LABELS.
_ECONOMIC_LABELS = frozenset(
    ("economic_positive", "economic_negative", "inconclusive_mixed")
)

#: Sentinel for a path walk that never reached a leaf.
_MISSING = object()


def _key_sort(key: RowKey) -> tuple[str, ...]:
    """A total-order projection of a RowKey for deterministic evidence
    lists (RowKey itself mixes None and str, which do not compare)."""

    return (
        str(key.period),
        str(key.cell),
        key.candidate,
        key.cost_level,
        key.fold_test_end,
    )


def _key_label(key: RowKey) -> str:
    return (
        f"period={key.period};cell={key.cell};candidate={key.candidate};"
        f"cost_level={key.cost_level};fold_test_end={key.fold_test_end}"
    )


def _walk(report: Mapping[str, Any], dotted_path: str) -> tuple[Any, str, Any]:
    """Walk a dotted path into the report.  Returns
    ``(container, leaf_name, value)``; on any missing step returns
    ``(None, leaf_name, _MISSING)``.  Never raises for absent data —
    the caller decides refusal semantics."""

    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        return None, "", _MISSING
    container: Any = report
    for part in parts[:-1]:
        if not isinstance(container, Mapping) or part not in container:
            return None, parts[-1], _MISSING
        container = container[part]
    leaf = parts[-1]
    if not isinstance(container, Mapping) or leaf not in container:
        return None, leaf, _MISSING
    return container, leaf, container[leaf]


def _bind_reconciliation(
    plan: RunPlan, recon: Reconciliation, report: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """§3.4 rule 3 — bind the reconciliation to THIS report before any
    judgment: same experiment, same row count, same actual key set
    rebuilt from the report rows.  Any divergence is an
    ``IntegrityRefusal`` carrying both measured values; a hand-built
    all-empty reconciliation against a populated report can never pass
    (empty keys vs non-empty rows), which physically closes the
    empty-true shortcut to an economic label."""

    if recon.experiment_id != plan.experiment_id:
        raise IntegrityRefusal(
            "reconciliation experiment_id does not match the plan\n"
            f"(recon.experiment_id={recon.experiment_id!r}, "
            f"plan.experiment_id={plan.experiment_id!r})"
        )
    if not isinstance(report, Mapping):
        raise IntegrityRefusal(
            "report is not a mapping and cannot be audited\n"
            f"(experiment={plan.experiment_id}, type={type(report).__name__})"
        )
    rows = report.get("rows")
    if not isinstance(rows, list) or not all(
        isinstance(row, Mapping) for row in rows
    ):
        raise IntegrityRefusal(
            "report has no auditable rows list\n"
            f"(experiment={plan.experiment_id}, locator=rows, "
            f"observed_type={type(rows).__name__})"
        )
    if recon.n_rows_observed != len(rows):
        raise IntegrityRefusal(
            "reconciliation row count does not match the report\n"
            f"(recon.n_rows_observed={recon.n_rows_observed}, "
            f"len(report rows)={len(rows)}, "
            f"experiment={plan.experiment_id})"
        )
    rebuilt = frozenset(row_key_of(row, plan) for row in rows)
    if recon.actual_keys != rebuilt:
        raise IntegrityRefusal(
            "reconciliation actual key set does not match the report rows\n"
            f"(recon keys={len(recon.actual_keys)}, "
            f"rebuilt from rows={len(rebuilt)}, "
            f"experiment={plan.experiment_id})"
        )
    return rows


def classify_status(
    plan: RunPlan,
    recon: Reconciliation,
    report: Mapping[str, Any],
) -> StatusReport:
    """Classify one report into the three R4 layers and the R5 terminal
    label.  Evaluation order is fixed (§3.4 rule 2):
    report_generated -> primary_arms_measured -> verdict_valid ->
    label; the label reads ``plan.verdict_map`` only when
    verdict_valid is True, and ``measurement_incomplete`` shadows every
    economic label.  Raises ``IntegrityRefusal`` when the inputs cannot
    be audited at all (unbound reconciliation, no known schema-version
    declaration for the experiment, a verdict word outside the frozen
    map) — findings, by contrast, are data and never raise."""

    rows = _bind_reconciliation(plan, recon, report)

    # --- layer 1: report_generated (§5.1 G1-G3) ------------------------
    if plan.experiment_id not in _REPORT_SCHEMA_VERSIONS:
        raise IntegrityRefusal(
            "no known report schema-version declaration for this "
            "experiment\n"
            f"(experiment={plan.experiment_id!r}, "
            f"known={sorted(_REPORT_SCHEMA_VERSIONS)})"
        )
    schema_key, schema_expected = _REPORT_SCHEMA_VERSIONS[plan.experiment_id]
    schema_observed = report.get(schema_key)
    g1_nonempty = bool(report)
    g2_schema_known = (
        schema_key in report
        and isinstance(schema_observed, int)
        and not isinstance(schema_observed, bool)
        and schema_observed == schema_expected
    )
    # G3: the completion stamp — its ONLY read in this module, feeding
    # ONLY the report_generated layer (INV4).  A report with no stamp
    # key has not declared itself generated -> g3 False, not a crash.
    g3_stamp_true = report.get("completed") is True
    report_generated = g1_nonempty and g2_schema_known and g3_stamp_true

    # --- layer 2: primary_arms_measured (§5.2 M1-M5, recon only) -------
    primary_ids = frozenset(
        (arm.candidate, arm.period) for arm in plan.primary_arms
    )
    primary_missing = tuple(
        key
        for key in sorted(recon.missing, key=_key_sort)
        if (key.candidate, key.period) in primary_ids
    )
    m1 = not primary_missing
    m2 = recon.duplicates == ()
    m3 = recon.primary_failed_rows == ()
    m4 = recon.benchmark_failed_rows == ()
    coverage_codes = tuple(
        sorted(
            {
                issue.code
                for issue in recon.issues
                if issue.code in _COVERAGE_ISSUE_CODES
            }
        )
    )
    m5 = not coverage_codes
    primary_arms_measured = report_generated and m1 and m2 and m3 and m4 and m5

    # --- layer 3: verdict_valid (§5.3 V0-V4, recon only) ---------------
    missing_keys = tuple(sorted(recon.missing, key=_key_sort))
    unexpected_keys = tuple(sorted(recon.unexpected, key=_key_sort))
    v0 = recon.missing == ()
    v1 = recon.unexpected == ()
    v2 = not any(issue.code == _UNEXPECTED_AXIS_CODE for issue in recon.issues)
    v3 = not any(issue.code == _SUMMARY_LAYER_CODE for issue in recon.issues)
    v4 = not any(issue.code == _META_COUNT_CODE for issue in recon.issues)
    verdict_valid = primary_arms_measured and v0 and v1 and v2 and v3 and v4

    # --- terminal label (§5.4 fixed order, first match wins) -----------
    verdict_evidence: dict[str, Any] = {}
    verdict_path = str(plan.sources.get("adjudication_path", ""))
    if not report_generated:
        label = "not_generated"
    elif not (m1 and m2 and m3 and m4 and m5 and v0 and v1 and v2):
        # measurement defects shadow every economic reading (M3), even
        # with a completed stamp and a frozen negative-looking verdict.
        label = "measurement_incomplete"
    elif not (v3 and v4):
        # rows intact and complete; only the report's own summary/meta
        # layer disagrees with them — the measurement exists, the
        # report's summarized evidence does not earn a verdict.
        label = "summary_layer_inconsistent"
    else:
        container, leaf, value = _walk(report, verdict_path)
        if value is _MISSING or container is None:
            raise IntegrityRefusal(
                "adjudication outcome word not found at the declared path\n"
                f"(path={verdict_path!r}, experiment={plan.experiment_id})"
            )
        if isinstance(value, bool):
            # bool outcomes normalize to a literal "<leaf>:<true|false>"
            # key (the r01 frozen true/false outcome pair) — generic,
            # never a hardcoded verdict word.
            verdict_word = f"{leaf}:{str(value).lower()}"
        elif isinstance(value, str) and value:
            verdict_word = value
        else:
            raise IntegrityRefusal(
                "adjudication outcome word has an unsupported type\n"
                f"(path={verdict_path!r}, "
                f"experiment={plan.experiment_id}, type={type(value).__name__})"
            )
        if verdict_word not in plan.verdict_map:
            raise IntegrityRefusal(
                "verdict word is not in the frozen verdict_map — word-table "
                "drift is refused, never silently defaulted\n"
                f"(word={verdict_word!r}, map_keys={sorted(plan.verdict_map)}, "
                f"experiment={plan.experiment_id})"
            )
        label = plan.verdict_map[verdict_word]
        if label not in _ECONOMIC_LABELS:
            raise IntegrityRefusal(
                "verdict_map maps a word to a non-economic label — corrupt "
                "plan data\n"
                f"(word={verdict_word!r}, mapped={label!r}, "
                f"experiment={plan.experiment_id})"
            )
        verdict_evidence = {
            "verdict_path": verdict_path,
            "verdict_word": verdict_word,
            "mapped_label": label,
        }

    # --- the frozen adjudication output, preserved verbatim ------------
    container, _leaf, _value = _walk(report, verdict_path)
    adjudication_as_recorded = (
        copy.deepcopy(container) if isinstance(container, Mapping) else None
    )

    if label not in STATUS_LABELS:  # defensive: unreachable via verdict_map
        raise IntegrityRefusal(
            f"classification produced an unknown label {label!r}\n"
            f"(experiment={plan.experiment_id})"
        )

    return StatusReport(
        experiment_id=plan.experiment_id,
        report_generated=report_generated,
        report_generated_evidence={
            "g1_report_nonempty": g1_nonempty,
            "g2_schema_key": schema_key,
            "g2_schema_version_expected": schema_expected,
            "g2_schema_version_observed": schema_observed,
            "g3_completion_stamp_is_true": g3_stamp_true,
        },
        primary_arms_measured=primary_arms_measured,
        primary_arms_evidence={
            "gates_on_report_generated": report_generated,
            "m1_primary_missing_keys": [_key_label(k) for k in primary_missing],
            "m2_duplicate_keys": [
                (_key_label(key), multiplicity)
                for key, multiplicity in recon.duplicates
            ],
            "m3_primary_failed_row_count": len(recon.primary_failed_rows),
            "m3_primary_failure_reasons": tuple(
                sorted({reason for _key, reason in recon.primary_failed_rows})
            ),
            "m4_benchmark_failed_row_count": len(recon.benchmark_failed_rows),
            "m5_coverage_issue_codes": list(coverage_codes),
        },
        verdict_valid=verdict_valid,
        verdict_valid_evidence={
            "gates_on_primary_arms_measured": primary_arms_measured,
            "v0_missing_key_count": len(missing_keys),
            "v0_missing_keys": [_key_label(k) for k in missing_keys],
            "v1_unexpected_key_count": len(unexpected_keys),
            "v1_unexpected_keys": [_key_label(k) for k in unexpected_keys],
            "v2_unexpected_axis_value_issues": sum(
                1 for i in recon.issues if i.code == _UNEXPECTED_AXIS_CODE
            ),
            "v3_summary_layer_mismatch_issues": sum(
                1 for i in recon.issues if i.code == _SUMMARY_LAYER_CODE
            ),
            "v4_meta_row_count_mismatch_issues": sum(
                1 for i in recon.issues if i.code == _META_COUNT_CODE
            ),
            **verdict_evidence,
        },
        label=label,
        arm_status=recon.arm_status,
        adjudication_as_recorded=adjudication_as_recorded,
        adjudication_quarantined=not verdict_valid,
    )
