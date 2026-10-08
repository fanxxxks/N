"""WP14 (p50) R3: expected-vs-actual reconciliation by unique key.

Contract: ``docs/p50_experiment_integrity_draft.md`` §3.3 (the contract
file lands with the WP14 contract-writing stage per DESIGN §1; every
check here is pinned to a DESIGN §3.3 clause and a measured fact
F3/F4/F8/F9/F10/F12/F19/F20).  The expected key set comes ONLY from the
materialized plan (``plan.expected_keys`` — cross(arms, cost_levels,
folds) with per-arm ladders, never a uniform ladder); the report's rows
layer is the authority for what actually happened (F12), and the
summary layer is REBUILT from rows and compared.

What this module deliberately does not do: it makes no economic
judgment (labels are status.py's job; findings are data, never
refusals); it never compares date coverage ACROSS experiments (r01's
fold calendar is 10 trading days shorter than r02/r03 by design — S §6;
groups are keyed (period, fold_test_end) WITHIN one report only); it
contains no ``if experiment_id == …`` classification branch — the
summary-layer location, arm domain and period prefilter are all read
from ``plan.sources`` declarations (S §9.3), and the only
experiment-keyed table below is the INPUT-IDENTITY map for the refusal
case "report does not belong to this plan's experiment" (§3.3:
IntegrityRefusal is reserved for inputs that cannot be audited at all).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .plan import expected_keys
from .types import (
    RECONCILE_SCHEMA_VERSION,
    ArmMeasurementStatus,
    IntegrityRefusal,
    ReconcileIssue,
    Reconciliation,
    RowKey,
    RunPlan,
)

#: Input-identity markers (G2's family key, one per audited experiment;
#: measured on the frozen artifacts — r01_baseline_report.json carries
#: r01_report_schema_version=3, the other three =1).  A report that does
#: not carry THIS plan's family key with the known value cannot be
#: audited against this plan at all → IntegrityRefusal (never a finding:
#: there is nothing locatable to report).
_REPORT_SCHEMA_KEYS: dict[str, tuple[str, int]] = {
    "r01_slow_baseline": ("r01_report_schema_version", 3),
    "r02_matrix": ("r02_report_schema_version", 1),
    "r02_periods": ("r02_periods_report_schema_version", 1),
    "r03_combos": ("r03_report_schema_version", 1),
    # WP18 (p56): each domain's report carries its OWN schema stamp, one
    # key per experiment exactly like the r0x families — the foreign-family
    # guard above assumes that invariant (a shared key name would make
    # every wp18 report look foreign to its own plan, and would let a
    # wp18_slow report reconcile against a wp18_medium plan down to row
    # findings instead of an upfront refusal).
    "wp18_medium": ("wp18_medium_report_schema_version", 1),
    "wp18_slow": ("wp18_slow_report_schema_version", 1),
}

#: The L1/L2 provenance anchors plan_sources snapshots and the r0x
#: reports record (measured: all four frozen reports carry
#: ``provenance.config_sha256`` and ``provenance.contract_sha256``,
#: 8/8 matching the current files — the r0x writers record UPPERCASE
#: config digests while ``_sha256_file`` yields lowercase, so hex
#: digests compare case-insensitively).
_PROVENANCE_ANCHORS: tuple[str, ...] = ("config", "contract")

#: The five RowKey axes and each axis's neutral value (types.RowKey
#: docstring: an axis the experiment does not use stays neutral).  An
#: axis NOT in plan.key_axes that carries a non-neutral value on a row
#: is an ``unexpected_axis_value`` finding (§3.3; r01 rows carrying a
#: cell, r02_matrix rows carrying a period).
_ALL_AXES = ("period", "cell", "candidate", "cost_level", "fold_test_end")
_NEUTRAL_AXIS_VALUE: dict[str, Any] = {
    "period": None,
    "cell": None,
    "candidate": "",
    "cost_level": "",
    "fold_test_end": "",
}

#: Sentinel classification string for a failed row whose failure_reason
#: field is absent (every real failed row carries one, F12; a synthetic
#: or corrupted stub must still classify by a visible, sorted string).
_MISSING_FAILURE_REASON = "<absent failure_reason>"


def _key_sort(key: RowKey) -> tuple[str, str, str, str, str]:
    """A deterministic sort tuple: neutral period/cell sort first."""

    return (
        key.period or "",
        key.cell or "",
        key.candidate,
        key.cost_level,
        key.fold_test_end,
    )


def _key_to_dict(key: RowKey) -> dict[str, Any]:
    return {
        "period": key.period,
        "cell": key.cell,
        "candidate": key.candidate,
        "cost_level": key.cost_level,
        "fold_test_end": key.fold_test_end,
    }


def row_key_of(row: Mapping[str, Any], plan: RunPlan) -> RowKey:
    """The five-axis uniqueness key of one materialized row (§3.3):
    ``period``/``cell`` via ``.get`` (benchmark rows carry no cell field
    at all, F8 — their neutral axis value is None) and
    candidate/cost_level/fold_test_end read directly.  ``plan`` is
    accepted for signature stability with DESIGN §3.3 (and stays the
    hook for any future per-experiment axis renaming); the projection
    itself is plan-independent so keys are auditable even for rows the
    plan never expected — those become ``unexpected_row`` findings, not
    refusals."""

    return RowKey(
        period=row.get("period"),
        cell=row.get("cell"),
        candidate=str(row.get("candidate", "")),
        cost_level=str(row.get("cost_level", "")),
        fold_test_end=str(row.get("fold_test_end", "")),
    )


def rebuild_failed_folds(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Rebuild each arm's ``failed_folds`` from the rows layer with the
    r0x semantics frozen at r01_baseline.py:492-497 — "a fold fails for
    an arm when ANY of its levels failed".  Grouping is by the row's
    full candidate string (cell::arm under r02/r03, so cells stay
    separate) and by fold_test_end; the caller decides whether rows
    must be period-prefiltered first (plan.sources[
    "summary_layer_period_prefilter"], F20)."""

    by_arm_fold: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for row in rows:
        candidate = str(row.get("candidate", ""))
        fold = str(row.get("fold_test_end", ""))
        level = str(row.get("cost_level", ""))
        by_arm_fold.setdefault(candidate, {}).setdefault(fold, {})[level] = row
    return {
        candidate: sum(
            1
            for per_level in folds.values()
            if any(bool(row.get("failed")) for row in per_level.values())
        )
        for candidate, folds in by_arm_fold.items()
    }


def _require_rows(
    plan: RunPlan, report: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """The fail-closed gate (§3.3): IntegrityRefusal — never a finding —
    when the input cannot be audited at all: a non-mapping report, a
    report without a rows layer, a row that is not a mapping (no key can
    be projected), or a report whose schema-family key does not match
    the plan's experiment."""

    if not isinstance(report, Mapping):
        raise IntegrityRefusal(
            f"report is not a mapping\n(experiment={plan.experiment_id}, "
            f"locator=report, type={type(report).__name__})"
        )
    schema_key, schema_value = _REPORT_SCHEMA_KEYS[plan.experiment_id]
    present = sorted(k for k, (_s, _v) in _REPORT_SCHEMA_KEYS.items() if _s in report)
    expected_id = plan.experiment_id
    if schema_key not in report:
        raise IntegrityRefusal(
            "report does not carry this experiment's schema-family key — "
            "plan/report experiment mismatch\n"
            f"(experiment={expected_id}, locator={schema_key}, "
            f"schema_keys_present={[ _REPORT_SCHEMA_KEYS[p][0] for p in present ]})"
        )
    if report[schema_key] != schema_value:
        raise IntegrityRefusal(
            "report carries an unknown schema version for this "
            "experiment's family — cannot be audited against this plan\n"
            f"(experiment={expected_id}, locator={schema_key}, "
            f"value={report[schema_key]!r}, known={schema_value!r})"
        )
    other_families = [
        _REPORT_SCHEMA_KEYS[exp][0]
        for exp in _REPORT_SCHEMA_KEYS
        if exp != expected_id and _REPORT_SCHEMA_KEYS[exp][0] in report
    ]
    if other_families:
        raise IntegrityRefusal(
            "report carries schema keys of a different experiment family — "
            "plan/report experiment mismatch\n"
            f"(experiment={expected_id}, locator=schema_family, "
            f"foreign_keys={other_families})"
        )
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise IntegrityRefusal(
            "report has no rows layer to reconcile\n"
            f"(experiment={expected_id}, locator=rows, "
            f"type={type(rows).__name__})"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise IntegrityRefusal(
                "row is not a mapping — no key can be projected\n"
                f"(experiment={expected_id}, locator=rows[{index}], "
                f"type={type(row).__name__})"
            )
    return rows


def _provenance_anchor_issues(
    plan: RunPlan, report: Mapping[str, Any]
) -> list[ReconcileIssue]:
    """The plan↔report provenance anchoring promised by
    ``plan.plan_sources``' docstring ("a later reconcile compares them
    against the report's recorded provenance"): the report's recorded
    ``provenance.config_sha256`` / ``provenance.contract_sha256`` are
    compared against the plan's L1/L2 source snapshots.

    A MISMATCH is an ``IntegrityRefusal`` carrying both measured
    values: the report was measured from different inputs than the
    plan describes, so every downstream comparison would run across
    mismatched calibers (the P45 lesson) — that is an input-identity
    defect, not a locatable row finding.  A MISSING anchor on either
    side is a locatable ``provenance_anchor_absent`` finding — the
    undisclosed identity is reported, never silently skipped and never
    inflated into a refusal.  Hex digests compare case-insensitively
    (the r0x writers record UPPERCASE config digests while
    ``_sha256_file`` yields lowercase — measured on all four frozen
    artifacts)."""

    recorded = report.get("provenance")
    recorded_map = recorded if isinstance(recorded, Mapping) else {}
    issues: list[ReconcileIssue] = []
    for name in _PROVENANCE_ANCHORS:
        entry = plan.sources.get(name)
        plan_sha = entry.get("sha256") if isinstance(entry, Mapping) else None
        report_sha = recorded_map.get(f"{name}_sha256")
        if report_sha is None or plan_sha is None:
            if report_sha is None:
                missing_side = (
                    f"report (no provenance.{name}_sha256 field — the "
                    "report never recorded this input's identity)"
                )
            else:
                missing_side = (
                    f"plan (sources[{name!r}] carries no sha256 — the "
                    "plan never snapshotted this input's identity)"
                )
            issues.append(
                ReconcileIssue(
                    code="provenance_anchor_absent",
                    key=None,
                    detail=(
                        f"provenance anchor '{name}' is absent on the "
                        f"{missing_side}; input identity for this anchor "
                        "is undisclosed, not verified"
                    ),
                )
            )
            continue
        if str(report_sha).strip().lower() != str(plan_sha).strip().lower():
            plan_path = entry.get("path") if isinstance(entry, Mapping) else None
            raise IntegrityRefusal(
                "report provenance does not match the plan's source "
                f"snapshot for the '{name}' anchor — the report was "
                "measured from different inputs than the plan "
                "describes, and reconciling the two would compare "
                "across mismatched calibers (the P45 lesson)\n"
                f"(experiment={plan.experiment_id}, "
                f"locator=provenance.{name}_sha256, "
                f"report={str(report_sha)!r}, plan={str(plan_sha)!r}, "
                f"plan path={str(plan_path)!r})"
            )
    return issues


def _axis_issues(
    plan: RunPlan, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[ReconcileIssue], list[RowKey]]:
    """Per-row inactive-axis scan + key projection in row order."""

    inactive_axes = tuple(
        axis for axis in _ALL_AXES if axis not in plan.key_axes
    )
    issues: list[ReconcileIssue] = []
    keys: list[RowKey] = []
    for index, row in enumerate(rows):
        key = row_key_of(row, plan)
        keys.append(key)
        for axis in inactive_axes:
            value = row.get(axis)
            if value != _NEUTRAL_AXIS_VALUE[axis]:
                issues.append(
                    ReconcileIssue(
                        code="unexpected_axis_value",
                        key=key,
                        detail=(
                            f"row carries a value on axis '{axis}' that the "
                            f"plan declares inactive (key_axes="
                            f"{list(plan.key_axes)}); rows[{index}], "
                            f"{axis}={value!r}"
                        ),
                    )
                )
    return issues, keys


def _coverage_layer(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[RowKey],
) -> tuple[list[ReconcileIssue], dict[str, Any]]:
    """Date-coverage and fold-window checks WITHIN one report (§3.3):
    successful rows grouped by (period, fold_test_end); inside a group
    n_dates must be single-valued, daily_dates vectors (when the report
    carries them — r01/r03 only, S §2.1) must be identical, and every
    row must satisfy n_dates == len(daily_returns).  Observed fold
    windows must be non-inverted and pairwise disjoint per period.
    Cross-experiment date comparison is structurally impossible here:
    the only key is (period, fold_test_end) of THIS report."""

    issues: list[ReconcileIssue] = []
    groups: dict[tuple[str | None, str], list[tuple[RowKey, Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for key, row in zip(keys, rows):
        if not row.get("failed"):
            groups[(key.period, key.fold_test_end)].append((key, row))

    group_records: dict[str, Any] = {}
    for (period, fold), members in sorted(
        groups.items(), key=lambda item: (item[0][0] or "", item[0][1])
    ):
        n_dates_values = sorted(
            {
                int(row["n_dates"])
                for _key, row in members
                if row.get("n_dates") is not None
            }
        )
        daily_vectors = {
            tuple(row.get("daily_dates", ()))
            for _key, row in members
            if row.get("daily_dates") is not None
        }
        locator = f"(period={period!r}, fold_test_end={fold!r})"
        n_dates_consistent = len(n_dates_values) <= 1
        daily_consistent = len(daily_vectors) <= 1
        if not n_dates_consistent:
            issues.append(
                ReconcileIssue(
                    code="date_coverage_mismatch",
                    key=None,
                    detail=(
                        f"successful rows disagree on n_dates inside one "
                        f"fold group {locator}: observed {n_dates_values}"
                    ),
                )
            )
        if not daily_consistent:
            issues.append(
                ReconcileIssue(
                    code="date_coverage_mismatch",
                    key=None,
                    detail=(
                        f"successful rows carry {len(daily_vectors)} distinct "
                        f"daily_dates vectors inside one fold group "
                        f"{locator}; the test calendar must be identical "
                        f"within a fold"
                    ),
                )
            )
        group_records[f"{period or '-'}|{fold}"] = {
            "n_rows": len(members),
            "n_dates_observed": n_dates_values,
            "n_dates_consistent": n_dates_consistent,
            "daily_dates_present": any(
                row.get("daily_dates") is not None for _key, row in members
            ),
            "daily_dates_consistent": daily_consistent,
        }
    for key, row in zip(keys, rows):
        if row.get("failed"):
            continue
        daily_returns = row.get("daily_returns")
        n_dates = row.get("n_dates")
        if (
            daily_returns is not None
            and n_dates is not None
            and int(n_dates) != len(daily_returns)
        ):
            issues.append(
                ReconcileIssue(
                    code="returns_length_mismatch",
                    key=key,
                    detail=(
                        f"n_dates={n_dates} != len(daily_returns)="
                        f"{len(daily_returns)} at {locator_of(key)}"
                    ),
                )
            )

    # Fold windows from OBSERVED rows (failed stubs still carry both
    # dates, S §2.3): per period, windows must satisfy train < test and
    # be pairwise disjoint when sorted by test_end.
    windows: dict[str | None, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        windows[row.get("period")].add(
            (str(row.get("fold_train_end", "")), str(row.get("fold_test_end", "")))
        )
    window_records: dict[str, list[list[str]]] = {}
    for period, period_windows in sorted(
        windows.items(), key=lambda item: item[0] or ""
    ):
        # lists (not tuples) so the coverage dict is JSON-native by
        # construction (reconcile_to_json round-trips losslessly).
        window_records[period or "-"] = [
            list(window) for window in sorted(period_windows)
        ]
        ordered = sorted(period_windows, key=lambda w: (w[1], w[0]))
        for train_end, test_end in ordered:
            if not train_end or not test_end or train_end >= test_end:
                issues.append(
                    ReconcileIssue(
                        code="fold_window_overlap",
                        key=None,
                        detail=(
                            f"observed fold window is inverted or undated "
                            f"(period={period!r}, train_end={train_end!r}, "
                            f"test_end={test_end!r}): train_end must be "
                            f"earlier than test_end"
                        ),
                    )
                )
        for (_tr1, te1), (tr2, _te2) in zip(ordered, ordered[1:]):
            if tr2 < te1:
                issues.append(
                    ReconcileIssue(
                        code="fold_window_overlap",
                        key=None,
                        detail=(
                            f"observed test windows overlap (period="
                            f"{period!r}): window(train_end={tr2!r}) starts "
                            f"before the previous window ends "
                            f"(test_end={te1!r})"
                        ),
                    )
                )
    return issues, {
        "groups": group_records,
        "fold_windows": window_records,
        "summary_layer_extra_arms": [],
    }


def locator_of(key: RowKey) -> str:
    """The locatable position string of one row key (C3 message style)."""

    return (
        f"(period={key.period!r}, cell={key.cell!r}, "
        f"candidate={key.candidate!r}, cost_level={key.cost_level!r}, "
        f"fold_test_end={key.fold_test_end!r})"
    )


def _summary_layer_issues(
    plan: RunPlan,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    coverage: dict[str, Any],
) -> list[ReconcileIssue]:
    """Summary-layer rebuild comparison (§3.3, F12/F20): rebuild
    failed_folds from rows and compare ONLY the arms the summary layer's
    own domain lists; row-layer arms outside the domain (r02_periods:
    cash + benchmark) are recorded in coverage[
    "summary_layer_extra_arms"] and never reported as issues.  The
    layer's location, domain and period prefilter all come from
    plan.sources — no experiment-id branch."""

    issues: list[ReconcileIssue] = []
    path = plan.sources["summary_layer_path"]
    prefilter = bool(plan.sources["summary_layer_period_prefilter"])
    extras: list[str] = []

    if path == "arm_summaries":
        summary = report.get("arm_summaries")
        if not isinstance(summary, Mapping):
            return [
                ReconcileIssue(
                    code="summary_layer_mismatch",
                    key=None,
                    detail=(
                        "the summary layer declared by the plan "
                        f"(summary_layer_path={path!r}) is absent from the "
                        "report — the rows layer cannot be cross-checked "
                        "against it"
                    ),
                )
            ]
        rebuilt = rebuild_failed_folds(rows)
        for arm in sorted(summary):
            recorded = summary[arm].get("failed_folds") if isinstance(
                summary[arm], Mapping
            ) else None
            rebuilt_value = rebuilt.get(str(arm), 0)
            if recorded != rebuilt_value:
                issues.append(
                    ReconcileIssue(
                        code="summary_layer_mismatch",
                        key=None,
                        detail=(
                            f"arm_summaries failed_folds disagrees with the "
                            f"rows-layer rebuild {locator_arm(arm)}: "
                            f"recorded={recorded!r}, rebuilt={rebuilt_value}"
                        ),
                    )
                )
        extras = sorted(set(rebuilt) - {str(arm) for arm in summary})
    elif path == "period_blocks.<freq>.arms":
        blocks = report.get("period_blocks")
        if not isinstance(blocks, Mapping):
            return [
                ReconcileIssue(
                    code="summary_layer_mismatch",
                    key=None,
                    detail=(
                        "the summary layer declared by the plan "
                        f"(summary_layer_path={path!r}) is absent from the "
                        "report — the rows layer cannot be cross-checked "
                        "against it"
                    ),
                )
            ]
        for period in sorted(blocks):
            block = blocks[period]
            arms = block.get("arms") if isinstance(block, Mapping) else None
            if not isinstance(arms, Mapping):
                issues.append(
                    ReconcileIssue(
                        code="summary_layer_mismatch",
                        key=None,
                        detail=(
                            f"period_blocks[{period!r}].arms is absent or not "
                            "a mapping — declared summary layer not auditable"
                        ),
                    )
                )
                continue
            # F20: prefilter rows per period before rebuilding, else the
            # two periods' same-named folds collapse onto one arm.
            period_rows = (
                [row for row in rows if row.get("period") == period]
                if prefilter
                else list(rows)
            )
            rebuilt = rebuild_failed_folds(period_rows)
            for arm in sorted(arms):
                recorded = arms[arm].get("failed_folds") if isinstance(
                    arms[arm], Mapping
                ) else None
                rebuilt_value = rebuilt.get(str(arm), 0)
                if recorded != rebuilt_value:
                    issues.append(
                        ReconcileIssue(
                            code="summary_layer_mismatch",
                            key=None,
                            detail=(
                                f"period_blocks[{period!r}].arms "
                                f"failed_folds disagrees with the rows-layer "
                                f"rebuild {locator_arm(arm)}: "
                                f"recorded={recorded!r}, "
                                f"rebuilt={rebuilt_value}"
                            ),
                        ),
                    )
            extras.extend(
                f"period={period!r}:{arm}"
                for arm in sorted(set(rebuilt) - {str(a) for a in arms})
            )
    else:
        return [
            ReconcileIssue(
                code="summary_layer_mismatch",
                key=None,
                detail=(
                    "plan declares an unknown summary_layer_path "
                    f"{path!r} — no comparison rule exists (fail-visible, "
                    "never a silent skip)"
                ),
            )
        ]
    coverage["summary_layer_extra_arms"] = sorted(set(extras))
    return issues


def locator_arm(arm: str) -> str:
    return f"(arm={arm!r})"


def _arm_status_layer(
    plan: RunPlan,
    expected: frozenset[RowKey],
    actual: frozenset[RowKey],
    failed_by_key: Counter[RowKey],
    failure_reasons_by_arm: dict[tuple[str | None, str | None, str], list[str]],
) -> tuple[ArmMeasurementStatus, ...]:
    """Per-arm measurement outcome over the plan's own arm list (R3 →
    R6 citability): n_expected = len(cost_levels) x len(folds) per arm;
    n_present counts PRESENT EXPECTED keys only (unexpected rows stay in
    the unexpected list); failed counts raw failed rows of the arm;
    status: program_error if any failed row, else missing (0 present) /
    partial (some present) / citable (all present, none failed)."""

    statuses = []
    for arm in plan.arms:
        triple = (arm.period, arm.cell, arm.candidate)
        present = sum(
            1
            for key in actual
            if (key.period, key.cell, key.candidate) == triple
            and key in expected
        )
        n_expected = len(arm.cost_levels) * len(plan.folds)
        n_failed = sum(
            count
            for key, count in failed_by_key.items()
            if (key.period, key.cell, key.candidate) == triple
        )
        reasons = tuple(sorted(set(failure_reasons_by_arm.get(triple, ()))))
        if n_failed >= 1:
            status = "program_error"
        elif present == 0:
            status = "missing"
        elif present < n_expected:
            status = "partial"
        else:
            status = "citable"
        statuses.append(
            ArmMeasurementStatus(
                candidate=arm.candidate,
                kind=arm.kind,
                is_primary=arm.is_primary,
                n_expected_rows=n_expected,
                n_present_rows=present,
                n_failed_rows=n_failed,
                failure_reasons=reasons,
                status=status,
            )
        )
    return tuple(statuses)


def reconcile_report(
    plan: RunPlan,
    report: Mapping[str, Any],
    run_meta: Mapping[str, Any] | None = None,
) -> Reconciliation:
    """Reconcile one report against its plan by unique key (§3.3):
    missing rows (expected − actual), duplicate rows (Counter
    multiplicity > 1 — never silently deduplicated), unexpected rows
    (actual − expected), failed rows (rows-layer authority, F12, split
    into primary / benchmark streams), date coverage within the report,
    the rebuilt summary layer, and the optional run_meta row count.
    Findings are DATA (returned in ``issues``); only unauditable inputs
    raise IntegrityRefusal."""

    if run_meta is not None and not isinstance(run_meta, Mapping):
        raise IntegrityRefusal(
            "run_meta is not a mapping\n"
            f"(experiment={plan.experiment_id}, locator=run_meta, "
            f"type={type(run_meta).__name__})"
        )
    rows = _require_rows(plan, report)
    # §3.3 provenance anchoring runs BEFORE any set comparison: a
    # mismatched input identity (config/contract drift) refuses here,
    # with both measured values in the message; missing anchors are
    # returned as locatable findings.
    provenance_issues = _provenance_anchor_issues(plan, report)

    axis_issues, keys = _axis_issues(plan, rows)
    expected = frozenset(expected_keys(plan))
    actual_counter: Counter[RowKey] = Counter(keys)
    actual = frozenset(actual_counter)

    missing = tuple(sorted(expected - actual, key=_key_sort))
    duplicates = tuple(
        (key, count)
        for key, count in sorted(
            ((k, c) for k, c in actual_counter.items() if c > 1),
            key=lambda item: _key_sort(item[0]),
        )
    )
    unexpected = tuple(sorted(actual - expected, key=_key_sort))

    issues: list[ReconcileIssue] = provenance_issues + list(axis_issues)
    for key in missing:
        issues.append(
            ReconcileIssue(
                code="missing_row",
                key=key,
                detail=f"planned row absent from the report at {locator_of(key)}",
            )
        )
    for key, multiplicity in duplicates:
        issues.append(
            ReconcileIssue(
                code="duplicate_row",
                key=key,
                detail=(
                    f"{multiplicity} rows share one key at {locator_of(key)} "
                    f"— reported, never silently overwritten"
                ),
            )
        )
    for key in unexpected:
        issues.append(
            ReconcileIssue(
                code="unexpected_row",
                key=key,
                detail=f"row outside the plan at {locator_of(key)}",
            )
        )

    primary_triples = {
        (arm.period, arm.cell, arm.candidate)
        for arm in plan.arms
        if arm.is_primary
    }
    benchmark_triples = {
        (arm.period, arm.cell, arm.candidate)
        for arm in plan.arms
        if arm.kind == "benchmark"
    }
    failed_rows: list[tuple[RowKey, str]] = []
    primary_failed_rows: list[tuple[RowKey, str]] = []
    benchmark_failed_rows: list[tuple[RowKey, str]] = []
    failed_by_key: Counter[RowKey] = Counter()
    reasons_by_arm: dict[tuple[str | None, str | None, str], list[str]] = (
        defaultdict(list)
    )
    for key, row in zip(keys, rows):
        if not row.get("failed"):
            continue
        reason = str(row.get("failure_reason") or _MISSING_FAILURE_REASON)
        triple = (key.period, key.cell, key.candidate)
        failed_rows.append((key, reason))
        failed_by_key[key] += 1
        reasons_by_arm[triple].append(reason)
        if triple in primary_triples:
            primary_failed_rows.append((key, reason))
        if triple in benchmark_triples:
            benchmark_failed_rows.append((key, reason))
    for key, reason in failed_rows:
        issues.append(
            ReconcileIssue(
                code="failed_row",
                key=key,
                detail=(
                    f"program failure (rows-layer authority) at "
                    f"{locator_of(key)}: {reason}"
                ),
            )
        )

    coverage_issues, coverage = _coverage_layer(rows, keys)
    issues.extend(coverage_issues)
    issues.extend(_summary_layer_issues(plan, report, rows, coverage))

    if run_meta is not None:
        recorded = run_meta.get("row_count")
        if recorded != len(rows):
            issues.append(
                ReconcileIssue(
                    code="meta_row_count_mismatch",
                    key=None,
                    detail=(
                        f"run_meta.row_count={recorded!r} != len(rows)="
                        f"{len(rows)} (experiment={plan.experiment_id})"
                    ),
                )
            )

    arm_status = _arm_status_layer(
        plan, expected, actual, failed_by_key, reasons_by_arm
    )
    return Reconciliation(
        experiment_id=plan.experiment_id,
        expected_keys=expected,
        actual_keys=actual,
        n_rows_observed=len(rows),
        missing=missing,
        duplicates=duplicates,
        unexpected=unexpected,
        failed_rows=tuple(failed_rows),
        primary_failed_rows=tuple(primary_failed_rows),
        benchmark_failed_rows=tuple(benchmark_failed_rows),
        coverage=coverage,
        arm_status=arm_status,
        issues=tuple(issues),
        reconcile_schema_version=RECONCILE_SCHEMA_VERSION,
    )


def reconcile_to_json(recon: Reconciliation) -> dict[str, Any]:
    """The CLI/audit JSON form (C10: the writer adds
    ``ensure_ascii=False, indent=2``; every field here is jsonable)."""

    return {
        "reconcile_schema_version": recon.reconcile_schema_version,
        "experiment_id": recon.experiment_id,
        "n_rows_observed": recon.n_rows_observed,
        "n_expected_keys": len(recon.expected_keys),
        "n_actual_keys": len(recon.actual_keys),
        "n_missing": len(recon.missing),
        "n_duplicates": len(recon.duplicates),
        "n_unexpected": len(recon.unexpected),
        "n_failed_rows": len(recon.failed_rows),
        "n_primary_failed_rows": len(recon.primary_failed_rows),
        "n_benchmark_failed_rows": len(recon.benchmark_failed_rows),
        "missing": [_key_to_dict(key) for key in recon.missing],
        "duplicates": [
            {"key": _key_to_dict(key), "multiplicity": multiplicity}
            for key, multiplicity in recon.duplicates
        ],
        "unexpected": [_key_to_dict(key) for key in recon.unexpected],
        "failed_rows": [
            {"key": _key_to_dict(key), "failure_reason": reason}
            for key, reason in recon.failed_rows
        ],
        "primary_failed_rows": [
            {"key": _key_to_dict(key), "failure_reason": reason}
            for key, reason in recon.primary_failed_rows
        ],
        "benchmark_failed_rows": [
            {"key": _key_to_dict(key), "failure_reason": reason}
            for key, reason in recon.benchmark_failed_rows
        ],
        "coverage": recon.coverage,
        "arm_status": [
            {
                "candidate": arm.candidate,
                "kind": arm.kind,
                "is_primary": arm.is_primary,
                "n_expected_rows": arm.n_expected_rows,
                "n_present_rows": arm.n_present_rows,
                "n_failed_rows": arm.n_failed_rows,
                "failure_reasons": list(arm.failure_reasons),
                "status": arm.status,
            }
            for arm in recon.arm_status
        ],
        "issues": [
            {
                "code": issue.code,
                "key": None if issue.key is None else _key_to_dict(issue.key),
                "detail": issue.detail,
            }
            for issue in recon.issues
        ],
    }
