"""WP14 (p50) experiment-integrity value types.

Contract: ``docs/p50_experiment_integrity_draft.md`` §2/§5.  Frozen
dataclasses only — no IO, no loader/engine imports, so every consumer
(plan materialization, R2 precheck) runs BEFORE any large data load.

What this module deliberately does not do: it does not re-run, patch or
re-adjudicate any historical artifact; it does not register anything
into ``artifact_schemas`` (measurement artifact; p34 §4 version-table
precedent, r02_matrix.py:53-55); it does not fold distinct economic
objects into one key (RowKey always carries every axis; the r02_periods
lesson — dropping ``period`` collapses 120 rows onto 60 keys, measured).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: p50 §2 schema versions (measurement artifacts; bump comments below).
#: v1 (WP14): initial — plan / reconcile / status / errata payloads.
PLAN_SCHEMA_VERSION = 1
RECONCILE_SCHEMA_VERSION = 1
STATUS_SCHEMA_VERSION = 1
ERRATA_SCHEMA_VERSION = 1

#: The arm kinds a plan distinguishes (R1).  benchmark/control are
#: comparison arms: measured and reported, never adjudicated as value
#: evidence (r01_baseline.py:819-821 precedent — decide_adjudication
#: skips them by name).
ARM_KINDS = ("single", "combo", "benchmark", "control")

#: Terminal classification labels (R4/R5).  ``measurement_incomplete``
#: SHADOWS every economic label: it wins whenever any measurement-
#: integrity defect exists, even if the report says completed=true and
#: the frozen adjudication output looks negative (M3 discipline).
STATUS_LABELS = (
    "not_generated",
    "measurement_incomplete",
    "summary_layer_inconsistent",
    "economic_negative",
    "economic_positive",
    "inconclusive_mixed",
)

#: Per-arm measurement outcome (R6: the five slow singles must come out
#: ``citable`` while the two combos come out ``program_error``).
ARM_MEASUREMENT_STATUSES = ("citable", "program_error", "missing", "partial")


class PlanValidationError(ValueError):
    """A plan/precheck violation (C3 layering: ValueError subclass).

    Raised by plan.py / precheck.py for constructible, locatable
    problems (wrong-domain member, dangling combo reference, invalid
    folds).  Messages are lowercase multi-line f-strings naming the
    violated clause and the (experiment, arm, member, fold) locator.
    """


class IntegrityRefusal(RuntimeError):
    """A fail-closed refusal of reconciliation / status / errata (C3).

    Raised when an input cannot be audited at all (unparseable report,
    plan/report experiment mismatch, write target outside the single
    permitted directory).  Integrity FINDINGS are data, not refusals.
    """


@dataclass(frozen=True)
class DirectionRule:
    """How member directions are produced for one arm (R1 requirement:
    the run plan must state the direction rule explicitly)."""

    rule: str            # direction.py rule id, e.g. "reference_return"
    rule_version: int    # direction.DIRECTION_RULE_VERSION at run time
    scope: str           # "per_fold" | "per_(period,fold)"
    learned_at: str      # e.g. "reference_cell=n20_m5000" | "none"


@dataclass(frozen=True)
class PlannedArm:
    """One preregistered arm AFTER cell/period expansion (R1): the full
    row-value identity plus its members, direction rule, cost ladder.

    ``cell`` is the arm's cell axis (F19): signal arms and control:cash
    carry their reference cell (cash rows have NO cell:: prefix in the
    candidate string yet cell="n20_m5000" in the row — measured), while
    benchmark stays None (benchmark rows carry no cell field at all).
    """

    candidate: str                     # full row value, incl. "cell::" prefix
    kind: str                          # one of ARM_KINDS
    members: tuple[str, ...]           # single: (own factor,); combo:
                                       # member factors; benchmark/control: ()
    direction_rule: DirectionRule | None   # None for benchmark/control
    cost_levels: tuple[str, ...]       # per-arm ladder — benchmark ("ref",),
                                       # r03 reference singles ("1x",),
                                       # r01 signal arms ("0x","1x","2x")
    is_primary: bool                   # R5 主要预注册臂: a program error on
                                       # it forces measurement_incomplete
    members_source: str                # provenance, e.g.
                                       # "config.protocol.baseline_signals"
                                       # or "code.R01_COMBO_ARMS"
    cell: str | None = None            # cell axis of the EXPANDED arm; None
                                       # for cell-free arms and benchmark
    period: str | None = None          # r02_periods arm expansion axis
    note: str = ""


@dataclass(frozen=True)
class PlannedFold:
    train_end: str    # YYYY-MM-DD
    test_end: str     # YYYY-MM-DD


@dataclass(frozen=True)
class RunPlan:
    """The materialized run plan (R1): everything a run must measure,
    BEFORE any data is loaded.  Rebuildable offline from L2 config +
    L3 code constants (P §0) and JSON-round-trippable for the CLI."""

    experiment_id: str        # one of EXPERIMENT_IDS
    research_domain: str
    tier_set: str | None
    folds: tuple[PlannedFold, ...]
    arms: tuple[PlannedArm, ...]      # cell/period-EXPANDED (r02_matrix:
                                       # 30 signal + control + benchmark = 32,
                                       # == its arm_summaries key count, F8)
    key_axes: tuple[str, ...]         # load-bearing identity axes, subset of
                                       # ("period","cell","candidate",
                                       #  "cost_level","fold_test_end")
    adjudication_rules: dict[str, float]
    verdict_map: dict[str, str]       # EXHAUSTIVE literal map from the
                                       # experiment's frozen verdict
                                       # vocabulary (enumerated verbatim
                                       # from the r0x code, F23) to an
                                       # economic label.  NO wildcard /
                                       # "everything else" entries — an
                                       # outcome word ABSENT from the map
                                       # is a fail-closed IntegrityRefusal,
                                       # never a silent default (§5.4)
    sources: dict[str, Any]           # L1-L4 snapshot: contract/config
                                       # paths + sha256s, code-constant
                                       # module/names, product paths,
                                       # adjudication_path (outcome-word
                                       # location), summary_layer_path and
                                       # summary_layer_arms (F20)
    plan_schema_version: int = PLAN_SCHEMA_VERSION

    @property
    def primary_arms(self) -> tuple[PlannedArm, ...]:
        return tuple(a for a in self.arms if a.is_primary)

    @property
    def expected_row_count(self) -> int:
        return sum(len(a.cost_levels) * len(self.folds) for a in self.arms)


#: The four audited experiments (P §0; row counts verified: F2/F8/F9/F10)
#: plus the two WP18 experiment ids (p56, FROZEN_18_PLAN §10.2 additive
#: registration: medium = M1 30 + M2 40 + M3 90 + C1 40 = 200 expected
#: rows, slow = S1 50 + S2 60 + S3 130 + C2 40 = 280; both ≤ 300 and the
#: FORMAL rows — M3/S3 account 1x + C1/C2 stress 0x/2x — sum to exactly
#: 300 across the two domains at full extension, §4.6).
EXPERIMENT_IDS = (
    "r01_slow_baseline",
    "r02_matrix",
    "r02_periods",
    "r03_combos",
    "wp18_medium",
    "wp18_slow",
)


@dataclass(frozen=True)
class RowKey:
    """Uniqueness key of one materialized row (F3/F8/F9/F10 verified
    collision-free on all four artifacts).

    ALL five axis fields always participate in equality; an axis the
    experiment does not use stays ``None`` (its neutral value).  Dropping
    a load-bearing axis folds distinct economic objects into one key —
    forbidden; the plan's ``key_axes`` documents which axes carry weight
    and reconcile flags non-neutral values on inactive axes.
    ``seed`` is deliberately absent (all-null across the artifacts, F6).
    """

    period: str | None = None
    cell: str | None = None
    candidate: str = ""
    cost_level: str = ""
    fold_test_end: str = ""


@dataclass(frozen=True)
class PrecheckIssue:
    """One R2 finding, locatable with NO data loaded."""

    code: str          # see precheck.py CHECK codes
    experiment_id: str
    arm: str | None
    member: str | None
    locator: str       # "(experiment=…, arm=…, member=…)" position string
    detail: str


@dataclass(frozen=True)
class PrecheckReport:
    issues: tuple[PrecheckIssue, ...]
    checks_run: tuple[str, ...]   # names of the check classes executed

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ReconcileIssue:
    """One R3 finding at a locatable row position."""

    code: str   # "missing_row" | "duplicate_row" | "unexpected_row" |
                # "unexpected_axis_value" | "failed_row" |
                # "date_coverage_mismatch" | "returns_length_mismatch" |
                # "fold_window_overlap" | "summary_layer_mismatch" |
                # "meta_row_count_mismatch" | "provenance_anchor_absent"
                # (the last: an L1/L2 config/contract identity anchor
                # absent on either side — disclosed, §3.3; a PRESENT
                # anchor that MISMATCHES is an IntegrityRefusal in
                # reconcile.py, never a finding)
    key: RowKey | None
    detail: str


@dataclass(frozen=True)
class ArmMeasurementStatus:
    """Per-arm measurement outcome (R3 -> R6 citability)."""

    candidate: str
    kind: str
    is_primary: bool
    n_expected_rows: int
    n_present_rows: int
    n_failed_rows: int
    failure_reasons: tuple[str, ...]     # distinct, sorted
    status: str                          # ARM_MEASUREMENT_STATUSES:
                                         # citable = all rows present & ok;
                                         # program_error = >=1 failed row;
                                         # missing = rows absent;
                                         # partial = some rows absent


@dataclass(frozen=True)
class Reconciliation:
    """R3 output: expected-vs-actual by unique key, plus the failure
    and coverage layers.  Rows are the authority; the summary layer is
    rebuilt from rows and compared (S §9.4)."""

    experiment_id: str
    expected_keys: frozenset[RowKey]
    actual_keys: frozenset[RowKey]
    n_rows_observed: int              # len(report["rows"]) as recorded by
                                       # reconcile — classify_status rebinds
                                       # recon to the passed report with it
                                       # (§3.4 rule 3: no orphan recon can
                                       # smuggle an all-clean verdict)
    missing: tuple[RowKey, ...]                    # expected - actual
    duplicates: tuple[tuple[RowKey, int], ...]     # key, multiplicity (>1)
    unexpected: tuple[RowKey, ...]                 # actual - expected
    failed_rows: tuple[tuple[RowKey, str], ...]    # key, failure_reason
    primary_failed_rows: tuple[tuple[RowKey, str], ...]
    benchmark_failed_rows: tuple[tuple[RowKey, str], ...]
    coverage: dict[str, Any]     # per (period, fold_test_end): the observed
                                 # n_dates set, daily_dates-consistency flag,
                                 # n_dates-vs-len(daily_returns) flag
    arm_status: tuple[ArmMeasurementStatus, ...]
    issues: tuple[ReconcileIssue, ...]
    reconcile_schema_version: int = RECONCILE_SCHEMA_VERSION


@dataclass(frozen=True)
class StatusReport:
    """R4/R5 three-layer status.  Each layer carries its OWN evidence
    dict; no layer is ever derived from a layer above it, and
    ``completed`` feeds ONLY report_generated (§5 anti-shortcut)."""

    experiment_id: str
    report_generated: bool
    report_generated_evidence: dict[str, Any]
    primary_arms_measured: bool
    primary_arms_evidence: dict[str, Any]
    verdict_valid: bool
    verdict_valid_evidence: dict[str, Any]
    label: str                      # STATUS_LABELS
    arm_status: tuple[ArmMeasurementStatus, ...]
    adjudication_as_recorded: dict[str, Any] | None   # verbatim copy
    adjudication_quarantined: bool  # True whenever verdict_valid is False:
                                    # the frozen adjudication output exists
                                    # but must NOT be read as an economic
                                    # result (r01 slow: worth_search=false
                                    # is quarantined, F7)
    status_schema_version: int = STATUS_SCHEMA_VERSION
