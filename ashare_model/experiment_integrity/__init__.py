"""WP14 (p50) experiment integrity: plan, precheck, reconcile, status.

Contract: ``docs/p50_experiment_integrity_draft.md`` (§1 layout).
Import-light on purpose: only frozen value types plus the pure plan
materialization are exported here; the r0x experiment modules are
imported lazily inside functions so R1/R2 consumers can run BEFORE any
large data load.

What this subpackage deliberately does not do (red lines): it never
re-runs a backtest, never mutates anything under ``AlphaGPT/data/**``
(the only permitted write is errata files inside ``data/r1/
r01_slow_baseline``, added as NEW files), and never upgrades a frozen
adjudication verdict — classification only ever narrows how a report
may be read.
"""

from __future__ import annotations

from .plan import (
    expected_keys,
    materialize_plan,
    plan_from_json,
    plan_sources,
    plan_to_json,
)
from .reconcile import (
    rebuild_failed_folds,
    reconcile_report,
    reconcile_to_json,
    row_key_of,
)
from .types import (
    ARM_KINDS,
    ARM_MEASUREMENT_STATUSES,
    ERRATA_SCHEMA_VERSION,
    EXPERIMENT_IDS,
    PLAN_SCHEMA_VERSION,
    RECONCILE_SCHEMA_VERSION,
    STATUS_LABELS,
    STATUS_SCHEMA_VERSION,
    ArmMeasurementStatus,
    DirectionRule,
    IntegrityRefusal,
    PlanValidationError,
    PlannedArm,
    PlannedFold,
    PrecheckIssue,
    PrecheckReport,
    ReconcileIssue,
    Reconciliation,
    RowKey,
    RunPlan,
    StatusReport,
)

__all__ = [
    "ARM_KINDS",
    "ARM_MEASUREMENT_STATUSES",
    "ERRATA_SCHEMA_VERSION",
    "EXPERIMENT_IDS",
    "PLAN_SCHEMA_VERSION",
    "RECONCILE_SCHEMA_VERSION",
    "STATUS_LABELS",
    "STATUS_SCHEMA_VERSION",
    "ArmMeasurementStatus",
    "DirectionRule",
    "IntegrityRefusal",
    "PlanValidationError",
    "PlannedArm",
    "PlannedFold",
    "PrecheckIssue",
    "PrecheckReport",
    "ReconcileIssue",
    "Reconciliation",
    "RowKey",
    "RunPlan",
    "StatusReport",
    "expected_keys",
    "rebuild_failed_folds",
    "reconcile_report",
    "reconcile_to_json",
    "row_key_of",
    "materialize_plan",
    "plan_from_json",
    "plan_sources",
    "plan_to_json",
]
