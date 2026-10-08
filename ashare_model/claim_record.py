"""The claim record and its fail-closed qualification (WP17 W1/W2).

Frozen spec: ``reports/wp17/FROZEN_17.md`` — the single source of truth.
This module owns the five field groups every formal significance claim
must carry (§1.1) and the qualification algorithm that decides whether
a nominal-5% significance claim is supported at the artifact's actual
operating point (§2).  It never runs calibration and never compares
method versions; it consumes registered evidence as-is.

Design rulings (frozen):

* A caller-supplied ``regime`` string NEVER grants qualification (§1.2)
  — the covering scenario set is computed by the adjudicator from the
  task's ``n_common_dates`` alone (§2.4).
* ``qualified = (reason_codes == [])`` — fail-closed; every missing
  piece of evidence is a reason code, never a silent pass.
* The task-side bootstrap ``seed`` is NOT an identity comparison key:
  the calibration swept 400-1000 seeds per scenario, so method coverage
  is seed-independent (§1.3); the task seed stays in the max_t block.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .eval_corrections import MAX_T_METHOD_VERSION
from .stat_calibration import CALIBRATION_VERSION

#: The frozen structural missing-fold policy (FROZEN §1.1 group 3): a
#: trial whose folds ALL failed is dropped entirely; a missing/mismatched
#: benchmark series raises (never zero-padded, p41 B1); a trial not
#: covering the adjudication window W leaves the joint matrix but stays
#: in the multiplicity count.  This is a structural fact descriptor, not
#: a switch — the only registered value.
MISSING_FOLD_POLICY = "structural_no_imputation_no_zero_padding_v1"

#: The DSR side-statistic identity: analytic, declared uncalibrated
#: (p48 draft §2 B1) — qualification acts on max-t only, the promotion
#: gate's bound input.
DSR_METHOD = "analytic_bldp_uncalibrated_v1"

METHOD_VERSION_ATTESTATION = (
    "max_t_method_version=1 is attested, not retro-invented: the evidence "
    "file predates versioning and carries no version field; the mapping is "
    "declared forward by docs/p48_stat_calibration_measurement_log.md "
    "(统计实现 eval_corrections 零改动) and the git history examined in "
    "reports/wp17/_recon/map.md §2."
)

#: The calibration-evidence registration (FROZEN §1.3): append-only list;
#: old entries are never edited.  Entry fields: the registered source
#: path, the sha256 the rows were measured under, and the
#: max_t_method_version the entry pins.
CALIBRATION_EVIDENCE_REGISTRATION: list[dict[str, Any]] = [
    {
        "source": "D:/minequant/reports/p48/calibration_report.json",
        "sha256": (
            "b85a652705f8768fe5629473de5aac7841b575d2ff46617388e6a49c1b296d0e"
        ),
        "max_t_method_version": 1,
    }
]

#: The identity fields a consulted evidence row registers — exactly the
#: JSON's existing fields, nothing added, nothing rewritten (§1.1/§4.4:
#: the historical claim strings are never carried into new reports).
IDENTITY_ROW_FIELDS = (
    "scenario", "kind", "n_perms", "block_size", "seed_start", "reps",
    "verdict",
)

# --- reason codes (FROZEN §1.2, frozen enumeration) ----------------------------

EVIDENCE_NOT_REGISTERED = "EVIDENCE_NOT_REGISTERED"
EVIDENCE_HASH_MISMATCH = "EVIDENCE_HASH_MISMATCH"
EVIDENCE_IDENTITY_MISMATCH = "EVIDENCE_IDENTITY_MISMATCH"
N_DATES_OUT_OF_CALIBRATED_DOMAIN = "N_DATES_OUT_OF_CALIBRATED_DOMAIN"
K_OUT_OF_CALIBRATED_DOMAIN = "K_OUT_OF_CALIBRATED_DOMAIN"
MISSING_FOLD_POLICY_UNREGISTERED = "MISSING_FOLD_POLICY_UNREGISTERED"
SCENARIO_UNCALIBRATED = "SCENARIO_UNCALIBRATED"
SCENARIO_INSUFFICIENT_SAMPLES = "SCENARIO_INSUFFICIENT_SAMPLES"
SCENARIO_UNRELIABLE_AT_NOMINAL = "SCENARIO_UNRELIABLE_AT_NOMINAL"

#: The frozen operating domain (FROZEN §2.3): the calibration measured
#: exactly n in {30, 250} (point membership, never an interval) and the
#: declared K band [4, 9].  Everything else is uncalibrated territory.
CALIBRATED_N_DATES = (30, 250)
K_BAND = (4, 9)

#: The covering scenario set is decided by n_common_dates alone — the
#: caller has no say (FROZEN §2.4, adjudicator-computed).
COVERING_SCENARIOS_BY_N: dict[int, tuple[str, ...]] = {
    250: (
        "independent_noise", "common_factor_095",
        "ar1_phi04", "duplicate_candidates",
    ),
    30: ("short_window_30d",),
}

#: Verdicts that keep a nominal-5% claim (B5 semantics).
PASSING_VERDICTS = ("calibrated", "conservative")

# --- the frozen G2 refusal strings (FROZEN §3.3, verbatim) ----------------------

STRING_A_NO_CLAIM_RECORD = (
    "artifact carries no claim record (wp17 W1): pre-wp17 artifacts "
    "cannot support promotion - the five field groups (statistics, "
    "method, task actuals, calibration evidence identity, "
    "qualification) are required"
)

STRING_B_INCONSISTENT_TEMPLATE = (
    "claim record is inconsistent with the artifact's own statistical "
    "blocks (wp17 W1): {field} records {recorded!r} but the artifact "
    "carries {actual!r}"
)

STRING_C_UNQUALIFIED_TEMPLATE = (
    "significance qualification fail-closed: no calibrated evidence "
    "covers this artifact's operating point (n_common_dates={n}, "
    "pool_trials={k}, n_perms={np}, block_size={bs}, "
    "missing_fold_policy={policy}); reason codes: {codes}; a "
    "nominal-5% significance claim is unsupported regardless of the "
    "p-value (wp17 W2/W3)"
)


def no_reliable_inference_reason(codes: list[str]) -> str:
    """The frozen no-reliable-inference conclusion (FROZEN §3.4)."""

    return (
        "No reliable inference: the significance method lacks "
        "calibrated-evidence qualification for this operating point "
        f"(reason codes: {', '.join(codes)}); descriptive statistics are "
        "retained, and no nominal-5% significance or promotion claim is "
        "made."
    )


def qualified_claim_reason(record: dict) -> str:
    """The frozen qualified conclusion (FROZEN §3.4) — only renderable
    when ``qualification.qualified`` is True."""

    stats = record.get("statistics") or {}
    task = record.get("task_actuals") or {}
    evidence = record.get("calibration_evidence") or {}
    p_value = stats.get("max_t_p_value")
    sha = str(evidence.get("sha256"))
    return (
        f"max-t p={p_value:.4f} is qualified for a nominal-5% "
        "significance claim at the calibrated operating point "
        f"(n={task.get('n_common_dates')} common dates, "
        f"K={task.get('n_pool_trials')} pooled trials, evidence "
        f"{evidence.get('source')} sha256 {sha[:8]})"
    )


# --- the evidence registry (object form, G1's RegimeRegistry precedent) --------

#: Sentinel: "use the registered entry" (distinct from an explicit None,
#: which means "no registration" => EVIDENCE_NOT_REGISTERED).
_DEFAULT_REGISTRATION = object()


def _normalized(path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _identity_rows(payload: dict) -> list[dict]:
    scenarios = (payload or {}).get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError(
            "calibration evidence payload carries no 'scenarios' mapping "
            "(wp17 W2: the identity rows cannot be registered)"
        )
    rows = []
    for name in sorted(scenarios):
        row = scenarios[name]
        if not isinstance(row, dict):
            raise ValueError(
                f"calibration evidence row {name!r} is not an object "
                "(wp17 W2)"
            )
        rows.append(
            {
                field: row[field]
                for field in IDENTITY_ROW_FIELDS
                if field in row
            }
        )
    return rows


class CalibrationEvidenceRegistry:
    """The registered calibration evidence, as an object (never a
    caller string).  Production loads via :meth:`from_file` — which
    always MEASURES the file's sha256 against the registered value at
    qualification time; tests may build in-memory registries via
    :meth:`from_payload` (the in-memory attestation defaults to the
    registration's hash unless ``measured_sha256`` says otherwise)."""

    def __init__(
        self,
        *,
        registration: dict | None,
        measured_sha256: str | None,
        rows: list[dict],
        source: str | None,
    ):
        self.registration = registration
        self.measured_sha256 = measured_sha256
        self.rows = rows
        self.source = source

    @classmethod
    def from_file(cls, path) -> "CalibrationEvidenceRegistry":
        file_path = Path(path)
        raw = file_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        registration = None
        for entry in CALIBRATION_EVIDENCE_REGISTRATION:
            if _normalized(entry["source"]) == _normalized(file_path):
                registration = dict(entry)
                break
        return cls(
            registration=registration,
            measured_sha256=hashlib.sha256(raw).hexdigest(),
            rows=_identity_rows(payload),
            source=str(file_path),
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        *,
        registration: Any = _DEFAULT_REGISTRATION,
        measured_sha256: str | None = None,
    ) -> "CalibrationEvidenceRegistry":
        """In-memory construction (tests / synthetic registries).

        ``registration`` defaults to the registered entry; an explicit
        ``None`` builds a registry that points at NO registered file
        (=> EVIDENCE_NOT_REGISTERED).  ``measured_sha256`` defaults to
        the registration's hash — an in-memory payload is its own
        attestation; production always measures real bytes via
        :meth:`from_file`."""

        if registration is _DEFAULT_REGISTRATION:
            registration = dict(CALIBRATION_EVIDENCE_REGISTRATION[0])
        measured = measured_sha256
        if measured is None:
            measured = (registration or {}).get("sha256")
        return cls(
            registration=registration,
            measured_sha256=measured,
            rows=_identity_rows(payload),
            source=(registration or {}).get("source"),
        )


def _auto_evidence() -> CalibrationEvidenceRegistry | None:
    """The then-available registry (FROZEN §1.1): the registered file,
    or None when unavailable — never a write blocker."""

    try:
        return CalibrationEvidenceRegistry.from_file(
            CALIBRATION_EVIDENCE_REGISTRATION[0]["source"]
        )
    except Exception:
        return None


_AUTO = object()


def build_claim_record(
    artifact_max_t: dict | None,
    artifact_dsr: dict | None,
    stitched_trials: list,
    max_t_perms: int,
    *,
    evidence: Any = _AUTO,
) -> dict:
    """Assemble the five field groups (FROZEN §1.1) — aggregation and
    registration only.  The qualification group is filled by
    :func:`qualify_claim_record` with the then-available evidence
    registry (``evidence`` defaults to the registered file; unavailable
    => qualified=False + EVIDENCE_NOT_REGISTERED, never a write
    blocker).  G2 recomputes the fifth group at promotion time
    (verify-don't-trust, FROZEN §3.1)."""

    mt = artifact_max_t or {}
    dsr_block = artifact_dsr or {}
    window = mt.get("window") or {}
    trial_set = mt.get("trial_set") or {}
    trials = [t for t in (stitched_trials or []) if isinstance(t, dict)]
    n_failed_folds = sum(
        len(t.get("failed_folds") or []) for t in trials
    )
    ev = _auto_evidence() if evidence is _AUTO else evidence
    registration = dict(CALIBRATION_EVIDENCE_REGISTRATION[0])
    rows_consulted = [dict(row) for row in ev.rows] if ev is not None else []
    record: dict[str, Any] = {
        # Group 1: the raw statistics and p-value.
        "statistics": {
            "dsr": dsr_block.get("dsr"),
            "max_t_observed": mt.get("observed_max_t"),
            "max_t_p_value": mt.get("p_value"),
            "max_t_significant_95": mt.get("significant_95"),
        },
        # Group 2: the method version and the parameters actually used.
        "method": {
            "max_t_method_version": MAX_T_METHOD_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "n_perms": mt.get("n_perms", max_t_perms),
            "block_size": mt.get("block_size"),
            "dsr_method": DSR_METHOD,
        },
        # Group 3: the task's actuals.
        "task_actuals": {
            "n_common_dates": window.get("n_dates"),
            "n_pool_trials": trial_set.get("n_trials"),
            "n_current_trials": trial_set.get("n_current"),
            "n_extra_trials": trial_set.get("n_extra"),
            "matrix_excluded": list(trial_set.get("matrix_excluded") or []),
            "n_failed_folds_total": int(n_failed_folds),
            "missing_fold_policy": MISSING_FOLD_POLICY,
        },
        # Group 4: the calibration evidence identity.
        "calibration_evidence": {
            "source": registration["source"],
            "sha256": registration["sha256"],
            "rows_consulted": rows_consulted,
            "method_version_attestation": METHOD_VERSION_ATTESTATION,
        },
    }
    record["qualification"] = qualify_claim_record(record, ev)
    return record


def qualify_claim_record(record: dict, evidence) -> dict:
    """The W2 algorithm (FROZEN §2): produce group five.  Fixed order,
    fail-closed; steps 1-2 terminate on failure, step 3's three domain
    checks all evaluate, then the covering-set conjunction (§2.4/§2.5).
    A caller-supplied regime string plays no part in the decision."""

    codes: list[str] = []
    covering: list[str] = []
    blocking: list[str] = []
    registration = evidence.registration if evidence is not None else None

    # Step 1: evidence consumability (FROZEN §2.1).
    if evidence is None or registration is None:
        codes.append(EVIDENCE_NOT_REGISTERED)
    elif evidence.measured_sha256 != registration.get("sha256"):
        codes.append(EVIDENCE_HASH_MISMATCH)
    else:
        # Step 2: method identity (FROZEN §2.2) — the task's
        # (max_t_method_version, n_perms, block_size) against the rows'
        # (n_perms, block_size) plus the registration's pinned version.
        method = record.get("method") or {}
        task = record.get("task_actuals") or {}
        rows_by_scenario = {
            row.get("scenario"): row for row in evidence.rows
        }
        ev_n_perms = {row.get("n_perms") for row in evidence.rows}
        ev_block = {row.get("block_size") for row in evidence.rows}
        if (
            method.get("max_t_method_version")
            != registration.get("max_t_method_version")
            or method.get("n_perms") not in ev_n_perms
            or method.get("block_size") not in ev_block
        ):
            codes.append(EVIDENCE_IDENTITY_MISMATCH)
        else:
            # Step 3: the coverage domain (FROZEN §2.3).
            n_dates = task.get("n_common_dates")
            k = task.get("n_pool_trials")
            if n_dates not in CALIBRATED_N_DATES:
                codes.append(N_DATES_OUT_OF_CALIBRATED_DOMAIN)
            if not (
                isinstance(k, int)
                and not isinstance(k, bool)
                and K_BAND[0] <= k <= K_BAND[1]
            ):
                codes.append(K_OUT_OF_CALIBRATED_DOMAIN)
            if task.get("missing_fold_policy") != MISSING_FOLD_POLICY:
                codes.append(MISSING_FOLD_POLICY_UNREGISTERED)
            # Step 4/5: the covering-set conjunction (FROZEN §2.4/§2.5).
            covering = list(COVERING_SCENARIOS_BY_N.get(n_dates, ()))
            for name in covering:
                row = rows_by_scenario.get(name)
                if row is None:
                    if SCENARIO_UNCALIBRATED not in codes:
                        codes.append(SCENARIO_UNCALIBRATED)
                    blocking.append(name)
                    continue
                verdict = row.get("verdict")
                if verdict == "insufficient_samples":
                    if SCENARIO_INSUFFICIENT_SAMPLES not in codes:
                        codes.append(SCENARIO_INSUFFICIENT_SAMPLES)
                    blocking.append(name)
                elif verdict == "unreliable_at_nominal":
                    if SCENARIO_UNRELIABLE_AT_NOMINAL not in codes:
                        codes.append(SCENARIO_UNRELIABLE_AT_NOMINAL)
                    blocking.append(name)
                elif verdict not in PASSING_VERDICTS:
                    # e.g. power_reported: not a size ruling (passthrough
                    # semantics preserved, test_p48:180-190).
                    if SCENARIO_UNCALIBRATED not in codes:
                        codes.append(SCENARIO_UNCALIBRATED)
                    blocking.append(name)

    qualified = not codes
    reason = (
        qualified_claim_reason(record)
        if qualified
        else no_reliable_inference_reason(codes)
    )
    return {
        "qualified": qualified,
        "status": "qualified" if qualified else "not_qualified",
        "reason_codes": codes,
        "reason": reason,
        "covering_scenarios": covering,
        "blocking_scenarios": blocking,
    }


def claim_record_consistency_reasons(record: dict, artifact: dict) -> list[str]:
    """FROZEN §3.1 step 2 (verify-don't-trust): the record's groups 1-3
    against the artifact's own dsr / max_t / stitched / matrix_excluded
    content, field by field.  Group 5 is NOT checked for consistency —
    it is recomputed by the caller instead."""

    mt = artifact.get("max_t") or {}
    dsr_block = artifact.get("dsr") or {}
    window = mt.get("window") or {}
    trial_set = mt.get("trial_set") or {}
    stitched = (artifact.get("stitched") or {}).get("trials") or []
    failed_total = sum(
        len(t.get("failed_folds") or [])
        for t in stitched
        if isinstance(t, dict)
    )
    statistics = record.get("statistics") or {}
    method = record.get("method") or {}
    task = record.get("task_actuals") or {}
    checks = (
        ("statistics.dsr", statistics.get("dsr"), dsr_block.get("dsr")),
        (
            "statistics.max_t_observed",
            statistics.get("max_t_observed"),
            mt.get("observed_max_t"),
        ),
        (
            "statistics.max_t_p_value",
            statistics.get("max_t_p_value"),
            mt.get("p_value"),
        ),
        (
            "statistics.max_t_significant_95",
            statistics.get("max_t_significant_95"),
            mt.get("significant_95"),
        ),
        ("method.n_perms", method.get("n_perms"), mt.get("n_perms")),
        ("method.block_size", method.get("block_size"), mt.get("block_size")),
        (
            "task_actuals.n_common_dates",
            task.get("n_common_dates"),
            window.get("n_dates"),
        ),
        (
            "task_actuals.n_pool_trials",
            task.get("n_pool_trials"),
            trial_set.get("n_trials"),
        ),
        (
            "task_actuals.n_current_trials",
            task.get("n_current_trials"),
            trial_set.get("n_current"),
        ),
        (
            "task_actuals.n_extra_trials",
            task.get("n_extra_trials"),
            trial_set.get("n_extra"),
        ),
        (
            "task_actuals.matrix_excluded",
            list(task.get("matrix_excluded") or []),
            list(trial_set.get("matrix_excluded") or []),
        ),
        (
            "task_actuals.n_failed_folds_total",
            task.get("n_failed_folds_total"),
            failed_total,
        ),
    )
    return [
        STRING_B_INCONSISTENT_TEMPLATE.format(
            field=field, recorded=recorded, actual=actual
        )
        for field, recorded, actual in checks
        if recorded != actual
    ]


def claim_record_stdout_lines(record) -> list[str]:
    """The evaluation-CLI qualification lines (FROZEN §3.4), inserted
    after the max_t line; ``[]`` when there is no record to report."""

    if not isinstance(record, dict):
        return []
    qualification = record.get("qualification")
    if not isinstance(qualification, dict):
        return []
    codes = list(qualification.get("reason_codes") or [])
    lines = [
        f"claim_record: qualified={qualification.get('qualified')} "
        f"reason_codes={codes}"
    ]
    if qualification.get("qualified") is False:
        lines.append(f"no reliable inference: {qualification.get('reason')}")
    return lines
