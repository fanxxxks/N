"""P55 WP17 claim-qualification RED/GREEN tests (WP 17).

Frozen spec: ``reports/wp17/FROZEN_17.md`` — the single source of truth
for W1-W6; every assertion below cites its section.  These tests pin,
before the implementation exists:

* W1 — the five claim-record field groups (§1.1) assembled by
  ``build_claim_record`` and carried by every ``build_result`` artifact
  (descriptive path never blocked, §3.2);
* W2 — the fail-closed qualification algorithm (§2): one test per
  reason code (§1.2 table), the qualified conjunction over the covering
  scenario set (§2.4/§2.5), and the ruling that a caller-supplied
  regime string never grants qualification (§1.2);
* W3 — G2's fourth layer (§3.1) with the frozen refusal strings A/B/C
  verbatim (§3.3): ``p=0.001`` but unqualified must not support
  promotion; the frozen evidence file blocks every operating point
  today (§2.6); the formal-conclusion templates (§3.4);
* W4 — the corrected positive calibration label, anchored on the
  [4.6%, 7.5%] numeric example (§4.2/§4.3);
* W5 — the task-scale power scenario registered but never run, the
  non-extrapolation note, and the frozen seed slices (§5.2/§5.3);
* W6 — the method-revision trigger guards G1-G5 verbatim with zero
  exemptions (§6.5).
"""

from __future__ import annotations

import copy
import functools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ashare_data.config import FoldConfig, ProtocolConfig, TierConfig
from ashare_model import evaluation
from ashare_model import promotion as promotion_module
from ashare_model.eval_corrections import MAX_T_METHOD_VERSION
from ashare_model.stat_calibration import (
    CALIBRATION_VERSION,
    SCENARIO_SPECS,
    calibrate_scenario,
    calibration_verdict,
    generate_scenario,
    scenario_seed_start,
    wilson_ci,
)

from ashare_model.claim_record import (  # noqa: E402  (RED: module absent)
    CALIBRATION_EVIDENCE_REGISTRATION,
    DSR_METHOD,
    METHOD_VERSION_ATTESTATION,
    MISSING_FOLD_POLICY,
    STRING_A_NO_CLAIM_RECORD,
    STRING_B_INCONSISTENT_TEMPLATE,
    STRING_C_UNQUALIFIED_TEMPLATE,
    CalibrationEvidenceRegistry,
    build_claim_record,
    claim_record_stdout_lines,
    no_reliable_inference_reason,
    qualify_claim_record,
    qualified_claim_reason,
)
from ashare_model.method_revision import (  # noqa: E402  (RED: module absent)
    G1_REFUSED,
    G2_REFUSED_TEMPLATE,
    G3_REFUSED_TEMPLATE,
    G4_REFUSED_TEMPLATE,
    G5_REFUSED_TEMPLATE,
    assert_method_version_endorsed,
    check_budget,
    open_episode,
    record_acceptance,
    register_method_version,
    require_open_episode,
)

from test_stitched_oos import _row

FIVE_GROUPS = {
    "statistics", "method", "task_actuals",
    "calibration_evidence", "qualification",
}

#: The covering scenario set for n=250 (FROZEN §2.4 table, row order).
COVERING_250 = [
    "independent_noise", "common_factor_095",
    "ar1_phi04", "duplicate_candidates",
]

FROZEN_EVIDENCE = Path(CALIBRATION_EVIDENCE_REGISTRATION[0]["source"])


# --- helpers -------------------------------------------------------------------


def _evidence_payload(verdicts=None, drop=()):
    """An in-memory calibration-report-shaped payload (identity rows only)."""

    merged = {
        "independent_noise": "calibrated",
        "common_factor_095": "calibrated",
        "ar1_phi04": "calibrated",
        "duplicate_candidates": "calibrated",
        "short_window_30d": "calibrated",
    }
    merged.update(verdicts or {})
    scenarios = {}
    for name, verdict in merged.items():
        if name in drop:
            continue
        scenarios[name] = {
            "scenario": name,
            "kind": "size",
            "n_perms": 999,
            "block_size": 10,
            "seed_start": 930_000,
            "reps": 1000,
            "verdict": verdict,
        }
    return {"scenarios": scenarios}


def _registry(
    payload=None,
    *,
    verdicts=None,
    drop=(),
    registration="default",
    measured_sha256=None,
):
    kwargs = {}
    if registration != "default":
        kwargs["registration"] = registration
    if measured_sha256 is not None:
        kwargs["measured_sha256"] = measured_sha256
    if payload is None:
        payload = _evidence_payload(verdicts=verdicts, drop=drop)
    return CalibrationEvidenceRegistry.from_payload(payload, **kwargs)


def _record(n_dates=250, k=8, n_perms=999, block_size=10, evidence="default"):
    """A claim record for one synthetic operating point (FROZEN §1.1)."""

    max_t = {
        "observed_max_t": 3.25,
        "p_value": 0.001,
        "significant_95": True,
        "n_perms": n_perms,
        "block_size": block_size,
        "window": {
            "first_date": "2026-01-04",
            "last_date": "2026-12-30",
            "n_dates": n_dates,
        },
        "trial_set": {
            "n_current": k, "n_extra": 0, "n_trials": k,
            "matrix_excluded": [],
        },
    }
    dsr = {"dsr": 0.97}
    ev = _registry() if evidence == "default" else evidence
    return build_claim_record(max_t, dsr, [], n_perms, evidence=ev)


@functools.lru_cache(maxsize=1)
def _in_domain_artifact_cached():
    """A real build_result artifact AT the calibrated operating point:
    8 current candidates (multiplicity pool), a 250-common-date stitched
    axis (2 folds x 125 sessions), max_t_perms=999 (FROZEN §7.2 row 1).
    The adjudicated challenger carries a planted edge => p=0.001."""

    from ashare_model.evaluation import build_result

    rng = np.random.default_rng(7)
    folds = [("2026-01-01", "2026-06-30"), ("2026-06-30", "2026-12-31")]
    rows = []
    for train_end, test_end in folds:
        daily = [0.004 + x for x in rng.normal(0.0, 0.001, 125)]
        rows.append(
            _row(
                "challenger", 42, train_end, test_end,
                daily=daily, formula_text="RET_1", eligible=True,
                rejection_reasons=[], direction=1, formula=[1],
                average_turnover=0.05, capacity_utilization=0.05,
            )
        )
    for i in range(7):  # the multiplicity pool: pure-noise trials
        for train_end, test_end in folds:
            daily = [x for x in rng.normal(0.0, 0.001, 125)]
            rows.append(
                _row(
                    f"trained_pool_{i}", 100 + i, train_end, test_end,
                    daily=daily, formula_text="RET_1", eligible=True,
                    rejection_reasons=[], direction=1, formula=[1],
                    average_turnover=0.05, capacity_utilization=0.05,
                )
            )
    proto = ProtocolConfig(
        folds=[FoldConfig(a, b) for a, b in folds],
        seeds=[42],
        baseline_signals=[],
        random_samples=0,
    )
    return build_result(
        proto, "confirmation", TierConfig(50, 256), rows,
        spec_id="a" * 64, run_id="b" * 32,
        data_end_date="2026-12-31", dataset_id="ds-current",
        max_t_perms=999,
    )


def _in_domain_artifact():
    return copy.deepcopy(_in_domain_artifact_cached())


def _default_point_artifact():
    """A default production artifact (k=1, n=500, n_perms=5000): the
    operating point NO registered evidence covers (FROZEN §2.2 — the
    honest default state, not a defect)."""

    from ashare_model.evaluation import build_result

    rng = np.random.default_rng(9)
    rows = []
    for train_end, test_end in [
        ("2026-01-01", "2026-06-30"), ("2026-06-30", "2026-12-31"),
    ]:
        daily = [0.004 + x for x in rng.normal(0.0, 0.001, 250)]
        rows.append(
            _row(
                "challenger", 42, train_end, test_end,
                daily=daily, formula_text="RET_1", eligible=True,
                rejection_reasons=[], direction=1, formula=[1],
                average_turnover=0.05, capacity_utilization=0.05,
            )
        )
    proto = ProtocolConfig(
        folds=[
            FoldConfig("2026-01-01", "2026-06-30"),
            FoldConfig("2026-06-30", "2026-12-31"),
        ],
        seeds=[42],
        baseline_signals=[],
        random_samples=0,
    )
    return build_result(
        proto, "confirmation", TierConfig(50, 256), rows,
        spec_id="a" * 64, run_id="b" * 32,
        data_end_date="2026-12-31", dataset_id="ds-current",
    )


def _g2(artifact, evidence=None):
    verdict = promotion_module.evaluate_challenger(
        artifact, calibration_evidence=evidence
    )
    return verdict["gates"]["significance"], verdict


# --- W1: the five field groups --------------------------------------------------


def test_w1_five_groups_structure():
    record = _record()
    assert set(record) == FIVE_GROUPS
    # Group 1: the raw statistics (FROZEN §1.1).
    assert set(record["statistics"]) == {
        "dsr", "max_t_observed", "max_t_p_value", "max_t_significant_95",
    }
    assert record["statistics"] == {
        "dsr": 0.97,
        "max_t_observed": 3.25,
        "max_t_p_value": 0.001,
        "max_t_significant_95": True,
    }
    # Group 2: method version and parameters.
    assert set(record["method"]) == {
        "max_t_method_version", "calibration_version",
        "n_perms", "block_size", "dsr_method",
    }
    assert record["method"]["max_t_method_version"] == 1
    assert record["method"]["max_t_method_version"] == MAX_T_METHOD_VERSION
    assert record["method"]["calibration_version"] == CALIBRATION_VERSION == 1
    assert record["method"]["n_perms"] == 999
    assert record["method"]["block_size"] == 10
    assert record["method"]["dsr_method"] == DSR_METHOD
    assert DSR_METHOD == "analytic_bldp_uncalibrated_v1"
    # Group 3: the task's actuals.
    assert set(record["task_actuals"]) == {
        "n_common_dates", "n_pool_trials", "n_current_trials",
        "n_extra_trials", "matrix_excluded", "n_failed_folds_total",
        "missing_fold_policy",
    }
    assert record["task_actuals"]["n_common_dates"] == 250
    assert record["task_actuals"]["n_pool_trials"] == 8
    assert record["task_actuals"]["n_current_trials"] == 8
    assert record["task_actuals"]["n_extra_trials"] == 0
    assert record["task_actuals"]["matrix_excluded"] == []
    assert record["task_actuals"]["n_failed_folds_total"] == 0
    assert (
        record["task_actuals"]["missing_fold_policy"] == MISSING_FOLD_POLICY
    )
    assert MISSING_FOLD_POLICY == (
        "structural_no_imputation_no_zero_padding_v1"
    )
    # Group 4: the calibration evidence identity.
    assert set(record["calibration_evidence"]) == {
        "source", "sha256", "rows_consulted", "method_version_attestation",
    }
    entry = CALIBRATION_EVIDENCE_REGISTRATION[0]
    assert record["calibration_evidence"]["source"] == entry["source"]
    assert record["calibration_evidence"]["sha256"] == entry["sha256"]
    for row in record["calibration_evidence"]["rows_consulted"]:
        # identity fields only — the historical claim strings are NOT
        # carried into new reports (FROZEN §4.4).
        assert set(row) == {
            "scenario", "kind", "n_perms", "block_size",
            "seed_start", "reps", "verdict",
        }
    # Group 5: the qualification.
    assert set(record["qualification"]) == {
        "qualified", "status", "reason_codes", "reason",
        "covering_scenarios", "blocking_scenarios",
    }


def test_w1_registration_entry_and_attestation_frozen():
    entry = CALIBRATION_EVIDENCE_REGISTRATION[0]
    assert entry["source"] == "D:/minequant/reports/p48/calibration_report.json"
    assert entry["sha256"] == (
        "b85a652705f8768fe5629473de5aac7841b575d2ff46617388e6a49c1b296d0e"
    )
    assert entry["max_t_method_version"] == 1
    assert METHOD_VERSION_ATTESTATION == (
        "max_t_method_version=1 is attested, not retro-invented: the "
        "evidence file predates versioning and carries no version field; "
        "the mapping is declared forward by "
        "docs/p48_stat_calibration_measurement_log.md "
        "(统计实现 eval_corrections 零改动) and the git history examined "
        "in reports/wp17/_recon/map.md §2."
    )


def test_w1_failed_folds_counted_across_stitched_trials():
    max_t = {
        "observed_max_t": 1.0, "p_value": 0.02, "significant_95": True,
        "n_perms": 999, "block_size": 10,
        "window": {"first_date": "a", "last_date": "b", "n_dates": 250},
        "trial_set": {"n_current": 4, "n_extra": 0, "n_trials": 4,
                      "matrix_excluded": []},
    }
    stitched = [
        {"failed_folds": [{"fold_test_end": "x"}, {"fold_test_end": "y"}]},
        {"failed_folds": []},
        {"failed_folds": [{"fold_test_end": "z"}]},
    ]
    record = build_claim_record(
        max_t, {"dsr": 0.99}, stitched, 999, evidence=_registry()
    )
    assert record["task_actuals"]["n_failed_folds_total"] == 3


def test_w1_w3_descriptive_artifact_never_blocked_by_qualification():
    # FROZEN §3.2: build_result fills all five groups unconditionally;
    # qualified=False (with reasons) never blocks the save.
    artifact = _default_point_artifact()
    cr = artifact["claim_record"]
    assert set(cr) == FIVE_GROUPS
    q = cr["qualification"]
    assert q["qualified"] is False
    assert q["status"] == "not_qualified"
    assert q["reason_codes"]
    if FROZEN_EVIDENCE.exists():
        # 5000 != 999: the identity step fires and terminates (§2.2).
        assert q["reason_codes"] == ["EVIDENCE_IDENTITY_MISMATCH"]
    else:
        assert q["reason_codes"] == ["EVIDENCE_NOT_REGISTERED"]
    # The first four groups are fully populated regardless.
    assert cr["statistics"]["max_t_p_value"] == artifact["max_t"]["p_value"]
    assert cr["task_actuals"]["n_common_dates"] == 500
    assert cr["task_actuals"]["n_pool_trials"] == 1


def test_w1_in_domain_artifact_carries_honest_record():
    artifact = _in_domain_artifact()
    cr = artifact["claim_record"]
    assert set(cr) == FIVE_GROUPS
    assert cr["task_actuals"]["n_common_dates"] == 250
    assert cr["task_actuals"]["n_pool_trials"] == 8
    assert cr["method"]["n_perms"] == 999
    assert cr["method"]["block_size"] == 10
    mt = artifact["max_t"]
    assert mt["p_value"] == pytest.approx(0.001)  # p=0.001, significant
    assert mt["significant_95"] is True


# --- W2: the fail-closed qualification algorithm --------------------------------


def test_w2_qualified_conjunction_over_covering_scenarios():
    q = qualify_claim_record(_record(), _registry())
    assert q["qualified"] is True
    assert q["status"] == "qualified"
    assert q["reason_codes"] == []
    assert q["covering_scenarios"] == COVERING_250
    assert q["blocking_scenarios"] == []
    assert q["reason"] == qualified_claim_reason(_record())


def test_w2_conservative_scenarios_also_qualify():
    payload = _evidence_payload(verdicts={
        name: "conservative" for name in COVERING_250
    })
    q = qualify_claim_record(_record(), _registry(payload))
    assert q["qualified"] is True  # B5 semantics (FROZEN §2.5)


def test_w2_n30_uses_the_short_window_covering_set():
    q = qualify_claim_record(_record(n_dates=30, k=4), _registry())
    assert q["qualified"] is True
    assert q["covering_scenarios"] == ["short_window_30d"]


def test_w2_evidence_not_registered():
    for bad in (None, _registry(registration=None)):
        q = qualify_claim_record(_record(), bad)
        assert q["qualified"] is False
        assert q["reason_codes"] == ["EVIDENCE_NOT_REGISTERED"]


def test_w2_evidence_hash_mismatch_terminates():
    q = qualify_claim_record(_record(), _registry(measured_sha256="0" * 64))
    assert q["qualified"] is False
    assert q["reason_codes"] == ["EVIDENCE_HASH_MISMATCH"]


def test_w2_identity_mismatch_n_perms():
    q = qualify_claim_record(_record(n_perms=5000), _registry())
    assert q["qualified"] is False
    assert q["reason_codes"] == ["EVIDENCE_IDENTITY_MISMATCH"]


def test_w2_identity_mismatch_block_size():
    q = qualify_claim_record(_record(block_size=20), _registry())
    assert q["qualified"] is False
    assert q["reason_codes"] == ["EVIDENCE_IDENTITY_MISMATCH"]


@pytest.mark.parametrize("n_dates", [180, 400, 31, 249])
def test_w2_n_dates_out_of_domain_is_point_membership(n_dates):
    # FROZEN §2.3: point-set membership in {30, 250} — interpolation and
    # extrapolation are both extrapolation of the evidence.
    q = qualify_claim_record(_record(n_dates=n_dates), _registry())
    assert q["qualified"] is False
    assert q["reason_codes"] == ["N_DATES_OUT_OF_CALIBRATED_DOMAIN"]


@pytest.mark.parametrize("k", [2, 12])
def test_w2_k_out_of_domain(k):
    q = qualify_claim_record(_record(k=k), _registry())
    assert q["qualified"] is False
    assert "K_OUT_OF_CALIBRATED_DOMAIN" in q["reason_codes"]


def test_w2_missing_fold_policy_unregistered():
    record = _record()
    record["task_actuals"]["missing_fold_policy"] = "zero_padding_v0"
    q = qualify_claim_record(record, _registry())
    assert q["qualified"] is False
    assert "MISSING_FOLD_POLICY_UNREGISTERED" in q["reason_codes"]


def test_w2_scenario_uncalibrated():
    q = qualify_claim_record(_record(), _registry(drop={"ar1_phi04"}))
    assert q["qualified"] is False
    assert q["reason_codes"] == ["SCENARIO_UNCALIBRATED"]
    assert q["blocking_scenarios"] == ["ar1_phi04"]
    assert q["covering_scenarios"] == COVERING_250


def test_w2_scenario_insufficient_samples():
    q = qualify_claim_record(_record(), _registry(verdicts={
        "short_window_30d": "insufficient_samples",
        "ar1_phi04": "insufficient_samples",
    }))
    assert q["qualified"] is False
    assert q["reason_codes"] == ["SCENARIO_INSUFFICIENT_SAMPLES"]
    assert q["blocking_scenarios"] == ["ar1_phi04"]


def test_w2_scenario_unreliable_at_nominal():
    q = qualify_claim_record(_record(), _registry(verdicts={
        "ar1_phi04": "unreliable_at_nominal",
    }))
    assert q["qualified"] is False
    assert q["reason_codes"] == ["SCENARIO_UNRELIABLE_AT_NOMINAL"]
    assert q["blocking_scenarios"] == ["ar1_phi04"]


def test_w2_power_rows_never_enter_the_covering_set():
    payload = _evidence_payload()
    payload["scenarios"]["power_sr05"] = {
        "scenario": "power_sr05", "kind": "power", "n_perms": 999,
        "block_size": 10, "seed_start": 950_000, "reps": 400,
        "verdict": "power_reported",
    }
    q = qualify_claim_record(_record(), _registry(payload))
    assert q["qualified"] is True
    assert q["covering_scenarios"] == COVERING_250  # power is not size


def test_w2_caller_regime_string_never_grants_qualification():
    record = _record(n_dates=180)
    record["task_actuals"]["regime"] = "iid"  # a caller-supplied string
    q = qualify_claim_record(record, _registry())
    assert q["qualified"] is False
    assert q["reason_codes"] == ["N_DATES_OUT_OF_CALIBRATED_DOMAIN"]
    # The assembled record itself carries no regime key anywhere.
    clean = _record()
    assert "regime" not in json.dumps(clean)


@pytest.mark.skipif(
    not FROZEN_EVIDENCE.exists(),
    reason="registered calibration evidence not present on this machine",
)
class TestFrozenEvidenceBlocksEveryOperatingPoint:
    """FROZEN §2.6: with the registered evidence, every task is
    not-qualified — the machine-executable form of the measurement
    log's frozen-inference ruling."""

    @classmethod
    def _registry(cls):
        return CalibrationEvidenceRegistry.from_file(FROZEN_EVIDENCE)

    def test_measured_sha_matches_registration(self):
        registry = self._registry()
        assert registry.measured_sha256 == (
            CALIBRATION_EVIDENCE_REGISTRATION[0]["sha256"]
        )

    def test_n250_blocked_by_ar1(self):
        q = qualify_claim_record(_record(), self._registry())
        assert q["qualified"] is False
        assert q["reason_codes"] == ["SCENARIO_UNRELIABLE_AT_NOMINAL"]
        assert q["blocking_scenarios"] == ["ar1_phi04"]

    def test_n30_blocked_by_short_window(self):
        q = qualify_claim_record(
            _record(n_dates=30, k=4), self._registry()
        )
        assert q["qualified"] is False
        assert q["reason_codes"] == ["SCENARIO_UNRELIABLE_AT_NOMINAL"]
        assert q["blocking_scenarios"] == ["short_window_30d"]


# --- W3: G2's fourth layer and the formal-conclusion surface ---------------------


def test_w3_g2_passes_when_qualified():
    artifact = _in_domain_artifact()
    gate, verdict = _g2(artifact, evidence=_registry())
    assert gate["passed"], gate["reasons"]
    assert verdict["qualification"]["qualified"] is True
    assert verdict["qualification"]["reason_codes"] == []
    assert verdict["qualification"]["covering_scenarios"] == COVERING_250
    assert verdict["claim_record"] == artifact["claim_record"]


def test_w3_g2_refuses_p_0001_when_no_evidence_registered():
    # FROZEN §3.1: three layers pass, the fourth fails => G2 FAILS.
    artifact = _in_domain_artifact()
    assert artifact["max_t"]["p_value"] == pytest.approx(0.001)
    gate, verdict = _g2(artifact, evidence=None)
    assert not gate["passed"]
    expected = STRING_C_UNQUALIFIED_TEMPLATE.format(
        n=250, k=8, np=999, bs=10,
        policy=MISSING_FOLD_POLICY,
        codes="EVIDENCE_NOT_REGISTERED",
    )
    assert gate["reasons"] == [expected]
    assert verdict["promoted"] is False
    assert verdict["qualification"]["qualified"] is False
    assert verdict["qualification"]["reason_codes"] == [
        "EVIDENCE_NOT_REGISTERED"
    ]


def test_w3_g2_refuses_p_0001_when_evidence_blocks_the_point():
    artifact = _in_domain_artifact()
    blocking = _registry(verdicts={"ar1_phi04": "unreliable_at_nominal"})
    gate, verdict = _g2(artifact, evidence=blocking)
    assert not gate["passed"]
    expected = STRING_C_UNQUALIFIED_TEMPLATE.format(
        n=250, k=8, np=999, bs=10,
        policy=MISSING_FOLD_POLICY,
        codes="SCENARIO_UNRELIABLE_AT_NOMINAL",
    )
    assert gate["reasons"] == [expected]
    assert verdict["promoted"] is False
    assert verdict["qualification"]["blocking_scenarios"] == ["ar1_phi04"]


def test_w3_string_a_verbatim():
    assert STRING_A_NO_CLAIM_RECORD == (
        "artifact carries no claim record (wp17 W1): pre-wp17 artifacts "
        "cannot support promotion - the five field groups (statistics, "
        "method, task actuals, calibration evidence identity, "
        "qualification) are required"
    )
    artifact = _in_domain_artifact()
    artifact.pop("claim_record", None)
    gate, _ = _g2(artifact, evidence=_registry())
    assert not gate["passed"]
    assert gate["reasons"] == [STRING_A_NO_CLAIM_RECORD]


def test_w3_string_b_verbatim():
    artifact = _in_domain_artifact()
    artifact["claim_record"]["statistics"]["max_t_p_value"] = 0.5
    gate, _ = _g2(artifact, evidence=_registry())
    assert not gate["passed"]
    expected = STRING_B_INCONSISTENT_TEMPLATE.format(
        field="statistics.max_t_p_value",
        recorded=0.5,
        actual=artifact["max_t"]["p_value"],
    )
    assert expected in gate["reasons"]
    assert expected == (
        "claim record is inconsistent with the artifact's own statistical "
        "blocks (wp17 W1): statistics.max_t_p_value records 0.5 but the "
        "artifact carries 0.001"
    )


def test_w3_string_b_catches_task_actual_tampering():
    artifact = _in_domain_artifact()
    artifact["claim_record"]["task_actuals"]["n_common_dates"] = 30
    gate, _ = _g2(artifact, evidence=_registry())
    assert not gate["passed"]
    expected = STRING_B_INCONSISTENT_TEMPLATE.format(
        field="task_actuals.n_common_dates",
        recorded=30,
        actual=artifact["max_t"]["window"]["n_dates"],
    )
    assert expected in gate["reasons"]


def test_w3_artifacts_own_qualification_flag_is_not_trusted():
    # FROZEN §3.1 step 2: the artifact's own qualification.qualified
    # boolean is never trusted — G2 recomputes from the evidence object.
    artifact = _in_domain_artifact()
    artifact["claim_record"]["qualification"] = {
        "qualified": True, "status": "qualified", "reason_codes": [],
        "reason": "tampered", "covering_scenarios": [],
        "blocking_scenarios": [],
    }
    gate, _ = _g2(artifact, evidence=None)
    assert not gate["passed"]
    assert any(
        STRING_C_UNQUALIFIED_TEMPLATE.startswith(r.split(" (n_common")[0])
        for r in gate["reasons"]
    )


def test_w3_frozen_conclusion_templates():
    assert no_reliable_inference_reason(["SCENARIO_UNRELIABLE_AT_NOMINAL"]) == (
        "No reliable inference: the significance method lacks "
        "calibrated-evidence qualification for this operating point "
        "(reason codes: SCENARIO_UNRELIABLE_AT_NOMINAL); descriptive "
        "statistics are retained, and no nominal-5% significance or "
        "promotion claim is made."
    )
    record = _record()
    assert qualified_claim_reason(record) == (
        "max-t p=0.0010 is qualified for a nominal-5% significance claim "
        "at the calibrated operating point (n=250 common dates, K=8 "
        "pooled trials, evidence D:/minequant/reports/p48/"
        "calibration_report.json sha256 b85a6527)"
    )


def test_w3_stdout_lines_frozen():
    record = _record()
    reason = no_reliable_inference_reason(["SCENARIO_UNRELIABLE_AT_NOMINAL"])
    record["qualification"] = {
        "qualified": False,
        "status": "not_qualified",
        "reason_codes": ["SCENARIO_UNRELIABLE_AT_NOMINAL"],
        "reason": reason,
        "covering_scenarios": COVERING_250,
        "blocking_scenarios": ["ar1_phi04"],
    }
    lines = claim_record_stdout_lines(record)
    assert lines == [
        "claim_record: qualified=False "
        "reason_codes=['SCENARIO_UNRELIABLE_AT_NOMINAL']",
        f"no reliable inference: {reason}",
    ]
    assert claim_record_stdout_lines(None) == []


def test_w3_cli_wiring_surfaces_exist():
    promotion_src = Path(promotion_module.__file__).read_text(encoding="utf-8")
    assert "--calibration-evidence" in promotion_src
    evaluation_src = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "claim_record_stdout_lines" in evaluation_src


# --- W4: the corrected positive calibration label -------------------------------


def test_w4_positive_label_numeric_example():
    v = calibration_verdict(rejections=59, reps=1000, alpha=0.05)
    assert v["verdict"] == "calibrated"
    assert v["ci_low"] == pytest.approx(0.0460143, rel=1e-6)
    assert v["ci_high"] == pytest.approx(0.0753609, rel=1e-6)
    assert v["ci_low"] < 0.05 < v["ci_high"]
    assert v["ci_high"] > 0.05  # the interval admits rates ABOVE 5%
    assert v["claim"] == (
        "No significant departure from the nominal 5% level detected: "
        "rate=0.059, 95% CI [0.046, 0.075] contains 5% — supports "
        "non-rejection only; not a verification that the false-positive "
        "rate is at most 5% (an interval containing 5% also admits rates "
        "above 5%)."
    )


def test_w4_no_overclaiming_wording_anywhere():
    claim = calibration_verdict(59, 1000)["claim"]
    assert "size supported" not in claim  # the old over-strong label is gone
    # "at most 5%" survives only inside the negated clause.
    assert claim.count("at most") == 1
    assert "not a verification that the false-positive rate is at most 5%" \
        in claim


def test_w4_other_three_branches_unchanged_verbatim():
    low, high = wilson_ci(15, 1000)
    assert calibration_verdict(15, 1000)["claim"] == (
        f"Size below nominal (CI [{low:.3f}, {high:.3f}] entirely under "
        "0.05): 5% claims are valid and conservative; power is reduced."
    )
    low, high = wilson_ci(120, 1000)
    assert calibration_verdict(120, 1000)["claim"] == (
        f"NO 5% significance claim: measured size 0.120, 95% CI "
        f"[{low:.3f}, {high:.3f}] lies entirely above 0.05. The honest "
        "statement is the measured effective size; fixing the method "
        "requires a new contract and a fresh disjoint seed block."
    )
    v = calibration_verdict(30, 400)
    assert v["verdict"] == "insufficient_samples"
    width = v["ci_high"] - v["ci_low"]
    assert v["claim"] == (
        "No reliable inference: fewer than 500 repetitions (or CI width "
        f"{width:.3f} > 0.04) — the measured rate cannot bound the size."
    )


# --- W5: the task-scale power scenario (registered, never run) --------------------


FROZEN_NOTE = (
    "power_sr05/power_sr025 are NOT extrapolable to the task scale: daily "
    "Sharpe 0.5/0.25 annualize to 7.94/3.97 (x sqrt(252)), i.e. 126%/63% "
    "annualized relative excess at scale=0.01 - 126x/63x the task-scale "
    "edge (<1% annualized relative excess; task_scale_excess_1pct plants "
    "sr=1/252 ~= 0.00397, annualized Sharpe 1/sqrt(252) ~= 0.063). "
    "Detection rates measured at 0.5/0.25 say nothing about detection at "
    "the task scale; no power claim is made at the task scale until "
    "task_scale_excess_1pct is calibrated under the wp17 W6 protocol."
)


def test_w5_task_scale_scenario_registered_not_run():
    spec = SCENARIO_SPECS["task_scale_excess_1pct"]
    assert spec["kind"] == "power"
    assert spec["k_trials"] == 8
    assert spec["n_dates"] == 250
    assert spec["sr"] == 1.0 / 252.0  # the exact rational, FROZEN §5.1
    assert spec["scale"] == 0.01
    assert spec["task_scale"] is True
    assert spec["note"] == FROZEN_NOTE


def test_w5_generator_plants_the_task_scale_edge():
    # Same first draw as independent_noise (same seed, same shape); the
    # power branch adds sr*scale to the planted row — deterministic check.
    rng_a = np.random.default_rng(11)
    task = generate_scenario("task_scale_excess_1pct", rng_a)
    rng_b = np.random.default_rng(11)
    base = 0.01 * rng_b.normal(0.0, 1.0, size=(8, 250))
    expected = base.copy()
    expected[0] = expected[0] + (1.0 / 252.0) * 0.01
    np.testing.assert_allclose(task, expected)


def test_w5_frozen_seed_slices_unchanged():
    # The seven p48 scenarios keep their registered slices; the new
    # scenario's name sorts AFTER short_window_30d so it takes the next
    # slice (FROZEN §5.2 naming constraint).
    assert scenario_seed_start("ar1_phi04") == 900_000
    assert scenario_seed_start("common_factor_095") == 910_000
    assert scenario_seed_start("duplicate_candidates") == 920_000
    assert scenario_seed_start("independent_noise") == 930_000
    assert scenario_seed_start("power_sr025") == 940_000
    assert scenario_seed_start("power_sr05") == 950_000
    assert scenario_seed_start("short_window_30d") == 960_000  # NOT pushed
    assert scenario_seed_start("task_scale_excess_1pct") == 970_000


def test_w5_scale_gap_arithmetic_frozen():
    # FROZEN §5.1: closed-form ratios 0.5*252=126, 0.25*252=63.
    assert 0.5 * 252 == 126
    assert 0.25 * 252 == 63
    assert math.isclose(0.5 * math.sqrt(252), 7.9373, abs_tol=1e-4)
    assert math.isclose(0.25 * math.sqrt(252), 3.9686, abs_tol=1e-4)
    assert math.isclose((1.0 / 252.0) * math.sqrt(252), 0.062994,
                        abs_tol=1e-6)
    assert (0.01 / 252.0) / 0.01 == pytest.approx(1.0 / 252.0)


# --- W6: the method-revision trigger guards (zero exemptions) --------------------


def _episode(
    episode_id="ep-1",
    versions=None,
    acceptance=None,
    status="open",
    purpose="method_revision",
    frozen_budget_seconds=3600,
    blocking_reason_codes=("SCENARIO_UNRELIABLE_AT_NOMINAL",),
):
    return {
        "episode_id": episode_id,
        "purpose": purpose,
        "opened_utc": "2026-09-23T00:00:00+00:00",
        "artifact": {
            "run_id": "b" * 32,
            "spec_id": "a" * 64,
            "candidate": "trained",
            "seed": 42,
            "artifact_sha256": "f" * 64,
        },
        "blocking_reason_codes": list(blocking_reason_codes),
        "target_operating_points": [
            {
                "n_dates": 250,
                "k_trials": 8,
                "n_perms": 999,
                "block_size": 10,
                "correlation_forms": list(COVERING_250),
            }
        ],
        "versions": list(versions or []),
        "frozen_budget_seconds": frozen_budget_seconds,
        "pilot_seeds": [500_000],
        "acceptance": acceptance,
        "status": status,
    }


def _accepted(version_id="v2", start=1_000_000, span=10_000):
    return {
        "version_id": version_id,
        "seed_block_start": start,
        "span": span,
        "reps": 1000,
        "verdict": "calibrated",
        "closed_utc": "2026-09-23T01:00:00+00:00",
    }


def _registry_file(path, episodes):
    path.write_text(
        json.dumps({"episodes": list(episodes)}), encoding="utf-8"
    )
    return path


def test_w6_g1_verbatim_and_default_refusal(tmp_path):
    assert G1_REFUSED == (
        "method revision refused: no open trigger record exists for this "
        "work - register the candidate worth formal testing (artifact "
        "identity + blocking reason codes) before any method-version "
        "comparison or acceptance calibration starts (wp17 W6)"
    )
    path = _registry_file(tmp_path / "reg.json", [_episode()])
    with pytest.raises(ValueError) as ei:
        require_open_episode(None, path=path)  # default: fail-closed
    assert str(ei.value) == G1_REFUSED


def test_w6_g1_calibrate_scenario_guarded_by_default():
    with pytest.raises(ValueError) as ei:
        calibrate_scenario("short_window_30d", reps=1)
    assert str(ei.value) == G1_REFUSED


def test_w6_g1_open_episode_token_accepted(tmp_path):
    path = _registry_file(tmp_path / "reg.json", [_episode()])
    episode = require_open_episode("ep-1", path=path)
    assert episode["status"] == "open"


def test_w6_g1_terminal_or_unknown_tokens_refused(tmp_path):
    path = _registry_file(
        tmp_path / "reg.json",
        [_episode(status="rejected"), _episode(episode_id="ep-2")],
    )
    for token in ("ep-1", "ep-3", "", "*"):
        with pytest.raises(ValueError) as ei:
            require_open_episode(token, path=path)
        assert str(ei.value) == G1_REFUSED


def test_w6_g2_third_method_version_refused(tmp_path):
    path = _registry_file(tmp_path / "reg.json", [_episode(versions=["v1"])])
    register_method_version("ep-1", "v2", path=path)  # second: allowed
    with pytest.raises(ValueError) as ei:
        register_method_version("ep-1", "v3", path=path)
    expected = G2_REFUSED_TEMPLATE.format(episode_id="ep-1", n=2)
    assert str(ei.value) == expected
    assert expected == (
        "method revision refused: episode ep-1 already lists 2 method "
        "versions; at most two may be compared per episode (wp17 W6)"
    )


def test_w6_g3_second_acceptance_refused(tmp_path):
    path = _registry_file(
        tmp_path / "reg.json",
        [_episode(versions=["v1", "v2"], acceptance=_accepted(),
                  status="rejected")],
    )
    with pytest.raises(ValueError) as ei:
        record_acceptance(
            "ep-1", version_id="v2", seed_block_start=2_000_000,
            span=10_000, reps=1000, verdict="rejected", path=path,
        )
    expected = G3_REFUSED_TEMPLATE.format(episode_id="ep-1")
    assert str(ei.value) == expected
    assert expected.endswith(
        "has already run its single independent-seed acceptance "
        "calibration; a failed or insufficient acceptance keeps "
        "'no reliable inference' - no re-runs and no additional seeds "
        "(wp17 W6)"
    )


def test_w6_g4_budget_exceeded_refused(tmp_path):
    path = _registry_file(
        tmp_path / "reg.json", [_episode(frozen_budget_seconds=3600)]
    )
    check_budget("ep-1", 3600.0, path=path)  # at the budget: allowed
    with pytest.raises(ValueError) as ei:
        check_budget("ep-1", 3601.0, path=path)
    expected = G4_REFUSED_TEMPLATE.format(
        episode_id="ep-1", budget_seconds=3600
    )
    assert str(ei.value) == expected
    assert expected == (
        "method revision refused: episode ep-1 exceeded its frozen budget "
        "of 3600s; the honest state remains 'no reliable inference' "
        "(wp17 W6)"
    )


@pytest.mark.parametrize(
    "start,span",
    [
        (10_000, 240),      # p40 tuning block
        (45_000, 10),       # p40 tuning block
        (900_000, 100_000), # p48 validation block
        (950_000, 2_000),   # p48 validation block
        (1_000_000, 500),   # prior episode's acceptance block
    ],
)
def test_w6_g5_overlapping_seed_blocks_refused(tmp_path, start, span):
    prior = _episode(
        episode_id="ep-0", status="rejected", acceptance=_accepted()
    )
    path = _registry_file(
        tmp_path / "reg.json", [prior, _episode(versions=["v1"])]
    )
    with pytest.raises(ValueError) as ei:
        record_acceptance(
            "ep-1", version_id="v1", seed_block_start=start, span=span,
            reps=1000, verdict="calibrated", path=path,
        )
    expected = G5_REFUSED_TEMPLATE.format(start=start, span=span)
    assert str(ei.value) == expected
    assert expected.startswith(
        "method revision refused: the requested seed block "
        f"[{start},{span}) overlaps a used block"
    )


def test_w6_g5_fresh_disjoint_block_accepted(tmp_path):
    prior = _episode(
        episode_id="ep-0", status="rejected", acceptance=_accepted()
    )
    path = _registry_file(
        tmp_path / "reg.json", [prior, _episode(versions=["v1"])]
    )
    record_acceptance(
        "ep-1", version_id="v1", seed_block_start=2_000_000, span=10_000,
        reps=1000, verdict="calibrated", path=path,
    )
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    ep1 = next(e for e in reloaded["episodes"] if e["episode_id"] == "ep-1")
    assert ep1["acceptance"]["seed_block_start"] == 2_000_000


def test_w6_open_episode_validates_and_appends(tmp_path):
    path = tmp_path / "reg.json"
    open_episode(_episode(), path=path)
    with pytest.raises(ValueError):
        open_episode(_episode(), path=path)  # duplicate id: append-only
    bad = _episode(blocking_reason_codes=[])
    with pytest.raises(ValueError):
        # (c) FROZEN §6.1: the trigger must attribute the G2 failure to
        # qualification reason codes.
        open_episode(bad, path=tmp_path / "reg2.json")


def test_w6_registry_schema_revalidated_on_read(tmp_path):
    path = tmp_path / "reg.json"
    _registry_file(path, [_episode(frozen_budget_seconds=-1)])
    with pytest.raises(ValueError):
        require_open_episode("ep-1", path=path)


def test_w6_method_constant_turnover_requires_accepted_episode(tmp_path):
    assert MAX_T_METHOD_VERSION == 1  # v1: attested forward (§1.3)
    empty = _registry_file(tmp_path / "empty.json", [])
    assert_method_version_endorsed(1, path=empty)
    with pytest.raises(ValueError):
        assert_method_version_endorsed(2, path=empty)
    accepted = _registry_file(
        tmp_path / "accepted.json",
        [_episode(status="accepted", acceptance=_accepted())],
    )
    assert_method_version_endorsed(2, path=accepted)


def test_w6_zero_exemptions_no_bypass_surface():
    # FROZEN §6.5: the exemption surface is EMPTY — no whitelist, no
    # skip flag, no environment-variable bypass.
    import ashare_model.method_revision as mr

    source = Path(mr.__file__).read_text(encoding="utf-8").lower()
    for banned in ("skip", "exempt", "whitelist", "getenv", "environ"):
        assert banned not in source, (
            f"method_revision must carry no bypass surface: {banned}"
        )
