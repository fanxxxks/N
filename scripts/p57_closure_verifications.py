"""P57 WP18 closure verifications (收口需求 R1/R3 + §9 items).

Runs the closure package's LIVE read-only checks and writes
``reports/wp18/closure_verifications.json``:

* R1 — the slow cost-stress closure: the skip record's dual carriers
  (cost_stress.json screen_adjudication vs the C2 trial ledger) are
  verbatim-identical with stress_rows_authorized=0 / row_count=0; the
  60 unrun stress row keys (= 20 medium C1:A + 40 slow C2) are
  PRESERVED in the integrity plans; measurement_incomplete is the
  designed-in screen-refused state.  ZERO chase: the §5 initial
  screen refused authorization, so no 0x/2x row is run (result-
  shopping red line).
* R3 — method-revision zero-trigger closure: registry episodes == []
  re-verified and the guards refuse on start (fail-closed liveness).
* §9.6 (E3) — the slow_ab cache entry's byte identity vs the medium_a
  entry of the same content-addressed key, against the sha256
  registered in errata E3: a VERIFIED no-op, not deleted (deleting
  would be a further mutation of an existing directory).
* §9.3 — the domain-gate shape deviation pipeline note for future
  packages (split the signal-diagnostics stage per domain, or align
  the gate definition with the stage partitioning).
* §9.2/§9.7/§9.8 — the standing items restated (user-overridable
  E2/E9 ratification; restart conditions; measurement limits).

Zero new measurement rows / backtests / scoring; data/** untouched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# `python scripts/p57_closure_verifications.py` sets sys.path[0] to
# scripts/ — make the package importable (repo convention).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WP18 = ROOT / "reports" / "wp18"
OUT = WP18 / "closure_verifications.json"

SLOW_STRESS = WP18 / "slow" / "cost_stress.json"
SLOW_LEDGER = (
    ROOT / "data" / "r1" / "wp18_slow" / "runs" / "wp18_c2"
    / "wp18-c2-20260927" / "trials.jsonl"
)
E3_SLOW_ENTRY = (
    ROOT / "data" / "r1" / "slow_ab" / "factor_tensor_cache"
    / "ed8dc37c37eecfe36af9f5bcf09effa1c8a7c3170a574091588e2b052cd851f0.npz"
)
E3_MEDIUM_ENTRY = (
    ROOT / "data" / "r1" / "medium_a" / "factor_tensor_cache"
    / "ed8dc37c37eecfe36af9f5bcf09effa1c8a7c3170a574091588e2b052cd851f0.npz"
)
E3_PREEXISTING_SLOW = (
    ROOT / "data" / "r1" / "slow_ab" / "factor_tensor_cache"
    / "4ab7587af59df64c45a37f9ccf4fbc7b6d1a2166b8e755f1a6ce123ef9e4210d.npz"
)
E3_PREEXISTING_MEDIUM = (
    ROOT / "data" / "r1" / "medium_a" / "factor_tensor_cache"
    / "4ab7587af59df64c45a37f9ccf4fbc7b6d1a2166b8e755f1a6ce123ef9e4210d.npz"
)
#: errata E3's registered sha256 of the medium_a entry (evidence field).
E3_REGISTERED_SHA256 = (
    "81db3d19f49424ddb51734519ff4e99284c5b694dce3116d00fd59ce9f4d8b4f"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 24)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_r1() -> dict:
    stress = _load(SLOW_STRESS)
    adjudication = stress["screen_adjudication"]
    rows = [
        json.loads(line)
        for line in SLOW_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    screen_rows = [r for r in rows if r.get("algorithm") == "screen:1x_initial"]
    letters = {
        "slow:combo:slow_all5:A": adjudication["A"]["skip_reason"],
        "slow:combo:slow_all5:B": adjudication["B"]["skip_reason"],
    }
    ledger_matches = {
        letter: any(
            (r.get("payload") or {}).get("skip_reason") == skip
            and (r.get("payload") or {}).get("stress_rows_authorized") == 0
            and str(r.get("candidate")) == letter
            for r in screen_rows
        )
        for letter, skip in letters.items()
    }
    slow_plan = _load(WP18 / "slow" / "integrity" / "plan.json")
    c2_arms = [
        a for a in slow_plan["arms"]
        if str(a.get("candidate", "")).startswith("C2:")
    ]
    slow_reconcile = _load(WP18 / "slow" / "integrity" / "reconcile_full.json")
    slow_status = _load(WP18 / "slow" / "integrity" / "status_full.json")
    medium_reconcile = _load(
        WP18 / "medium" / "integrity" / "reconcile_full.json"
    )
    medium_status = _load(WP18 / "medium" / "integrity" / "status_full.json")
    return {
        "item": "slow cost-stress closure (1x initial screen refused "
                "authorization for BOTH rules)",
        "carriers": {
            "cost_stress.screen_adjudication": str(SLOW_STRESS.relative_to(ROOT)),
            "trial_ledger.screen:1x_initial": str(
                SLOW_LEDGER.relative_to(ROOT)
            ),
        },
        "screen_outcome": {
            "A": {
                "failed_conditions": adjudication["A"]["failed_conditions"],
                "skip_reason": adjudication["A"]["skip_reason"],
                "dynamic_cumulative_net_excess": (
                    adjudication["A"]["dynamic_cumulative_net_excess"]
                ),
            },
            "B": {
                "failed_conditions": adjudication["B"]["failed_conditions"],
                "skip_reason": adjudication["B"]["skip_reason"],
                "dynamic_cumulative_net_excess": (
                    adjudication["B"]["dynamic_cumulative_net_excess"]
                ),
            },
        },
        "dual_carrier_verbatim_identical": all(ledger_matches.values()),
        "dual_carrier_detail": ledger_matches,
        "stress_rows_authorized": 0,
        "row_count": stress["row_count"],
        "ledger_rows_total": len(rows),
        "ledger_screen_rows": len(screen_rows),
        "unrun_row_keys_preserved": {
            "medium_c1_a": medium_reconcile["n_missing"],
            "slow_c2": slow_reconcile["n_missing"],
            "total": medium_reconcile["n_missing"] + slow_reconcile["n_missing"],
            "slow_c2_arms_in_plan": len(c2_arms),
            "slow_missing_by_stage": slow_reconcile["missing_by_stage"],
            "medium_missing_by_stage": medium_reconcile["missing_by_stage"],
            "slow_missing_adjudication_note": (
                slow_reconcile["missing_adjudication"]["note"]
            ),
        },
        "measurement_incomplete_semantics": (
            "designed-in state: the missing rows are EXACTLY the "
            "§5-screened-out stress arms (initial screen refused "
            "authorization); NOT a measurement failure and NOT evidence "
            "of a negative economic result (WP18_REPORT.md:38,194)"
        ),
        "status_labels": {
            "slow": {
                "label": slow_status["label"],
                "verdict_valid": slow_status["verdict_valid"],
                "primary_arms_measured": slow_status["primary_arms_measured"],
            },
            "medium": {
                "label": medium_status["label"],
                "verdict_valid": medium_status["verdict_valid"],
            },
        },
        "disposition": (
            "closed by design — no chase (不追跑): running 0x/2x rows for a "
            "rule that failed the §5 initial screen would be "
            "selection-shopping an already-refused result; the 60 row keys "
            "stay materialized in the plans for any future round under a "
            "changed screen criterion"
        ),
    }


def verify_r3() -> dict:
    from ashare_model import method_revision as mr

    registry = _load(mr.DEFAULT_REGISTRY_PATH)
    refusal = None
    try:
        mr.require_open_episode(None)
    except ValueError as exc:
        refusal = "G1_REFUSED" if str(exc) == mr.G1_REFUSED else str(exc)
    attested_ok = (
        mr.assert_method_version_endorsed(mr.ATTESTED_METHOD_VERSION) is None
    )
    return {
        "item": "method-revision zero-trigger closure (WP17 W6)",
        "registry": str(mr.DEFAULT_REGISTRY_PATH.relative_to(ROOT)),
        "episodes": registry.get("episodes"),
        "guard_liveness": {
            "require_open_episode_none": refusal,
            "attested_version_needs_no_episode": attested_ok,
        },
        "closure": (
            "no trigger record was ever opened; the guards refuse any "
            "method-version comparison or acceptance calibration on start "
            "(fail-closed, FROZEN_17 §6.5 G1).  Formally closed with this "
            "package; reopening requires the W6 episode protocol."
        ),
    }


def verify_e3() -> dict:
    slow_hash = _sha256(E3_SLOW_ENTRY)
    medium_hash = _sha256(E3_MEDIUM_ENTRY)
    return {
        "item": "errata E3 — slow_ab content-addressed cache entry byte "
                "identity (disclosed write; verified no-op)",
        "files": {
            "slow_ab_entry": str(E3_SLOW_ENTRY.relative_to(ROOT)),
            "medium_a_entry": str(E3_MEDIUM_ENTRY.relative_to(ROOT)),
        },
        "sizes_bytes": {
            "slow_ab": E3_SLOW_ENTRY.stat().st_size,
            "medium_a": E3_MEDIUM_ENTRY.stat().st_size,
        },
        "sha256": {"slow_ab": slow_hash, "medium_a": medium_hash},
        "registered_sha256_in_errata_e3": E3_REGISTERED_SHA256,
        "byte_identical": slow_hash == medium_hash,
        "matches_errata_registration": slow_hash == E3_REGISTERED_SHA256,
        "preexisting_entry_untouched": {
            "name": E3_PREEXISTING_SLOW.name,
            "present_in_slow_ab": E3_PREEXISTING_SLOW.exists(),
            "same_size_as_medium_a_twin": (
                E3_PREEXISTING_SLOW.stat().st_size
                == E3_PREEXISTING_MEDIUM.stat().st_size
            ),
            "note": "the directory's only pre-E3 entry; left as-is "
                    "(zero mutations to data/** in this package)",
        },
        "action": "verified_no_op_not_deleted",
        "rationale": (
            "the appended entry is the deterministic rebuild of the same "
            "domain-independent tensor (identity key = v2 duckdb + "
            "dataset_id + dates + ts_codes); deleting it would be a "
            "further mutation of an existing directory — disclosure over "
            "restoration, per errata E3"
        ),
    }


def domain_gate_note() -> dict:
    return {
        "item": "§9.3 domain-gate shape deviation (errata E4) — pipeline "
                "note for future packages",
        "source": ["WP18_REPORT.md §9.3", "errata E4"],
        "note": (
            "future packages replicating this pipeline must either split "
            "the signal-diagnostics stage per-domain (medium diagnostics "
            "gate the slow leg) or align the gate definition with the "
            "stage partitioning, so the frozen-form gate "
            "(full-domain integrity + runtime cap BEFORE entering the "
            "next domain) is executable as written instead of the "
            "post-hoc subset-gate + second-checkpoint + ledger-backfill "
            "shape WP18 landed"
        ),
        "disposition": "orchestration-layer note recorded; no rerun",
    }


def standing_items() -> dict:
    return {
        "item": "§9.2/§9.7/§9.8 standing items restated",
        "s9_2_contract_ruling_user_overridable": (
            "the E2/E9 adjudication (FROZEN_A §8.3 dual-constant refusal "
            "gate; re-learn of the 19 refused keys) is a run-owner-level "
            "ratification the user may still overturn — rollback actions "
            "listed in errata E9 standing"
        ),
        "s9_7_restart_conditions": (
            "the hypothesis batch (medium 3-factor + slow 5-factor equal-"
            "weight combos, A/B orientation, reference grid) is PAUSED "
            "(stop-table rows 4 and 6); only a new economic hypothesis or "
            "a new information source justifies restart; B-fulltrain "
            "would need separate pre-registration (contract §2.3)"
        ),
        "s9_8_measurement_limits": (
            "DIVIDEND_YIELD's 'no information' reading over 0.291-0.361 "
            "coverage is coverage-limited (measurement_incomplete "
            "semantics — not an economic negative result); slow 2021 fold "
            "budget constriction degrades dynamic to hold (Q3 mechanism "
            "note)"
        ),
    }


def main() -> int:
    result = {
        "package": "p57_wp18_closure",
        "purpose": (
            "WP18 closure verifications — zero new measurement rows / "
            "backtests / scoring; data/** untouched"
        ),
        "r1_slow_cost_stress_closure": verify_r1(),
        "r3_method_revision_zero_trigger": verify_r3(),
        "e3_cache_entry_byte_identity": verify_e3(),
        "domain_gate_pipeline_note": domain_gate_note(),
        "standing_items_restated": standing_items(),
    }
    OUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"p57 closure verifications -> {OUT}")
    r1 = result["r1_slow_cost_stress_closure"]
    e3 = result["e3_cache_entry_byte_identity"]
    print(
        f"  R1 dual-carrier identical: "
        f"{r1['dual_carrier_verbatim_identical']}; preserved row keys: "
        f"{r1['unrun_row_keys_preserved']['total']}"
    )
    print(f"  E3 byte-identical: {e3['byte_identical']} "
          f"(registered match: {e3['matches_errata_registration']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
