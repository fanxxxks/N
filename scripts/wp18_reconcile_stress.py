"""WP18 (p56) C1/C2 reconciliation: experiment_integrity reconcile +
status against the FULL per-domain plan and the EXECUTED subset plan.

Contract: ``reports/wp18/FROZEN_18_PLAN.md`` §10.2 — after the stress
stage the products MUST pass reconcile (missing/duplicate/unexpected/
failed + sha anchoring); status classification is additionally saved.

Two reconciliations, both honest:

* FULL plan (M1/M2/M3+C1 or S1/S2/S3+C2): the missing rows must name
  EXACTLY the §5-screened-OUT stress subset — the arms of the rules
  that failed the 1x screen and were therefore never executed (the
  conditional stage is planned at the 40-row upper bound, §4.5).  The
  executed subset and the skip decisions live in
  ``reports/wp18/<domain>/cost_stress.json``; this script CROSS-CHECKS
  the missing set against that adjudication and refuses on any
  divergence (anything missing that was not screened out, or anything
  screened out yet observed, is a defect).
* EXECUTED-SUBSET plan (stages 1+2+3 + the executed stress arms only):
  must reconcile clean (0 missing / 0 duplicate / 0 unexpected /
  0 failed) — the completeness proof of the executed row set.

The earlier artifacts (plan.json / plan_stages12.json /
plan_stages123.json and their reconcile/status views) are historical
records of those executions and are NOT touched; the full-plan views
(reconcile_full.json / status_full.json) are regenerated to describe
the CURRENT artifact.

Usage (cwd = AlphaGPT): python scripts/wp18_reconcile_stress.py --domain medium|slow
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_model.experiment_integrity.plan import plan_from_json  # noqa: E402
from ashare_model.experiment_integrity.reconcile import (  # noqa: E402
    reconcile_report,
    reconcile_to_json,
)
from ashare_model.experiment_integrity.status import classify_status  # noqa: E402
from ashare_model.experiment_integrity.types import IntegrityRefusal  # noqa: E402
from ashare_model.wp18_signal_vs_account import parse_wp18_candidate  # noqa: E402

DONE_STAGES = {"M1", "M2", "M3", "S1", "S2", "S3"}
BARE_CONTROLS = {"benchmark:equal_weight", "control:cash"}


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _status_payload(status) -> dict:
    payload = dataclasses.asdict(status)
    payload["arm_status"] = [
        dataclasses.asdict(arm) for arm in status.arm_status
    ]
    return payload


def _missing_by_stage(recon) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key in recon.missing:
        stage, _rule, _mode, cand = parse_wp18_candidate(key.candidate)
        counts[stage or ("<control>" if cand in BARE_CONTROLS else "?")] += 1
    return dict(sorted(counts.items()))


def reconcile_domain(domain: str) -> dict:
    integ = _ROOT / f"reports/wp18/{domain}/integrity"
    plan_path = integ / "plan.json"
    report_path = _ROOT / f"data/r1/wp18_{domain}/wp18_{domain}_report.json"
    run_meta_path = (
        _ROOT / f"data/r1/wp18_{domain}/wp18_{domain}_run_meta.json"
    )
    stress_path = _ROOT / f"reports/wp18/{domain}/cost_stress.json"
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    stress_stage = str(stress["stage"])
    adjudication = stress["screen_adjudication"]
    executed_letters = {
        letter for letter, rec in adjudication.items() if rec["passed"]
    }
    skipped_letters = set(adjudication) - executed_letters

    plan = plan_from_json(json.loads(plan_path.read_text(encoding="utf-8")))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    # --- full-plan reconcile (regenerated: describes the CURRENT state) --
    recon_full = reconcile_report(plan, report, run_meta)
    full_json = reconcile_to_json(recon_full)
    full_json["missing_by_stage"] = _missing_by_stage(recon_full)
    full_json["issue_codes"] = dict(
        Counter(i.code for i in recon_full.issues)
    )

    # --- §5 cross-check: the missing set must be EXACTLY the screened-out
    # stress subset (never anything else, never a screened-out row observed)
    expected_missing = {
        key for key in recon_full.expected_keys
        if parse_wp18_candidate(key.candidate)[0] == stress_stage
        and parse_wp18_candidate(key.candidate)[1] in skipped_letters
    }
    observed_missing = set(recon_full.missing)
    if observed_missing != expected_missing:
        raise SystemExit(
            f"[{domain}] full-plan reconcile missing set diverges from the "
            f"§5 screening adjudication: missing={len(observed_missing)} "
            f"expected(screened-out)={len(expected_missing)}; "
            f"extra={sorted(observed_missing - expected_missing)[:3]} "
            f"unaccounted={sorted(expected_missing - observed_missing)[:3]}"
        )
    full_json["missing_adjudication"] = {
        "stress_stage": stress_stage,
        "executed_letters": sorted(executed_letters),
        "skipped_letters": sorted(skipped_letters),
        "note": (
            "the missing rows are EXACTLY the §5-screened-out stress arms "
            "(planned at the 40-row upper bound, §4.5; the skip decisions "
            "with full evidence live in cost_stress.json screen_adjudication)"
        ),
    }
    _write(integ / "reconcile_full.json", full_json)

    # --- EXECUTED-SUBSET plan (stages 1+2+3 + executed stress arms) -------
    full_payload = json.loads(plan_path.read_text(encoding="utf-8"))

    def _executed(arm) -> bool:
        stage, letter, _mode, _cand = parse_wp18_candidate(arm["candidate"])
        if arm["candidate"] in BARE_CONTROLS:
            return True
        if stage in DONE_STAGES:
            return True
        return stage == stress_stage and letter in executed_letters

    subset_arms = [arm for arm in full_payload["arms"] if _executed(arm)]
    subset_payload = dict(full_payload)
    subset_payload["arms"] = subset_arms
    expected = sum(
        len(a["cost_levels"]) * len(full_payload["folds"]) for a in subset_arms
    )
    subset_payload["expected_row_count"] = expected
    _write(integ / "plan_stages123c.json", subset_payload)
    plan_subset = plan_from_json(subset_payload)
    recon_sub = reconcile_report(plan_subset, report, run_meta)
    sub_json = reconcile_to_json(recon_sub)
    sub_json["issue_codes"] = dict(
        Counter(i.code for i in recon_sub.issues)
    )
    _write(integ / "reconcile_stages123c.json", sub_json)

    # --- status (full plan = the overall value; subset recorded too) ------
    status_full = classify_status(plan, recon_full, report)
    _write(integ / "status_full.json", _status_payload(status_full))
    subset_status_payload: dict = {"note": (
        "the executed-subset classification: the verdict walk looks for "
        "the adjudication word, which the stress execution never mints "
        "(the report writer owns §6.1) — recorded verbatim, never faked"
    )}
    try:
        status_sub = classify_status(plan_subset, recon_sub, report)
        subset_status_payload["status"] = _status_payload(status_sub)
    except IntegrityRefusal as exc:
        subset_status_payload["integrity_refusal"] = str(exc)
    _write(integ / "status_stages123c.json", subset_status_payload)

    return {
        "domain": domain,
        "stress_stage": stress_stage,
        "screen": {
            letter: {
                "passed": rec["passed"],
                "failed_conditions": rec["failed_conditions"],
            }
            for letter, rec in adjudication.items()
        },
        "full": {
            "expected_rows": full_json["n_expected_keys"],
            "observed_rows": full_json["n_rows_observed"],
            "missing": full_json["n_missing"],
            "duplicates": full_json["n_duplicates"],
            "unexpected": full_json["n_unexpected"],
            "failed": full_json["n_failed_rows"],
            "missing_by_stage": full_json["missing_by_stage"],
            "missing_equals_screened_out_subset": True,
            "issue_codes": full_json["issue_codes"],
            "status_label": status_full.label,
        },
        "executed_subset": {
            "expected_rows": sub_json["n_expected_keys"],
            "observed_rows": sub_json["n_rows_observed"],
            "missing": sub_json["n_missing"],
            "duplicates": sub_json["n_duplicates"],
            "unexpected": sub_json["n_unexpected"],
            "failed": sub_json["n_failed_rows"],
            "issue_codes": sub_json["issue_codes"],
        },
        "outputs": [
            str(integ / "reconcile_full.json"),
            str(integ / "plan_stages123c.json"),
            str(integ / "reconcile_stages123c.json"),
            str(integ / "status_full.json"),
            str(integ / "status_stages123c.json"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("medium", "slow"))
    args = parser.parse_args()
    result = reconcile_domain(args.domain)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
