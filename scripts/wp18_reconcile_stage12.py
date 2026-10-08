"""WP18 (p56) stage-1+2 reconciliation: experiment_integrity reconcile +
status against both the FULL per-domain plan and a stages-1+2 subset plan.

Contract: ``reports/wp18/FROZEN_18_PLAN.md`` §10.2 — after any run the
products MUST pass reconcile (missing/duplicate/unexpected/failed + sha
anchoring); status classification is additionally saved.  The wp14 CLI's
``reconcile``/``status`` subcommands are registered stubs (exit 1), so
this driver calls the landed module authorities directly
(``reconcile.reconcile_report`` / ``status.classify_status``).

Two reconciliations per domain, both honest:

* FULL plan (M1+M2+M3+C1 / S1+S2+S3+C2): the missing rows must name
  EXACTLY the not-yet-run stages (M3/C1 or S3/C2 — account + conditional
  stress, outside this stage-1+2 execution).  Anything else is a defect.
* STAGES-1+2 SUBSET plan (measurement arms only): must reconcile clean
  (0 missing / 0 duplicate / 0 unexpected / 0 failed) — this is the
  completeness proof of the executed 180-row set.

The overall status label is taken from the FULL plan (the plan-registered
artifact); the subset status is additionally recorded — with the stages
3/4 rows absent it either reports the not-yet-complete layers or refuses
on the adjudication word, which is recorded verbatim (no verdict is
minted by a measurement stage).

Usage (cwd = AlphaGPT): python scripts/wp18_reconcile_stage12.py
"""

from __future__ import annotations

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

MEASUREMENT_STAGES = {"M1", "M2", "S1", "S2"}


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
        stage, _rule, _mode, _cand = parse_wp18_candidate(key.candidate)
        counts[stage or "<control>"] += 1
    return dict(sorted(counts.items()))


def reconcile_domain(domain: str) -> dict:
    integ = _ROOT / f"reports/wp18/{domain}/integrity"
    plan_path = integ / "plan.json"
    report_path = _ROOT / f"data/r1/wp18_{domain}/wp18_{domain}_report.json"
    run_meta_path = (
        _ROOT / f"data/r1/wp18_{domain}/wp18_{domain}_run_meta.json"
    )
    plan = plan_from_json(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

    # --- full-plan reconcile ---------------------------------------------
    recon_full = reconcile_report(plan, report, run_meta)
    full_json = reconcile_to_json(recon_full)
    full_json["missing_by_stage"] = _missing_by_stage(recon_full)
    full_json["issue_codes"] = dict(
        Counter(i.code for i in recon_full.issues)
    )
    _write(integ / "reconcile_full.json", full_json)

    # --- stages-1+2 subset plan + reconcile --------------------------------
    full_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    subset_arms = [
        arm
        for arm in full_payload["arms"]
        if parse_wp18_candidate(arm["candidate"])[0] in MEASUREMENT_STAGES
    ]
    subset_payload = dict(full_payload)
    subset_payload["arms"] = subset_arms
    expected = sum(
        len(a["cost_levels"]) * len(full_payload["folds"]) for a in subset_arms
    )
    subset_payload["expected_row_count"] = expected
    _write(integ / "plan_stages12.json", subset_payload)
    plan_subset = plan_from_json(subset_payload)
    recon_sub = reconcile_report(plan_subset, report, run_meta)
    sub_json = reconcile_to_json(recon_sub)
    sub_json["issue_codes"] = dict(
        Counter(i.code for i in recon_sub.issues)
    )
    _write(integ / "reconcile_stages12.json", sub_json)

    # --- status (full plan = the overall value; subset recorded too) ------
    status_full = classify_status(plan, recon_full, report)
    _write(integ / "status_full.json", _status_payload(status_full))
    subset_status_payload: dict = {"note": (
        "the stages-1+2 subset classification: with all subset rows "
        "measured the verdict walk looks for the adjudication word, which "
        "a measurement stage never mints — recorded verbatim, never faked"
    )}
    try:
        status_sub = classify_status(plan_subset, recon_sub, report)
        subset_status_payload["status"] = _status_payload(status_sub)
    except IntegrityRefusal as exc:
        subset_status_payload["integrity_refusal"] = str(exc)
    _write(integ / "status_stages12.json", subset_status_payload)

    return {
        "domain": domain,
        "full": {
            "expected_rows": full_json["n_expected_keys"],
            "observed_rows": full_json["n_rows_observed"],
            "missing": full_json["n_missing"],
            "duplicates": full_json["n_duplicates"],
            "unexpected": full_json["n_unexpected"],
            "failed": full_json["n_failed_rows"],
            "missing_by_stage": full_json["missing_by_stage"],
            "issue_codes": full_json["issue_codes"],
            "status_label": status_full.label,
        },
        "stages12_subset": {
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
            str(integ / "reconcile_stages12.json"),
            str(integ / "plan_stages12.json"),
            str(integ / "status_full.json"),
            str(integ / "status_stages12.json"),
        ],
    }


def main() -> int:
    results = [reconcile_domain(domain) for domain in ("medium", "slow")]
    summary = {
        "wp18_reconcile_stage12": True,
        "domains": results,
    }
    _write(
        _ROOT / "reports/wp18/integrity_stage12_summary.json", summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
