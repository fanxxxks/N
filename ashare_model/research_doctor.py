"""Read-only research doctor (P0-03).

Aggregates the repository's research state into one report:

* **code** — git commit / branch / dirty flag (read-only git queries);
* **data** — the content-addressed ``dataset_id`` of the current database
  (latest dataset manifest) plus manifest metadata;
* **gates** — the G1–G7 production gate outcome (``ProductionGateRunner``,
  formal mode, read-only queries);
* **artifacts** — strategy / protocol artifact versions (searcher,
  reward_version, protocol_version, model_version, dataset_id) and their
  legacy status (stamped ``legacy`` / ``legacy_reason`` fields, see
  ``ashare_model.artifact_versions`` for the stamping contract);
* **dependencies** — versions of the research-critical packages;
* **runtime estimates** — estimated minutes for the default train budget
  and the protocol screening/confirmation tiers, anchored on the
  documented local pace (never a measurement).

Conflict rules (severity ``error`` -> exit code 1):

* any G1–G7 gate fails;
* the database has no ``dataset_id`` (no dataset manifest);
* an artifact records a reward/protocol/model version that differs from
  the current code generation **and is not marked legacy** — a legacy
  artifact is reported as informational (it is labeled, never a
  champion);
* an artifact's recorded ``dataset_id`` differs from the current database
  and is not marked legacy;
* the default searcher is ``gp`` but DEAP is not installed.

The doctor never writes data, artifacts or git state; the only side
effect is the report file the caller explicitly requests with
``--output``.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ashare_data.config import (
    load_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
)
from ashare_data.db import AshareDB
from ashare_data.gates import ProductionGateRunner
from ashare_data.io_utils import read_json_safe
from ashare_data.manifest import latest_manifest

from .alphagpt import MODEL_VERSION
from .artifact_versions import classify_artifact
from .evaluation import PROTOCOL_VERSION
from .reward import REWARD_VERSION

# Legacy stamp field contract (written by scripts/stamp_legacy_artifacts.py,
# consumed by this doctor and the web API): a legacy artifact carries
# ``legacy: true`` plus a human-readable ``legacy_reason`` list.
LEGACY_FIELD = "legacy"
LEGACY_REASON_FIELD = "legacy_reason"

# Documented local pace (README): ~1.3–1.8 s/step at batch 256, data load
# ~5–6 min; per-step time scales ~sqrt(batch/256).  Estimates only.
_SEC_PER_STEP_AT_256 = 1.55
_DATA_LOAD_MINUTES = 5.5

_ARTIFACT_FIELDS = {
    "strategy": (
        "searcher",
        "reward_version",
        "protocol_version",
        "model_version",
        "dataset_id",
        "unique_semantic_evals",
    ),
    "protocol": (
        "protocol_version",
        "reward_version",
        "dataset_id",
        "ledger",
        "stitched",
    ),
}


def estimate_run_minutes(steps: int, batch_size: int) -> float:
    """Estimated single-run minutes for a ``steps x batch_size`` budget.

    Anchored on the README-documented local pace (1.3–1.8 s/step at batch
    256 plus ~5–6 min data load); per-step time scales with the square
    root of the batch size.  This is an estimate for planning, not a
    measurement.
    """

    sec_per_step = _SEC_PER_STEP_AT_256 * math.sqrt(max(batch_size, 1) / 256.0)
    return round((steps * sec_per_step) / 60.0 + _DATA_LOAD_MINUTES, 1)


def runtime_estimates(model_config, protocol_config) -> dict:
    """Estimated runtimes of the default train budget and protocol tiers."""

    screening = estimate_run_minutes(
        protocol_config.screening.steps, protocol_config.screening.batch_size
    )
    confirmation = estimate_run_minutes(
        protocol_config.confirmation.steps, protocol_config.confirmation.batch_size
    )
    return {
        "basis": (
            "README-documented local pace (1.3-1.8 s/step @ batch 256, "
            "sqrt-scaled by batch; data load ~5-6 min); estimates, not "
            "measurements"
        ),
        "train_default_minutes": estimate_run_minutes(
            model_config.train_steps, model_config.batch_size
        ),
        "protocol_screening_single_run_minutes": screening,
        "protocol_confirmation_single_run_minutes": confirmation,
        "protocol_full_training_minutes_estimate": round(
            screening * len(protocol_config.folds) * len(protocol_config.seeds), 1
        ),
        "note": (
            "protocol totals cover per-fold training only; OOS scoring, "
            "baselines and multiple-testing corrections add extra wall time"
        ),
    }


def build_report(
    *,
    code: dict[str, Any],
    data: dict[str, Any],
    gates: dict[str, Any],
    artifacts: list[dict[str, Any]],
    dependencies: dict[str, str | None],
    model_searcher: str,
    runtime_estimates_: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate the gathered sections into the final doctor report.

    Pure function over plain data: every conflict rule lives here so the
    report logic is fully unit-testable without a database or git repo.
    """

    findings: list[dict[str, str]] = []

    for check in gates.get("checks", []):
        if not check.get("ok"):
            findings.append(
                {
                    "severity": "error",
                    "message": f"gate {check.get('name')} failed: {check.get('detail')}",
                }
            )

    dataset_id = (data or {}).get("dataset_id")
    if not dataset_id:
        findings.append(
            {
                "severity": "error",
                "message": (
                    "database has no dataset_id: no dataset manifest exists "
                    "(run `python -m ashare_data.manifest` to create one)"
                ),
            }
        )

    for artifact in artifacts:
        if not artifact.get("exists"):
            continue
        legacy = bool(artifact.get("legacy"))
        reasons = list(artifact.get("legacy_reasons") or [])
        fields = artifact.get("fields") or {}
        name = artifact.get("name", "artifact")
        for key, current in (
            ("reward_version", REWARD_VERSION),
            ("protocol_version", PROTOCOL_VERSION),
            ("model_version", MODEL_VERSION),
        ):
            recorded = fields.get(key)
            if recorded is not None and str(recorded) != str(current):
                message = (
                    f"{name} artifact {key} {recorded} != current {current}"
                )
                if legacy:
                    findings.append(
                        {
                            "severity": "info",
                            "message": f"{message} (legacy: {'; '.join(reasons)})",
                        }
                    )
                else:
                    findings.append(
                        {
                            "severity": "error",
                            "message": (
                                f"{message} and the artifact is not marked "
                                "legacy — it must not be treated as the "
                                "current champion"
                            ),
                        }
                    )
        recorded_dataset = fields.get("dataset_id")
        if (
            recorded_dataset is not None
            and dataset_id is not None
            and str(recorded_dataset) != str(dataset_id)
        ):
            message = (
                f"{name} artifact dataset_id {recorded_dataset} != current "
                f"database {dataset_id}"
            )
            findings.append(
                {
                    "severity": "info" if legacy else "error",
                    "message": (
                        message
                        if legacy
                        else message + " and the artifact is not marked legacy"
                    ),
                }
            )
        if not legacy and recorded_dataset is None:
            findings.append(
                {
                    "severity": "warning",
                    "message": (
                        f"{name} artifact carries no dataset_id (pre-T1-01); "
                        "its data binding cannot be verified"
                    ),
                }
            )

    if model_searcher == "gp" and dependencies.get("deap") is None:
        findings.append(
            {
                "severity": "error",
                "message": (
                    "default searcher is gp but deap is not installed; the "
                    "default train path cannot run"
                ),
            }
        )

    errors = [f for f in findings if f["severity"] == "error"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": code,
        "data": data,
        "gates": gates,
        "artifacts": artifacts,
        "dependencies": dependencies,
        "runtime_estimates": runtime_estimates_,
        "findings": findings,
        "healthy": not errors,
    }


def gather_code_version(repo_root: Path) -> dict[str, Any]:
    """Read-only git facts (commit / branch / dirty flag)."""

    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    dirty_raw = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("branch", "--show-current"),
        "dirty": dirty_raw is not None and bool(dirty_raw),
        "note": "read-only git queries" if commit is None else None,
    }


def gather_data_version(data_config) -> dict[str, Any]:
    """Latest dataset manifest of the local database (read-only)."""

    result: dict[str, Any] = {
        "dataset_id": None,
        "manifest_version": None,
        "created_at": None,
        "total_rows": None,
        "db_path": str(data_config.duckdb_path),
    }
    try:
        with AshareDB(data_config.duckdb_path, read_only=True) as db:
            manifest = latest_manifest(db, data_config)
    except Exception as exc:  # noqa: BLE001 - report, never crash
        result["error"] = str(exc)
        return result
    if manifest is not None:
        result.update(
            {
                "dataset_id": manifest.dataset_id,
                "manifest_version": manifest.manifest_version,
                "created_at": manifest.created_at,
                "total_rows": manifest.total_rows,
            }
        )
    return result


def gather_gates(data_config, min_eligible: int = 100) -> dict[str, Any]:
    """G1–G7 outcome through the same runner every formal entry uses."""

    result = ProductionGateRunner(
        data_config, min_eligible=min_eligible
    ).run(mode="formal")
    return {
        "mode": result.mode,
        "ok": result.ok,
        "degraded": result.degraded,
        "checks": [
            {"name": check.name, "ok": check.ok, "detail": check.detail}
            for check in result.checks
        ],
    }


def gather_artifacts(data_dir: Path) -> list[dict[str, Any]]:
    """Version fields + legacy status of the strategy/protocol artifacts."""

    snapshots: list[dict[str, Any]] = []
    for name, filename in (
        ("strategy", "best_ashare_strategy.json"),
        ("protocol", "protocol_result.json"),
    ):
        path = Path(data_dir) / filename
        payload = read_json_safe(path)
        if not isinstance(payload, dict):
            snapshots.append(
                {
                    "name": name,
                    "path": str(path),
                    "exists": False,
                    "legacy": False,
                    "legacy_reasons": [],
                    "fields": {},
                }
            )
            continue
        snapshots.append(
            {
                "name": name,
                "path": str(path),
                "exists": True,
                "legacy": bool(payload.get(LEGACY_FIELD)),
                "legacy_reasons": list(payload.get(LEGACY_REASON_FIELD) or []),
                # Pure version classification (P0-04): an old artifact that
                # is not stamped is visible as classified-legacy here while
                # the conflict rules still reject it (no stamp = error).
                "classification": classify_artifact(name, payload),
                "fields": {key: payload.get(key) for key in _ARTIFACT_FIELDS[name]},
            }
        )
    return snapshots


def gather_dependencies() -> dict[str, str | None]:
    """Installed versions of the research-critical packages (None = absent)."""

    versions: dict[str, str | None] = {}
    for name in (
        "torch", "numpy", "pandas", "deap", "optuna", "duckdb", "akshare",
        "cvxpy", "fastapi", "loguru", "pytest", "scipy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to ashare_config.yaml")
    parser.add_argument(
        "--json", action="store_true", help="print the report as JSON to stdout"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="write the report as JSON to PATH (the only write this tool does)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=100,
        help="production gate G6: minimum eligible stocks (default: 100)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    try:
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        model_config = make_model_config(raw)
        protocol_config = make_protocol_config(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"doctor: config error: {exc}", file=sys.stderr)
        return 2

    report = build_report(
        code=gather_code_version(root),
        data=gather_data_version(data_config),
        gates=gather_gates(data_config, min_eligible=args.min_eligible),
        artifacts=gather_artifacts(data_config.data_dir),
        dependencies=gather_dependencies(),
        model_searcher=model_config.searcher,
        runtime_estimates_=runtime_estimates(model_config, protocol_config),
    )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"doctor report written to {out}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    _print_human(report)
    if not report["healthy"]:
        print(
            "\nstatus: NOT HEALTHY "
            f"({sum(1 for f in report['findings'] if f['severity'] == 'error')} "
            "error-level findings)",
            file=sys.stderr,
        )
        return 1
    print("\nstatus: HEALTHY")
    return 0


def _print_human(report: dict[str, Any]) -> None:
    code = report["code"]
    data = report["data"]
    gates = report["gates"]
    print("AlphaGPT research doctor")
    print("=======================")
    print(
        "code: {} (branch {}, {})".format(
            code.get("commit") or "unknown",
            code.get("branch") or "unknown",
            "dirty" if code.get("dirty") else "clean",
        )
    )
    print(
        "data: dataset_id={} manifest v{} (created {}, {} rows, {})".format(
            data.get("dataset_id") or "MISSING",
            data.get("manifest_version") or "-",
            data.get("created_at") or "-",
            data.get("total_rows") or "-",
            data.get("db_path") or "-",
        )
    )
    passed = sum(1 for c in gates.get("checks", []) if c.get("ok"))
    total = len(gates.get("checks", []))
    print(f"gates: {passed}/{total} PASS (mode={gates.get('mode')})")
    for artifact in report["artifacts"]:
        status = (
            "LEGACY" if artifact.get("legacy")
            else "present" if artifact.get("exists")
            else "absent"
        )
        print(f"  {artifact['name']}: {artifact['path']} — {status}")
    print("dependencies: " + ", ".join(
        f"{name}={version or 'MISSING'}"
        for name, version in sorted(report["dependencies"].items())
    ))
    estimates = report["runtime_estimates"]
    print(
        "runtime estimates (minutes): train_default={}, screening_run={}, "
        "confirmation_run={}, protocol_full_training={}".format(
            estimates.get("train_default_minutes"),
            estimates.get("protocol_screening_single_run_minutes"),
            estimates.get("protocol_confirmation_single_run_minutes"),
            estimates.get("protocol_full_training_minutes_estimate"),
        )
    )
    for finding in report["findings"]:
        print(f"  [{finding['severity']}] {finding['message']}")


if __name__ == "__main__":
    raise SystemExit(main())
