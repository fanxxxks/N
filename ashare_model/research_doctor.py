"""Read-only research doctor (P0-03).

Aggregates the repository's research state into one report:

* **code** — git commit / branch / dirty flag (read-only git queries);
* **data** — the content-addressed ``dataset_id`` of the current database
  (latest dataset manifest) plus manifest metadata;
* **gates** — the G1–G8 production gate outcome (``ProductionGateRunner``,
  formal mode, read-only queries);
* **fairness_probes** — the F2-(a) fairness re-probe obligation
  (``campaign_closure_decisions_20260902.md`` ⑥ / t59, machine-triggered by
  IP-03): the two t59 probes (``build_action_mask`` family-⑤ admission and
  the GP ``build_pset`` terminal enumeration) run LIVE on whatever tree the
  doctor inspects, and the recorded probe baseline
  (``docs/fairness_probe_baseline.json``) is compared against the live
  vocabulary fingerprint;
* **artifacts** — strategy / protocol artifact versions (searcher,
  reward_version, protocol_version, model_version, dataset_id) and their
  legacy status (stamped ``legacy`` / ``legacy_reason`` fields, see
  ``ashare_model.artifact_versions`` for the stamping contract);
* **dependencies** — versions of the research-critical packages;
* **runtime estimates** — estimated minutes for the default train budget
  and the protocol screening/confirmation tiers, anchored on the
  documented local pace (never a measurement).

Conflict rules (severity ``error`` -> exit code 1):

* any G1–G8 gate fails;
* any F2-(a) fairness probe fails, or the probe harness cannot run
  (torch/deap/vocab import) — formal runs are blocked until the probes
  pass on the actual tree; a stale or missing probe baseline is reported
  at warning severity (evidence bookkeeping, not a fairness break);
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
from .artifact_versions import classify_artifact, version_matches
from .evaluation import PROTOCOL_VERSION
from .reward import REWARD_VERSION
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import (
    EXECUTION_SPEC_VERSION,
    validate_portfolio_config_provenance,
)

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
        "execution_version",
        "portfolio_constructor_version",
        "portfolio_config",
        "dataset_id",
        "unique_semantic_evals",
    ),
    "protocol": (
        "protocol_version",
        "reward_version",
        "execution_version",
        "portfolio_constructor_version",
        "portfolio_config",
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
    fairness_probes: dict[str, Any] | None = None,
    fundamental_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the gathered sections into the final doctor report.

    Pure function over plain data: every conflict rule lives here so the
    report logic is fully unit-testable without a database or git repo.
    ``fairness_probes`` is ``None`` when the caller did not gather the
    section (the CLI always gathers it); ``None`` contributes no findings
    and the report simply omits the section.
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

    if fairness_probes is not None:
        for check in fairness_probes.get("probes", []):
            if not check.get("ok"):
                findings.append(
                    {
                        "severity": "error",
                        "message": (
                            f"F2-(a) fairness probe {check.get('name')} "
                            f"failed: {check.get('detail')} — formal runs "
                            "are blocked until the probes pass on this "
                            "tree (campaign_closure_decisions_20260902.md "
                            "⑥; evidence record: "
                            "docs/test_runtime_measurement_log.md)"
                        ),
                    }
                )
        if fairness_probes.get("all_ok"):
            baseline = fairness_probes.get("baseline")
            if baseline is None:
                findings.append(
                    {
                        "severity": "warning",
                        "message": (
                            "no fairness probe baseline recorded "
                            "(docs/fairness_probe_baseline.json): the live "
                            "probes passed on this tree, but decision ⑥ "
                            "evidence should be recorded (see "
                            "docs/test_runtime_measurement_log.md)"
                        ),
                    }
                )
            elif fairness_probes.get("baseline_stale"):
                findings.append(
                    {
                        "severity": "warning",
                        "message": (
                            "fairness probe baseline is stale: recorded "
                            f"vocab_fingerprint "
                            f"{baseline.get('vocab_fingerprint')!r} != "
                            f"current "
                            f"{fairness_probes.get('vocab_fingerprint')!r} "
                            "— re-record the probe evidence after the "
                            "vocabulary change (decision ⑥)"
                        ),
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
        # Version equality uses the single comparison primitive from
        # ashare_model.artifact_versions; the doctor's own semantics
        # (missing fields are skipped here and reported separately below,
        # legacy artifacts downgrade to info) stay local.
        for key, current in (
            ("reward_version", REWARD_VERSION),
            ("protocol_version", PROTOCOL_VERSION),
            ("model_version", MODEL_VERSION),
            ("execution_version", EXECUTION_SPEC_VERSION),
            (
                "portfolio_constructor_version",
                PORTFOLIO_CONSTRUCTOR_VERSION,
            ),
        ):
            recorded = fields.get(key)
            if recorded is not None and not version_matches(recorded, current):
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
        for key in ("execution_version", "portfolio_constructor_version"):
            if fields.get(key) is None:
                message = f"{name} artifact carries no {key} (pre-P3)"
                findings.append(
                    {
                        "severity": "info" if legacy else "error",
                        "message": (
                            f"{message} (legacy: {'; '.join(reasons)})"
                            if legacy
                            else message + " and is not marked legacy"
                        ),
                    }
                )
        recorded_portfolio_config = fields.get("portfolio_config")
        if not isinstance(recorded_portfolio_config, dict):
            message = f"{name} artifact carries no portfolio_config (pre-P3)"
            findings.append(
                {
                    "severity": "info" if legacy else "error",
                    "message": (
                        f"{message} (legacy: {'; '.join(reasons)})"
                        if legacy
                        else message + " and is not marked legacy"
                    ),
                }
            )
        else:
            try:
                validate_portfolio_config_provenance(
                    recorded_portfolio_config
                )
            except ValueError as exc:
                message = f"{name} artifact {exc}"
                findings.append(
                    {
                        "severity": "info" if legacy else "error",
                        "message": (
                            f"{message} (legacy: {'; '.join(reasons)})"
                            if legacy
                            else message + " and is not marked legacy"
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
    report = {
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
    if fairness_probes is not None:
        report["fairness_probes"] = fairness_probes
    if fundamental_coverage is not None:
        report["fundamental_coverage"] = fundamental_coverage
    return report


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


#: IP-13 (03-F-09 / 05-④): slow-fundamental registry features whose underlying
#: ``fundamental_pit`` columns exist but with known low coverage
#: (~10.1% / 10.1% / 1.4% at the P13 backfill snapshot).  Disclosure is
#: REPORT-ONLY (no gate, no finding): the numbers make the coverage
#: auditable; filling them is a future preregistered decision
#: (docs/fundamental_coverage_ledger.md, U7).
FUNDAMENTAL_COVERAGE_FIELDS: tuple[str, ...] = (
    "roa",
    "debt_ratio",
    "dividend_yield",
)


def gather_fundamental_coverage(data_config) -> dict[str, Any]:
    """Finite-value coverage of the slow-fundamental columns (read-only).

    Denominator mirrors the P13 §4.1 measurement style: finite values
    (non-NULL, non-NaN) over all ``fundamental_pit`` rows.  Report-only
    by contract (IP-13): this section never raises findings and never
    gates anything.
    """

    result: dict[str, Any] = {
        "table": "fundamental_pit",
        "total_rows": None,
        "distinct_ts_code": None,
        "definition": "finite values (non-NULL, non-NaN) / total rows",
        "fields": {
            name: {"finite": None, "coverage": None}
            for name in FUNDAMENTAL_COVERAGE_FIELDS
        },
        "report_only": True,
        "error": None,
    }
    finite = ", ".join(
        f'COUNT(*) FILTER (WHERE "{name}" IS NOT NULL AND NOT isnan("{name}")) '
        f'AS "{name}"'
        for name in FUNDAMENTAL_COVERAGE_FIELDS
    )
    query = (
        "SELECT COUNT(*) AS total_rows, "
        "COUNT(DISTINCT ts_code) AS distinct_ts_code, "
        f"{finite} FROM fundamental_pit"
    )
    try:
        with AshareDB(data_config.duckdb_path, read_only=True) as db:
            frame = db.query(query)
    except Exception as exc:  # noqa: BLE001 - report, never crash
        result["error"] = str(exc)
        return result
    if frame.empty:
        result["error"] = "fundamental_pit coverage query returned no rows"
        return result
    row = frame.iloc[0]
    total = int(row["total_rows"]) if row["total_rows"] is not None else 0
    result["total_rows"] = total or None
    distinct = row["distinct_ts_code"]
    result["distinct_ts_code"] = int(distinct) if distinct is not None else None
    for name in FUNDAMENTAL_COVERAGE_FIELDS:
        finite_count = int(row[name]) if row[name] is not None else 0
        result["fields"][name] = {
            "finite": finite_count,
            "coverage": (finite_count / total) if total else None,
        }
    return result


def gather_gates(data_config, min_eligible: int = 100) -> dict[str, Any]:
    """G1–G8 outcome through the same runner every formal entry uses."""

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


# -- F2-(a) fairness re-probe (IP-03; campaign_closure_decisions_20260902.md ⑥) -

#: Recorded probe evidence (committed, decision ⑥): the doctor compares its
#: live vocabulary fingerprint against this baseline.  Provenance fields
#: (``code_commit``) are informational; only the vocabulary fingerprint
#: participates in the staleness comparison, so routine merges never flag.
FAIRNESS_PROBE_BASELINE_PATH = Path("docs") / "fairness_probe_baseline.json"

#: P13 family ⑤ (docs/p13_fundamental_fields_contract.md §5.5; decision ⑥):
#: the appended slow-fundamental block whose mask/GP admission the t57/t58
#: repair chain touched.  Frozen here deliberately — the probe pins the
#: probed layout, and a legal vocabulary evolution must go through the
#: re-probe flow (fail-closed) instead of silently re-deriving.
_PROBE_FAMILY5_NAMES = (
    "CASHFLOW_QUALITY",
    "ACCRUALS",
    "ASSET_GROWTH",
    "EARNINGS_ACCEL",
)

#: The frozen grammar-6 layout the t59 probes were recorded against
#: (feature ids 74-77 contiguous append; 77 features, 12 deprecated,
#: 65 samplable; vocabulary fingerprint 0e64ad614bfd).
_PROBE_FROZEN = {
    "family5_ids": (74, 75, 76, 77),
    "feature_count": 77,
    "deprecated_count": 12,
    "samplable_feature_count": 65,
}


def gather_fairness_probes(repo_root: Path) -> dict[str, Any]:
    """Run the two t59 fairness probes LIVE (in-memory, read-only).

    Probe harness imports are lazy and individually guarded: a tree that
    cannot run the probes reports a failed ``probe_harness_import`` check
    (fail-closed) instead of crashing the whole doctor.  The probes are
    deterministic on-tree checks (engineering evidence, never a research
    conclusion).
    """

    probes: list[dict[str, Any]] = []
    try:
        import torch

        from .alphagpt import build_action_mask
        from .gp_search import build_pset
        from .vocab import DEPRECATED_FEATURE_NAMES, FORMULA_VOCAB
    except Exception as exc:  # noqa: BLE001 - the probe must fail closed
        probes.append(
            {
                "name": "probe_harness_import",
                "ok": False,
                "detail": (
                    "probe infrastructure unavailable "
                    f"({exc!r}); formal-run readiness cannot be certified"
                ),
            }
        )
        return _probe_section(probes, None, repo_root)

    vocab = FORMULA_VOCAB
    fingerprint = vocab.feature_version
    family_ids = tuple(
        vocab.feature_offset + vocab.feature_names.index(name)
        for name in _PROBE_FAMILY5_NAMES
    )
    deprecated_ids = {
        vocab.feature_offset + vocab.feature_names.index(name)
        for name in DEPRECATED_FEATURE_NAMES
    }

    layout_ok = (
        family_ids == _PROBE_FROZEN["family5_ids"]
        and vocab.feature_count == _PROBE_FROZEN["feature_count"]
        and len(DEPRECATED_FEATURE_NAMES) == _PROBE_FROZEN["deprecated_count"]
    )
    probes.append(
        {
            "name": "frozen_vocab_layout",
            "ok": layout_ok,
            "detail": (
                f"family-⑤ ids {list(family_ids)} vs frozen "
                f"{list(_PROBE_FROZEN['family5_ids'])}; feature_count "
                f"{vocab.feature_count} vs {_PROBE_FROZEN['feature_count']}; "
                f"deprecated {len(DEPRECATED_FEATURE_NAMES)} vs "
                f"{_PROBE_FROZEN['deprecated_count']}; fingerprint "
                f"{fingerprint} (vocabulary size {vocab.size}, EOS "
                f"{vocab.eos_token_id})"
            ),
        }
    )

    # Probe 1 (t59 ⑥): build_action_mask with feature_ids=[74..77] admits
    # exactly the four family-⑤ tokens at step 0.
    try:
        stack_sizes = torch.zeros(1, dtype=torch.long)
        done = torch.zeros(1, dtype=torch.bool)
        stack_types = torch.zeros(1, 6, dtype=torch.long)
        mask = build_action_mask(
            stack_sizes,
            done,
            0,
            6,
            vocab,
            feature_ids=list(family_ids),
            stack_types=stack_types,
        )
        legal = sorted(
            int(token)
            for token in (mask == 0.0)[0].nonzero(as_tuple=True)[0]
        )
        probes.append(
            {
                "name": "action_mask_family5_step0_exact",
                "ok": legal == list(_PROBE_FROZEN["family5_ids"]),
                "detail": f"step-0 legal set {legal}",
            }
        )

        # Probe 1b (t59 ⑥): the unrestricted mask offers exactly the
        # samplable features (77 - 12 deprecated = 65) and no deprecated
        # token leaks into the legal set.
        full_mask = build_action_mask(
            stack_sizes,
            done,
            0,
            6,
            vocab,
            feature_ids=None,
            stack_types=stack_types,
        )
        legal_full = sorted(
            int(token)
            for token in (full_mask == 0.0)[0].nonzero(as_tuple=True)[0]
        )
        legal_features = [
            token
            for token in legal_full
            if vocab.feature_offset <= token < vocab.operator_offset
        ]
        leaks = sorted(deprecated_ids & set(legal_features))
        probes.append(
            {
                "name": "action_mask_full_vocab_deprecated_excluded",
                "ok": len(legal_features)
                == _PROBE_FROZEN["samplable_feature_count"]
                and not leaks,
                "detail": (
                    f"step-0 legal features {len(legal_features)} vs "
                    f"{_PROBE_FROZEN['samplable_feature_count']}; "
                    f"deprecated leaks {leaks}"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash
        probes.append(
            {
                "name": "action_mask_family5_step0_exact",
                "ok": False,
                "detail": f"probe raised: {exc!r}",
            }
        )
        probes.append(
            {
                "name": "action_mask_full_vocab_deprecated_excluded",
                "ok": False,
                "detail": f"probe raised: {exc!r}",
            }
        )

    # Probe 2 (t59 ⑥): the GP primitive set enumerates the single registry
    # — exactly the four family-⑤ terminals under restriction, exactly the
    # 65 samplable features unrestricted, no duplicates, no deprecated.
    try:
        restricted = build_pset(vocab, feature_ids=list(family_ids))
        restricted_names = [
            getattr(terminal, "name", str(terminal))
            for terminals in restricted.terminals.values()
            for terminal in terminals
        ]
        probes.append(
            {
                "name": "gp_pset_family5_terminals",
                "ok": sorted(restricted_names)
                == sorted(_PROBE_FAMILY5_NAMES),
                "detail": (
                    f"restricted terminals {sorted(restricted_names)} "
                    f"(count {len(restricted_names)})"
                ),
            }
        )

        full = build_pset(vocab, feature_ids=None)
        full_names = [
            getattr(terminal, "name", str(terminal))
            for terminals in full.terminals.values()
            for terminal in terminals
        ]
        deprecated_terminals = sorted(set(full_names) & set(DEPRECATED_FEATURE_NAMES))
        probes.append(
            {
                "name": "gp_pset_full_vocab_terminal_count",
                "ok": len(set(full_names))
                == _PROBE_FROZEN["samplable_feature_count"]
                and len(full_names) == len(set(full_names))
                and not deprecated_terminals,
                "detail": (
                    f"unique terminals {len(set(full_names))} vs "
                    f"{_PROBE_FROZEN['samplable_feature_count']} "
                    f"(total {len(full_names)}); deprecated terminals "
                    f"{deprecated_terminals}"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - report, never crash
        probes.append(
            {
                "name": "gp_pset_family5_terminals",
                "ok": False,
                "detail": f"probe raised: {exc!r}",
            }
        )
        probes.append(
            {
                "name": "gp_pset_full_vocab_terminal_count",
                "ok": False,
                "detail": f"probe raised: {exc!r}",
            }
        )

    return _probe_section(probes, fingerprint, repo_root)


def _probe_section(
    probes: list[dict[str, Any]], fingerprint: str | None, repo_root: Path
) -> dict[str, Any]:
    """Assemble the fairness_probes report section incl. the baseline."""

    baseline = read_json_safe(repo_root / FAIRNESS_PROBE_BASELINE_PATH)
    baseline = baseline if isinstance(baseline, dict) else None
    stale = (
        baseline is not None
        and fingerprint is not None
        and baseline.get("vocab_fingerprint") != fingerprint
    )
    return {
        "vocab_fingerprint": fingerprint,
        "probes": probes,
        "baseline": baseline,
        "baseline_stale": stale,
        "all_ok": all(check["ok"] for check in probes),
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
        fairness_probes=gather_fairness_probes(root),
        fundamental_coverage=gather_fundamental_coverage(data_config),
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
    probes = report.get("fairness_probes")
    if probes is not None:
        probe_total = len(probes.get("probes", []))
        probe_passed = sum(1 for c in probes.get("probes", []) if c.get("ok"))
        baseline = probes.get("baseline")
        baseline_state = (
            "stale" if probes.get("baseline_stale")
            else "recorded" if baseline is not None
            else "missing"
        )
        print(
            f"fairness probes (F2-(a) ⑥): {probe_passed}/{probe_total} PASS "
            f"(vocab fingerprint {probes.get('vocab_fingerprint')}, "
            f"baseline {baseline_state})"
        )
        for check in probes.get("probes", []):
            if not check.get("ok"):
                print(f"  FAIL {check.get('name')}: {check.get('detail')}")
    coverage = report.get("fundamental_coverage")
    if coverage is not None:
        if coverage.get("error"):
            print(
                "fundamental coverage (report-only): unavailable "
                f"({coverage['error']})"
            )
        else:
            fields = coverage.get("fields", {})
            parts = []
            for name in ("roa", "debt_ratio", "dividend_yield"):
                value = (fields.get(name) or {}).get("coverage")
                parts.append(
                    f"{name}={value:.4f}"
                    if isinstance(value, (int, float))
                    else f"{name}=n/a"
                )
            print(
                "fundamental coverage (report-only, finite/rows): "
                + ", ".join(parts)
            )
    for artifact in report["artifacts"]:
        status = (
            "LEGACY" if artifact.get("legacy")
            else "present" if artifact.get("exists")
            else "absent"
        )
        print(f"  {artifact['name']}: {artifact['path']} - {status}")
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
