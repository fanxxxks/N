"""A05 layered readiness report (p30): per-purpose eligibility, four
statuses, no unified HEALTHY.

The doctor's legacy ``healthy`` flag answers one narrow question — "do
the engineering conflict rules (version/dataset bindings/gates/fairness
errors) trigger?" — and the 2026-09-09 audit proved that a HEALTHY
doctor coexisted with real blockers (F06).  This module composes the
EXISTING authorities into the six per-purpose dimensions the plan
requires and never re-derives any judgement:

* **runtime** — config load, input paths, package deps, RunStore root;
* **snapshot_integrity** — the A02 composite snapshot confirmation
  (``runspec.confirm_composite_snapshot``, full uncached re-hash);
* **data_eligibility** — the G1–G8 production gates plus, for a bound
  run, the D7 verdict replayed from the run's own FACTOR_SET_QUALIFIED
  evidence (per-formula-dependency by p17 §12; a global verdict stays a
  preregistered non-goal);
* **research_ready** — lifecycle edge activation, fairness probes, and
  the carried runtime/snapshot/gates prerequisites;
* **promotion_ready** — artifact binding conflicts (non-legacy) and, for
  a bound run, the p27 R7/I10 source-run qualification;
* **paper_ready** — the lifecycle-bound paper account scan (no hardcoded
  ``"current"``), the sim-state cross-check, and the carried G8 online
  freshness.

Every check is one of ``pass`` / ``fail`` / ``unknown`` (not computable,
reason mandatory) / ``not_applicable`` (the use does not exist here,
reason mandatory); N/A and unknown can never aggregate into a pass
(p30 §2.4).  The whole layer is read-only: DuckDB connections are
read-only, the v2 file is never opened for writing, the RunStore is
replayed without taking the writer lock.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

READINESS_REPORT_VERSION = "1"

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_UNKNOWN = "unknown"
STATUS_NOT_APPLICABLE = "not_applicable"

_STATUSES = frozenset(
    {STATUS_PASS, STATUS_FAIL, STATUS_UNKNOWN, STATUS_NOT_APPLICABLE}
)

# The doctor is always an audit-mode action (p27 §3 action-mode table):
# the report declares it so a reader can never mistake it for a formal
# admission decision.
ACTION_MODE = "audit"

_SHANGHAI = ZoneInfo("Asia/Shanghai")

#: Dimensions, in report order (p30 §3.1).
DIMENSIONS = (
    "runtime",
    "snapshot_integrity",
    "data_eligibility",
    "research_ready",
    "promotion_ready",
    "paper_ready",
)


def aggregate_status(statuses) -> str:
    """The frozen p30 §2.4 aggregation: fail > unknown > pass, and an
    all-N/A set stays not_applicable (N/A never counts as a pass)."""

    values = list(statuses)
    invalid = sorted(set(values) - _STATUSES)
    if invalid:
        raise ValueError(
            f"unregistered readiness status {invalid[0]!r}; the four-"
            "status vocabulary is the single authority (p30 §2.3)"
        )
    if any(value == STATUS_FAIL for value in values):
        return STATUS_FAIL
    if any(value == STATUS_UNKNOWN for value in values):
        return STATUS_UNKNOWN
    if not values or all(
        value == STATUS_NOT_APPLICABLE for value in values
    ):
        return STATUS_NOT_APPLICABLE
    return STATUS_PASS


def _check(
    name: str,
    status: str,
    detail: str,
    *,
    release_condition: str | None = None,
    carried_from: str | None = None,
) -> dict[str, Any]:
    if status not in _STATUSES:
        raise ValueError(f"check {name!r} carries status {status!r}")
    check: dict[str, Any] = {
        "name": name,
        "status": status,
        "detail": detail,
    }
    if release_condition is not None:
        check["release_condition"] = release_condition
    if carried_from is not None:
        check["carried_from"] = carried_from
    return check


def _dimension(name: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "status": aggregate_status(check["status"] for check in checks),
        "checks": checks,
    }


# -- gatherers (the DB/FS-touching inputs of the pure builder) ----------------


def shanghai_today() -> str:
    """The evaluation day (Asia/Shanghai local date, YYYYMMDD)."""

    return datetime.now(_SHANGHAI).date().strftime("%Y%m%d")


def fairness_status(fairness_probes: dict | None) -> dict | None:
    """Aggregate the doctor's fairness-probe section into a readiness
    input (decision ⑥ semantics preserved: a missing/stale baseline
    warns, never blocks; a failed probe blocks formal runs)."""

    if fairness_probes is None:
        return None
    probes = fairness_probes.get("probes", [])
    failed = [check for check in probes if not check.get("ok")]
    if failed:
        return {
            "status": STATUS_FAIL,
            "detail": "; ".join(
                f"{check.get('name')}: {check.get('detail')}"
                for check in failed
            ),
        }
    baseline_state = (
        "stale" if fairness_probes.get("baseline_stale")
        else "recorded" if fairness_probes.get("baseline") is not None
        else "missing"
    )
    detail = "probes passed on this tree"
    if baseline_state == "stale":
        detail += (
            "; baseline is stale — re-record the probe evidence after "
            "the vocabulary change (decision ⑥)"
        )
    elif baseline_state == "missing":
        detail += (
            "; no probe baseline recorded — record the evidence "
            "(decision ⑥)"
        )
    return {"status": STATUS_PASS, "detail": detail}


def lifecycle_edges_status(lifecycle_checks: dict | None) -> dict:
    """The readiness input for the static edge-activation check (the
    doctor's ``gather_lifecycle_checks`` result → one status)."""

    if lifecycle_checks is None:
        return {
            "status": STATUS_UNKNOWN,
            "detail": "not computed (lifecycle checks not gathered)",
        }
    for check in lifecycle_checks.get("checks", []):
        if check.get("name") == "lifecycle_edges_activated":
            return {
                "status": STATUS_PASS if check.get("ok") else STATUS_FAIL,
                "detail": str(check.get("detail", "")),
            }
    error = next(
        (
            check
            for check in lifecycle_checks.get("checks", [])
            if check.get("name") == "lifecycle_error"
        ),
        None,
    )
    return {
        "status": STATUS_UNKNOWN,
        "detail": str(error.get("detail")) if error else "check missing",
    }


def gather_store_root(project_root: Path) -> dict[str, Any]:
    """The A01 runtime-path resolver, audit mode — the same RunStore-root
    authority every write-side consumer uses (never a second rule)."""

    from ashare_data.config import ConfigPathError, resolve_runtime_paths

    try:
        paths = resolve_runtime_paths(
            None, project_root=Path(project_root), mode="audit"
        )
    except (ConfigPathError, FileNotFoundError, OSError) as exc:
        return {
            "root": None,
            "source": "unresolved",
            "detail": f"runtime path resolution failed: {exc}",
        }
    return {
        "root": str(paths.data_dir),
        "source": "config",
        "detail": "runtime paths resolved (audit mode)",
    }


def gather_config_binding(config_path: Path | None) -> dict[str, Any]:
    """Config provenance: the resolved path and the sha256 of the file
    bytes (a binding fact for the report, not a semantics authority)."""

    import hashlib

    path = Path(config_path) if config_path is not None else None
    if path is None:
        return {
            "status": STATUS_UNKNOWN,
            "path": None,
            "sha256": None,
            "detail": "not computable: no config path resolved",
        }
    if not path.is_file():
        return {
            "status": STATUS_FAIL,
            "path": str(path),
            "sha256": None,
            "detail": f"config file missing: {path}",
        }
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return {
            "status": STATUS_UNKNOWN,
            "path": str(path),
            "sha256": None,
            "detail": f"config file unreadable: {exc}",
        }
    return {
        "status": STATUS_PASS,
        "path": str(path),
        "sha256": digest,
        "detail": "config file read for binding",
    }


def gather_input_paths(data_config) -> dict[str, Any]:
    """The two INPUT databases must exist before any run; output
    directories are allowed to be created and are never required here."""

    from ashare_data.config import resolve_v2_duckdb_path

    try:
        required = {
            "primary": Path(data_config.duckdb_path),
            "v2": Path(resolve_v2_duckdb_path(data_config)),
        }
    except Exception as exc:  # noqa: BLE001 - a report, never a crash
        return {
            "status": STATUS_UNKNOWN,
            "detail": (
                "not computable: the input paths could not be resolved "
                f"({type(exc).__name__}: {exc})"
            ),
            "missing": [],
        }
    missing: list[str] = []
    for name, path in required.items():
        try:
            exists = path.is_file()
        except OSError as exc:
            return {
                "status": STATUS_UNKNOWN,
                "detail": (
                    f"not computable: {name} input path {path} is "
                    f"unreadable ({exc})"
                ),
                "missing": [],
            }
        if not exists:
            missing.append(f"{name} input database missing: {path}")
    if missing:
        return {
            "status": STATUS_FAIL,
            "detail": "; ".join(missing),
            "missing": missing,
        }
    return {
        "status": STATUS_PASS,
        "detail": "primary and v2 input databases present",
        "missing": [],
    }


def gather_snapshot(data_config) -> dict[str, Any]:
    """Content-confirm the dual-database composite snapshot through the
    A02 authority (full uncached re-hash on both sides)."""

    from ashare_data.config import resolve_v2_duckdb_path
    from ashare_data.db import AshareDB
    from ashare_data.manifest import DatasetIdMismatch
    from ashare_model.runspec import confirm_composite_snapshot

    result: dict[str, Any] = {
        "status": STATUS_UNKNOWN,
        "detail": "",
        "composite_id": None,
        "publication_id": None,
        "primary_dataset_id": None,
        "v2_dataset_id": None,
    }
    try:
        with AshareDB(
            data_config.duckdb_path, read_only=True
        ) as primary_db, AshareDB(
            resolve_v2_duckdb_path(data_config), read_only=True
        ) as v2_db:
            try:
                composite = confirm_composite_snapshot(
                    primary_db, data_config, v2_db
                )
            except DatasetIdMismatch as exc:
                result["status"] = STATUS_FAIL
                result["detail"] = str(exc)
                return result
        result["status"] = STATUS_PASS
        result["detail"] = (
            "composite snapshot confirmed (publication "
            f"{composite.publication_id}; full uncached re-hash)"
        )
        result["composite_id"] = composite.composite_id
        result["publication_id"] = composite.publication_id
        result["primary_dataset_id"] = composite.primary.dataset_id
        result["v2_dataset_id"] = composite.v2.dataset_id
        return result
    except Exception as exc:  # noqa: BLE001 - a report, never a crash
        result["status"] = STATUS_UNKNOWN
        result["detail"] = (
            "not computable: the snapshot identity could not be confirmed "
            f"({type(exc).__name__}: {exc})"
        )
        return result


def _bound_paper_account(store: Any, spec_id: str, run_id: str) -> dict[str, Any]:
    """The bound account of one PAPER_OBSERVING run through the single
    read path (``lifecycle.bound_paper_account_id``); strictly read-only.

    The handle is deliberately NOT ``close()``d: this handle never took
    the single-writer lock, and ``RunHandle.close()`` unconditionally
    unlinks ``runs/<spec>/<run>.lock`` — closing here would delete the
    lock of an ACTIVE writer mid-simulation (dual-review HIGH finding;
    ``clear_stale_lock`` is the only sanctioned lock removal and it is
    an explicit administrative action, never automatic).  The handle
    holds no OS resources (the ledger opens the trials file per append),
    so dropping it without close is safe.
    """

    from ashare_model import lifecycle
    from ashare_model.ledger import LedgerIntegrityError
    from ashare_model.run_store import RunHandle, RunStoreError

    try:
        spec = lifecycle._load_spec_from_run(store, spec_id, run_id)
        run_dir = store.runs_dir / spec_id / run_id
        handle = RunHandle(store, spec, run_id, run_dir)
        account = lifecycle.bound_paper_account_id(handle)
    except (RunStoreError, lifecycle.LifecycleError, LedgerIntegrityError, OSError) as exc:
        return {"account_id": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"account_id": account, "error": None}


def gather_paper(
    store_root: str | Path | None, *, sim_state_path: str | Path | None
) -> dict[str, Any]:
    """Replay the RunStore for lifecycle-bound paper observations and
    cross-check the sim state file — the false-positive-free replacement
    of the hardcoded ``assert_paper_account_writable("current")``."""

    from ashare_model import lifecycle
    from ashare_model.run_store import RunStore

    state_path = Path(sim_state_path) if sim_state_path is not None else None
    state_exists: bool | None = None
    if state_path is not None:
        try:
            state_exists = state_path.is_file()
        except OSError:
            state_exists = None

    binding: dict[str, Any] = {"status": STATUS_UNKNOWN, "detail": "", "bound": []}
    if store_root is None:
        binding["status"] = STATUS_UNKNOWN
        binding["detail"] = (
            "not computable: the RunStore root is unresolved "
            "(the runtime dimension carries the reason)"
        )
    else:
        root = Path(store_root)
        store = RunStore(root)
        bound: list[dict[str, Any]] = []
        scan_errors: list[str] = []
        if store.runs_dir.is_dir():
            for run_dir in sorted(store.runs_dir.glob("*/*")):
                if not run_dir.is_dir():
                    continue
                spec_id, run_id = run_dir.parent.name, run_dir.name
                try:
                    state = lifecycle.replay_run(root, spec_id, run_id)
                except lifecycle.LifecycleError as exc:
                    scan_errors.append(f"{spec_id}/{run_id}: {exc}")
                    continue
                if state != "PAPER_OBSERVING":
                    continue
                account = _bound_paper_account(store, spec_id, run_id)
                bound.append(
                    {
                        "spec_id": spec_id,
                        "run_id": run_id,
                        "account_id": account["account_id"],
                        "state": state,
                    }
                )
                if account["error"]:
                    scan_errors.append(
                        f"{spec_id}/{run_id}: binding read failed "
                        f"({account['error']})"
                    )
                elif account["account_id"] is None:
                    scan_errors.append(
                        f"{spec_id}/{run_id}: the PAPER_OBSERVING binding "
                        "evidence carries no account id"
                    )
        if bound:
            missing_accounts = [
                entry for entry in bound if entry["account_id"] is None
            ]
            if missing_accounts:
                binding["status"] = STATUS_UNKNOWN
                binding["detail"] = (
                    "a PAPER_OBSERVING run exists but its binding evidence "
                    "does not yield an account id; audit the binding "
                    "artifacts before any paper claim"
                )
            else:
                binding["status"] = STATUS_PASS
                binding["detail"] = (
                    f"{len(bound)} lifecycle-bound paper observation(s) "
                    "replayed from the RunStore"
                )
            binding["bound"] = bound
        elif scan_errors:
            # A run could not be replayed: "no observation exists" is an
            # UNPROVEN claim (the run may be one of the unread ones) —
            # unknown, never not_applicable (p30 §2.3).
            binding["status"] = STATUS_UNKNOWN
            binding["detail"] = (
                "not computable: the RunStore scan hit "
                f"{len(scan_errors)} replay/binding failure(s) — a paper "
                "observation cannot be ruled out or confirmed: "
                + "; ".join(scan_errors[:5])
            )
        else:
            binding["status"] = STATUS_NOT_APPLICABLE
            binding["detail"] = (
                "no lifecycle-bound paper observation exists under "
                f"{root}; resume/continue claims are refused without one "
                "(p27 I9/R6; p8 contract §8)"
            )
        if scan_errors:
            binding["scan_errors"] = scan_errors

    # The sim state file must never look resumable without a binding.
    if binding["status"] == STATUS_UNKNOWN or state_exists is None:
        sim_status = STATUS_UNKNOWN
        sim_detail = (
            "not computable: the sim state / binding cross-check needs "
            "both a resolvable store root and a readable state path"
        )
    elif state_exists and binding["status"] == STATUS_PASS:
        sim_status = STATUS_PASS
        sim_detail = (
            "paper state file present and consistent with the binding "
            "evidence"
        )
    elif state_exists:
        sim_status = STATUS_FAIL
        sim_detail = (
            f"paper state file exists at {state_path} but no "
            "PAPER_OBSERVING run binds an account — resuming would be an "
            "unbound continuation (p27 I9/R6; p8 contract §8)"
        )
    elif binding["status"] == STATUS_PASS:
        sim_status = STATUS_UNKNOWN
        sim_detail = (
            f"bound paper run exists but no sim state file was found at "
            f"{state_path}"
        )
    else:
        sim_status = STATUS_NOT_APPLICABLE
        sim_detail = "no paper state file and no binding evidence"

    return {
        "binding": binding,
        "sim_state": {
            "status": sim_status,
            "detail": sim_detail,
            "state_path": str(state_path) if state_path is not None else None,
            "state_exists": state_exists,
        },
    }


def gather_bound_run(
    store_root: str | Path | None, spec_id: str, run_id: str
) -> dict[str, Any]:
    """Replay ONE real run: its lifecycle state, its recorded D7 data
    eligibility (from the integrity-verified FACTOR_SET_QUALIFIED
    evidence — a historical fact labelled as such), and the p27 R7/I10
    source-run qualification consumed by the promotion side."""

    from ashare_model import lifecycle
    from ashare_model.run_store import RunStore, RunStoreError

    report: dict[str, Any] = {
        "spec_id": spec_id,
        "run_id": run_id,
        "state": None,
        "state_status": STATUS_UNKNOWN,
        "state_detail": "",
        "eligibility": {
            "status": STATUS_UNKNOWN,
            "detail": "not computable: the run state could not be replayed",
        },
        "source_qualified": {
            "status": STATUS_UNKNOWN,
            "detail": "not computable: the run state could not be replayed",
        },
    }
    if store_root is None:
        for key in ("state_status",):
            report[key] = STATUS_UNKNOWN
        report["state_detail"] = (
            "not computable: the RunStore root is unresolved"
        )
        report["eligibility"]["detail"] = (
            "not computable: the RunStore root is unresolved"
        )
        report["source_qualified"]["detail"] = (
            "not computable: the RunStore root is unresolved"
        )
        return report

    root = Path(store_root)
    try:
        state = lifecycle.replay_run(root, spec_id, run_id)
    except lifecycle.LifecycleError as exc:
        report["state_detail"] = f"replay failed: {exc}"
        report["eligibility"]["detail"] = f"replay failed: {exc}"
        report["source_qualified"]["detail"] = f"replay failed: {exc}"
        return report
    report["state"] = state
    if state is None:
        report["state_status"] = STATUS_UNKNOWN
        report["state_detail"] = (
            "the run has an empty lifecycle event chain (legacy/unverified)"
        )
    else:
        report["state_status"] = STATUS_PASS
        report["state_detail"] = f"replayed lifecycle state {state!r}"

    # D7 verdict as recorded at the run's own FACTOR_SET_QUALIFIED
    # transition (historical fact; today's composite confirmation is the
    # snapshot_integrity dimension's answer).
    try:
        store = RunStore(root)
        index = store.load_index(spec_id, run_id)
        run_dir = store.runs_dir / spec_id / run_id
        events = lifecycle.load_events(
            run_dir / run_store.LIFECYCLE_FILENAME,
            expect_spec_id=spec_id,
            expect_run_id=run_id,
        )
        event = next(
            (
                entry
                for entry in reversed(events)
                if entry.to_state == "FACTOR_SET_QUALIFIED"
            ),
            None,
        )
        if event is None:
            report["eligibility"] = {
                "status": STATUS_NOT_APPLICABLE,
                "detail": (
                    "the run has not reached FACTOR_SET_QUALIFIED: no D7 "
                    "verdict exists yet (per-formula-dependency "
                    "eligibility is recorded at that transition)"
                ),
            }
        else:
            from ashare_model.run_store import read_registered_artifact

            payload = read_registered_artifact(
                run_dir, index, str(event.evidence["artifact_id"])
            )
            d7 = payload.get("data_eligibility")
            if not isinstance(d7, dict):
                report["eligibility"] = {
                    "status": STATUS_UNKNOWN,
                    "detail": (
                        "the FACTOR_SET_QUALIFIED evidence payload carries "
                        "no data_eligibility block"
                    ),
                }
            elif d7.get("ok"):
                report["eligibility"] = {
                    "status": STATUS_PASS,
                    "detail": (
                        "D7 verdict recorded at the run's "
                        "FACTOR_SET_QUALIFIED transition: all required "
                        "sources eligible (as-recorded historical fact)"
                    ),
                }
            else:
                reasons = "; ".join(str(r) for r in d7.get("reasons", []))
                report["eligibility"] = {
                    "status": STATUS_FAIL,
                    "detail": (
                        "D7 verdict recorded at the run's "
                        f"FACTOR_SET_QUALIFIED transition failed: {reasons}"
                    ),
                }
    except (RunStoreError, lifecycle.LifecycleError, KeyError, OSError) as exc:
        report["eligibility"] = {
            "status": STATUS_UNKNOWN,
            "detail": f"not computable: the evidence replay failed ({exc})",
        }

    # p27 R7/I10 consumption side: the promotion entry's floor.
    try:
        qualified_state = lifecycle.assert_qualified_source_run(
            root, spec_id, run_id
        )
    except lifecycle.LifecycleError as exc:
        report["source_qualified"] = {
            "status": STATUS_FAIL,
            "detail": str(exc),
        }
    else:
        report["source_qualified"] = {
            "status": STATUS_PASS,
            "detail": (
                "source run qualified: lifecycle chain at or above the "
                f"search floor ({qualified_state!r})"
            ),
        }
    return report


# -- the pure builder ----------------------------------------------------------


def build_readiness_report(
    *,
    now: str,
    evaluation_day: str,
    config_status: dict[str, Any],
    input_paths: dict[str, Any],
    dependencies: dict[str, str | None],
    model_searcher: str | None,
    store_root: dict[str, Any],
    snapshot: dict[str, Any] | None,
    gates: dict[str, Any] | None,
    artifacts: list[dict[str, Any]] | None,
    artifact_findings: list[dict[str, Any]],
    fairness: dict[str, Any] | None,
    lifecycle_edges: dict[str, Any],
    bound_run: dict[str, Any] | None,
    paper: dict[str, Any],
    git_commit: str | None = None,
    repo_root: str | None = None,
) -> dict[str, Any]:
    """Assemble the p30 report from gathered inputs (pure; JSON-ready).

    Every authority verdict is carried, never re-computed here; the only
    logic in this function is the frozen four-status aggregation and the
    mandatory bindings block.
    """

    # -- runtime ---------------------------------------------------------------
    runtime_checks = [
        _check(
            "config_loaded",
            str(config_status.get("status", STATUS_UNKNOWN)),
            str(config_status.get("detail", "not computable")),
            release_condition=(
                "fix the configuration error named in the detail; the "
                "doctor cannot evaluate any path while the config fails"
                if config_status.get("status") == STATUS_FAIL
                else (
                    "restore the config file named in the detail, then "
                    "re-run the doctor"
                    if config_status.get("status") == STATUS_UNKNOWN
                    else None
                )
            ),
        ),
        _check(
            "input_paths_present",
            str(input_paths.get("status", STATUS_UNKNOWN)),
            str(input_paths.get("detail", "not computable")),
            release_condition=(
                "restore or sync the named input database(s); an empty "
                "substitute file is never a fix"
                if input_paths.get("status") == STATUS_FAIL
                else (
                    "resolve the config error first, then re-run the "
                    "doctor"
                    if input_paths.get("status") == STATUS_UNKNOWN
                    else None
                )
            ),
        ),
    ]
    if model_searcher is None:
        runtime_checks.append(
            _check(
                "runtime_deps",
                STATUS_UNKNOWN,
                "not computable: the searcher is unknown (config error)",
                release_condition=(
                    "resolve the config error first, then re-run the "
                    "doctor"
                ),
            )
        )
    else:
        required = ["torch", "numpy", "pandas"]
        if str(model_searcher) == "gp":
            required.append("deap")
        missing = [
            name
            for name in required
            if dependencies.get(name) is None
        ]
        if missing:
            runtime_checks.append(
                _check(
                    "runtime_deps",
                    STATUS_FAIL,
                    "missing research-critical dependencies: "
                    + ", ".join(missing),
                    release_condition=(
                        "install the named packages (the environment "
                        "contract in docs/PROJECT_ONBOARDING.md lists the "
                        "supported set)"
                    ),
                )
            )
        else:
            runtime_checks.append(
                _check(
                    "runtime_deps",
                    STATUS_PASS,
                    "required dependencies present ("
                    + ", ".join(
                        f"{name}={dependencies.get(name)}"
                        for name in required
                    )
                    + ")",
                )
            )
    store_root_ok = bool(store_root.get("root"))
    # An explicit gatherer status wins (e.g. "unknown — not computable:
    # config error"); otherwise the root presence decides.
    store_root_status = str(
        store_root.get(
            "status",
            STATUS_PASS if store_root_ok else STATUS_FAIL,
        )
    )
    runtime_checks.append(
        _check(
            "store_root_resolvable",
            store_root_status,
            str(store_root.get("detail", "not computable")),
            release_condition=(
                "fix the runtime-path configuration so the RunStore root "
                "resolves (the same resolver the write side uses)"
                if store_root_status == STATUS_FAIL
                else (
                    "resolve the config error first, then re-run the "
                    "doctor"
                    if store_root_status == STATUS_UNKNOWN
                    else None
                )
            ),
        )
    )
    runtime = _dimension("runtime", runtime_checks)

    # -- snapshot_integrity ------------------------------------------------------
    if snapshot is None:
        snapshot_checks = [
            _check(
                "composite_identity_confirmed",
                STATUS_UNKNOWN,
                "not computed (the caller did not gather this section)",
                release_condition=(
                    "run `python -m ashare_model.research_doctor` for "
                    "the full composite confirmation"
                ),
            )
        ]
    else:
        snapshot_checks = [
            _check(
                "composite_identity_confirmed",
                str(snapshot.get("status", STATUS_UNKNOWN)),
                str(snapshot.get("detail", "not computable")),
                release_condition=(
                    "publish a new composite snapshot generation under the "
                    "current protocol (A02 p26 I7; see "
                    "docs/p26_a02_dataset_snapshot_identity_draft.md)"
                    if snapshot.get("status") == STATUS_FAIL
                    else (
                        "resolve the database access error named in the "
                        "detail (a locked/missing file is an "
                        "environmental condition), then re-run the doctor"
                        if snapshot.get("status") == STATUS_UNKNOWN
                        else None
                    )
                ),
            )
        ]
    snapshot_dim = _dimension("snapshot_integrity", snapshot_checks)

    # -- data_eligibility --------------------------------------------------------
    if gates is None:
        gates_status = STATUS_UNKNOWN
        gates_detail = (
            "not computable: the G1–G8 runner did not produce a result"
        )
        gate_checks: list[dict[str, Any]] = []
    else:
        gates_status = str(gates.get("status", STATUS_UNKNOWN))
        raw_checks = list(gates.get("checks", []))
        passed = sum(1 for check in raw_checks if check.get("ok"))
        gates_detail = (
            f"G1–G8: {passed}/{len(raw_checks)} checks passed "
            f"(mode={gates.get('mode', 'unknown')})"
        )
        gate_checks = [
            {
                "name": str(check.get("name", "")),
                "ok": bool(check.get("ok")),
                "detail": str(check.get("detail", "")),
            }
            for check in raw_checks
        ]
    eligibility_checks = [
        _check(
            "production_gates_g1_g8",
            gates_status,
            gates_detail,
            release_condition=(
                "close the failed gate(s) named in the per-gate detail "
                "(data sync/repair is a stateful action needing its own "
                "authorization); dev mode never substitutes for a formal "
                "gate pass"
                if gates_status == STATUS_FAIL
                else (
                    "resolve the config error, then re-run "
                    "`python -m ashare_model.research_doctor` for the "
                    "gate pass"
                    if gates_status == STATUS_UNKNOWN
                    else None
                )
            ),
        )
    ]
    if gates is not None:
        # p30 §3.1: the per-gate detail travels with the check so an
        # API consumer sees WHICH gate failed, not just a count.
        eligibility_checks[0]["per_gate"] = gate_checks
    if bound_run is None:
        eligibility_checks.append(
            _check(
                "runspec_data_eligibility",
                STATUS_NOT_APPLICABLE,
                "no bound run: per-formula-dependency eligibility is "
                "run-grained (p17 §12; a global verdict is a "
                "preregistered non-goal) — bind a run with "
                "--spec-id/--run-id or read its FACTOR_SET_QUALIFIED "
                "evidence",
            )
        )
    else:
        eligibility = bound_run.get("eligibility", {})
        eligibility_status = str(eligibility.get("status", STATUS_UNKNOWN))
        eligibility_checks.append(
            _check(
                "runspec_data_eligibility",
                eligibility_status,
                str(eligibility.get("detail", "not computable")),
                release_condition=(
                    "repair the recorded D7 reasons (the failing sources "
                    "are named in the detail) or re-qualify the run "
                    "through the p8 §5.2 chain"
                    if eligibility_status == STATUS_FAIL
                    else (
                        "repair the run's event-chain/evidence integrity "
                        "named in the detail, then re-run the doctor"
                        if eligibility_status == STATUS_UNKNOWN
                        else None
                    )
                ),
            )
        )
    data_dim = _dimension("data_eligibility", eligibility_checks)

    # -- research_ready ------------------------------------------------------------
    research_checks = [
        _check(
            "lifecycle_edges_activated",
            str(lifecycle_edges.get("status", STATUS_UNKNOWN)),
            str(lifecycle_edges.get("detail", "not computable")),
            release_condition=(
                "complete the P8-15 activation procedure "
                "(docs/p8_research_lifecycle_contract.md §9) so every "
                "legal edge is activated"
                if lifecycle_edges.get("status") == STATUS_FAIL
                else (
                    "resolve the lifecycle check error named in the "
                    "detail, then re-run the doctor"
                    if lifecycle_edges.get("status") == STATUS_UNKNOWN
                    else None
                )
            ),
        ),
        _check(
            "fairness_probes",
            (
                str(fairness.get("status", STATUS_UNKNOWN))
                if fairness is not None
                else STATUS_UNKNOWN
            ),
            (
                str(fairness.get("detail", "not computable"))
                if fairness is not None
                else "not computable: the probe harness did not run"
            ),
            release_condition=(
                "repair the failed probe named in the detail on this "
                "tree — formal runs stay blocked until the probes pass "
                "(campaign decision ⑥)"
                if fairness is not None
                and fairness.get("status") == STATUS_FAIL
                else (
                    "resolve the probe-harness error (torch/deap/vocab "
                    "import) named in the detail, then re-run the doctor"
                    if fairness is None
                    or fairness.get("status") == STATUS_UNKNOWN
                    else None
                )
            ),
        ),
        _check(
            "runtime_prerequisite",
            runtime["status"],
            f"carried from the runtime dimension ({runtime['status']})",
            carried_from="runtime",
            release_condition=(
                "resolve the failing runtime checks above (their release "
                "conditions apply)"
                if runtime["status"] == STATUS_FAIL
                else None
            ),
        ),
        _check(
            "snapshot_prerequisite",
            snapshot_dim["status"],
            "carried from the snapshot_integrity dimension "
            f"({snapshot_dim['status']})",
            carried_from="snapshot_integrity",
            release_condition=(
                "resolve the snapshot_integrity blockers above (their "
                "release conditions apply)"
                if snapshot_dim["status"] == STATUS_FAIL
                else None
            ),
        ),
        _check(
            "gates_prerequisite",
            gates_status,
            "carried from data_eligibility.production_gates "
            f"({gates_status})",
            carried_from="data_eligibility.production_gates",
            release_condition=(
                "resolve the data_eligibility.production_gates blockers "
                "(their release conditions apply)"
                if gates_status == STATUS_FAIL
                else None
            ),
        ),
    ]
    research_dim = _dimension("research_ready", research_checks)

    # -- promotion_ready -----------------------------------------------------------
    # Only ERROR-severity conflicts block promotion: the doctor's scan
    # downgrades properly-legacy-stamped mismatches to info (auditable),
    # and a legacy artifact must never fail this dimension — it is
    # simply not a candidate (p30 §3.1: N/A, never pass).
    artifact_conflicts = [
        str(finding.get("message", ""))
        for finding in artifact_findings
        if finding.get("severity") == "error"
    ]
    if artifacts is None:
        promotion_binding = _check(
            "artifact_binding",
            STATUS_UNKNOWN,
            "not computable: the artifact inventory was not gathered "
            "(config error or API on-demand mode); run "
            "`python -m ashare_model.research_doctor` for the full pass",
        )
    else:
        non_legacy = [
            artifact
            for artifact in artifacts
            if artifact.get("exists") and not artifact.get("legacy")
        ]
        has_existing = any(
            artifact.get("exists") for artifact in artifacts
        )
        legacy_only = has_existing and not non_legacy
        if artifact_conflicts:
            promotion_binding = _check(
                "artifact_binding",
                STATUS_FAIL,
                "; ".join(artifact_conflicts),
                release_condition=(
                    "stamp the artifact legacy (scripts/"
                    "stamp_legacy_artifacts.py) or re-produce it under "
                    "the current versions; an unstamped mismatched "
                    "artifact must never be treated as the current "
                    "champion"
                ),
            )
        elif non_legacy:
            promotion_binding = _check(
                "artifact_binding",
                STATUS_PASS,
                f"{len(non_legacy)} non-legacy artifact(s) consistent "
                "with the current version set",
            )
        elif legacy_only:
            promotion_binding = _check(
                "artifact_binding",
                STATUS_NOT_APPLICABLE,
                "only legacy artifacts present: they remain auditable "
                "and are never promotable (p8 contract §8)",
            )
        else:
            promotion_binding = _check(
                "artifact_binding",
                STATUS_NOT_APPLICABLE,
                "no candidate artifact exists in the configured data_dir",
            )
    if bound_run is None:
        source_check = _check(
            "source_run_qualified",
            STATUS_NOT_APPLICABLE,
            "no bound run: bind one with --spec-id/--run-id to replay "
            "the p27 R7/I10 source-run qualification",
        )
    else:
        source = bound_run.get("source_qualified", {})
        source_status = str(source.get("status", STATUS_UNKNOWN))
        source_check = _check(
            "source_run_qualified",
            source_status,
            str(source.get("detail", "not computable")),
            release_condition=(
                "drive the source run through the qualification chain "
                "(p27 R7/I10) or bind a qualified source run before any "
                "promotion claim"
                if source_status == STATUS_FAIL
                else (
                    "repair the source run's event-chain integrity named "
                    "in the detail, then re-run the doctor"
                    if source_status == STATUS_UNKNOWN
                    else None
                )
            ),
        )
    promotion_dim = _dimension(
        "promotion_ready", [promotion_binding, source_check]
    )

    # -- paper_ready ------------------------------------------------------------------
    binding = paper.get("binding", {})
    sim_state = paper.get("sim_state", {})
    binding_status = str(binding.get("status", STATUS_UNKNOWN))
    binding_detail = str(binding.get("detail", "not computable"))
    if binding.get("scan_errors"):
        # The scan-level failures must be VISIBLE in the report, not
        # only in the gatherer output (dual-review finding 5).
        binding_detail += " | scan errors: " + "; ".join(
            str(error) for error in binding["scan_errors"][:5]
        )
    paper_checks = [
        _check(
            "paper_binding_evidence",
            binding_status,
            binding_detail,
            release_condition=(
                "retire the observed run or repair its binding evidence "
                "(the account binding comes from the PAPER_OBSERVING "
                "transition, p27 I9) — never resume without one"
                if binding_status == STATUS_FAIL
                else (
                    "repair the replay/binding failures listed in the "
                    "detail, then re-run the doctor"
                    if binding_status == STATUS_UNKNOWN
                    else None
                )
            ),
        ),
        _check(
            "sim_state_vs_binding",
            str(sim_state.get("status", STATUS_UNKNOWN)),
            str(sim_state.get("detail", "not computable")),
            release_condition=(
                "retire or explicitly rebuild the unbound state through "
                "the audited migration path; a state file without a "
                "lifecycle-bound account is never resumable"
                if sim_state.get("status") == STATUS_FAIL
                else (
                    "resolve the store-root/state-path issue named in "
                    "the detail, then re-run the doctor"
                    if sim_state.get("status") == STATUS_UNKNOWN
                    else None
                )
            ),
        ),
    ]
    g8 = next(
        (
            check
            for check in gate_checks
            if check["name"].startswith("G8")
        ),
        None,
    ) if gate_checks else None
    if binding_status == STATUS_NOT_APPLICABLE:
        # No online observation exists: freshness has nothing to be
        # fresh FOR (the frozen snapshot answer is snapshot_integrity's).
        paper_checks.append(
            _check(
                "online_data_freshness",
                STATUS_NOT_APPLICABLE,
                "no online paper observation exists; the G8 freshness "
                "verdict applies to a live observation, not to a frozen "
                "snapshot (see snapshot_integrity)",
                carried_from="data_eligibility.production_gates",
            )
        )
    elif g8 is None:
        paper_checks.append(
            _check(
                "online_data_freshness",
                STATUS_UNKNOWN,
                "not computable: the G8 freshness gate did not produce a "
                "result in this report",
                carried_from="data_eligibility.production_gates",
                release_condition=(
                    "run `python -m ashare_model.research_doctor` for "
                    "the full G1–G8 pass"
                ),
            )
        )
    else:
        paper_checks.append(
            _check(
                "online_data_freshness",
                STATUS_PASS if g8["ok"] else STATUS_FAIL,
                (
                    f"carried G8 freshness verdict: {g8['detail']} "
                    "(today-dependent, unlike the frozen snapshot)"
                ),
                carried_from="data_eligibility.production_gates",
                release_condition=(
                    "bring the daily-bar window back within the G8 "
                    "open-session tolerance (data sync is a stateful "
                    "action needing its own authorization; p16)"
                    if not g8["ok"]
                    else None
                ),
            )
        )
    paper_dim = _dimension("paper_ready", paper_checks)

    dimensions = {
        "runtime": runtime,
        "snapshot_integrity": snapshot_dim,
        "data_eligibility": data_dim,
        "research_ready": research_dim,
        "promotion_ready": promotion_dim,
        "paper_ready": paper_dim,
    }

    # -- summary ---------------------------------------------------------------------
    blocking: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    by_status = {
        STATUS_PASS: 0,
        STATUS_FAIL: 0,
        STATUS_UNKNOWN: 0,
        STATUS_NOT_APPLICABLE: 0,
    }
    for name, dimension in dimensions.items():
        for check in dimension["checks"]:
            by_status[check["status"]] += 1
            entry = {
                "dimension": name,
                "check": check["name"],
                "detail": check["detail"],
                "release_condition": check.get("release_condition", ""),
            }
            if check["status"] == STATUS_FAIL:
                blocking.append(entry)
            elif check["status"] == STATUS_UNKNOWN:
                unresolved.append(entry)

    snapshot_binding: dict[str, Any]
    if snapshot is None or snapshot.get("status") != STATUS_PASS:
        snapshot_binding = {
            "error": (
                str(snapshot.get("detail", "not computed"))
                if snapshot is not None
                else "not computed"
            )
        }
    else:
        snapshot_binding = {
            "composite_id": snapshot.get("composite_id"),
            "publication_id": snapshot.get("publication_id"),
            "primary_dataset_id": snapshot.get("primary_dataset_id"),
            "v2_dataset_id": snapshot.get("v2_dataset_id"),
        }

    return {
        "readiness_version": READINESS_REPORT_VERSION,
        "action_mode": ACTION_MODE,
        "bindings": {
            "readiness_version": READINESS_REPORT_VERSION,
            "action_mode": ACTION_MODE,
            "evaluated_at": now,
            "evaluation_day": evaluation_day,
            "config_path": config_status.get("path"),
            "config_sha256": config_status.get("sha256"),
            "git_commit": git_commit,
            "repo_root": repo_root,
            "store_root": store_root.get("root"),
            "store_root_source": store_root.get("source"),
            "bound_run": (
                {
                    "spec_id": bound_run.get("spec_id"),
                    "run_id": bound_run.get("run_id"),
                    "state": bound_run.get("state"),
                }
                if bound_run is not None
                else None
            ),
            "snapshot": snapshot_binding,
        },
        "dimensions": dimensions,
        "summary": {
            "blocking": blocking,
            "unresolved": unresolved,
            "by_status": by_status,
        },
    }
