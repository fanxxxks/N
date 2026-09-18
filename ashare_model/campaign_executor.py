"""Resumable campaign execution over the TaskExecutor (B03, p32).

The R1 campaign driver (``scripts/r1_campaign.ps1``) used to skip legs by
bare ``.done`` markers, lose partial failures behind ``exit 0`` and carry
no wall-clock or memory control (audit F09).  This module is the campaign
adapter the plan requires: it maps every config leg onto the existing
:class:`ashare_model.task_executor.TaskExecutor` semantics — one leg, one
independent worker subprocess, one ledger-bound attempt — and adds the
campaign layer the executor deliberately does not own:

* **Identity-bound reuse** — a completed leg is reused only when its
  identity (config content hash + the confirmed composite snapshot +
  code commit + seeds + requested budget + the leg's research semantics,
  p32 §3.2) still matches.  ``.done`` files are *derived display*,
  written from terminal success and never read for decisions.
* **Stage telemetry** — the attempt's trial payload declares the
  ``loading / feature_build / search / evaluation / artifact_write``
  stage plan at start; the row command records milestones through
  :func:`ashare_model.task_executor.row_phase`.
* **Budgets** — row wall-clock / row peak-RSS budgets are enforced by
  the worker; the campaign wall cap stops leg launches
  (``stopped_budget``), mirroring the P10/P14 searcher-campaign breaker
  pattern without duplicating its implementation.
* **Terminal meanings** — ``succeeded`` / ``failed`` / ``aborted`` /
  ``not_run`` are distinct; a partial failure is a campaign failure and
  never ``exit 0``.  Resume attaches the same run for interrupted legs
  (new attempt, evidence preserved) and leaves terminal-failed legs
  untouched unless ``retry_failed`` opens a NEW run explicitly.
* **Single writer + ownership-verified kills** — a campaign lock refuses
  a second driver; a stale lock is cleared only explicitly, after
  liveness verification of the holder pid.

The campaign state file is a schema-versioned, atomically written
(sole authority :func:`ashare_data.io_utils.atomic_write_json`) driver
checkpoint that *points into* the RunStore ledgers — it never replaces
them.  Legs run as explicit ``--lifecycle-mode engineering`` probes
(p27 I2): this layer produces execution/recovery evidence, never formal
research identity (formal legs are R01's preregistered wiring).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import psutil
from loguru import logger

from ashare_data.io_utils import atomic_write_json, read_json_safe

from .identity import content_hash
from .ledger import LedgerIntegrityError
from .run_store import RunStore, RunStoreError
from .runspec import resolve_runtime_runspec
from .task_executor import (
    TaskExecutor,
    TaskExecutorError,
    TaskRow,
)

CAMPAIGN_STATE_SCHEMA_VERSION = 1
"""Schema version of the campaign state file (fail-closed on read)."""

CAMPAIGN_STATE_FILENAME = "campaign_state.json"
CAMPAIGN_LOCK_FILENAME = "campaign.lock"

LEG_STAGE_PLAN: tuple[str, ...] = (
    "loading",
    "feature_build",
    "search",
    "evaluation",
    "artifact_write",
)
"""The stage plan registered with every attempt (p32 §3.1)."""

#: Campaign exit codes (p32 §3.4): completed / any leg failed or budget
#: stop / user abort / operational refusal.
EXIT_COMPLETED = 0
EXIT_FAILED = 1
EXIT_ABORTED = 130
EXIT_REFUSED = 2


class CampaignExecutorError(Exception):
    """Campaign contract violation (maps to exit code 2)."""


class CampaignLockedError(CampaignExecutorError):
    """Another driver holds the campaign lock (live or stale)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class CampaignLeg:
    """One config leg of a campaign (one TaskExecutor run when executed)."""

    leg_id: str
    config: Path
    output: Path
    ledger: Path
    extra_args: tuple[str, ...] = ()
    tier: str = "screening"
    #: Row command override: a fixed argv sequence or a
    #: ``Callable[[TaskRow], Sequence[str]]`` (tests / special legs).
    #: ``None`` builds the standard engineering-mode evaluation command.
    command: Sequence[str] | Callable[[TaskRow], Sequence[str]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.leg_id, str) or not self.leg_id.strip():
            raise ValueError("leg_id must be a non-empty string")
        for name, value in (
            ("config", self.config),
            ("output", self.output),
            ("ledger", self.ledger),
        ):
            if not isinstance(value, (str, Path)):
                raise ValueError(f"{name} must be a path")
        if self.command is None or callable(self.command):
            return
        if isinstance(self.command, str) or not list(self.command):
            raise ValueError(
                "command must be a non-empty argv sequence, a callable, "
                f"or None; got {self.command!r}"
            )


@dataclass(frozen=True)
class CampaignManifest:
    """The frozen campaign plan: legs + the budget block (p32 §3.3).

    Budget values are frozen HERE at campaign planning time — never
    hardcoded in code.  ``None`` means explicitly uncapped.
    """

    name: str
    legs: tuple[CampaignLeg, ...]
    row_wall_clock_budget_s: float | None = None
    row_peak_rss_budget_mb: float | None = None
    campaign_wall_clock_budget_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("campaign name must be a non-empty string")
        if not self.legs:
            raise ValueError("a campaign manifest needs at least one leg")
        seen: set[str] = set()
        for leg in self.legs:
            if not isinstance(leg, CampaignLeg):
                raise TypeError(
                    f"manifest legs must be CampaignLeg, got {type(leg)!r}"
                )
            if leg.leg_id in seen:
                raise ValueError(f"duplicate campaign leg id: {leg.leg_id}")
            seen.add(leg.leg_id)
        for name in (
            "row_wall_clock_budget_s",
            "row_peak_rss_budget_mb",
            "campaign_wall_clock_budget_s",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive number or None")


def default_evaluation_command(leg: CampaignLeg) -> list[str]:
    """The standard engineering-mode evaluation row command (p32 §3.7).

    Engineering is the explicit non-formal declaration (p27 I2).  Formal
    legs need preregistered spec/run identity and are R01's wiring.
    """

    import sys

    return [
        str(sys.executable),
        "-m",
        "ashare_model.evaluation",
        "--config",
        str(Path(leg.config).resolve()),
        "--tier",
        str(leg.tier),
        "--output",
        str(Path(leg.output).resolve()),
        "--ledger",
        str(Path(leg.ledger).resolve()),
        "--lifecycle-mode",
        "engineering",
        *leg.extra_args,
    ]


class CampaignRunner:
    """Single-writer driver that runs a manifest's legs to terminal state."""

    def __init__(
        self,
        manifest: CampaignManifest,
        state_dir: str | Path,
        *,
        repo_root: str | Path | None = None,
        identity_provider: Callable[[], tuple[str, str]] | None = None,
        clear_stale_lock: bool = False,
        retry_failed: bool = False,
        heartbeat_interval_s: float = 30.0,
        heartbeat_timeout_s: float = 600.0,
        poll_interval_s: float = 0.2,
        kill_grace_s: float = 3.0,
        done_dir: str | Path | None = None,
        marker_path: str | Path | None = None,
    ):
        self.manifest = manifest
        self.state_dir = Path(state_dir)
        self.repo_root = (
            Path(repo_root)
            if repo_root is not None
            else Path(__file__).resolve().parents[1]
        )
        # Derived-display and human-history locations (review A#14: the
        # generic module takes them as parameters; the R1 CLI pins the
        # data/r1 convention).
        self.done_dir = (
            Path(done_dir) if done_dir is not None else self.repo_root / "data" / "r1"
        )
        self.marker_log_path = (
            Path(marker_path)
            if marker_path is not None
            else self.repo_root / "data" / "r1" / "markers.log"
        )
        self._identity_provider = identity_provider
        self._clear_stale_lock_enabled = bool(clear_stale_lock)
        self._retry_failed = bool(retry_failed)
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._heartbeat_timeout_s = float(heartbeat_timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._kill_grace_s = float(kill_grace_s)
        self.state_path = self.state_dir / CAMPAIGN_STATE_FILENAME
        self.lock_path = self.state_dir / CAMPAIGN_LOCK_FILENAME
        self._lock_acquired = False
        self._stale_campaign_lock_cleared = False
        self._dataset_id: str | None = None
        self._data_cutoff: str | None = None
        self._reference_data_config: Any = None

    # -- locking -------------------------------------------------------------

    def _read_lock(self) -> dict[str, Any]:
        holder = read_json_safe(self.lock_path)
        return holder if isinstance(holder, dict) else {}

    def _acquire_lock(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(
                self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            holder = self._read_lock()
            pid = holder.get("pid")
            alive = isinstance(pid, int) and psutil.pid_exists(pid)
            if alive:
                raise CampaignLockedError(
                    f"campaign {self.manifest.name!r} is already running "
                    f"(holder pid {pid}); a second driver is a contract "
                    "violation (p32 §3.6)"
                )
            if not self._clear_stale_lock_enabled:
                raise CampaignLockedError(
                    f"stale campaign lock for {self.manifest.name!r} "
                    f"(holder pid {pid!r} is not alive); recovery is the "
                    "explicit --clear-stale-lock action (p32 §3.6)"
                )
            self._clear_stale_lock()
            self._acquire_lock()
            return
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": _now_iso()}))
        self._lock_acquired = True

    def _clear_stale_lock(self) -> bool:
        """Explicitly remove a stale campaign lock after verifying the
        recorded holder pid is dead (a live holder is never cleared)."""

        if not self.lock_path.exists():
            return False
        holder = self._read_lock()
        pid = holder.get("pid")
        if isinstance(pid, int) and psutil.pid_exists(pid):
            raise CampaignLockedError(
                f"campaign lock holder pid {pid} is still alive; refusing "
                "to clear a live lock"
            )
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            # A concurrent driver won the removal race; treat as cleared
            # (its own acquire will fail with the live-holder refusal).
            return False
        self._stale_campaign_lock_cleared = True
        logger.warning(
            "campaign {} stale lock cleared (holder pid {!r} verified dead)",
            self.manifest.name,
            pid,
        )
        return True

    def _release_lock(self) -> None:
        if self._lock_acquired:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            self._lock_acquired = False

    # -- identity ------------------------------------------------------------

    def _resolve_data_configs(self) -> dict[str, Any]:
        """Parse every leg config once and enforce data-config consistency
        across the campaign (p32 §3.2/I4): the legs share one data
        identity, so a mismatched data block refuses the campaign."""

        from ashare_data.config import load_config, make_data_config

        configs: dict[str, Any] = {}
        reference_payload: dict[str, Any] | None = None
        for leg in self.manifest.legs:
            try:
                raw = load_config(leg.config, project_root=self.repo_root)
                data_config = make_data_config(raw, self.repo_root)
            except CampaignExecutorError:
                raise
            except Exception as exc:  # noqa: BLE001 - typed refusal (exit 2)
                raise CampaignExecutorError(
                    f"leg {leg.leg_id!r} config resolution refused: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            payload = asdict(data_config)
            if reference_payload is None:
                reference_payload = payload
                self._reference_data_config = data_config
            elif payload != reference_payload:
                raise CampaignExecutorError(
                    f"leg {leg.leg_id!r} resolves a different data config "
                    "than the first campaign leg; a campaign shares one "
                    "data identity (p32 §3.2/I4)"
                )
            configs[leg.leg_id] = data_config
        return configs

    def _resolve_identity(self) -> tuple[str, str]:
        """One composite identity confirmation per campaign launch, through
        the same loader authority the formal entries use (A02).  The
        reuse decision is the *consumption side* of the F02 lesson: a
        published id without content confirmation is never a reuse
        authority."""

        if self._dataset_id and self._data_cutoff:
            return self._dataset_id, self._data_cutoff
        try:
            if self._identity_provider is not None:
                dataset_id, data_cutoff = self._identity_provider()
            else:
                from ashare_data.config import (
                    load_config,
                    make_model_config,
                )
                from ashare_model.data_loader import AshareDataLoader

                if self._reference_data_config is None:
                    self._resolve_data_configs()
                loader = AshareDataLoader(
                    self._reference_data_config,
                    make_model_config(
                        load_config(
                            self.manifest.legs[0].config,
                            project_root=self.repo_root,
                        )
                    ),
                )
                loader.confirm_input_snapshot()
                dataset_id = loader.dataset_id
                data_cutoff = loader.frozen_data_cutoff()
        except CampaignExecutorError:
            raise
        except Exception as exc:  # noqa: BLE001 - typed refusal (exit 2)
            # DatasetIdMismatch and every other identity-resolution
            # refusal is an OPERATIONAL refusal (p32 §3.4): fail-closed
            # with exit 2, never a leg failure and never a downgrade.
            raise CampaignExecutorError(
                f"campaign identity resolution refused: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not dataset_id or not data_cutoff:
            raise CampaignExecutorError(
                "campaign launch requires a confirmed composite identity "
                "(dataset_id and data_cutoff); refusing to run legs "
                "without one (p32 §3.2/I4)"
            )
        self._dataset_id = str(dataset_id)
        self._data_cutoff = str(data_cutoff)
        return self._dataset_id, self._data_cutoff

    def _leg_spec(self, leg: CampaignLeg, data_config: Any):
        """Build the leg's frozen RunSpec through the single runtime
        authority (the same factory the evaluation entry uses).  A
        resolution failure is an operational refusal (exit 2)."""

        from ashare_data.config import (
            load_config,
            make_backtest_config,
            make_model_config,
            make_protocol_config,
        )

        try:
            raw = load_config(leg.config, project_root=self.repo_root)
            model_config = make_model_config(raw)
            backtest_config = make_backtest_config(raw)
            proto_cfg = make_protocol_config(raw)
            tier = getattr(proto_cfg, leg.tier)
            # The evaluation entry's budget identity formula
            # (evaluation.main): requested = tier.steps * tier.batch_size,
            # minimum one.
            requested_budget = max(1, int(tier.steps) * int(tier.batch_size))
            return resolve_runtime_runspec(
                dataset_id=self._dataset_id,
                data_cutoff=self._data_cutoff,
                purpose="historical_research",
                data_config=data_config,
                backtest_config=backtest_config,
                requested_budget=requested_budget,
                n_folds=len(proto_cfg.folds),
                research_domain=proto_cfg.domain,
                seeds=tuple(int(s) for s in proto_cfg.seeds),
                searcher=model_config.searcher,
                max_formula_len=int(model_config.max_formula_len),
                require_clean_tree=False,
            )
        except CampaignExecutorError:
            raise
        except Exception as exc:  # noqa: BLE001 - typed refusal (exit 2)
            raise CampaignExecutorError(
                f"leg {leg.leg_id!r} RunSpec resolution refused: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _leg_identity(self, leg: CampaignLeg, spec) -> tuple[str, dict[str, Any]]:
        """The content identity a reuse decision must match exactly
        (config/snapshot/code/seed/budget + the leg's research semantics,
        p32 §3.2/I3, rev 1: the frozen ``spec_id`` is part of the
        identity — the runtime_overrides/.env deep-merge channel and a
        dirty tree can change the effective run parameters without
        touching the leg YAML, and ``spec_id`` is computed from the
        resolved spec either way)."""

        config_bytes = Path(leg.config).read_bytes()
        payload = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "dataset_id": self._dataset_id,
            "data_cutoff": self._data_cutoff,
            "git_commit": spec.git_commit,
            "spec_id": spec.spec_id,
            "seeds": [int(seed) for seed in spec.seeds],
            "requested_budget": int(spec.requested_budget),
            "frequency": str(spec.frequency),
            "horizon": int(spec.horizon),
            "research_domain": str(spec.research_domain),
            "searcher": str(spec.searcher),
            "max_formula_len": int(spec.max_formula_len),
            "lifecycle_mode": "engineering",
        }
        return content_hash("campaign_leg_identity", payload), payload

    # -- state ---------------------------------------------------------------

    def _load_state(self) -> dict[str, Any] | None:
        state = read_json_safe(self.state_path)
        if state is None:
            return None
        version = state.get("campaign_state_schema_version")
        if version != CAMPAIGN_STATE_SCHEMA_VERSION:
            raise CampaignExecutorError(
                f"unknown campaign state schema version {version!r} in "
                f"{self.state_path} (this code understands "
                f"<= {CAMPAIGN_STATE_SCHEMA_VERSION})"
            )
        # Review A#7: the state file belongs to exactly one campaign — a
        # different manifest pointed at the same state_dir would mix leg
        # records and cross-campaign reuse.  Fail closed (exit 2).
        state_campaign = state.get("campaign_id")
        if state_campaign != self.manifest.name:
            raise CampaignExecutorError(
                f"campaign state {self.state_path} belongs to campaign "
                f"{state_campaign!r}, not {self.manifest.name!r}; refusing "
                "to mix leg records across campaigns (p32 §3.6, rev 1)"
            )
        return state

    def _persist_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now_iso()
        atomic_write_json(self.state_path, state)

    def _init_state(self) -> dict[str, Any]:
        state = {
            "campaign_state_schema_version": CAMPAIGN_STATE_SCHEMA_VERSION,
            "campaign_id": self.manifest.name,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "identity_basis": {
                "dataset_id": self._dataset_id,
                "data_cutoff": self._data_cutoff,
            },
            "budgets": {
                "row_wall_clock_budget_s": (
                    self.manifest.row_wall_clock_budget_s
                ),
                "row_peak_rss_budget_mb": self.manifest.row_peak_rss_budget_mb,
                "campaign_wall_clock_budget_s": (
                    self.manifest.campaign_wall_clock_budget_s
                ),
            },
            "legs": {},
            "summary": None,
        }
        self._persist_state(state)
        return state

    # -- per-leg execution ------------------------------------------------------

    def _leg_command(self, leg: CampaignLeg, row: TaskRow) -> list[str]:
        if leg.command is None:
            return default_evaluation_command(leg)
        if callable(leg.command):
            return [str(part) for part in leg.command(row)]
        return [str(part) for part in leg.command]

    def _derived_done_path(self, leg: CampaignLeg) -> Path:
        return self.done_dir / f"{leg.leg_id}.done"

    def _write_derived_done(self, leg: CampaignLeg, identity_hash: str) -> None:
        """Derived display only (p32 §3.2): written from terminal success,
        never read for decisions."""

        path = self._derived_done_path(leg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"derived from campaign state {self.state_path} at {_now_iso()}\n"
            f"leg={leg.leg_id} identity={identity_hash}\n",
            encoding="utf-8",
        )

    def _marker(self, name: str, detail: str) -> None:
        """Append the human-readable campaign history (p21 markers.log
        convention; operational continuity for the R1 runbook)."""

        line = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {name} {detail}"
        )
        try:
            self.marker_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.marker_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            logger.warning("campaign marker write failed: {}", exc)
        logger.info("campaign | {}", line)

    def _preflight(self, leg: CampaignLeg) -> None:
        """A01 C5/C6 pre-flight: refuse to start a leg when a required
        input path is missing (read-only config resolver check)."""

        from ashare_data.config import main as config_main

        exit_code = config_main(
            ["--config", str(Path(leg.config).resolve()), "--check"]
        )
        if exit_code != 0:
            raise CampaignExecutorError(
                f"leg {leg.leg_id!r} failed the A01 path pre-flight "
                f"(config --check exit {exit_code}); required input missing"
            )

    def _leg_telemetry(self, leg: CampaignLeg) -> dict[str, Any]:
        """Telemetry extracted from the leg's own protocol artifact
        (p32 §3.7/I11): requested/consumed unique semantic evaluations,
        proposal/duplicate/invalid counts and the candidate count.  A
        field is ``None`` unless its source is present — per-field (a
        row without a field contributes nothing and never turns the
        field into a fabricated 0)."""

        payload = read_json_safe(leg.output)
        if not isinstance(payload, dict):
            return {
                "requested_budget": None,
                "consumed_budget": None,
                "proposal_count": None,
                "invalid_proposals": None,
                "semantic_duplicates": None,
                "n_candidates": None,
            }
        keys = (
            "requested_budget",
            "consumed_budget",
            "proposal_count",
            "invalid_proposals",
            "semantic_duplicates",
        )
        sums = {key: 0 for key in keys}
        found = {key: False for key in keys}
        for row in payload.get("rows", []) or []:
            search_result = (
                row.get("search_result") if isinstance(row, dict) else None
            )
            if not isinstance(search_result, dict):
                continue
            for key in keys:
                value = search_result.get(key)
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    sums[key] += int(value)
                    found[key] = True
        telemetry: dict[str, Any] = {
            key: (sums[key] if found[key] else None) for key in keys
        }
        n_candidates = payload.get("n_candidates")
        telemetry["n_candidates"] = (
            int(n_candidates)
            if isinstance(n_candidates, (int, float))
            and not isinstance(n_candidates, bool)
            else None
        )
        return telemetry

    def _execute_leg(
        self,
        leg: CampaignLeg,
        data_config: Any,
        state: dict[str, Any],
        record: dict[str, Any],
        spec,
        identity_hash: str,
        identity_payload: dict[str, Any],
    ) -> str:
        """Run one leg through the TaskExecutor and update its state
        record.  Returns the terminal leg status."""

        resume_candidate = bool(
            record.get("run_id")
            and record.get("status") in ("running", "aborted")
        )
        identity_stale = resume_candidate and (
            record.get("identity_hash") != identity_hash
        )
        resume = bool(resume_candidate and not identity_stale)
        update: dict[str, Any] = {}
        if identity_stale:
            # The leg died in flight and its identity changed since: the
            # recovery decision (NEW run; the old run stays discoverable)
            # is explicit in the state, not only in the log (p32 §3.5
            # rev 1, review A#5).
            update["superseded_run_id"] = record.get("run_id")
            update["superseded_reason"] = (
                "identity changed while the attempt was in flight; the "
                "old run keeps its open trial as evidence"
            )
        if resume:
            run_id = record["run_id"]
        else:
            run_id = f"leg-{leg.leg_id}-{spec.spec_id[:8]}-{_short_id()}"

        store = RunStore(data_config.data_dir)
        if resume and self._stale_campaign_lock_cleared:
            # The previous campaign driver was verified dead when this
            # runner cleared the stale CAMPAIGN lock (p32 §3.6), so any
            # writer lock this driver's killed predecessor left on the
            # leg's own RunStore run is stale by the same verified fact.
            # Clearing it here is the explicit recovery action — never a
            # blind unlink on a live writer.
            if store.clear_stale_lock(spec.spec_id, run_id):
                logger.warning(
                    "campaign leg {} cleared the stale writer lock of run "
                    "{} (previous campaign driver verified dead)",
                    leg.leg_id,
                    run_id,
                )
        row = TaskRow(
            seed=0,
            backend="evaluation",
            track="campaign",
            fold=leg.leg_id,
            payload={
                "leg_id": leg.leg_id,
                "stages": list(LEG_STAGE_PLAN),
                "identity_hash": identity_hash,
            },
        )
        try:
            executor = TaskExecutor(
                store=store,
                spec=spec,
                rows=[row],
                command_builder=lambda _row: self._leg_command(leg, _row),
                run_id=run_id,
                resume=resume,
                heartbeat_interval_s=self._heartbeat_interval_s,
                heartbeat_timeout_s=self._heartbeat_timeout_s,
                poll_interval_s=self._poll_interval_s,
                kill_grace_s=self._kill_grace_s,
                row_wall_clock_budget_s=self.manifest.row_wall_clock_budget_s,
                row_peak_rss_budget_mb=self.manifest.row_peak_rss_budget_mb,
            )
        except (
            RunStoreError,
            TaskExecutorError,
            LedgerIntegrityError,
        ) as exc:
            if resume:
                # The stored run's frozen spec no longer matches (identity
                # changed under an in-flight record): never force-attach.
                # Open a NEW run; the superseded pointer + refusal reason
                # stay in the state (p32 §3.5 rev 1), the old evidence
                # stays untouched.
                logger.warning(
                    "campaign leg {} attach of run {} refused ({}); "
                    "opening a new run (old evidence preserved)",
                    leg.leg_id,
                    run_id,
                    exc,
                )
                return self._execute_leg(
                    leg,
                    data_config,
                    state,
                    {
                        **record,
                        "run_id": None,
                        "status": "failed",
                        "superseded_run_id": run_id,
                        "superseded_reason": f"attach refused: {exc}",
                    },
                    spec,
                    identity_hash,
                    identity_payload,
                )
            # A ledger-integrity or store refusal on a fresh attempt is an
            # OPERATIONAL refusal (evidence problem), not a leg work
            # failure: exit 2, never a silently failed leg (review A#2).
            raise CampaignExecutorError(
                f"leg {leg.leg_id!r} executor construction refused: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # I1 (p32 §3.1): the attempt start is registered immediately.
        self._mark_leg(
            state,
            leg,
            {
                "status": "running",
                "reused": False,
                "identity_hash": identity_hash,
                "identity": identity_payload,
                "run_id": run_id,
                "spec_id": spec.spec_id,
                **update,
            },
        )
        try:
            payload = executor.execute()
        except KeyboardInterrupt:
            self._mark_leg(
                state,
                leg,
                {
                    "status": "aborted",
                    "error": "aborted by user (KeyboardInterrupt)",
                },
            )
            raise
        except TaskExecutorError as exc:
            # Attach-time reconciliation can raise TaskExecutorError
            # (plan drift / concurrent supervisor) — an operational
            # refusal, never a leg work failure (review A#2).
            self._mark_leg(
                state,
                leg,
                {
                    "status": "failed",
                    "error": f"executor refused: {exc}",
                },
            )
            raise CampaignExecutorError(
                f"leg {leg.leg_id!r} executor refused: {exc}"
            ) from exc
        row_result = payload["rows"][row.row_id]
        status = row_result["status"]
        trial_id = row_result.get("trial_id")
        # Review A#11: the executor's own payload carries the row metrics
        # (single authority; no layout re-derivation on this side).
        metrics = row_result.get("metrics") or {}
        update = {
            "status": status,
            "trial_id": trial_id,
            "error": row_result.get("error"),
            "wall_seconds": metrics.get("wall_seconds"),
            "peak_rss_mb": metrics.get("peak_rss_mb"),
            "active_phase": metrics.get("active_phase"),
        }
        if status == "succeeded":
            update["telemetry"] = self._leg_telemetry(leg)
            update["output"] = str(leg.output)
            self._write_derived_done(leg, identity_hash)
        self._mark_leg(state, leg, update)
        return status

    def _mark_leg(
        self, state: dict[str, Any], leg: CampaignLeg, update: dict[str, Any]
    ) -> None:
        legs = state.setdefault("legs", {})
        record = dict(legs.get(leg.leg_id) or {})
        record.update(update)
        record["leg_id"] = leg.leg_id
        record["updated_at"] = _now_iso()
        legs[leg.leg_id] = record
        self._persist_state(state)
        self._marker(
            leg.leg_id.upper(),
            f"{update.get('status')} run={update.get('run_id')}",
        )

    # -- main loop ---------------------------------------------------------------

    def run(self, leg_ids: Sequence[str] | None = None) -> dict[str, Any]:
        """Run the manifest's legs (or the named subset) to terminal state.

        Returns the campaign summary including ``exit_code`` (p32 §3.4).
        Operational refusals raise :class:`CampaignExecutorError`; a user
        abort re-raises :class:`KeyboardInterrupt` after recording the
        in-flight leg as ``aborted`` and persisting the aborted summary.
        """

        selected = list(self.manifest.legs)
        if leg_ids is not None:
            known = {leg.leg_id for leg in self.manifest.legs}
            unknown = [leg_id for leg_id in leg_ids if leg_id not in known]
            if unknown:
                raise CampaignExecutorError(
                    f"unknown campaign legs: {unknown}"
                )
            wanted = set(leg_ids)
            selected = [leg for leg in self.manifest.legs if leg.leg_id in wanted]
        self._acquire_lock()
        try:
            data_configs = self._resolve_data_configs()
            self._resolve_identity()
            state = self._load_state()
            if state is None:
                state = self._init_state()
            state["identity_basis"] = {
                "dataset_id": self._dataset_id,
                "data_cutoff": self._data_cutoff,
            }
            state.setdefault("legs", {})
            started = time.perf_counter()
            try:
                for leg in selected:
                    record = state["legs"].get(leg.leg_id) or {}
                    status_before = record.get("status")
                    # Review A#2: a config/spec resolution failure is an
                    # OPERATIONAL refusal (exit 2), never a leg failure —
                    # typed wrapping happens in _resolve_data_configs /
                    # _resolve_identity / _leg_spec callers below.
                    spec = self._leg_spec(leg, data_configs[leg.leg_id])
                    identity_hash, identity_payload = self._leg_identity(
                        leg, spec
                    )
                    if (
                        status_before == "succeeded"
                        and record.get("identity_hash") == identity_hash
                        and record.get("output") == str(leg.output)
                    ):
                        # Rev 1 (review A#3): reuse additionally requires
                        # the recorded output pointer to match — an
                        # identity cannot silently vouch for results
                        # written somewhere else.
                        record.update({"reused": True})
                        state["legs"][leg.leg_id] = record
                        self._persist_state(state)
                        self._marker(
                            leg.leg_id.upper(),
                            "SKIP reuse: identity match (done file display only)",
                        )
                        continue
                    record.update({"reused": False})
                    if status_before == "failed" and not self._retry_failed:
                        record.update(
                            {
                                "status": "failed",
                                "reason": (
                                    "terminal failed attempt preserved; use "
                                    "--retry-failed to open a new run "
                                    "(p32 §3.5)"
                                ),
                            }
                        )
                        state["legs"][leg.leg_id] = record
                        self._persist_state(state)
                        self._marker(
                            leg.leg_id.upper(),
                            "SKIP failed: old failure preserved (no auto retry)",
                        )
                        continue
                    if (
                        self.manifest.campaign_wall_clock_budget_s is not None
                        and time.perf_counter() - started
                        > float(self.manifest.campaign_wall_clock_budget_s)
                    ):
                        # Rev 1 (review A#6): a stale "running" record is an
                        # INTERRUPTED leg from a killed prior driver — the
                        # attach-same-run recovery target (p32 §3.5).  The
                        # sweep must not demote it to not_run (that would
                        # orphan the open trial); it stays attachable.
                        for remaining in selected:
                            remaining_status = (
                                state["legs"]
                                .get(remaining.leg_id, {})
                                .get("status")
                            )
                            if remaining_status in (None, "not_run"):
                                self._mark_leg(
                                    state,
                                    remaining,
                                    {
                                        "status": "not_run",
                                        "reason": "campaign wall-clock budget "
                                        "exhausted before this leg "
                                        "(stopped_budget)",
                                    },
                                )
                        break
                    # Review B#2 (p32 §3.4): the A01 pre-flight refusal is
                    # an OPERATIONAL refusal (exit 2), not a leg failure —
                    # it propagates out of run().
                    self._preflight(leg)
                    self._execute_leg(
                        leg,
                        data_configs[leg.leg_id],
                        state,
                        record,
                        spec,
                        identity_hash,
                        identity_payload,
                    )
            except KeyboardInterrupt:
                self._finalize(state, selected, started, aborted=True)
                raise
            return self._finalize(state, selected, started)
        finally:
            self._release_lock()

    def _finalize(
        self,
        state: dict[str, Any],
        selected: Sequence[CampaignLeg],
        started: float,
        *,
        aborted: bool = False,
    ) -> dict[str, Any]:
        legs_state = state.get("legs", {})
        counts = {
            "total": len(selected),
            "succeeded": 0,
            "failed": 0,
            "aborted": 0,
            "reused": 0,
            "not_run": 0,
            "running": 0,
        }
        statuses: dict[str, str] = {}
        for leg in selected:
            record = legs_state.get(leg.leg_id) or {}
            status = str(record.get("status") or "not_run")
            statuses[leg.leg_id] = status
            if record.get("reused"):
                counts["reused"] += 1
            if status in counts:
                counts[status] += 1
        if aborted or any(s == "aborted" for s in statuses.values()):
            # A user abort is sticky for the campaign (review B#5): a
            # later budget stop must not downgrade the abort evidence.
            campaign_status = "aborted"
            exit_code = EXIT_ABORTED
        elif any(
            s in ("failed", "running") for s in statuses.values()
        ):
            # A leftover "running" leg is an interrupted attempt the cap
            # sweep deliberately left attachable (rev 1): the campaign is
            # not completed — recovery happens on the next launch.
            campaign_status = "failed"
            exit_code = EXIT_FAILED
        elif any(s == "not_run" for s in statuses.values()):
            campaign_status = "stopped_budget"
            exit_code = EXIT_FAILED
        else:
            campaign_status = "completed"
            exit_code = EXIT_COMPLETED
        summary = {
            "campaign": self.manifest.name,
            "campaign_status": campaign_status,
            "exit_code": exit_code,
            "counts": counts,
            "legs": {
                leg.leg_id: {
                    "status": statuses[leg.leg_id],
                    "reused": bool(
                        (legs_state.get(leg.leg_id) or {}).get("reused")
                    ),
                    "run_id": (legs_state.get(leg.leg_id) or {}).get("run_id"),
                    "error": (legs_state.get(leg.leg_id) or {}).get("error"),
                    "reason": (legs_state.get(leg.leg_id) or {}).get("reason"),
                }
                for leg in selected
            },
            "wall_seconds": time.perf_counter() - started,
        }
        state["summary"] = summary
        self._persist_state(state)
        logger.info(
            "campaign {} status={} exit={} counts={}",
            self.manifest.name,
            campaign_status,
            exit_code,
            counts,
        )
        return summary
