"""Reliable long-task executor (E3, t5): one independent subprocess per
``(seed, backend, track, fold)`` row, bound to the run's RunStore ledger.

Replaces ad-hoc WMI/batch scheduling for long research grids.  The
module is *transport and evidence only*: the row command is opaque
(``command_builder(row) -> argv``), so no research semantics live here.

Contract (AGENTS.md §4.3/§4.5):

* **Independent subprocess per row** — the supervisor spawns
  ``python -m ashare_model.task_executor worker <job.json>`` per row; a
  worker (or its command) crashing cannot affect any other row.  Exit
  code 0 of the row command means the row's work succeeded; anything
  else is a recorded measurement, never silently dropped.
* **Heartbeat + resource metrics** — the worker atomically rewrites
  ``<row>.heartbeat.json`` every ``heartbeat_interval_s`` with CPU%,
  RSS and wall time (its own and its command child's); the supervisor
  kills and marks a row whose last heartbeat is older than
  ``heartbeat_timeout_s``.  A worker that never gets to its first beat
  is covered by the same clock (age counts from the attempt start), so
  ``heartbeat_timeout_s`` must stay far above process-spawn time
  (default 600 s).
* **RunStore binding** — every attempt is a trial in the run's bound,
  hash-chained ``trials.jsonl`` (spec_id/run_id identity via
  :class:`ashare_model.run_store.RunStore`); the row plan is registered
  as a content-addressed ``task_executor_plan`` artifact and a resume
  with a different plan is refused (fail-closed identity).
* **Row-level atomic results** — ``<row>.result.json`` is written tmp +
  ``os.replace`` (:func:`ashare_data.io_utils.atomic_write_json`, the
  single atomic-JSON authority): an injected write failure leaves zero
  half rows.
* **Stage telemetry (B03, p32)** — the worker exports the row phase file
  path to the row command through ``TASK_EXECUTOR_ROW_PHASE_FILE``; the
  command records coarse stage milestones via :func:`row_phase`
  (``loading / feature_build / search / evaluation / artifact_write``).
  The phase journal flows into the row result (``phases`` / ``phase``)
  and from there into the ledger trial metrics — best-effort telemetry
  that must never break a row.
* **Row budgets (B03, p32)** — optional ``row_wall_clock_budget_s`` /
  ``row_peak_rss_budget_mb`` are enforced by the worker on its own
  command child: exceeding either terminates the child and records the
  row failed with a budget-specific error (distinct from crashes and
  heartbeat timeouts).  ``None`` means uncapped; values are frozen in
  the caller's campaign manifest, never hardcoded here.
* **Ownership-verified kills (B03, p32)** — before the supervisor tears
  a worker tree down (heartbeat timeout or user abort) it verifies the
  target pid's command line still references this row's job file, so a
  recycled pid can never route a kill into an unrelated process.
* **Resume reconciliation** — a terminal ledger record (succeeded OR
  failed) is final for the run: failed trials stay in the chain and are
  never overwritten, renamed or re-executed; redoing a failed row is a
  *rerun*, which means a NEW run id / output directory (fresh runs get
  fresh ``executor_rows`` dirs by construction).  An *interrupted*
  attempt (open trial with no valid result file — the on-disk state a
  hard kill leaves) is closed as failed and the row re-executed once;
  an interrupted attempt whose worker did leave a valid result is
  adopted from that result without re-running the row.
* **Single writer** — inherited from RunStore: a second executor on the
  same run raises :class:`ashare_model.run_store.RunLockedError`.

Telemetry reuses the existing authorities only (RunStore ledger +
artifacts, per-row JSON files, loguru); no second logging/ledger path.
Dependency direction (§9): ashare_model + ashare_data primitives only,
no CLI/Web imports.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import psutil
from loguru import logger

from ashare_data.io_utils import atomic_write_json, read_json_safe

from .ledger import ExperimentLedger
from .run_store import RunStore
from .runspec import RunSpec

TASK_EXECUTOR_SCHEMA_VERSION = 1
"""Schema version of heartbeat / result / plan payloads (fail-closed on
read: unknown versions are refused, never leniently parsed)."""

WORKER_POLL_INTERVAL_S = 0.1
DEFAULT_HEARTBEAT_INTERVAL_S = 30.0
DEFAULT_HEARTBEAT_TIMEOUT_S = 600.0
DEFAULT_KILL_GRACE_S = 3.0

_PLAN_ARTIFACT_TYPE = "task_executor_plan"
_RESULT_SUFFIX = ".result.json"
_HEARTBEAT_SUFFIX = ".heartbeat.json"
_JOB_SUFFIX = ".job.json"
_LOG_SUFFIX = ".log"
_PHASES_SUFFIX = ".phases.jsonl"

#: Environment variable exporting the row phase journal path to the row
#: command (B03 p32 §3.1).  Unset outside a TaskExecutor worker, so
#: :func:`row_phase` is a no-op for every direct caller.
ROW_PHASE_ENV = "TASK_EXECUTOR_ROW_PHASE_FILE"


class TaskExecutorError(Exception):
    """Executor contract violation (fail-closed)."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


@dataclass(frozen=True)
class TaskRow:
    """One (seed, backend, track, fold) unit of work."""

    seed: int
    backend: str
    track: str
    fold: str
    fold_train_end: str | None = None
    fold_test_end: str | None = None
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        for name, value in (
            ("backend", self.backend),
            ("track", self.track),
            ("fold", self.fold),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")

    @property
    def row_id(self) -> str:
        """Logical identity inside payloads and the ledger."""

        return f"{self.seed}:{self.backend}:{self.track}:{self.fold}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "backend": str(self.backend),
            "track": str(self.track),
            "fold": str(self.fold),
            "fold_train_end": self.fold_train_end,
            "fold_test_end": self.fold_test_end,
            "payload": dict(self.payload or {}),
        }


def row_slug(row: TaskRow) -> str:
    """Filesystem-safe file stem for a row (":" is illegal on Windows)."""

    return _slug_part(row.row_id)


def build_plan(rows: Sequence[TaskRow]) -> tuple[TaskRow, ...]:
    """Validate a row plan fail-closed: non-empty, unique row ids, unique
    file slugs (two distinct rows that map to one filename are a contract
    violation, never a silent overwrite)."""

    plan = tuple(rows)
    if not plan:
        raise ValueError("task plan must not be empty")
    seen_ids: set[str] = set()
    seen_slugs: set[str] = set()
    for row in plan:
        if not isinstance(row, TaskRow):
            raise TypeError(f"plan entries must be TaskRow, got {type(row)!r}")
        if row.row_id in seen_ids:
            raise ValueError(
                f"duplicate plan row (seed, backend, track, fold): {row.row_id}"
            )
        slug = row_slug(row)
        if slug in seen_slugs:
            raise ValueError(
                f"plan rows {row.row_id} collide on file slug {slug!r}; "
                "refusing to overwrite row files (fail-closed)"
            )
        seen_ids.add(row.row_id)
        seen_slugs.add(slug)
    return plan


def write_row_result(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one row result (tmp + os.replace — a crash or an
    injected write failure leaves zero half rows)."""

    atomic_write_json(path, dict(payload))


def _read_valid_result(path: Path, row: TaskRow) -> dict[str, Any] | None:
    """Load a row result fail-closed; ``None`` when missing or invalid."""

    payload = read_json_safe(path)
    if not isinstance(payload, dict):
        return None
    if payload.get("result_schema_version") != TASK_EXECUTOR_SCHEMA_VERSION:
        return None
    if payload.get("row_id") != row.row_id:
        return None
    if payload.get("status") not in ("succeeded", "failed"):
        return None
    return payload


def row_phase(name: str) -> None:
    """Record one coarse stage milestone for the running executor row.

    Writes one JSON line (``{"phase": ..., "entered_at": ...}``) to the
    row phase journal the worker exported via ``ROW_PHASE_ENV``.  A no-op
    unless called inside a TaskExecutor worker's row command, and a
    failed write is logged — never raised — so telemetry cannot break a
    research row (B03 p32 §3.1).
    """

    path = os.environ.get(ROW_PHASE_ENV)
    if not path:
        return
    line = json.dumps(
        {"phase": str(name), "entered_at": _now_iso()},
        ensure_ascii=False,
    )
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError as exc:
        logger.warning(
            "task_executor row phase write failed ({}): {}", name, exc
        )


def read_phase_journal(path: str | Path | None) -> list[dict[str, Any]]:
    """Tolerant phase journal reader: malformed lines (a killed child can
    tear the last line) are skipped, never trusted."""

    if path is None:
        return []
    journal = Path(path)
    if not journal.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("phase"):
            entries.append(
                {
                    "phase": str(payload["phase"]),
                    "entered_at": str(payload.get("entered_at") or ""),
                }
            )
    return entries


def _process_cmdline_has(pid: int, needle: str) -> bool:
    """Whether the process behind ``pid`` still names ``needle`` on its
    command line (the ownership check in front of every supervisor kill;
    ``False`` when the pid is gone or unreadable)."""

    try:
        cmdline = psutil.Process(pid).cmdline()
    except psutil.Error:
        return False
    return any(needle in str(part) for part in cmdline)


def _subtree_rss_mb(proc: psutil.Process) -> float | None:
    """Total resident memory of the command process and its whole
    descendant tree (MB), or ``None`` when nothing could be read.

    The row command may be a launcher whose real interpreter is a
    grandchild (venv shims), so per-child RSS alone under-measures the
    row's footprint and would silently never trip a memory budget.
    """

    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        procs = [proc]
    total: float | None = None
    for member in procs:
        try:
            rss = member.memory_info().rss / (1024.0 * 1024.0)
        except psutil.Error:
            continue
        total = rss if total is None else total + rss
    return total


def _kill_tree(pid: int, grace_s: float) -> None:
    """Terminate a worker and its whole command subtree (best effort)."""

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    victims = set()
    try:
        victims.update(child.pid for child in parent.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    victims.add(pid)
    for victim_pid in victims:
        try:
            psutil.Process(victim_pid).terminate()
        except psutil.NoSuchProcess:
            pass
    # A victim can exit between pid_exists() and Process(): guard the
    # reaping sweep exactly like the terminate loop (a raw NoSuchProcess
    # here would otherwise kill the supervisor's own unwind).
    procs = []
    for victim_pid in victims:
        try:
            procs.append(psutil.Process(victim_pid))
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(procs, timeout=grace_s)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


# ------------------------------------------------------------------ worker


def _worker_beat(
    *,
    row_id: str,
    worker_pid: int,
    child_pid: int | None,
    self_proc: "psutil.Process",
    child_proc: "psutil.Process | None",
    started: float,
    heartbeat_path: Path,
) -> None:
    """One atomic heartbeat: worker + command-child CPU/RSS/wall."""

    def _metrics(proc: "psutil.Process | None") -> tuple[Any, float | None]:
        if proc is None:
            return None, None
        try:
            return proc.cpu_percent(None), proc.memory_info().rss / (1024.0 * 1024.0)
        except psutil.Error:
            return None, None

    cpu_percent, rss_mb = _metrics(self_proc)
    child_cpu_percent, child_rss_mb = _metrics(child_proc)
    atomic_write_json(
        heartbeat_path,
        {
            "heartbeat_schema_version": TASK_EXECUTOR_SCHEMA_VERSION,
            "row_id": row_id,
            "pid": worker_pid,
            "child_pid": child_pid,
            "ts": _now_iso(),
            "wall_seconds": time.perf_counter() - started,
            "cpu_percent": cpu_percent,
            "rss_mb": rss_mb,
            "child_cpu_percent": child_cpu_percent,
            "child_rss_mb": child_rss_mb,
        },
    )


def worker_main(job_path: str | Path) -> int:
    """Run one row inside its dedicated subprocess.

    Loads the supervisor-written job file (fail-closed), spawns the row
    command with output appended to the row log, emits periodic
    heartbeats with resource metrics, and writes the row result
    atomically exactly once (even on failure paths).
    """

    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("job_schema_version") != TASK_EXECUTOR_SCHEMA_VERSION:
        raise TaskExecutorError(
            f"unknown job schema version {job.get('job_schema_version')!r} "
            f"in {job_path}"
        )
    row_id = str(job["row_id"])
    command = list(job["command"])
    if not command:
        raise TaskExecutorError(f"empty row command for {row_id}")
    result_path = Path(job["result_path"])
    heartbeat_path = Path(job["heartbeat_path"])
    log_path = Path(job["log_path"])
    interval_s = float(job["heartbeat_interval_s"])
    cwd = job.get("cwd")
    phase_path = job.get("phase_path")
    row_wall_clock_budget_s = job.get("row_wall_clock_budget_s")
    row_peak_rss_budget_mb = job.get("row_peak_rss_budget_mb")

    worker_pid = os.getpid()
    self_proc = psutil.Process(worker_pid)
    started_perf = time.perf_counter()
    started_iso = _now_iso()

    child: subprocess.Popen | None = None
    child_proc: "psutil.Process | None" = None
    exit_code: int | None = None
    error: str | None = None
    peak_rss_mb: float | None = None
    cpu_seconds = 0.0
    budget_error: str | None = None
    beat_stop = threading.Event()

    def beat_loop() -> None:
        while not beat_stop.wait(interval_s):
            try:
                _worker_beat(
                    row_id=row_id,
                    worker_pid=worker_pid,
                    child_pid=child.pid if child is not None else None,
                    self_proc=self_proc,
                    child_proc=child_proc,
                    started=started_perf,
                    heartbeat_path=heartbeat_path,
                )
            except Exception:  # a lost beat must never kill the worker
                pass

    try:
        log_handle = log_path.open("ab")
        try:
            env = dict(os.environ)
            if phase_path:
                # B03 p32 §3.1: the row command learns its phase journal
                # only through the environment (additive, optional).
                env[ROW_PHASE_ENV] = str(phase_path)
            child = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                env=env,
            )
            child_proc = psutil.Process(child.pid)
        finally:
            log_handle.close()

        _worker_beat(  # first beat: the supervisor's timeout clock anchor
            row_id=row_id,
            worker_pid=worker_pid,
            child_pid=child.pid,
            self_proc=self_proc,
            child_proc=child_proc,
            started=started_perf,
            heartbeat_path=heartbeat_path,
        )
        beat_thread = threading.Thread(target=beat_loop, daemon=True)
        beat_thread.start()
        try:
            while child.poll() is None:
                try:
                    rss = _subtree_rss_mb(child_proc)
                    # A child that dies mid-sample yields None: keep the
                    # previous peak instead of comparing None (the row is
                    # already terminal — poll() catches it next turn).
                    if rss is not None:
                        peak_rss_mb = (
                            rss
                            if peak_rss_mb is None
                            else max(peak_rss_mb, rss)
                        )
                except psutil.Error:
                    rss = None
                try:
                    times = child_proc.cpu_times()
                    cpu_seconds = float(times.user) + float(times.system)
                except psutil.Error:
                    pass
                # B03 p32 §3.3: the worker enforces the row budgets on its
                # own child — a budget kill is a recorded measurement with
                # a budget-specific reason, never a silent drop.
                if budget_error is None and (
                    row_wall_clock_budget_s is not None
                    and time.perf_counter() - started_perf
                    > float(row_wall_clock_budget_s)
                ):
                    budget_error = (
                        f"row {row_id}: wall-clock budget "
                        f"{float(row_wall_clock_budget_s):.1f}s exceeded "
                        f"after {time.perf_counter() - started_perf:.1f}s "
                        "(command terminated by the worker)"
                    )
                if budget_error is None and (
                    row_peak_rss_budget_mb is not None
                    and rss is not None
                    and rss > float(row_peak_rss_budget_mb)
                ):
                    budget_error = (
                        f"row {row_id}: peak RSS budget "
                        f"{float(row_peak_rss_budget_mb):.1f} MB exceeded "
                        f"({rss:.1f} MB resident; command terminated by "
                        "the worker)"
                    )
                if budget_error is not None:
                    # Kill the whole command subtree: the row command may
                    # be a launcher (venv shim) whose real work runs in a
                    # grandchild — terminating the direct child alone
                    # would orphan the real work process.
                    _kill_tree(child.pid, DEFAULT_KILL_GRACE_S)
                    try:
                        child.wait(timeout=DEFAULT_KILL_GRACE_S + 5.0)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    break
                time.sleep(WORKER_POLL_INTERVAL_S)
            exit_code = child.returncode
        finally:
            beat_stop.set()
            beat_thread.join(timeout=5.0)
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        error = f"{type(exc).__name__}: {exc}"
        beat_stop.set()

    if peak_rss_mb is None and child_proc is not None:
        try:  # fast-exiting children: sample once after reaping
            peak_rss_mb = _subtree_rss_mb(child_proc)
        except psutil.Error:
            peak_rss_mb = None
    if budget_error is not None:
        error = budget_error
    if exit_code is None and error is None:
        error = "row command did not produce an exit code"
    status = "succeeded" if error is None and exit_code == 0 else "failed"
    if status == "failed" and error is None:
        error = f"row {row_id}: command exited {exit_code}"

    phases = read_phase_journal(phase_path)
    write_row_result(
        result_path,
        {
            "result_schema_version": TASK_EXECUTOR_SCHEMA_VERSION,
            "row_id": row_id,
            "status": status,
            "exit_code": exit_code,
            "wall_seconds": time.perf_counter() - started_perf,
            "cpu_seconds": cpu_seconds,
            "peak_rss_mb": peak_rss_mb,
            "error": error,
            "started_at": started_iso,
            "finished_at": _now_iso(),
            "worker_pid": worker_pid,
            "child_pid": child.pid if child is not None else None,
            "phases": phases,
            "phase": phases[-1]["phase"] if phases else None,
            "author": "worker",
        },
    )
    return 0 if status == "succeeded" else 1


# -------------------------------------------------------------- supervisor


class TaskExecutor:
    """Single-writer supervisor over one RunStore run's row plan."""

    def __init__(
        self,
        *,
        store: RunStore,
        spec: RunSpec,
        rows: Sequence[TaskRow],
        command_builder: Callable[[TaskRow], Sequence[str]],
        run_id: str | None = None,
        resume: bool = False,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
        heartbeat_timeout_s: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
        poll_interval_s: float = 0.2,
        kill_grace_s: float = DEFAULT_KILL_GRACE_S,
        row_wall_clock_budget_s: float | None = None,
        row_peak_rss_budget_mb: float | None = None,
    ):
        self._plan = build_plan(rows)
        if heartbeat_interval_s <= 0 or heartbeat_timeout_s <= 0:
            raise ValueError("heartbeat intervals must be positive")
        for name, value in (
            ("row_wall_clock_budget_s", row_wall_clock_budget_s),
            ("row_peak_rss_budget_mb", row_peak_rss_budget_mb),
        ):
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be a positive number or None")
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.kill_grace_s = float(kill_grace_s)
        self.row_wall_clock_budget_s = row_wall_clock_budget_s
        self.row_peak_rss_budget_mb = row_peak_rss_budget_mb
        self._command_builder = command_builder
        self._executed = False
        self._handle = None
        self._active: tuple[subprocess.Popen | None, Path | None] = (None, None)
        try:
            if resume:
                if not run_id or not str(run_id).strip():
                    raise TaskExecutorError(
                        "resume=True requires the run_id of an existing run"
                    )
                self._handle = store.attach_run(spec, run_id)
            else:
                self._handle = store.open_run(spec, run_id=run_id)
            self.spec = spec
            self.run_id = self._handle.run_id
            self.run_dir = self._handle.run_dir
            self.rows_dir = self.run_dir / "executor_rows"
            self._register_or_check_plan()
            # Prior-attempt state: one integrity-verified read of the run's
            # bound ledger at attach time (appends this session go through
            # the handle; state is tracked in memory afterwards).
            self._prior = self._scan_prior_attempts()
        except Exception:
            if self._handle is not None:
                self._handle.close()
            raise

    # -- plan identity --------------------------------------------------------

    def _plan_payload(self) -> dict[str, Any]:
        return {
            "plan_schema_version": TASK_EXECUTOR_SCHEMA_VERSION,
            "rows": [row.to_payload() for row in self._plan],
        }

    def _register_or_check_plan(self) -> None:
        payload = self._plan_payload()
        index = self._handle.load_index()
        existing = [
            record
            for record in index["artifacts"]
            if record["artifact_type"] == _PLAN_ARTIFACT_TYPE
        ]
        if existing:
            registered = self._handle.read_artifact(existing[0]["artifact_id"])
            if registered != payload:
                raise TaskExecutorError(
                    f"plan drift on resume for run {self.run_id}: the stored "
                    "task_executor_plan artifact does not match the requested "
                    "rows (fail-closed); rerun with a new run id instead"
                )
            return
        self._handle.write_artifact(_PLAN_ARTIFACT_TYPE, payload)

    # -- prior-attempt reconciliation ------------------------------------------

    def _scan_prior_attempts(self) -> dict[str, list[dict[str, Any]]]:
        ledger = ExperimentLedger(
            self.run_dir / "trials.jsonl",
            run_id=self.run_id,
            spec_id=self.spec.spec_id,
        )
        opens: dict[str, str] = {}
        attempts: dict[str, list[dict[str, Any]]] = {}
        for entry in ledger.iter_entries():
            row_id = str((entry.payload or {}).get("row_id"))
            if not row_id:
                continue  # non-executor trials in the same run
            if entry.status == "running":
                opens[entry.trial_id] = row_id
                attempts.setdefault(row_id, []).append(
                    {"trial_id": entry.trial_id, "status": "running"}
                )
            else:
                if opens.get(entry.trial_id) != row_id:
                    raise TaskExecutorError(
                        f"ledger terminal entry {entry.trial_id} does not "
                        f"close an open executor trial of {row_id}"
                    )
                attempts[row_id][-1]["status"] = entry.status
        return attempts

    def _open_trial_ids(self, row_id: str) -> list[str]:
        return [
            attempt["trial_id"]
            for attempt in self._prior.get(row_id, [])
            if attempt["status"] == "running"
        ]

    # -- execution ---------------------------------------------------------------

    def _heartbeat_age_s(self, heartbeat_path: Path, attempt_start: float) -> float:
        beat = read_json_safe(heartbeat_path)
        last = attempt_start
        if isinstance(beat, dict) and beat.get("ts"):
            try:
                last = max(
                    last, datetime.fromisoformat(str(beat["ts"])).timestamp()
                )
            except ValueError:
                pass
        return time.time() - last

    def _spawn_worker(self, job_path: Path) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ashare_model.task_executor",
                "worker",
                str(job_path),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _finish_attempt(
        self,
        *,
        row: TaskRow,
        trial_id: str,
        error: str,
        wall_seconds: float,
        exit_code: int | None,
        author: str,
        phase_path: Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Close an attempt abnormally: supervisor-written result record
        (the worker produced none) + terminal failed ledger entry.
        Returns ``(status, metrics)`` so the summary can carry the same
        evidence the ledger holds."""

        phases = read_phase_journal(phase_path)
        write_row_result(
            self.rows_dir / f"{row_slug(row)}{_RESULT_SUFFIX}",
            {
                "result_schema_version": TASK_EXECUTOR_SCHEMA_VERSION,
                "row_id": row.row_id,
                "status": "failed",
                "exit_code": exit_code,
                "wall_seconds": wall_seconds,
                "cpu_seconds": None,
                "peak_rss_mb": None,
                "error": error,
                "started_at": None,
                "finished_at": _now_iso(),
                "phases": phases,
                "phase": phases[-1]["phase"] if phases else None,
                "author": author,
            },
        )
        metrics: dict[str, Any] = {"wall_seconds": wall_seconds}
        if phases:
            metrics["phases"] = phases
            metrics["active_phase"] = phases[-1]["phase"]
        self._handle.fail_trial(trial_id, error=error, metrics=metrics)
        return "failed", metrics

    def _run_row(self, row: TaskRow, attempt: int) -> dict[str, Any]:
        rows_dir = self.rows_dir
        rows_dir.mkdir(parents=True, exist_ok=True)
        slug = row_slug(row)
        result_path = rows_dir / f"{slug}{_RESULT_SUFFIX}"
        heartbeat_path = rows_dir / f"{slug}{_HEARTBEAT_SUFFIX}"
        log_path = rows_dir / f"{slug}{_LOG_SUFFIX}"
        phase_path = rows_dir / f"{slug}{_PHASES_SUFFIX}"
        command = [str(part) for part in self._command_builder(row)]
        if not command:
            raise TaskExecutorError(
                f"command_builder returned an empty command for {row.row_id}"
            )
        trial_id = self._handle.record_trial(
            algorithm=f"backend:{row.backend}",
            candidate=row.row_id,
            seed=row.seed,
            fold_train_end=row.fold_train_end,
            fold_test_end=row.fold_test_end,
            payload={
                "row_id": row.row_id,
                "backend": row.backend,
                "track": row.track,
                "fold": row.fold,
                "attempt": attempt,
                **dict(row.payload or {}),
            },
        )
        atomic_write_json(
            rows_dir / f"{slug}{_JOB_SUFFIX}",
            {
                "job_schema_version": TASK_EXECUTOR_SCHEMA_VERSION,
                "row_id": row.row_id,
                "spec_id": self.spec.spec_id,
                "run_id": self.run_id,
                "row": row.to_payload(),
                "command": command,
                "result_path": str(result_path),
                "heartbeat_path": str(heartbeat_path),
                "log_path": str(log_path),
                "phase_path": str(phase_path),
                "heartbeat_interval_s": self.heartbeat_interval_s,
                "row_wall_clock_budget_s": self.row_wall_clock_budget_s,
                "row_peak_rss_budget_mb": self.row_peak_rss_budget_mb,
                "cwd": str(Path(__file__).resolve().parents[1]),
            },
        )
        logger.info(
            "task_executor run={} row={} attempt={} trial={} spawning worker",
            self.run_id,
            row.row_id,
            attempt,
            trial_id,
        )
        job_path = rows_dir / f"{slug}{_JOB_SUFFIX}"
        proc = self._spawn_worker(job_path)
        self._active = (proc, job_path)
        attempt_start = time.time()
        timed_out = False
        try:
            while proc.poll() is None:
                age = self._heartbeat_age_s(heartbeat_path, attempt_start)
                if age > self.heartbeat_timeout_s:
                    logger.error(
                        "task_executor row={} trial={} heartbeat timeout after "
                        "{:.1f}s without a beat; killing worker tree",
                        row.row_id,
                        trial_id,
                        age,
                    )
                    self._terminate_worker_tree(proc, job_path)
                    timed_out = True
                    break
                time.sleep(self.poll_interval_s)
            proc.wait()
        except KeyboardInterrupt:
            # B03 p32 §3.6/R9: the except clause runs BEFORE the finally
            # below, so the ownership-verified kill executes while the
            # worker reference is still alive — a user abort must never
            # orphan the worker or its row command (the interrupted
            # attempt stays open for the resume to close + re-execute).
            self._terminate_worker_tree(proc, job_path)
            raise
        finally:
            self._active = (None, None)

        if timed_out:
            error = (
                f"row {row.row_id}: heartbeat timeout after "
                f"{self.heartbeat_timeout_s:.1f}s without a beat "
                "(worker killed by the supervisor)"
            )
            status, metrics = self._finish_attempt(
                row=row,
                trial_id=trial_id,
                error=error,
                wall_seconds=time.time() - attempt_start,
                exit_code=None,
                author="supervisor",
                phase_path=phase_path,
            )
            return {
                "status": status,
                "trial_id": trial_id,
                "error": error,
                "metrics": metrics,
            }

        result = _read_valid_result(result_path, row)
        if result is None:
            error = (
                f"row {row.row_id}: worker exited {proc.returncode} "
                "without a valid result record"
            )
            status, metrics = self._finish_attempt(
                row=row,
                trial_id=trial_id,
                error=error,
                wall_seconds=time.time() - attempt_start,
                exit_code=proc.returncode,
                author="supervisor",
                phase_path=phase_path,
            )
            return {
                "status": status,
                "trial_id": trial_id,
                "error": error,
                "metrics": metrics,
            }

        metrics = {
            "wall_seconds": result.get("wall_seconds"),
            "cpu_seconds": result.get("cpu_seconds"),
            "peak_rss_mb": result.get("peak_rss_mb"),
            "exit_code": result.get("exit_code"),
        }
        if result.get("phases"):
            metrics["phases"] = result["phases"]
            metrics["active_phase"] = result.get("phase")
        if result["status"] == "succeeded":
            self._handle.complete_trial(trial_id, metrics=metrics)
            return {
                "status": "succeeded",
                "trial_id": trial_id,
                "error": None,
                "metrics": metrics,
            }
        error = str(result.get("error")) or f"row {row.row_id} failed"
        self._handle.fail_trial(trial_id, error=error, metrics=metrics)
        return {
            "status": "failed",
            "trial_id": trial_id,
            "error": error,
            "metrics": metrics,
        }

    def _abort_active_worker(self) -> None:
        """User-abort path (B03 p32 §3.6/R9): terminate the active worker
        tree (ownership-verified) so a KeyboardInterrupt never leaves an
        orphaned worker or row command behind."""

        proc, job_path = self._active
        if proc is None:
            return
        logger.warning(
            "task_executor run={} user abort: terminating the active "
            "worker tree (pid={})",
            self.run_id,
            proc.pid,
        )
        self._terminate_worker_tree(proc, job_path)

    def _terminate_worker_tree(
        self, proc: subprocess.Popen, job_path: Path | None
    ) -> None:
        """Kill the worker and its command subtree — but only after the
        ownership check: the pid's command line must still name this
        run's job file, so a recycled pid can never route a kill into an
        unrelated process (B03 p32 §3.6).

        Fallback invariant (review A#10/B#4): when the cmdline check
        fails, the Popen-handle ``proc.kill()`` below still targets the
        exact process this supervisor spawned — while ``proc`` is
        unreaped (``poll() is None`` pins it on Windows via the open
        handle, on POSIX as an unreaped zombie) the OS cannot recycle
        that pid, so the fallback can never hit a foreign process.  It
        is deliberately NOT a tree kill: an orphaned command grandchild
        after a genuine worker death is recorded evidence, not a target.
        """

        if job_path is not None and _process_cmdline_has(
            proc.pid, str(job_path)
        ):
            _kill_tree(proc.pid, self.kill_grace_s)
            return
        logger.error(
            "task_executor worker pid={} no longer matches its job file "
            "{}; skipping the tree kill and falling back to the owned "
            "Popen handle only (recycled-pid guard)",
            proc.pid,
            job_path,
        )
        try:
            proc.kill()
        except OSError:
            pass

    def execute(self) -> dict[str, Any]:
        """Run the plan to its terminal state and return the summary.

        Rows with a terminal ledger record are never re-executed; an
        interrupted attempt (open trial) is closed as failed and re-run
        once, or adopted from its worker result when one exists.
        """

        if self._executed:
            raise TaskExecutorError(
                "this TaskExecutor instance already executed; construct a "
                "new instance (or a new run) instead"
            )
        self._executed = True
        try:
            rows: dict[str, dict[str, Any]] = {}
            for row in self._plan:
                open_ids = self._open_trial_ids(row.row_id)
                prior = self._prior.get(row.row_id, [])
                terminal = [a for a in prior if a["status"] != "running"]
                if len(open_ids) > 1:
                    raise TaskExecutorError(
                        f"row {row.row_id} has {len(open_ids)} open trials; "
                        "concurrent supervisors are a contract violation"
                    )
                if open_ids:
                    result = _read_valid_result(
                        self.rows_dir / f"{row_slug(row)}{_RESULT_SUFFIX}", row
                    )
                    if result is not None:
                        # Adopt: the worker's verdict is on disk; record it
                        # without re-running the row.
                        self.rows_dir.mkdir(parents=True, exist_ok=True)
                        metrics = {
                            "wall_seconds": result.get("wall_seconds"),
                            "cpu_seconds": result.get("cpu_seconds"),
                            "peak_rss_mb": result.get("peak_rss_mb"),
                            "exit_code": result.get("exit_code"),
                        }
                        if result.get("phases"):
                            metrics["phases"] = result["phases"]
                            metrics["active_phase"] = result.get("phase")
                        if result["status"] == "succeeded":
                            self._handle.complete_trial(
                                open_ids[0], metrics=metrics
                            )
                            rows[row.row_id] = {
                                "status": "succeeded",
                                "trial_id": open_ids[0],
                                "error": None,
                                "metrics": metrics,
                            }
                        else:
                            error = (
                                str(result.get("error"))
                                or f"row {row.row_id} failed"
                            )
                            self._handle.fail_trial(
                                open_ids[0], error=error, metrics=metrics
                            )
                            rows[row.row_id] = {
                                "status": "failed",
                                "trial_id": open_ids[0],
                                "error": error,
                                "metrics": metrics,
                            }
                        logger.info(
                            "task_executor run={} row={} adopted interrupted "
                            "attempt trial={} from its result file",
                            self.run_id,
                            row.row_id,
                            open_ids[0],
                        )
                        continue
                    # Interrupted with no verdict on disk: close the stale
                    # attempt as failed (append-only evidence preserved)
                    # and fill the unfinished row with a fresh attempt.
                    error = (
                        f"row {row.row_id}: interrupted attempt "
                        f"{open_ids[0]} ended without a terminal record "
                        "(restart recovery); re-executing"
                    )
                    self._handle.fail_trial(open_ids[0], error=error)
                    logger.warning(
                        "task_executor run={} row={} closed interrupted "
                        "trial={} and re-executing",
                        self.run_id,
                        row.row_id,
                        open_ids[0],
                    )
                    attempt = len(prior) + 1
                    rows[row.row_id] = self._run_row(row, attempt)
                    continue
                if terminal:
                    # Terminal records are final for this run: a failed
                    # trial stays in the chain and is never re-executed
                    # (redoing it is a rerun = new run id / output dir).
                    last = terminal[-1]["status"]
                    rows[row.row_id] = {
                        "status": last,
                        "trial_id": terminal[-1]["trial_id"],
                        "error": None,
                    }
                    continue
                rows[row.row_id] = self._run_row(row, attempt=1)

            statuses = [info["status"] for info in rows.values()]
            payload = {
                "task_executor_schema_version": TASK_EXECUTOR_SCHEMA_VERSION,
                "spec_id": self.spec.spec_id,
                "run_id": self.run_id,
                "rows_dir": str(self.rows_dir),
                "plan": [row.row_id for row in self._plan],
                "rows": rows,
                "counts": {
                    "total": len(self._plan),
                    "succeeded": statuses.count("succeeded"),
                    "failed": statuses.count("failed"),
                },
                "status": (
                    "completed"
                    if statuses and all(s == "succeeded" for s in statuses)
                    else "failed"
                ),
            }
            logger.info(
                "task_executor run={} status={} succeeded={} failed={}",
                self.run_id,
                payload["status"],
                payload["counts"]["succeeded"],
                payload["counts"]["failed"],
            )
            return payload
        except KeyboardInterrupt:
            # B03 p32 §3.6: a user abort kills the active worker tree
            # (ownership-verified) before unwinding.  The interrupted
            # attempt stays OPEN in the ledger — the resume closes it as
            # failed and re-executes, preserving the append-only evidence.
            self._abort_active_worker()
            raise
        finally:
            self._handle.close()


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "worker":
        return worker_main(argv[1])
    print(
        "usage: python -m ashare_model.task_executor worker <job.json>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
