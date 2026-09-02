"""Content-addressed RunStore bound to per-run ledgers (P8-04).

Layout (lifecycle contract §3, ``runs/<spec_id>/<run_id>/``)::

    runspec.json      write-once frozen RunSpec payload (content = spec_id)
    lifecycle.jsonl   lifecycle events (machine: P8-06; created empty here)
    trials.jsonl      the run's own hash-chained trial ledger (schema v1:
                      every entry carries spec_id/run_id)
    artifacts/        content-addressed artifact files (<artifact_id>.json)
    index.json        the single trust anchor: schema version + artifact
                      records (artifact_id, type, path, identity)

Guarantees:

* **Write-once RunSpec** — an existing ``runspec.json`` with different
  content is rejected (:class:`RunSpecMismatchError`); identical content
  is idempotent.
* **Single writer** — :meth:`RunStore.open_run` takes an
  ``O_CREAT|O_EXCL`` lock per run; a second writer gets
  :class:`RunLockedError`. Recovery from a crashed writer is an explicit
  :meth:`RunStore.clear_stale_lock` action, never automatic.
* **No overwrite** — an existing run directory is never reopened by
  ``open_run``; reruns get a fresh ``run_id``.
* **Artifacts** — payload must serialize through the strict identity
  canonicalization; ``artifact_id = content_hash("artifact", payload)``;
  the file is written atomically *before* the index registration is
  atomically updated. The index is the only trust anchor: an orphan file
  without an index record is not trusted, and a registered file whose
  content no longer hashes to its id is rejected on read.
* **Legacy ledgers** — unbound ledgers keep the historical v0 byte
  behavior (see ``ashare_model.ledger``); v0 files are never rewritten or
  bound to a spec.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ashare_model.identity import canonical_json_strict, content_hash
from ashare_model.ledger import ExperimentLedger
from ashare_model.runspec import RunSpec, new_run_id

INDEX_SCHEMA_VERSION = 1

RUNSPEC_FILENAME = "runspec.json"
LIFECYCLE_FILENAME = "lifecycle.jsonl"
TRIALS_FILENAME = "trials.jsonl"
INDEX_FILENAME = "index.json"
ARTIFACTS_DIRNAME = "artifacts"
LOCK_SUFFIX = ".lock"


class RunStoreError(Exception):
    """RunStore contract violation (fail-closed)."""


class RunExistsError(RunStoreError):
    """The run already exists; reruns must use a new run_id."""


class RunSpecMismatchError(RunStoreError):
    """A stored runspec.json exists with different content (write-once)."""


class RunLockedError(RunStoreError):
    """Another writer holds the single-writer lock for this run."""


class ArtifactNotRegisteredError(RunStoreError):
    """The artifact is not registered in the run index (not trusted)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid4().hex[:8]}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_runspec_once(run_dir: Path, spec: RunSpec) -> None:
    """Write ``runspec.json`` write-once: identical content is idempotent,
    different content under the same spec_id is a contract violation."""

    path = run_dir / RUNSPEC_FILENAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != spec.to_payload():
            raise RunSpecMismatchError(
                f"runspec.json under {run_dir} exists with different content "
                f"(spec_id {spec.spec_id}); refusing to overwrite evidence"
            )
        return
    _atomic_write_text(path, canonical_json_strict(spec.to_payload()))


@dataclass(frozen=True)
class ArtifactRecord:
    """One registered artifact (index record)."""

    artifact_id: str
    artifact_type: str
    path: str
    spec_id: str
    run_id: str
    candidate_id: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "spec_id": self.spec_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactRecord":
        try:
            return cls(
                artifact_id=str(payload["artifact_id"]),
                artifact_type=str(payload["artifact_type"]),
                path=str(payload["path"]),
                spec_id=str(payload["spec_id"]),
                run_id=str(payload["run_id"]),
                candidate_id=(
                    None if payload.get("candidate_id") is None
                    else str(payload["candidate_id"])
                ),
                created_at=str(payload["created_at"]),
            )
        except KeyError as exc:
            raise RunStoreError(f"malformed artifact record: missing {exc}") from exc


def read_registered_artifact(
    run_dir: Path, index: dict[str, Any], artifact_id: str
) -> dict[str, Any]:
    """Load a registered artifact payload and verify its content hash.

    Shared verification path for :meth:`RunHandle.read_artifact` and the
    P8-06 lifecycle replay (one implementation — the index is the single
    trust anchor; unregistered ids and content drift are contract
    violations).
    """

    record = next(
        (
            ArtifactRecord.from_dict(entry)
            for entry in index["artifacts"]
            if entry["artifact_id"] == artifact_id
        ),
        None,
    )
    if record is None:
        raise ArtifactNotRegisteredError(
            f"artifact {artifact_id} is not registered in the index of "
            f"{run_dir}; unregistered files are not trusted"
        )
    target = run_dir / record.path
    if not target.is_file():
        raise RunStoreError(
            f"registered artifact {record.path} is missing from disk"
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if content_hash("artifact", payload) != artifact_id:
        raise RunStoreError(
            f"artifact {record.path} content does not hash to its "
            f"registered id {artifact_id}; evidence tampered or corrupted"
        )
    return payload


class RunHandle:
    """Single-writer handle for one open run."""

    def __init__(self, store: "RunStore", spec: RunSpec, run_id: str, run_dir: Path):
        self._store = store
        self.spec = spec
        self.run_id = run_id
        self.run_dir = run_dir
        self._closed = False
        self._ledger = ExperimentLedger(
            run_dir / TRIALS_FILENAME, run_id=run_id, spec_id=spec.spec_id
        )

    # -- trials ---------------------------------------------------------------

    def record_trial(self, **kwargs) -> str:
        """Open a trial in this run's bound ledger (entries carry identity)."""

        self._require_open()
        return self._ledger.record_trial(**kwargs)

    def complete_trial(self, trial_id: str, metrics: dict | None = None) -> None:
        self._require_open()
        self._ledger.complete_trial(trial_id, metrics=metrics)

    def fail_trial(self, trial_id: str, error: str, metrics: dict | None = None) -> None:
        self._require_open()
        self._ledger.fail_trial(trial_id, error=error, metrics=metrics)

    # -- artifacts ------------------------------------------------------------

    def write_artifact(
        self,
        artifact_type: str,
        payload: dict[str, Any],
        candidate_id: str | None = None,
    ) -> ArtifactRecord:
        """Write an artifact content-addressed, then register it atomically.

        The file lands first, the index (trust anchor) second; a crash in
        between leaves an untrusted orphan that a later identical write
        recovers.
        """

        self._require_open()
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            raise RunStoreError("artifact_type must be a non-empty string")
        # Strict identity canonicalization rejects NaN/Infinity/implicit
        # types directly (CanonicalJSONError) — artifact content identity
        # is fail-closed at the primitive level.
        text = canonical_json_strict(payload)
        artifact_id = content_hash("artifact", payload)
        relative = f"{ARTIFACTS_DIRNAME}/{artifact_id}.json"
        target = self.run_dir / relative
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != payload:
                raise RunStoreError(
                    f"artifact file {relative} exists with different content "
                    f"(hash collision is a contract violation)"
                )
        else:
            _atomic_write_text(target, text)

        index = self._load_index()
        existing = {
            entry["artifact_id"]: entry for entry in index["artifacts"]
        }
        record = ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            path=relative,
            spec_id=self.spec.spec_id,
            run_id=self.run_id,
            candidate_id=candidate_id,
            created_at=_now(),
        )
        if artifact_id in existing:
            previous = ArtifactRecord.from_dict(existing[artifact_id])
            if previous.to_dict() != record.to_dict():
                raise RunStoreError(
                    f"artifact {artifact_id} already registered with different "
                    "metadata; refusing to overwrite evidence"
                )
            return record
        index["artifacts"].append(record.to_dict())
        _atomic_write_text(
            self.run_dir / INDEX_FILENAME, canonical_json_strict(index)
        )
        return record

    def read_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Load a registered artifact, verifying content against its id."""

        self._require_open()
        return read_registered_artifact(
            self.run_dir, self._load_index(), artifact_id
        )

    def load_index(self) -> dict[str, Any]:
        self._require_open()
        return self._load_index()

    def _load_index(self) -> dict[str, Any]:
        path = self.run_dir / INDEX_FILENAME
        if not path.exists():
            raise RunStoreError(f"missing artifact index at {path}")
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            raise RunStoreError(
                f"unknown artifact index schema version "
                f"{index.get('index_schema_version')!r} in {path}"
            )
        return index

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Release the single-writer lock (idempotent)."""

        if self._closed:
            return
        self._closed = True
        lock = self._store._lock_path(self.spec.spec_id, self.run_id)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass

    def _require_open(self) -> None:
        if self._closed:
            raise RunStoreError("run handle is closed; reopen with open_run")

    def __enter__(self) -> "RunHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RunStore:
    """Filesystem store of content-addressed runs under ``<root>/runs``."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"

    # -- paths ----------------------------------------------------------------

    def _spec_dir(self, spec_id: str) -> Path:
        return self.runs_dir / spec_id

    def _run_dir(self, spec_id: str, run_id: str) -> Path:
        return self._spec_dir(spec_id) / run_id

    def _lock_path(self, spec_id: str, run_id: str) -> Path:
        return self._spec_dir(spec_id) / f"{run_id}{LOCK_SUFFIX}"

    def _acquire_writer_lock(self, spec_id: str, run_id: str) -> None:
        """Take the single-writer ``O_CREAT|O_EXCL`` lock (shared by
        :meth:`open_run` and :meth:`attach_run`); recovery from a stale
        lock is the explicit :meth:`clear_stale_lock` action, never
        automatic."""

        lock_path = self._lock_path(spec_id, run_id)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RunLockedError(
                f"run {spec_id}/{run_id} is already locked by a writer; "
                "recovery is the explicit clear_stale_lock action"
            ) from exc
        try:
            os.write(fd, f"pid={os.getpid()} created_at={_now()}\n".encode("utf-8"))
        finally:
            os.close(fd)

    # -- writer ---------------------------------------------------------------

    def open_run(self, spec: RunSpec, run_id: str | None = None) -> RunHandle:
        """Open a NEW run (single-writer). Existing runs are never reopened
        or overwritten; reruns pass a fresh run_id (the default)."""

        if not isinstance(spec, RunSpec):
            raise RunStoreError("open_run requires a frozen RunSpec instance")
        run_id = run_id or new_run_id()
        if not run_id.strip():
            raise RunStoreError("run_id must be a non-empty string")
        spec_dir = self._spec_dir(spec.spec_id)
        spec_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_writer_lock(spec.spec_id, run_id)
        lock_path = self._lock_path(spec.spec_id, run_id)
        try:
            run_dir = self._run_dir(spec.spec_id, run_id)
            if run_dir.exists():
                raise RunExistsError(
                    f"run {spec.spec_id}/{run_id} already exists; reruns must "
                    "use a new run_id — existing runs are never overwritten"
                )
            run_dir.mkdir()
            write_runspec_once(run_dir, spec)
            (run_dir / LIFECYCLE_FILENAME).touch()
            (run_dir / TRIALS_FILENAME).touch()
            (run_dir / ARTIFACTS_DIRNAME).mkdir(exist_ok=True)
            index = {
                "index_schema_version": INDEX_SCHEMA_VERSION,
                "spec_id": spec.spec_id,
                "run_id": run_id,
                "artifacts": [],
            }
            _atomic_write_text(
                run_dir / INDEX_FILENAME, canonical_json_strict(index)
            )
        except Exception:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            raise
        return RunHandle(self, spec, run_id, run_dir)

    def attach_run(self, spec: RunSpec, run_id: str) -> RunHandle:
        """Attach a single writer to an EXISTING run (P8-06 lifecycle
        management continuation: the FAILED/RETIRED edges apply to runs
        whose creating session already ended).  The stored
        ``runspec.json`` must match ``spec`` exactly (write-once,
        idempotent check); the same single-writer lock protocol as
        :meth:`open_run` guards concurrent attachment.  Never creates a
        run, never modifies the frozen layout."""

        if not isinstance(spec, RunSpec):
            raise RunStoreError("attach_run requires a frozen RunSpec instance")
        if not str(run_id).strip():
            raise RunStoreError("run_id must be a non-empty string")
        run_dir = self._run_dir(spec.spec_id, run_id)
        if not run_dir.is_dir():
            raise RunStoreError(
                f"run {spec.spec_id}/{run_id} does not exist under "
                f"{self.runs_dir}; open_run creates new runs, attach_run "
                "never does"
            )
        write_runspec_once(run_dir, spec)
        self._acquire_writer_lock(spec.spec_id, run_id)
        return RunHandle(self, spec, run_id, run_dir)

    # -- recovery / readers -----------------------------------------------------

    def clear_stale_lock(self, spec_id: str, run_id: str) -> bool:
        """Explicitly remove a stale single-writer lock (crash recovery).

        This is a deliberate administrative action — never automatic — and
        it is the only sanctioned way to clear a lock without a live writer.
        """

        lock = self._lock_path(spec_id, run_id)
        try:
            lock.unlink()
        except FileNotFoundError:
            return False
        return True

    def read_runspec_payload(self, spec_id: str, run_id: str) -> dict[str, Any]:
        path = self._run_dir(spec_id, run_id) / RUNSPEC_FILENAME
        return json.loads(path.read_text(encoding="utf-8"))

    def load_index(self, spec_id: str, run_id: str) -> dict[str, Any]:
        path = self._run_dir(spec_id, run_id) / INDEX_FILENAME
        if not path.exists():
            raise RunStoreError(f"missing artifact index at {path}")
        index = json.loads(path.read_text(encoding="utf-8"))
        if index.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            raise RunStoreError(
                f"unknown artifact index schema version "
                f"{index.get('index_schema_version')!r} in {path}"
            )
        return index
