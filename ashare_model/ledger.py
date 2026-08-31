"""Automatic experiment trial ledger (T4-01).

The ledger is the structural guarantee behind "every trial is recorded —
no manual omission of failed trials".  It is an **append-only JSONL**
file where each line is one immutable :class:`LedgerEntry`:

* every trial opens with a ``running`` entry **before** any work starts
  and must close with exactly one terminal entry (``succeeded`` or
  ``failed``); the :meth:`ExperimentLedger.trial` context manager closes
  it automatically on both the success and the exception path, so a
  crashed trial is still recorded as ``failed`` — never silently dropped;
* a trial that never closes (hard kill mid-trial) stays visible as an
  *open* trial: :meth:`ExperimentLedger.finalize` returns it, and a
  protocol run that finishes with open trials is tainted, never silently
  clean;
* closing is one-shot: a trial whose terminal entry already exists cannot
  be closed again;
* every entry carries ``sha256`` of the canonical JSON of the previous
  entry (hash chain), and reload verifies both the chain and sequence
  continuity, raising :class:`LedgerIntegrityError` on any gap or
  tampering — so an edited or truncated ledger is detected, not trusted.

The ledger is a file-level record only: no server, no locking beyond the
write itself, and no loss when the process dies between lines.  Protocol
runs append to it through the shared context manager (see
``ashare_model.evaluation.run_protocol``); experiment archives reference
the ledger path and run id, and Champion/Challenger promotion (T4-01,
``ashare_model.promotion``) queries it for trial history.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ashare_model.identity import canonical_json, sha256_hex


class LedgerIntegrityError(Exception):
    """The ledger file is missing entries, reordered, or tampered with."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# P8-02: canonical JSON and entry hashing live in the single identity module
# (ashare_model.identity); ``canonical_json``/``sha256_hex`` are imported
# above and re-exported so the historical import surface stays intact and
# every existing ledger line re-hashes byte-identically.


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable ledger line."""

    sequence: int
    run_id: str
    trial_id: str
    status: str  # "running" | "succeeded" | "failed"
    algorithm: str
    candidate: str
    seed: int | None = None
    fold_train_end: str | None = None
    fold_test_end: str | None = None
    started_at: str = ""
    finished_at: str | None = None
    error: str | None = None
    metrics: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    prev_hash: str = ""
    entry_hash: str = ""

    @classmethod
    def _fields_except(cls, *excluded: str) -> tuple:
        return tuple(f.name for f in fields(cls) if f.name not in excluded)

    @classmethod
    def from_json(cls, text: str) -> "LedgerEntry":
        payload = json.loads(text)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise LedgerIntegrityError(f"malformed ledger entry: {exc}") from exc

    def to_json(self) -> str:
        """The full line as written to the file (including ``entry_hash``)."""

        return canonical_json(
            {name: getattr(self, name) for name in self._fields_except()}
        )

    @classmethod
    def with_hash(cls, entry: "LedgerEntry") -> "LedgerEntry":
        """Return ``entry`` with its ``entry_hash`` filled in.

        The hash covers every field except ``entry_hash`` itself, so a
        reload can recompute it from the line alone.
        """

        text = canonical_json(
            {name: getattr(entry, name) for name in cls._fields_except("entry_hash")}
        )
        return cls(
            **{
                name: getattr(entry, name)
                for name in cls._fields_except("entry_hash")
            },
            entry_hash=sha256_hex(text),
        )


class ExperimentLedger:
    """Append-only, hash-chained trial ledger backed by one JSONL file.

    ``run_id`` identifies the experiment run the entries belong to
    (auto-generated when omitted).  Opening an existing file verifies the
    hash chain and sequence continuity; any violation raises
    :class:`LedgerIntegrityError` before a single entry is trusted.
    """

    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run-{_now().replace(':', '').replace('-', '')}"
        self._entries: list[LedgerEntry] = []
        if self.path.exists():
            self._load()
        self._closed_trials: set[str] = {
            e.trial_id for e in self._entries if e.status != "running"
        }

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        previous_hash = ""
        for index, line in enumerate(lines, start=1):
            entry = LedgerEntry.from_json(line)
            if entry.sequence != index:
                raise LedgerIntegrityError(
                    f"sequence gap: expected {index}, found {entry.sequence} "
                    f"in {self.path}"
                )
            if entry.prev_hash != previous_hash:
                raise LedgerIntegrityError(
                    f"hash chain broken at sequence {index} in {self.path}"
                )
            expected = sha256_hex(
                canonical_json(
                    {
                        name: getattr(entry, name)
                        for name in LedgerEntry._fields_except("entry_hash")
                    }
                )
            )
            if entry.entry_hash != expected:
                raise LedgerIntegrityError(
                    f"entry hash mismatch at sequence {index} in {self.path}"
                )
            previous_hash = entry.entry_hash
            self._entries.append(entry)

    def _append(self, entry: LedgerEntry) -> LedgerEntry:
        previous_hash = self._entries[-1].entry_hash if self._entries else ""
        entry = LedgerEntry.with_hash(
            LedgerEntry(
                **{
                    name: getattr(entry, name)
                    for name in LedgerEntry._fields_except("prev_hash", "entry_hash")
                },
                prev_hash=previous_hash,
            )
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json() + "\n")
        self._entries.append(entry)
        return entry

    # -- recording -----------------------------------------------------------

    def record_trial(
        self,
        *,
        algorithm: str,
        candidate: str,
        seed: int | None = None,
        fold_train_end: str | None = None,
        fold_test_end: str | None = None,
        payload: dict | None = None,
    ) -> str:
        """Open a trial (append a ``running`` entry) and return its id.

        The caller must close it with :meth:`complete_trial` or
        :meth:`fail_trial` — normally via the :meth:`trial` context
        manager, which guarantees the close on both paths.
        """

        sequence = len(self._entries) + 1
        trial_id = f"{self.run_id}-{sequence:05d}"
        entry = self._append(
            LedgerEntry(
                sequence=sequence,
                run_id=self.run_id,
                trial_id=trial_id,
                status="running",
                algorithm=algorithm,
                candidate=candidate,
                seed=seed,
                fold_train_end=fold_train_end,
                fold_test_end=fold_test_end,
                started_at=_now(),
                payload=dict(payload or {}),
            )
        )
        return entry.trial_id

    def _close(
        self, trial_id: str, status: str, error: str | None, metrics: dict | None
    ) -> None:
        if trial_id not in {e.trial_id for e in self._entries}:
            raise KeyError(f"unknown trial: {trial_id}")
        if trial_id in self._closed_trials:
            raise ValueError(f"trial already closed: {trial_id}")
        opening = next(e for e in self._entries if e.trial_id == trial_id)
        self._closed_trials.add(trial_id)
        sequence = len(self._entries) + 1
        self._append(
            LedgerEntry(
                sequence=sequence,
                run_id=self.run_id,
                trial_id=trial_id,
                status=status,
                algorithm=opening.algorithm,
                candidate=opening.candidate,
                seed=opening.seed,
                fold_train_end=opening.fold_train_end,
                fold_test_end=opening.fold_test_end,
                started_at=opening.started_at,
                finished_at=_now(),
                error=error,
                metrics=dict(metrics or {}),
                payload=opening.payload,
            )
        )

    def complete_trial(self, trial_id: str, metrics: dict | None = None) -> None:
        """Close a trial as ``succeeded`` (one-shot)."""

        self._close(trial_id, "succeeded", None, metrics)

    def fail_trial(self, trial_id: str, error: str, metrics: dict | None = None) -> None:
        """Close a trial as ``failed`` with the error text (one-shot)."""

        self._close(trial_id, "failed", error, metrics)

    @contextmanager
    def trial(self, **kwargs) -> Iterator[str]:
        """Open a trial and guarantee its close on both paths.

        Usage::

            with ledger.trial(algorithm="trained", candidate="trained",
                              seed=seed, fold_train_end=..., fold_test_end=...) as tid:
                ...  # on exception the trial is recorded as failed
        """

        trial_id = self.record_trial(**kwargs)
        try:
            yield trial_id
        except Exception as exc:
            self.fail_trial(trial_id, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.complete_trial(trial_id)

    # -- queries -------------------------------------------------------------

    def iter_entries(self) -> Iterator[LedgerEntry]:
        return iter(self._entries)

    def entries_for(self, trial_id: str) -> list[LedgerEntry]:
        return [e for e in self._entries if e.trial_id == trial_id]

    def trials_for(
        self,
        *,
        algorithm: str | None = None,
        seed: int | None = None,
    ) -> Iterator[LedgerEntry]:
        """Terminal (closed) trial records matching the filters.

        A closed trial appears exactly once — its terminal entry — so
        callers can iterate "the trials of a run" without lifecycle noise.
        """

        for entry in self._entries:
            if entry.status == "running":
                continue
            if algorithm is not None and entry.algorithm != algorithm:
                continue
            if seed is not None and entry.seed != seed:
                continue
            yield entry

    def open_trials(self) -> list[LedgerEntry]:
        """Trials that never closed (leak detector).

        The ``running`` record of a *closed* trial stays in the append-only
        history; only trials with no terminal entry are open.
        """

        return [
            e
            for e in self._entries
            if e.status == "running" and e.trial_id not in self._closed_trials
        ]

    def finalize(self) -> list[LedgerEntry]:
        """Return the open (never-closed) trials at the end of a run.

        A non-empty result means the run was interrupted mid-trial: the
        run is tainted and must not be presented as a clean record.
        """

        return self.open_trials()

    def __len__(self) -> int:
        return len(self._entries)
