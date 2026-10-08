"""Paper observation lineage: append-only, hash-chained, machine-proven
monotonic extension (E1b/P8-09; contract §7).

Paper observation data is **append-only**: each extension records
``{dataset_id, [start, end], manifest hash, previous lineage hash,
expected session count, missing-session report}`` and the chain is
hash-linked (``OBSERVATION_LINEAGE_VERSION = 1``, contract §7 — the
lineage owns the version instead of bumping ``MANIFEST_VERSION``).

**Machine monotonic proof** (validated before an append is accepted;
all must hold, any failure rejects the append and the caller must
freeze the account):

* every session in the new window is an open session of the unique
  production calendar;
* the new window starts strictly after the recorded horizon and covers
  every open session up to its end (no gaps: 新窗口 == 旧窗口 ∪
  [old_max+1, new_max]);
* the reported missing sessions are exactly the calendar's non-open
  days inside the window (every missing day is proven to be a
  non-trading day);
* the carried ``manifest_hash`` is recorded per entry for the
  paper-window completeness comparison.

History rewrites cannot pass the contiguity proof: a window that starts
at or before the recorded horizon fails the monotonic check and is
rejected (contract §7: 历史分区改写即拒绝并冻结，无 warning-continue 路径).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ashare_data.db import AshareDB
from ashare_data.io_utils import durable_append_line
from ashare_model.identity import canonical_json_strict, content_hash

OBSERVATION_LINEAGE_VERSION = 1

_GENESIS_PREV_HASH = "0" * 64


class ObservationLineageError(Exception):
    """Observation lineage contract violation (fail-closed)."""


@dataclass(frozen=True)
class ObservationLineageEntry:
    """One hash-chained observation extension (a lineage.jsonl line)."""

    seq: int
    prev_hash: str
    entry_hash: str
    dataset_id: str
    window_start: str
    window_end: str
    sessions: tuple[str, ...]
    manifest_hash: str
    expected_sessions: int
    missing_sessions: tuple[str, ...]

    def payload(self) -> dict:
        """The strict-JSON hash input (everything except ``entry_hash``)."""

        return {
            "observation_lineage_version": OBSERVATION_LINEAGE_VERSION,
            "seq": self.seq,
            "prev_hash": self.prev_hash,
            "dataset_id": self.dataset_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "sessions": list(self.sessions),
            "manifest_hash": self.manifest_hash,
            "expected_sessions": int(self.expected_sessions),
            "missing_sessions": list(self.missing_sessions),
        }


def _entry_from_line(line: str) -> ObservationLineageEntry:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ObservationLineageError(
            f"lineage.jsonl contains a non-JSON line: {exc}"
        ) from exc
    if not isinstance(raw, dict) or "entry_hash" not in raw:
        raise ObservationLineageError("lineage entry missing entry_hash")
    entry_hash = raw["entry_hash"]
    payload = {key: value for key, value in raw.items() if key != "entry_hash"}
    if payload.get("observation_lineage_version") != OBSERVATION_LINEAGE_VERSION:
        raise ObservationLineageError(
            f"unknown observation_lineage_version "
            f"{payload.get('observation_lineage_version')!r}"
        )
    if content_hash("observation-lineage", payload) != entry_hash:
        raise ObservationLineageError(
            f"lineage entry seq={payload.get('seq')!r} hash mismatch; "
            "append-only observation history was rewritten"
        )
    try:
        return ObservationLineageEntry(
            seq=int(raw["seq"]),
            prev_hash=str(raw["prev_hash"]),
            entry_hash=str(entry_hash),
            dataset_id=str(raw["dataset_id"]),
            window_start=str(raw["window_start"]),
            window_end=str(raw["window_end"]),
            sessions=tuple(str(s) for s in raw["sessions"]),
            manifest_hash=str(raw["manifest_hash"]),
            expected_sessions=int(raw["expected_sessions"]),
            missing_sessions=tuple(str(s) for s in raw["missing_sessions"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ObservationLineageError(
            f"malformed lineage entry: {exc}"
        ) from exc


def load_lineage(path: Path) -> list[ObservationLineageEntry]:
    """Load and chain-verify a ``lineage.jsonl`` (append-only, ordered,
    hash-linked).  Any tampering fails closed."""

    entries: list[ObservationLineageEntry] = []
    if not path.exists():
        raise ObservationLineageError(f"missing observation lineage at {path}")
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        entry = _entry_from_line(line)
        if entry.seq != len(entries) + 1:
            raise ObservationLineageError(
                f"lineage sequence gap at line {number}: seq {entry.seq} "
                f"after {len(entries)} entries"
            )
        expected_prev = (
            entries[-1].entry_hash if entries else _GENESIS_PREV_HASH
        )
        if entry.prev_hash != expected_prev:
            raise ObservationLineageError(
                f"lineage chain break at seq {entry.seq}"
            )
        entries.append(entry)
    return entries


def _open_sessions(config, start: str, end: str) -> list[str] | None:
    """The calendar's open sessions in ``[start, end]`` ascending
    (the unique production trade calendar; p17 §6.1 forbids a second
    calendar).  ``None`` = the calendar is unavailable."""

    try:
        with AshareDB(config.duckdb_path, read_only=True) as db:
            rows = db.query(
                f"SELECT trade_date FROM {config.calendar_table} "
                "WHERE is_open=TRUE AND trade_date >= ? AND trade_date <= ? "
                "ORDER BY trade_date",
                [start, end],
            )
    except Exception:  # noqa: BLE001 - missing calendar is not computable
        return None
    return [str(value) for value in rows["trade_date"].tolist()]


class ObservationLineageWriter:
    """Append-only writer enforcing the §7 machine monotonic proof."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = (
            load_lineage(self.path) if self.path.exists() else []
        )

    @property
    def covered_sessions(self) -> tuple[str, ...]:
        """The union of all recorded sessions (ascending)."""

        seen: set[str] = set()
        for entry in self.entries:
            seen.update(entry.sessions)
        return tuple(sorted(seen))

    def append(
        self,
        *,
        dataset_id: str,
        sessions: list[str],
        manifest_hash: str,
        config,
    ) -> ObservationLineageEntry:
        """Extend the observation window, proving monotonic append.

        The caller supplies the observation sessions actually recorded
        for the new window; the calendar proves both the session
        identity and the missing (non-trading) days.
        """

        if not sessions:
            raise ObservationLineageError(
                "an observation extension must carry at least one session"
            )
        window_start, window_end = sessions[0], sessions[-1]
        calendar = _open_sessions(config, window_start, window_end)
        if calendar is None:
            raise ObservationLineageError(
                "not computable: the trade calendar is unavailable; the "
                "§7 monotonic proof cannot run"
            )
        calendar_set = set(calendar)
        unknown = [s for s in sessions if s not in calendar_set]
        if unknown:
            raise ObservationLineageError(
                f"observation sessions are not calendar open sessions: "
                f"{unknown[:5]}"
            )
        recorded = list(self.covered_sessions)
        if recorded:
            if sessions[0] <= recorded[-1]:
                raise ObservationLineageError(
                    "observation history rewrite rejected: the new window "
                    f"starts at {sessions[0]}, at or before the recorded "
                    f"horizon {recorded[-1]} (contract §7: append-only, "
                    "no historical partition rewrite)"
                )
            expected_tail = [
                date for date in calendar if date > recorded[-1]
            ]
            missing_proven = [
                date
                for date in expected_tail
                if date <= window_end and date not in set(sessions)
            ]
            if missing_proven:
                raise ObservationLineageError(
                    "monotonic append rejected: the new window leaves a "
                    f"gap in the open-session tail {missing_proven[:5]} "
                    "(contract §7: 新窗口 == 旧窗口 ∪ [old_max+1, new_max], "
                    "无缝隙)"
                )
        missing_sessions = sorted(
            {
                date
                for date in calendar_set
                if window_start <= date <= window_end
                and date not in set(sessions)
            }
        )
        previous_hash = (
            self.entries[-1].entry_hash if self.entries else _GENESIS_PREV_HASH
        )
        payload = {
            "observation_lineage_version": OBSERVATION_LINEAGE_VERSION,
            "seq": len(self.entries) + 1,
            "prev_hash": previous_hash,
            "dataset_id": str(dataset_id),
            "window_start": window_start,
            "window_end": window_end,
            "sessions": list(sessions),
            "manifest_hash": str(manifest_hash),
            "expected_sessions": len(calendar),
            "missing_sessions": missing_sessions,
        }
        entry = ObservationLineageEntry(
            seq=int(payload["seq"]),
            prev_hash=previous_hash,
            entry_hash=content_hash("observation-lineage", payload),
            dataset_id=str(dataset_id),
            window_start=window_start,
            window_end=window_end,
            sessions=tuple(sessions),
            manifest_hash=str(manifest_hash),
            expected_sessions=len(calendar),
            missing_sessions=tuple(missing_sessions),
        )
        line = canonical_json_strict(
            {**entry.payload(), "entry_hash": entry.entry_hash}
        )
        # Evidence durability (P1 fsync review): one append per observation
        # window extension (data-refresh frequency) — flush + fsync per
        # line, so a returned append cannot vanish in a power loss.
        durable_append_line(self.path, line)
        self.entries.append(entry)
        return entry
