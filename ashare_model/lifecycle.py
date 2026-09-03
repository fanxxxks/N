"""P8-06 research lifecycle: append-only, hash-chained state machine.

Contract authority: ``docs/p8_research_lifecycle_contract.md`` (§3 state set
and evidence matrix, §4 legal transitions and staged activation, §5.1 data
qualification, §9 activation boundary).  Scope baseline: t16 (AgentTeams
alphagpt-p0-p1, L-line requirement freeze).  ``LIFECYCLE_CONTRACT_VERSION``
per contract §10 (P8-06 row).

Hard properties (each pinned by ``tests/test_run_store.py``):

* **Replay-only state** — the current state is computed by replaying
  ``runs/<spec_id>/<run_id>/lifecycle.jsonl`` (append-only, hash-chained).
  There is no authority status JSON and no state setter; a forged
  ``status.json`` is ignored by the reader (contract §3).
* **Single identity path** — every content hash goes through
  :func:`ashare_model.identity.content_hash` (kind-domain separated);
  the lifecycle chain hash kind is ``"lifecycle-event"``.  No second
  hash/identity implementation (contract §2).
* **Evidence-gated transitions** — every transition consumes a registered,
  content-verified evidence artifact (the RunStore index is the trust
  anchor, re-verified on every replay).  Unregistered or drifted evidence
  fails closed (contract §3 evidence matrix).
* **Staged activation** — the full §4 graph is known; only the P8-06 edge
  set is activatable.  Requests for later-stage edges are rejected with
  :class:`StageNotActivatedError` (contract §4: "未激活的边在状态机上
  fail-closed，不得静默放行").  Replay does *not* enforce the activation
  set (a chain written by a later stage stays replayable) — activation is
  an append-time gate.
* **Terminal discipline** — ``REJECTED``/``FAILED``/``RETIRED`` have no
  out-edges; ``PROMOTED``'s only out-edge is ``-> RETIRED``; ``IDEA``/
  ``SPEC_LOCKED`` can never enter ``REJECTED`` (no research verdict exists
  there; contract §4).
* **Genesis** — an uninitialized run (empty chain) has state ``None``.  The
  only legal first event is the genesis into ``IDEA`` carrying a registered
  IdeaRecord artifact (the §3 evidence for ``IDEA``); the genesis cannot
  repeat.

Known staged limitation (t16 baseline §6 L1): the ``-> REJECTED`` edges are
activated at P8-08, so in P8-06 a completed-but-failing data gate is refused
with its evidence artifact preserved; it cannot enter ``REJECTED`` yet.
Known honest gap (t16 baseline §6 L2): the §5.1 loader-frame degradation
channel has no typed producer in the data layer yet, so
:func:`build_data_qualification_report` truthfully emits
``degraded_sources.loader_frames = LOADER_FRAME_SOURCE_UNTYPED`` and the
verdict rule refuses such reports — no silent green.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ashare_model import run_store
from ashare_model.identity import canonical_json_strict, content_hash
from ashare_model.run_store import (
    RunHandle,
    RunStore,
    RunStoreError,
    read_registered_artifact,
)
from ashare_model.runspec import RunSpec

__all__ = [
    "ACTIVATED_EDGES",
    "LIFECYCLE_CONTRACT_VERSION",
    "LEGAL_EDGES",
    "LEGAL_TRANSITION_LINES",
    "LifecycleError",
    "LifecycleEvent",
    "LifecycleIntegrityError",
    "LifecycleWriter",
    "IdentityMismatchError",
    "IllegalTransitionError",
    "STATUSES",
    "StageNotActivatedError",
    "TERMINAL_STATUSES",
    "build_data_qualification_report",
    "legal_edge_pairs",
    "load_events",
    "main",
    "replay_run",
    "replay_state",
]

# -- contract constants (docs/p8_research_lifecycle_contract.md §3/§4/§10) ---

LIFECYCLE_CONTRACT_VERSION = 1

#: Canonical status set, contract §3 (order verbatim).
STATUSES: tuple[str, ...] = (
    "IDEA",
    "SPEC_LOCKED",
    "DATA_QUALIFIED",
    "FACTOR_SET_QUALIFIED",
    "SEARCH_PLAN_ADMITTED",
    "OOS_QUALIFIED",
    "PAPER_OBSERVING",
    "PROMOTED",
    "REJECTED",
    "FAILED",
    "RETIRED",
)

#: Terminal states (contract §4): no out-edges, meanings not interchangeable.
TERMINAL_STATUSES = frozenset({"REJECTED", "FAILED", "RETIRED"})

#: Contract §4 machine-readable transition block, 15 lines verbatim.
LEGAL_TRANSITION_LINES: tuple[str, ...] = (
    "IDEA -> SPEC_LOCKED",
    "IDEA -> FAILED | RETIRED",
    "SPEC_LOCKED -> DATA_QUALIFIED",
    "SPEC_LOCKED -> REJECTED | FAILED | RETIRED",
    "DATA_QUALIFIED -> FACTOR_SET_QUALIFIED",
    "DATA_QUALIFIED -> REJECTED | FAILED | RETIRED",
    "FACTOR_SET_QUALIFIED -> SEARCH_PLAN_ADMITTED",
    "FACTOR_SET_QUALIFIED -> REJECTED | FAILED | RETIRED",
    "SEARCH_PLAN_ADMITTED -> OOS_QUALIFIED",
    "SEARCH_PLAN_ADMITTED -> REJECTED | FAILED | RETIRED",
    "OOS_QUALIFIED -> PAPER_OBSERVING",
    "OOS_QUALIFIED -> REJECTED | FAILED | RETIRED",
    "PAPER_OBSERVING -> PROMOTED",
    "PAPER_OBSERVING -> REJECTED | FAILED | RETIRED",
    "PROMOTED -> RETIRED",
)


def _parse_transition_lines(
    lines: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    edges: dict[str, set[str]] = {}
    for line in lines:
        src, dsts = line.split(" -> ")
        edges.setdefault(src.strip(), set()).update(
            dst.strip() for dst in dsts.split("|")
        )
    known = set(STATUSES)
    for src, dsts in edges.items():
        unknown = dsts - known
        if src not in known or unknown:
            raise RuntimeError(
                f"contract transition block references unknown state: "
                f"{src!r} -> {sorted(unknown)}"
            )
    return {src: frozenset(dsts) for src, dsts in edges.items()}


#: Legal-edge graph derived from the verbatim contract lines (fail-fast).
LEGAL_EDGES: dict[str, frozenset[str]] = _parse_transition_lines(
    LEGAL_TRANSITION_LINES
)


def legal_edge_pairs() -> frozenset[tuple[str, str]]:
    """All ``(from_state, to_state)`` pairs of the contract §4 graph."""

    return frozenset(
        (src, dst) for src, dsts in LEGAL_EDGES.items() for dst in dsts
    )


#: Contract §4 staged activation, P8-06 row: ``IDEA -> SPEC_LOCKED``,
#: ``SPEC_LOCKED -> DATA_QUALIFIED``, every legal ``-> FAILED`` edge and
#: every legal ``-> RETIRED`` edge.  Later-stage edges stay in the graph but
#: are fail-closed at append time until their stage activates them.
ACTIVATED_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("IDEA", "SPEC_LOCKED"),
        ("SPEC_LOCKED", "DATA_QUALIFIED"),
    }
    | {pair for pair in legal_edge_pairs() if pair[1] in ("FAILED", "RETIRED")}
)


# -- event schema -------------------------------------------------------------

_EVENT_HASH_KIND = "lifecycle-event"
_GENESIS_PREV_HASH = "0" * 64
_RECORD_TYPE_IDEA = "idea_record"
_RECORD_TYPE_SPEC_LOCK = "spec_lock"
_RECORD_TYPE_DATA_REPORT = "data_qualification_report"
_IDEA_FIELDS = ("hypothesis", "scope", "economic_mechanism", "non_goals")
_LOADER_FRAME_SOURCE_UNTYPED = "untyped_in_current_data_layer"


class LifecycleError(Exception):
    """Lifecycle contract violation (fail-closed)."""


class IllegalTransitionError(LifecycleError):
    """The requested edge is not part of the contract §4 graph, or the
    current state is terminal."""


class StageNotActivatedError(LifecycleError):
    """The edge is legal but its activation stage has not landed yet
    (contract §4 staged activation; P8-06 set only in this version)."""


class IdentityMismatchError(LifecycleError):
    """An event's spec_id/run_id does not match the run it claims."""


class LifecycleIntegrityError(LifecycleError):
    """The event chain is broken: bad JSON, sequence gap, hash mismatch or
    an impossible replay."""


class EvidenceError(LifecycleError):
    """Transition evidence is missing, unregistered or content-drifted."""


class DataQualificationError(LifecycleError):
    """The DataQualificationReport does not satisfy the §5.1 verdict rules."""


@dataclass(frozen=True)
class LifecycleEvent:
    """One hash-chained lifecycle event (a line of ``lifecycle.jsonl``)."""

    seq: int
    prev_hash: str
    event_hash: str
    event_kind: str  # "genesis" | "transition"
    from_state: str | None
    to_state: str
    spec_id: str
    run_id: str
    evidence: dict[str, str] | None
    completed: bool
    reason: str
    ts: str

    def payload(self) -> dict[str, Any]:
        """The strict-JSON hash input (everything except ``event_hash``)."""

        return {
            "lifecycle_contract_version": LIFECYCLE_CONTRACT_VERSION,
            "seq": self.seq,
            "prev_hash": self.prev_hash,
            "event_kind": self.event_kind,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "spec_id": self.spec_id,
            "run_id": self.run_id,
            "evidence": self.evidence,
            "completed": self.completed,
            "reason": self.reason,
            "ts": self.ts,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _event_from_line(line: str) -> LifecycleEvent:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise LifecycleIntegrityError(
            f"lifecycle.jsonl contains a non-JSON line: {exc}"
        ) from exc
    if not isinstance(raw, dict) or "event_hash" not in raw:
        raise LifecycleIntegrityError("lifecycle event missing event_hash")
    event_hash = raw["event_hash"]
    payload = {key: value for key, value in raw.items() if key != "event_hash"}
    if payload.get("lifecycle_contract_version") != LIFECYCLE_CONTRACT_VERSION:
        raise LifecycleIntegrityError(
            f"unknown lifecycle_contract_version "
            f"{payload.get('lifecycle_contract_version')!r}"
        )
    if content_hash(_EVENT_HASH_KIND, payload) != event_hash:
        raise LifecycleIntegrityError(
            f"lifecycle event seq={payload.get('seq')!r} hash mismatch; "
            "append-only evidence was rewritten"
        )
    try:
        evidence = raw.get("evidence")
        return LifecycleEvent(
            seq=int(raw["seq"]),
            prev_hash=str(raw["prev_hash"]),
            event_hash=str(event_hash),
            event_kind=str(raw["event_kind"]),
            from_state=raw.get("from_state"),
            to_state=str(raw["to_state"]),
            spec_id=str(raw["spec_id"]),
            run_id=str(raw["run_id"]),
            evidence=None if evidence is None else dict(evidence),
            completed=bool(raw["completed"]),
            reason=str(raw.get("reason", "")),
            ts=str(raw["ts"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LifecycleIntegrityError(
            f"malformed lifecycle event: {exc}"
        ) from exc


def load_events(
    path: Path,
    *,
    expect_spec_id: str | None = None,
    expect_run_id: str | None = None,
) -> list[LifecycleEvent]:
    """Load and chain-verify a ``lifecycle.jsonl`` (append-only, ordered,
    hash-linked).  Identity mismatches and any tampering fail closed."""

    events: list[LifecycleEvent] = []
    if not path.exists():
        raise LifecycleIntegrityError(f"missing lifecycle event log at {path}")
    text = path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        event = _event_from_line(line)
        if event.seq != len(events) + 1:
            raise LifecycleIntegrityError(
                f"lifecycle sequence gap at line {number}: seq {event.seq} "
                f"after {len(events)} events"
            )
        expected_prev = events[-1].event_hash if events else _GENESIS_PREV_HASH
        if event.prev_hash != expected_prev:
            raise LifecycleIntegrityError(
                f"lifecycle chain break at seq {event.seq}: prev_hash does "
                "not link to the previous event"
            )
        if expect_spec_id is not None and event.spec_id != expect_spec_id:
            raise IdentityMismatchError(
                f"lifecycle event seq {event.seq} carries spec_id "
                f"{event.spec_id!r}, expected {expect_spec_id!r}"
            )
        if expect_run_id is not None and event.run_id != expect_run_id:
            raise IdentityMismatchError(
                f"lifecycle event seq {event.seq} carries run_id "
                f"{event.run_id!r}, expected {expect_run_id!r}"
            )
        events.append(event)
    return events


def replay_state(events: list[LifecycleEvent]) -> str | None:
    """Derive the current state purely from the verified event chain.

    Returns ``None`` for an uninitialized run (empty chain).  The §4
    activation set is an append-time gate and is deliberately not enforced
    here, so chains written by later activation stages stay replayable.
    """

    state: str | None = None
    for event in events:
        if event.event_kind == "genesis":
            if state is not None or event.to_state != "IDEA":
                raise LifecycleIntegrityError(
                    "genesis must be the first event and may only enter IDEA"
                )
        elif event.event_kind == "transition":
            if state is None or event.from_state != state:
                raise LifecycleIntegrityError(
                    f"event seq {event.seq} claims from_state "
                    f"{event.from_state!r} but replayed state is {state!r}"
                )
            if event.to_state not in LEGAL_EDGES.get(state, frozenset()):
                raise LifecycleIntegrityError(
                    f"event seq {event.seq} uses non-contract edge "
                    f"{state!r} -> {event.to_state!r}"
                )
        else:
            raise LifecycleIntegrityError(
                f"unknown event_kind {event.event_kind!r}"
            )
        state = event.to_state
    return state


def replay_run(root: Path, spec_id: str, run_id: str) -> str | None:
    """Trust-root reader: verify the run's identity anchoring, event chain,
    and every event's evidence artifact, then return the replayed state.

    Identity anchoring: the stored ``runspec.json`` must hash to the index's
    ``spec_id``; events must carry that spec_id and the index's run_id; each
    evidence artifact must be index-registered with a matching type and a
    content hash equal to its id.  Any forgery (including a planted
    ``status.json``, which is never read) fails closed.
    """

    store = RunStore(root)
    run_dir = store.runs_dir / spec_id / run_id
    if not run_dir.is_dir():
        raise LifecycleIntegrityError(
            f"run {spec_id}/{run_id} does not exist under {store.runs_dir}"
        )
    index = store.load_index(spec_id, run_id)
    if index.get("spec_id") != spec_id or index.get("run_id") != run_id:
        raise IdentityMismatchError(
            f"index identity {index.get('spec_id')!r}/{index.get('run_id')!r} "
            f"does not match the requested run {spec_id!r}/{run_id!r}"
        )
    stored_spec = json.loads(
        (run_dir / run_store.RUNSPEC_FILENAME).read_text(encoding="utf-8")
    )
    if content_hash("runspec", stored_spec) != spec_id:
        raise IdentityMismatchError(
            f"stored runspec.json does not hash to spec_id {spec_id!r}"
        )
    events = load_events(
        run_dir / run_store.LIFECYCLE_FILENAME,
        expect_spec_id=spec_id,
        expect_run_id=run_id,
    )
    for event in events:
        evidence = event.evidence
        if evidence is None:
            continue
        try:
            read_registered_artifact(run_dir, index, str(evidence["artifact_id"]))
        except RunStoreError as exc:
            raise EvidenceError(
                f"event seq {event.seq} evidence integrity failure: {exc}"
            ) from exc
        record = next(
            (
                entry
                for entry in index["artifacts"]
                if entry["artifact_id"] == evidence["artifact_id"]
            ),
            None,
        )
        if record is None or record["artifact_type"] != str(
            evidence["artifact_type"]
        ):
            raise EvidenceError(
                f"event seq {event.seq} evidence type mismatch: "
                f"{evidence!r} vs index record {record!r}"
            )
    return replay_state(events)


# -- evidence report schemas (contract §3 evidence matrix / §5.1) -------------


def _require_non_empty_str(payload: dict[str, Any], key: str, what: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LifecycleError(f"{what}: {key} must be a non-empty string")
    return value


def _gate_check_names(checks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Map G1..G8 from the producer's naming convention (``G<n> ...``)."""

    found: dict[int, dict[str, Any]] = {}
    for check in checks:
        match = re.match(r"^G([1-8])\b", str(check.get("name", "")))
        if match is not None:
            found.setdefault(int(match.group(1)), check)
    return found


def build_data_qualification_report(
    spec: RunSpec,
    gate_result: Any,
    *,
    expected_trading_days: int,
    actual_trading_days: int,
    missing_trading_days: list[str],
    coverage: float,
) -> dict[str, Any]:
    """Build the §5.1 ``DataQualificationReport`` payload from a real
    :class:`ashare_data.gates.GateResult` (the G1–G8 authority — 判定逻辑
    零复制; G8 freshness added by p16).

    ``degraded_sources.loader_frames`` is truthfully
    :data:`LOADER_FRAME_SOURCE_UNTYPED` until the data layer grows a typed
    degradation source (t16 baseline §6 L2); the §5.1 verdict rule therefore
    refuses reports from real runs — fail closed, never a silent green.
    """

    checks = [
        {"name": check.name, "ok": bool(check.ok), "detail": str(check.detail)}
        for check in gate_result.checks
    ]
    universe_contract = (
        dataclasses.asdict(gate_result.status)
        if gate_result.status is not None
        else None
    )
    gate_payload = {
        "mode": str(gate_result.mode),
        "degraded": bool(gate_result.degraded),
        "checks": checks,
    }
    min_eligible = spec.resolved_thresholds.get("min_eligible")
    completed = set(_gate_check_names(checks)) == set(range(1, 9))
    return {
        "report_schema_version": 1,
        "spec_id": spec.spec_id,
        "dataset_id": spec.dataset_id,
        "data_cutoff": spec.data_cutoff,
        "expected_trading_days": int(expected_trading_days),
        "actual_trading_days": int(actual_trading_days),
        "missing_trading_days": [str(day) for day in missing_trading_days],
        "mode": str(gate_result.mode),
        "degraded": bool(gate_result.degraded),
        "gate_checks": checks,
        "gate_result_hash": content_hash("gate_result", gate_payload),
        "min_eligible": None if min_eligible is None else float(min_eligible),
        "coverage": float(coverage),
        "universe_contract": universe_contract,
        "degraded_sources": {"loader_frames": _LOADER_FRAME_SOURCE_UNTYPED},
        "completed": bool(completed),
    }


def _validate_data_report(report: dict[str, Any], spec: RunSpec) -> None:
    """Machine verdict rules of contract §5.1 (any failure refuses the
    DATA_QUALIFIED transition; the evidence artifact is preserved)."""

    if report.get("report_schema_version") != 1:
        raise DataQualificationError("report_schema_version must be 1")
    if report.get("spec_id") != spec.spec_id:
        raise DataQualificationError("report spec_id does not match the run")
    if report.get("dataset_id") != spec.dataset_id:
        raise DataQualificationError(
            "report dataset_id does not match the spec"
        )
    if report.get("data_cutoff") != spec.data_cutoff:
        raise DataQualificationError(
            "report data_cutoff does not match the spec"
        )
    if report.get("mode") != "formal":
        raise DataQualificationError(
            f"mode must be 'formal' (dev results can never Data Qualify), "
            f"got {report.get('mode')!r}"
        )
    if report.get("degraded") is not False:
        raise DataQualificationError(
            f"degraded must be false, got {report.get('degraded')!r}"
        )
    if report.get("missing_trading_days") != []:
        raise DataQualificationError(
            f"missing_trading_days must be empty, got "
            f"{report.get('missing_trading_days')!r}"
        )
    sources = report.get("degraded_sources")
    loader_frames = (
        sources.get("loader_frames") if isinstance(sources, dict) else None
    )
    if loader_frames != "clean":
        raise DataQualificationError(
            f"loader-frame degradation source is {loader_frames!r}; the "
            "§5.1 'no undisclosed degraded' rule cannot be certified"
        )
    if report.get("completed") is not True:
        raise DataQualificationError("gate evaluation is not completed")
    checks = report.get("gate_checks")
    if not isinstance(checks, list) or not checks:
        raise DataQualificationError("gate_checks must be a non-empty list")
    found = _gate_check_names(checks)
    if set(found) != set(range(1, 9)):
        raise DataQualificationError(
            f"gate evaluation incomplete: G checks found {sorted(found)}"
        )
    failed = [found[i] for i in range(1, 9) if not found[i].get("ok")]
    if failed:
        raise DataQualificationError(
            "gate check failed: "
            + "; ".join(
                f"{check.get('name')}: {check.get('detail')}" for check in failed
            )
        )
    gate_payload = {
        "mode": report["mode"],
        "degraded": report["degraded"],
        "checks": checks,
    }
    if content_hash("gate_result", gate_payload) != report.get(
        "gate_result_hash"
    ):
        raise DataQualificationError(
            "gate_result_hash does not match the reported gate payload"
        )
    expected_min = spec.resolved_thresholds.get("min_eligible")
    if expected_min is not None and report.get("min_eligible") != float(
        expected_min
    ):
        raise DataQualificationError(
            "min_eligible does not match the spec resolved threshold"
        )


# -- writer / reader ----------------------------------------------------------


def _event_line(event: LifecycleEvent) -> str:
    """The strict canonical ``lifecycle.jsonl`` line for one event."""

    return canonical_json_strict({**event.payload(), "event_hash": event.event_hash})


class LifecycleWriter:
    """Single-writer lifecycle appender bound to one open run handle.

    All state changes flow through :meth:`transition` (one guarded path);
    the explicit ``record_idea``/``lock_spec``/``qualify_data``/``fail``/
    ``retire`` methods are the documented P8-06 flows.  Evidence artifacts
    are written through the handle's content-addressed store before the
    event is appended, so a crash can only leave a registered artifact
    without its event — never the reverse.
    """

    def __init__(self, handle: RunHandle):
        handle.load_index()  # refuses a closed handle (fail-closed)
        self._handle = handle
        self._path = handle.run_dir / run_store.LIFECYCLE_FILENAME
        self.events = load_events(
            self._path,
            expect_spec_id=handle.spec.spec_id,
            expect_run_id=handle.run_id,
        )

    @property
    def current_state(self) -> str | None:
        return replay_state(self.events)

    def _append(
        self,
        *,
        event_kind: str,
        from_state: str | None,
        to_state: str,
        evidence: dict[str, str] | None,
        completed: bool,
        reason: str,
    ) -> LifecycleEvent:
        previous = self.events[-1] if self.events else None
        payload = {
            "lifecycle_contract_version": LIFECYCLE_CONTRACT_VERSION,
            "seq": len(self.events) + 1,
            "prev_hash": previous.event_hash if previous else _GENESIS_PREV_HASH,
            "event_kind": event_kind,
            "from_state": from_state,
            "to_state": to_state,
            "spec_id": self._handle.spec.spec_id,
            "run_id": self._handle.run_id,
            "evidence": evidence,
            "completed": completed,
            "reason": reason,
            "ts": _now(),
        }
        event = LifecycleEvent(
            seq=int(payload["seq"]),
            prev_hash=str(payload["prev_hash"]),
            event_hash=content_hash(_EVENT_HASH_KIND, payload),
            event_kind=event_kind,
            from_state=from_state,
            to_state=to_state,
            spec_id=str(payload["spec_id"]),
            run_id=str(payload["run_id"]),
            evidence=evidence,
            completed=completed,
            reason=reason,
            ts=str(payload["ts"]),
        )
        with open(self._path, "a", encoding="utf-8") as stream:
            stream.write(_event_line(event) + "\n")
        self.events.append(event)
        return event

    def _write_evidence(
        self, artifact_type: str, payload: dict[str, Any]
    ) -> dict[str, str]:
        record = self._handle.write_artifact(artifact_type, payload)
        return {
            "artifact_id": record.artifact_id,
            "artifact_type": record.artifact_type,
        }

    def record_idea(self, idea: dict[str, Any]) -> LifecycleEvent:
        """Genesis: register the typed IdeaRecord and enter ``IDEA``.

        Input validation happens before anything is registered — an invalid
        idea is a construction error, not evidence to preserve.
        """

        if self.current_state is not None:
            raise LifecycleError(
                "genesis already happened; a run enters IDEA exactly once"
            )
        if idea.get("idea_record_schema_version") != 1:
            raise LifecycleError("idea_record_schema_version must be 1")
        for field in _IDEA_FIELDS:
            _require_non_empty_str(idea, field, "IdeaRecord")
        evidence = self._write_evidence(_RECORD_TYPE_IDEA, idea)
        return self._append(
            event_kind="genesis",
            from_state=None,
            to_state="IDEA",
            evidence=evidence,
            completed=True,
            reason="",
        )

    def lock_spec(self) -> LifecycleEvent:
        """``IDEA -> SPEC_LOCKED``: register the spec-lock evidence (frozen
        RunSpec identity + clean git commit + dependency lock hash, all
        re-read from the frozen spec, never re-derived)."""

        spec = self._handle.spec
        payload = {
            "spec_lock_schema_version": 1,
            "spec_id": spec.spec_id,
            "run_id": self._handle.run_id,
            "git_commit": spec.git_commit,
            "dependency_lock_hash": spec.dependency_lock_hash,
            "runspec_schema_version": spec.runspec_schema_version,
        }
        evidence = self._write_evidence(_RECORD_TYPE_SPEC_LOCK, payload)
        return self.transition("SPEC_LOCKED", evidence=evidence)

    def qualify_data(self, report: dict[str, Any]) -> LifecycleEvent:
        """``SPEC_LOCKED -> DATA_QUALIFIED`` under the §5.1 verdict rules.

        The report artifact is registered first (evidence preservation), the
        machine verdict second; a refused report keeps the run in
        ``SPEC_LOCKED`` with its evidence on file (the ``-> REJECTED`` edges
        activate at P8-08 — t16 baseline §6 L1).
        """

        evidence = self._write_evidence(_RECORD_TYPE_DATA_REPORT, report)
        _validate_data_report(report, self._handle.spec)
        return self.transition("DATA_QUALIFIED", evidence=evidence)

    def transition(
        self,
        to_state: str,
        *,
        evidence: dict[str, str] | None = None,
        reason: str = "",
        completed: bool = True,
    ) -> LifecycleEvent:
        """The single guarded transition path (fail-closed)."""

        current = self.current_state
        if current is None:
            raise LifecycleError(
                "run is uninitialized; record_idea must come first"
            )
        if to_state not in LEGAL_EDGES.get(current, frozenset()):
            raise IllegalTransitionError(
                f"illegal lifecycle edge {current!r} -> {to_state!r} "
                "(contract §4 graph; terminal states have no out-edges)"
            )
        if (current, to_state) not in ACTIVATED_EDGES:
            raise StageNotActivatedError(
                f"edge {current!r} -> {to_state!r} is legal but its "
                "activation stage has not landed (contract §4 分期)"
            )
        if to_state in ("FAILED", "RETIRED") and not str(reason).strip():
            raise LifecycleError(
                f"{to_state} requires a non-empty reason (machine "
                "discrimination forbids post-hoc reclassification)"
            )
        if evidence is not None:
            index = self._handle.load_index()
            artifact_id = str(evidence.get("artifact_id"))
            read_registered_artifact(self._handle.run_dir, index, artifact_id)
            record = next(
                (
                    entry
                    for entry in index["artifacts"]
                    if entry["artifact_id"] == artifact_id
                ),
                None,
            )
            if (
                record is None
                or record["artifact_type"] != str(evidence.get("artifact_type"))
            ):
                raise EvidenceError(
                    f"transition evidence {evidence!r} is not a trusted "
                    "registered artifact"
                )
        return self._append(
            event_kind="transition",
            from_state=current,
            to_state=to_state,
            evidence=evidence,
            completed=completed,
            reason=str(reason),
        )

    def fail(self, reason: str) -> LifecycleEvent:
        """Any non-terminal state ``-> FAILED`` (runtime failure; the
        machine discriminator: evaluation NOT completed)."""

        return self.transition("FAILED", reason=reason, completed=False)

    def retire(
        self, reason: str, replacement_spec_id: str | None = None
    ) -> LifecycleEvent:
        """Any non-terminal state ``-> RETIRED`` (administrative)."""

        event = self.transition("RETIRED", reason=reason, completed=False)
        if replacement_spec_id:
            self._handle.write_artifact(
                "retirement",
                {
                    "retirement_schema_version": 1,
                    "spec_id": self._handle.spec.spec_id,
                    "run_id": self._handle.run_id,
                    "replacement_spec_id": str(replacement_spec_id),
                },
            )
        return event


# -- minimal new-entry CLI (contract §9: 新机制只通过新入口生效) ----------------


def _load_spec_from_run(store: RunStore, spec_id: str, run_id: str) -> RunSpec:
    payload = json.loads(
        (store.runs_dir / spec_id / run_id / run_store.RUNSPEC_FILENAME)
        .read_text(encoding="utf-8")
    )
    if content_hash("runspec", payload) != spec_id:
        raise LifecycleError(
            f"stored runspec.json does not hash to spec_id {spec_id!r}"
        )
    return RunSpec.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    """Lifecycle CLI: ``start`` / ``show-state`` / ``fail`` / ``retire``.

    Every subcommand operates only through the same guarded machine the
    tests exercise; existing formal entries (train/protocol/promotion/
    paper) are untouched (contract §9).
    """

    parser = argparse.ArgumentParser(
        prog="python -m ashare_model.lifecycle",
        description="P8-06 research lifecycle state machine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser(
        "start", help="open a run, record the idea, lock the spec"
    )
    start.add_argument("--root", required=True)
    start.add_argument("--spec-file", required=True, help="RunSpec payload JSON")
    start.add_argument("--idea-file", required=True, help="IdeaRecord JSON")

    def _identity_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", required=True)
        p.add_argument("--spec-id", required=True)
        p.add_argument("--run-id", required=True)

    show = sub.add_parser("show-state", help="replay and print the run state")
    _identity_args(show)

    fail = sub.add_parser("fail", help="mark an existing run FAILED")
    _identity_args(fail)
    fail.add_argument("--reason", required=True)

    retire = sub.add_parser("retire", help="retire an existing run")
    _identity_args(retire)
    retire.add_argument("--reason", required=True)
    retire.add_argument("--replacement-spec-id", default=None)

    args = parser.parse_args(argv)

    if args.command == "start":
        spec_payload = json.loads(
            Path(args.spec_file).read_text(encoding="utf-8")
        )
        spec = RunSpec.model_validate(spec_payload)
        idea = json.loads(Path(args.idea_file).read_text(encoding="utf-8"))
        store = RunStore(args.root)
        handle = store.open_run(spec)
        try:
            writer = LifecycleWriter(handle)
            writer.record_idea(idea)
            writer.lock_spec()
            print(
                json.dumps(
                    {
                        "spec_id": spec.spec_id,
                        "run_id": handle.run_id,
                        "state": writer.current_state,
                    }
                )
            )
        finally:
            handle.close()
        return 0

    if args.command == "show-state":
        state = replay_run(Path(args.root), args.spec_id, args.run_id)
        print(
            json.dumps(
                {
                    "spec_id": args.spec_id,
                    "run_id": args.run_id,
                    "state": state,
                }
            )
        )
        return 0

    store = RunStore(args.root)
    spec = _load_spec_from_run(store, args.spec_id, args.run_id)
    handle = store.attach_run(spec, args.run_id)
    try:
        writer = LifecycleWriter(handle)
        if args.command == "fail":
            writer.fail(reason=args.reason)
        elif args.command == "retire":
            writer.retire(
                reason=args.reason,
                replacement_spec_id=args.replacement_spec_id,
            )
        print(
            json.dumps(
                {
                    "spec_id": args.spec_id,
                    "run_id": args.run_id,
                    "state": writer.current_state,
                }
            )
        )
        return 0
    finally:
        handle.close()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
