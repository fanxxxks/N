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
    "ACTION_MODES",
    "ACTION_AUDIT",
    "ACTION_ENGINEERING",
    "ACTION_PAPER",
    "ACTION_PROTECTED_OOS",
    "ACTION_RESEARCH",
    "ENTRY_MIN_STATE",
    "FORMAL_ENTRY_TYPES",
    "LIFECYCLE_CONTRACT_VERSION",
    "LEGAL_EDGES",
    "LEGAL_TRANSITION_LINES",
    "LifecycleError",
    "LifecycleEvent",
    "LifecycleIntegrityError",
    "LifecycleWriter",
    "IdentityMismatchError",
    "IllegalTransitionError",
    "NON_FORMAL_ACTION_MODES",
    "RunContext",
    "SEARCH_ACTION_MIN_STATE",
    "STATUSES",
    "StageNotActivatedError",
    "TERMINAL_STATUSES",
    "DataQualificationError",
    "FactorQualificationError",
    "SearchAdmissionError",
    "OOSQualificationError",
    "PaperBindingError",
    "PromotionEvidenceError",
    "assert_paper_account_writable",
    "assert_protected_oos_access",
    "assert_qualified_source_run",
    "bound_paper_account_id",
    "build_data_qualification_report",
    "build_factor_qualification_report",
    "build_oos_qualification_evidence",
    "build_paper_account_binding",
    "build_promotion_evidence",
    "build_search_admission_evidence",
    "legal_edge_pairs",
    "load_events",
    "main",
    "replay_run",
    "replay_state",
]

# -- contract constants (docs/p8_research_lifecycle_contract.md §3/§4/§10) ---

# A03 (p27 §4.2/A03-A8, approved: reports/alphagpt-ab
# 00_authority_exception_record.md §11 ruling 1): 1 -> 2.  The entry gate's
# enforcement semantics changed from "vacuous when no identity is passed"
# to "real identity + action mode mandatory" (p27 I2/I3/I4), so gate
# generations are distinguishable in artifacts and logs.  The state set,
# transition graph and evidence matrix are unchanged (p8 remains the sole
# machine authority; p27 non-goal 1).
LIFECYCLE_CONTRACT_VERSION = 2
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


#: Contract §4 staged activation.  E1b (t21) executed the P8-07, P8-08 and
#: P8-09 rows on top of the P8-06 set, so the activated set is now the full
#: contract §4 graph: every legal edge is activatable **with its stage's
#: evidence validation** (the per-edge evidence gates live in
#: :class:`LifecycleWriter` methods and the §5 validators).  A global
#: "all sources fresh" gate stays a preregistered non-goal; the formal-entry
#: no-bypass activation (contract §9 激活点 = P8-15) is tracked in the
#: activation matrix of the delivery report.
ACTIVATED_EDGES: frozenset[tuple[str, str]] = legal_edge_pairs()


# -- action modes and entry floors (A03, p27 §3/I3 + I4 + I5) -----------------
#
# The five action classes of the remediation plan §四 A03.  The table below
# is the single authority for "which lifecycle state allows which action"
# (p27 §9 ruling 3: extending it requires a contract revision, never a
# local edit).

#: 工程探针: explicit non-formal probe.  Any state read-only; may load and
#: run small budgets; never consumes the protected OOS; its runs carry no
#: lifecycle chain and can therefore never enter formal consumption (R7).
ACTION_ENGINEERING = "engineering"
#: 历史审计: explicit read-only audit of legacy/current evidence.
ACTION_AUDIT = "audit"
#: 正式研究: formal research entry (train/evaluation/backtest).
ACTION_RESEARCH = "research"
#: 受保护 OOS: the only mode allowed to evaluate the locked OOS window.
ACTION_PROTECTED_OOS = "protected_oos"
#: 正式 paper: paper observation against the bound account.
ACTION_PAPER = "paper"

ACTION_MODES: frozenset[str] = frozenset(
    {
        ACTION_ENGINEERING,
        ACTION_AUDIT,
        ACTION_RESEARCH,
        ACTION_PROTECTED_OOS,
        ACTION_PAPER,
    }
)

#: Explicit non-formal modes — the ONLY place the legacy "no identity"
#: passthrough survives (p27 I2; approval record §11 ruling 1).
NON_FORMAL_ACTION_MODES: frozenset[str] = frozenset(
    {ACTION_ENGINEERING, ACTION_AUDIT}
)

#: Formal entry types and their minimum lifecycle state at RunContext
#: establishment (p27 §3/I4 + I5: fixed separately per entry, pinned by
#: tests/test_p27_a03_lifecycle_runcontext.py R4/R15; ``paper_sim`` keeps
#: its exact-state semantics from P8-15).
ENTRY_MIN_STATE: dict[str, str] = {
    "train": "SPEC_LOCKED",
    "evaluation": "SEARCH_PLAN_ADMITTED",
    "backtest": "SEARCH_PLAN_ADMITTED",
    "paper_sim": "PAPER_OBSERVING",
}

FORMAL_ENTRY_TYPES: frozenset[str] = frozenset(ENTRY_MIN_STATE)

#: I5: the search action is a different action from loading — before any
#: search compute the run must have consumed the full qualification chain
#: (data + factor set + search plan).
SEARCH_ACTION_MIN_STATE = "SEARCH_PLAN_ADMITTED"


# -- event schema -------------------------------------------------------------

_EVENT_HASH_KIND = "lifecycle-event"
_GENESIS_PREV_HASH = "0" * 64
_RECORD_TYPE_IDEA = "idea_record"
_RECORD_TYPE_SPEC_LOCK = "spec_lock"
_RECORD_TYPE_DATA_REPORT = "data_qualification_report"
_RECORD_TYPE_FACTOR_REPORT = "factor_qualification_report"
_RECORD_TYPE_SEARCH_ADMISSION = "search_admission_evidence"
_RECORD_TYPE_OOS = "oos_qualification_evidence"
_RECORD_TYPE_PAPER_BINDING = "paper_account_binding"
_RECORD_TYPE_PROMOTION = "promotion_evidence"
_IDEA_FIELDS = ("hypothesis", "scope", "economic_mechanism", "non_goals")
_LOADER_FRAME_SOURCE_UNTYPED = "untyped_in_current_data_layer"
_ADAPTER_FEATURE_DOMAIN_SOURCE = "ashare_domain.domain_of_feature"


class LifecycleError(Exception):
    """Lifecycle contract violation (fail-closed)."""


class IllegalTransitionError(LifecycleError):
    """The requested edge is not part of the contract §4 graph, or the
    current state is terminal."""


class StageNotActivatedError(LifecycleError):
    """The edge is legal but its activation stage has not landed yet
    (contract §4 staged activation; enforced via ACTIVATED_EDGES)."""


class IdentityMismatchError(LifecycleError):
    """An event's spec_id/run_id does not match the run it claims."""


class LifecycleIntegrityError(LifecycleError):
    """The event chain is broken: bad JSON, sequence gap, hash mismatch or
    an impossible replay."""


class EvidenceError(LifecycleError):
    """Transition evidence is missing, unregistered or content-drifted."""


class DataQualificationError(LifecycleError):
    """The DataQualificationReport does not satisfy the §5.1 verdict rules."""


class FactorQualificationError(LifecycleError):
    """The FactorQualificationReport does not satisfy the §5.2 verdict
    rules (any of the seven checks failing refuses the transition)."""


class SearchAdmissionError(LifecycleError):
    """The SearchAdmissionEvidence does not satisfy the §5.3 rules."""


class OOSQualificationError(LifecycleError):
    """The OOSQualificationEvidence does not satisfy the §6 rules (a
    tainted protected OOS window permanently refuses the transition)."""


class PaperBindingError(LifecycleError):
    """The paper account binding does not satisfy the §7 rules."""


class PromotionEvidenceError(LifecycleError):
    """The promotion evidence does not satisfy the §3/§7 rules (the paper
    window must be complete and lineage-bound)."""


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


# -- §5.2/§5.3/§6/§7 evidence builders and validators (E1b, t21) --------------


_FACTOR_REPORT_SCHEMA_VERSION = 1
_SEARCH_ADMISSION_SCHEMA_VERSION = 1
_OOS_EVIDENCE_SCHEMA_VERSION = 1
_PAPER_BINDING_SCHEMA_VERSION = 1
_PROMOTION_EVIDENCE_SCHEMA_VERSION = 1
_UNIFIED_DOMAIN_ID = "unified"


def build_factor_qualification_report(
    spec: RunSpec,
    *,
    feature_names: Iterable[str],
    universe_codes: list[str],
    config,
    coverage_by_feature: dict[str, float],
    baseline_evidence: dict[str, Any],
    ablation_evidence: dict[str, Any],
    evaluation_day: str | None = None,
) -> dict[str, Any]:
    """Build the §5.2 ``FactorQualificationReport``.

    Consumes the D7 per-formula-dependency data eligibility gate
    (:func:`ashare_model.data_eligibility.check_runspec_data_eligibility`
    — the §12 source checks for the factor set's data dependencies) and
    derives the registry-level checks from the single authorities
    (``feature_registry`` pit levels, ``feature_metadata.FEATURE_METADATA``
    promotion permissions, ``research_domain.domain_of_feature``).  The
    baseline/ablation evidence blocks are caller-produced
    (``baseline_harness.run_matched_baseline`` results, hash-stamped); the
    validator enforces their presence and completeness.
    """

    from ashare_model.data_eligibility import (
        check_runspec_data_eligibility,
    )
    from ashare_model.feature_metadata import FEATURE_METADATA
    from ashare_model.feature_registry import family_of, pit_level_of
    from ashare_model.research_domain import domain_of_feature
    from ashare_model.data_tier import tier_of_pit_level

    names = list(feature_names)
    eligibility = check_runspec_data_eligibility(
        spec, names, universe_codes, config, evaluation_day=evaluation_day
    )
    threshold = spec.resolved_thresholds.get("factor_min_coverage")
    features: list[dict[str, Any]] = []
    for name in names:
        meta = FEATURE_METADATA[name]
        pit_level = pit_level_of(name)
        domain = domain_of_feature(name)
        domain_consistent = (
            spec.research_domain == _UNIFIED_DOMAIN_ID
            or domain.id == spec.research_domain
        )
        tier = tier_of_pit_level(pit_level)
        features.append(
            {
                "name": name,
                "family": family_of(name),
                "pit_level": pit_level.value,
                "data_tier": tier.value,
                "promotion_allowed": bool(meta.promotion_allowed),
                "domain_id": domain.id,
                "domain_consistent": domain_consistent,
                "in_domain_execution": (
                    spec.research_domain == _UNIFIED_DOMAIN_ID
                    or (
                        spec.frequency in domain.frequencies
                        and int(spec.horizon) in domain.horizon_range
                    )
                ),
                "coverage": (
                    float(coverage_by_feature[name])
                    if name in coverage_by_feature
                    else None
                ),
            }
        )
    domain_execution_legal = all(
        feature["domain_consistent"] and feature["in_domain_execution"]
        for feature in features
    )
    weakest_tier = (
        max((feature["data_tier"] for feature in features), default="A")
    )
    return {
        "report_schema_version": _FACTOR_REPORT_SCHEMA_VERSION,
        "spec_id": spec.spec_id,
        "research_domain": spec.research_domain,
        "frequency": spec.frequency,
        "horizon": int(spec.horizon),
        "feature_count": len(features),
        "features": features,
        "domain_execution_legal": bool(domain_execution_legal),
        "weakest_data_tier": weakest_tier,
        # p17/D5: industry-family features stay Tier C (the tier record is
        # a fact; the promotion-path C restriction is data_tier.py's
        # promotion-gate rule, unchanged here).
        "promotion_input_eligible": weakest_tier != "C",
        "data_eligibility": {
            "ok": bool(eligibility.ok),
            "reasons": list(eligibility.reasons),
            "checks": {
                source: {"ok": check.ok, "required": check.required}
                for source, check in eligibility.checks.items()
            },
        },
        "baseline_evidence": dict(baseline_evidence),
        "ablation_evidence": dict(ablation_evidence),
        "completed": True,
    }


def _validate_factor_report(report: dict[str, Any], spec: RunSpec) -> None:
    """Machine verdict rules of contract §5.2 (any failure refuses the
    FACTOR_SET_QUALIFIED transition; the evidence artifact is preserved)."""

    if report.get("report_schema_version") != _FACTOR_REPORT_SCHEMA_VERSION:
        raise FactorQualificationError("report_schema_version must be 1")
    if report.get("spec_id") != spec.spec_id:
        raise FactorQualificationError("report spec_id does not match the run")
    if report.get("research_domain") != spec.research_domain:
        raise FactorQualificationError(
            "report research_domain does not match the spec"
        )
    features = report.get("features")
    if not isinstance(features, list) or not features:
        raise FactorQualificationError("features must be a non-empty list")
    for feature in features:
        if feature.get("promotion_allowed") is not True:
            raise FactorQualificationError(
                f"feature {feature.get('name')!r} is not promotion_allowed "
                "(§5.2 check 3: the factor set cannot enter the promotion "
                "path)"
            )
        if feature.get("domain_consistent") is not True:
            raise FactorQualificationError(
                f"feature {feature.get('name')!r} does not belong to the "
                "RunSpec research domain (§5.2 check 4)"
            )
        if feature.get("in_domain_execution") is not True:
            raise FactorQualificationError(
                f"feature {feature.get('name')!r}: the spec's "
                "frequency/horizon is not a legal execution point of its "
                "research domain (§5.2 check 4)"
            )
        threshold = spec.resolved_thresholds.get("factor_min_coverage")
        if threshold is not None:
            coverage = feature.get("coverage")
            if coverage is None or float(coverage) < float(threshold):
                raise FactorQualificationError(
                    f"feature {feature.get('name')!r} coverage "
                    f"{coverage!r} < resolved threshold {threshold!r} "
                    "(§5.2 check 5)"
                )
    if report.get("domain_execution_legal") is not True:
        raise FactorQualificationError(
            "domain execution legality failed (§5.2 check 4)"
        )
    eligibility = report.get("data_eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("ok") is not True:
        reasons = eligibility.get("reasons") if isinstance(eligibility, dict) else None
        raise FactorQualificationError(
            "the D7 data eligibility gate failed for the factor set's "
            f"sources: {reasons!r}"
        )
    for block, what in (
        ("baseline_evidence", "simple baseline"),
        ("ablation_evidence", "leave-family-out ablation"),
    ):
        evidence = report.get(block)
        if not isinstance(evidence, dict) or evidence.get("completed") is not True:
            raise FactorQualificationError(
                f"§5.2 check 6: the {what} evidence is missing or incomplete"
            )
        if not str(evidence.get("result_hash", "")).strip():
            raise FactorQualificationError(
                f"§5.2 check 6: the {what} evidence carries no result hash"
            )
    if report.get("completed") is not True:
        raise FactorQualificationError("report is not completed")


def build_search_admission_evidence(
    spec: RunSpec, *, budget_tier: str = "standard"
) -> dict[str, Any]:
    """Build the §5.3 ``SearchAdmissionEvidence`` from the frozen RunSpec
    (searcher, contract versions, domain, budget, paired seeds — all
    re-read from the spec, never re-derived)."""

    return {
        "evidence_schema_version": _SEARCH_ADMISSION_SCHEMA_VERSION,
        "spec_id": spec.spec_id,
        "searcher": spec.searcher,
        "search_contract_version": int(spec.search_contract_version),
        "grammar_version": int(spec.grammar_version),
        "research_domain": spec.research_domain,
        "requested_budget": int(spec.requested_budget),
        "budget_accounting": str(spec.budget_accounting),
        "max_formula_len": int(spec.max_formula_len),
        "seeds": [int(seed) for seed in spec.seeds],
        "paired_seed_plan_hash": content_hash(
            "seed_plan", {"seeds": [int(seed) for seed in spec.seeds]}
        ),
        "budget_tier": str(budget_tier),
        # §5.3 check 5: semantic legality is not configurable — the
        # evidence structurally asserts the absence of any switch.
        "semantic_legality_configurable": False,
        "rl_admission": None,
        "completed": True,
    }


def _validate_search_admission(
    evidence: dict[str, Any], spec: RunSpec
) -> None:
    """Machine verdict rules of contract §5.3."""

    if evidence.get("evidence_schema_version") != _SEARCH_ADMISSION_SCHEMA_VERSION:
        raise SearchAdmissionError("evidence_schema_version must be 1")
    if evidence.get("spec_id") != spec.spec_id:
        raise SearchAdmissionError("evidence spec_id does not match the run")
    if evidence.get("searcher") != spec.searcher:
        raise SearchAdmissionError("evidence searcher does not match the spec")
    if evidence.get("search_contract_version") != int(
        spec.search_contract_version
    ):
        raise SearchAdmissionError("search_contract_version drift")
    if evidence.get("grammar_version") != int(spec.grammar_version):
        raise SearchAdmissionError("grammar_version drift")
    if evidence.get("research_domain") != spec.research_domain:
        raise SearchAdmissionError("research_domain drift")
    if evidence.get("requested_budget") != int(spec.requested_budget):
        raise SearchAdmissionError("requested_budget drift")
    if evidence.get("max_formula_len") != int(spec.max_formula_len):
        raise SearchAdmissionError(
            "max_formula_len drift (the EOS-inclusive length口径 must be "
            "shared by every backend)"
        )
    if evidence.get("seeds") != [int(seed) for seed in spec.seeds]:
        raise SearchAdmissionError(
            "seed drift: the paired seed plan is preregistered and locked"
        )
    expected_plan_hash = content_hash(
        "seed_plan", {"seeds": [int(seed) for seed in spec.seeds]}
    )
    if evidence.get("paired_seed_plan_hash") != expected_plan_hash:
        raise SearchAdmissionError("paired_seed_plan_hash does not match")
    if evidence.get("semantic_legality_configurable") is not False:
        raise SearchAdmissionError(
            "§5.3 check 5: semantic legality must not be configurable"
        )
    if spec.searcher == "rl":
        verdict = evidence.get("rl_admission")
        if not isinstance(verdict, dict) or not verdict.get("completed"):
            raise SearchAdmissionError(
                "§5.3 check 6: RL admission requires the encapsulated "
                "decide_p4_admission verdict"
            )
        if verdict.get("promotion_blockers"):
            raise SearchAdmissionError(
                "§5.3 check 6: the RL admission verdict carries "
                f"promotion_blockers {verdict.get('promotion_blockers')!r}"
            )
    if evidence.get("completed") is not True:
        raise SearchAdmissionError("evidence is not completed")


def build_oos_qualification_evidence(
    spec: RunSpec, *, tainted: bool = False
) -> dict[str, Any]:
    """Build the §6 ``OOSQualificationEvidence``: the write-once locked
    OOS window (the RunSpec holdout window) with its lock hash and the
    taint flag."""

    if spec.holdout_window is None:
        raise OOSQualificationError(
            "the RunSpec has no locked holdout window; OOS cannot qualify"
        )
    window = {
        "start": spec.holdout_window.start,
        "end": spec.holdout_window.end,
    }
    return {
        "evidence_schema_version": _OOS_EVIDENCE_SCHEMA_VERSION,
        "spec_id": spec.spec_id,
        "oos_window": window,
        "lock_hash": content_hash(
            "oos_lock", {"spec_id": spec.spec_id, **window}
        ),
        "tainted": bool(tainted),
        "resolved_thresholds": dict(spec.resolved_thresholds),
        "completed": True,
    }


def _validate_oos_evidence(evidence: dict[str, Any], spec: RunSpec) -> None:
    """Machine verdict rules of contract §6 (a tainted protected OOS
    window permanently refuses OOS_QUALIFIED)."""

    if evidence.get("evidence_schema_version") != _OOS_EVIDENCE_SCHEMA_VERSION:
        raise OOSQualificationError("evidence_schema_version must be 1")
    if evidence.get("spec_id") != spec.spec_id:
        raise OOSQualificationError("evidence spec_id does not match the run")
    holdout = spec.holdout_window
    window = evidence.get("oos_window")
    if holdout is None:
        raise OOSQualificationError(
            "the RunSpec has no locked holdout window"
        )
    expected_window = {"start": holdout.start, "end": holdout.end}
    if window != expected_window:
        raise OOSQualificationError(
            f"oos_window {window!r} does not match the locked RunSpec "
            f"window {expected_window!r} (write-once)"
        )
    expected_lock = content_hash(
        "oos_lock", {"spec_id": spec.spec_id, **expected_window}
    )
    if evidence.get("lock_hash") != expected_lock:
        raise OOSQualificationError("lock_hash does not match the window")
    if evidence.get("tainted") is not False:
        raise OOSQualificationError(
            "the protected OOS window is tainted: OOS_QUALIFIED is "
            "permanently refused (contract §6)"
        )
    if evidence.get("completed") is not True:
        raise OOSQualificationError("evidence is not completed")


def build_paper_account_binding(
    spec: RunSpec,
    run_id: str,
    *,
    account_id: str,
    candidate_id: str,
    observation_lineage_hash: str,
) -> dict[str, Any]:
    """Build the §7 paper account binding (spec/candidate/account/dataset
    identity + the initialized observation lineage hash)."""

    return {
        "binding_schema_version": _PAPER_BINDING_SCHEMA_VERSION,
        "spec_id": spec.spec_id,
        "run_id": str(run_id),
        "account_id": str(account_id),
        "candidate_id": str(candidate_id),
        "dataset_id": spec.dataset_id,
        "observation_lineage_hash": str(observation_lineage_hash),
        "completed": True,
    }


def _validate_paper_binding(
    evidence: dict[str, Any], spec: RunSpec, run_id: str
) -> None:
    """Machine rules of contract §7 (paper events reference the account
    and lineage; the binding fixes the identity set)."""

    if evidence.get("binding_schema_version") != _PAPER_BINDING_SCHEMA_VERSION:
        raise PaperBindingError("binding_schema_version must be 1")
    if evidence.get("spec_id") != spec.spec_id:
        raise PaperBindingError("binding spec_id does not match the run")
    if evidence.get("run_id") != run_id:
        raise PaperBindingError("binding run_id does not match the run")
    if evidence.get("dataset_id") != spec.dataset_id:
        raise PaperBindingError("binding dataset_id does not match the spec")
    for field in ("account_id", "candidate_id", "observation_lineage_hash"):
        if not str(evidence.get(field, "")).strip():
            raise PaperBindingError(
                f"binding {field} must be a non-empty string"
            )
    if evidence.get("completed") is not True:
        raise PaperBindingError("binding is not completed")


def build_promotion_evidence(
    spec: RunSpec,
    *,
    promotion_verdict_hash: str,
    paper_window_start: str,
    paper_window_end: str,
    observation_lineage_hash: str,
) -> dict[str, Any]:
    """Build the §3/§7 ``PROMOTED`` evidence: the complete paper window
    bound to the observation lineage plus the promotion gate verdict
    hash (``promotion.evaluate_challenger`` remains the gate authority)."""

    return {
        "evidence_schema_version": _PROMOTION_EVIDENCE_SCHEMA_VERSION,
        "spec_id": spec.spec_id,
        "promotion_verdict_hash": str(promotion_verdict_hash),
        "paper_window": {
            "start": str(paper_window_start),
            "end": str(paper_window_end),
        },
        "observation_lineage_hash": str(observation_lineage_hash),
        "paper_window_complete": True,
        "completed": True,
    }


def _validate_promotion_evidence(
    evidence: dict[str, Any], spec: RunSpec
) -> None:
    """Machine rules for ``PAPER_OBSERVING -> PROMOTED``: the promotion
    verdict hash and the complete, lineage-bound paper window."""

    if evidence.get("evidence_schema_version") != _PROMOTION_EVIDENCE_SCHEMA_VERSION:
        raise PromotionEvidenceError("evidence_schema_version must be 1")
    if evidence.get("spec_id") != spec.spec_id:
        raise PromotionEvidenceError("evidence spec_id does not match the run")
    if not str(evidence.get("promotion_verdict_hash", "")).strip():
        raise PromotionEvidenceError(
            "the promotion gate verdict hash is required (promotion.py "
            "G1–G7 remains the gate authority)"
        )
    window = evidence.get("paper_window")
    if not isinstance(window, dict) or not str(window.get("start", "")).strip() \
            or not str(window.get("end", "")).strip():
        raise PromotionEvidenceError("the paper window must be recorded")
    if evidence.get("paper_window_complete") is not True:
        raise PromotionEvidenceError(
            "the paper window is incomplete: promotion evidence cannot be "
            "generated before the window completes (contract §7)"
        )
    if not str(evidence.get("observation_lineage_hash", "")).strip():
        raise PromotionEvidenceError(
            "the paper window must be bound to the observation lineage"
        )
    if evidence.get("completed") is not True:
        raise PromotionEvidenceError("evidence is not completed")


def assert_protected_oos_access(
    state: str | None,
    run_type: str,
    *,
    action_mode: str = ACTION_RESEARCH,
) -> None:
    """Contract §6 isolation rule: evaluating the protected OOS window
    requires a lifecycle state at or after ``SEARCH_PLAN_ADMITTED`` and a
    run type of ``research`` or ``promotion``.  Engineering runs and
    pre-admission states are refused (the OOS never enters PR development
    feedback).  A03 (p27 I8/R8): the engineering/audit action modes are
    refused in **any** state — the locked window is unreachable from the
    explicit non-formal modes by construction."""

    if action_mode not in ACTION_MODES:
        raise LifecycleError(
            f"unregistered action mode {action_mode!r} (p27 §3/I3); "
            "the action-mode table is the single authority"
        )
    if action_mode in NON_FORMAL_ACTION_MODES:
        raise LifecycleError(
            f"protected OOS access refused for action mode {action_mode!r}; "
            "engineering/audit runs never read the locked OOS window in "
            "any state (contract §6, p27 I8)"
        )
    if run_type not in ("research", "promotion"):
        raise LifecycleError(
            f"protected OOS access refused for run type {run_type!r}; "
            "only research/promotion runs may evaluate the locked OOS "
            "window (contract §6)"
        )
    allowed = {"SEARCH_PLAN_ADMITTED", "OOS_QUALIFIED", "PAPER_OBSERVING",
               "PROMOTED"}
    if state not in allowed:
        raise LifecycleError(
            f"protected OOS access refused for lifecycle state {state!r}; "
            "the run must be at least SEARCH_PLAN_ADMITTED (contract §6)"
        )


def assert_formal_run_lifecycle(
    entry_type: str,
    data_dir: Path,
    spec_id: str | None = None,
    run_id: str | None = None,
    *,
    action_mode: str = ACTION_RESEARCH,
) -> str | None:
    """A03 entry gate (p27 I2, approved semantics: 「可空转」→「强制」).

    Formal action modes (``research``/``protected_oos``/``paper``) require
    a **real, established** lifecycle identity: a missing ``spec_id``/
    ``run_id``, a missing or empty event log, a terminal state, or a state
    below the entry's floor (p27 §3/I4) is refused — ``None`` is never a
    formal pass.  The pre-A03 "no identity → pre-activation legacy"
    passthrough survives **only** under the explicitly declared
    non-formal modes (``engineering``/``audit``), where it returns the
    replayed state or ``None`` (legacy/unverified marker).

    Returns the replayed lifecycle state (``None`` only in the explicit
    non-formal modes without identity).

    Entry-type → minimum-state mapping (p27 §3/I4 + I5, fixed per entry):

    * ``train`` → at least ``SPEC_LOCKED`` (the search action itself is
      gated separately at ``SEARCH_PLAN_ADMITTED`` — loading ≠ search)
    * ``evaluation`` → at least ``SEARCH_PLAN_ADMITTED``
    * ``backtest`` → at least ``SEARCH_PLAN_ADMITTED`` (the source run of
      a formally consumed strategy must be a qualified run)
    * ``paper_sim`` → exactly ``PAPER_OBSERVING``
    """

    if entry_type not in FORMAL_ENTRY_TYPES:
        raise LifecycleError(f"unknown formal entry type {entry_type!r}")
    if action_mode not in ACTION_MODES:
        raise LifecycleError(
            f"unregistered action mode {action_mode!r} (p27 §3/I3); the "
            "action-mode table is the single authority"
        )
    if action_mode in NON_FORMAL_ACTION_MODES:
        # The only sanctioned legacy passthrough (p27 I2/§5.3): an explicit
        # non-formal declaration may proceed without lifecycle identity.
        if spec_id is None or run_id is None:
            return None
        try:
            return replay_run(Path(data_dir), spec_id, run_id)
        except (FileNotFoundError, KeyError):
            return None  # no lifecycle events file → legacy/unverified
    # Formal modes: real identity is mandatory (R1) — refuse, never guess.
    if not str(spec_id or "").strip() or not str(run_id or "").strip():
        raise LifecycleError(
            f"{entry_type} entry in {action_mode!r} mode requires a real, "
            "established lifecycle identity (spec_id/run_id); a missing "
            "identity is a refusal, not a pre-activation pass (p27 I2, "
            "approval record §11 ruling 1)"
        )
    try:
        state = replay_run(Path(data_dir), spec_id, run_id)
    except (FileNotFoundError, KeyError) as exc:
        raise LifecycleError(
            f"{entry_type} entry refused: the run {str(spec_id)[:12]}…/"
            f"{str(run_id)[:12]}… has no readable lifecycle event log "
            f"({exc}); a formal {action_mode} run requires the "
            "append-only event chain (p27 I2/R3)"
        ) from exc
    if state is None:
        raise LifecycleError(
            f"{entry_type} entry refused: the run "
            f"{str(spec_id)[:12]}…/{str(run_id)[:12]}… has an empty "
            "lifecycle event log; preregister the run (genesis + spec "
            "lock) before a formal entry (p27 I2/R3)"
        )
    if state in TERMINAL_STATUSES:
        raise LifecycleError(
            f"{entry_type} entry refused: the run is TERMINAL at "
            f"{state!r}; terminal states have no out-edges and a "
            "terminated run is never continued (p27 I4/R5)"
        )
    floor = ENTRY_MIN_STATE[entry_type]
    if entry_type == "paper_sim":
        if state != floor:
            raise LifecycleError(
                f"paper simulation requires lifecycle state "
                f"{floor!r}; got {state!r} (contract §6: paper "
                "observation starts only from OOS_QUALIFIED)"
            )
    else:
        current_idx = stage_order.index(state) if state in stage_order else -1
        required_idx = stage_order.index(floor)
        if current_idx < required_idx:
            raise LifecycleError(
                f"{entry_type} entry in {action_mode!r} mode requires "
                f"lifecycle state at least {floor!r}; got {state!r} "
                "(p27 §3/I4 action-state table; removal condition: drive "
                "the run's qualification chain first)"
            )
    return state


stage_order = [
    "IDEA", "SPEC_LOCKED", "DATA_QUALIFIED", "FACTOR_SET_QUALIFIED",
    "SEARCH_PLAN_ADMITTED", "OOS_QUALIFIED", "PAPER_OBSERVING",
    "PROMOTED",
]


def _stage_sort_key(state: str) -> int:
    return stage_order.index(state) if state in stage_order else -1


def assert_qualified_source_run(
    store_root,
    spec_id: str,
    run_id: str,
    *,
    minimum_state: str = SEARCH_ACTION_MIN_STATE,
) -> str:
    """Consumption-side gate of p27 R7/I10 (A03 follow-up, user ruling
    1a of 2026-09-12): a formal consumer of a current-schema artifact
    must verify the artifact's source run has a lifecycle chain at or
    above ``minimum_state`` — schema alone is never qualification.

    Read-only: replays the trust-rooted chain (integrity/identity
    failures propagate as :class:`LifecycleError` subclasses) and takes
    **no** writer lock.  Used by the promotion entry (the fifth formal
    entry, p27 §2.1 assumption 4 / ruling 1a); the producing entries
    enforce the same floor through :meth:`RunContext.attach`.
    """

    if not str(spec_id or "").strip() or not str(run_id or "").strip():
        raise LifecycleError(
            "qualified-source check requires the artifact's lineage "
            "(spec_id/run_id); a current schema without lifecycle "
            "identity is never qualification evidence (p27 I10/R7)"
        )
    state = replay_run(Path(store_root), spec_id, run_id)
    if state is None:
        raise LifecycleError(
            f"formal consumption refused: the source run "
            f"{str(spec_id)[:12]}…/{str(run_id)[:12]}… has an empty "
            "lifecycle event log — the artifact was produced outside "
            "the lifecycle machine (engineering/audit products are "
            "never promotion evidence, p27 I10/R7/R14)"
        )
    if state in TERMINAL_STATUSES:
        raise LifecycleError(
            f"formal consumption refused: the source run is TERMINAL at "
            f"{state!r}; artifacts of a rejected/failed/retired run are "
            "not promotion input (p27 I10)"
        )
    if _stage_sort_key(state) < _stage_sort_key(minimum_state):
        raise LifecycleError(
            f"formal consumption refused: the source run is at "
            f"{state!r}, below the required {minimum_state!r}; the "
            "protocol evidence must come from a fully admitted search "
            "plan (p27 I5/R7)"
        )
    return state


def bound_paper_account_id(handle: RunHandle) -> str | None:
    """The account identity bound by the run's ``PAPER_OBSERVING``
    transition evidence (contract §7/p27 I9) — read from the registered
    binding artifact through the trust-rooted index, never assumed."""

    index = handle.load_index()
    for event in reversed(
        load_events(
            handle.run_dir / run_store.LIFECYCLE_FILENAME,
            expect_spec_id=handle.spec.spec_id,
            expect_run_id=handle.run_id,
        )
    ):
        if event.to_state != "PAPER_OBSERVING":
            continue
        evidence = event.evidence or {}
        artifact_id = str(evidence.get("artifact_id", ""))
        if not artifact_id:
            return None
        payload = read_registered_artifact(handle.run_dir, index, artifact_id)
        account = payload.get("account_id")
        return str(account) if account else None
    return None


class RunContext:
    """The single execution-identity object of the formal entries (A03
    p27 I1/I12): the open run handle, the frozen spec, the replayed
    lifecycle state, and the declared action mode.

    Formal entries receive **this** type — never a per-entry re-assembly
    of near-identity fields.  :meth:`attach` is the only construction
    path: it attaches the single writer, replays the event chain, and
    enforces the action-state table (p27 §3/I4) fail-closed.
    """

    def __init__(
        self,
        handle: RunHandle,
        *,
        entry_type: str,
        action_mode: str,
        writer: "LifecycleWriter",
        run_type: str | None = None,
    ):
        self.handle = handle
        self.entry_type = str(entry_type)
        self.action_mode = str(action_mode)
        self.run_type = run_type
        self._writer = writer

    # -- construction ---------------------------------------------------------

    @classmethod
    def attach(
        cls,
        store: RunStore,
        *,
        spec_id: str,
        run_id: str,
        entry_type: str,
        action_mode: str,
        run_type: str | None = None,
        spec: RunSpec | None = None,
    ) -> "RunContext":
        """Attach to an existing, preregistered run and enforce the entry
        gate with real identity (p27 I2/I4; R2/R3/R4/R5/R15)."""

        if entry_type not in FORMAL_ENTRY_TYPES:
            raise LifecycleError(
                f"unknown formal entry type {entry_type!r}"
            )
        if action_mode not in ACTION_MODES:
            raise LifecycleError(
                f"unregistered action mode {action_mode!r} (p27 §3/I3)"
            )
        if action_mode in NON_FORMAL_ACTION_MODES:
            raise LifecycleError(
                "RunContext is the formal execution identity; declare the "
                f"{action_mode!r} mode through the entry's explicit "
                "non-formal switch instead (p27 I2)"
            )
        if not str(spec_id or "").strip() or not str(run_id or "").strip():
            raise LifecycleError(
                f"{entry_type} entry requires a real lifecycle identity "
                "(spec_id/run_id); a missing identity is a refusal "
                "(p27 I2/R1)"
            )
        if spec is None:
            spec = _load_spec_from_run(store, str(spec_id), str(run_id))
        if spec.spec_id != str(spec_id):
            raise LifecycleError(
                f"context spec identity mismatch: resolved spec_id "
                f"{spec.spec_id!r} != requested {spec_id!r} (p27 I12/R2)"
            )
        handle = store.attach_run(spec, str(run_id))
        try:
            writer = LifecycleWriter(handle)
            context = cls(
                handle,
                entry_type=entry_type,
                action_mode=action_mode,
                writer=writer,
                run_type=run_type,
            )
            # One gate authority: the same formal entry gate, now with the
            # real identity (empty chain → R3, terminal → R5, floor →
            # R4/R15, all fail-closed).
            assert_formal_run_lifecycle(
                entry_type,
                store.root,
                spec.spec_id,
                handle.run_id,
                action_mode=action_mode,
            )
        except Exception:
            handle.close()
            raise
        return context

    # -- identity views -------------------------------------------------------

    @property
    def state(self) -> str | None:
        return self._writer.current_state

    @property
    def spec(self) -> RunSpec:
        return self.handle.spec

    @property
    def run_id(self) -> str:
        return self.handle.run_id

    @property
    def run_dir(self) -> Path:
        return self.handle.run_dir

    # -- action gates ---------------------------------------------------------

    def assert_search_ready(self) -> str:
        """I5: the search action requires the full qualification chain
        (≥ ``SEARCH_PLAN_ADMITTED``) at action time — a loading check is
        a different action and never implies search eligibility."""

        state = self.state
        if state is None or _stage_sort_key(state) < _stage_sort_key(
            SEARCH_ACTION_MIN_STATE
        ):
            raise LifecycleError(
                f"search action refused: the run is at {state!r}; formal "
                f"search requires the complete qualification chain (at "
                f"least {SEARCH_ACTION_MIN_STATE!r}: data + factor set + "
                "search plan). Removal condition: consume the §5.1/§5.2/"
                "§5.3 qualification transitions before searching "
                "(p27 I5/A03-A3)"
            )
        return state

    def assert_protected_oos_access(self) -> None:
        """I8: the context's state/mode/run-type must allow protected-OOS
        evaluation (delegates to the single authority)."""

        assert_protected_oos_access(
            self.state,
            self.run_type or "research",
            action_mode=self.action_mode,
        )

    def assert_paper_account(self, target_account_id: str | None) -> str:
        """I9/R6: the paper write target must be the account bound by the
        run's ``PAPER_OBSERVING`` evidence; a missing or foreign account
        is refused (no hardcoded ``current``)."""

        bound = bound_paper_account_id(self.handle)
        target = str(target_account_id or "").strip()
        if not bound:
            raise LifecycleError(
                "paper write refused: the run has no PAPER_OBSERVING "
                "account binding evidence; the writable account comes "
                "from the lifecycle binding, never from a default "
                "(p27 I9/A03-A7)"
            )
        if not target:
            raise LifecycleError(
                "paper write refused: no target account identity was "
                "provided (p27 I9)"
            )
        if bound != target:
            raise LifecycleError(
                f"paper write refused: cross-account write — the run is "
                f"bound to account {bound!r} but the target account is "
                f"{target!r} (p27 I9/R6)"
            )
        return bound

    def close(self) -> None:
        self.handle.close()


def assert_paper_account_writable(account_kind: str) -> None:
    """Contract §8: v1/v0 paper accounts are read-only by default —
    writes (resume/append/promotion) are refused; only current
    lifecycle-bound accounts are writable.  The explicit retire/precise
    reconstruction paths are administrative actions outside this guard."""

    if str(account_kind) != "current":
        raise LifecycleError(
            f"paper account kind {account_kind!r} is read-only by default "
            "(contract §8: v1/v0 accounts are never migrated implicitly; "
            "retire or an explicit precise-reconstruction migration only)"
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

    def qualify_factor_set(self, report: dict[str, Any]) -> LifecycleEvent:
        """``DATA_QUALIFIED -> FACTOR_SET_QUALIFIED`` under the §5.2
        verdict rules, consuming the D7 per-formula-dependency data
        eligibility gate (the report embeds its verdict).  Evidence first,
        machine verdict second; a refused report keeps the run in
        ``DATA_QUALIFIED`` with its evidence on file."""

        evidence = self._write_evidence(_RECORD_TYPE_FACTOR_REPORT, report)
        _validate_factor_report(report, self._handle.spec)
        return self.transition("FACTOR_SET_QUALIFIED", evidence=evidence)

    def admit_search_plan(self, evidence_payload: dict[str, Any]) -> LifecycleEvent:
        """``FACTOR_SET_QUALIFIED -> SEARCH_PLAN_ADMITTED`` under the §5.3
        rules (searcher legality, contract/version/domain/budget match,
        locked paired seeds, RL verdict encapsulation)."""

        evidence = self._write_evidence(_RECORD_TYPE_SEARCH_ADMISSION, evidence_payload)
        _validate_search_admission(evidence_payload, self._handle.spec)
        return self.transition("SEARCH_PLAN_ADMITTED", evidence=evidence)

    def qualify_oos(self, evidence_payload: dict[str, Any]) -> LifecycleEvent:
        """``SEARCH_PLAN_ADMITTED -> OOS_QUALIFIED`` under the §6 rules
        (write-once locked window, untainted)."""

        evidence = self._write_evidence(_RECORD_TYPE_OOS, evidence_payload)
        _validate_oos_evidence(evidence_payload, self._handle.spec)
        return self.transition("OOS_QUALIFIED", evidence=evidence)

    def start_paper_observation(
        self, evidence_payload: dict[str, Any]
    ) -> LifecycleEvent:
        """``OOS_QUALIFIED -> PAPER_OBSERVING``: bind the paper account
        (spec/candidate/account/dataset + observation lineage hash)."""

        evidence = self._write_evidence(_RECORD_TYPE_PAPER_BINDING, evidence_payload)
        _validate_paper_binding(
            evidence_payload, self._handle.spec, self._handle.run_id
        )
        return self.transition("PAPER_OBSERVING", evidence=evidence)

    def promote(self, evidence_payload: dict[str, Any]) -> LifecycleEvent:
        """``PAPER_OBSERVING -> PROMOTED``: the complete, lineage-bound
        paper window plus the promotion gate verdict hash."""

        evidence = self._write_evidence(_RECORD_TYPE_PROMOTION, evidence_payload)
        _validate_promotion_evidence(evidence_payload, self._handle.spec)
        return self.transition("PROMOTED", evidence=evidence)

    def reject(
        self, evidence: dict[str, str], *, completed: bool = True
    ) -> LifecycleEvent:
        """A gate state ``-> REJECTED`` (contract §4 machine
        discriminator: the stage's qualification gate COMPLETED with a
        failing verdict; its full evidence is registered and carried by
        the event)."""

        return self.transition(
            "REJECTED", evidence=evidence, completed=completed
        )

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
        if to_state == "REJECTED":
            # Contract §4 machine discriminator: a REJECTED transition is
            # only legal when the stage's qualification gate COMPLETED with
            # a failing verdict — its full evidence artifact must be
            # registered and the event carries completed=True.  An
            # evaluation that did not complete is a FAILED, never a
            # REJECTED (the meanings are not interchangeable).
            if completed is not True:
                raise LifecycleError(
                    "REJECTED requires completed=True (the qualification "
                    "gate finished with a failing verdict); an evaluation "
                    "that did not complete is a FAILED"
                )
            if evidence is None:
                raise LifecycleError(
                    "REJECTED requires the stage's complete gate-evidence "
                    "artifact (fail verdict with all check results)"
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
