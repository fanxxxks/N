"""Content-addressed RunStore bound to the ledger (P8-04).

Layout per lifecycle contract §3 (``runs/<spec_id>/<run_id>/``):
``runspec.json`` (write-once), ``lifecycle.jsonl``, ``trials.jsonl``,
``artifacts/`` (content-addressed files) and ``index.json`` — the index
is the single trust anchor for artifacts. The store is single-writer
(O_CREAT|O_EXCL lock); reruns use a new run_id and never overwrite an
existing run. Ledger entries written through the store carry
``spec_id``/``run_id`` and ``LEDGER_SCHEMA_VERSION``; legacy (unversioned)
ledger lines keep their exact historical byte behavior and stay read-only.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

import pytest

from ashare_model import run_store
from ashare_model.identity import (
    canonical_json,
    canonical_json_strict,
    content_hash,
)
from ashare_model.ledger import (
    LEDGER_SCHEMA_VERSION,
    ExperimentLedger,
    LedgerEntry,
    LedgerIntegrityError,
)
from ashare_model.run_store import (
    INDEX_SCHEMA_VERSION,
    ArtifactRecord,
    ArtifactNotRegisteredError,
    RunExistsError,
    RunLockedError,
    RunSpecMismatchError,
    RunStore,
    RunStoreError,
    write_runspec_once,
)
from ashare_model.runspec import WindowSpec, build_runspec
from ashare_data.config import BacktestConfig
from ashare_portfolio.execution_spec import portfolio_config_provenance


def _spec() -> "object":
    return build_runspec(
        dataset_id="ds-store-0001",
        data_cutoff="20241231",
        frequency="daily",
        target="forward_return",
        horizon=1,
        requested_budget=2000,
        n_folds=5,
        dev_window=WindowSpec(start="20150101", end="20231231"),
        holdout_window=WindowSpec(start="20240101", end="20241231"),
        preregistered_contract_hash="a" * 64,
        resolved_thresholds={"factor_min_coverage": 0.9, "min_eligible": 100.0},
        portfolio_config=portfolio_config_provenance(BacktestConfig()),
        require_clean_tree=False,
    )


ARTIFACT_PAYLOAD = {
    "artifact_schema_version": 1,
    "formula": [3, 1, 2],
    "direction": 1,
    "metrics": {"sharpe": 1.25},
}


# -- layout and write-once semantics -----------------------------------------


def test_open_run_creates_contract_layout(tmp_path):
    store = RunStore(tmp_path)
    spec = _spec()
    with store.open_run(spec) as handle:
        run_dir = handle.run_dir
        assert run_dir == tmp_path / "runs" / spec.spec_id / handle.run_id
        assert (run_dir / "runspec.json").is_file()
        assert (run_dir / "lifecycle.jsonl").is_file()
        assert (run_dir / "trials.jsonl").is_file()
        assert (run_dir / "artifacts").is_dir()
        index = handle.load_index()
        assert index["index_schema_version"] == INDEX_SCHEMA_VERSION == 1
        assert index["spec_id"] == spec.spec_id
        assert index["run_id"] == handle.run_id
        assert index["artifacts"] == []


def test_runspec_json_is_write_once(tmp_path):
    spec = _spec()
    run_dir = tmp_path / "runs" / spec.spec_id / "run-x"
    run_dir.mkdir(parents=True)
    write_runspec_once(run_dir, spec)  # idempotent for identical content
    write_runspec_once(run_dir, spec)
    conflicting = json.loads((run_dir / "runspec.json").read_text(encoding="utf-8"))
    conflicting["dataset_id"] = "ds-other"
    (run_dir / "runspec.json").write_text(
        canonical_json(conflicting), encoding="utf-8"
    )
    with pytest.raises(RunSpecMismatchError):
        write_runspec_once(run_dir, spec)


def test_existing_run_is_never_overwritten(tmp_path):
    store = RunStore(tmp_path)
    spec = _spec()
    with store.open_run(spec, run_id="run-a") as first:
        first.record_trial(algorithm="trained", candidate="trained", seed=42)
    before = (first.run_dir / "trials.jsonl").read_text(encoding="utf-8")
    with pytest.raises(RunExistsError):
        store.open_run(spec, run_id="run-a")
    with store.open_run(spec) as second:
        assert second.run_id != "run-a"
        second.record_trial(algorithm="trained", candidate="trained", seed=43)
    assert (first.run_dir / "trials.jsonl").read_text(encoding="utf-8") == before
    assert (second.run_dir / "trials.jsonl").read_text(encoding="utf-8") != before


def test_concurrent_double_open_has_exactly_one_writer(tmp_path):
    store = RunStore(tmp_path)
    spec = _spec()
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def attempt():
        barrier.wait()
        try:
            handle = store.open_run(spec, run_id="run-race")
        except (RunLockedError, RunExistsError) as exc:
            outcomes.append(type(exc).__name__)
        else:
            outcomes.append("opened")
            handle.close()

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("opened") == 1
    assert len(outcomes) == 2


def test_operations_after_close_are_refused(tmp_path):
    store = RunStore(tmp_path)
    handle = store.open_run(_spec())
    handle.close()
    with pytest.raises(RunStoreError):
        handle.record_trial(algorithm="trained", candidate="trained")
    with pytest.raises(RunStoreError):
        handle.write_artifact("strategy", ARTIFACT_PAYLOAD)


# -- artifacts: content-addressed, index as the trust anchor -----------------


def test_artifact_roundtrip_and_index_registration(tmp_path):
    store = RunStore(tmp_path)
    spec = _spec()
    with store.open_run(spec) as handle:
        record = handle.write_artifact("strategy", ARTIFACT_PAYLOAD)
        assert isinstance(record, ArtifactRecord)
        assert record.spec_id == spec.spec_id
        assert record.run_id == handle.run_id
        assert record.path.startswith("artifacts/")
        assert handle.read_artifact(record.artifact_id) == ARTIFACT_PAYLOAD
        index = handle.load_index()
        assert [entry["artifact_id"] for entry in index["artifacts"]] == [
            record.artifact_id
        ]


def test_artifact_id_is_content_identity_regardless_of_key_order(tmp_path):
    store = RunStore(tmp_path)
    with store.open_run(_spec()) as handle:
        first = handle.write_artifact("strategy", ARTIFACT_PAYLOAD)
        reordered = {
            "metrics": {"sharpe": 1.25},
            "direction": 1,
            "formula": [3, 1, 2],
            "artifact_schema_version": 1,
        }
        second = handle.write_artifact("strategy", reordered)
        assert first.artifact_id == second.artifact_id
        assert len(handle.load_index()["artifacts"]) == 1


def test_artifact_id_matches_independent_oracle(tmp_path):
    store = RunStore(tmp_path)
    with store.open_run(_spec()) as handle:
        record = handle.write_artifact("strategy", ARTIFACT_PAYLOAD)
        expected = hashlib.sha256(
            json.dumps(
                {"kind": "artifact", "value": ARTIFACT_PAYLOAD},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        assert record.artifact_id == expected


def test_artifact_payload_must_be_strictly_serializable(tmp_path):
    from ashare_model.identity import CanonicalJSONError

    store = RunStore(tmp_path)
    with store.open_run(_spec()) as handle:
        with pytest.raises(CanonicalJSONError):
            handle.write_artifact("strategy", {"metrics": {"sharpe": float("nan")}})
        assert handle.load_index()["artifacts"] == []
        assert not any((handle.run_dir / "artifacts").iterdir())


def test_interrupted_write_leaves_no_trusted_half_product(tmp_path):
    store = RunStore(tmp_path)
    spec = _spec()
    with store.open_run(spec) as handle:
        # Simulate a crash after the file write but before index
        # registration: the orphan file must not be trusted.
        orphan_id = hashlib.sha256(
            json.dumps(
                {"kind": "artifact", "value": ARTIFACT_PAYLOAD},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        orphan = handle.run_dir / "artifacts" / f"{orphan_id}.json"
        orphan.parent.mkdir(exist_ok=True)
        orphan.write_text(canonical_json(ARTIFACT_PAYLOAD), encoding="utf-8")
        with pytest.raises(ArtifactNotRegisteredError):
            handle.read_artifact(orphan_id)
        # A well-behaved write of the same content registers it (recovery).
        record = handle.write_artifact("strategy", ARTIFACT_PAYLOAD)
        assert record.artifact_id == orphan_id
        assert handle.read_artifact(orphan_id) == ARTIFACT_PAYLOAD


def test_tampered_artifact_file_is_detected(tmp_path):
    store = RunStore(tmp_path)
    with store.open_run(_spec()) as handle:
        record = handle.write_artifact("strategy", ARTIFACT_PAYLOAD)
        target = handle.run_dir / record.path
        target.write_text('{"tampered": true}', encoding="utf-8")
        with pytest.raises(RunStoreError, match="hash"):
            handle.read_artifact(record.artifact_id)


# -- ledger binding: spec_id/run_id/schema version ---------------------------


def test_store_ledger_entries_carry_identity_and_version(tmp_path):
    store = RunStore(tmp_path)
    spec = _spec()
    with store.open_run(spec) as handle:
        trial_id = handle.record_trial(
            algorithm="trained", candidate="trained", seed=42
        )
        handle.complete_trial(trial_id, metrics={"sharpe": 1.0})
    lines = (handle.run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert payload["ledger_schema_version"] == LEDGER_SCHEMA_VERSION == 1
        assert payload["spec_id"] == spec.spec_id
        assert payload["run_id"] == handle.run_id
        fields_except_hash = {
            key: value for key, value in payload.items() if key != "entry_hash"
        }
        expected = hashlib.sha256(
            canonical_json(fields_except_hash).encode("utf-8")
        ).hexdigest()
        assert payload["entry_hash"] == expected
    # Reloading through the store verifies chain + identity.
    reloaded = ExperimentLedger(
        handle.run_dir / "trials.jsonl", run_id=handle.run_id, spec_id=spec.spec_id
    )
    assert len(reloaded) == 2


def test_legacy_ledger_lines_keep_historical_bytes_and_stay_openable(tmp_path):
    path = tmp_path / "legacy.jsonl"
    legacy = ExperimentLedger(path, run_id="run-legacy")
    legacy.record_trial(algorithm="trained", candidate="trained", seed=42)
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    # No new keys may appear in legacy-mode output.
    assert "spec_id" not in line
    assert "ledger_schema_version" not in line
    legacy_fields = LedgerEntry._fields_except()
    assert set(line) == set(legacy_fields) - {"spec_id", "ledger_schema_version"}
    expected = hashlib.sha256(
        canonical_json(
            {key: value for key, value in line.items() if key != "entry_hash"}
        ).encode("utf-8")
    ).hexdigest()
    assert line["entry_hash"] == expected
    reloaded = ExperimentLedger(path, run_id="run-legacy")
    assert len(reloaded) == 1


def test_legacy_ledger_cannot_be_bound_to_a_spec(tmp_path):
    path = tmp_path / "legacy.jsonl"
    legacy = ExperimentLedger(path, run_id="run-legacy")
    legacy.record_trial(algorithm="trained", candidate="trained", seed=42)
    with pytest.raises(LedgerIntegrityError, match="legacy"):
        ExperimentLedger(path, run_id="run-legacy", spec_id="spec-x")


def test_identity_mismatch_in_bound_ledger_rejected(tmp_path):
    path = tmp_path / "bound.jsonl"
    bound = ExperimentLedger(path, run_id="run-1", spec_id="spec-a")
    bound.record_trial(algorithm="trained", candidate="trained")
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    line["spec_id"] = "spec-b"
    # The tampered line no longer hashes correctly either — rebuild the
    # hash so the test isolates the identity check, not the hash check.
    fields = {k: v for k, v in line.items() if k != "entry_hash"}
    line["entry_hash"] = hashlib.sha256(
        canonical_json(fields).encode("utf-8")
    ).hexdigest()
    path.write_text(canonical_json(line) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="spec_id"):
        ExperimentLedger(path, run_id="run-1", spec_id="spec-a")


def test_tampered_v1_entry_detected(tmp_path):
    path = tmp_path / "bound.jsonl"
    bound = ExperimentLedger(path, run_id="run-1", spec_id="spec-a")
    bound.record_trial(algorithm="trained", candidate="trained")
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    line["algorithm"] = "forged"
    path.write_text(canonical_json(line) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        ExperimentLedger(path, run_id="run-1", spec_id="spec-a")


def test_sequence_gap_detected(tmp_path):
    path = tmp_path / "bound.jsonl"
    bound = ExperimentLedger(path, run_id="run-1", spec_id="spec-a")
    first = bound.record_trial(algorithm="trained", candidate="trained")
    bound.complete_trial(first)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[1] + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="sequence"):
        ExperimentLedger(path, run_id="run-1", spec_id="spec-a")


def test_future_ledger_schema_version_rejected(tmp_path):
    path = tmp_path / "future.jsonl"
    bound = ExperimentLedger(path, run_id="run-1", spec_id="spec-a")
    bound.record_trial(algorithm="trained", candidate="trained")
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    line["ledger_schema_version"] = LEDGER_SCHEMA_VERSION + 1
    fields = {k: v for k, v in line.items() if k != "entry_hash"}
    line["entry_hash"] = hashlib.sha256(
        canonical_json(fields).encode("utf-8")
    ).hexdigest()
    path.write_text(canonical_json(line) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="schema version"):
        ExperimentLedger(path, run_id="run-1", spec_id="spec-a")


# -- P8-06 research lifecycle: replay-only hash-chained state machine --------
#
# Contract: docs/p8_research_lifecycle_contract.md §3/§4/§5.1 (P8-06 staged
# activation per §4).  Scope baseline: t16 (AgentTeams alphagpt-p0-p1).
# State is derived ONLY by replaying ``lifecycle.jsonl``; every transition
# consumes a registered, hash-verified evidence artifact; edges outside the
# P8-06 activation set are fail-closed rejected.  There is no authority
# status JSON and no state setter.

LIFECYCLE_IDEA = {
    "idea_record_schema_version": 1,
    "hypothesis": "Industry-relative reversal predicts medium-horizon returns",
    "scope": "daily-bar features, formal mode, fold-0",
    "economic_mechanism": "stock-specific overreaction vs peers reverts",
    "non_goals": "no promotion claim, no paper operations, no searcher change",
}

_CONTRACT_PATH = Path(__file__).resolve().parents[1] / (
    "docs/p8_research_lifecycle_contract.md"
)


def _gate_checks(ok=True):
    """Eight G1..G8 checks using the producer's real naming convention
    (p16: the data-qualification space is G1–G8 once the freshness gate
    lands; the completeness rule demands all eight)."""
    return [
        {"name": f"G{i} synthetic check", "ok": ok, "detail": "ok" if ok else "fail"}
        for i in range(1, 9)
    ]


def _clean_data_report(spec, ok=True):
    """A formal, complete DataQualificationReport payload (all G ok unless
    ``ok=False``); ``gate_result_hash`` is a real content hash over the
    gate payload so the guard's hash-consistency check passes."""
    checks = _gate_checks(ok=ok)
    gate_payload = {"mode": "formal", "degraded": False, "checks": checks}
    return {
        "report_schema_version": 1,
        "spec_id": spec.spec_id,
        "dataset_id": spec.dataset_id,
        "data_cutoff": spec.data_cutoff,
        "expected_trading_days": 200,
        "actual_trading_days": 200,
        "missing_trading_days": [],
        "mode": "formal",
        "degraded": False,
        "gate_checks": checks,
        "gate_result_hash": content_hash("gate_result", gate_payload),
        "min_eligible": 100.0,
        "coverage": 0.99,
        "universe_contract": {"mode": "strict", "degraded": False},
        "degraded_sources": {"loader_frames": "clean"},
        "completed": True,
    }


def _lifecycle_writer(store, spec):
    from ashare_model import lifecycle

    handle = store.open_run(spec)
    return handle, lifecycle.LifecycleWriter(handle)


def test_lifecycle_status_set_version_and_terminals():
    from ashare_model import lifecycle

    assert lifecycle.LIFECYCLE_CONTRACT_VERSION == 1
    assert lifecycle.STATUSES == (
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
    assert lifecycle.TERMINAL_STATUSES == frozenset(
        {"REJECTED", "FAILED", "RETIRED"}
    )
    for terminal in lifecycle.TERMINAL_STATUSES:
        assert terminal not in lifecycle.LEGAL_EDGES  # no out-edges


def test_lifecycle_edge_graph_matches_contract_code_block():
    from ashare_model import lifecycle

    text = _CONTRACT_PATH.read_text(encoding="utf-8")
    match = re.search(r"```text\n(.*?)```", text, re.DOTALL)
    assert match is not None, "contract §4 transition block missing"
    lines = tuple(
        " ".join(line.split())
        for line in match.group(1).strip().splitlines()
        if line.strip()
    )
    assert lines == lifecycle.LEGAL_TRANSITION_LINES
    # Every mentioned state is a canonical status; the pair graph matches.
    for line in lines:
        src, dsts = line.split(" -> ")
        assert src.strip() in lifecycle.STATUSES
        for dst in dsts.split("|"):
            assert dst.strip() in lifecycle.STATUSES
            assert dst.strip() in lifecycle.LEGAL_EDGES[src.strip()]


def test_lifecycle_activated_edge_set_is_exactly_p8_06():
    from ashare_model import lifecycle

    legal = lifecycle.legal_edge_pairs()
    expected = {("IDEA", "SPEC_LOCKED"), ("SPEC_LOCKED", "DATA_QUALIFIED")}
    expected |= {pair for pair in legal if pair[1] in ("FAILED", "RETIRED")}
    assert lifecycle.ACTIVATED_EDGES == expected
    future = legal - lifecycle.ACTIVATED_EDGES
    assert ("DATA_QUALIFIED", "FACTOR_SET_QUALIFIED") in future
    assert ("SPEC_LOCKED", "REJECTED") in future
    assert ("OOS_QUALIFIED", "PAPER_OBSERVING") in future
    assert ("PROMOTED", "RETIRED") not in future  # management edge, active


def test_lifecycle_replay_only_state_and_status_json_ignored(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    writer.qualify_data(_clean_data_report(spec))
    assert writer.current_state == "DATA_QUALIFIED"
    assert lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id) == (
        "DATA_QUALIFIED"
    )
    # A forged authority-status file must be ignored: state derives only
    # from the append-only event chain (no status JSON authority).
    (handle.run_dir / "status.json").write_text(
        '{"state": "PROMOTED"}', encoding="utf-8"
    )
    assert lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id) == (
        "DATA_QUALIFIED"
    )


def test_lifecycle_genesis_requires_registered_idea_record(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle = store.open_run(spec)
    writer = lifecycle.LifecycleWriter(handle)
    assert writer.current_state is None  # uninitialized before genesis
    incomplete = dict(LIFECYCLE_IDEA)
    incomplete.pop("economic_mechanism")
    with pytest.raises(lifecycle.LifecycleError):
        writer.record_idea(incomplete)
    assert writer.current_state is None
    assert handle.load_index()["artifacts"] == []  # nothing registered
    writer.record_idea(LIFECYCLE_IDEA)
    assert writer.current_state == "IDEA"
    index = handle.load_index()
    assert index["artifacts"][0]["artifact_type"] == "idea_record"
    with pytest.raises(lifecycle.LifecycleError):
        writer.record_idea(LIFECYCLE_IDEA)  # genesis only once


def test_lifecycle_skip_level_transitions_rejected(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    with pytest.raises(lifecycle.LifecycleError):
        writer.lock_spec()  # still uninitialized (no genesis)
    writer.record_idea(LIFECYCLE_IDEA)
    with pytest.raises(lifecycle.IllegalTransitionError):
        writer.qualify_data(_clean_data_report(spec))  # IDEA -> DATA_QUALIFIED
    assert writer.current_state == "IDEA"


def test_lifecycle_future_stage_edges_fail_closed(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    with pytest.raises(lifecycle.StageNotActivatedError):
        writer.transition("REJECTED", reason="not before P8-08")
    writer.qualify_data(_clean_data_report(spec))
    with pytest.raises(lifecycle.StageNotActivatedError):
        writer.transition("FACTOR_SET_QUALIFIED")
    assert writer.current_state == "DATA_QUALIFIED"


def test_lifecycle_identity_mismatch_rejected_at_replay(tmp_path):
    from ashare_model import lifecycle
    from ashare_model.identity import content_hash

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    # Forge one more correctly-hashed event carrying a foreign spec_id.
    path = handle.run_dir / "lifecycle.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    previous = json.loads(lines[-1])
    payload = {
        "lifecycle_contract_version": lifecycle.LIFECYCLE_CONTRACT_VERSION,
        "seq": previous["seq"] + 1,
        "prev_hash": previous["event_hash"],
        "event_kind": "transition",
        "from_state": "IDEA",
        "to_state": "SPEC_LOCKED",
        "spec_id": "spec-foreign",
        "run_id": handle.run_id,
        "evidence": None,
        "completed": True,
        "reason": "",
        "ts": "2026-01-01T00:00:00+00:00",
    }
    forged = dict(payload, event_hash=content_hash("lifecycle-event", payload))
    path.write_text(
        "\n".join(lines + [canonical_json_strict(forged)]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(lifecycle.IdentityMismatchError):
        lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id)


def test_lifecycle_tampered_chain_detected(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    path = handle.run_dir / "lifecycle.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[0])
    forged["reason"] = "rewritten history"
    lines[0] = canonical_json_strict(forged)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(lifecycle.LifecycleIntegrityError):
        lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id)


def test_lifecycle_sequence_gap_detected(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    path = handle.run_dir / "lifecycle.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # Drop the genesis line: the chain then starts at seq 2 — a gap.
    path.write_text(lines[1] + "\n", encoding="utf-8")
    with pytest.raises(lifecycle.LifecycleIntegrityError, match="sequence"):
        lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id)


def test_lifecycle_terminal_states_have_no_out_edges(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.retire(reason="superseded idea")
    assert writer.current_state == "RETIRED"
    with pytest.raises(lifecycle.IllegalTransitionError):
        writer.fail(reason="after terminal")
    handle.close()
    assert lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id) == (
        "RETIRED"
    )


def test_lifecycle_gate_fail_refused_with_evidence_preserved(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    report = _clean_data_report(spec)
    report["gate_checks"] = _gate_checks(ok=False)
    with pytest.raises(lifecycle.DataQualificationError, match="G"):
        writer.qualify_data(report)
    assert writer.current_state == "SPEC_LOCKED"
    types = [e["artifact_type"] for e in handle.load_index()["artifacts"]]
    assert types.count("data_qualification_report") == 1  # evidence preserved


def test_lifecycle_rejects_seven_gate_report_as_incomplete(tmp_path):
    """p16 §4.1: the completeness rule is G1–G8 — a report carrying only
    the pre-G8 seven checks must be refused (fail-closed against a
    verdict that would silently ignore the freshness gate)."""
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    report = _clean_data_report(spec)
    report["gate_checks"] = _gate_checks(ok=True)[:7]
    report["gate_result_hash"] = lifecycle.content_hash(
        "gate_result",
        {
            "mode": report["mode"],
            "degraded": report["degraded"],
            "checks": report["gate_checks"],
        },
    )
    with pytest.raises(lifecycle.DataQualificationError, match="incomplete"):
        writer.qualify_data(report)


def test_lifecycle_rejects_failed_g8_check(tmp_path):
    """p16 §6.4: a failing G8 must refuse DATA_QUALIFIED with the check
    named in the error (never silently swallowed by the verdict)."""
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    report = _clean_data_report(spec)
    checks = _gate_checks(ok=True)
    checks[-1]["ok"] = False
    checks[-1]["detail"] = "daily 20260821 vs reference 20260901, lag=8 > 3"
    report["gate_checks"] = checks
    report["gate_result_hash"] = lifecycle.content_hash(
        "gate_result",
        {
            "mode": report["mode"],
            "degraded": report["degraded"],
            "checks": report["gate_checks"],
        },
    )
    with pytest.raises(lifecycle.DataQualificationError, match="G8"):
        writer.qualify_data(report)


def test_lifecycle_degraded_or_dev_or_missing_days_refused(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    dev = _clean_data_report(spec)
    dev["mode"] = "dev"
    with pytest.raises(lifecycle.DataQualificationError, match="mode"):
        writer.qualify_data(dev)
    degraded = _clean_data_report(spec)
    degraded["degraded"] = True
    with pytest.raises(lifecycle.DataQualificationError, match="degraded"):
        writer.qualify_data(degraded)
    gaps = _clean_data_report(spec)
    gaps["missing_trading_days"] = ["20240105"]
    with pytest.raises(lifecycle.DataQualificationError, match="missing"):
        writer.qualify_data(gaps)
    untyped = _clean_data_report(spec)
    untyped["degraded_sources"] = {
        "loader_frames": "untyped_in_current_data_layer"
    }
    with pytest.raises(lifecycle.DataQualificationError, match="loader"):
        writer.qualify_data(untyped)
    assert writer.current_state == "SPEC_LOCKED"


def test_lifecycle_clean_report_advances_to_data_qualified(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    writer.qualify_data(_clean_data_report(spec))
    assert writer.current_state == "DATA_QUALIFIED"
    handle.close()
    assert lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id) == (
        "DATA_QUALIFIED"
    )


def test_lifecycle_evidence_tamper_and_unregistered_rejected(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    writer.qualify_data(_clean_data_report(spec))
    # Content drift on a registered evidence artifact must fail the replay.
    report_id = handle.load_index()["artifacts"][-1]["artifact_id"]
    target = handle.run_dir / "artifacts" / f"{report_id}.json"
    original = target.read_text(encoding="utf-8")
    payload = json.loads(original)
    payload["coverage"] = 1.0
    target.write_text(canonical_json_strict(payload), encoding="utf-8")
    with pytest.raises(lifecycle.EvidenceError):
        lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id)
    target.write_text(original, encoding="utf-8")  # restore
    # An event referencing an unregistered artifact must fail the replay.
    path = handle.run_dir / "lifecycle.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    previous = json.loads(lines[-1])
    event = {
        "lifecycle_contract_version": lifecycle.LIFECYCLE_CONTRACT_VERSION,
        "seq": previous["seq"] + 1,
        "prev_hash": previous["event_hash"],
        "event_kind": "transition",
        "from_state": "DATA_QUALIFIED",
        "to_state": "FAILED",
        "spec_id": spec.spec_id,
        "run_id": handle.run_id,
        "evidence": {"artifact_id": "f" * 64, "artifact_type": "x"},
        "completed": False,
        "reason": "boom",
        "ts": "2026-01-01T00:00:00+00:00",
    }
    forged = dict(event, event_hash=content_hash("lifecycle-event", event))
    path.write_text(
        "\n".join(lines + [canonical_json_strict(forged)]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(lifecycle.EvidenceError):
        lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id)


def test_lifecycle_nan_evidence_rejected(tmp_path):
    from ashare_model import lifecycle
    from ashare_model.identity import CanonicalJSONError

    store = RunStore(tmp_path)
    handle = store.open_run(_spec())
    writer = lifecycle.LifecycleWriter(handle)
    poisoned = dict(LIFECYCLE_IDEA, coverage=float("nan"))
    with pytest.raises(CanonicalJSONError):
        writer.record_idea(poisoned)
    assert writer.current_state is None


def test_lifecycle_fail_and_retire_management_edges(tmp_path):
    from ashare_model import lifecycle

    store = RunStore(tmp_path)
    spec = _spec()
    handle, writer = _lifecycle_writer(store, spec)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.fail(reason="loader crashed mid-window")
    assert writer.current_state == "FAILED"
    assert writer.events[-1].completed is False
    handle.close()
    # Management continuation attaches to an EXISTING run explicitly.
    with pytest.raises(RunStoreError):
        store.attach_run(spec, "run-missing")
    # A runspec.json that no longer matches the spec object is a mismatch.
    runspec_path = handle.run_dir / "runspec.json"
    stored = json.loads(runspec_path.read_text(encoding="utf-8"))
    stored["dataset_id"] = "ds-tampered"
    runspec_path.write_text(canonical_json_strict(stored), encoding="utf-8")
    with pytest.raises(RunSpecMismatchError):
        store.attach_run(spec, handle.run_id)
    runspec_path.write_text(
        canonical_json_strict(spec.to_payload()), encoding="utf-8"
    )
    resumed = store.attach_run(spec, handle.run_id)
    writer2 = lifecycle.LifecycleWriter(resumed)
    assert writer2.current_state == "FAILED"
    # FAILED is terminal: retirement after failure is a non-contract edge.
    with pytest.raises(lifecycle.IllegalTransitionError):
        writer2.retire(reason="abandoned after failure")
    resumed.close()
    assert lifecycle.replay_run(tmp_path, spec.spec_id, handle.run_id) == (
        "FAILED"
    )
    # RETIRED applies to a non-terminal existing run (administrative).
    retire_handle = store.open_run(spec)  # fresh run_id, same spec
    retire_writer = lifecycle.LifecycleWriter(retire_handle)
    retire_writer.record_idea(LIFECYCLE_IDEA)
    retire_writer.retire(reason="superseded idea")
    assert retire_writer.current_state == "RETIRED"
    retire_handle.close()
    assert lifecycle.replay_run(
        tmp_path, spec.spec_id, retire_handle.run_id
    ) == "RETIRED"


def test_lifecycle_cli_start_show_state_and_retire(tmp_path, capsys):
    from ashare_model import lifecycle

    spec = _spec()
    idea_file = tmp_path / "idea.json"
    idea_file.write_text(canonical_json_strict(LIFECYCLE_IDEA), encoding="utf-8")
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        canonical_json_strict(spec.to_payload()), encoding="utf-8"
    )
    root = tmp_path / "store"
    code = lifecycle.main(
        [
            "start",
            "--root",
            str(root),
            "--spec-file",
            str(spec_file),
            "--idea-file",
            str(idea_file),
        ]
    )
    assert code == 0
    listing = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert listing["spec_id"] == spec.spec_id
    assert listing["state"] == "SPEC_LOCKED"
    state = lifecycle.main(
        [
            "show-state",
            "--root",
            str(root),
            "--spec-id",
            listing["spec_id"],
            "--run-id",
            listing["run_id"],
        ]
    )
    assert state == 0
    shown = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert shown["state"] == "SPEC_LOCKED"
    code = lifecycle.main(
        [
            "retire",
            "--root",
            str(root),
            "--spec-id",
            listing["spec_id"],
            "--run-id",
            listing["run_id"],
            "--reason",
            "superseded",
        ]
    )
    assert code == 0
    state = lifecycle.main(
        [
            "show-state",
            "--root",
            str(root),
            "--spec-id",
            listing["spec_id"],
            "--run-id",
            listing["run_id"],
        ]
    )
    assert state == 0
    shown = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert shown["state"] == "RETIRED"
