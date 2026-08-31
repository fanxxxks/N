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
import threading
from pathlib import Path

import pytest

from ashare_model import run_store
from ashare_model.identity import canonical_json, canonical_json_strict
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
