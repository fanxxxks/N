"""T4-01 contract tests: the automatic experiment trial ledger.

The ledger is the structural guarantee that **every** trial is recorded —
including failed and crashed ones — with no manual bookkeeping.  These
tests pin the contract before the implementation exists:

* append-only JSONL with a per-entry hash chain (tamper detection);
* every trial opens as ``running`` and must close as ``succeeded`` or
  ``failed``; an exception inside the trial context still closes as
  ``failed`` (never silently dropped);
* a trial that never closes stays visible as an open (leaked) trial, so
  an interrupted run is detected instead of silently omitted;
* closing is one-shot: a trial cannot be closed twice;
* reload verifies sequence continuity and the hash chain, raising
  ``LedgerIntegrityError`` on any gap or tamper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_model.ledger import (
    ExperimentLedger,
    LedgerIntegrityError,
    LedgerEntry,
)

ENTRY_FIELDS = (
    "sequence",
    "run_id",
    "trial_id",
    "status",
    "algorithm",
    "candidate",
    "seed",
    "fold_train_end",
    "fold_test_end",
    "started_at",
    "finished_at",
    "error",
    "metrics",
    "payload",
)


def _baseline_kwargs(**overrides):
    kwargs = dict(
        algorithm="trained",
        candidate="trained",
        seed=42,
        fold_train_end="2023-12-31",
        fold_test_end="2024-12-31",
    )
    kwargs.update(overrides)
    return kwargs


def _ledger(tmp_path: Path, **kwargs) -> ExperimentLedger:
    return ExperimentLedger(tmp_path / "ledger.jsonl", **kwargs)


def test_append_reload_roundtrip(tmp_path):
    ledger = _ledger(tmp_path, run_id="run-1")
    trial_id = ledger.record_trial(**_baseline_kwargs())
    ledger.complete_trial(trial_id, metrics={"sharpe": 1.23})

    reloaded = _ledger(tmp_path, run_id="run-1")
    entries = list(reloaded.iter_entries())
    assert [e.sequence for e in entries] == [1, 2]
    assert entries[0].status == "running"
    assert entries[1].status == "succeeded"
    assert entries[1].metrics == {"sharpe": 1.23}
    assert entries[0].trial_id == trial_id == entries[1].trial_id
    # Every entry carries the full record schema.
    for entry in entries:
        for field in ENTRY_FIELDS:
            assert hasattr(entry, field), field


def test_trial_context_marks_succeeded_on_success(tmp_path):
    ledger = _ledger(tmp_path)
    with ledger.trial(**_baseline_kwargs()) as trial_id:
        # The trial is recorded as open *before* any work starts.
        assert [e.trial_id for e in ledger.open_trials()] == [trial_id]
    assert ledger.open_trials() == []
    terminal = [e for e in ledger.iter_entries() if e.trial_id == trial_id]
    assert [e.status for e in terminal] == ["running", "succeeded"]
    assert terminal[1].finished_at is not None


def test_trial_context_marks_failed_on_exception(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with ledger.trial(**_baseline_kwargs()) as trial_id:
            raise RuntimeError("boom")
    assert ledger.open_trials() == []
    terminal = [e for e in ledger.iter_entries() if e.trial_id == trial_id]
    assert [e.status for e in terminal] == ["running", "failed"]
    assert "boom" in (terminal[1].error or "")


def test_failed_trial_is_never_silently_dropped(tmp_path):
    # A trial that crashes hard (no context-manager cleanup) must remain
    # visible as an open trial — the run is tainted, never silently clean.
    ledger = _ledger(tmp_path)
    trial_id = ledger.record_trial(**_baseline_kwargs())
    open_trials = ledger.open_trials()
    assert [e.trial_id for e in open_trials] == [trial_id]
    leaked = ledger.finalize()
    assert [e.trial_id for e in leaked] == [trial_id]


def test_finalize_clean_run_returns_nothing(tmp_path):
    ledger = _ledger(tmp_path)
    with ledger.trial(**_baseline_kwargs()):
        pass
    assert ledger.finalize() == []


def test_double_close_rejected(tmp_path):
    ledger = _ledger(tmp_path)
    trial_id = ledger.record_trial(**_baseline_kwargs())
    ledger.complete_trial(trial_id)
    with pytest.raises(ValueError):
        ledger.complete_trial(trial_id)
    with pytest.raises(ValueError):
        ledger.fail_trial(trial_id, error="late failure")
    # Fail after success is also a double close.
    trial_id2 = ledger.record_trial(**_baseline_kwargs(seed=99))
    ledger.fail_trial(trial_id2, error="first")
    with pytest.raises(ValueError):
        ledger.complete_trial(trial_id2)


def test_close_unknown_trial_raises(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(KeyError):
        ledger.complete_trial("nope")
    with pytest.raises(KeyError):
        ledger.fail_trial("nope", error="x")


def test_tampered_metrics_detected(tmp_path):
    ledger = _ledger(tmp_path)
    trial_id = ledger.record_trial(**_baseline_kwargs())
    ledger.complete_trial(trial_id, metrics={"sharpe": 1.0})
    path = tmp_path / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    # Flip a recorded metric value in place (keep the same line count).
    tampered = []
    for line in lines:
        payload = json.loads(line)
        if payload.get("metrics"):
            payload["metrics"] = {"sharpe": 99.0}
        tampered.append(json.dumps(payload, sort_keys=True))
    path.write_text("\n".join(tampered) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        _ledger(tmp_path)


def test_tampered_prev_hash_detected(tmp_path):
    ledger = _ledger(tmp_path)
    trial_id = ledger.record_trial(**_baseline_kwargs())
    ledger.complete_trial(trial_id)
    path = tmp_path / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])
    payload["prev_hash"] = "0" * 64
    lines[1] = json.dumps(payload, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        _ledger(tmp_path)


def test_truncated_ledger_raises(tmp_path):
    ledger = _ledger(tmp_path)
    trial_id = ledger.record_trial(**_baseline_kwargs())
    ledger.complete_trial(trial_id)
    ledger.record_trial(**_baseline_kwargs(seed=99))  # second, still-open trial
    path = tmp_path / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    # Drop the middle line: the surviving lines create a sequence gap.
    path.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError):
        _ledger(tmp_path)


def test_entries_for_trial_lifecycle(tmp_path):
    ledger = _ledger(tmp_path)
    trial_id = ledger.record_trial(**_baseline_kwargs())
    ledger.fail_trial(trial_id, error="nope")
    lifecycle = ledger.entries_for(trial_id)
    assert [e.status for e in lifecycle] == ["running", "failed"]
    assert ledger.entries_for("missing") == []


def test_query_by_algorithm_and_seed(tmp_path):
    ledger = _ledger(tmp_path)
    with ledger.trial(**_baseline_kwargs(algorithm="trained", seed=7)):
        pass
    with ledger.trial(**_baseline_kwargs(algorithm="trained", seed=8)):
        pass
    with ledger.trial(**_baseline_kwargs(algorithm="random_search", seed=7)):
        pass
    trained7 = list(ledger.trials_for(algorithm="trained", seed=7))
    assert len(trained7) == 1 and trained7[0].status == "succeeded"
    assert len(list(ledger.trials_for(algorithm="trained"))) == 2
    assert len(list(ledger.trials_for(seed=7))) == 2
    assert len(list(ledger.trials_for(algorithm="baseline"))) == 0


def test_hashes_deterministic_across_reloads(tmp_path):
    ledger = _ledger(tmp_path, run_id="run-x")
    with ledger.trial(**_baseline_kwargs(seed=1)):
        pass
    with ledger.trial(**_baseline_kwargs(seed=2)):
        pass
    hashes = [e.entry_hash for e in ledger.iter_entries()]
    reloaded = _ledger(tmp_path, run_id="run-x")
    assert [e.entry_hash for e in reloaded.iter_entries()] == hashes


def test_run_id_isolation(tmp_path):
    # One shared ledger file accumulates runs; handles are opened
    # sequentially (each new handle reloads the file).
    path = tmp_path / "ledger.jsonl"
    ledger_a = ExperimentLedger(path, run_id="run-a")
    with ledger_a.trial(**_baseline_kwargs(seed=1)):
        pass
    ledger_b = ExperimentLedger(path, run_id="run-b")
    with ledger_b.trial(**_baseline_kwargs(seed=2)):
        pass
    reloaded = ExperimentLedger(path, run_id="run-a")
    assert {e.run_id for e in reloaded.iter_entries()} == {"run-a", "run-b"}
    assert len(list(reloaded.trials_for(algorithm="trained", seed=2))) == 1


def test_empty_ledger_ok(tmp_path):
    ledger = _ledger(tmp_path)
    assert list(ledger.iter_entries()) == []
    assert ledger.finalize() == []


def test_payload_and_metrics_roundtrip(tmp_path):
    ledger = _ledger(tmp_path)
    payload = {"tier": "screening", "steps": 150, "batch": {"x": [1, 2, 3]}}
    trial_id = ledger.record_trial(**_baseline_kwargs(payload=payload))
    ledger.complete_trial(trial_id, metrics={"nested": {"a": 1.5, "b": None}})
    reloaded = _ledger(tmp_path)
    entries = list(reloaded.iter_entries())
    assert entries[0].payload == payload
    assert entries[1].metrics == {"nested": {"a": 1.5, "b": None}}


def test_append_only_file_format(tmp_path):
    ledger = _ledger(tmp_path)
    with ledger.trial(**_baseline_kwargs()):
        pass
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # One JSON object per line: no wrapping, no array brackets.
    for line in lines:
        payload = json.loads(line)
        assert isinstance(payload, dict)
        assert payload["entry_hash"]
