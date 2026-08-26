"""T1-01 contracts: the optional MLflow tracking bridge.

The archive JSON remains the primary experiment record; MLflow is an
additive, opt-in channel.  The bridge must never break an offline run:
every failure mode is a structured, logged outcome, and the mlflow import
is lazy so the module stays importable without mlflow installed.
"""

from __future__ import annotations

import sys

import pytest

from ashare_model.experiment_tracking import TrackingOutcome, log_run


class _StubActiveRun:
    info = type("Info", (), {"run_id": "run-123"})()


class _StubMlflow:
    """Records every call the bridge is allowed to make."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.fail_on = None

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name):
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name=None):
        self.calls.append(("start_run", run_name))
        if self.fail_on == "start_run":
            raise RuntimeError("boom")
        return _StubActiveRun()

    def active_run(self):
        return _StubActiveRun()

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics):
        self.calls.append(("log_metrics", metrics))

    def set_tag(self, key, value):
        self.calls.append(("set_tag", (key, value)))

    def log_artifact(self, path):
        self.calls.append(("log_artifact", path))

    def end_run(self):
        self.calls.append(("end_run", None))


@pytest.fixture
def stub_mlflow(monkeypatch):
    stub = _StubMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", stub)
    return stub


@pytest.fixture
def tracking_uri(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    yield


def test_disabled_via_env_var(monkeypatch):
    monkeypatch.setenv("ALPHAGPT_DISABLE_TRACKING", "1")
    outcome = log_run(experiment_name="x", metrics={"a": 1})
    assert isinstance(outcome, TrackingOutcome)
    assert not outcome.tracked
    assert outcome.reason == "disabled"


def test_mlflow_missing_is_structured(monkeypatch):
    # sys.modules[name] = None makes importlib raise ImportError, exactly
    # like an environment without mlflow installed.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    outcome = log_run(experiment_name="x")
    assert not outcome.tracked
    assert outcome.reason == "mlflow_not_installed"
    assert outcome.run_id is None


def test_uri_unset_does_not_touch_mlflow(stub_mlflow):
    outcome = log_run(experiment_name="x")
    assert not outcome.tracked
    assert outcome.reason == "tracking_uri_unset"
    assert stub_mlflow.calls == []  # the bridge never touched the stub


def test_uri_from_env_drives_tracking(stub_mlflow, tracking_uri):
    outcome = log_run(
        experiment_name="alphagpt/screening",
        params={"steps": 150},
        metrics={"icir": 0.42},
        tags={"git_commit": "abc123"},
        artifacts=["/tmp/manifest.json"],
    )
    assert outcome.tracked
    assert outcome.run_id == "run-123"
    assert outcome.reason == "ok"
    assert ("set_tracking_uri", "file:///tmp/mlruns") in stub_mlflow.calls
    assert ("set_experiment", "alphagpt/screening") in stub_mlflow.calls
    assert ("start_run", None) in stub_mlflow.calls
    assert ("log_params", {"steps": 150}) in stub_mlflow.calls
    assert ("log_metrics", {"icir": 0.42}) in stub_mlflow.calls
    assert ("set_tag", ("git_commit", "abc123")) in stub_mlflow.calls
    assert ("log_artifact", "/tmp/manifest.json") in stub_mlflow.calls
    assert ("end_run", None) in stub_mlflow.calls


def test_explicit_uri_overrides_env(stub_mlflow, tracking_uri):
    outcome = log_run(
        experiment_name="x", tracking_uri="file:///other", metrics={}
    )
    assert outcome.tracked
    assert ("set_tracking_uri", "file:///other") in stub_mlflow.calls


def test_tracking_failure_is_structured(stub_mlflow, tracking_uri):
    stub_mlflow.fail_on = "start_run"
    outcome = log_run(experiment_name="x")
    assert not outcome.tracked
    assert outcome.reason == "tracking_failed"
    assert "boom" in outcome.detail
