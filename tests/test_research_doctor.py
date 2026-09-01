"""P0-03 contract tests: the read-only research doctor.

The doctor aggregates code/data/gate/artifact/dependency/runtime facts
into one report and flags version conflicts.  Conflict rules are pure
(``build_report``), so the tests drive them with plain inputs; the CLI
gatherers are stubbed for exit-code/output tests.
"""

from __future__ import annotations

import json

import pytest

from ashare_data.config import BacktestConfig
from ashare_model.alphagpt import MODEL_VERSION
from ashare_model.evaluation import PROTOCOL_VERSION
from ashare_model.reward import REWARD_VERSION
from ashare_model import research_doctor as doctor
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import execution_provenance
from ashare_trading.golden import EXECUTION_SPEC_VERSION


def _current_portfolio_config() -> dict:
    return execution_provenance(BacktestConfig())["portfolio_config"]


def _green(legacy: bool = False, reasons: tuple[str, ...] = ()) -> dict:
    """All-healthy report inputs (current versions, dataset bound, gates ok)."""

    current = {
        "reward_version": REWARD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_version": MODEL_VERSION,
        "execution_version": EXECUTION_SPEC_VERSION,
        "portfolio_constructor_version": PORTFOLIO_CONSTRUCTOR_VERSION,
        "portfolio_config": _current_portfolio_config(),
    }
    return dict(
        code={"commit": "abc123", "branch": "main", "dirty": False},
        data={
            "dataset_id": "d1",
            "manifest_version": "1",
            "created_at": "t",
            "total_rows": 10,
            "db_path": "/db",
        },
        gates={
            "mode": "formal",
            "ok": True,
            "degraded": False,
            "checks": [
                {"name": f"G{i}", "ok": True, "detail": "ok"} for i in range(1, 8)
            ],
        },
        artifacts=[
            {
                "name": "strategy",
                "path": "/s",
                "exists": True,
                "legacy": legacy,
                "legacy_reasons": list(reasons),
                "fields": {"searcher": "gp", "dataset_id": "d1", **current},
            },
            {
                "name": "protocol",
                "path": "/p",
                "exists": True,
                "legacy": legacy,
                "legacy_reasons": list(reasons),
                "fields": {
                    "protocol_version": PROTOCOL_VERSION,
                    "reward_version": REWARD_VERSION,
                    "dataset_id": "d1",
                    "execution_version": EXECUTION_SPEC_VERSION,
                    "portfolio_constructor_version": (
                        PORTFOLIO_CONSTRUCTOR_VERSION
                    ),
                    "portfolio_config": _current_portfolio_config(),
                },
            },
        ],
        dependencies={"deap": "1.4.4", "torch": "2.11.0+cu128"},
        model_searcher="gp",
        runtime_estimates_={"basis": "b", "train_default_minutes": 6.0},
    )


def _report(**overrides) -> dict:
    inputs = _green()
    inputs.update(overrides)
    return doctor.build_report(**inputs)


def _errors(report: dict) -> list[str]:
    return [
        f["message"] for f in report["findings"] if f["severity"] == "error"
    ]


def test_healthy_report_when_everything_current():
    report = _report()
    assert report["healthy"] is True
    assert report["findings"] == []
    assert report["code"]["commit"] == "abc123"
    assert report["data"]["dataset_id"] == "d1"


def test_version_mismatch_without_legacy_flag_is_error():
    artifacts = _green()["artifacts"]
    artifacts[0]["fields"]["reward_version"] = "10"
    report = _report(artifacts=artifacts)
    assert report["healthy"] is False
    errors = _errors(report)
    assert any(
        f"strategy artifact reward_version 10 != current {REWARD_VERSION}" in e
        and "not marked legacy" in e
        for e in errors
    )


def test_partial_portfolio_config_without_legacy_flag_is_error():
    artifacts = _green()["artifacts"]
    artifacts[0]["fields"]["portfolio_config"] = {
        "portfolio_method": "equal_weight"
    }
    report = _report(artifacts=artifacts)
    assert report["healthy"] is False
    assert any("portfolio_config" in message for message in _errors(report))


def test_version_mismatch_with_legacy_flag_is_informational():
    inputs = _green(legacy=True, reasons=("reward_version 10 != current 13",))
    inputs["artifacts"][0]["fields"]["reward_version"] = "10"
    report = doctor.build_report(**inputs)
    assert report["healthy"] is True
    infos = [f for f in report["findings"] if f["severity"] == "info"]
    assert any("reward_version 10 != current" in f["message"] for f in infos)
    assert not any(f["severity"] == "error" for f in report["findings"])


def test_gate_failure_is_error():
    gates = _green()["gates"]
    gates["checks"][2]["ok"] = False
    gates["checks"][2]["detail"] = "boom"
    report = _report(gates=gates)
    assert report["healthy"] is False
    assert any("gate G3 failed: boom" in e for e in _errors(report))


def test_missing_dataset_id_is_error():
    data = _green()["data"]
    data["dataset_id"] = None
    report = _report(data=data)
    assert report["healthy"] is False
    assert any("no dataset_id" in e for e in _errors(report))


def test_artifact_dataset_mismatch_is_error_unless_legacy():
    artifacts = _green()["artifacts"]
    artifacts[1]["fields"]["dataset_id"] = "other"
    report = _report(artifacts=artifacts)
    assert report["healthy"] is False
    assert any("protocol artifact dataset_id other != current database d1" in e
               for e in _errors(report))

    inputs = _green(legacy=True, reasons=("old dataset",))
    inputs["artifacts"][1]["fields"]["dataset_id"] = "other"
    report = doctor.build_report(**inputs)
    assert report["healthy"] is True
    assert any(
        f["severity"] == "info" and "dataset_id other != current" in f["message"]
        for f in report["findings"]
    )


def test_missing_artifact_dataset_id_warns_when_not_legacy():
    artifacts = _green()["artifacts"]
    artifacts[0]["fields"]["dataset_id"] = None
    report = _report(artifacts=artifacts)
    warnings = [f for f in report["findings"] if f["severity"] == "warning"]
    assert any("carries no dataset_id" in w["message"] for w in warnings)
    assert report["healthy"] is True


def test_deap_required_for_gp_default_searcher():
    deps = _green()["dependencies"]
    deps["deap"] = None
    report = _report(dependencies=deps)
    assert report["healthy"] is False
    assert any("deap is not installed" in e for e in _errors(report))

    report = _report(dependencies=deps, model_searcher="rl")
    assert report["healthy"] is True


def test_absent_artifacts_are_not_conflicts():
    artifacts = _green()["artifacts"]
    for artifact in artifacts:
        artifact["exists"] = False
    report = _report(artifacts=artifacts)
    assert report["healthy"] is True
    assert report["findings"] == []


def test_estimate_run_minutes_is_monotonic_in_budget():
    small = doctor.estimate_run_minutes(150, 256)
    larger_steps = doctor.estimate_run_minutes(300, 256)
    larger_batch = doctor.estimate_run_minutes(150, 512)
    assert small > 0
    assert larger_steps > small
    assert larger_batch > small


class _StubModel:
    searcher = "gp"
    train_steps = 150
    batch_size = 256


class _StubDataConfig:
    data_dir = "."


class _StubProtocol:
    class _Tier:
        def __init__(self, steps, batch_size):
            self.steps = steps
            self.batch_size = batch_size

    screening = _Tier(150, 256)
    confirmation = _Tier(200, 512)
    folds = [1, 2]
    seeds = [42]


@pytest.fixture
def cli_env(monkeypatch):
    """Stub the CLI gatherers/config so main() is fully deterministic."""

    monkeypatch.setattr(doctor, "load_config", lambda *a, **k: {})
    monkeypatch.setattr(
        doctor, "make_data_config", lambda *a, **k: _StubDataConfig()
    )
    monkeypatch.setattr(doctor, "make_model_config", lambda *a, **k: _StubModel())
    monkeypatch.setattr(
        doctor, "make_protocol_config", lambda *a, **k: _StubProtocol()
    )
    monkeypatch.setattr(
        doctor, "gather_code_version", lambda root: {"commit": "c1", "branch": "b", "dirty": False}
    )
    monkeypatch.setattr(
        doctor,
        "gather_data_version",
        lambda cfg: {"dataset_id": "d1", "manifest_version": "1", "created_at": "t", "total_rows": 1},
    )
    monkeypatch.setattr(
        doctor,
        "gather_gates",
        lambda cfg, min_eligible=100: {
            "mode": "formal", "ok": True, "degraded": False,
            "checks": [{"name": f"G{i}", "ok": True, "detail": "ok"} for i in range(1, 8)],
        },
    )
    monkeypatch.setattr(
        doctor,
        "gather_artifacts",
        lambda data_dir: [
            {
                "name": "strategy", "path": "/s", "exists": True,
                "legacy": False, "legacy_reasons": [],
                "fields": {"searcher": "gp", "dataset_id": "d1",
                           "reward_version": REWARD_VERSION,
                           "protocol_version": PROTOCOL_VERSION,
                           "model_version": MODEL_VERSION,
                           "execution_version": EXECUTION_SPEC_VERSION,
                           "portfolio_constructor_version": (
                               PORTFOLIO_CONSTRUCTOR_VERSION
                           ),
                           "portfolio_config": _current_portfolio_config()},
            }
        ],
    )
    monkeypatch.setattr(
        doctor, "gather_dependencies", lambda: {"deap": "1.4.4", "torch": "2.11.0+cu128"}
    )


def test_cli_healthy_exits_zero_and_prints_json(capsys, cli_env):
    rc = doctor.main(["--json"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "HEALTHY" in captured.out
    payload = json.loads(captured.out.split("AlphaGPT research doctor")[0])
    assert payload["healthy"] is True
    assert payload["code"]["commit"] == "c1"


def test_cli_unhealthy_exits_one(capsys, cli_env, monkeypatch):
    monkeypatch.setattr(
        doctor, "gather_data_version", lambda cfg: {"dataset_id": None}
    )
    rc = doctor.main([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "NOT HEALTHY" in captured.err


def test_cli_output_writes_report_file(tmp_path, capsys, cli_env):
    out = tmp_path / "sub" / "doctor.json"
    rc = doctor.main(["--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["healthy"] is True
    assert payload["data"]["dataset_id"] == "d1"
