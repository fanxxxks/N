from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig, SimConfig
from ashare_data.db import AshareDB
from webapi import auth, service
from webapi.app import app


def _make_root(tmp_path: Path) -> Path:
    """A minimal project root with a config baseline (no DuckDB needed)."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "ashare_config.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data",
                "duckdb_path": "data/ashare.duckdb",
                "parquet_dir": "data/parquet",
                "backtest": {
                    "initial_capital": 100000.0,
                    "top_n": 30,
                    "commission_rate": 0.00025,
                    "min_commission": 5.0,
                    "stamp_tax_rate": 0.0005,
                    "transfer_fee_rate": 0.00001,
                    "slippage_rate": 0.0005,
                },
                "sim": {
                    "initial_capital": 100000.0,
                    "max_positions": 30,
                    "state_path": "data/sim_portfolio_state.json",
                    "orders_dir": "data/sim_orders",
                    "trades_dir": "data/sim_trades",
                    "stop_signal_path": "STOP_SIGNAL",
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_sim_config_patch_writes_overrides_and_effective(tmp_path: Path):
    root = _make_root(tmp_path)
    result = service.write_sim_config(
        service.SimConfigPatch(
            initial_capital=200000.0,
            max_positions=10,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
        ),
        root=root,
    )
    assert result["ok"]
    assert result["effective"]["initial_capital"] == 200000.0
    assert result["effective"]["max_positions"] == 10
    assert result["effective"]["commission_rate"] == 0.0003
    assert result["effective"]["stamp_tax_rate"] == 0.001
    # Untouched keys fall back to the YAML baseline.
    assert result["effective"]["min_commission"] == 5.0
    assert result["effective"]["slippage_rate"] == 0.0005

    # The override file is written next to the baseline; the baseline itself
    # is never touched.
    overrides = yaml.safe_load(
        (tmp_path / "config" / "runtime_overrides.yaml").read_text(encoding="utf-8")
    )
    assert overrides["sim"]["initial_capital"] == 200000.0
    assert overrides["backtest"]["commission_rate"] == 0.0003
    baseline = yaml.safe_load(
        (tmp_path / "config" / "ashare_config.yaml").read_text(encoding="utf-8")
    )
    assert baseline["sim"]["initial_capital"] == 100000.0


def test_sim_config_none_removes_override_key(tmp_path: Path):
    root = _make_root(tmp_path)
    service.write_sim_config(
        service.SimConfigPatch(initial_capital=300000.0, slippage_rate=0.01),
        root=root,
    )
    service.write_sim_config(
        service.SimConfigPatch(initial_capital=None, slippage_rate=None),
        root=root,
    )
    result = service.get_sim_config(root=root)
    assert result["effective"]["initial_capital"] == 100000.0
    assert result["effective"]["slippage_rate"] == 0.0005
    assert result["overrides"] == {}


def test_sim_config_pending_reset_flags_initial_capital(tmp_path: Path):
    root = _make_root(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sim_portfolio_state.json").write_text(
        json.dumps(
            {
                "initial_capital": 100000.0,
                "cash": 100000.0,
                "trade_count": 0,
                "positions": {},
                "equity_history": [],
            }
        ),
        encoding="utf-8",
    )
    assert service.get_sim_config(root=root)["pending_reset"] is False

    service.write_sim_config(
        service.SimConfigPatch(initial_capital=250000.0), root=root
    )
    assert service.get_sim_config(root=root)["pending_reset"] is True


def test_sim_config_patch_validation(tmp_path: Path):
    root = _make_root(tmp_path)
    with pytest.raises(ValueError):
        service.SimConfigPatch(initial_capital=-5.0)
    with pytest.raises(ValueError):
        service.SimConfigPatch(max_positions=0)
    with pytest.raises(ValueError):
        service.SimConfigPatch(commission_rate=1.5)
    with pytest.raises(ValueError):
        service.SimConfigPatch(min_commission=-1.0)
    assert service.get_sim_config(root=root)["overrides"] == {}


def test_sim_config_not_degraded_on_healthy_config(tmp_path: Path):
    root = _make_root(tmp_path)
    result = service.get_sim_config(root=root)
    assert result["config_degraded"] is False
    assert result["degraded_reason"] is None


def test_sim_config_marks_corrupted_baseline_as_degraded(tmp_path: Path):
    """A corrupted YAML baseline must not be presented as healthy factory
    defaults (§9 UI clause): the response must carry an explicit
    config_degraded marker and the reason, while still returning a
    defensively-shaped effective block."""
    root = _make_root(tmp_path)
    (tmp_path / "config" / "ashare_config.yaml").write_text(
        "backtest: [unclosed\n  : :", encoding="utf-8"
    )
    result = service.get_sim_config(root=root)
    assert result["config_degraded"] is True
    assert isinstance(result["degraded_reason"], str)
    assert result["degraded_reason"]
    # The defensive fallback keeps the response shape consumable, but the
    # caller can now see that these are defaults, not the real baseline.
    for key in (
        "initial_capital",
        "max_positions",
        "single_weight_cap",
        "commission_rate",
        "min_commission",
        "stamp_tax_rate",
        "transfer_fee_rate",
        "slippage_rate",
    ):
        assert key in result["effective"]


def test_sim_start_enforces_production_universe_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
    )
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
    monkeypatch.setattr(
        service,
        "_get_configs",
        lambda: (data_config, ModelConfig(), BacktestConfig(), SimConfig()),
    )
    spawned = False

    def fail_if_spawned():
        nonlocal spawned
        spawned = True
        raise AssertionError("job manager must not be constructed")

    monkeypatch.setattr(service, "_job_manager", fail_if_spawned)
    result = service.sim_start(service.SimStartRequest())
    assert result["ok"] is False
    assert "production universe contract violation" in result["reason"]
    assert spawned is False


# --- M6: control-endpoint authorization -------------------------------------


def _request(host: str, token: str | None = None) -> SimpleNamespace:
    headers = {"X-API-Token": token} if token is not None else {}
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=host),
    )


def test_mutation_dependency_without_token_file_allows_loopback_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ASHARE_WEBAPI_ROOT", str(tmp_path))
    # No token file: loopback passes, anything else is rejected with a hint.
    asyncio.run(auth.require_mutation_token(_request("127.0.0.1")))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.require_mutation_token(_request("192.168.1.20")))
    assert exc_info.value.status_code == 403
    assert ".webapi_token" in exc_info.value.detail


def test_mutation_dependency_requires_matching_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "config"
    root.mkdir(parents=True)
    (root / ".webapi_token").write_text("s3cret", encoding="utf-8")
    monkeypatch.setenv("ASHARE_WEBAPI_ROOT", str(tmp_path))
    asyncio.run(auth.require_mutation_token(_request("192.168.1.20", token="s3cret")))
    for bad in (None, "", "wrong"):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                auth.require_mutation_token(_request("192.168.1.20", token=bad))
            )
        assert exc_info.value.status_code == 401


def test_mutating_endpoints_enforce_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # With a token file, POST/PUT control endpoints demand the matching
    # header before the service layer is even reached.
    root = tmp_path / "config"
    root.mkdir(parents=True)
    (root / ".webapi_token").write_text("s3cret", encoding="utf-8")
    monkeypatch.setenv("ASHARE_WEBAPI_ROOT", str(tmp_path))
    monkeypatch.setattr(
        service, "sim_stop_run", lambda: {"ok": True, "state": "idle"}
    )
    client = TestClient(app)
    assert client.post("/api/sim/stop").status_code == 401
    assert client.post("/api/sim/stop", headers={"X-API-Token": "wrong"}).status_code == 401
    ok = client.post("/api/sim/stop", headers={"X-API-Token": "s3cret"})
    assert ok.status_code == 200
    assert ok.json()["ok"] is True
    # Read endpoints stay unauthenticated.
    assert client.get("/api/health").status_code == 200
