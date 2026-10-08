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


def _request(
    host: str, token: str | None = None, host_header: str | None = None
) -> SimpleNamespace:
    # Keys use the exact casing auth reads ("host" is lowercase, matching
    # Starlette's case-insensitive lookup of the Host header).
    headers: dict = {}
    if token is not None:
        headers["X-API-Token"] = token
    if host_header is not None:
        headers["host"] = host_header
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


# --- E2/t26 re-dispatch: RunStore-backed run inventory + detail ------------


def _webapi_run(tmp_path: Path):
    from tests.conftest import make_test_spec
    from ashare_model import lifecycle
    from ashare_model.run_store import RunStore

    # A01 C8: the RunStore root is the config-derived ``data_dir`` (the
    # same resolver the write side uses), not the repository ROOT.  The
    # fixture therefore writes a real config baseline and opens the run
    # under ``<root>/data/runs``.
    _make_root(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    spec = make_test_spec(dataset_id='ds-webapi-0001')
    store = RunStore(data_dir)
    handle = store.open_run(spec)
    writer = lifecycle.LifecycleWriter(handle)
    idea = {
        'idea_record_schema_version': 1,
        'hypothesis': 'webapi run view',
        'scope': 'webapi contract tests',
        'economic_mechanism': 'n/a',
        'non_goals': 'no research claim',
    }
    writer.record_idea(idea)
    return tmp_path, store, handle, writer, spec


def test_list_runs_reads_runstore_directly(tmp_path, monkeypatch):
    from webapi import service

    root, store, handle, writer, spec = _webapi_run(tmp_path)
    writer.lock_spec()
    monkeypatch.setattr(service, 'ROOT', root)
    result = service.list_runs()
    # A01 C8 (test updated under AGENTS §10.1 whitelist category 2): the
    # default root is the config-derived data_dir, not the repository
    # ROOT — the old assertion encoded defect F05.
    assert result['runs_dir'] == str(root / 'data' / 'runs')
    assert result['runs_dir_source'] == 'config'
    assert result['runs_dir_exists'] is True
    assert len(result['runs']) == 1
    entry = result['runs'][0]
    assert entry['spec_id'] == spec.spec_id
    assert entry['run_id'] == handle.run_id
    assert entry['state'] == 'SPEC_LOCKED'
    assert entry['dataset_id'] == 'ds-webapi-0001'
    assert entry['error'] is None


def test_list_runs_empty_store_is_empty_not_error(tmp_path):
    from webapi import service

    monkeypatch_needed = False
    result = service.list_runs(tmp_path)
    assert result['runs'] == []


def test_run_detail_endpoint_state_events_and_404(tmp_path, monkeypatch):
    from webapi import service
    from webapi.app import app

    root, store, handle, writer, spec = _webapi_run(tmp_path)
    writer.lock_spec()
    monkeypatch.setattr(service, 'ROOT', root)
    client = TestClient(app)
    response = client.get('/api/runs/' + spec.spec_id + '/' + handle.run_id)
    assert response.status_code == 200
    payload = response.json()
    assert payload['state'] == 'SPEC_LOCKED'
    assert payload['dataset_id'] == 'ds-webapi-0001'
    assert payload['events'][0]['to_state'] == 'IDEA'
    assert payload['events'][1]['to_state'] == 'SPEC_LOCKED'
    missing = client.get('/api/runs/nope/nope')
    assert missing.status_code == 404


def test_artifact_classification_delegates_to_authority(tmp_path):
    from ashare_model.artifact_schemas import classify_schema_version
    from webapi.service import _artifact_classification

    payload = {'artifact_schema_version': 1, 'formula': [1], 'direction': 1}
    path = tmp_path / 'strategy.json'
    import json as _json
    path.write_text(_json.dumps(payload), encoding='utf-8')
    assert _artifact_classification(path) == classify_schema_version(payload)
    # Non-JSON artifacts classify as binary (not silently healthy/unknown).
    binary = tmp_path / 'model.pt'
    binary.write_bytes(b'x')
    assert _artifact_classification(binary) == 'binary'


# --- A01: one RunStore root for readers and writers (F05) -------------------


def test_a01_default_run_root_is_config_data_dir(tmp_path, monkeypatch):
    """A01 C8: the default list_runs root equals the write-side root."""

    from ashare_model.run_store import RunStore
    from webapi import service

    root = _make_root(tmp_path)
    monkeypatch.setattr(service, 'ROOT', root)
    result = service.list_runs()
    assert Path(result['runs_dir']) == RunStore(root / 'data').runs_dir
    assert result['runs_dir_source'] == 'config'
    assert result['runs_dir_exists'] is False
    assert result['runs'] == []


def test_a01_list_runs_reports_missing_root_not_silently_empty(tmp_path):
    """A01 C9: an unresolvable/explicit root must be visible, not silently
    indistinguishable from 'no runs'."""

    from webapi import service

    result = service.list_runs(tmp_path)
    assert result['runs'] == []
    assert result['runs_dir_source'] == 'explicit'
    assert result['runs_dir_exists'] is False


def test_a01_run_under_config_data_dir_visible_via_api(tmp_path, monkeypatch):
    """A01 C8: a run written under the configured data_dir is visible
    through the default (config-resolved) list and detail paths."""

    from ashare_model.run_store import RunStore
    from tests.conftest import make_test_spec
    from webapi import service

    root = _make_root(tmp_path)
    data_dir = root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    store = RunStore(data_dir)
    spec = make_test_spec(dataset_id='ds-a01-roundtrip')
    handle = store.open_run(spec)
    monkeypatch.setattr(service, 'ROOT', root)

    listed = service.list_runs()
    assert listed['runs_dir'] == str(data_dir / 'runs')
    assert [entry['run_id'] for entry in listed['runs']] == [handle.run_id]
    assert listed['runs'][0]['dataset_id'] == 'ds-a01-roundtrip'

    detail = service.get_run_detail(None, spec.spec_id, handle.run_id)
    assert detail['spec_id'] == spec.spec_id
    assert detail['run_id'] == handle.run_id
    assert detail['dataset_id'] == 'ds-a01-roundtrip'


# --- P1-1: a corrupt/missing runspec must not 500 the run views --------------


def _extra_run(root, dataset_id: str):
    from ashare_model.run_store import RunStore
    from tests.conftest import make_test_spec

    store = RunStore(root / 'data')
    spec = make_test_spec(dataset_id=dataset_id)
    handle = store.open_run(spec)
    return spec, handle


def test_list_runs_corrupt_or_missing_runspec_degrades_entry(
    tmp_path, monkeypatch
):
    """P1-1: a truncated/half-written or missing runspec.json degrades that
    ONE entry (error set, dataset_id None) instead of crashing /api/runs;
    healthy runs in the same store keep their clean fields."""
    from webapi import service
    from webapi.app import app

    root, store, handle, writer, spec = _webapi_run(tmp_path)
    writer.lock_spec()
    corrupt_spec, corrupt_handle = _extra_run(root, 'ds-corrupt')
    missing_spec, missing_handle = _extra_run(root, 'ds-missing')
    runs_dir = root / 'data' / 'runs'
    (runs_dir / corrupt_spec.spec_id / corrupt_handle.run_id / 'runspec.json').write_text(
        '{"dataset_id": "ds-corrupt', encoding='utf-8'  # truncated mid-write
    )
    (runs_dir / missing_spec.spec_id / missing_handle.run_id / 'runspec.json').unlink()

    monkeypatch.setattr(service, 'ROOT', root)
    result = service.list_runs()
    by_run = {entry['run_id']: entry for entry in result['runs']}
    assert len(by_run) == 3

    healthy = by_run[handle.run_id]
    assert healthy['error'] is None
    assert healthy['dataset_id'] == 'ds-webapi-0001'

    corrupt = by_run[corrupt_handle.run_id]
    assert corrupt['dataset_id'] is None
    assert corrupt['error'] is not None

    missing = by_run[missing_handle.run_id]
    assert missing['dataset_id'] is None
    assert missing['error'] is not None

    # The endpoint must stay 200 with the degraded entries, not 500.
    client = TestClient(app)
    response = client.get('/api/runs')
    assert response.status_code == 200
    payload = response.json()
    degraded = {
        entry['run_id']: entry for entry in payload['runs']
    }
    assert degraded[corrupt_handle.run_id]['error']
    assert degraded[handle.run_id]['error'] is None


def test_run_detail_corrupt_runspec_is_404_not_500(tmp_path, monkeypatch):
    """P1-1: get_run_detail on a corrupt runspec surfaces as an explicit
    RunStoreError (404 with a readable detail), never a raw
    JSONDecodeError escaping as a 500."""
    from webapi import service
    from webapi.app import app

    root, store, handle, writer, spec = _webapi_run(tmp_path)
    writer.lock_spec()
    (root / 'data' / 'runs' / spec.spec_id / handle.run_id / 'runspec.json').write_text(
        '{"dataset_id": "bro', encoding='utf-8'
    )
    monkeypatch.setattr(service, 'ROOT', root)
    client = TestClient(app)
    response = client.get('/api/runs/' + spec.spec_id + '/' + handle.run_id)
    assert response.status_code == 404
    assert 'unreadable' in response.json()['detail']


def test_run_detail_missing_runspec_is_404_not_500(tmp_path, monkeypatch):
    from webapi import service
    from webapi.app import app

    root, store, handle, writer, spec = _webapi_run(tmp_path)
    writer.lock_spec()
    (root / 'data' / 'runs' / spec.spec_id / handle.run_id / 'runspec.json').unlink()
    monkeypatch.setattr(service, 'ROOT', root)
    client = TestClient(app)
    response = client.get('/api/runs/' + spec.spec_id + '/' + handle.run_id)
    assert response.status_code == 404
    assert 'unreadable' in response.json()['detail']


# --- P1-2: forwarded-Host hardening of the no-token loopback path ------------


def test_mutation_without_token_requires_loopback_host_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Direct loopback connections pass; a loopback client presenting an
    external Host header (the reverse-proxy / ssh -L / VS Code
    port-forward topology) is rejected with 403."""
    monkeypatch.setenv("ASHARE_WEBAPI_ROOT", str(tmp_path))
    for host_header in (
        "127.0.0.1:8000",
        "localhost:8000",
        "[::1]:8000",
        "::1",
        "",  # HTTP/1.0-style empty Host counts as loopback
    ):
        asyncio.run(
            auth.require_mutation_token(
                _request("127.0.0.1", host_header=host_header)
            )
        )
    # IPv4-mapped IPv6 loopback peer (dual-stack listener) is loopback too.
    asyncio.run(
        auth.require_mutation_token(
            _request("::ffff:127.0.0.1", host_header="127.0.0.1:8000")
        )
    )
    # The proxy scenario: loopback TCP peer, forwarded external Host.
    for host_header in (
        "quant.example.com:8000",
        "quant.example.com",
        "192.168.1.20:8000",
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                auth.require_mutation_token(
                    _request("127.0.0.1", host_header=host_header)
                )
            )
        assert exc_info.value.status_code == 403
        assert ".webapi_token" in exc_info.value.detail


def test_mutation_with_token_skips_host_header_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With a token file the Host check must not apply: a correctly
    tokened request through any proxy/Host is the documented exposure."""
    root = tmp_path / "config"
    root.mkdir(parents=True)
    (root / ".webapi_token").write_text("s3cret", encoding="utf-8")
    monkeypatch.setenv("ASHARE_WEBAPI_ROOT", str(tmp_path))
    asyncio.run(
        auth.require_mutation_token(
            _request("192.168.1.20", token="s3cret", host_header="quant.example.com")
        )
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            auth.require_mutation_token(
                _request(
                    "192.168.1.20", token="wrong", host_header="quant.example.com"
                )
            )
        )
    assert exc_info.value.status_code == 401


def test_mutation_non_ascii_token_file_is_401_not_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """hmac.compare_digest raises TypeError on non-ASCII str inputs; a
    non-ASCII token file (and any token attempt against it) must degrade
    to 401, not crash the dependency with a 500."""
    root = tmp_path / "config"
    root.mkdir(parents=True)
    (root / ".webapi_token").write_text("令牌-非ASCII", encoding="utf-8")
    monkeypatch.setenv("ASHARE_WEBAPI_ROOT", str(tmp_path))
    for attempt in ("令牌-非ASCII", "s3cret"):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                auth.require_mutation_token(_request("127.0.0.1", token=attempt))
            )
        assert exc_info.value.status_code == 401
    # A binary (non-UTF-8) token file is likewise an invalid token.
    (root / ".webapi_token").write_bytes(b"\xff\xfe\x00s3cret")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.require_mutation_token(_request("127.0.0.1", token="x")))
    assert exc_info.value.status_code == 401
