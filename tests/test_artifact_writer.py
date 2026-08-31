"""RunStore-bound formal artifact write/read path (P8-05).

Covers the P8-05 acceptance list at the core level: current round-trip
for every artifact type, missing-identity rejection, cross-artifact
lineage consistency, v1-is-legacy-audit-but-never-formal, unknown/future
rejection, and failed writes that leave every existing file intact.
Identity oracles are the independent stdlib canonicalization calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ashare_data.config import BacktestConfig
from ashare_model.artifact_schemas import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactSchemaError,
    BacktestResultArtifact,
    PaperStateArtifact,
    ProtocolResultArtifact,
    StrategyArtifact,
    assert_consistent_lineage,
    classify_schema_version,
    require_current_schema,
)
from ashare_model.artifact_writer import (
    artifact_id_of,
    read_boundary_artifact,
    reconstruct_runspec_from_lineage,
    write_boundary_artifact,
)
from ashare_model.identity import candidate_id, formula_hash
from ashare_model.run_store import RunStore
from ashare_model.runspec import WindowSpec, build_runspec
from ashare_portfolio.execution_spec import portfolio_config_provenance

SPEC = None  # built lazily in the fixture (build_runspec needs tmp repo)


def _spec(tmp_path: Path):
    return build_runspec(
        dataset_id="ds-v2-0001",
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
        repo_root=tmp_path,
        require_clean_tree=False,
    )


@pytest.fixture
def handle(tmp_path: Path):
    # tmp git repo so git capture is hermetic
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=str(root), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=str(root), check=True, capture_output=True
    )
    (root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(root), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=str(root), check=True, capture_output=True
    )
    spec = build_runspec(
        dataset_id="ds-v2-0001",
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
        repo_root=root,
    )
    store = RunStore(tmp_path / "store")
    return store.open_run(spec)


def _strategy_input() -> dict:
    return {
        "formula": [4, 45, 3],
        "formula_text": "(VOL_20 CORR60 ATR_14)",
        "source": "rl",
        "direction": -1,
        "data_tier": None,
        "val_reward": 0.11,
        "val_icir": 0.24,
        "train_reward": -1.0,
        "train_icir": 0.13,
        "complexity_penalty": 0.0,
        "complexity_cost": 1.0,
        "active_ir": 0.5,
        "risk_exposure": 0.2,
        "average_turnover": 0.1,
        "capacity_utilization": 0.3,
        "eligible": True,
        "rejection_reasons": [],
        "history": [{"step": 0.0, "avg_reward": -1.0}],
        "searcher": "rl",
        "init_seed": 42,
        "device": "cpu",
        "model_version": 3,
        "feature_names": ["RET_1"],
        "operator_names": ["ADD"],
        "feature_version": "29ac4001dd3c",
        "grammar_version": 3,
        "reward_version": "14",
        "protocol_version": "25",
        "research_domain": "unified",
        "research_domain_version": 1,
        "execution_version": 2,
        "portfolio_constructor_version": 1,
        "portfolio_config": portfolio_config_provenance(BacktestConfig()),
        "semantic_cache_version": 1,
        "unique_semantic_evals": 100,
        "semantic_cache_stats": {"hits": 0},
        "search_contract_version": 2,
        "search_result": None,
        "dataset_id": "abc123",
        "searcher_candidate_label": "rl:4,45,3",
    }


def _protocol_input() -> dict:
    return {
        "protocol_version": "25",
        "data_tier_version": 1,
        "reward_version": "14",
        "research_domain": "unified",
        "research_domain_version": 1,
        "execution_version": 2,
        "portfolio_constructor_version": 1,
        "portfolio_config": portfolio_config_provenance(BacktestConfig()),
        "dataset_id": "abc123",
        "frequency": "daily",
        "horizon": 1,
        "tier": "screening",
        "steps": 150,
        "batch_size": 256,
        "seeds": [42, 7, 2024],
        "random_samples": 100,
        "random_seed": 123,
        "random_budget_matched": True,
        "random_budget": 38400,
        "folds": [{"train_end": "2020-12-31", "test_end": "2021-12-31"}],
        "baseline_signals": ["MOMENTUM_20"],
        "data_end_date": "2025-12-31",
        "universe_policy": None,
        "data_regime": None,
        "ledger": None,
        "n_candidates": 0,
        "rows": [],
        "aggregates": {},
        "stitched": {"n_trials": 0, "trials": []},
        "top_trial": None,
        "dsr": None,
        "max_t": None,
        "dsr_extra_trials": 0,
    }


def _backtest_input() -> dict:
    return {
        "formula": [4, 45, 3],
        "formula_text": "(VOL_20 CORR60 ATR_14)",
        "direction": -1,
        "dataset_id": "abc123",
        "metrics": {"sharpe": 1.2},
        "dates": ["20240102"],
        "equity_curve": [100.0],
        "benchmark": "all",
        "benchmark_equity": [100.0],
        "positions": [],
        "universe_policy": None,
    }


def _paper_input() -> dict:
    return {
        "initial_capital": 100000.0,
        "cash": 99000.0,
        "trade_count": 1,
        "last_exec_date": "20240102",
        "positions": {},
        "equity_history": [{"trade_date": "20240102", "equity": 100000.0}],
    }


ACCOUNT = "0123456789abcdef0123456789abcdef"


# -- schema version pin and identity derivation -------------------------------


def test_schema_version_is_v2():
    assert ARTIFACT_SCHEMA_VERSION == 2
    assert classify_schema_version({"artifact_schema_version": 1}) == "legacy"
    assert classify_schema_version({"artifact_schema_version": 2}) == "current"
    assert classify_schema_version({"artifact_schema_version": 3}) == "unknown"


def test_candidate_id_is_deterministic_content_identity():
    first = candidate_id("s" * 64, [4, 45, 3], -1)
    assert first == candidate_id("s" * 64, [4, 45, 3], -1)
    assert first != candidate_id("s" * 64, [4, 45, 3], 1)
    assert first != candidate_id("s" * 64, [4, 45], -1)
    assert first != candidate_id("t" * 64, [4, 45, 3], -1)
    assert first != formula_hash([4, 45, 3])
    assert len(first) == 64


# -- current round-trip for every artifact type -------------------------------


def test_round_trip_all_four_types(handle, tmp_path):
    spec = handle.spec
    cand = candidate_id(spec.spec_id, [4, 45, 3], -1)
    plan = (
        (StrategyArtifact, _strategy_input(), "strategy", cand, None),
        (ProtocolResultArtifact, _protocol_input(), "protocol", cand, None),
        (BacktestResultArtifact, _backtest_input(), "backtest", cand, None),
        (
            PaperStateArtifact,
            _paper_input(),
            "paper_state",
            cand,
            ACCOUNT,
        ),
    )
    for model_cls, payload, artifact_type, cid, account in plan:
        record = write_boundary_artifact(
            handle,
            artifact_type=artifact_type,
            model_cls=model_cls,
            payload=payload,
            candidate_id=cid,
            account_id=account,
            convenience_path=tmp_path / f"{artifact_type}.json",
        )
        stored = handle.read_artifact(record.artifact_id)
        assert stored["artifact_schema_version"] == 2
        assert stored["spec_id"] == spec.spec_id
        assert stored["run_id"] == handle.run_id
        assert stored["candidate_id"] == cid
        if account is not None:
            assert stored["account_id"] == ACCOUNT
        expected_id = hashlib.sha256(
            json.dumps(
                {"kind": "artifact", "value": stored},
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        assert record.artifact_id == expected_id == artifact_id_of(stored)
        mirrored = json.loads(
            (tmp_path / f"{artifact_type}.json").read_text(encoding="utf-8")
        )
        assert mirrored == stored


# -- missing identity rejected -------------------------------------------------


def test_missing_identity_rejected(handle):
    spec = handle.spec
    with pytest.raises(ArtifactSchemaError):
        write_boundary_artifact(
            handle,
            artifact_type="strategy",
            model_cls=StrategyArtifact,
            payload=_strategy_input(),
            candidate_id="",
        )
    with pytest.raises(ArtifactSchemaError):
        write_boundary_artifact(
            handle,
            artifact_type="paper_state",
            model_cls=PaperStateArtifact,
            payload=_paper_input(),
            candidate_id=candidate_id(spec.spec_id, [4, 45, 3], -1),
        )


def test_identity_mismatch_with_handle_rejected(handle):
    payload = dict(_strategy_input())
    payload["spec_id"] = "f" * 64  # forged identity differing from the run
    with pytest.raises(ArtifactSchemaError):
        write_boundary_artifact(
            handle,
            artifact_type="strategy",
            model_cls=StrategyArtifact,
            payload=payload,
            candidate_id=candidate_id(handle.spec.spec_id, [4, 45, 3], -1),
        )


# -- cross-artifact lineage consistency ---------------------------------------


def test_lineage_consistency_matrix(handle):
    spec = handle.spec
    cand = candidate_id(spec.spec_id, [4, 45, 3], -1)
    strategy = write_boundary_artifact(
        handle,
        artifact_type="strategy",
        model_cls=StrategyArtifact,
        payload=_strategy_input(),
        candidate_id=cand,
    )
    stored_strategy = handle.read_artifact(strategy.artifact_id)
    protocol = write_boundary_artifact(
        handle,
        artifact_type="protocol",
        model_cls=ProtocolResultArtifact,
        payload=_protocol_input(),
        candidate_id=cand,
    )
    stored_protocol = handle.read_artifact(protocol.artifact_id)
    same = assert_consistent_lineage(stored_strategy, stored_protocol)
    assert same == (spec.spec_id, handle.run_id)
    other = dict(stored_protocol)
    other["run_id"] = "b" * 32
    with pytest.raises(ArtifactSchemaError):
        assert_consistent_lineage(stored_strategy, other)
    with pytest.raises(ArtifactSchemaError):
        assert_consistent_lineage(stored_strategy, spec_id="c" * 64)
    forged = dict(stored_strategy)
    forged["run_id"] = "not-a-uuid"
    with pytest.raises(ArtifactSchemaError):
        assert_consistent_lineage(forged)


def test_reconstruct_runspec_from_registered_lineage(handle, tmp_path):
    cand = candidate_id(handle.spec.spec_id, [4, 45, 3], -1)
    record = write_boundary_artifact(
        handle,
        artifact_type="strategy",
        model_cls=StrategyArtifact,
        payload=_strategy_input(),
        candidate_id=cand,
    )
    payload = handle.read_artifact(record.artifact_id)
    rebuilt = reconstruct_runspec_from_lineage(
        RunStore(tmp_path / "store"), payload
    )
    assert rebuilt == handle.spec
    assert rebuilt.spec_id == payload["spec_id"]


def test_reconstruct_runspec_rejects_missing_or_inexact_source(handle, tmp_path):
    cand = candidate_id(handle.spec.spec_id, [4, 45, 3], -1)
    record = write_boundary_artifact(
        handle,
        artifact_type="strategy",
        model_cls=StrategyArtifact,
        payload=_strategy_input(),
        candidate_id=cand,
    )
    payload = handle.read_artifact(record.artifact_id)
    store = RunStore(tmp_path / "store")

    missing = dict(payload)
    missing["run_id"] = "f" * 32
    with pytest.raises(ArtifactSchemaError, match="reconstruct"):
        reconstruct_runspec_from_lineage(store, missing)

    runspec_path = handle.run_dir / "runspec.json"
    runspec_payload = json.loads(runspec_path.read_text(encoding="utf-8"))
    runspec_payload["requested_budget"] += 1
    runspec_path.write_text(json.dumps(runspec_payload), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match="spec_id"):
        reconstruct_runspec_from_lineage(store, payload)


# -- legacy is audit-only; formal consumption rejects -------------------------


def test_v1_payload_is_legacy_audit_only(tmp_path):
    v1 = {
        "artifact_schema_version": 1,
        "candidate_id": "rl:4,45,3",  # historical searcher-internal label
        "initial_capital": 1.0,
        "cash": 1.0,
        "trade_count": 0,
        "last_exec_date": None,
        "positions": {},
        "equity_history": [],
    }
    path = tmp_path / "sim_portfolio_state.json"
    path.write_text(json.dumps(v1), encoding="utf-8")
    assert classify_schema_version(v1) == "legacy"
    with pytest.raises(ArtifactSchemaError):
        require_current_schema(v1, artifact="paper")
    with pytest.raises(ArtifactSchemaError):
        read_boundary_artifact(path, model_cls=PaperStateArtifact, formal=True)
    audit = read_boundary_artifact(path, model_cls=PaperStateArtifact, formal=False)
    assert audit is not None and audit[1] == "legacy"
    assert audit[0]["candidate_id"] == "rl:4,45,3"


def test_unknown_future_version_rejected(tmp_path):
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps({"artifact_schema_version": 3}), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError):
        read_boundary_artifact(path, model_cls=StrategyArtifact, formal=False)
    with pytest.raises(ArtifactSchemaError):
        read_boundary_artifact(path, model_cls=StrategyArtifact, formal=True)


# -- failed writes leave every existing file intact ----------------------------


def test_failed_write_does_not_overwrite_convenience(handle, tmp_path):
    spec = handle.spec
    cand = candidate_id(spec.spec_id, [4, 45, 3], -1)
    target = tmp_path / "strategy.json"
    record = write_boundary_artifact(
        handle,
        artifact_type="strategy",
        model_cls=StrategyArtifact,
        payload=_strategy_input(),
        candidate_id=cand,
        convenience_path=target,
    )
    before_store = handle.load_index()["artifacts"]
    before_file = target.read_text(encoding="utf-8")
    broken = _strategy_input()
    broken["eligible"] = "not-a-bool"
    with pytest.raises(ArtifactSchemaError):
        write_boundary_artifact(
            handle,
            artifact_type="strategy",
            model_cls=StrategyArtifact,
            payload=broken,
            candidate_id=cand,
            convenience_path=target,
        )
    assert target.read_text(encoding="utf-8") == before_file
    assert handle.load_index()["artifacts"] == before_store
    assert handle.read_artifact(record.artifact_id)["eligible"] is True
