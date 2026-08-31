"""Typed artifact schemas (P7-C, docs/p7_artifact_schema_contract.md).

The contract: the four boundary artifacts (strategy / protocol / backtest /
paper state) carry ``artifact_schema_version``; writers validate
fail-closed before persisting; readers run the classify matrix — current
validates, legacy flows through the pre-contract path byte-identically,
unknown/future versions are hard-rejected.  These tests encode §6 (RED
first): schema round-trips, missing-provenance rejection, top-level
forbid, the classify matrix and the write/read wiring.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from ashare_model.artifact_schemas import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactSchemaError,
    BacktestResultArtifact,
    PaperStateArtifact,
    ProtocolResultArtifact,
    StrategyArtifact,
    classify_schema_version,
)

# --- minimal valid payloads (keys enumerated from the writer code, §3) ----


def _strategy_payload() -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "spec_id": "a" * 64,
        "run_id": "b" * 32,
        "candidate_id": "c" * 64,
        "formula": [4, 45, 3],
        "searcher_candidate_label": "rl:4,45,3",
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
        "grammar_version": 2,
        "reward_version": "14",
        "protocol_version": "24",
        "research_domain": "unified",
        "research_domain_version": 1,
        "execution_version": 2,
        "portfolio_constructor_version": 1,
        "portfolio_config": {"top_n": 10},
        "semantic_cache_version": 1,
        "unique_semantic_evals": 100,
        "semantic_cache_stats": {"hits": 0},
        "search_contract_version": 1,
        "search_result": None,
        "dataset_id": "abc123",
        # RL-only optional fields
        "rl_initialization": "random",
        "imitation": None,
        "experimental": True,
    }


def _protocol_payload() -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "spec_id": "a" * 64,
        "run_id": "b" * 32,
        "candidate_id": "c" * 64,
        "protocol_version": "24",
        "data_tier_version": 1,
        "reward_version": "14",
        "research_domain": "unified",
        "research_domain_version": 1,
        "execution_version": 2,
        "portfolio_constructor_version": 1,
        "portfolio_config": {"top_n": 10},
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


# --- §4 classify matrix -----------------------------------------------------


class TestClassifyMatrix:
    def test_missing_key_is_legacy(self):
        assert classify_schema_version({"formula": []}) == "legacy"

    def test_current_version(self):
        assert (
            classify_schema_version(
                {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION}
            )
            == "current"
        )

    @pytest.mark.parametrize("bad", [3, 0, -1, "1", 1.0, True, [1]])
    def test_unknown_versions(self, bad):
        assert (
            classify_schema_version({"artifact_schema_version": bad}) == "unknown"
        )


# --- §3 schema spine --------------------------------------------------------


class TestStrategySchema:
    def test_round_trip(self):
        payload = _strategy_payload()
        artifact = StrategyArtifact.validate_payload(payload)
        assert artifact.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
        assert artifact.to_payload()["formula"] == [4, 45, 3]

    def test_missing_dataset_id_key_fails(self):
        # §6.1: the key may be null (pre-T1-01 databases) but must exist.
        payload = _strategy_payload()
        del payload["dataset_id"]
        with pytest.raises(ValidationError):
            StrategyArtifact.model_validate(payload)

    def test_unknown_top_level_key_forbidden(self):
        # §6.6: top-level extra=forbid — new provenance needs a schema bump.
        payload = _strategy_payload()
        payload["surprise"] = 1
        with pytest.raises(ValidationError):
            StrategyArtifact.model_validate(payload)

    def test_unknown_schema_version_rejected(self):
        payload = _strategy_payload()
        payload["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION + 1
        with pytest.raises(ArtifactSchemaError):
            StrategyArtifact.validate_payload(payload)

    def test_legacy_classification_rejected_for_current_validation(self):
        payload = _strategy_payload()
        del payload["artifact_schema_version"]
        with pytest.raises(ArtifactSchemaError):
            StrategyArtifact.validate_payload(payload)

    def test_legacy_stamp_fields_tolerated(self):
        payload = _strategy_payload()
        payload.update(
            {
                "legacy": True,
                "legacy_reason": ["reward_version 10 != current 14"],
                "legacy_stamped_at": "2026-08-27T10:31:32+00:00",
            }
        )
        artifact = StrategyArtifact.validate_payload(payload)
        assert artifact.legacy is True


class TestProtocolSchema:
    def test_round_trip(self):
        artifact = ProtocolResultArtifact.validate_payload(_protocol_payload())
        assert artifact.n_candidates == 0

    def test_missing_rows_key_fails(self):
        payload = _protocol_payload()
        del payload["rows"]
        with pytest.raises(ValidationError):
            ProtocolResultArtifact.model_validate(payload)

    def test_unknown_top_level_key_forbidden(self):
        payload = _protocol_payload()
        payload["surprise"] = 1
        with pytest.raises(ValidationError):
            ProtocolResultArtifact.model_validate(payload)

    def test_unknown_schema_version_rejected(self):
        payload = _protocol_payload()
        payload["artifact_schema_version"] = 99
        with pytest.raises(ArtifactSchemaError):
            ProtocolResultArtifact.validate_payload(payload)


# --- write-path wiring ------------------------------------------------------


class TestStrategyWritePath:
    def _fake_trainer(self, tmp_path):
        """Minimal AshareTrainer instance for _write_artifact (no __init__)."""
        from ashare_data.config import BacktestConfig, DataConfig, ModelConfig
        from ashare_model.candidates import CandidateScore
        from ashare_model.train import AshareTrainer
        from ashare_model.vocab import FORMULA_VOCAB

        trainer = AshareTrainer.__new__(AshareTrainer)
        score = CandidateScore(
            tokens=(4, 45, 3),
            candidate_id="rl:4,45,3",
            formula_text="(VOL_20 CORR60 ATR_14)",
            source="rl",
            direction=-1,
            val_reward=0.11,
            val_icir=0.24,
            train_reward=-1.0,
            train_icir=0.13,
            complexity_penalty=0.0,
            eligible=True,
            rejection_reasons=(),
        )
        trainer.selection_result = SimpleNamespace(selected=score)
        trainer.best_tokens = [4, 45, 3]
        trainer.history = []
        trainer.init_seed = 42
        trainer.vocab = FORMULA_VOCAB
        trainer.domain_id = "unified"
        trainer.model_config = ModelConfig(max_formula_len=6)
        trainer.backtest_config = BacktestConfig()
        trainer.semantic_cache = SimpleNamespace(budget_used=0, stats=lambda: {})
        trainer.search_result = None
        trainer.loader = SimpleNamespace(
            dataset_id="abc123", dates=["20240102", "20241231"]
        )
        trainer.data_config = DataConfig(
            data_dir=tmp_path,
            duckdb_path=tmp_path / "ashare.duckdb",
            parquet_dir=tmp_path / "parquet",
            start_date="20150101",
        )
        trainer.rl_initialization = "random"
        trainer.imitation_result = None
        trainer.model = SimpleNamespace(state_dict=lambda: {})
        return trainer

    def test_write_artifact_stamps_and_validates(self, tmp_path):
        from ashare_model.time_contract import TrainingTimeContract  # noqa: F401

        trainer = self._fake_trainer(tmp_path)
        import torch

        out = trainer._write_artifact(
            contract=None,
            vm_device=torch.device("cpu"),
            seed=42,
            requested_budget=100,
        )
        assert out == [4, 45, 3]
        payload = json.loads(
            (tmp_path / "best_ashare_strategy.json").read_text(encoding="utf-8")
        )
        assert payload["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
        StrategyArtifact.validate_payload(payload)
        from ashare_model.artifact_writer import artifact_id_of
        from ashare_model.run_store import RunStore

        index = RunStore(tmp_path).load_index(payload["spec_id"], payload["run_id"])
        assert index["artifacts"][0]["artifact_id"] == artifact_id_of(payload)

    def test_write_artifact_fail_closed(self, tmp_path, monkeypatch):
        """§2.1: a payload that fails validation must never reach disk."""
        trainer = self._fake_trainer(tmp_path)
        import torch

        def _boom(payload):
            raise ArtifactSchemaError("deliberate strategy validation failure")

        monkeypatch.setattr(
            StrategyArtifact, "validate_payload", staticmethod(_boom)
        )
        with pytest.raises(ArtifactSchemaError):
            trainer._write_artifact(
                contract=None,
                vm_device=torch.device("cpu"),
                seed=42,
                requested_budget=100,
            )
        assert not (tmp_path / "best_ashare_strategy.json").exists()


class TestProtocolWritePath:
    def test_build_result_stamps_and_validates(self):
        from ashare_data.config import ProtocolConfig, TierConfig
        from ashare_model.evaluation import build_result

        result = build_result(
            ProtocolConfig(),
            "screening",
            TierConfig(0, 0),
            [],
            spec_id="a" * 64,
            run_id="b" * 32,
            dataset_id="abc123",
        )
        assert result["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
        ProtocolResultArtifact.validate_payload(result)


# --- read-path wiring -------------------------------------------------------


class TestReadPath:
    def test_load_trial_rows_rejects_unknown_version(self, tmp_path):
        from ashare_model.evaluation import load_trial_rows

        path = tmp_path / "protocol_result.json"
        path.write_text(
            json.dumps({"artifact_schema_version": 99, "rows": []}),
            encoding="utf-8",
        )
        with pytest.raises(ArtifactSchemaError):
            load_trial_rows(str(path))

    def test_load_trial_rows_legacy_unchanged(self, tmp_path):
        from ashare_model.evaluation import load_trial_rows

        path = tmp_path / "protocol_result.json"
        path.write_text(
            json.dumps({"dataset_id": None, "rows": [{"candidate": "x"}]}),
            encoding="utf-8",
        )
        assert load_trial_rows(str(path)) == [{"candidate": "x"}]

    def test_load_trial_rows_current_validates(self, tmp_path):
        from ashare_model.evaluation import load_trial_rows

        payload = _protocol_payload()
        payload["rows"] = [{"candidate": "x"}]
        path = tmp_path / "protocol_result.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert load_trial_rows(str(path)) == [{"candidate": "x"}]

    def test_run_sim_load_formula_rejects_unknown_version(self, tmp_path):
        from ashare_trading.run_sim import SimulationRunner

        (tmp_path / "best_ashare_strategy.json").write_text(
            json.dumps({"artifact_schema_version": 99, "formula": [1]}),
            encoding="utf-8",
        )
        sim = SimulationRunner.__new__(SimulationRunner)
        sim.data_config = SimpleNamespace(data_dir=tmp_path)
        sim.loader = SimpleNamespace(dataset_id="abc123")
        with pytest.raises(ArtifactSchemaError):
            sim.load_formula()

    def test_run_sim_load_formula_legacy_unchanged(self, tmp_path):
        from ashare_trading.run_sim import SimulationRunner

        # Minimal pre-contract payload: no schema version, resolvable formula.
        (tmp_path / "best_ashare_strategy.json").write_text(
            json.dumps({"formula": [1], "formula_text": "RET_1"}),
            encoding="utf-8",
        )
        sim = SimulationRunner.__new__(SimulationRunner)
        sim.data_config = SimpleNamespace(data_dir=tmp_path)
        sim.loader = SimpleNamespace(dataset_id=None)
        sim.load_formula()
        assert sim.formula_tokens == [1]

    def test_on_disk_artifacts_classify_legacy(self):
        """§3 invariant: the repo's current data/ artifacts (pre-contract)
        keep the legacy read path byte-identically."""
        from pathlib import Path

        strategy = Path("data/best_ashare_strategy.json")
        protocol = Path("data/protocol_result.json")
        if strategy.exists():
            payload = json.loads(strategy.read_text(encoding="utf-8"))
            assert classify_schema_version(payload) == "legacy"
        if protocol.exists():
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            assert classify_schema_version(payload) == "legacy"


# --- C2: backtest result + paper state --------------------------------------


def _backtest_payload() -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "spec_id": "a" * 64,
        "run_id": "b" * 32,
        "candidate_id": "c" * 64,
        "formula": [4, 45, 3],
        "formula_text": "(VOL_20 CORR60 ATR_14)",
        "direction": -1,
        "dataset_id": "abc123",
        "metrics": {"total_return": 0.1, "sharpe": 1.2},
        "dates": ["2024-01-02", "2024-01-03"],
        "equity_curve": [1.0, 1.1],
        "benchmark": "equal_weight",
        "benchmark_equity": [1.0, 1.05],
        "positions": [{"date": "2024-01-03", "weights": {"000001.SZ": 0.5}}],
        "universe_policy": None,
    }


def _paper_payload() -> dict:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "spec_id": "a" * 64,
        "run_id": "b" * 32,
        "candidate_id": "c" * 64,
        "account_id": "d" * 32,
        "initial_capital": 100000.0,
        "cash": 95000.0,
        "trade_count": 1,
        "last_exec_date": "20240105",
        "positions": {
            "000001.SZ": {
                "ts_code": "000001.SZ",
                "name": "PingAn",
                "quantity": 100.0,
                "available_quantity": 100.0,
                "avg_cost": 10.0,
                "last_price": 10.5,
                "last_date": "20240105",
            }
        },
        "equity_history": [{"trade_date": "20240105", "equity": 100500.0}],
    }


class TestBacktestSchema:
    def test_round_trip(self):
        artifact = BacktestResultArtifact.validate_payload(_backtest_payload())
        assert artifact.direction == -1

    def test_missing_dataset_id_key_fails(self):
        payload = _backtest_payload()
        del payload["dataset_id"]
        with pytest.raises(ValidationError):
            BacktestResultArtifact.model_validate(payload)

    def test_unknown_top_level_key_forbidden(self):
        payload = _backtest_payload()
        payload["surprise"] = 1
        with pytest.raises(ValidationError):
            BacktestResultArtifact.model_validate(payload)

    def test_unknown_schema_version_rejected(self):
        payload = _backtest_payload()
        payload["artifact_schema_version"] = 99
        with pytest.raises(ArtifactSchemaError):
            BacktestResultArtifact.validate_payload(payload)


class TestPaperStateSchema:
    def test_round_trip(self):
        artifact = PaperStateArtifact.validate_payload(_paper_payload())
        assert artifact.positions["000001.SZ"].quantity == 100.0

    def test_missing_cash_key_fails(self):
        payload = _paper_payload()
        del payload["cash"]
        with pytest.raises(ValidationError):
            PaperStateArtifact.model_validate(payload)

    def test_unknown_top_level_key_forbidden(self):
        payload = _paper_payload()
        payload["surprise"] = 1
        with pytest.raises(ValidationError):
            PaperStateArtifact.model_validate(payload)

    def test_bad_position_shape_fails(self):
        payload = _paper_payload()
        del payload["positions"]["000001.SZ"]["avg_cost"]
        with pytest.raises(ValidationError):
            PaperStateArtifact.model_validate(payload)


class TestPaperStateIO:
    """§5 fail-closed runtime semantics of SimulationPortfolio."""

    def test_unbound_save_is_rejected_without_creating_state(self, tmp_path):
        from ashare_trading.portfolio import SimulationPortfolio

        path = tmp_path / "state.json"
        portfolio = SimulationPortfolio(100000.0, path)
        with pytest.raises(ArtifactSchemaError, match="not bound"):
            portfolio.save()
        assert not path.exists()

    def test_save_stamps_and_validates(self, tmp_path):
        from ashare_trading.portfolio import SimulationPortfolio
        from tests.conftest import open_test_handle

        portfolio = SimulationPortfolio(100000.0, tmp_path / "state.json")
        with open_test_handle(tmp_path / "store") as handle:
            portfolio.bind_lineage(
                handle, candidate_id="c" * 64, account_id="d" * 32
            )
            portfolio.save()
        payload = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert payload["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
        PaperStateArtifact.validate_payload(payload)

    def test_load_corrupt_file_raises_and_never_overwrites(self, tmp_path):
        from ashare_trading.portfolio import SimulationPortfolio

        path = tmp_path / "state.json"
        path.write_text("{not json", encoding="utf-8")
        before = path.read_bytes()
        with pytest.raises(ArtifactSchemaError):
            SimulationPortfolio(100000.0, path)
        assert path.read_bytes() == before

    def test_load_unknown_version_raises_and_never_overwrites(self, tmp_path):
        from ashare_trading.portfolio import SimulationPortfolio

        path = tmp_path / "state.json"
        payload = _paper_payload()
        payload["artifact_schema_version"] = 99
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = path.read_bytes()
        with pytest.raises(ArtifactSchemaError):
            SimulationPortfolio(100000.0, path)
        assert path.read_bytes() == before

    def test_load_non_dict_raises(self, tmp_path):
        from ashare_trading.portfolio import SimulationPortfolio

        path = tmp_path / "state.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ArtifactSchemaError):
            SimulationPortfolio(100000.0, path)

    @pytest.mark.parametrize("legacy_version", [None, 1])
    def test_load_legacy_is_read_only_and_save_preserves_bytes(
        self, tmp_path, legacy_version
    ):
        """P8 §8: v0/v1 paper state is audit-only and never auto-migrated."""
        from ashare_trading.portfolio import SimulationPortfolio

        path = tmp_path / "state.json"
        legacy = _paper_payload()
        for key in ("spec_id", "run_id", "candidate_id", "account_id"):
            legacy.pop(key)
        if legacy_version is None:
            legacy.pop("artifact_schema_version")
        else:
            legacy["artifact_schema_version"] = legacy_version
        path.write_text(json.dumps(legacy), encoding="utf-8")
        before = path.read_bytes()
        portfolio = SimulationPortfolio(100000.0, path)
        assert portfolio.cash == 95000.0
        assert portfolio.trade_count == 1
        assert portfolio.legacy_read_only
        with pytest.raises(ArtifactSchemaError, match="legacy"):
            portfolio.save()
        assert path.read_bytes() == before

    def test_legacy_reset_requires_explicit_archival_authority(self, tmp_path):
        from ashare_trading.portfolio import SimulationPortfolio

        path = tmp_path / "state.json"
        legacy = _paper_payload()
        for key in (
            "artifact_schema_version",
            "spec_id",
            "run_id",
            "candidate_id",
            "account_id",
        ):
            legacy.pop(key)
        path.write_text(json.dumps(legacy), encoding="utf-8")
        before = path.read_bytes()
        portfolio = SimulationPortfolio(100000.0, path)
        with pytest.raises(ArtifactSchemaError, match="archive/retire"):
            portfolio.reset()
        assert path.read_bytes() == before

    def test_missing_file_is_fresh_state(self, tmp_path):
        from ashare_trading.portfolio import SimulationPortfolio

        portfolio = SimulationPortfolio(100000.0, tmp_path / "state.json")
        assert portfolio.cash == 100000.0
        assert portfolio.trade_count == 0
