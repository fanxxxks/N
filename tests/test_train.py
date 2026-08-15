from __future__ import annotations

import numpy as np
import pytest
import torch

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig
from ashare_model.data_loader import AshareDataLoader
from ashare_model.train import AshareTrainer
from ashare_model.vocab import FORMULA_VOCAB


def _make_trainer(tmp_path, loader=None):
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
    )
    model_config = ModelConfig(batch_size=4, train_steps=1, max_formula_len=6)
    return AshareTrainer(
        data_config,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-01-20"),
        loader,
    )


def test_train_end_index_before_data(populated_db: DataConfig):
    from ashare_model.data_loader import AshareDataLoader

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    trainer = _make_trainer(populated_db.data_dir, loader)
    idx = trainer._train_end_index()
    assert 0 < idx <= len(loader.dates)


def test_update_stack_state_feature_and_operator():
    action = torch.tensor([1, FORMULA_VOCAB.operator_offset], dtype=torch.long)
    open_slots = torch.ones(2, dtype=torch.long)
    stack_sizes = torch.zeros(2, dtype=torch.long)
    AshareTrainer._update_stack_state(action, open_slots, stack_sizes)
    assert stack_sizes[0].item() == 1
    assert open_slots[0].item() == 0


def test_fast_reward_smoke(tmp_path):
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
    )
    model_config = ModelConfig()
    loader = AshareDataLoader(data_config, model_config)
    loader.factor_tensor = torch.zeros(1)
    trainer = _make_trainer(tmp_path, loader)
    signal = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    target = np.array([[0.01, -0.01, 0.01, -0.01], [0.02, 0.02, 0.02, 0.02], [0.03, 0.03, 0.03, 0.03]])
    reward, mean = trainer._fast_reward(signal, target, ["A", "B", "C"], ["d1", "d2", "d3", "d4"])
    assert isinstance(reward, float)
    assert isinstance(mean, float)


def test_fast_reward_charges_turnover_costs(tmp_path):
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
    )
    model_config = ModelConfig()
    loader = AshareDataLoader(data_config, model_config)
    loader.factor_tensor = torch.zeros(1)
    trainer = _make_trainer(tmp_path, loader)

    # Identical targets for every stock, so any top-2 basket earns the same
    # gross return and only turnover costs can differentiate the signals.
    target = np.full((3, 4), 0.002)
    static_signal = np.array(
        [[3.0, 3.0, 3.0, 3.0], [2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]
    )
    _, mean_static = trainer._fast_reward(
        static_signal, target, ["A", "B", "C"], ["d1", "d2", "d3", "d4"]
    )
    # The basket flips completely on days 1 and 2 -> 2 full turnovers over 3
    # daily returns.
    churn_signal = np.array(
        [[3.0, 1.0, 3.0, 1.0], [2.0, 2.0, 2.0, 2.0], [1.0, 3.0, 1.0, 3.0]]
    )
    _, mean_churn = trainer._fast_reward(
        churn_signal, target, ["A", "B", "C"], ["d1", "d2", "d3", "d4"]
    )
    cost_per_turnover = (
        2.0 * trainer.backtest_config.commission_rate
        + 2.0 * trainer.backtest_config.slippage_rate
        + 2.0 * trainer.backtest_config.transfer_fee_rate
        + trainer.backtest_config.stamp_tax_rate
    )
    assert cost_per_turnover == pytest.approx(0.00202)
    assert mean_static == pytest.approx(0.002)
    assert mean_static - mean_churn == pytest.approx(
        (2.0 / 3.0) * cost_per_turnover, abs=1e-9
    )


def test_fast_reward_survives_nonfinite_targets(tmp_path):
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
    )
    loader = AshareDataLoader(data_config, ModelConfig())
    loader.factor_tensor = torch.zeros(1)
    trainer = _make_trainer(tmp_path, loader)
    signal = np.array([[3.0] * 4, [2.0] * 4, [1.0] * 4])
    target = np.array(
        [[0.01, np.inf, 0.01, np.nan], [0.02, 0.02, 0.02, 0.02], [0.03, 0.03, 0.03, 0.03]]
    )
    reward, mean = trainer._fast_reward(signal, target, ["A", "B", "C"], ["d1", "d2", "d3", "d4"])
    assert np.isfinite(reward)
    assert np.isfinite(mean)


def test_validation_split_bounds(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    trainer = _make_trainer(populated_db.data_dir, loader)
    train_end = trainer._train_end_index()
    val_start = trainer._validation_start(train_end)
    assert 1 <= val_start < train_end


def test_best_formula_selected_on_validation_window(tmp_path, populated_db: DataConfig, monkeypatch):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
    )
    train_end = trainer._train_end_index()
    val_start = trainer._validation_start(train_end)
    target = loader.target_ret[:, :train_end].numpy()
    # Crafted signal: actively bad in-sample, perfectly aligned in the
    # validation tail.  Only a validation-driven selection can score 5.0.
    crafted = np.zeros((len(loader.ts_codes), train_end))
    crafted[:, :val_start] = -target[:, :val_start] * 100.0
    crafted[:, val_start:] = target[:, val_start:] * 100.0
    crafted_t = torch.tensor(crafted, dtype=torch.float32)
    monkeypatch.setattr(trainer.vm, "execute", lambda tokens, ft: crafted_t)

    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is not None
    assert trainer.best_reward == pytest.approx(5.0, abs=1e-6)
    assert "value_loss" in trainer.history[0]


def test_train_artifact_records_vocab_provenance(tmp_path, populated_db: DataConfig, monkeypatch):
    import json

    from ashare_model.vocab import resolve_formula_tokens

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
    )
    train_end = trainer._train_end_index()
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            len(loader.ts_codes) * train_end, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end),
    )
    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is not None

    artifact = json.loads(
        (populated_db.data_dir / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
    assert artifact["feature_names"] == list(FORMULA_VOCAB.feature_names)
    assert artifact["operator_names"] == list(FORMULA_VOCAB.operator_names)
    assert artifact["feature_version"] == FORMULA_VOCAB.feature_version
    # The recorded metadata must be enough to resolve the formula back.
    assert resolve_formula_tokens(artifact) == list(tokens)
