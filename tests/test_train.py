from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig
from ashare_model.data_loader import AshareDataLoader
from ashare_model.reward import REWARD_VERSION
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
    assert artifact["reward_version"] == REWARD_VERSION
    # The recorded metadata must be enough to resolve the formula back.
    assert resolve_formula_tokens(artifact) == list(tokens)


def test_train_can_skip_artifacts(populated_db: DataConfig, monkeypatch):
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
    tokens = trainer.train(steps=1, batch_size=1, save_artifacts=False)
    assert tokens is not None
    # Ablation runs must not clobber the working strategy/model artifacts.
    assert not (populated_db.data_dir / "best_ashare_strategy.json").exists()
    assert not (populated_db.data_dir / "ashare_model.pt").exists()


def test_train_end_date_override_truncates_training_window(
    tmp_path, populated_db: DataConfig, monkeypatch
):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
    )
    captured: dict[str, int] = {}

    def fake_execute(tokens, ft):
        captured["cols"] = ft.shape[2]
        return torch.zeros(ft.shape[1], ft.shape[2])

    monkeypatch.setattr(trainer.vm, "execute", fake_execute)
    trainer.train(
        steps=1, batch_size=1, save_artifacts=False, train_end_date="2024-01-10"
    )
    override_end = trainer._train_end_index("2024-01-10")
    assert captured["cols"] == override_end
    # The override lands strictly before the config-driven window.
    assert override_end < trainer._train_end_index()


def test_validation_start_moves_with_overridden_train_end(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    trainer = _make_trainer(populated_db.data_dir, loader)
    early = trainer._train_end_index("2024-01-10")
    late = trainer._train_end_index()
    early_val = trainer._validation_start(early)
    late_val = trainer._validation_start(late)
    assert 1 <= early_val < early
    assert 1 <= late_val < late
    assert early_val <= late_val


def test_train_invalid_formula_gets_clip_low(populated_db: DataConfig, monkeypatch):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
    )
    monkeypatch.setattr(trainer.vm, "execute", lambda tokens, ft: None)
    tokens = trainer.train(steps=1, batch_size=1, save_artifacts=False)
    assert tokens is None
    assert trainer.history[0]["avg_reward"] == trainer.reward_config.reward_clip_low


def test_train_constant_formula_gets_bad_reward(populated_db: DataConfig, monkeypatch):
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
        lambda tokens, ft: torch.zeros(len(loader.ts_codes), train_end),
    )
    tokens = trainer.train(steps=1, batch_size=1, save_artifacts=False)
    # Constant signals are rejected outright and never enter the
    # best-formula selection, so no formula is saved.
    assert tokens is None
    assert trainer.history[0]["avg_reward"] == trainer.reward_config.bad_reward


# --- reward deduplication / cross-step cache --------------------------------


def test_reward_cache_reuses_evaluations_across_steps(
    populated_db: DataConfig, monkeypatch
):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=4, train_steps=2, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
    )
    train_end = trainer._train_end_index()
    calls: dict[str, int] = {"n": 0}

    def counting_execute(tokens, ft):
        calls["n"] += 1
        return torch.arange(
            1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end)

    monkeypatch.setattr(trainer.vm, "execute", counting_execute)
    # Force every sampled token to be feature 1: all four sequences in the
    # batch are the same formula, so only one unique evaluation per step.
    monkeypatch.setattr(
        Categorical,
        "sample",
        lambda self: torch.full(
            (self.logits.shape[0],),
            1,
            dtype=torch.long,
            device=self.logits.device,
        ),
    )
    trainer.train(steps=2, batch_size=4, save_artifacts=False)
    # Step 1 evaluates the unique formula once; step 2 hits the cache and
    # never touches the VM again (invalid formulas are cached too).
    assert calls["n"] == 1


def test_reward_cache_is_bounded_lru(monkeypatch):
    from ashare_model.train import AshareTrainer

    monkeypatch.setattr(AshareTrainer, "_REWARD_CACHE_CAP", 4)
    cache: OrderedDict = OrderedDict()
    monkeypatch.setattr(
        AshareTrainer, "_reward_cache", cache, raising=False
    )
    trainer = object.__new__(AshareTrainer)
    for i in range(6):
        trainer._cache_put((i,), (float(i), None))
    assert len(cache) == 4
    assert list(cache.keys()) == [(2,), (3,), (4,), (5,)]
    # Touching moves the entry to the most-recent end.
    trainer._cache_touch((2,))
    assert list(cache.keys()) == [(3,), (4,), (5,), (2,)]


def test_duplicate_formulas_share_one_batched_evaluation(
    populated_db: DataConfig, monkeypatch
):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=4, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
    )
    train_end = trainer._train_end_index()
    calls: dict[str, int] = {"n": 0}
    add_token = FORMULA_VOCAB.operator_offset  # first operator (ADD)
    pattern = [1, 1, add_token, 0]
    state = {"pos": -1}

    def counting_execute(tokens, ft):
        calls["n"] += 1
        return torch.arange(
            1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end)

    monkeypatch.setattr(trainer.vm, "execute", counting_execute)

    def fixed_sample(self):
        state["pos"] += 1
        token = pattern[state["pos"] % len(pattern)]
        return torch.full(
            (self.logits.shape[0],),
            token,
            dtype=torch.long,
            device=self.logits.device,
        )

    monkeypatch.setattr(Categorical, "sample", fixed_sample)
    trainer.train(steps=1, batch_size=4, save_artifacts=False)
    # Four identical valid formulas: one VM execution, one batched scoring,
    # and the best formula is recorded from the shared evaluation.
    assert calls["n"] == 1
    assert trainer.best_tokens is not None
    assert trainer.best_tokens[:3] == [1, 1, add_token]
    assert trainer.best_reward > -float("inf")
