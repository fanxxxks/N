from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig, RewardConfig
from ashare_model.data_loader import AshareDataLoader
from ashare_model.reward import REWARD_VERSION
from ashare_model.train import AshareTrainer, resolve_device
from ashare_model.vocab import FORMULA_VOCAB


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device() == torch.device("cuda")
    assert resolve_device("auto") == torch.device("cuda")
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == torch.device("cpu")
    assert resolve_device("auto") == torch.device("cpu")


def test_resolve_device_rejects_unknown_values():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def _reward_cfg(**kwargs):
    # The 3-stock fixtures fall below the production ic_min_stocks; the
    # floors are relaxed so the fixtures exercise the training loop itself.
    defaults = dict(ic_min_stocks=3, min_val_reward=-1e9, min_val_icir=-1e9)
    defaults.update(kwargs)
    return RewardConfig(**defaults)


def _make_trainer(tmp_path, loader=None, reward_config=None):
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
        reward_config=reward_config or _reward_cfg(),
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
    windows = trainer._validation_windows(train_end)
    assert windows and windows[0][0] == val_start and windows[-1][1] == train_end
    assert all(b > a for a, b in windows)
    assert all(b - a >= 3 for a, b in windows)


def test_validation_windows_cover_tail_disjointly():
    trainer = object.__new__(AshareTrainer)
    trainer.model_config = ModelConfig(validation_fraction=0.2, validation_splits=3)
    trainer.reward_config = _reward_cfg()
    train_end = 100
    windows = trainer._validation_windows(train_end)
    assert len(windows) == 3
    starts = [a for a, _ in windows]
    ends = [b for _, b in windows]
    assert starts[0] == trainer._validation_start(train_end)
    assert ends[-1] == train_end
    assert ends[:-1] == starts[1:]  # disjoint and contiguous


def test_validation_windows_degrade_to_single_window_when_too_short():
    trainer = object.__new__(AshareTrainer)
    trainer.model_config = ModelConfig(validation_fraction=0.5, validation_splits=3)
    trainer.reward_config = _reward_cfg()
    train_end = 10
    windows = trainer._validation_windows(train_end)
    assert windows == [(trainer._validation_start(train_end), train_end)]


def test_best_formula_selected_on_validation_window(tmp_path, populated_db: DataConfig, monkeypatch):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    reward_config = _reward_cfg()
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=reward_config,
    )
    train_end = trainer._train_end_index()
    val_start = trainer._validation_start(train_end)
    target = loader.target_ret[:, :train_end].numpy()
    # Crafted signal: actively bad in-sample, perfectly aligned in the
    # validation tail.  Only a validation-driven selection can score the
    # clip ceiling.
    crafted = np.zeros((len(loader.ts_codes), train_end))
    crafted[:, :val_start] = -target[:, :val_start] * 100.0
    crafted[:, val_start:] = target[:, val_start:] * 100.0
    crafted_t = torch.tensor(crafted, dtype=torch.float32)
    monkeypatch.setattr(trainer.vm, "execute", lambda tokens, ft: crafted_t)
    # Force an operator-bearing formula so the bare-copy penalty does not
    # apply: the validation reward then clips at exactly the ceiling.
    add_token = FORMULA_VOCAB.operator_offset
    pattern = [1, 1, add_token, 0]
    state = {"pos": -1}

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
    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is not None
    assert trainer.best_reward == pytest.approx(
        trainer.reward_config.reward_clip_high, abs=1e-6
    )
    assert "value_loss" in trainer.history[0]
    assert "entropy" in trainer.history[0]


def test_bare_factor_penalty_applied_but_operator_formula_not(
    tmp_path, populated_db: DataConfig, monkeypatch
):
    """A bare single-feature formula pays complexity_penalty on both the
    gradient reward and the validation selection; an operator-bearing
    formula does not."""

    import ashare_model.train as train_module

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    reward_config = _reward_cfg(complexity_penalty=0.25)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=reward_config,
    )
    train_end = trainer._train_end_index()
    signal = torch.arange(
        1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
    ).reshape(len(loader.ts_codes), train_end)
    monkeypatch.setattr(trainer.vm, "execute", lambda tokens, ft: signal)
    monkeypatch.setattr(
        train_module,
        "batched_basket_rewards",
        lambda signals, target, bt, rc, val_windows, **kwargs: (
            np.full(signals.shape[0], 0.5),
            np.full(signals.shape[0], 0.4),
            np.full(signals.shape[0], 0.45),
            np.full(signals.shape[0], 0.35),
        ),
    )
    add_token = FORMULA_VOCAB.operator_offset
    row_a = [1, 0, 0, 0]
    row_b = [1, 1, add_token, 0]
    state = {"pos": -1}

    def fixed_sample(self):
        state["pos"] += 1
        pos = state["pos"] % 4
        return torch.tensor(
            [row_a[pos], row_b[pos]], dtype=torch.long, device=self.logits.device
        )

    monkeypatch.setattr(Categorical, "sample", fixed_sample)
    trainer.train(steps=1, batch_size=2, save_artifacts=False)
    bare = (1, 0, 0, 0)
    combined = (1, 1, add_token, 0)
    assert trainer._reward_cache[bare] == pytest.approx((0.25, 0.15, 0.45, 0.35))
    assert trainer._reward_cache[combined] == pytest.approx((0.5, 0.4, 0.45, 0.35))
    # The selection prefers the unpenalized operator formula.
    assert trainer.best_tokens == list(combined)
    assert trainer.best_reward == pytest.approx(0.4)
    assert trainer.best_icir == pytest.approx(0.45)


def test_quality_floor_blocks_save_and_returns_none(
    tmp_path, populated_db: DataConfig, monkeypatch
):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    reward_config = _reward_cfg(min_val_reward=0.0)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=reward_config,
    )
    train_end = trainer._train_end_index()
    # Perfectly *inverted* alignment: negative ICIR everywhere, so the best
    # validation reward stays below the quality floor.
    target = loader.target_ret[:, :train_end].numpy()
    inverted = torch.tensor(-target * 100.0, dtype=torch.float32)
    monkeypatch.setattr(trainer.vm, "execute", lambda tokens, ft: inverted)
    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is None
    assert trainer.best_reward < 0.0
    assert not (populated_db.data_dir / "best_ashare_strategy.json").exists()


def test_policy_update_loss_clips_advantage_and_applies_entropy():
    log_probs = [torch.tensor([-0.5, -0.7]), torch.tensor([-0.2, -0.3])]
    rewards = torch.tensor([1.0, 1.0])  # zero spread -> unbounded advantage
    baseline = torch.tensor([0.0, 0.0])
    entropies = [torch.tensor([1.5, 1.5]), torch.tensor([1.5, 1.5])]
    loss, policy_loss, value_loss, entropy = AshareTrainer._policy_update_loss(
        log_probs,
        rewards,
        baseline,
        entropies,
        value_loss_weight=0.5,
        advantage_clip=10.0,
        entropy_coef=0.01,
    )
    # Zero spread must not explode: the clipped advantage is bounded.
    assert torch.isfinite(policy_loss) and abs(float(policy_loss)) <= 10.0
    assert entropy == pytest.approx(1.5, abs=1e-6)
    assert torch.isfinite(value_loss)
    # The entropy bonus shifts the total loss below the plain actor-critic
    # sum (loss = policy + w * value - coef * entropy).
    assert loss == pytest.approx(
        float(policy_loss + 0.5 * value_loss - 0.01 * entropy), abs=1e-6
    )
    # Zero spread drives the raw advantage to ~1e6, so the loss is exactly
    # the clipped value: sum(log_probs) = [-0.7, -1.0], adv = +10 both rows.
    assert policy_loss == pytest.approx(8.5, abs=1e-6)
    _, policy_loss_5, _, _ = AshareTrainer._policy_update_loss(
        log_probs, rewards, baseline, entropies,
        value_loss_weight=0.5, advantage_clip=5.0, entropy_coef=0.01,
    )
    assert policy_loss_5 == pytest.approx(4.25, abs=1e-6)


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
        reward_config=_reward_cfg(),
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


def test_train_artifact_records_device(tmp_path, populated_db: DataConfig, monkeypatch):
    import json

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=_reward_cfg(),
    )
    train_end = trainer._train_end_index()
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            len(loader.ts_codes) * train_end, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end),
    )
    tokens = trainer.train(steps=1, batch_size=1, device="cpu")
    assert tokens is not None
    artifact = json.loads(
        (populated_db.data_dir / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
    assert artifact["device"] == "cpu"
    assert artifact["init_seed"] == 42


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_train_one_step_on_cuda(populated_db: DataConfig, monkeypatch):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=4, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=_reward_cfg(),
    )
    train_end = trainer._train_end_index()
    # The VM must receive the CUDA-resident factor tensor; signals come
    # back over the .cpu() bridge into the numpy reward path while the
    # policy itself stays on CPU.
    captured: dict[str, object] = {}

    def execute_on(tokens, ft):
        captured["vm_device"] = ft.device.type
        return (
            torch.arange(
                1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
            )
            .reshape(len(loader.ts_codes), train_end)
            .to(ft.device)
        )

    monkeypatch.setattr(trainer.vm, "execute", execute_on)
    tokens = trainer.train(steps=1, batch_size=4, device="cuda", save_artifacts=False)
    assert tokens is not None
    assert captured["vm_device"] == "cuda"
    assert next(trainer.model.parameters()).device.type == "cpu"
    assert np.isfinite(trainer.history[0]["avg_reward"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_same_seed_samples_same_formulas_on_cpu_and_cuda(
    populated_db: DataConfig, monkeypatch
):
    # The policy and sampling stay on CPU regardless of the VM device, so
    # the same (init_seed, seed) must produce the identical sampled formula
    # sequence on CPU and CUDA — the cross-device reproducibility contract.
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_cfg = ModelConfig(batch_size=8, train_steps=1, max_formula_len=6)
    bt = BacktestConfig(top_n=2, train_end_date="2024-02-01")
    sequences: dict[str, list[tuple[int, ...]]] = {"cpu": [], "cuda": []}

    def recorder(key: str):
        def execute(tokens, ft):
            sequences[key].append(tuple(tokens))
            return torch.zeros(ft.shape[1], ft.shape[2], device=ft.device)

        return execute

    cpu_trainer = AshareTrainer(populated_db, model_cfg, bt, loader, init_seed=7)
    cuda_trainer = AshareTrainer(populated_db, model_cfg, bt, loader, init_seed=7)
    monkeypatch.setattr(cpu_trainer.vm, "execute", recorder("cpu"))
    monkeypatch.setattr(cuda_trainer.vm, "execute", recorder("cuda"))
    cpu_trainer.train(steps=1, batch_size=8, seed=11, device="cpu", save_artifacts=False)
    cuda_trainer.train(steps=1, batch_size=8, seed=11, device="cuda", save_artifacts=False)
    assert sequences["cpu"]
    assert sequences["cpu"] == sequences["cuda"]


def test_train_can_skip_artifacts(populated_db: DataConfig, monkeypatch):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=_reward_cfg(),
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
        reward_config=_reward_cfg(),
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
        reward_config=_reward_cfg(),
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
    # never touches the VM again (invalid formulas are cached too).  The
    # second VM call overall is the single end-of-training execution that
    # computes the best formula's learned trade direction.
    assert calls["n"] == 2


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
        reward_config=_reward_cfg(),
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
    # and the best formula is recorded from the shared evaluation; the
    # second VM call is the single end-of-training direction computation.
    assert calls["n"] == 2
    assert trainer.best_tokens is not None
    assert trainer.best_tokens[:3] == [1, 1, add_token]
    assert trainer.best_reward > -float("inf")


# --- streaming reward scoring (memory-bounded pending buffer) ----------------


def test_reward_chunk_size_bounds():
    f = AshareTrainer._reward_chunk_size
    # Tiny windows hit the 64 cap; huge signals floor at 1 so a single
    # oversized signal still makes progress.
    assert f(0) == 64
    assert f(1024) == 64
    assert f(1 << 30) == 1
    # Monotone: larger signals never allow a larger chunk.
    sizes = [f(sb) for sb in (1, 2**10, 2**16, 2**20, 2**24, 2**28, 2**30)]
    assert sizes == sorted(sizes, reverse=True)


def test_streamed_reward_chunks_match_single_pass(
    populated_db: DataConfig, monkeypatch
):
    """Scoring pending formulas in small chunks yields the same training
    outcome as scoring them in one pass: streaming is a memory optimization,
    not a semantic change."""

    import ashare_model.train as train_module

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=64, train_steps=1, max_formula_len=4)
    # Capture the real function once: inside run_trainer the module attribute
    # may already be patched by the previous run.
    real_batched = train_module.batched_basket_rewards

    def fake_execute(tokens, ft):
        # Deterministic non-constant signal per formula, shaped [stocks, dates].
        key = tuple(int(t) for t in tokens)
        seed = sum((i + 1) * t for i, t in enumerate(key)) % 1000 + 1
        rng = np.random.default_rng(seed)
        return torch.tensor(
            rng.normal(size=(ft.shape[1], ft.shape[2])).astype(np.float32)
        )

    def run_trainer(chunk_size: int, calls: list[int]) -> AshareTrainer:
        monkeypatch.setattr(
            AshareTrainer,
            "_reward_chunk_size",
            staticmethod(lambda signal_bytes: chunk_size),
        )
        trainer = AshareTrainer(
            populated_db,
            model_config,
            BacktestConfig(top_n=2, train_end_date="2024-02-01"),
            loader,
            reward_config=_reward_cfg(),
        )
        monkeypatch.setattr(trainer.vm, "execute", fake_execute)

        def counting(signals, target_ret, bt_cfg, reward_cfg, val_windows, **kwargs):
            calls.append(int(signals.shape[0]))
            return real_batched(
                signals, target_ret, bt_cfg, reward_cfg, val_windows, **kwargs
            )

        monkeypatch.setattr(train_module, "batched_basket_rewards", counting)
        trainer.train(steps=1, batch_size=64, save_artifacts=False)
        return trainer

    calls_streamed: list[int] = []
    calls_single: list[int] = []
    streamed = run_trainer(8, calls_streamed)
    single = run_trainer(128, calls_single)

    # The streamed path splits the exact same formula set into chunks of at
    # most 8; the single-pass path scores it in exactly one call.
    assert calls_streamed and all(c <= 8 for c in calls_streamed)
    assert len(calls_single) == 1
    assert sum(calls_streamed) == calls_single[0]

    # Identical sampling (init_seed/seed pinned) and identical scoring:
    # both paths must agree on every training observable.
    assert streamed.best_tokens == single.best_tokens
    assert streamed.best_reward == pytest.approx(single.best_reward)
    assert streamed.history[0]["avg_reward"] == pytest.approx(
        single.history[0]["avg_reward"]
    )
    assert streamed.history[0]["loss"] == pytest.approx(
        single.history[0]["loss"]
    )


# --- v4: signal-quality gate, direction, collapse monitoring, sampler -------


def test_icir_gate_blocks_weak_signal(tmp_path, populated_db: DataConfig, monkeypatch):
    import ashare_model.train as train_module

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    reward_config = _reward_cfg(min_val_reward=-1e9, min_val_icir=0.05)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=reward_config,
    )
    train_end = trainer._train_end_index()
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            len(loader.ts_codes) * train_end, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end),
    )
    # The scorer reports a strong cost-adjusted reward but a sub-threshold
    # ICIR: the quality gate must block the save even though the
    # cost-adjusted floor passes.
    monkeypatch.setattr(
        train_module,
        "batched_basket_rewards",
        lambda signals, target, bt, rc, val_windows, **kwargs: (
            np.full(signals.shape[0], 0.8),
            np.full(signals.shape[0], 0.8),
            np.full(signals.shape[0], 0.01),
            np.full(signals.shape[0], 0.01),
        ),
    )
    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is None
    # The bare-copy penalty (0.02 by default config) shaves the reported
    # cost-adjusted reward; the raw ICIR stays below the gate.
    assert trainer.best_reward == pytest.approx(0.78)
    assert trainer.best_icir == pytest.approx(0.01)
    assert not (populated_db.data_dir / "best_ashare_strategy.json").exists()


def test_icir_gate_passes_strong_signal(tmp_path, populated_db: DataConfig, monkeypatch):
    import ashare_model.train as train_module

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    reward_config = _reward_cfg(min_val_reward=0.0, min_val_icir=0.05)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=reward_config,
    )
    train_end = trainer._train_end_index()
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            len(loader.ts_codes) * train_end, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end),
    )
    monkeypatch.setattr(
        train_module,
        "batched_basket_rewards",
        lambda signals, target, bt, rc, val_windows, **kwargs: (
            np.full(signals.shape[0], 0.3),
            np.full(signals.shape[0], 0.3),
            np.full(signals.shape[0], 0.4),
            np.full(signals.shape[0], 0.4),
        ),
    )
    tokens = trainer.train(steps=1, batch_size=1, save_artifacts=False)
    assert tokens is not None
    assert trainer.best_icir == pytest.approx(0.4)
    assert trainer.best_direction in (-1, 1)


def test_artifact_records_direction_and_icir(tmp_path, populated_db: DataConfig, monkeypatch):
    import json

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=1, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=_reward_cfg(),
    )
    train_end = trainer._train_end_index()
    # Positive ramp signal with a positively correlated target: the IC is
    # positive, so the learned direction must be +1 and the artifact must
    # record it together with the quality-gate ICIR.
    signal = torch.arange(
        1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
    ).reshape(len(loader.ts_codes), train_end)
    monkeypatch.setattr(trainer.vm, "execute", lambda tokens, ft: signal)
    monkeypatch.setattr(
        loader, "target_ret", (signal / 100.0).to(loader.target_ret.dtype)
    )
    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is not None
    artifact = json.loads(
        (populated_db.data_dir / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
    assert artifact["direction"] == trainer.best_direction
    assert artifact["best_icir"] == pytest.approx(trainer.best_icir)


def test_collapse_warning_fires_after_consecutive_collapsed_steps(
    tmp_path, populated_db: DataConfig, monkeypatch
):
    import ashare_model.train as train_module

    warnings: list[str] = []
    monkeypatch.setattr(
        train_module.logger, "warning", lambda msg: warnings.append(msg)
    )
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(
        batch_size=4,
        train_steps=1,
        max_formula_len=4,
        collapse_warn_fraction=0.9,
        collapse_warn_steps=2,
    )
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=_reward_cfg(),
    )
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            len(loader.ts_codes) * trainer._train_end_index(),
            dtype=torch.float32,
        ).reshape(len(loader.ts_codes), trainer._train_end_index()),
    )
    # All rows sample the same feature: unique fraction 1/4 < 0.9.
    monkeypatch.setattr(
        Categorical,
        "sample",
        lambda self: torch.full(
            (self.logits.shape[0],), 1, dtype=torch.long, device=self.logits.device
        ),
    )
    first = trainer.train(steps=1, batch_size=4, save_artifacts=False)
    assert first is not None
    assert not warnings  # first collapsed step: below the streak threshold
    second = trainer.train(steps=1, batch_size=4, save_artifacts=False)
    assert second is not None
    assert trainer.history[-1]["unique_frac"] == pytest.approx(0.25)
    # The second consecutive collapsed step crosses the threshold.
    assert len(warnings) == 1
    assert "policy sampling collapsed" in warnings[0]


def test_sample_random_formulas_are_valid_and_deterministic():
    from ashare_model.train import sample_random_formulas
    from ashare_model.vm import StackVM

    n, max_len = 200, 12
    first = sample_random_formulas(7, FORMULA_VOCAB, max_len, n)
    second = sample_random_formulas(7, FORMULA_VOCAB, max_len, n)
    assert first == second
    assert len(first) == n
    vm = StackVM(FORMULA_VOCAB)
    factor_tensor = torch.zeros((FORMULA_VOCAB.feature_count, 5, 8))
    for tokens in first:
        assert len(tokens) == max_len
        # Every sampled sequence is a structurally valid postfix formula:
        # the VM executes it into exactly one result tensor.
        assert vm.execute(list(tokens), factor_tensor) is not None
    # Different seeds sample different formula sets.
    assert sample_random_formulas(8, FORMULA_VOCAB, max_len, n) != first


def test_validation_windows_new_defaults_split_longer_tail():
    from ashare_model.train import validation_start, validation_windows

    train_end = 1000
    assert validation_start(train_end, ModelConfig()) == 650
    windows = validation_windows(train_end, ModelConfig())
    assert len(windows) == 4
    assert windows[0][0] == 650 and windows[-1][1] == 1000
    assert windows[0][1] == windows[1][0]  # disjoint and contiguous
    # Short tails still degrade to a single window.
    assert validation_windows(10, ModelConfig()) == [
        (validation_start(10, ModelConfig()), 10)
    ]
