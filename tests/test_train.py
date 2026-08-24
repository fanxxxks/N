from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig, RewardConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.data_loader import AshareDataLoader
from ashare_model.reward import REWARD_VERSION
from ashare_model.train import (
    AshareTrainer,
    resolve_device,
    validation_start,
)
from ashare_model.vocab import FORMULA_VOCAB, GRAMMAR_VERSION


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
    idx = trainer._training_contract().train_label_end
    assert 0 < idx <= len(loader.dates)


def test_update_stack_state_feature_and_operator():
    action = torch.tensor([1, FORMULA_VOCAB.operator_offset], dtype=torch.long)
    stack_sizes = torch.zeros(2, dtype=torch.long)
    done = torch.zeros(2, dtype=torch.bool)
    AshareTrainer._update_stack_state(action, stack_sizes, done)
    assert stack_sizes[0].item() == 1
    assert done[0].item() is False


def test_update_stack_state_eos_terminates_and_pad_latches_done():
    eos = FORMULA_VOCAB.eos_token_id
    action = torch.tensor([1, eos], dtype=torch.long)
    # Row 0 samples a feature (stack 0 -> 1); row 1 samples EOS on an
    # already-complete stack.
    stack_sizes = torch.tensor([0, 1], dtype=torch.long)
    done = torch.zeros(2, dtype=torch.bool)
    AshareTrainer._update_stack_state(action, stack_sizes, done)
    assert stack_sizes[0].item() == 1
    assert done[1].item() is True  # row 1 sampled EOS
    assert done[0].item() is False  # row 0 sampled a feature
    # PAD after EOS keeps the done latch and never touches the stack.
    AshareTrainer._update_stack_state(
        torch.tensor([0, 0], dtype=torch.long), stack_sizes, done
    )
    assert done.all() and (stack_sizes == 1).all()


def test_validation_split_bounds(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    trainer = _make_trainer(populated_db.data_dir, loader)
    train_end = trainer._training_contract().train_label_end
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
    train_end = trainer._training_contract().train_label_end
    train_signal_end = trainer._training_contract().train_signal_end
    val_start = trainer._validation_start(train_signal_end)
    target = open_to_open_returns(
        loader.raw_data_cache["open"][:, :train_end].numpy()
    )
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
    pattern = [1, 1, add_token, FORMULA_VOCAB.eos_token_id]
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
    train_end = trainer._training_contract().train_label_end
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
    bare_score = trainer._reward_cache[bare]
    combined_score = trainer._reward_cache[combined]
    assert bare_score.train_reward == pytest.approx(0.25)
    assert bare_score.val_reward == pytest.approx(0.15)
    assert bare_score.train_icir == pytest.approx(0.45)
    assert bare_score.val_icir == pytest.approx(0.35)
    assert combined_score.train_reward == pytest.approx(0.5)
    assert combined_score.val_reward == pytest.approx(0.4)
    assert combined_score.train_icir == pytest.approx(0.45)
    assert combined_score.val_icir == pytest.approx(0.35)
    # The selection prefers the unpenalized operator formula.
    assert trainer.best_tokens == list(combined)
    assert trainer.best_val_reward == pytest.approx(0.4)
    assert trainer.best_val_icir == pytest.approx(0.35)
    assert trainer.best_train_reward == pytest.approx(0.5)
    assert trainer.best_train_icir == pytest.approx(0.45)


def test_trainer_passes_universe_mask_to_scorer(
    tmp_path, populated_db: DataConfig, monkeypatch
):
    """The RL loop must hand the PIT universe mask to the candidate scorer
    (sliced to the exact training window), not rely on NaN VM outputs."""

    import ashare_model.train as train_module

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
    train_end = trainer._training_contract().train_label_end
    captured: dict[str, object] = {}

    def fake_rewards(signals, target, bt, rc, val_windows, **kwargs):
        captured["universe_mask"] = kwargs.get("universe_mask")
        n = signals.shape[0]
        return (
            np.full(n, 0.5),
            np.full(n, 0.4),
            np.full(n, 0.45),
            np.full(n, 0.35),
        )

    monkeypatch.setattr(train_module, "batched_basket_rewards", fake_rewards)
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end),
    )

    add_token = FORMULA_VOCAB.operator_offset
    # A legal operator-bearing sequence under the EOS grammar: two features,
    # ADD, EOS.  The mask makes each of these tokens legal in turn.
    pattern = [1, 1, add_token, FORMULA_VOCAB.eos_token_id]
    state = {"pos": -1}

    def fixed_sample(self):
        state["pos"] += 1
        return torch.full(
            (self.logits.shape[0],),
            pattern[state["pos"] % len(pattern)],
            dtype=torch.long,
            device=self.logits.device,
        )

    monkeypatch.setattr(Categorical, "sample", fixed_sample)
    trainer.train(steps=1, batch_size=1, save_artifacts=False)
    mask = captured.get("universe_mask")
    assert mask is not None
    assert mask.shape == (len(loader.ts_codes), train_end)
    assert np.array_equal(mask, loader.universe_mask[:, :train_end])


def test_policy_gradient_reward_window_excludes_validation_tail(
    populated_db: DataConfig, monkeypatch
):
    """M2 contract: the reward window feeding the policy gradient stops
    where the validation tail begins.

    The validation tail is the *selection* data (it decides the best
    formula); it must not also be the *learning* signal, or the policy is
    trained toward whatever overfits the tail.  A spy around the batched
    reward path records the primary scoring window of every call the
    trainer makes and asserts its end never crosses the first validation
    column, while the window itself stays non-degenerate.
    """
    import ashare_model.train as train_module

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    trainer = AshareTrainer(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        loader,
        reward_config=_reward_cfg(),
    )
    contract = trainer._training_contract()
    val_start = validation_start(contract.train_signal_end, model_config)

    real = train_module.batched_basket_rewards
    seen: list[tuple[int, int] | None] = []

    def spy(signals, target, bt, rc, val_windows, **kwargs):
        seen.append(kwargs.get("train_signal_range"))
        return real(signals, target, bt, rc, val_windows, **kwargs)

    monkeypatch.setattr(train_module, "batched_basket_rewards", spy)
    # Deterministic legal sampling (the seed-dependent real sampling could
    # draw a constant formula and never reach the reward path); the signal
    # is monotone so the candidate is never rejected as near-constant.
    add_token = FORMULA_VOCAB.operator_offset
    pattern = [1, 1, add_token, FORMULA_VOCAB.eos_token_id]
    state = {"pos": -1}

    def fixed_sample(self):
        state["pos"] += 1
        return torch.full(
            (self.logits.shape[0],),
            pattern[state["pos"] % len(pattern)],
            dtype=torch.long,
            device=self.logits.device,
        )

    monkeypatch.setattr(Categorical, "sample", fixed_sample)
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            1.0, len(loader.ts_codes) * contract.train_label_end + 1.0,
            dtype=torch.float32,
        ).reshape(len(loader.ts_codes), contract.train_label_end),
    )
    trainer.train(steps=1, batch_size=2, seed=42, save_artifacts=False)
    assert seen, "the trainer never invoked the reward path"
    assert all(rng is not None for rng in seen)
    assert all(start < end <= val_start for (start, end) in seen)


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
    train_end = trainer._training_contract().train_label_end
    # Both orientations score below the validation floor. Direction symmetry
    # must not allow an ineligible candidate to become the saved best.
    signal = torch.arange(
        1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
    ).reshape(len(loader.ts_codes), train_end)
    monkeypatch.setattr(trainer.vm, "execute", lambda tokens, ft: signal)
    import ashare_model.train as train_module
    monkeypatch.setattr(
        train_module,
        "batched_basket_rewards",
        lambda signals, target, bt, rc, val_windows, **kwargs: (
            np.full(signals.shape[0], -0.5),
            np.full(signals.shape[0], -0.5),
            np.full(signals.shape[0], 0.2),
            np.full(signals.shape[0], 0.2),
        ),
    )
    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is None
    assert trainer.best_val_reward == -float("inf")
    assert trainer.selection_result.best_rejected is not None
    assert trainer.selection_result.best_rejected.val_reward < 0.0
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
    train_end = trainer._training_contract().train_label_end
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
    assert artifact["grammar_version"] == GRAMMAR_VERSION
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
    train_end = trainer._training_contract().train_label_end
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
    train_end = trainer._training_contract().train_label_end
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
    train_end = trainer._training_contract().train_label_end
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
    override_end = trainer._training_contract("2024-01-10").train_label_end
    assert captured["cols"] == override_end
    # The override lands strictly before the config-driven window.
    assert override_end < trainer._training_contract().train_label_end


def test_validation_start_moves_with_overridden_train_end(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    trainer = _make_trainer(populated_db.data_dir, loader)
    early = trainer._training_contract("2024-01-10").train_label_end
    late = trainer._training_contract().train_label_end
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
    train_end = trainer._training_contract().train_label_end
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
    train_end = trainer._training_contract().train_label_end
    calls: dict[str, int] = {"n": 0}

    def counting_execute(tokens, ft):
        calls["n"] += 1
        return torch.arange(
            1.0, len(loader.ts_codes) * train_end + 1.0, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end)

    monkeypatch.setattr(trainer.vm, "execute", counting_execute)
    # Force one legal formula for every row (two features + ADD + EOS): all
    # four sequences in the batch are the same formula, so only one unique
    # evaluation per step.
    add_token = FORMULA_VOCAB.operator_offset
    pattern = [1, 1, add_token, FORMULA_VOCAB.eos_token_id]
    state = {"pos": -1}

    def fixed_sample(self):
        state["pos"] += 1
        return torch.full(
            (self.logits.shape[0],),
            pattern[state["pos"] % len(pattern)],
            dtype=torch.long,
            device=self.logits.device,
        )

    monkeypatch.setattr(Categorical, "sample", fixed_sample)
    trainer.train(steps=2, batch_size=4, save_artifacts=False)
    # Step 1 evaluates the unique formula once; step 2 hits the cache and
    # never touches the VM again (invalid formulas are cached too).  The
    # Direction is scored in the same shared +/- batch, so no late VM
    # execution is needed after selection.
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
        reward_config=_reward_cfg(),
    )
    train_end = trainer._training_contract().train_label_end
    calls: dict[str, int] = {"n": 0}
    add_token = FORMULA_VOCAB.operator_offset  # first operator (ADD)
    pattern = [1, 1, add_token, FORMULA_VOCAB.eos_token_id]
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
    # Direction is selected by the same batched evaluation.
    assert calls["n"] == 1
    assert trainer.best_tokens is not None
    assert trainer.best_tokens[:3] == [1, 1, add_token]
    assert trainer.best_reward > -float("inf")


# --- streaming reward scoring (memory-bounded pending buffer) ----------------


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
            train_module,
            "score_chunk_size",
            lambda signal_bytes: chunk_size,
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
    # Each pending candidate contributes both +signal and -signal to the
    # shared direction-symmetric scoring call.
    assert calls_streamed and all(c <= 2 * 8 for c in calls_streamed)
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
    train_end = trainer._training_contract().train_label_end
    monkeypatch.setattr(
        trainer.vm,
        "execute",
        lambda tokens, ft: torch.arange(
            len(loader.ts_codes) * train_end, dtype=torch.float32
        ).reshape(len(loader.ts_codes), train_end),
    )
    # The scorer reports a strong cost-adjusted reward but a sub-threshold
    # ICIR: the quality gate must block the save even though the
    # cost-adjusted floor passes.  A bare feature formula is forced so the
    # bare-copy penalty (0.02) deterministically shaves the reward.
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
    state = {"pos": -1}

    def fixed_sample(self):
        state["pos"] += 1
        pattern = [1, FORMULA_VOCAB.eos_token_id, 0, 0]
        return torch.full(
            (self.logits.shape[0],),
            pattern[state["pos"] % len(pattern)],
            dtype=torch.long,
            device=self.logits.device,
        )

    monkeypatch.setattr(Categorical, "sample", fixed_sample)
    tokens = trainer.train(steps=1, batch_size=1)
    assert tokens is None
    # The bare-copy penalty (0.02 by default config) shaves the reported
    # cost-adjusted reward; the raw ICIR stays below the gate.
    assert trainer.best_val_reward == -float("inf")
    rejected = trainer.selection_result.best_rejected
    assert rejected is not None
    assert rejected.val_reward == pytest.approx(0.78)
    assert rejected.val_icir == pytest.approx(0.01)
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
    train_end = trainer._training_contract().train_label_end
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
    assert trainer.best_val_icir == pytest.approx(0.4)
    assert trainer.best_train_icir == pytest.approx(0.4)
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
    train_end = trainer._training_contract().train_label_end
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
    assert artifact["val_reward"] == pytest.approx(trainer.best_val_reward)
    assert artifact["val_icir"] == pytest.approx(trainer.best_val_icir)
    assert artifact["train_reward"] == pytest.approx(
        trainer.best_train_reward
    )
    assert artifact["train_icir"] == pytest.approx(
        trainer.best_train_icir
    )
    assert "best_reward" not in artifact and "best_icir" not in artifact
    assert "full_window_reward" not in artifact and "full_window_icir" not in artifact


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
            len(loader.ts_codes) * trainer._training_contract().train_label_end,
            dtype=torch.float32,
        ).reshape(len(loader.ts_codes), trainer._training_contract().train_label_end),
    )
    # All rows sample the same legal formula (features + ADD + EOS):
    # unique fraction 1/4 < 0.9.  The EOS grammar forbids padding before
    # completion, so the forced sequence is the shortest operator-bearing
    # formula the mask admits.
    add_token = FORMULA_VOCAB.operator_offset
    pattern = [1, 1, add_token, FORMULA_VOCAB.eos_token_id]
    state = {"pos": -1}

    def fixed_sample(self):
        state["pos"] += 1
        return torch.full(
            (self.logits.shape[0],),
            pattern[state["pos"] % len(pattern)],
            dtype=torch.long,
            device=self.logits.device,
        )

    monkeypatch.setattr(Categorical, "sample", fixed_sample)
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
    vm = StackVM(
        FORMULA_VOCAB,
        universe_mask=torch.ones((5, 8), dtype=torch.bool),
    )
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
