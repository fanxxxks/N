from __future__ import annotations

import torch

from lord.experiment import (
    ModularAdditionDataset,
    NewtonSchulzLowRankDecay,
    get_stable_rank,
)


def test_modular_addition_dataset_splits():
    train = ModularAdditionDataset(p=10, split="train", train_frac=0.5, seed=42)
    val = ModularAdditionDataset(p=10, split="val", train_frac=0.5, seed=42)
    assert len(train) + len(val) == 100
    x, y = train[0]
    assert x.shape == (3,)
    assert y.shape == ()


def test_newton_schulz_step_changes_parameters():
    params = [("q_proj", torch.randn(4, 4, requires_grad=True))]
    decay = NewtonSchulzLowRankDecay(
        params,
        decay_rate=1e-3,
        num_iterations=2,
        target_keywords=["q_proj"],
    )
    before = params[0][1].detach().clone()
    decay.step()
    assert not torch.allclose(params[0][1], before)


def test_stable_rank_smoke():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4)

    model = DummyModel()
    rank = get_stable_rank(model)
    assert isinstance(rank, float)
