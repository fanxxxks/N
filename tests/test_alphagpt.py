from __future__ import annotations

import torch

from ashare_data.config import ModelConfig
from ashare_model.alphagpt import (
    AlphaGPTModel,
    LoopedTransformer,
    LoopedTransformerLayer,
    MTPHead,
    QKNorm,
    RMSNorm,
    SwiGLU,
    build_action_mask,
)
from ashare_model.vocab import FORMULA_VOCAB


def test_rms_norm():
    layer = RMSNorm(8)
    x = torch.randn(2, 3, 8)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_qk_norm():
    layer = QKNorm(8)
    q = torch.randn(2, 3, 2, 8)
    k = torch.randn(2, 3, 2, 8)
    qn, kn = layer(q, k)
    assert qn.shape == q.shape and kn.shape == k.shape


def test_swiglu():
    layer = SwiGLU(8, 16)
    out = layer(torch.randn(2, 3, 8))
    assert out.shape == (2, 3, 8)


def test_mtp_head():
    head = MTPHead(8, FORMULA_VOCAB.size)
    out, probs = head(torch.randn(2, 8))
    assert out.shape == (2, FORMULA_VOCAB.size)
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))


def test_looped_transformer_forward():
    layer = LoopedTransformerLayer(8, 2, 16, num_loops=2)
    out = layer(torch.randn(2, 4, 8), is_causal=True)
    assert out.shape == (2, 4, 8)
    block = LoopedTransformer(8, 2, 2, 16, num_loops=2)
    assert block(torch.randn(2, 4, 8)).shape == (2, 4, 8)


def test_alphagpt_model_forward():
    model = AlphaGPTModel(ModelConfig(d_model=32, nhead=4, num_layers=1))
    idx = torch.tensor([[1, 2, 3], [2, 3, 4]])
    logits, value, probs = model(idx)
    assert logits.shape == (2, FORMULA_VOCAB.size)
    assert value.shape == (2, 1)
    assert probs.shape == (2, 3)


def test_build_action_mask_done_padding():
    open_slots = torch.zeros(8, dtype=torch.long)
    stack_sizes = torch.ones(8, dtype=torch.long)
    mask = build_action_mask(open_slots, stack_sizes, 0, 6, FORMULA_VOCAB)
    assert (mask == 0.0).sum() == 8
    assert (mask[:, 0] == 0.0).all()


def test_build_action_mask_has_legal_feature_tokens():
    open_slots = torch.ones(8, dtype=torch.long)
    stack_sizes = torch.zeros(8, dtype=torch.long)
    mask = build_action_mask(open_slots, stack_sizes, 0, 6, FORMULA_VOCAB)
    assert (mask[:, 1 : FORMULA_VOCAB.operator_offset] == 0.0).all()
    assert not torch.isfinite(mask[:, FORMULA_VOCAB.operator_offset]).any()
