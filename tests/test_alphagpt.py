from __future__ import annotations

import torch

from ashare_data.config import ModelConfig
from ashare_model.alphagpt import (
    MODEL_VERSION,
    AlphaGPTModel,
    LoopedTransformer,
    LoopedTransformerLayer,
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


def test_looped_transformer_forward():
    layer = LoopedTransformerLayer(8, 2, 16, num_loops=2)
    out = layer(torch.randn(2, 4, 8), is_causal=True)
    assert out.shape == (2, 4, 8)
    block = LoopedTransformer(8, 2, 2, 16, num_loops=2)
    assert block(torch.randn(2, 4, 8)).shape == (2, 4, 8)


def test_alphagpt_model_forward():
    model = AlphaGPTModel(ModelConfig(d_model=32, nhead=4, num_layers=1))
    idx = torch.tensor([[1, 2, 3], [2, 3, 4]])
    logits, value = model(idx)
    assert logits.shape == (2, FORMULA_VOCAB.size)
    assert value.shape == (2, 1)


def test_model_version_is_pinned():
    # P4 contract §6: v3 records mandatory elite-imitation initialization;
    # v2 checkpoints cannot prove that training contract and are retrained.
    assert MODEL_VERSION == 3
    model = AlphaGPTModel(ModelConfig(d_model=32, nhead=4, num_layers=1))
    state = model.state_dict()
    assert not any("mtp_head" in name for name in state)
    assert "head.weight" in state and "head.bias" in state


def _stack_types(sizes, capacity, fill=None):
    """Semantic-type state helper (P7-E): every live slot carries one
    continuous type (return_like by default)."""
    from ashare_model.feature_metadata import SemanticType
    from ashare_model.semantic_sampling import type_id

    if fill is None:
        fill = type_id(SemanticType.RETURN_LIKE)
    st = torch.zeros(len(sizes), capacity, dtype=torch.long)
    for i, size in enumerate(sizes):
        st[i, :size] = fill
    return st


def test_build_action_mask_done_padding():
    stack_sizes = torch.ones(8, dtype=torch.long)
    done = torch.ones(8, dtype=torch.bool)
    mask = build_action_mask(
        stack_sizes, done, 0, 6, FORMULA_VOCAB,
        stack_types=_stack_types([1] * 8, 6),
    )
    assert (mask == 0.0).sum() == 8
    assert (mask[:, 0] == 0.0).all()


def test_build_action_mask_has_legal_feature_tokens():
    stack_sizes = torch.zeros(8, dtype=torch.long)
    done = torch.zeros(8, dtype=torch.bool)
    mask = build_action_mask(
        stack_sizes, done, 0, 6, FORMULA_VOCAB,
        stack_types=_stack_types([0] * 8, 6),
    )
    assert (mask[:, FORMULA_VOCAB.feature_offset : FORMULA_VOCAB.operator_offset] == 0.0).all()
    assert not torch.isfinite(mask[:, FORMULA_VOCAB.operator_offset]).any()


def test_build_action_mask_eos_requires_single_stack_value():
    # EOS is legal exactly when the stack holds one value and the formula
    # is not done yet.
    stack_one = torch.ones(4, dtype=torch.long)
    stack_two = torch.full((4,), 2, dtype=torch.long)
    done = torch.zeros(4, dtype=torch.bool)
    mask = build_action_mask(
        stack_one, done, 1, 6, FORMULA_VOCAB,
        stack_types=_stack_types([1] * 4, 6),
    )
    assert (mask[:, FORMULA_VOCAB.eos_token_id] == 0.0).all()
    mask = build_action_mask(
        stack_two, done, 1, 6, FORMULA_VOCAB,
        stack_types=_stack_types([2] * 4, 6),
    )
    assert not torch.isfinite(mask[:, FORMULA_VOCAB.eos_token_id]).any()


def test_build_action_mask_pad_only_after_eos():
    # Not-done rows may never pad; done rows may only pad.  The EOS token
    # is a real, independent token of the vocabulary.
    stack_sizes = torch.tensor([1, 3, 1, 1], dtype=torch.long)
    done = torch.tensor([False, False, True, False], dtype=torch.bool)
    mask = build_action_mask(
        stack_sizes, done, 1, 6, FORMULA_VOCAB,
        stack_types=_stack_types([1, 3, 1, 1], 6),
    )
    assert (mask[~done, FORMULA_VOCAB.pad_token_id] != 0.0).all()
    assert (mask[done, FORMULA_VOCAB.pad_token_id] == 0.0).all()
    assert (mask[done, 1:] != 0.0).all()


def test_build_action_mask_binary_requires_two_stack_values():
    stack_sizes = torch.tensor([1, 2], dtype=torch.long)
    done = torch.zeros(2, dtype=torch.bool)
    mask = build_action_mask(
        stack_sizes, done, 0, 12, FORMULA_VOCAB,
        stack_types=_stack_types([1, 2], 12),
    )
    op = FORMULA_VOCAB.operator_offset  # ADD is the first operator
    assert not torch.isfinite(mask[0, op]).any()
    assert (mask[1, op] == 0.0).all()


def test_build_action_mask_binary_requires_same_type():
    # P7-E: ADD is family-preserving — two values of different semantic
    # types make the operator illegal even with stack >= 2.
    from ashare_model.feature_metadata import SemanticType
    from ashare_model.semantic_sampling import type_id

    ret = type_id(SemanticType.RETURN_LIKE)
    price = type_id(SemanticType.PRICE_LIKE)
    stack_sizes = torch.tensor([2, 2], dtype=torch.long)
    done = torch.zeros(2, dtype=torch.bool)
    st = torch.zeros(2, 12, dtype=torch.long)
    st[0, :2] = ret  # same family -> ADD legal
    st[1, 0], st[1, 1] = ret, price  # mixed -> ADD illegal
    mask = build_action_mask(stack_sizes, done, 0, 12, FORMULA_VOCAB, stack_types=st)
    op = FORMULA_VOCAB.operator_offset  # ADD
    assert (mask[0, op] == 0.0).all()
    assert not torch.isfinite(mask[1, op]).any()


def test_build_action_mask_enforces_open_boolean_depth_discipline():
    """P7-E: an open event may only be consumed as GATE's deepest
    argument; the two branch values must be continuous and same-family."""
    from ashare_model.feature_metadata import SemanticType
    from ashare_model.semantic_sampling import type_id

    bool_id = type_id(SemanticType.BOOLEAN_EVENT_SIGNAL)
    ret_id = type_id(SemanticType.RETURN_LIKE)
    price_id = type_id(SemanticType.PRICE_LIKE)
    stack_sizes = torch.tensor([1, 2, 3, 3], dtype=torch.long)
    done = torch.zeros(4, dtype=torch.bool)
    stack_types = torch.zeros(4, 12, dtype=torch.long)
    stack_types[0, 0] = bool_id
    stack_types[1, :2] = torch.tensor([bool_id, ret_id])
    stack_types[2, :3] = torch.tensor([bool_id, ret_id, ret_id])
    stack_types[3, :3] = torch.tensor([bool_id, ret_id, price_id])

    mask = build_action_mask(
        stack_sizes,
        done,
        0,
        12,
        FORMULA_VOCAB,
        stack_types=stack_types,
    )
    ret = FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index("RET_1")
    price = (
        FORMULA_VOCAB.feature_offset
        + FORMULA_VOCAB.feature_names.index("CLOSE_POSITION")
    )
    event = (
        FORMULA_VOCAB.feature_offset
        + FORMULA_VOCAB.feature_names.index("LIMIT_UP_EVENT")
    )
    add = FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index("ADD")
    gate = FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index("GATE")

    assert mask[0, ret] == 0.0
    assert mask[0, price] == 0.0
    assert not torch.isfinite(mask[0, event])
    assert mask[1, ret] == 0.0
    assert not torch.isfinite(mask[1, price])
    assert not torch.isfinite(mask[1, gate])
    assert mask[2, gate] == 0.0
    assert not torch.isfinite(mask[2, add])
    assert not torch.isfinite(mask[3, gate])


def test_build_action_mask_open_boolean_escape_bounds_are_tight():
    """P7 handoff invariant: the shortest gated event formula is exactly
    ``event, x, y, GATE, EOS``; every prefix is admitted at equality and
    rejected with one fewer remaining position."""
    from ashare_model.feature_metadata import SemanticType
    from ashare_model.semantic_sampling import type_id

    bool_id = type_id(SemanticType.BOOLEAN_EVENT_SIGNAL)
    ret_id = type_id(SemanticType.RETURN_LIKE)
    event = (
        FORMULA_VOCAB.feature_offset
        + FORMULA_VOCAB.feature_names.index("LIMIT_UP_EVENT")
    )
    ret = FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index("RET_1")
    gate = FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index("GATE")

    def legal(stack: list[int], max_len: int, token: int) -> bool:
        stack_types = torch.zeros(1, 6, dtype=torch.long)
        if stack:
            stack_types[0, : len(stack)] = torch.tensor(stack)
        mask = build_action_mask(
            torch.tensor([len(stack)]),
            torch.tensor([False]),
            0,
            max_len,
            FORMULA_VOCAB,
            stack_types=stack_types,
        )
        return bool(mask[0, token] == 0.0)

    # Closed -> event producer: four positions must remain for x/y/GATE/EOS.
    assert legal([], 5, event)
    assert not legal([], 4, event)
    # d=1 -> first branch: y/GATE/EOS remain.
    assert legal([bool_id], 4, ret)
    assert not legal([bool_id], 3, ret)
    # d=2 -> matching second branch: GATE/EOS remain.
    assert legal([bool_id, ret_id], 3, ret)
    assert not legal([bool_id, ret_id], 2, ret)
    # d=3 -> GATE: EOS remains.
    assert legal([bool_id, ret_id, ret_id], 2, gate)
    assert not legal([bool_id, ret_id, ret_id], 1, gate)


def test_build_action_mask_always_leaves_a_terminating_path():
    # Every legal prefix must keep at least one token legal, and the only
    # way to reach the final position is with stack == 1 so EOS fits.
    torch.manual_seed(0)
    stack_sizes = torch.zeros(32, dtype=torch.long)
    stack_types = torch.zeros(32, 12, dtype=torch.long)
    done = torch.zeros(32, dtype=torch.bool)
    max_len = 12
    for step in range(max_len):
        mask = build_action_mask(
            stack_sizes, done, step, max_len, FORMULA_VOCAB,
            stack_types=stack_types,
        )
        totals = (mask == 0.0).sum(dim=1)
        assert (totals >= 1).all()
        # Greedy uniform resample, like the trainer's loop.
        allowed = (mask == 0.0).float()
        probs = allowed / allowed.sum(dim=1, keepdim=True)
        action = torch.multinomial(probs, 1).squeeze(1)
        from ashare_model.train import AshareTrainer

        AshareTrainer._update_stack_state(
            action, stack_sizes, stack_types, done
        )
    assert done.all()
