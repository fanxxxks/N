"""LoopedTransformer policy model for A-share factor discovery.

The model is a single-task policy: one linear head over the formula
vocabulary plus a scalar critic.  The former multi-task head (MTPHead)
was removed in MODEL_VERSION 2 — it had no multi-task supervision and
the trainer discarded its router probabilities, so it was dead
complexity; the model keeps exactly one head with real training signal.
MODEL_VERSION 3 adds the audited elite-imitation initialization contract;
the tensor architecture is unchanged, but v2 checkpoints cannot establish
whether this required initialization occurred and are therefore not promoted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_metadata import SemanticType
from .ops import OPS_BY_NAME
from .semantic_sampling import (
    feature_types,
    fixed_output,
    signature_legal,
    type_id,
)
from .vocab import FORMULA_VOCAB

# Model architecture generation.  v1 shipped the multi-task head (MTPHead);
# v2 removed it (no supervision existed); v3 requires an explicit, recorded
# random or elite-imitation initialization.  Recorded in training artifacts so
# a checkpoint is never silently interpreted under the wrong training contract.
MODEL_VERSION = 3

# P7-E per-operator signature lookup tables, cached against a fingerprint
# of the registry state the lookups were derived from.  The mask must
# never drift from ``OPERATOR_REGISTRY``: any registry change produces a
# new fingerprint and rebuilds the tables (verified by
# tests/test_semantic_sampling.py::test_checker_is_registry_driven).
_OP_RULE_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def _op_rule_lookups(vocab, device: torch.device) -> dict[str, torch.Tensor]:
    """Flat legality lookup per operator over semantic-type-id tuples.

    Entry ``(t_1, ..., t_arity)`` (ids follow ``list(SemanticType)``
    order, base-6 encoding) is True iff
    :func:`ashare_model.semantic_sampling.signature_legal` accepts the
    application — one source of truth, evaluated eagerly per combo so the
    mask check stays vectorized.
    """

    import itertools

    from ashare_model.operator_registry import operator_meta_of

    fingerprint = (
        tuple(
            (
                name,
                operator_meta_of(name).inputs,
                operator_meta_of(name).output,
            )
            for name in vocab.operator_names
        ),
        device.type,
        device.index,
    )
    cached = _OP_RULE_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    types = tuple(SemanticType)
    n_types = len(types)
    lookups: dict[str, torch.Tensor] = {}
    for name in vocab.operator_names:
        try:
            arity = OPS_BY_NAME[name][1]
        except KeyError:
            raise ValueError(
                f"operator {name!r} has no implementation/signature in OPS_BY_NAME"
            ) from None
        lookup = torch.zeros(n_types**arity, dtype=torch.bool)
        for combo in itertools.product(range(n_types), repeat=arity):
            lookup[
                _combo_index(combo, n_types)
            ] = signature_legal(name, tuple(types[t] for t in combo))
        lookups[name] = lookup.to(device=device)
    _OP_RULE_CACHE.clear()  # keep only the current registry generation
    _OP_RULE_CACHE[fingerprint] = lookups
    return lookups


def _combo_index(combo: tuple[int, ...], n_types: int) -> int:
    index = 0
    for t in combo:
        index = index * n_types + t
    return index


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class QKNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, 1, 1, d_model) * (d_model**-0.5))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return q_norm * self.scale, k_norm * self.scale


class SwiGLU(nn.Module):
    def __init__(self, d_in: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_in, d_ff * 2)
        self.fc = nn.Linear(d_ff, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_glu = self.w(x)
        x, gate = x_glu.chunk(2, dim=-1)
        return self.fc(x * F.silu(gate))


class LoopedTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_loops: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_loops = num_loops
        self.d_model = d_model
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal and mask is None:
            seq_len = x.shape[1]
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                diagonal=1,
            )
        for _ in range(self.num_loops):
            x_norm = self.norm1(x)
            attn_out, _ = self.attention(
                x_norm, x_norm, x_norm, attn_mask=mask, is_causal=False
            )
            x = x + self.dropout(attn_out)
            x_norm = self.norm2(x)
            x = x + self.dropout(self.ffn(x_norm))
        return x


class LoopedTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_loops: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LoopedTransformerLayer(
                    d_model, nhead, dim_feedforward, num_loops, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=is_causal)
        return x


class AlphaGPTModel(nn.Module):
    """Policy and value model over the A-share formula vocabulary."""

    def __init__(self, config, vocab=None):
        super().__init__()
        self.vocab = vocab or FORMULA_VOCAB
        self.vocab_size = self.vocab.size
        self.d_model = config.d_model
        self.max_len = config.max_formula_len

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.max_len + 1, self.d_model)
        )
        self.blocks = LoopedTransformer(
            d_model=self.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            num_loops=config.num_loops,
            dropout=config.dropout,
        )
        self.ln_f = RMSNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)
        self.head_critic = nn.Linear(self.d_model, 1)

    def forward(self, idx: torch.Tensor):
        b, t = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device),
            diagonal=1,
        )
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        logits = self.head(last)
        value = self.head_critic(last)
        return logits, value


def build_action_mask(
    stack_sizes: torch.Tensor,
    done: torch.Tensor,
    step: int,
    max_len: int,
    vocab=None,
    *,
    feature_ids: list[int] | None = None,
    stack_types: torch.Tensor,
) -> torch.Tensor:
    """Mask tokens so sampled sequences remain legal postfix formulas.

    Legality is stack-only (the legacy ``open_slots`` hybrid is gone; see
    :mod:`ashare_model.ir`):

    * feature: ``stack + 1``, and the pushed value must still be reducible
      to one result plus an EOS within the remaining positions;
    * unary: ``stack >= 1`` (stack unchanged);
    * binary: ``stack >= 2``, ``stack - 1`` after execution;
    * ternary: ``stack >= 3``, ``stack - 2`` after execution;
    * EOS: only when ``stack == 1``;
    * PAD: only after EOS (``done``); EOS is emitted explicitly, so PAD
      never appears mid-formula.

    ``done`` marks rows that already emitted EOS; their only legal token is
    PAD.  A completed formula therefore always carries an explicit EOS
    before its padding.

    ``feature_ids`` (P6 §4.2) restricts the open feature tokens to the
    given global vocab ids (research-domain search spaces); ``None`` keeps
    the pre-P6 behavior of opening every feature token.

    ``stack_types`` (P7-E, GRAMMAR v3; docs/p7_semantic_types_contract.md)
    is the LongTensor ``[batch, capacity]`` of semantic-type ids on the
    stack (0 = empty slot; ids follow ``list(SemanticType)`` order + 1).
    Operator tokens are only legal when the top-of-stack argument types
    satisfy the operator's registered semantic signature
    (:func:`ashare_model.semantic_sampling.signature_legal`); the type
    state itself is advanced by
    :func:`ashare_model.train.AshareTrainer._update_stack_state`.  The
    parameter is required — there is no untyped fallback (fail-closed).

    The termination bound is type-aware and tight under the P7-E
    open-boolean discipline:
    boolean-event values (JUMP output, event features) can only be
    consumed as a GATE condition.  At most one boolean may sit on the
    stack; depth 1 admits a continuous branch push, depth 2 admits only a
    same-family branch push, and depth 3 admits only GATE.  Their exact
    post-action reserves are respectively ``s + 2``, ``s`` and ``s - 2``.
    A boolean producer reserves ``s' + 3`` actions (x/y/GATE plus the
    closed-stack reduction/EOS cost); a continuous post-state of size
    ``s'`` reserves ``s'``.  EOS remains legal whenever ``stack == 1``, so
    a bare event formula is still valid.  These bounds keep every admitted
    prefix on a terminating path.
    """

    vocab = vocab or FORMULA_VOCAB
    batch = stack_sizes.shape[0]
    if done.shape != stack_sizes.shape:
        raise ValueError("done and stack_sizes must have the same shape")
    if stack_types.ndim != 2 or stack_types.shape[0] != batch:
        raise ValueError(
            "stack_types must be a [batch, capacity] tensor aligned with stack_sizes"
        )
    if stack_types.device != stack_sizes.device or done.device != stack_sizes.device:
        raise ValueError("stack_sizes, stack_types and done must share one device")
    if stack_types.dtype != torch.long:
        raise ValueError("stack_types must be a LongTensor")
    capacity = stack_types.shape[1]
    if capacity == 0 or bool((stack_sizes < 0).any()) or bool((stack_sizes > capacity).any()):
        raise ValueError("stack_sizes must stay within stack_types capacity")

    # Resolve every vocabulary member before the early-done return.  Missing
    # feature metadata or operator signatures therefore fail at mask build,
    # even for an all-done batch (contract §5: no untyped fallback).
    feature_type_ids = torch.tensor(
        [type_id(t) for t in feature_types(vocab)],
        dtype=torch.long,
        device=stack_sizes.device,
    )
    lookups = _op_rule_lookups(vocab, stack_sizes.device)
    n_types = len(SemanticType)
    bool_id = type_id(SemanticType.BOOLEAN_EVENT_SIGNAL)

    allowed_features = torch.ones(
        vocab.feature_count, dtype=torch.bool, device=stack_sizes.device
    )
    if feature_ids is not None:
        allowed_features.zero_()
        tokens = torch.as_tensor(
            [int(token) for token in feature_ids],
            dtype=torch.long,
            device=stack_sizes.device,
        )
        if tokens.numel() and bool(
            ((tokens < vocab.feature_offset) | (tokens >= vocab.operator_offset)).any()
        ):
            raise ValueError("feature_ids may contain only feature-token ids")
        if tokens.numel():
            allowed_features[tokens - vocab.feature_offset] = True

    # P9 §4.2 (grammar v4): deprecated features leave the sampling space
    # while keeping their token ids and computations — they are never
    # legal to sample, with or without a domain feature_ids restriction.
    # The policy rides the vocabulary object, so toy/legacy vocabularies
    # (empty deprecated set) are unaffected.
    deprecated_ids = torch.tensor(
        [
            index
            for index, name in enumerate(vocab.feature_names)
            if name in vocab.deprecated_names
        ],
        dtype=torch.long,
        device=stack_sizes.device,
    )
    if deprecated_ids.numel():
        allowed_features[deprecated_ids] = False

    mask = torch.full(
        (batch, vocab.size), float("-inf"), device=stack_sizes.device
    )
    if done.any():
        mask[done, vocab.pad_token_id] = 0.0

    active = ~done
    if not active.any():
        return mask

    # Per-row open-boolean state (P7-E): an admitted prefix contains at
    # most one event value, at depth 1, 2 or 3 until GATE consumes it.
    live_slots = torch.arange(capacity, device=stack_types.device).unsqueeze(0)
    live = live_slots < stack_sizes.unsqueeze(1)
    invalid_live_type = (
        active.unsqueeze(1)
        & live
        & ((stack_types < 1) | (stack_types > n_types))
    )
    if invalid_live_type.any():
        raise ValueError("every live stack slot must carry a valid semantic-type id")
    is_bool_cell = (stack_types == bool_id) & live
    bool_count = is_bool_cell.sum(dim=1)
    bool_slot = torch.argmax(is_bool_cell.to(torch.long), dim=1)
    bool_depth = torch.where(
        bool_count == 1, stack_sizes - bool_slot, torch.zeros_like(stack_sizes)
    )
    top_type = stack_types.gather(
        1, (stack_sizes - 1).clamp(min=0).unsqueeze(1)
    ).squeeze(1)

    remaining_after = max_len - step - 1

    eos = vocab.eos_token_id
    if eos is not None:
        can_eos = active & (stack_sizes == 1)
        mask[can_eos, eos] = 0.0

    # --- features ---------------------------------------------------------
    # Closed continuous stacks of post-size s' need exactly s' actions to
    # reduce to one value and emit EOS.  Producing an event opens a boolean
    # at depth 1 and reserves x/y/GATE plus that same closed-stack escape.
    closed = active & (bool_count == 0)
    continuous_feature = feature_type_ids != bool_id
    event_feature = feature_type_ids == bool_id
    closed_continuous = closed & (remaining_after >= stack_sizes + 1)
    closed_event_producer = closed & (remaining_after >= stack_sizes + 1 + 3)

    # Open-boolean exact escape costs after the candidate feature:
    # d=1 -> d=2 needs y + GATE + closed reduction/EOS (s + 2 actions);
    # d=2 -> d=3 needs GATE + closed reduction/EOS (s actions).
    open_d1 = (
        active
        & (bool_count == 1)
        & (bool_depth == 1)
        & (remaining_after >= stack_sizes + 2)
    )
    open_d2 = (
        active
        & (bool_count == 1)
        & (bool_depth == 2)
        & (remaining_after >= stack_sizes)
    )

    legal_features = (
        closed_continuous.unsqueeze(1) & continuous_feature.unsqueeze(0)
    ) | (
        closed_event_producer.unsqueeze(1) & event_feature.unsqueeze(0)
    ) | (
        open_d1.unsqueeze(1) & continuous_feature.unsqueeze(0)
    ) | (
        open_d2.unsqueeze(1)
        & continuous_feature.unsqueeze(0)
        & (feature_type_ids.unsqueeze(0) == top_type.unsqueeze(1))
    )
    legal_features &= allowed_features.unsqueeze(0)
    mask.narrow(1, vocab.feature_offset, vocab.feature_count).masked_fill_(
        legal_features, 0.0
    )

    # Operators are resolved by name through OPS_BY_NAME so the mask stays
    # correctly aligned for any vocabulary whose operator names are a
    # subset of the operator table (e.g. toy test vocabularies).  P7-E:
    # an operator is only legal when the top-of-stack argument types
    # satisfy its registered semantic signature; the per-row gathered
    # types make the check vectorized over the batch, with per-operator
    # lookup tables cached against the registry fingerprint (a registry
    # change invalidates the cache, so the mask can never drift from it).
    if vocab.operator_names:
        max_arity = max(OPS_BY_NAME[name][1] for name in vocab.operator_names)
        offsets = torch.arange(
            max_arity, 0, -1, device=stack_sizes.device
        )
        slots = (stack_sizes.unsqueeze(1) - offsets).clamp(min=0)
        top_types = stack_types.gather(1, slots)

        for index, name in enumerate(vocab.operator_names):
            token = vocab.operator_offset + index
            arity = OPS_BY_NAME[name][1]
            enough_args = active & (stack_sizes >= arity)
            cols = top_types[:, max_arity - arity :]
            valid_type_ids = ((cols >= 1) & (cols <= n_types)).all(dim=1)
            flat = torch.zeros(batch, dtype=torch.long, device=cols.device)
            for column in cols.unbind(dim=1):
                flat = flat * n_types + (column - 1).clamp(
                    min=0, max=n_types - 1
                )
            signature_ok = valid_type_ids & lookups[name][flat]
            post_size = stack_sizes - arity + 1

            output = fixed_output(name)
            if output is SemanticType.BOOLEAN_EVENT_SIGNAL:
                closed_feasible = remaining_after >= post_size + 3
            else:
                closed_feasible = remaining_after >= post_size
            can_closed = (
                enough_args
                & closed
                & signature_ok
                & closed_feasible
            )

            # Once an event is open, every operator except GATE is blocked.
            # At depth 3 GATE sees (condition, x, y) in push order; after it
            # returns a continuous branch value, a closed stack of size
            # ``post_size == s - 2`` needs exactly that many actions.
            can_gate_open = (
                enough_args
                & (bool_count == 1)
                & (bool_depth == 3)
                & (name == "GATE")
                & signature_ok
                & (remaining_after >= post_size)
            )
            mask[can_closed | can_gate_open, token] = 0.0
    return mask
