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

from .ops import OPS_BY_NAME
from .vocab import FORMULA_VOCAB

# Model architecture generation.  v1 shipped the multi-task head (MTPHead);
# v2 removed it (no supervision existed); v3 requires an explicit, recorded
# random or elite-imitation initialization.  Recorded in training artifacts so
# a checkpoint is never silently interpreted under the wrong training contract.
MODEL_VERSION = 3


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
    before its padding.  The feasibility bound is conservative (binary-only
    reductions: finishing from stack ``s`` needs at least ``s`` tokens), so
    the mask never admits a sequence that cannot terminate.

    ``feature_ids`` (P6 §4.2) restricts the open feature tokens to the
    given global vocab ids (research-domain search spaces); ``None`` keeps
    the pre-P6 behavior of opening every feature token.
    """

    vocab = vocab or FORMULA_VOCAB
    b = stack_sizes.shape[0]
    mask = torch.full((b, vocab.size), float("-inf"), device=stack_sizes.device)
    remaining = max_len - step

    if done.any():
        mask[done, vocab.pad_token_id] = 0.0

    active = ~done
    if not active.any():
        return mask

    eos = vocab.eos_token_id
    if eos is not None:
        can_eos = active & (stack_sizes == 1)
        mask[can_eos, eos] = 0.0

    # A feature pushes one value; from stack s the formula needs at least
    # s+1 more tokens (s binary reductions + EOS) to terminate.
    can_feature = active & (stack_sizes + 1 <= remaining - 1)
    if feature_ids is None:
        mask[can_feature, vocab.feature_offset : vocab.operator_offset] = 0.0
    else:
        # Fancy indexing pairs a row mask with a column list element-wise,
        # so the open set is built as a column mask first and combined
        # with the row mask via broadcasting (every open row gets every
        # allowed feature token).
        allowed = torch.zeros(
            vocab.size, dtype=torch.bool, device=mask.device
        )
        allowed[
            torch.as_tensor(
                [int(token) for token in feature_ids],
                dtype=torch.long,
                device=mask.device,
            )
        ] = True
        mask[can_feature.unsqueeze(1) & allowed.unsqueeze(0)] = 0.0

    # Operators are resolved by name through OPS_BY_NAME so the mask stays
    # correctly aligned for any vocabulary whose operator names are a
    # subset of the operator table (e.g. toy test vocabularies).
    for i, name in enumerate(vocab.operator_names):
        token = vocab.operator_offset + i
        arity = OPS_BY_NAME[name][1]
        can_op = (
            active
            & (stack_sizes >= arity)
            & (stack_sizes - arity + 1 <= remaining - 1)
        )
        mask[can_op, token] = 0.0
    return mask
