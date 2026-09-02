"""Training-window helpers shared by every searcher entry point.

Extracted from ``train.py`` (P7 Phase B1) by reason-to-change: this module
owns *window arithmetic and search-space sampling* — the validation-tail
split, the bound ``_TrainWindow`` context, uniform random-formula sampling
and device resolution.  It changes when window/sampling *contracts* change
— not when the RL update rule, artifact schemas or the CLI change.

The module is import-leaf-ward of the ``train`` facade: it never imports
``ashare_model.train``.  Random and RL sampling share the vocabulary-aware
state transition in :mod:`ashare_model.semantic_sampling`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .alphagpt import build_action_mask
from .search_length_prior import sample_target_content_length
from .semantic_sampling import advance_stack_state
from .time_contract import TrainingTimeContract


@dataclass
class _TrainWindow:
    """One training window bound for every searcher (RL, random, GP): the
    tensors, masks and windows the candidate scorer consumes, plus the
    semantic-budget cache and the calibration fingerprint executor."""

    contract: "TrainingTimeContract"
    train_end_idx: int
    val_windows: list[tuple[int, int]]
    val_start: int
    train_signal_range: tuple[int, int]
    factor_tensor: torch.Tensor
    vm_device: torch.device
    train_universe_mask: np.ndarray
    target_ret: np.ndarray
    realized_ret: np.ndarray
    rebalance_mask: np.ndarray
    blocked_buy: np.ndarray
    blocked_sell: np.ndarray
    reward_chunk: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validation_start(train_signal_end: int, model_config) -> int:
    """First index of the validation tail inside the training window.

    The tail keeps at least two dates (a reward needs one daily return);
    windows too small to hold anything out validate on the full window.
    """

    if train_signal_end <= 2:
        return 0
    val_frac = float(np.clip(model_config.validation_fraction, 0.0, 0.5))
    val_start = int(round(train_signal_end * (1.0 - val_frac)))
    return max(1, min(val_start, train_signal_end - 2))


def validation_windows(
    train_signal_end: int,
    model_config,
    *,
    rebalance_mask: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Independent validation sub-windows covering the training tail.

    The tail is split into ``model_config.validation_splits`` disjoint
    ``(start, end)`` column intervals of at least 3 columns each
    (a rank-ICIR needs a minimum number of cross-sections to be
    meaningful).  Tails too short to split degrade to a single window,
    preserving the historical single-holdout behavior.  Shared by the
    trainer and the protocol's random-search baseline so both score
    against the identical selection windows.
    """

    val_start = validation_start(train_signal_end, model_config)
    val_len = train_signal_end - val_start
    splits = max(1, int(model_config.validation_splits))
    min_len = 3
    if val_len < splits * min_len:
        windows = [(val_start, train_signal_end)]
    else:
        base = val_len // splits
        windows = []
        for k in range(splits):
            start = val_start + k * base
            end = (
                val_start + (k + 1) * base
                if k < splits - 1
                else train_signal_end
            )
            windows.append((start, end))
    if rebalance_mask is not None:
        due = np.asarray(rebalance_mask, dtype=bool)
        if due.ndim != 1 or due.shape[0] < train_signal_end:
            raise ValueError(
                f"rebalance_mask shape {due.shape} does not cover "
                f"train_signal_end={train_signal_end}"
            )
        for start, end in windows:
            observations = int(np.count_nonzero(due[start:end]))
            if observations < 2:
                raise ValueError(
                    f"validation window {start}:{end} has {observations} "
                    "rebalance observations; at least 2 are required for ICIR"
                )
    return windows


def _walk_uniform_batch(
    count: int,
    vocab,
    max_len: int,
    effective_len: int,
    feature_ids: list[int] | None,
) -> list[tuple[int, ...]]:
    """One batched uniform walk over the legal action mask (the P14 length
    prior bounds each batch by its effective position count, so the mask
    forces EOS termination at the bound; rows are then PAD-padded back to
    ``max_len`` — the historical shape of every sampled sequence).
    Consumes the global torch RNG sequentially — deterministic under a
    prior ``torch.manual_seed``."""

    stack_sizes = torch.zeros(count, dtype=torch.long)
    # P7-E: the sampling state carries one semantic-type id per stack slot
    # so the action mask can enforce operator signature legality.
    stack_types = torch.zeros(count, max_len, dtype=torch.long)
    done = torch.zeros(count, dtype=torch.bool)
    seqs: list[torch.Tensor] = []
    for pos in range(effective_len):
        mask = build_action_mask(
            stack_sizes, done, pos, effective_len, vocab, feature_ids=feature_ids,
            stack_types=stack_types,
        )
        allowed = (mask == 0.0).float()
        totals = allowed.sum(dim=1, keepdim=True)
        no_legal = torch.nonzero(
            totals.squeeze(1) == 0, as_tuple=False
        ).flatten()
        if no_legal.numel():
            rows = no_legal[:16].tolist()
            suffix = "" if no_legal.numel() <= 16 else f" (+{no_legal.numel() - 16} more)"
            raise RuntimeError(
                f"no legal token at step {pos} for rows {rows}{suffix}; "
                "typed sampling fails closed"
            )
        probs = allowed / totals
        action = torch.multinomial(probs, 1).squeeze(1)
        seqs.append(action)
        advance_stack_state(
            action,
            stack_sizes,
            stack_types,
            done,
            vocab=vocab,
        )
    pad = vocab.pad_token_id
    rows = torch.stack(seqs, dim=1).tolist()
    return [
        tuple(int(t) for t in row) + (pad,) * (max_len - len(row))
        for row in rows
    ]


def sample_random_formulas(
    seed: int, vocab, max_len: int, n: int, feature_ids: list[int] | None = None
) -> list[tuple[int, ...]]:
    """Sample ``n`` structurally valid postfix formulas over the legal
    action mask with P14 per-sequence EOS length targets (profile
    ``p14-uniform-2-11-v1``): each sequence walks under an effective
    position bound of ``target + 1``, so the mask forces EOS termination
    exactly at the drawn target and the proposal content-length
    distribution is the preregistered discrete-uniform target (for
    ``max_len < 4`` the profile is inactive and the walk keeps the legacy
    full-length bound).  The exact legality rules the policy samples under
    are unchanged, so the random-search baseline and the RL policy share
    one search space.  Deterministic in ``seed``; pins torch's CPU RNG
    exactly like the trainer does.  Every sequence is EOS-terminated, so
    its effective (non-padded) length is always >= 2.

    ``feature_ids`` (P6 §4.2) restricts sampling to the given feature
    tokens; ``None`` keeps the pre-P6 full-vocabulary search space.
    """

    torch.manual_seed(seed)
    length_rng = random.Random(seed)
    if max_len >= 4:
        targets = [length_rng.randint(2, max_len - 1) for _ in range(n)]
    else:
        targets = [None] * n
    groups: dict[int | None, int] = {}
    for target in targets:
        groups[target] = groups.get(target, 0) + 1
    out: list[tuple[int, ...]] = []
    # Deterministic group order: the inactive profile first, then targets
    # ascending.  Each group is one batched walk under its effective bound;
    # rows are PAD-padded back to ``max_len`` (the historical shape).
    for target in sorted(groups, key=lambda t: (t is None, t or 0)):
        count = groups[target]
        effective_len = max_len if target is None else target + 1
        out.extend(
            _walk_uniform_batch(count, vocab, max_len, effective_len, feature_ids)
        )
    return out


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a ``--device`` value to a concrete torch device.

    ``auto`` (the default) uses CUDA when available and falls back to CPU,
    so the same entry point runs accelerated on a GPU machine and unchanged
    on GPU-less CI.  Unknown values fail loudly.
    """

    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError(
            f"unknown device {requested!r}; expected 'auto', 'cpu' or 'cuda'"
        )
    return torch.device(requested)
