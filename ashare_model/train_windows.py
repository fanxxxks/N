"""Training-window helpers shared by every searcher entry point.

Extracted from ``train.py`` (P7 Phase B1) by reason-to-change: this module
owns *window arithmetic and search-space sampling* — the validation-tail
split, the bound ``_TrainWindow`` context, uniform random-formula sampling
and device resolution.  It changes when window/sampling *contracts* change
— not when the RL update rule, artifact schemas or the CLI change.

The module is import-leaf-ward of the ``train`` facade: it never imports
``ashare_model.train`` at module level.  ``sample_random_formulas`` needs
the trainer's static stack-state update (the single implementation lives
on ``AshareTrainer``, called directly by existing tests), so it resolves
the class lazily inside the function body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .alphagpt import build_action_mask
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


def sample_random_formulas(
    seed: int, vocab, max_len: int, n: int, feature_ids: list[int] | None = None
) -> list[tuple[int, ...]]:
    """Sample ``n`` structurally valid postfix formulas under a uniform
    prior over the legal action mask (the exact legality rules the policy
    samples under, so the random-search baseline and the RL policy share
    one search space).  Deterministic in ``seed``; pins torch's CPU RNG
    exactly like the trainer does.  Every sequence is EOS-terminated, so
    its effective (non-padded) length is always >= 2.

    ``feature_ids`` (P6 §4.2) restricts sampling to the given feature
    tokens; ``None`` keeps the pre-P6 full-vocabulary search space.
    """

    # Late import: the stack-state update's single implementation lives on
    # the trainer (existing tests call it there); a module-level import
    # would cycle through the train facade's re-export of this module.
    from .train import AshareTrainer  # noqa: PLC0415

    torch.manual_seed(seed)
    stack_sizes = torch.zeros(n, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    seqs: list[torch.Tensor] = []
    for pos in range(max_len):
        mask = build_action_mask(
            stack_sizes, done, pos, max_len, vocab, feature_ids=feature_ids
        )
        allowed = (mask == 0.0).float()
        totals = allowed.sum(dim=1, keepdim=True)
        # The legal mask guarantees at least one allowed token per row; the
        # uniform fallback keeps sampling total even if that invariant is
        # ever violated by a vocabulary change.
        safe = torch.where(
            totals > 0, allowed, torch.full_like(allowed, 1.0 / vocab.size)
        )
        probs = safe / safe.sum(dim=1, keepdim=True)
        action = torch.multinomial(probs, 1).squeeze(1)
        seqs.append(action)
        AshareTrainer._update_stack_state(action, stack_sizes, done)
    return [tuple(int(t) for t in row) for row in torch.stack(seqs, dim=1).tolist()]


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
