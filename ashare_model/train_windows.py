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
from .semantic_cache import (
    CalibrationSlice,
    SemanticCache,
    make_calibration_execute,
)
from .semantic_sampling import advance_stack_state
from .time_contract import TrainingTimeContract
from .versions import PROTOCOL_VERSION
from .reward import REWARD_VERSION
from .targets import causal_target_returns
from ashare_data.processor import open_to_open_returns
from ashare_portfolio.rebalance import RebalancePolicy


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


def _window_bounds(
    trainer,
    train_end_date: str | None,
    window_cap: tuple[int, int] | None,
):
    """Resolve the training contract, the (optionally capped) window
    bounds and the rebalance policy/mask (B3: contiguous extraction of
    the historical ``prepare_window`` prologue)."""

    contract = trainer._training_contract(train_end_date)
    train_end_idx = contract.train_label_end
    stock_slice = slice(None)
    if window_cap is not None:
        stocks, dates = int(window_cap[0]), int(window_cap[1])
        if stocks <= 0 or dates <= 0:
            raise ValueError("window_cap must be positive (stocks, dates)")
        stock_slice = slice(0, stocks)
        train_end_idx = min(train_end_idx, dates)
    # The capped window's signal end reserves the exact entry+holding
    # context declared by the target contract.
    train_signal_end = min(
        contract.train_signal_end,
        max(0, train_end_idx - contract.exit_offset),
    )
    policy = RebalancePolicy.from_config(trainer.backtest_config)
    full_rebalance_mask = policy.rebalance_mask(trainer.loader.dates)
    rebalance_mask = full_rebalance_mask[:train_end_idx]
    return (
        contract,
        train_end_idx,
        stock_slice,
        train_signal_end,
        policy,
        rebalance_mask,
    )


def _bind_vm_window(
    trainer,
    vm_device: torch.device,
    stock_slice: slice,
    train_end_idx: int,
) -> np.ndarray:
    """Move the VM's industry groups and PIT mask onto the compute device
    and return the sliced numpy eligibility mask for the scorer."""

    # The VM executes on the compute device; the industry-group tensor
    # for CS_NEUTRALIZE and the PIT eligibility mask for the
    # cross-sectional operators move with the factor stack (the loader
    # always builds the mask in load_data, so there is no fallback).
    industry_codes = getattr(trainer.loader, "industry_codes", None)
    trainer.vm.industry_codes = (
        industry_codes[stock_slice, :train_end_idx].to(vm_device)
        if industry_codes is not None
        else None
    )
    universe_mask = trainer.loader.universe_mask
    trainer.vm.universe_mask = torch.tensor(
        universe_mask[stock_slice, :train_end_idx],
        dtype=torch.bool,
        device=vm_device,
    )
    # The candidate scorer needs the same PIT mask on the numpy side:
    # every quality statistic (rank IC, basket selection, near-constant
    # rejection, direction) is gated to signal-date eligible cells.
    return universe_mask[stock_slice, :train_end_idx]


def _window_targets(
    trainer,
    policy: RebalancePolicy,
    stock_slice: slice,
    train_end_idx: int,
    rebalance_mask: np.ndarray,
    train_universe_mask: np.ndarray,
):
    """Build the causal targets, realized returns and tradability masks
    for the (optionally capped) training window."""

    # Recompute labels only from prices inside the inclusive training
    # anchor. Precomputed global targets at the tail may reference fold-
    # external t+1/t+2 prices and are intentionally never sliced here.
    train_open = trainer.loader.raw_data_cache["open"][
        stock_slice, :train_end_idx
    ].numpy()
    target_ret = causal_target_returns(
        train_open,
        trainer.loader.dates[:train_end_idx],
        policy,
        rebalance_mask=rebalance_mask,
    )
    realized_ret = open_to_open_returns(train_open)
    # Same semantics as loader.mask_by_universe, applied to the capped
    # window (the loader's mask is the full stock axis): values outside
    # the PIT eligibility mask become NaN.
    target_ret = np.asarray(target_ret, dtype=np.float64).copy()
    target_ret[~np.asarray(train_universe_mask, dtype=bool)] = np.nan
    # Tradability masks (buy/sell blocked per stock and date) align the
    # training basket with the backtest engine's execution rules; both
    # matrices are shared by every formula scored this run.
    blocked_buy, blocked_sell = trainer.loader.tradability_masks()
    blocked_buy = blocked_buy[stock_slice, :train_end_idx]
    blocked_sell = blocked_sell[stock_slice, :train_end_idx]
    return target_ret, realized_ret, blocked_buy, blocked_sell


def prepare_training_window(
    trainer,
    train_end_date: str | None,
    vm_device: torch.device,
    window_cap: tuple[int, int] | None = None,
) -> _TrainWindow:
    """Bind everything the searchers (RL, random, GP) need about the
    training window: the factor tensor, VM masks, target, windows, the
    semantic-budget cache and the calibration fingerprint executor.

    Shared by :meth:`train` (RL) and :meth:`train_search` (the
    non-RL backends) so every searcher measures the identical window.

    ``window_cap`` is a measurement override ``(stocks, dates)`` that
    slices the window head for tractable experiments (the admission
    tier): every searcher then measures the identical capped window,
    and the validation windows / signal range are re-derived inside
    the cap.  The cap is recorded by callers that persist results.

    B3 (IP-07b): moved verbatim from ``train.py`` and split into the
    contiguous stage helpers above; side-effect order (VM bindings,
    semantic-cache construction, calibration executor) is unchanged.
    """

    (
        contract,
        train_end_idx,
        stock_slice,
        train_signal_end,
        policy,
        rebalance_mask,
    ) = _window_bounds(trainer, train_end_date, window_cap)
    # Hold out the tail of the training window for out-of-sample best
    # formula selection, split into independent sub-windows; the best
    # formula is decided on the *median* validation reward so a single
    # lucky tail stretch cannot win the selection.
    val_windows = trainer._validation_windows(
        train_signal_end,
        rebalance_mask=rebalance_mask,
    )
    # The *learning* window is the in-sample head only: the policy
    # gradient scores candidates on columns strictly before the first
    # validation window, so the selection data never doubles as the
    # training signal (val rewards rank formulas; IS rewards teach the
    # policy — two jobs, two windows).
    val_start = validation_start(train_signal_end, trainer.model_config)
    train_signal_range = (contract.train_signal_start, val_start)
    factor_tensor = trainer.loader.factor_tensor[
        :, stock_slice, :train_end_idx
    ].to(vm_device)
    train_universe_mask = _bind_vm_window(
        trainer, vm_device, stock_slice, train_end_idx
    )
    target_ret, realized_ret, blocked_buy, blocked_sell = _window_targets(
        trainer,
        policy,
        stock_slice,
        train_end_idx,
        rebalance_mask,
        train_universe_mask,
    )
    reward_chunk = _bind_semantic_ledger(
        trainer,
        contract=contract,
        val_windows=val_windows,
        policy=policy,
        factor_tensor=factor_tensor,
        stock_slice=stock_slice,
        train_end_idx=train_end_idx,
    )
    return _TrainWindow(
        contract=contract,
        train_end_idx=train_end_idx,
        val_windows=val_windows,
        val_start=val_start,
        train_signal_range=train_signal_range,
        factor_tensor=factor_tensor,
        vm_device=vm_device,
        train_universe_mask=train_universe_mask,
        target_ret=target_ret,
        realized_ret=realized_ret,
        rebalance_mask=rebalance_mask,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        reward_chunk=reward_chunk,
    )


def _bind_semantic_ledger(
    trainer,
    *,
    contract,
    val_windows: list[tuple[int, int]],
    policy: RebalancePolicy,
    factor_tensor: torch.Tensor,
    stock_slice: slice,
    train_end_idx: int,
) -> int:
    """Bind the evaluation-budget ledger (the semantic cache), the
    calibration fingerprint executor and the streamed-reward chunk size
    onto the trainer; returns the chunk size (contiguous extraction of
    the historical ``prepare_window`` tail)."""

    # T2-01: the semantic cache is the evaluation-budget ledger.  Its
    # key carries the full evaluation context (dataset_id, reward and
    # protocol versions, the training window), so scores are never
    # reused across datasets or measurement generations, and its budget
    # counts **unique semantic evaluations** — structurally identical
    # (canonical AST hash) and numerically equivalent (calibration
    # fingerprint) formulas never bill twice.
    trainer.semantic_cache = SemanticCache(
        dataset_id=trainer.loader.dataset_id,
        reward_version=REWARD_VERSION,
        protocol_version=PROTOCOL_VERSION,
        window_id=trainer._window_id(
            contract, val_windows, policy, domain_id=trainer.domain_id
        ),
        cap=trainer._REWARD_CACHE_CAP,
    )
    calibration_slice = CalibrationSlice.of(factor_tensor.shape[2])
    trainer._calibration_execute = make_calibration_execute(
        trainer.vm,
        factor_tensor,
        trainer.loader.universe_mask[stock_slice, :train_end_idx],
        trainer.vm.industry_codes,
        calibration_slice,
    )
    # Chunk the batched reward evaluation so the stacked signal matrix
    # (original stack + batched copy + both-direction copy) stays within
    # a fixed memory budget (~512 MB of float64 signals).  The chunk size
    # resolves through the trainer facade (``_reward_chunk_size``) so the
    # ``monkeypatch.setattr(train_module, "score_chunk_size", ...)``
    # surface in tests/test_train.py keeps late-binding (B3, IP-07b).
    signal_bytes = factor_tensor.shape[1] * train_end_idx * 8
    return trainer._reward_chunk_size(signal_bytes)


class WindowPreparationMixin:
    """Window-binding seams of ``AshareTrainer`` (B3/B5, IP-07b): the
    facade entry points delegate to this module's functions and helpers.
    Methods stay class attributes of the composed trainer, so callers and
    class-attribute patches keep one stable surface."""

    def prepare_window(
        self,
        train_end_date: str | None,
        vm_device: torch.device,
        window_cap: tuple[int, int] | None = None,
    ) -> "_TrainWindow":
        """Delegate to :func:`prepare_training_window` (B3, IP-07b)."""

        return prepare_training_window(
            self, train_end_date, vm_device, window_cap
        )

    def _training_contract(
        self, train_end_date: str | None = None
    ) -> TrainingTimeContract:
        return TrainingTimeContract.resolve(
            self.loader.dates,
            train_end_date or self.backtest_config.train_end_date,
            horizon=self.backtest_config.target_horizon,
        )

    def _validation_start(self, train_end_idx: int) -> int:
        """First index of the validation tail inside the training window.

        Delegates to the module-level :func:`validation_start` so the
        protocol's random-search baseline can share the exact selection
        windows without instantiating a trainer.
        """
        return validation_start(train_end_idx, self.model_config)

    def _validation_windows(
        self,
        train_end_idx: int,
        *,
        rebalance_mask: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        """Independent validation sub-windows covering the training tail.

        Delegates to the module-level :func:`validation_windows`.
        """
        return validation_windows(
            train_end_idx,
            self.model_config,
            rebalance_mask=rebalance_mask,
        )

    @staticmethod
    def _window_id(
        contract,
        val_windows: list[tuple[int, int]],
        policy: RebalancePolicy,
        domain_id: str = "unified",
    ) -> str:
        """Deterministic id of the training window: the exact columns the
        semantic-cache key binds scores to.  A non-unified research domain
        (P6 §4.3) appends its id so domain scores never mix with unified
        or other-domain scores."""

        val = "|".join(f"{a}:{b}" for a, b in val_windows)
        domain = (
            ""
            if str(domain_id) == "unified"
            else f":domain:{str(domain_id)}"
        )
        return (
            f"train:{contract.train_signal_start}:{contract.train_signal_end}:"
            f"label:{contract.train_label_end}:frequency:{policy.frequency}:"
            f"horizon:{policy.horizon}:val:{val}{domain}"
        )
