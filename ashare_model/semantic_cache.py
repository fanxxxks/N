"""Semantic cache and the unique-semantic-evaluation budget (T2-01).

Every searcher in Phase 2 (random, GP, TPE, RL) is billed in **unique
semantic formula evaluations**: a formula that was already evaluated —
structurally (same canonical AST hash) or semantically (same rank
fingerprint on the fixed calibration slice **and** the same complexity
bill, so the score including its complexity penalty is exactly reusable)
— never consumes budget again.

The cache key carries the canonical AST hash plus the full evaluation
context (``dataset_id``, reward version, protocol version, window), so a
score is never reused across datasets or measurement generations.

The semantic fingerprint is computed on the **fixed calibration slice**
(the head of the training window: all stocks, first
:data:`CALIBRATION_DATES` columns, a deterministic rule shared by every
searcher): the per-date cross-sectional rank pattern, quantized to 8
levels and hashed.  Two formulas with the same fingerprint rank every
cross-section identically on the slice — the exact property the reward
and the basket selection consume — so they are numerically equivalent
for evaluation purposes (e.g. ``x`` and ``ADD(x, x)``), and the
complexity bill keeps their net-of-penalty scores exact.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import numpy as np
import torch

from .ir import canonical_ast, structural_hash
from .vocab import FormulaVocab, FORMULA_VOCAB

# Bump when the fingerprint computation or the key layout changes: scores
# cached under an older version are not comparable with a newer one.
SEMANTIC_CACHE_VERSION = 1

# Fixed calibration slice: the head of the training window (all stocks,
# first CALIBRATION_DATES columns).  All stocks are included so the
# cross-sectional operators (CS_RANK / CS_ZSCORE / CS_NEUTRALIZE) rank
# inside the same cross-section as the full evaluation.
CALIBRATION_DATES = 60

_FINGERPRINT_LEVELS = 8


@dataclass(frozen=True)
class CalibrationSlice:
    """The fixed calibration window formulas are fingerprinted on."""

    date_start: int
    date_end: int

    @classmethod
    def of(cls, n_dates: int) -> "CalibrationSlice":
        return cls(0, min(CALIBRATION_DATES, n_dates))

    def __str__(self) -> str:
        return f"dates:{self.date_start}:{self.date_end}"


@dataclass(frozen=True)
class SemanticCacheKey:
    """Cache key: canonical AST hash plus the full evaluation context."""

    structural_hash: str
    dataset_id: str | None
    reward_version: int
    protocol_version: int
    window_id: str


T = TypeVar("T")


def _quantized_ranks(signal: np.ndarray) -> np.ndarray:
    """Per-date cross-sectional ranks quantized to 8 levels (uint8).

    Only finite cells rank; non-finite cells stay 0.  Ties keep their
    stable-sort positions (consecutive ranks), which is deterministic.
    A date with fewer than two finite cells contributes an all-zero
    column, so degenerate signals collapse to one semantic class.
    """

    out = np.zeros(signal.shape, dtype=np.uint8)
    for d in range(signal.shape[1]):
        col = signal[:, d]
        finite = np.isfinite(col)
        n = int(finite.sum())
        if n < 2:
            continue
        order = np.argsort(col, kind="stable")
        ranks = np.empty(col.size, dtype=np.float64)
        ranks[order] = np.arange(col.size)
        normalized = ranks[finite] / (n - 1)
        out[finite, d] = np.clip(
            np.floor(normalized * _FINGERPRINT_LEVELS), 0, _FINGERPRINT_LEVELS - 1
        ).astype(np.uint8)
    return out


class SemanticCache(Generic[T]):
    """Bounded cache of evaluated formulas plus budget accounting.

    ``get``/``put`` operate on :class:`SemanticCacheKey` (canonical AST
    hash + context).  ``put`` consumes one budget unit exactly when the
    entry's semantic class (fingerprint + complexity bill) was never
    evaluated before; re-putting a known class (a numerically equivalent,
    identically-billed formula under a different AST) counts as a semantic
    dedup and costs nothing.  Entries are LRU-bounded; eviction removes
    the class mapping but never refunds budget — budget counts evaluations
    ever performed.
    """

    def __init__(
        self,
        *,
        dataset_id: str | None,
        reward_version: int,
        protocol_version: int,
        window_id: str,
        cap: int = 16384,
        vocab: FormulaVocab | None = None,
    ):
        if cap <= 0:
            raise ValueError("cap must be positive")
        self._dataset_id = dataset_id
        self._reward_version = int(reward_version)
        self._protocol_version = int(protocol_version)
        self._window_id = window_id
        self._cap = int(cap)
        self._vocab = vocab or FORMULA_VOCAB
        self._entries: OrderedDict[SemanticCacheKey, T] = OrderedDict()
        # fingerprint -> key of the first evaluation of that semantic class
        self._fingerprints: dict[str, SemanticCacheKey] = {}
        self._budget_used = 0
        self._n_hits = 0
        self._n_semantic_dedups = 0

    # --- keys ------------------------------------------------------------

    def key_for(
        self, tokens, vocab: FormulaVocab | None = None
    ) -> SemanticCacheKey | None:
        """Canonical cache key of a token sequence.

        Returns ``None`` for structurally invalid or degenerate
        (constant-producing) formulas — they are never evaluated and never
        consume budget.
        """

        vocab = vocab or self._vocab
        ast = canonical_ast(tokens, vocab)
        if ast is None:
            return None
        return SemanticCacheKey(
            structural_hash=structural_hash(ast),
            dataset_id=self._dataset_id,
            reward_version=self._reward_version,
            protocol_version=self._protocol_version,
            window_id=self._window_id,
        )

    # --- fingerprint ------------------------------------------------------

    @staticmethod
    def fingerprint(
        tokens,
        execute: Callable[[tuple[int, ...]], np.ndarray | None],
        vocab: FormulaVocab | None = None,
    ) -> str | None:
        """Semantic fingerprint on the calibration slice (12 hex chars).

        ``execute`` evaluates the token sequence on the calibration slice
        and returns a ``[stock, date]`` signal or ``None`` (execution
        failure).  The fingerprint is the hash of the per-date quantized
        cross-sectional ranks: monotone-equivalent signals share it, sign
        flips do not, and every constant signal collapses to one class.
        """

        signal = execute(tokens)
        if signal is None:
            return None
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        quantized = _quantized_ranks(arr)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:12]

    # --- entries ----------------------------------------------------------

    def get(self, key: SemanticCacheKey) -> T | None:
        score = self._entries.get(key)
        if score is not None:
            self._n_hits += 1
            self._entries.move_to_end(key)
        return score

    # --- semantic classes --------------------------------------------------

    @staticmethod
    def _class_key(fingerprint: str, complexity_bill: float | None) -> str:
        if complexity_bill is None:
            return fingerprint
        return f"{fingerprint}:{float(complexity_bill):.6f}"

    def put(
        self,
        key: SemanticCacheKey,
        score: T,
        fingerprint: str | None = None,
        complexity_bill: float | None = None,
    ) -> None:
        """Store an evaluated score.

        Consumes one budget unit only when ``fingerprint`` is not ``None``
        and no previous evaluation registered that semantic class (the
        fingerprint **plus the complexity bill**: two numerically
        equivalent formulas with different bills carry different penalties,
        so their scores are not interchangeable).  Re-putting an existing
        key overwrites its value (callers may store a placeholder first and
        the real score once scoring returns).
        """

        if key in self._entries:
            self._entries[key] = score
            self._entries.move_to_end(key)
            return
        if fingerprint is not None:
            class_key = self._class_key(fingerprint, complexity_bill)
            if class_key in self._fingerprints:
                self._n_semantic_dedups += 1
            else:
                self._fingerprints[class_key] = key
                self._budget_used += 1
        self._entries[key] = score
        while len(self._entries) > self._cap:
            evicted_key, _ = self._entries.popitem(last=False)
            for fp, owner in list(self._fingerprints.items()):
                if owner == evicted_key:
                    del self._fingerprints[fp]

    def is_claimed(
        self, fingerprint: str, complexity_bill: float | None = None
    ) -> bool:
        """True when the semantic class is already registered — evaluated,
        or claimed by an in-flight evaluation whose score has not landed
        yet (callers skip the duplicate evaluation entirely)."""

        return self._class_key(fingerprint, complexity_bill) in self._fingerprints

    def score_by_fingerprint(
        self, fingerprint: str, complexity_bill: float | None = None
    ) -> T | None:
        """Score of the first evaluation of a semantic class, if cached."""
        class_key = self._class_key(fingerprint, complexity_bill)
        key = self._fingerprints.get(class_key)
        if key is None:
            return None
        return self._entries.get(key)

    # --- accounting -------------------------------------------------------

    @property
    def budget_used(self) -> int:
        """Unique semantic evaluations performed (the Phase 2 budget)."""
        return self._budget_used

    @property
    def n_hits(self) -> int:
        return self._n_hits

    @property
    def n_semantic_dedups(self) -> int:
        return self._n_semantic_dedups

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int | str]:
        return {
            "budget_used": self._budget_used,
            "n_hits": self._n_hits,
            "n_semantic_dedups": self._n_semantic_dedups,
            "n_entries": len(self._entries),
            "cap": self._cap,
            "dataset_id": self._dataset_id,
            "reward_version": self._reward_version,
            "protocol_version": self._protocol_version,
            "window_id": self._window_id,
            "version": SEMANTIC_CACHE_VERSION,
        }


def make_calibration_execute(
    vm,
    factors: torch.Tensor,
    universe_mask: np.ndarray,
    industry_codes: torch.Tensor | None,
    calibration_slice: CalibrationSlice,
) -> Callable[[tuple[int, ...]], np.ndarray | None]:
    """Bind a VM to the calibration slice: ``execute(tokens) -> signal``.

    The factor tensor is date-sliced to the calibration window (all stocks,
    so the cross-sectional operators rank inside the same cross-section as
    the full evaluation); the VM's universe mask and industry codes are
    swapped for the matching slices for the duration of the call and
    restored afterwards, so the caller's full-window state is untouched.
    """

    date_slice = slice(calibration_slice.date_start, calibration_slice.date_end)
    cal_factors = factors[:, :, date_slice]
    device = factors.device
    cal_mask = torch.tensor(
        universe_mask[:, date_slice], dtype=torch.bool, device=device
    )
    cal_industry = (
        industry_codes[:, date_slice].to(device)
        if industry_codes is not None
        else None
    )

    def execute(tokens: tuple[int, ...]) -> np.ndarray | None:
        prev_mask, prev_industry = vm.universe_mask, vm.industry_codes
        vm.universe_mask, vm.industry_codes = cal_mask, cal_industry
        try:
            signal = vm.execute(tokens, cal_factors)
        finally:
            vm.universe_mask, vm.industry_codes = prev_mask, prev_industry
        if signal is None:
            return None
        return signal.detach().cpu().numpy()

    return execute
