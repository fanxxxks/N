"""Single candidate scoring and deterministic selection contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, Sequence

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig

from .reward import batched_basket_rewards

# Peak transient of one scoring chunk: ``score_many`` holds the original
# signal stack (1x), the stacked batch (1x) and the interleaved
# both-direction copy (2x) at once, i.e. ~4x one signal.
_SCORE_CHUNK_BUDGET_BYTES = 512 * (1 << 20)
_SCORE_CHUNK_SIGNALS_MULTIPLIER = 4


def score_chunk_size(signal_bytes: int) -> int:
    """Formulas per scoring chunk under the ~512 MB float64 budget.

    ``signal_bytes`` is the byte size of one candidate signal
    (``stocks x dates x 8``).  The chunk keeps ``4x`` one signal inside the
    budget so the stacked batch, the both-direction copy and the caller's
    pending originals never exceed it together; it is capped at 64 (tiny
    windows would otherwise build huge chunk counts) and floored at 1 so a
    single oversized signal still makes progress.
    """

    return max(
        1,
        min(
            64,
            _SCORE_CHUNK_BUDGET_BYTES
            // max(_SCORE_CHUNK_SIGNALS_MULTIPLIER * int(signal_bytes), 1),
        ),
    )


@dataclass(frozen=True)
class CandidateSpec:
    """Stable candidate identity independent of how it was generated."""

    candidate_id: str
    formula_text: str
    source: str
    tokens: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.tokens is not None and not isinstance(self.tokens, tuple):
            object.__setattr__(self, "tokens", tuple(int(t) for t in self.tokens))

    @property
    def deterministic_key(self) -> str:
        if self.tokens is not None:
            return ",".join(f"{int(token):08d}" for token in self.tokens)
        return self.candidate_id


@dataclass(frozen=True)
class CandidateScore:
    tokens: tuple[int, ...] | None
    candidate_id: str
    formula_text: str
    source: str
    direction: int
    val_reward: float
    val_icir: float
    train_reward: float
    train_icir: float
    complexity_penalty: float
    eligible: bool
    rejection_reasons: tuple[str, ...]

    @property
    def deterministic_key(self) -> str:
        if self.tokens is not None:
            return ",".join(f"{int(token):08d}" for token in self.tokens)
        return self.candidate_id

    def to_dict(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return float(value) if math.isfinite(float(value)) else None

        return {
            "tokens": list(self.tokens) if self.tokens is not None else None,
            "candidate_id": self.candidate_id,
            "formula_text": self.formula_text,
            "source": self.source,
            "direction": self.direction,
            "val_reward": finite_or_none(self.val_reward),
            "val_icir": finite_or_none(self.val_icir),
            "train_reward": finite_or_none(self.train_reward),
            "train_icir": finite_or_none(self.train_icir),
            "complexity_penalty": self.complexity_penalty,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_payload(cls, payload: dict, *, source: str = "artifact") -> "CandidateScore":
        """Read explicit v6 fields, with a one-way legacy artifact fallback."""

        def number(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return math.nan

        tokens = payload.get("formula", payload.get("tokens"))
        val_reward = payload.get("val_reward", payload.get("best_reward"))
        # v10 renamed full_window_* to train_*; earlier artifacts (and the
        # pre-v6 names) resolve through the same fallback chain.
        train_icir = payload.get(
            "train_icir",
            payload.get("full_window_icir", payload.get("best_icir")),
        )
        reasons = tuple(str(reason) for reason in payload.get("rejection_reasons", []))
        return cls(
            tokens=tuple(int(t) for t in tokens) if tokens is not None else None,
            candidate_id=str(payload.get("candidate_id", "artifact")),
            formula_text=str(payload.get("formula_text", "")),
            source=str(payload.get("source", source)),
            direction=int(payload.get("direction", 1)),
            val_reward=number(val_reward),
            val_icir=number(payload.get("val_icir")),
            train_reward=number(
                payload.get("train_reward", payload.get("full_window_reward"))
            ),
            train_icir=number(train_icir),
            complexity_penalty=number(payload.get("complexity_penalty", 0.0)),
            eligible=bool(payload.get("eligible", not reasons)),
            rejection_reasons=reasons,
        )


@dataclass(frozen=True)
class SelectionResult:
    selected: CandidateScore | None
    best_rejected: CandidateScore | None
    candidates: tuple[CandidateScore, ...]

    @property
    def eligible_count(self) -> int:
        return sum(score.eligible for score in self.candidates)

    def to_dict(self, *, compact: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "selected": self.selected.to_dict() if self.selected else None,
            "best_rejected": (
                self.best_rejected.to_dict() if self.best_rejected else None
            ),
            "candidate_count": len(self.candidates),
            "eligible_count": self.eligible_count,
        }
        if not compact:
            payload["candidates"] = [score.to_dict() for score in self.candidates]
        return payload


class CandidateSelector:
    """Eligibility-first deterministic candidate selector."""

    @staticmethod
    def _metric(value: float) -> float:
        return float(value) if math.isfinite(float(value)) else -math.inf

    @classmethod
    def _sort_key(cls, score: CandidateScore) -> tuple[float, float, float, float, str]:
        return (
            -cls._metric(score.val_reward),
            -cls._metric(score.val_icir),
            -cls._metric(score.train_reward),
            -cls._metric(score.train_icir),
            score.deterministic_key,
        )

    def select(self, scores: Iterable[CandidateScore]) -> SelectionResult:
        candidates = tuple(scores)
        eligible = sorted((s for s in candidates if s.eligible), key=self._sort_key)
        rejected = sorted((s for s in candidates if not s.eligible), key=self._sort_key)
        return SelectionResult(
            selected=eligible[0] if eligible else None,
            best_rejected=rejected[0] if rejected else None,
            candidates=candidates,
        )


class CandidateScorer:
    """Direction-symmetric scoring shared by RL, random search and baselines."""

    def __init__(
        self,
        backtest_config: BacktestConfig,
        reward_config: RewardConfig,
        *,
        near_constant_threshold: float = 1e-4,
        reward_function: Callable = batched_basket_rewards,
    ) -> None:
        self.backtest_config = backtest_config
        self.reward_config = reward_config
        self.near_constant_threshold = float(near_constant_threshold)
        self.reward_function = reward_function

    def complexity_penalty(self, spec: CandidateSpec) -> float:
        """Penalty for bare single-factor formulas (no operator anywhere).

        The AST is the single source of truth: the bytecode is decoded once
        and the penalty applies exactly when the AST is a bare
        :class:`~ashare_model.ir.Feature` — EOS and padding tokens never
        count as operators.
        """

        from .ir import FormulaSyntaxError, decode, operator_names

        tokens = spec.tokens or ()
        if not tokens:
            return float(self.reward_config.complexity_penalty)
        try:
            ast = decode(tokens)
        except FormulaSyntaxError:
            return float(self.reward_config.complexity_penalty)
        if operator_names(ast):
            return 0.0
        return float(self.reward_config.complexity_penalty)

    def score(
        self,
        spec: CandidateSpec,
        signal: np.ndarray | None,
        target_ret: np.ndarray,
        val_windows: list[tuple[int, int]],
        *,
        universe_mask: np.ndarray,
        blocked_buy: np.ndarray | None = None,
        blocked_sell: np.ndarray | None = None,
        formula_valid: bool = True,
        train_signal_range: tuple[int, int] | None = None,
    ) -> CandidateScore:
        return self.score_many(
            [spec],
            [signal],
            target_ret,
            val_windows,
            universe_mask=universe_mask,
            blocked_buy=blocked_buy,
            blocked_sell=blocked_sell,
            formula_valid=[formula_valid],
            train_signal_range=train_signal_range,
        )[0]

    def score_many(
        self,
        specs: Sequence[CandidateSpec],
        signals: Sequence[np.ndarray | None],
        target_ret: np.ndarray,
        val_windows: list[tuple[int, int]],
        *,
        universe_mask: np.ndarray,
        blocked_buy: np.ndarray | None = None,
        blocked_sell: np.ndarray | None = None,
        formula_valid: Sequence[bool] | None = None,
        train_signal_range: tuple[int, int] | None = None,
    ) -> list[CandidateScore]:
        """Score a batch of candidates under one PIT eligibility mask.

        ``universe_mask`` (``[stock, date]`` bool, aligned with
        ``target_ret``) is mandatory — it cannot be inferred from NaN VM
        outputs because bare-factor signals never pass through the VM.  The
        mask gates every quality decision — the near-constant rejection, the
        reward function (both directions share the identical mask) and the
        direction tie-break — so a future member's extreme finite values
        can neither move a score nor flip a direction or rejection reason
        before its join day.

        ``train_signal_range`` is the caller's *learning* window (the
        primary scoring pass): the trainer passes the in-sample window
        that ends where the validation tail begins, so the reward that
        feeds the policy gradient never reads the selection data; the
        protocol passes the same IS boundary so artifacts carry one
        uniform window semantics.  The near-constant rejection scans the
        same window: a formula constant on the learning window carries no
        gradient signal even if it varies on the validation tail.
        """

        if len(specs) != len(signals):
            raise ValueError("specs and signals must have the same length")
        if not val_windows:
            raise ValueError("candidate scoring requires at least one validation window")
        valid_flags = list(formula_valid or [True] * len(specs))
        if len(valid_flags) != len(specs):
            raise ValueError("formula_valid must align with specs")
        target = np.asarray(target_ret, dtype=np.float64)
        if target.ndim != 2:
            raise ValueError("target_ret must be [stock, date]")
        universe_mask = np.asarray(universe_mask, dtype=bool)
        if universe_mask.shape != target.shape:
            raise ValueError(
                f"universe_mask shape {universe_mask.shape} does not match "
                f"target shape {target.shape}"
            )
        if train_signal_range is None:
            train_signal_range = (0, max(target.shape[1] - 2, 0))
        train_start, train_end = train_signal_range
        scoring_ref = universe_mask[:, train_start:train_end]

        results: list[CandidateScore | None] = [None] * len(specs)
        batch_indices: list[int] = []
        batch_signals: list[np.ndarray] = []
        for index, (spec, signal, formula_is_valid) in enumerate(
            zip(specs, signals, valid_flags)
        ):
            if not formula_is_valid or signal is None:
                results[index] = self._rejected_score(
                    spec, "invalid_formula", self.reward_config.reward_clip_low
                )
                continue
            array = np.asarray(signal, dtype=np.float64)
            if array.shape != target.shape:
                raise ValueError(
                    f"signal shape {array.shape} does not match target {target.shape}"
                )
            scoring_values = array[:, train_start:train_end]
            ref = np.isfinite(scoring_values) & scoring_ref
            finite = scoring_values[ref]
            if finite.size == 0 or float(np.std(finite)) < self.near_constant_threshold:
                results[index] = self._rejected_score(
                    spec,
                    "constant_or_near_constant_signal",
                    self.reward_config.bad_reward,
                )
                continue
            batch_indices.append(index)
            batch_signals.append(array)

        if batch_signals:
            raw = np.stack(batch_signals)
            # Interleaving keeps each candidate's +signal and -signal scores
            # adjacent and makes mirror invariance straightforward to audit.
            oriented = np.empty((raw.shape[0] * 2, *raw.shape[1:]), dtype=np.float64)
            oriented[0::2] = raw
            oriented[1::2] = -raw
            rewards, val_rewards, icir, val_icir = self.reward_function(
                oriented,
                target,
                self.backtest_config,
                self.reward_config,
                val_windows,
                blocked_buy=blocked_buy,
                blocked_sell=blocked_sell,
                train_signal_range=train_signal_range,
                universe_mask=universe_mask,
            )
            if val_rewards is None or val_icir is None:
                # A reward function that drops the validation results is a
                # programming error, not a candidate outcome — never let
                # this disappear under python -O (a bare assert would).
                raise RuntimeError(
                    "reward function returned no validation rewards "
                    "despite non-empty val_windows"
                )
            for batch_index, result_index in enumerate(batch_indices):
                spec = specs[result_index]
                plus = batch_index * 2
                minus = plus + 1
                direction = self._choose_direction(
                    raw[batch_index],
                    val_windows,
                    float(val_rewards[plus]),
                    float(val_icir[plus]),
                    float(val_rewards[minus]),
                    float(val_icir[minus]),
                    universe_mask,
                )
                chosen = plus if direction == 1 else minus
                penalty = self.complexity_penalty(spec)
                score = CandidateScore(
                    tokens=spec.tokens,
                    candidate_id=spec.candidate_id,
                    formula_text=spec.formula_text,
                    source=spec.source,
                    direction=direction,
                    val_reward=float(val_rewards[chosen]) - penalty,
                    val_icir=float(val_icir[chosen]),
                    train_reward=float(rewards[chosen]) - penalty,
                    train_icir=float(icir[chosen]),
                    complexity_penalty=penalty,
                    eligible=False,
                    rejection_reasons=(),
                )
                results[result_index] = self._apply_eligibility(score)

        return [score for score in results if score is not None]

    def _apply_eligibility(self, score: CandidateScore) -> CandidateScore:
        reasons: list[str] = []
        if not math.isfinite(score.val_reward):
            reasons.append("val_reward_not_finite")
        if not math.isfinite(score.val_icir):
            reasons.append("val_icir_not_finite")
        if math.isfinite(score.val_reward) and score.val_reward < float(
            self.reward_config.min_val_reward
        ):
            reasons.append("val_reward_below_minimum")
        if math.isfinite(score.val_icir) and score.val_icir < float(
            self.reward_config.min_val_icir
        ):
            reasons.append("val_icir_below_minimum")
        return replace(
            score,
            eligible=not reasons,
            rejection_reasons=tuple(reasons),
        )

    def _rejected_score(
        self,
        spec: CandidateSpec,
        reason: str,
        training_reward: float,
    ) -> CandidateScore:
        return CandidateScore(
            tokens=spec.tokens,
            candidate_id=spec.candidate_id,
            formula_text=spec.formula_text,
            source=spec.source,
            direction=1,
            val_reward=math.nan,
            val_icir=math.nan,
            train_reward=float(training_reward),
            train_icir=math.nan,
            complexity_penalty=self.complexity_penalty(spec),
            eligible=False,
            rejection_reasons=(reason,),
        )

    @staticmethod
    def _choose_direction(
        signal: np.ndarray,
        val_windows: list[tuple[int, int]],
        plus_reward: float,
        plus_icir: float,
        minus_reward: float,
        minus_icir: float,
        universe_mask: np.ndarray,
    ) -> int:
        plus_key = (
            plus_reward if math.isfinite(plus_reward) else -math.inf,
            plus_icir if math.isfinite(plus_icir) else -math.inf,
        )
        minus_key = (
            minus_reward if math.isfinite(minus_reward) else -math.inf,
            minus_icir if math.isfinite(minus_icir) else -math.inf,
        )
        if plus_key > minus_key:
            return 1
        if minus_key > plus_key:
            return -1
        # Mirror-stable canonical orientation: make the first finite non-zero
        # validation observation positive.  Mirroring the input therefore
        # flips direction while leaving the applied signal identical.  Only
        # signal-date eligible validation observations are scanned, so a
        # future member's values can never decide the canonical orientation.
        for start, end in val_windows:
            values = np.asarray(signal[:, start:end])
            values = np.where(universe_mask[:, start:end], values, np.nan)
            for value in values.T.ravel():
                if math.isfinite(float(value)) and float(value) != 0.0:
                    return 1 if float(value) > 0.0 else -1
        return 1
