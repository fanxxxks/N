"""Versioned contract shared by every formula-search backend.

The contract is deliberately independent of GP, Optuna and torch policy
implementations.  Importing the production GP path therefore does not pull in
experimental RL or optional TPE runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol, runtime_checkable

from .candidates import CandidateScore

# v2 (P7-E): the effective search space (legal formula set) is narrowed by
# semantic-type sampling legality (docs/p7_semantic_types_contract.md);
# results from different contract versions are never matched comparisons.
SEARCH_CONTRACT_VERSION = 2

TERMINATION_REASONS = frozenset(
    {
        "budget_exhausted",
        "steps_exhausted",
        "proposal_stagnation",
        "candidate_pool_exhausted",
        "no_eligible_candidate",
        "backend_error",
    }
)


@dataclass(frozen=True)
class SearchRequest:
    """One backend-independent search request.

    ``budget`` is always the requested count of unique semantic formula
    evaluations.  RL may additionally expose its step/batch split, but that
    split cannot change the budget.
    """

    seed: int
    budget: int
    max_formula_len: int
    steps: int | None = None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        if int(self.budget) <= 0:
            raise ValueError("budget must be positive")
        if int(self.max_formula_len) < 2:
            raise ValueError("max_formula_len must be at least 2")
        split = (self.steps is not None, self.batch_size is not None)
        if split[0] != split[1]:
            raise ValueError("steps and batch_size must either both be set or both omitted")
        if self.steps is not None:
            if int(self.steps) <= 0 or int(self.batch_size) <= 0:
                raise ValueError("steps and batch_size must be positive")
            if int(self.steps) * int(self.batch_size) != int(self.budget):
                raise ValueError("steps and batch_size must multiply to budget")


@dataclass(frozen=True)
class SearchResult:
    """The sole result schema returned by GP, TPE, Random and RL."""

    backend: str
    seed: int
    requested_budget: int
    consumed_budget: int
    termination_reason: str
    stagnation_reason: str | None = None
    best_so_far: tuple[tuple[int, float], ...] = ()
    scores: tuple[CandidateScore, ...] = ()
    selected: CandidateScore | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    proposal_count: int = 0
    invalid_proposals: int = 0
    semantic_duplicates: int = 0
    elite_archive: Any | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    contract_version: int = SEARCH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        requested = int(self.requested_budget)
        consumed = int(self.consumed_budget)
        if requested <= 0:
            raise ValueError("requested_budget must be positive")
        if consumed < 0:
            raise ValueError("consumed_budget must be non-negative")
        if consumed > requested:
            raise ValueError("consumed budget cannot exceed requested budget")
        if self.termination_reason not in TERMINATION_REASONS:
            raise ValueError(
                f"unknown termination_reason {self.termination_reason!r}; "
                f"expected one of {sorted(TERMINATION_REASONS)}"
            )
        if self.termination_reason == "proposal_stagnation":
            if not self.stagnation_reason:
                raise ValueError(
                    "proposal_stagnation requires a non-empty stagnation_reason"
                )
        elif self.stagnation_reason is not None:
            raise ValueError(
                "stagnation_reason is only valid with proposal_stagnation"
            )
        if consumed == 0 and self.best_so_far:
            raise ValueError("zero consumed budget requires an empty best-so-far curve")

        previous_x = 0
        previous_reward = -math.inf
        normalized: list[tuple[int, float]] = []
        for raw_x, raw_reward in self.best_so_far:
            x = int(raw_x)
            reward = float(raw_reward)
            if x <= previous_x:
                raise ValueError("best_so_far budget coordinates must be strictly increasing")
            if x > consumed:
                raise ValueError("best_so_far cannot extend beyond consumed budget")
            if math.isnan(reward) or reward < previous_reward:
                raise ValueError("best_so_far reward must be non-decreasing")
            normalized.append((x, reward))
            previous_x = x
            previous_reward = reward
        if consumed > 0:
            if not normalized or normalized[0][0] != 1:
                raise ValueError("best_so_far must start at consumed budget 1")
            if normalized[-1][0] != consumed:
                raise ValueError("best_so_far must end at consumed budget")
        object.__setattr__(self, "best_so_far", tuple(normalized))
        if int(self.proposal_count) < 0:
            raise ValueError("proposal_count must be non-negative")
        if int(self.invalid_proposals) < 0 or int(self.semantic_duplicates) < 0:
            raise ValueError("proposal diagnostic counts must be non-negative")

    # Compatibility accessors for callers of the former
    # BaselineHarnessResult.  There remains one result type and one source of
    # truth; these names only make the v1 in-process migration explicit.
    @property
    def budget(self) -> int:
        return self.requested_budget

    @property
    def n_evaluated(self) -> int:
        return self.consumed_budget

    @property
    def n_invalid(self) -> int:
        return self.invalid_proposals

    @property
    def n_semantic_dedups(self) -> int:
        return self.semantic_duplicates

    @property
    def rejections(self) -> dict[str, int]:
        return self.rejection_reasons

    def to_dict(self, *, include_scores: bool = False) -> dict[str, object]:
        archive = self.elite_archive
        payload: dict[str, object] = {
            "contract_version": self.contract_version,
            "backend": self.backend,
            "seed": self.seed,
            "requested_budget": self.requested_budget,
            "consumed_budget": self.consumed_budget,
            "termination_reason": self.termination_reason,
            "stagnation_reason": self.stagnation_reason,
            "best_so_far": [[x, reward] for x, reward in self.best_so_far],
            "selected": self.selected.to_dict() if self.selected else None,
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "proposal_count": self.proposal_count,
            "invalid_proposals": self.invalid_proposals,
            "semantic_duplicates": self.semantic_duplicates,
            "elite_archive": (
                archive.to_dict() if hasattr(archive, "to_dict") else archive
            ),
            "diagnostics": self.diagnostics,
            "n_scores": len(self.scores),
        }
        if include_scores:
            payload["scores"] = [score.to_dict() for score in self.scores]
        return payload


@runtime_checkable
class SearchBackend(Protocol):
    """Stateless backend interface used by the trainer and experiments."""

    name: str

    def search(
        self,
        request: SearchRequest,
        evaluator: object | None,
        **backend_context: object,
    ) -> SearchResult:
        ...
