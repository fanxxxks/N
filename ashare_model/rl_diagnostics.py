"""Pure, serializable diagnostics for the experimental RL searcher."""

from __future__ import annotations

import math
from typing import Iterable

import torch

from .candidates import CandidateScore

RL_DIAGNOSTICS_VERSION = 1


def reward_distribution(values: torch.Tensor) -> dict[str, float | int]:
    """Population distribution used for rewards and run-level metrics."""

    flat = torch.as_tensor(values, dtype=torch.float64).detach().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("diagnostic distribution requires at least one value")
    if not torch.isfinite(flat).all():
        raise ValueError("diagnostic distribution values must be finite")
    quantiles = torch.quantile(
        flat, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    )
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
    }


def rejection_reason_counts(
    scores: Iterable[CandidateScore],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for score in scores:
        for reason in score.rejection_reasons:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items()))


def gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Global L2 norm without clipping or mutating any gradient."""

    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(dtype=torch.float64)
        squared += float(torch.sum(grad * grad))
    return math.sqrt(squared)


def _formula_length_summary(lengths: Iterable[int]) -> dict[str, int | float | None]:
    values = [int(length) for length in lengths]
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": float(sum(values) / len(values)),
        "max": max(values),
    }


def summarize_rl_step(
    *,
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    entropy: float,
    gradient_norm: float,
    scores: Iterable[CandidateScore],
    formula_lengths: Iterable[int],
    operator_names: Iterable[str],
    semantic_duplicates: int,
    proposal_count: int,
) -> dict[str, object]:
    """Build the exact P4 per-step diagnostic schema."""

    proposal_count = int(proposal_count)
    semantic_duplicates = int(semantic_duplicates)
    if proposal_count <= 0:
        raise ValueError("proposal_count must be positive")
    if not 0 <= semantic_duplicates <= proposal_count:
        raise ValueError("semantic_duplicates must be within proposal_count")
    advantage = torch.as_tensor(advantages, dtype=torch.float64).detach().reshape(-1)
    if advantage.numel() == 0 or not torch.isfinite(advantage).all():
        raise ValueError("advantages must be a non-empty finite tensor")
    return {
        "version": RL_DIAGNOSTICS_VERSION,
        "reward_distribution": reward_distribution(rewards),
        "rejection_reasons": rejection_reason_counts(scores),
        "entropy": float(entropy),
        "semantic_duplicates": semantic_duplicates,
        "semantic_duplicate_rate": float(semantic_duplicates / proposal_count),
        "advantage_variance": float(advantage.var(unbiased=False)),
        "gradient_norm": float(gradient_norm),
        "formula_length": _formula_length_summary(formula_lengths),
        "operator_coverage": sorted(set(str(name) for name in operator_names)),
    }


def aggregate_rl_run(
    *,
    reward_values: Iterable[float],
    step_summaries: Iterable[dict[str, object]],
    rejection_reasons: dict[str, int],
    formula_lengths: Iterable[int],
    operator_names: Iterable[str],
    semantic_duplicates: int,
    proposal_count: int,
) -> dict[str, object]:
    """Aggregate exact run-level evidence without changing training state."""

    steps = tuple(step_summaries)
    rewards = torch.tensor(list(reward_values), dtype=torch.float64)
    entropy = [float(step["entropy"]) for step in steps]
    advantage_variance = [float(step["advantage_variance"]) for step in steps]
    gradient_norms = [float(step["gradient_norm"]) for step in steps]
    proposal_count = int(proposal_count)
    semantic_duplicates = int(semantic_duplicates)
    return {
        "version": RL_DIAGNOSTICS_VERSION,
        "reward_distribution": reward_distribution(rewards),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "mean_entropy": float(sum(entropy) / len(entropy)) if entropy else None,
        "semantic_duplicates": semantic_duplicates,
        "semantic_duplicate_rate": (
            float(semantic_duplicates / proposal_count) if proposal_count else 0.0
        ),
        "mean_advantage_variance": (
            float(sum(advantage_variance) / len(advantage_variance))
            if advantage_variance
            else None
        ),
        "gradient_norm": (
            {
                "min": min(gradient_norms),
                "mean": float(sum(gradient_norms) / len(gradient_norms)),
                "max": max(gradient_norms),
            }
            if gradient_norms
            else {"min": None, "mean": None, "max": None}
        ),
        "formula_length": _formula_length_summary(formula_lengths),
        "operator_coverage": sorted(set(str(name) for name in operator_names)),
        "steps": list(steps),
    }
