"""Matched-budget baseline harness (T1-05).

The harness answers the phase-1 research-validity questions with the
shared production measurement path (scorer + engine), not a proxy:

* :func:`run_matched_baseline` gives the random-search baseline the
  **same evaluation budget** as a trained candidate — ``steps x
  batch_size`` unique formula evaluations — so RL-vs-baseline
  comparisons are budget-fair (the protocol's fixed ``random_samples``
  knob is replaced by the matched budget unless explicitly disabled).
* :func:`oos_active_ir` reduces a strategy's OOS excess series to its
  annualized active IR — the same quantity the protocol's trial matrix
  uses for the DS/max-t corrections.
* :func:`reward_oos_correlation` measures whether the training reward
  orders candidates the way the OOS active IR does (Spearman rho with a
  deterministic permutation p-value).  A stable positive correlation is
  the phase-1 completion gate: the reward must be a faithful proxy of
  out-of-sample portfolio quality, not a self-referential in-sample
  number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig

from .candidates import (
    CandidateScore,
    CandidateScorer,
    CandidateSelector,
    CandidateSpec,
)
from .train import sample_random_formulas
from .vocab import FORMULA_VOCAB

_ANNUALIZATION = 252
_CAP = 1e9


def oos_active_ir(
    daily_returns: np.ndarray,
    benchmark_daily_returns: np.ndarray,
) -> float:
    """Annualized active IR of one OOS excess series.

    ``mean(excess) / std(excess) * sqrt(252)``; a constant non-zero
    excess has an unbounded ratio (capped finite); fewer than two
    observations score 0.0.
    """

    strat = np.asarray(daily_returns, dtype=np.float64)
    bench = np.asarray(benchmark_daily_returns, dtype=np.float64)
    if bench.shape != strat.shape:
        bench = np.zeros_like(strat)
    excess = strat - bench
    if excess.size < 2:
        return 0.0
    mean = float(excess.mean())
    std = float(excess.std(ddof=1)) if excess.size > 1 else 0.0
    if std <= 0.0:
        return float(np.sign(mean) * _CAP)
    ir = mean / std * np.sqrt(_ANNUALIZATION)
    return float(np.clip(ir, -_CAP, _CAP))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        obs = np.empty(x.size, dtype=bool)
        obs[0] = True
        np.not_equal(x_sorted[1:], x_sorted[:-1], out=obs[1:])
        dense = np.cumsum(obs) - 1
        counts = np.bincount(dense)
        cum = np.cumsum(counts)
        avg = (cum - counts + 1 + cum) / 2.0
        out = np.empty_like(dense)
        out[order] = avg[dense]
        return out

    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra @ ra) * (rb @ rb)))
    if den <= 0.0:
        return 0.0
    return float(ra @ rb) / den


def reward_oos_correlation(
    pairs: list[tuple[float, float]],
    seed: int = 0,
    n_perms: int = 999,
) -> dict[str, float]:
    """Spearman correlation between training rewards and OOS active IR.

    ``pairs`` is ``[(reward, oos_active_ir), ...]`` with at least two
    entries.  The p-value is a deterministic two-sided permutation test
    (the reward labels are permuted under the null of no association).
    """

    if len(pairs) < 2:
        raise ValueError("reward_oos_correlation needs at least two pairs")
    rewards = np.asarray([float(r) for r, _ in pairs], dtype=np.float64)
    oos = np.asarray([float(o) for _, o in pairs], dtype=np.float64)
    observed = _spearman(rewards, oos)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_perms)):
        shuffled = rng.permutation(oos)
        perm_rho = _spearman(rewards, shuffled)
        if abs(perm_rho) >= abs(observed):
            count += 1
    p_value = float((count + 1) / (int(n_perms) + 1))
    return {
        "rho": observed,
        "p_value": p_value,
        "n": len(pairs),
        "n_perms": int(n_perms),
        "seed": int(seed),
        "positive_stable": bool(observed > 0.0 and p_value < 0.05),
    }


@dataclass(frozen=True)
class BaselineHarnessResult:
    """One matched-budget baseline run plus its validity statistics."""

    budget: int
    n_evaluated: int
    n_invalid: int
    scores: tuple[CandidateScore, ...]
    selected: CandidateScore | None
    reward_oos: dict[str, float] | None = None
    rejections: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "n_evaluated": self.n_evaluated,
            "n_invalid": self.n_invalid,
            "selected": self.selected.to_dict() if self.selected else None,
            "reward_oos": self.reward_oos,
            "rejections": self.rejections,
            "n_scores": len(self.scores),
        }


def run_matched_baseline(
    *,
    target: np.ndarray,
    universe_mask: np.ndarray,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig,
    val_windows: list[tuple[int, int]],
    train_signal_range: tuple[int, int],
    budget: int,
    seed: int,
    execute: Callable[[tuple[int, ...]], np.ndarray | None],
    max_formula_len: int = 12,
    tie_break_keys: np.ndarray | None = None,
    adv: np.ndarray | None = None,
) -> BaselineHarnessResult:
    """Score ``budget`` **unique** structurally-valid formulas with the
    shared candidate scorer and select the best under the Pareto
    objectives — the budget-matched random-search baseline.

    ``execute`` is the formula evaluator (``tokens -> [stock, date]
    signal`` or ``None`` for invalid formulas); the production caller
    binds it to the StackVM and the factor tensor.  The budget is
    matched to the trained candidate's evaluation budget
    (``steps x batch_size`` unique evaluations); duplicates never count
    against it.
    """

    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    vocab = FORMULA_VOCAB
    # Sample with headroom and dedupe: duplicates never count against the
    # matched budget (the trainer's reward cache behaves the same way).
    sampled = sample_random_formulas(seed, vocab, max_formula_len, budget * 2)
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for key in sampled:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
        if len(unique) >= budget:
            break
    unique = unique[:budget]

    scorer = CandidateScorer(backtest_config, reward_config)
    selector = CandidateSelector()
    specs: list[CandidateSpec] = []
    signals: list[np.ndarray | None] = []
    valid_flags: list[bool] = []
    n_invalid = 0
    for key in unique:
        specs.append(
            CandidateSpec(
                candidate_id="baseline:" + ",".join(str(t) for t in key),
                formula_text="",
                source="matched_baseline",
                tokens=key,
            )
        )
        signal = execute(key)
        if signal is None:
            signals.append(None)
            valid_flags.append(False)
            n_invalid += 1
        else:
            signals.append(np.asarray(signal, dtype=np.float64))
            valid_flags.append(True)
    scores = scorer.score_many(
        specs,
        signals,
        target,
        val_windows,
        universe_mask=universe_mask,
        formula_valid=valid_flags,
        train_signal_range=train_signal_range,
        tie_break_keys=tie_break_keys,
        adv=adv,
    )
    selection = selector.select(scores, pareto_objectives=(
        ("active_ir", 1),
        ("risk_exposure", -1),
        ("average_turnover", -1),
        ("capacity_utilization", -1),
    ))
    rejections: dict[str, int] = {}
    for score in scores:
        for reason in score.rejection_reasons:
            rejections[reason] = rejections.get(reason, 0) + 1
    return BaselineHarnessResult(
        budget=budget,
        n_evaluated=len(unique),
        n_invalid=n_invalid,
        scores=tuple(scores),
        selected=selection.selected,
        rejections=rejections,
    )
