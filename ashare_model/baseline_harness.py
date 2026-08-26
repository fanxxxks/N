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
from .complexity import complexity_bill
from .ir import FormulaSyntaxError, canonical_ast, canonical_tokens
from .semantic_cache import CalibrationSlice, SemanticCache, make_calibration_execute
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
    """One matched-budget baseline run plus its validity statistics.

    ``budget`` is the requested evaluation budget; ``n_evaluated`` is the
    number of evaluations actually performed (unique semantic evaluations
    in semantic mode, unique canonical evaluations otherwise);
    ``n_semantic_dedups`` counts formulas that reused a numerically
    equivalent evaluation without billing.
    """

    budget: int
    n_evaluated: int
    n_invalid: int
    scores: tuple[CandidateScore, ...]
    selected: CandidateScore | None
    reward_oos: dict[str, float] | None = None
    rejections: dict[str, int] | None = None
    n_semantic_dedups: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget,
            "n_evaluated": self.n_evaluated,
            "n_invalid": self.n_invalid,
            "n_semantic_dedups": self.n_semantic_dedups,
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
    dataset_id: str | None = None,
    protocol_version: int | str | None = None,
    window_id: str | None = None,
    fingerprint_execute: Callable[[tuple[int, ...]], np.ndarray | None] | None = None,
) -> BaselineHarnessResult:
    """Score ``budget`` unique formulas with the shared candidate scorer
    and select the best under the Pareto objectives — the budget-matched
    random-search baseline.

    ``execute`` is the formula evaluator (``tokens -> [stock, date]``
    signal or ``None`` for invalid formulas); the production caller binds
    it to the StackVM and the factor tensor.  The budget is matched to the
    trained candidate's evaluation budget (``steps x batch_size`` unique
    evaluations); duplicates never count against it.

    Budget semantics (T2-01): with the semantic context (``dataset_id`` /
    ``protocol_version`` / ``window_id``) and ``fingerprint_execute``
    provided, the budget counts **unique semantic formula evaluations** —
    structurally invalid and degenerate formulas are skipped, canonical
    duplicates are merged, and numerically equivalent formulas (same
    calibration fingerprint) are scored once.  Without them the legacy
    token-sequence budget applies (T1-05), with canonical deduplication.
    """

    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    vocab = FORMULA_VOCAB
    semantic_mode = (
        dataset_id is not None
        and protocol_version is not None
        and window_id is not None
        and fingerprint_execute is not None
    )
    cache: SemanticCache | None = None
    if semantic_mode:
        from .reward import REWARD_VERSION

        cache = SemanticCache(
            dataset_id=dataset_id,
            reward_version=REWARD_VERSION,
            protocol_version=int(protocol_version),
            window_id=window_id,
            cap=budget,
        )

    # Sample with headroom and dedupe canonically: degenerate formulas are
    # dropped and structurally identical sequences collapse, so the pool
    # reaches the budget with fresh forms.
    headroom = budget * 8 if semantic_mode else budget * 2
    sampled = sample_random_formulas(seed, vocab, max_formula_len, headroom)
    canonical_forms: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for key in sampled:
        try:
            canonical = canonical_tokens(key, vocab)
        except FormulaSyntaxError:
            continue  # structurally invalid: never evaluated
        if canonical is None:
            continue  # degenerate (constant-producing): never evaluated
        ctuple = tuple(canonical)
        if ctuple in seen:
            continue
        seen.add(ctuple)
        canonical_forms.append(ctuple)
        if len(canonical_forms) >= headroom:
            break

    scorer = CandidateScorer(backtest_config, reward_config)
    selector = CandidateSelector()
    specs: list[CandidateSpec] = []
    signals: list[np.ndarray | None] = []
    valid_flags: list[bool] = []
    n_invalid = 0
    n_evaluated = 0
    n_semantic_dedups = 0
    for key in canonical_forms:
        if cache is not None:
            ckey = cache.key_for(key)
            if ckey is None:
                continue
            canonical = canonical_ast(key, vocab)
            bill = complexity_bill(canonical) if canonical is not None else None
            fingerprint = cache.fingerprint(key, fingerprint_execute, vocab)
            if fingerprint is not None:
                if cache.is_claimed(fingerprint, bill):
                    # Same semantic class already claimed this run: skip
                    # the duplicate evaluation entirely.
                    n_semantic_dedups += 1
                    continue
                score = cache.score_by_fingerprint(fingerprint, bill)
                if score is not None:
                    n_semantic_dedups += 1
                    cache.put(ckey, score, fingerprint, bill)
                    continue
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
        n_evaluated += 1
        if cache is not None:
            # Claim the semantic class now: the full evaluation is being
            # performed, so it consumes one budget unit even though the
            # real score lands later (the placeholder is overwritten below).
            cache.put(ckey, None, fingerprint, bill)
        if n_evaluated >= budget:
            break
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
    if cache is not None:
        for key, score in zip(specs, scores):
            ckey = cache.key_for(key.tokens)
            if ckey is not None:
                cache.put(ckey, score, None)
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
        n_evaluated=n_evaluated,
        n_invalid=n_invalid,
        scores=tuple(scores),
        selected=selection.selected,
        rejections=rejections,
        n_semantic_dedups=n_semantic_dedups,
    )
