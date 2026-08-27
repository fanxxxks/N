"""Matched-budget baseline harness (T1-05) and the Phase-2 search core.

The harness answers the phase-1 research-validity questions with the
shared production measurement path (scorer + engine), not a proxy:

* :func:`run_matched_baseline` gives the random-search baseline the
  **same evaluation budget** as a trained candidate — ``steps x
  batch_size`` unique formula evaluations — so RL-vs-baseline
  comparisons are budget-fair (the protocol's fixed ``random_samples``
  knob is replaced by the matched budget unless explicitly disabled).
* :class:`SemanticBudgetEvaluator` is the shared evaluation core of every
  Phase-2 searcher (random, GP, TPE, RL): proposals are billed in unique
  semantic formula evaluations (T2-01) and scored through the shared
  candidate scorer; the best-so-far curve and the Pareto selection are
  recorded for the admission comparisons (T2-03).
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
    score_chunk_size,
)
from .complexity import complexity_bill
from .ir import FormulaSyntaxError, canonical_ast, canonical_tokens
from .reward import REWARD_VERSION
from .semantic_cache import SemanticCache
from .train import sample_random_formulas
from .vm import formula_decode
from .vocab import FormulaVocab, FORMULA_VOCAB

_ANNUALIZATION = 252
_CAP = 1e9


def canonical_form_pool(
    seed: int,
    vocab: FormulaVocab,
    max_formula_len: int,
    target_count: int,
) -> list[tuple[int, ...]]:
    """Uniform mask-legal samples, canonically deduplicated.

    Structurally invalid and degenerate (constant-producing) formulas are
    dropped; commuted and otherwise identical canonical forms collapse to
    one entry.  Returns at most ``target_count`` canonical token
    sequences.  The random baseline's proposal pool, shared by the
    harness, the protocol row and the train entry.
    """

    sampled = sample_random_formulas(seed, vocab, max_formula_len, target_count * 8)
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
        if len(canonical_forms) >= target_count:
            break
    return canonical_forms


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
    equivalent evaluation without billing; ``best_so_far`` is the
    ``(cumulative budget, best validation reward)`` curve of the search.
    """

    budget: int
    n_evaluated: int
    n_invalid: int
    scores: tuple[CandidateScore, ...]
    selected: CandidateScore | None
    reward_oos: dict[str, float] | None = None
    rejections: dict[str, int] | None = None
    n_semantic_dedups: int = 0
    best_so_far: tuple[tuple[float, float], ...] = ()

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
            "best_so_far": list(self.best_so_far),
        }


class SemanticBudgetEvaluator:
    """Shared evaluation core of the Phase-2 searchers (T2-02).

    Every searcher (random, GP, TPE) proposes token sequences through
    :meth:`propose`; the evaluator converts them into evaluated candidates
    under the unique-semantic-evaluation budget: structurally invalid and
    degenerate formulas never evaluate, canonical duplicates and
    numerically equivalent classes (calibration fingerprint + complexity
    bill) never bill twice, and every full evaluation is scored through
    the shared candidate scorer in bounded chunks.  The best-so-far curve
    and the final Pareto selection are recorded for the admission
    comparisons (T2-03).
    """

    def __init__(
        self,
        *,
        target: np.ndarray,
        universe_mask: np.ndarray,
        backtest_config: BacktestConfig,
        reward_config: RewardConfig,
        val_windows: list[tuple[int, int]],
        train_signal_range: tuple[int, int],
        budget: int,
        execute: Callable[[tuple[int, ...]], np.ndarray | None],
        fingerprint_execute: Callable[[tuple[int, ...]], np.ndarray | None],
        dataset_id: str | None,
        protocol_version: int | str,
        window_id: str,
        tie_break_keys: np.ndarray | None = None,
        adv: np.ndarray | None = None,
        blocked_buy: np.ndarray | None = None,
        blocked_sell: np.ndarray | None = None,
        source: str = "search",
        candidate_prefix: str = "search",
        chunk: int | None = None,
        vocab: FormulaVocab | None = None,
        cache: SemanticCache | None = None,
    ):
        self._target = np.asarray(target, dtype=np.float64)
        self._universe_mask = np.asarray(universe_mask, dtype=bool)
        self._val_windows = val_windows
        self._train_signal_range = train_signal_range
        self._tie_break_keys = tie_break_keys
        self._adv = adv
        self._blocked_buy = (
            np.asarray(blocked_buy, dtype=bool) if blocked_buy is not None else None
        )
        self._blocked_sell = (
            np.asarray(blocked_sell, dtype=bool) if blocked_sell is not None else None
        )
        self._budget = int(budget)
        self._execute = execute
        self._fingerprint_execute = fingerprint_execute
        self._source = source
        self._candidate_prefix = candidate_prefix
        self._vocab = vocab or FORMULA_VOCAB
        # Default chunk = 1: every evaluated proposal is scored eagerly so
        # search loops (GP/TPE) can read the score immediately.  Batched
        # callers pass a larger chunk for memory-bounded score_many calls.
        self._chunk = int(chunk) if chunk is not None else 1
        self._scorer = CandidateScorer(backtest_config, reward_config)
        self._selector = CandidateSelector()
        self._cache = cache if cache is not None else SemanticCache(
            dataset_id=dataset_id,
            reward_version=REWARD_VERSION,
            protocol_version=int(protocol_version),
            window_id=window_id,
            cap=max(int(budget), 1024),
        )
        self._pending: list[tuple[CandidateSpec, np.ndarray]] = []
        self._claimed: set[object] = set()
        self._claim_sequence: list[int] = []
        self._scores: list[CandidateScore] = []
        self._best_so_far: list[tuple[float, float]] = []
        self._best_reward = -float("inf")
        self._n_evaluated = 0
        self._n_invalid = 0
        self._n_semantic_dedups = 0

    # --- proposal pipeline -------------------------------------------------

    def propose(self, tokens) -> tuple[CandidateScore | None, bool]:
        """Score one proposed formula under the semantic budget.

        Returns ``(score, consumed_budget)``.  ``score`` is ``None`` when
        the proposal was not evaluated (structurally invalid, degenerate,
        or a semantic class already claimed this run) — its fitness is the
        caller's choice (e.g. the current best).  ``consumed_budget`` is
        True exactly when a full evaluation happened.  With ``chunk == 1``
        the score is returned eagerly; with larger chunks it lands at the
        next :meth:`flush`.
        """

        ckey = self._cache.key_for(tokens)
        if ckey is None:
            return None, False  # invalid or degenerate: never evaluated
        score = self._cache.get(ckey)
        if score is not None:
            return score, False  # canonical duplicate of an evaluation
        if ckey in self._claimed:
            return None, False  # same canonical form claimed this run
        canonical = canonical_ast(tokens, self._vocab)
        bill = complexity_bill(canonical) if canonical is not None else None
        fingerprint = self._cache.fingerprint(
            tokens, self._fingerprint_execute, self._vocab
        )
        if fingerprint is not None:
            if self._cache.is_claimed(fingerprint, bill):
                self._n_semantic_dedups += 1
                return None, False
            score = self._cache.score_by_fingerprint(fingerprint, bill)
            if score is not None:
                self._n_semantic_dedups += 1
                self._cache.put(ckey, score, fingerprint, bill)
                return score, False
        spec = CandidateSpec(
            candidate_id=self._candidate_prefix
            + ":"
            + ",".join(str(int(t)) for t in tokens),
            formula_text=formula_decode(list(tokens), self._vocab),
            source=self._source,
            tokens=tuple(int(t) for t in tokens),
        )
        signal = self._execute(tokens)
        self._n_evaluated += 1
        self._claimed.add(ckey)
        self._cache.put(ckey, None, fingerprint, bill)  # claim the class
        self._claim_sequence.append(self._cache.budget_used)
        if signal is None:
            self._n_invalid += 1
            score = self._scorer.score(
                spec,
                None,
                self._target,
                self._val_windows,
                blocked_buy=self._blocked_buy,
                blocked_sell=self._blocked_sell,
                formula_valid=False,
                train_signal_range=self._train_signal_range,
                universe_mask=self._universe_mask,
            )
            self._cache.put(ckey, score, None)
            self._register(score, x=self._claim_sequence[-1])
            return score, True
        self._pending.append((spec, np.asarray(signal, dtype=np.float64)))
        if len(self._pending) >= self._chunk:
            self.flush()
        if not self._pending:
            # Eager mode (chunk == 1): the score landed in the flush above.
            return self._scores[-1], True
        return None, True

    def score_of(self, tokens) -> CandidateScore | None:
        """Cached score of an evaluated proposal (``None`` when not
        evaluated).  Lets batched searchers read every proposal's real
        score after a :meth:`flush`, without re-evaluating."""

        ckey = self._cache.key_for(tokens)
        if ckey is None:
            return None
        return self._cache.get(ckey)

    def flush(self) -> None:
        """Score the buffered pending signals (bounded memory chunks)."""

        if not self._pending:
            return
        specs = [spec for spec, _ in self._pending]
        signals = [signal for _, signal in self._pending]
        scored = self._scorer.score_many(
            specs,
            signals,
            self._target,
            self._val_windows,
            blocked_buy=self._blocked_buy,
            blocked_sell=self._blocked_sell,
            train_signal_range=self._train_signal_range,
            universe_mask=self._universe_mask,
            tie_break_keys=self._tie_break_keys,
            adv=self._adv,
        )
        base = len(self._scores)
        for i, (key, score) in enumerate(zip(specs, scored)):
            ckey = self._cache.key_for(key.tokens)
            if ckey is not None:
                self._cache.put(ckey, score, None)  # real score overwrites
            self._register(score, x=self._claim_sequence[base + i])
        self._pending = []

    def _register(self, score: CandidateScore, x: float) -> None:
        self._scores.append(score)
        if score.val_reward > self._best_reward:
            self._best_reward = float(score.val_reward)
        self._best_so_far.append((float(x), self._best_reward))

    # --- accounting --------------------------------------------------------

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def vocab(self) -> FormulaVocab:
        return self._vocab

    @property
    def budget_used(self) -> int:
        return self._cache.budget_used

    @property
    def best_reward(self) -> float:
        return self._best_reward

    def stats(self) -> dict[str, int | str]:
        return {
            **self._cache.stats(),
            "n_evaluated": self._n_evaluated,
            "n_invalid": self._n_invalid,
            "n_semantic_dedups": self._n_semantic_dedups,
        }

    def finish(self) -> BaselineHarnessResult:
        self.flush()
        selection = self._selector.select(
            self._scores,
            pareto_objectives=(
                ("active_ir", 1),
                ("risk_exposure", -1),
                ("average_turnover", -1),
                ("capacity_utilization", -1),
            ),
        )
        rejections: dict[str, int] = {}
        for score in self._scores:
            for reason in score.rejection_reasons:
                rejections[reason] = rejections.get(reason, 0) + 1
        return BaselineHarnessResult(
            budget=self._budget,
            n_evaluated=self._n_evaluated,
            n_invalid=self._n_invalid,
            scores=tuple(self._scores),
            selected=selection.selected,
            rejections=rejections,
            n_semantic_dedups=self._n_semantic_dedups,
            best_so_far=tuple(self._best_so_far),
        )


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

    # Sample with headroom and dedupe canonically: degenerate formulas are
    # dropped and structurally identical sequences collapse, so the pool
    # reaches the budget with fresh forms.
    headroom = budget * 8 if semantic_mode else budget * 2
    canonical_forms = canonical_form_pool(
        seed, vocab, max_formula_len, headroom
    )

    if semantic_mode:
        # The semantic budget is enforced by the shared evaluator (the same
        # core the GP and TPE baselines run on); the chunk keeps the
        # batched basket simulation memory-bounded for large budgets.
        signal_bytes = np.asarray(target).shape[0] * np.asarray(target).shape[1] * 8
        evaluator = SemanticBudgetEvaluator(
            target=target,
            universe_mask=universe_mask,
            backtest_config=backtest_config,
            reward_config=reward_config,
            val_windows=val_windows,
            train_signal_range=train_signal_range,
            budget=budget,
            execute=execute,
            fingerprint_execute=fingerprint_execute,
            dataset_id=dataset_id,
            protocol_version=protocol_version,
            window_id=window_id,
            tie_break_keys=tie_break_keys,
            adv=adv,
            source="matched_baseline",
            candidate_prefix="baseline",
            chunk=score_chunk_size(signal_bytes),
        )
        for key in canonical_forms:
            evaluator.propose(key)
            if evaluator.budget_used >= budget:
                break
        return evaluator.finish()

    scorer = CandidateScorer(backtest_config, reward_config)
    selector = CandidateSelector()
    specs: list[CandidateSpec] = []
    signals: list[np.ndarray | None] = []
    valid_flags: list[bool] = []
    n_invalid = 0
    n_evaluated = 0
    for key in canonical_forms:
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
    )
