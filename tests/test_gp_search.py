"""T2-02 contracts: the DEAP strongly-typed GP baseline.

The GP baseline is budget-fair: it is billed in unique semantic formula
evaluations (the same ledger as random search and RL), every proposed
formula fits the policy's max formula length, and the search is
deterministic in its seed.  DEAP supplies the typed tree structure and the
variation operators; this module maps trees to the formula grammar and
binds evaluation to the shared semantic budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig

from ashare_model.baseline_harness import SemanticBudgetEvaluator
from ashare_model.gp_search import run_gp_baseline
from ashare_model.ir import decode, encode, infix
from ashare_model.vocab import FORMULA_VOCAB


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=100000.0,
        top_n=5,
        single_weight_cap=0.5,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _reward(**kwargs) -> RewardConfig:
    defaults = dict(
        ic_min_stocks=5,
        min_val_reward=-1e9,
        min_val_icir=-1e9,
        min_valid_ic_days=2,
        min_effective_stocks=2,
        min_coverage=0.0,
        min_activity=0.0,
        min_sign_stability=0.0,
        min_val_window_q25=-1e9,
    )
    defaults.update(kwargs)
    return RewardConfig(**defaults)


def _context(seed=5, n_stocks=12, n_dates=48):
    rng = np.random.default_rng(seed)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    mask = np.ones(target.shape, dtype=bool)

    def execute(tokens):
        # Deterministic, formula-dependent, non-constant signal: enough
        # structure for the search to differentiate candidates.
        key_sum = sum(int(t) for t in tokens)
        rng_f = np.random.default_rng(key_sum % 1000 + 1)
        return target + rng_f.normal(0, 0.02, size=target.shape)

    return dict(
        target=target,
        universe_mask=mask,
        backtest_config=_cfg(),
        reward_config=_reward(),
        val_windows=[(24, 34), (34, 44)],
        train_signal_range=(0, 24),
        budget=12,
        max_formula_len=10,
        dataset_id="dataset-a",
        protocol_version=18,
        window_id="train:0:100",
        execute=execute,
        fingerprint_execute=execute,
    )


def _evaluator(**overrides):
    kwargs = _context()
    kwargs.update(overrides)
    kwargs.pop("max_formula_len", None)
    return SemanticBudgetEvaluator(**kwargs)


def _tokens(*names: str) -> list[int]:
    from ashare_model.ops import OPS_CONFIG

    vocab = FORMULA_VOCAB
    out = [
        vocab.feature_offset + vocab.feature_names.index(n) for n in names
    ]
    return out


# --- tree <-> grammar mapping ----------------------------------------------


def test_tree_maps_to_valid_bounded_tokens():
    from ashare_model.gp_search import tree_to_tokens, tokens_to_tree

    vocab = FORMULA_VOCAB
    add = vocab.operator_offset
    tokens = tuple(_tokens("RET_1", "RET_5") + [add])
    tree = tokens_to_tree(tokens, vocab)
    assert tree is not None
    assert tuple(tree_to_tokens(tree, vocab)) == tokens
    # Every generated token sequence stays within the policy's length
    # budget (max_formula_len == 10 -> at most 5 nodes -> 9 tokens + EOS).
    assert len(tree_to_tokens(tree, vocab)) <= 9


def test_typed_primitive_set_covers_vocabulary():
    from ashare_model.gp_search import build_pset

    pset = build_pset(FORMULA_VOCAB)
    # P7-E: terminals and primitives are keyed by semantic-type class, and
    # every operator appears once per legal signature (same name, same
    # arity); the vocabulary coverage is aggregated over the type keys.
    terminals = {
        t.name for entries in pset.terminals.values() for t in entries
    }
    primitives = {
        p.name: p.arity
        for entries in pset.primitives.values()
        for p in entries
    }
    # P9 §4.2 (whitelist §10.1 case 2, contract APPROVED): deprecated
    # features leave every backend's sampling space, so the GP terminal
    # set covers exactly the live vocabulary.
    from ashare_model.vocab import DEPRECATED_FEATURE_NAMES

    assert terminals == set(FORMULA_VOCAB.feature_names) - set(
        DEPRECATED_FEATURE_NAMES
    )
    assert not (terminals & set(DEPRECATED_FEATURE_NAMES))
    assert set(primitives) == set(FORMULA_VOCAB.operator_names)
    from ashare_model.ops import OPS_CONFIG

    for name, _, arity in OPS_CONFIG:
        assert primitives[name] == arity, name


def test_infix_of_tree_matches_ast():
    from ashare_model.gp_search import tokens_to_tree, tree_to_tokens

    vocab = FORMULA_VOCAB
    add = vocab.operator_offset
    tokens = tuple(_tokens("RET_1", "RET_5") + [add])
    tree = tokens_to_tree(tokens, vocab)
    assert tree is not None
    assert infix(decode(tuple(tree_to_tokens(tree, vocab)) + (vocab.eos_token_id,))) == (
        "(RET_1 ADD RET_5)"
    )
    assert encode(decode(tuple(tree_to_tokens(tree, vocab)) + (vocab.eos_token_id,)))[
        :-1
    ] == tree_to_tokens(tree, vocab)


# --- budget and determinism -------------------------------------------------


def test_run_gp_baseline_respects_semantic_budget():
    evaluator = _evaluator(budget=10)
    result = run_gp_baseline(seed=7, evaluator=evaluator, max_formula_len=10)
    assert result.n_evaluated == 10
    assert result.budget == 10
    assert len(result.best_so_far) == 10
    rewards = [v for _, v in result.best_so_far]
    assert rewards == sorted(rewards)  # best-so-far never drops
    assert result.selected is not None


def test_run_gp_baseline_deterministic_in_seed():
    first = run_gp_baseline(
        seed=11, evaluator=_evaluator(budget=8), max_formula_len=10
    )
    second = run_gp_baseline(
        seed=11, evaluator=_evaluator(budget=8), max_formula_len=10
    )
    assert first.best_so_far == second.best_so_far
    assert first.selected.formula_text == second.selected.formula_text


def test_run_gp_baseline_proposals_fit_max_length():
    # Every formula the GP proposes (evaluated or not) stays within the
    # policy's length budget: the tree node cap mirrors max_formula_len.
    seen = set()

    def execute(tokens):
        seen.add(tuple(int(t) for t in tokens))
        rng = np.random.default_rng(sum(tokens) % 1000 + 1)
        return _context()["target"] + rng.normal(0, 0.02, size=(12, 48))

    evaluator = _evaluator(budget=6, execute=execute, fingerprint_execute=execute)
    run_gp_baseline(seed=3, evaluator=evaluator, max_formula_len=10)
    for tokens in seen:
        assert len(tuple(int(t) for t in tokens)) <= 9  # 10 incl. EOS


def test_run_gp_baseline_uses_shared_budget_ledger():
    # The GP and the random baseline bill the identical semantic classes:
    # a formula the GP already evaluated never bills a second time.
    evaluator = _evaluator(budget=6)
    run_gp_baseline(seed=13, evaluator=evaluator, max_formula_len=10)
    stats = evaluator.stats()
    assert stats["budget_used"] == 6
    assert stats["n_semantic_dedups"] >= 0


# --- P14 §5.1/§5.2: punitive fitness for skipped proposals ------------------
# Contract: docs/p14_search_digest_preregistration.md (search digest) —
# skipped proposals (invalid / semantic-duplicate / claimed-before-flush)
# must receive the run's worst registered finite reward (never the current
# best): the old "neutral = best" anchor let invalid and duplicate
# individuals tie with the best and drove the P10 population collapse
# (GP consumed 5–14% of its budget).


def test_evaluator_worst_reward_is_the_punitive_anchor():
    evaluator = _evaluator(budget=8)
    assert evaluator.worst_reward == float("-inf")  # nothing registered yet
    vocab = FORMULA_VOCAB
    ret1 = vocab.feature_offset + vocab.feature_names.index("RET_1")
    ret5 = vocab.feature_offset + vocab.feature_names.index("RET_5")
    add = vocab.operator_offset + vocab.operator_names.index("ADD")
    s1, consumed1 = evaluator.propose((ret1, ret5, add, vocab.eos_token_id))
    assert consumed1 and s1 is not None
    s2, consumed2 = evaluator.propose((ret5, vocab.eos_token_id))
    assert consumed2 and s2 is not None
    vals = [float(s.val_reward) for s in (s1, s2)]
    assert evaluator.worst_reward == min(vals)
    assert evaluator.best_reward == max(vals)
    # An invalid proposal registers nothing: the anchor is unchanged.
    evaluator.propose(())
    assert evaluator.worst_reward == min(vals)


def test_evaluator_execute_none_proposals_never_break_best_so_far():
    """T2-01 + p14 §5.1: an execution failure is not billed and produces
    no reward, so it must not emit a best-so-far coordinate — the old code
    registered it at a duplicate budget coordinate and crashed the
    SearchResult construction (latent defect exposed by the p14 punitive
    scenarios)."""

    base = _context()
    target = base["target"]
    vocab = FORMULA_VOCAB
    blocked_token = vocab.feature_offset + vocab.feature_names.index("RET_5")
    ret1 = vocab.feature_offset + vocab.feature_names.index("RET_1")
    eos = vocab.eos_token_id

    def execute(tokens):
        # Deterministic per-formula execution failure: RET_5 never executes.
        if any(int(t) == blocked_token for t in tokens):
            return None
        rng = np.random.default_rng(sum(int(t) for t in tokens) % 1000 + 1)
        return target + rng.normal(0, 0.02, size=target.shape)

    evaluator = _evaluator(budget=8, execute=execute, fingerprint_execute=execute)
    evaluator.propose((ret1, eos))  # bills (class 1)
    score, consumed = evaluator.propose((blocked_token, eos))  # fails to execute
    assert consumed is True and score is not None  # attempted, not billed
    assert evaluator.budget_used == 1  # the failed execution is not billed
    evaluator.propose((ret1, eos))  # canonical duplicate of the first
    result = evaluator.finish(
        backend="gp", seed=1, termination_reason="budget_exhausted"
    )
    coordinates = [x for x, _ in result.best_so_far]
    assert coordinates == sorted(set(coordinates))  # strictly increasing


def test_run_gp_baseline_punitive_and_neutral_arms_deterministic_under_pinned_threads(
    monkeypatch,
):
    """FP-jitter guard (p14 §8-2; t49 repair-round-2, t24-ruling): both
    anchors are pinned to single-threaded BLAS and each arm must be exactly
    run-to-run deterministic (identical consumed budget and best-so-far).

    History (recorded for t24/t31): the original t18 test asserted a
    cross-arm consumption differential (punitive > neutral-best) on this
    toy.  Empirically that differential is dominated by environment
    nondeterminism, not by the anchor: across process hash seeds and BLAS
    threading the pairs measured (punitive, neutral) = (288, 84), (115, 83)
    and — fully pinned with PYTHONHASHSEED=0 + single-thread BLAS —
    (60, 103), i.e. the direction itself flips.  A toy whose reachable
    semantic-class space is this degenerate cannot carry a collapse
    signature; asserting one was a t18 test-design error (§10.1: built on
    an unstable oracle).  The collapse metric therefore stays where p14
    §9 pre-registered it: the formal-run judgment (GP consumption >= 50%,
    measured by the P10-faithful campaign on real data), not this unit
    test.  This guard pins the deterministic part: the anchor mechanism is
    unit-tested via ``test_evaluator_worst_reward_is_the_punitive_anchor``
    and ``test_evaluator_execute_none_proposals_never_break_best_so_far``.
    """

    from threadpoolctl import threadpool_limits

    saved_property = SemanticBudgetEvaluator.worst_reward
    try:
        with threadpool_limits(limits=1):
            # Punitive anchor (p14 §5.1): run-to-run deterministic.
            first = run_gp_baseline(
                seed=7, evaluator=_evaluator(budget=300), max_formula_len=10
            )
            second = run_gp_baseline(
                seed=7, evaluator=_evaluator(budget=300), max_formula_len=10
            )
            assert first.consumed_budget == second.consumed_budget
            assert first.best_so_far == second.best_so_far
            # Old anchor emulated (skipped proposals tied with the current
            # best): equally deterministic.
            monkeypatch.setattr(
                SemanticBudgetEvaluator,
                "worst_reward",
                property(lambda self: self.best_reward),
            )
            third = run_gp_baseline(
                seed=7, evaluator=_evaluator(budget=300), max_formula_len=10
            )
            fourth = run_gp_baseline(
                seed=7, evaluator=_evaluator(budget=300), max_formula_len=10
            )
            assert third.consumed_budget == fourth.consumed_budget
            assert third.best_so_far == fourth.best_so_far
    finally:
        SemanticBudgetEvaluator.worst_reward = saved_property


def test_run_gp_baseline_punitive_fitness_deterministic():
    first = run_gp_baseline(
        seed=7, evaluator=_evaluator(budget=120), max_formula_len=10
    )
    second = run_gp_baseline(
        seed=7, evaluator=_evaluator(budget=120), max_formula_len=10
    )
    assert first.best_so_far == second.best_so_far
    assert first.consumed_budget == second.consumed_budget
