"""T2-02 contracts: the Optuna TPE baseline.

The TPE baseline searches the same token space as the RL policy (position-
wise categoricals over the full vocabulary), is billed in unique semantic
formula evaluations, and is deterministic in its seed.  Invalid and
degenerate proposals cost nothing; proposals that were already evaluated
reuse the stored score without billing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_model.baseline_harness import SemanticBudgetEvaluator
from ashare_model.tpe_search import run_tpe_baseline


def _evaluator(**overrides):
    from ashare_data.config import BacktestConfig, RewardConfig

    rng = np.random.default_rng(5)
    target = rng.normal(0.001, 0.01, size=(12, 48))
    mask = np.ones(target.shape, dtype=bool)

    def execute(tokens):
        key_sum = sum(int(t) for t in tokens)
        rng_f = np.random.default_rng(key_sum % 1000 + 1)
        return target + rng_f.normal(0, 0.02, size=target.shape)

    kwargs = dict(
        target=target,
        universe_mask=mask,
        backtest_config=BacktestConfig(
            initial_capital=100000.0, top_n=5, single_weight_cap=0.5
        ),
        reward_config=RewardConfig(
            ic_min_stocks=5,
            min_val_reward=-1e9,
            min_val_icir=-1e9,
            min_valid_ic_days=2,
            min_effective_stocks=2,
            min_coverage=0.0,
            min_activity=0.0,
            min_sign_stability=0.0,
            min_val_window_q25=-1e9,
        ),
        val_windows=[(24, 34), (34, 44)],
        train_signal_range=(0, 24),
        budget=10,
        dataset_id="dataset-a",
        protocol_version=18,
        window_id="train:0:100",
        execute=execute,
        fingerprint_execute=execute,
    )
    kwargs.update(overrides)
    return SemanticBudgetEvaluator(**kwargs)


def test_run_tpe_baseline_respects_semantic_budget():
    evaluator = _evaluator(budget=10)
    result = run_tpe_baseline(seed=7, evaluator=evaluator, max_formula_len=10)
    assert result.budget == 10
    assert result.n_evaluated == 10
    assert len(result.best_so_far) == 10
    rewards = [v for _, v in result.best_so_far]
    assert rewards == sorted(rewards)  # best-so-far never drops
    assert result.selected is not None


def test_run_tpe_baseline_deterministic_in_seed():
    first = run_tpe_baseline(
        seed=11, evaluator=_evaluator(budget=8), max_formula_len=10
    )
    second = run_tpe_baseline(
        seed=11, evaluator=_evaluator(budget=8), max_formula_len=10
    )
    assert first.best_so_far == second.best_so_far
    assert first.selected.formula_text == second.selected.formula_text


def test_run_tpe_baseline_reuses_evaluated_formulas_without_billing():
    # The same formula proposed twice bills once: the second proposal
    # reuses the stored score through the shared ledger.
    evaluator = _evaluator(budget=6)
    from ashare_model.vocab import FORMULA_VOCAB

    vocab = FORMULA_VOCAB
    ret1 = vocab.feature_offset + vocab.feature_names.index("RET_1")
    tokens = (ret1, vocab.eos_token_id) + (vocab.pad_token_id,) * 8
    score, consumed = evaluator.propose(tokens)
    assert score is not None and consumed
    reused, consumed_again = evaluator.propose(tokens)
    assert reused is not None and not consumed_again
    assert evaluator.budget_used == 1
    assert evaluator.stats()["n_hits"] >= 1


def test_run_tpe_baseline_invalid_proposals_cost_nothing():
    evaluator = _evaluator(budget=4)
    # An empty sequence is structurally invalid: never evaluated, never
    # billed, and the search continues.
    score, consumed = evaluator.propose(())
    assert score is None and not consumed
    result = run_tpe_baseline(seed=17, evaluator=evaluator, max_formula_len=10)
    assert result.n_evaluated == 4
