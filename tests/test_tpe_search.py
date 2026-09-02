"""T2-02 contracts: the Optuna TPE baseline.

The TPE baseline searches the same token space as the RL policy (position-
wise categoricals over the full vocabulary), is billed in unique semantic
formula evaluations, and is deterministic in its seed.  Invalid and
degenerate proposals cost nothing; proposals that were already evaluated
reuse the stored score without billing.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from ashare_model.baseline_harness import SemanticBudgetEvaluator
from ashare_model.tpe_search import run_tpe_baseline
from ashare_model.vocab import FORMULA_VOCAB


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


# --- P14 §5.2/§5.3: punitive tell + proposal length prior -------------------
# Contract: docs/p14_search_digest_preregistration.md — skipped proposals
# must be told the run's worst registered finite reward (never the current
# best: at the P10 0.98 plateau every duplicate was told the plateau value,
# poisoning the surrogate), and the per-step EOS prior (profile
# p14-uniform-2-11-v1) must break the 91% cap-length stacking.


def _coarse_evaluator(**overrides):
    """Coarse calibration fingerprint (token-sum mod 7): after the first
    handful of (fingerprint, bill) classes every further proposal is a
    semantic duplicate with score_of -> None — the punitive tell path."""
    rng = np.random.default_rng(5)
    target = rng.normal(0.001, 0.01, size=(12, 48))

    def fingerprint_execute(tokens):
        cls = sum(int(t) for t in tokens) % 7
        return target * (1.0 + 0.001 * cls)

    return _evaluator(fingerprint_execute=fingerprint_execute, **overrides)


def test_tpe_tell_gives_skipped_proposals_the_punitive_worst_so_far(monkeypatch):
    import ashare_model.tpe_search as tpe_module

    told: list[float] = []
    real_create = tpe_module.optuna.create_study

    class RecordingStudy:
        def __init__(self, study):
            self._study = study

        def ask(self):
            return self._study.ask()

        def tell(self, trial, value=None, **kwargs):
            told.append(float(value))
            return self._study.tell(trial, value, **kwargs)

        def __getattr__(self, name):
            return getattr(self._study, name)

    monkeypatch.setattr(
        tpe_module.optuna,
        "create_study",
        lambda **kwargs: RecordingStudy(real_create(**kwargs)),
    )
    evaluator = _coarse_evaluator(budget=64)
    result = run_tpe_baseline(seed=7, evaluator=evaluator, max_formula_len=10)
    assert result.proposal_count > result.consumed_budget, "fixture produced no skips"
    real = [
        float(score.val_reward)
        for score in result.scores
        if score.val_reward is not None and math.isfinite(float(score.val_reward))
    ]
    assert len(real) == result.consumed_budget
    assert all(math.isfinite(value) for value in told)  # never NaN/inf
    r_min, r_max = min(real), max(real)
    # Skipped proposals are told the punitive worst-so-far: after the
    # minimum has registered, every skip adds a told value == r_min
    # (strictly more than the real evaluations of that value), and no skip
    # is ever told the current best (the r_max tell count stays at the real
    # count).  The old behavior told best_reward for every skip: r_max
    # inflation (strictly more than real) and no r_min inflation.
    assert Counter(told)[r_min] > Counter(real)[r_min], (
        Counter(told)[r_min],
        Counter(real)[r_min],
    )
    assert Counter(told)[r_max] == Counter(real)[r_max]


def test_tpe_length_prior_induced_distribution():
    """Additional smoke (NOT the frozen bound — see
    ``test_tpe_emission_length_distribution_frozen_bounds``): the per-step
    EOS prior must break the cap stacking: on the toy max_len=10 the
    dominant (cap) content length must fall far below the P10-era ~91% and
    short formulas must receive real coverage."""
    seen: list[tuple[int, ...]] = []

    evaluator = _coarse_evaluator(budget=240)

    original_propose = evaluator.propose

    def recording_propose(tokens):
        seen.append(tuple(int(t) for t in tokens))
        return original_propose(tokens)

    evaluator.propose = recording_propose
    result = run_tpe_baseline(seed=11, evaluator=evaluator, max_formula_len=10)
    assert len(seen) >= result.proposal_count * 0.9
    eos = FORMULA_VOCAB.eos_token_id

    def content_length(tokens):
        return tokens.index(eos) if eos in tokens else len(tokens)

    contents = [content_length(tokens) for tokens in seen]
    cap = sum(1 for c in contents if c >= 9) / len(contents)
    short = sum(1 for c in contents if c <= 7) / len(contents)
    assert cap <= 0.35, f"cap stacking unchanged: {cap:.2f}"
    assert short >= 0.40, f"no short-formula coverage: {short:.2f}"


def test_tpe_emission_length_distribution_frozen_bounds():
    """P14 §8-4 冻结口径（t24 对账裁决观察项 A，t43 修复）：生产词表
    （FORMULA_VOCAB，B-泛化终态：118 tokens / EOS 恒 113——4 个新特征
    ID 114–117 追加在 EOS 之后（tail_feature_names 机制），legacy 公式
    逐字可解码；计数随词表演进更新，判据不变。词表口径经 t46
    A→B-泛化终态裁定，此为第三次翻转、此后定稿）、max_len=12、≥2000 条
    ask→propose→记录循环的发射序列——

    * content=11（即 12 token 含 EOS 的上限长度）占比 **≤ 0.25**；
    * content ≤ 8 占比 **≥ 0.40**；
    * 全部发射序列 ≤ 12 token（未终结序列按发射长度计，RED-4 原文口径）。

    发射循环无需真实研究评价：桩评价器对每个不同 token 元组给出唯一
    calibration fingerprint，使每个发射序列恰好计费一次（零语义重复、
    零停滞），因此 ask→propose 流即为完整发射记录，且运行必然以
    budget_exhausted 终止（发射数 ≥ 预算由记账结构保证）。机制上
    cap 占比 ≤ P(target_content=11) = 1/10 —— 冻结界 0.25 有充足裕度。
    """
    import hashlib

    from ashare_data.config import BacktestConfig, RewardConfig

    rng0 = np.random.default_rng(3)
    base_fp = rng0.normal(0.001, 0.01, size=(6, 60))
    base_ex = rng0.normal(0.001, 0.01, size=(6, 40))

    def _key(tokens) -> int:
        digest = hashlib.md5(
            ",".join(str(int(t)) for t in tokens).encode()
        ).hexdigest()[:8]
        return int(digest, 16)

    def fingerprint_execute(tokens):
        r = np.random.default_rng(_key(tokens))
        return base_fp + r.normal(0, 0.01, size=base_fp.shape)

    def execute(tokens):
        r = np.random.default_rng(_key(tokens))
        return base_ex + r.normal(0, 0.01, size=base_ex.shape)

    evaluator = SemanticBudgetEvaluator(
        target=base_ex,
        universe_mask=np.ones(base_ex.shape, dtype=bool),
        backtest_config=BacktestConfig(
            initial_capital=100000.0, top_n=2, single_weight_cap=0.5
        ),
        reward_config=RewardConfig(
            ic_min_stocks=3,
            min_val_reward=-1e9,
            min_val_icir=-1e9,
            min_valid_ic_days=2,
            min_effective_stocks=2,
            min_coverage=0.0,
            min_activity=0.0,
            min_sign_stability=0.0,
            min_val_window_q25=-1e9,
        ),
        val_windows=[(12, 18), (18, 24)],
        train_signal_range=(0, 12),
        budget=2000,
        dataset_id="dataset-a",
        protocol_version=18,
        window_id="train:0:100",
        execute=execute,
        fingerprint_execute=fingerprint_execute,
    )

    seen: list[tuple[int, ...]] = []
    original_propose = evaluator.propose

    def recording_propose(tokens):
        seen.append(tuple(int(t) for t in tokens))
        return original_propose(tokens)

    evaluator.propose = recording_propose
    result = run_tpe_baseline(seed=7, evaluator=evaluator, max_formula_len=12)

    assert result.termination_reason == "budget_exhausted"
    assert result.proposal_count == len(seen) >= 2000
    eos = FORMULA_VOCAB.eos_token_id
    # RED-4 原文口径：未终结序列按发射长度计。
    contents = [tokens.index(eos) if eos in tokens else len(tokens) for tokens in seen]
    assert all(length <= 12 for length in contents)
    cap = sum(1 for c in contents if c == 11) / len(contents)
    short = sum(1 for c in contents if c <= 8) / len(contents)
    assert cap <= 0.25, f"frozen bound violated at the cap: {cap:.4f}"
    assert short >= 0.40, f"frozen bound violated on short coverage: {short:.4f}"


def test_length_prior_target_lengths_match_preregistered_profile():
    import random as _random

    from ashare_model import search_length_prior as lp

    assert lp.LENGTH_PRIOR_PROFILE == "p14-uniform-2-11-v1"
    assert lp.eos_probability(0, 12) == 0.0
    assert lp.eos_probability(1, 12) == 0.0
    # Target: P(content = L) = 1/10 for L in 2..11 at max_len 12.
    q = [lp.eos_probability(step, 12) for step in range(12)]
    survive = 1.0
    for content in range(2, 12):
        assert abs(survive * q[content] - 0.1) < 1e-9, content
        survive *= 1.0 - q[content]
    assert abs(q[11] - 1.0) < 1e-9  # the cap position forces completion
    # Degradation: below 4 positions the profile is inactive (no targets).
    assert lp.sample_target_content_length(_random.Random(1), 3) is None
    rng = _random.Random(7)
    draws = [lp.sample_target_content_length(rng, 12) for _ in range(400)]
    assert set(draws) == set(range(2, 12))  # the full preregistered range
