"""Phase-1 completion-gate acceptance tests (T1-05).

The four completion gates, asserted on the full production scoring chain
with the **default** production configuration (no fixture relaxations):

1. all-zero, two-valid-day and extremely sparse signals are rejected;
2. stock-row permutation never changes any measurement;
3. the reward correlates stably positively with OOS active IR (the
   dedicated harness test in test_baseline_harness);
4. ``single_weight_cap`` is never broken (the dedicated sweep in
   test_baseline_harness).

These tests are the "research validity" acceptance layer: they run the
same scorer/engine the production pipeline uses, not a simplified proxy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_model.candidates import CandidateScorer, CandidateSpec


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=100000.0,
        top_n=10,
        single_weight_cap=0.1,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _default_reward() -> RewardConfig:
    """The production default gates (T1-03), untouched by fixtures."""
    return RewardConfig()


def _market(n_stocks=40, n_dates=80, seed=3):
    rng = np.random.default_rng(seed)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.001, size=target.shape)
    mask = np.ones(target.shape, dtype=bool)
    return signal, target, mask


def _score(signal, target, mask, tokens=(1,)):
    scorer = CandidateScorer(_cfg(), _default_reward())
    windows = [(40, 60), (60, 78)]
    return scorer.score(
        CandidateSpec("gate", "gate", "gate", tokens),
        signal, target, windows,
        universe_mask=mask,
        tie_break_keys=np.asarray([f"{i:04d}" for i in range(signal.shape[0])]),
    )


def test_gate_all_zero_signal_rejected():
    signal, target, mask = _market()
    zero = np.zeros_like(signal)
    score = _score(zero, target, mask)
    assert not score.eligible
    assert "constant_or_near_constant_signal" in score.rejection_reasons


def test_gate_two_valid_day_signal_rejected():
    signal, target, mask = _market()
    sparse = np.full_like(signal, np.nan)
    sparse[:, :2] = np.tile(
        np.arange(signal.shape[0], dtype=float)[:, None], (1, 2)
    )
    score = _score(sparse, target, mask)
    assert not score.eligible
    # Rejected by the near-constant gate (all values equal) or the hard
    # valid-IC-days gate — both are rejections of the degenerate class.
    assert score.rejection_reasons


def test_gate_extremely_sparse_signal_rejected():
    signal, target, mask = _market()
    # Only five stocks selectable with the production ic_min_stocks=10:
    # the effective-stocks gate rejects.
    sparse_mask = np.zeros(mask.shape, dtype=bool)
    sparse_mask[:5] = True
    score = _score(signal, target, sparse_mask)
    assert not score.eligible
    assert "effective_stocks_below_minimum" in score.rejection_reasons


def test_gate_healthy_signal_passes_default_gates():
    signal, target, mask = _market()
    score = _score(signal, target, mask)
    assert score.eligible, score.rejection_reasons
    assert score.complexity_cost == pytest.approx(1.0)


def test_gate_permutation_invariance_full_chain():
    signal, target, mask = _market(n_stocks=30, n_dates=70, seed=8)
    keys = np.asarray([f"{i:04d}" for i in range(signal.shape[0])])
    scorer = CandidateScorer(_cfg(), _default_reward())
    windows = [(40, 60), (60, 68)]
    base = scorer.score(
        CandidateSpec("gate", "gate", "gate", (1,)),
        signal, target, windows,
        universe_mask=mask,
        tie_break_keys=keys,
    )
    order = np.random.default_rng(2).permutation(signal.shape[0])
    other = scorer.score(
        CandidateSpec("gate", "gate", "gate", (1,)),
        signal[order], target[order], windows,
        universe_mask=mask[order],
        tie_break_keys=keys[order],
    )
    for field in (
        "val_reward", "val_icir", "train_reward", "train_icir",
        "active_ir", "risk_exposure", "average_turnover",
        "eligible", "rejection_reasons",
    ):
        if isinstance(getattr(base, field), float):
            # Float summation order over permuted rows leaves ~1e-13
            # noise; the measurement itself is invariant.
            assert getattr(base, field) == pytest.approx(
                getattr(other, field), abs=1e-9, rel=1e-9
            ), field
        else:
            assert getattr(base, field) == getattr(other, field), field
