"""T1-03 contracts: robust IC statistics, hard signal-quality gates and
AST complexity billing.

* ICIR uses effective-sample-size shrinkage and HAC (Newey-West-style)
  variance: autocorrelated IC series are worth less than their raw length.
* The block bootstrap provides the sign-stability probability of the mean
  IC (P(mean > 0)) and the bootstrap CI, with a deterministic seed.
* Hard gates reject the degenerate classes — all-zero, two-valid-day and
  extremely sparse signals — by coverage, valid-IC days, effective stock
  count and signal activity.
* Complexity is billed from the AST: node count, depth, longest operator
  window and per-operator cost, with a monotone bill and a hard cap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_model.candidates import CandidateScorer, CandidateSpec
from ashare_model.complexity import (
    complexity_bill,
    longest_window,
    max_depth,
    node_count,
    operation_cost,
)
from ashare_model.ir import decode
from ashare_model.reward import rank_ic_series
from ashare_model.signal_quality import (
    SignalQualityGates,
    block_bootstrap_ic,
    effective_n,
    effective_stock_count,
    evaluate_quality_gates,
    hac_variance,
    robust_icir,
    summarize_signal_quality_batch,
)
from ashare_model.vocab import FORMULA_VOCAB


def _ic_series(n=120, rho=0.0, seed=3):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.05, size=n)
    series = np.empty(n)
    series[0] = noise[0]
    for i in range(1, n):
        series[i] = rho * series[i - 1] + noise[i]
    return series


# --- HAC variance / effective-n / robust ICIR ------------------------------


def test_hac_variance_matches_naive_for_independent_series():
    ic = _ic_series(n=200, rho=0.0)
    naive = float(np.var(ic, ddof=1))
    hac = hac_variance(ic)
    assert hac == pytest.approx(naive, rel=0.15)
    assert effective_n(ic) == pytest.approx(len(ic), rel=0.15)


def test_hac_variance_inflates_for_autocorrelated_series():
    ic = _ic_series(n=300, rho=0.6)
    hac = hac_variance(ic)
    naive = float(np.var(ic, ddof=1))
    assert hac > naive * 1.5
    assert effective_n(ic) < len(ic) * 0.7


def test_robust_icir_shrinks_with_autocorrelation():
    ic = _ic_series(n=300, rho=0.6)
    ic = ic + 0.01  # positive mean
    naive = float(ic.mean()) / float(ic.std(ddof=1))
    robust = robust_icir(ic)
    eff = effective_n(ic)
    # Exact identity: icir_hac = naive * sqrt(n_eff / n).
    assert robust == pytest.approx(naive * math.sqrt(eff / len(ic)), rel=1e-9)
    # Shrinkage: autocorrelation discounts the ratio below the naive one.
    assert 0.0 < robust < naive


def test_robust_icir_matches_naive_for_independent_series():
    ic = _ic_series(n=200, rho=0.0) + 0.02
    naive = float(ic.mean()) / float(ic.std(ddof=1))
    # HAC lag estimation adds sampling noise: the ratio stays ~naive.
    assert robust_icir(ic) == pytest.approx(naive, rel=0.25)
    # Independent series keep ~all their sample size.
    assert effective_n(ic) == pytest.approx(len(ic), rel=0.25)


def test_robust_icir_perfect_stable_ic_is_unbounded():
    ic = np.ones(50) * 0.03
    assert robust_icir(ic) == float("inf")
    assert robust_icir(-ic) == float("-inf")


def test_effective_stock_count_is_harmonic_mean():
    assert effective_stock_count(np.array([100, 100])) == pytest.approx(100.0)
    assert effective_stock_count(np.array([100, 50])) == pytest.approx(200.0 / 3.0)
    assert effective_stock_count(np.array([])) == 0.0
    assert effective_stock_count(np.array([0, 0])) == 0.0


# --- block bootstrap --------------------------------------------------------


def test_block_bootstrap_deterministic_and_bounded():
    ic = _ic_series(n=100, rho=0.0)
    first = block_bootstrap_ic(ic, n_boot=200, block_size=5, seed=42)
    second = block_bootstrap_ic(ic, n_boot=200, block_size=5, seed=42)
    assert first["sign_stability"] == second["sign_stability"]
    assert 0.0 <= first["sign_stability"] <= 1.0
    assert first["ci_low"] < first["ci_high"]
    assert first["n_boot"] == 200


def test_block_bootstrap_sign_stability_strong_signal():
    ic = _ic_series(n=150, rho=0.3) + 0.05  # strongly positive mean
    result = block_bootstrap_ic(ic, n_boot=300, block_size=5, seed=7)
    assert result["sign_stability"] > 0.95
    assert result["mean"] > 0.0


def test_block_bootstrap_noise_is_balanced():
    ic = np.random.default_rng(11).normal(0.0, 0.05, size=200)
    result = block_bootstrap_ic(ic, n_boot=300, block_size=5, seed=3)
    assert 0.05 <= result["sign_stability"] <= 0.95


# --- hard quality gates ----------------------------------------------------


def _gates(**kwargs) -> SignalQualityGates:
    defaults = dict(
        min_valid_ic_days=8,
        min_effective_stocks=3,
        min_coverage=0.2,
        min_activity=0.05,
        min_sign_stability=0.5,
        min_val_window_q25=0.0,
    )
    defaults.update(kwargs)
    return SignalQualityGates(**defaults)


def test_gates_reject_all_zero_signal():
    gates = _gates()
    n_stocks, n_dates = 20, 40
    signal = np.zeros((n_stocks, n_dates))
    target = np.random.default_rng(0).normal(0.001, 0.01, size=(n_stocks, n_dates))
    mask = np.ones(signal.shape, dtype=bool)
    summary = summarize_signal_quality_batch(
        signal[None], target, [(0, n_dates - 2)], (0, n_dates - 2),
        universe_mask=mask, ic_min_stocks=5,
    )[0]
    ok, reasons = evaluate_quality_gates(summary, gates)
    assert not ok
    assert "signal_activity_below_minimum" in reasons


def test_gates_reject_two_valid_ic_days():
    gates = _gates()
    n_stocks, n_dates = 20, 30
    target = np.random.default_rng(1).normal(0.001, 0.01, size=(n_stocks, n_dates))
    # The signal is valid on exactly the first two dates and NaN elsewhere.
    signal = np.full((n_stocks, n_dates), np.nan)
    signal[:, :2] = np.random.default_rng(2).normal(size=(n_stocks, 2))
    mask = np.ones(signal.shape, dtype=bool)
    summary = summarize_signal_quality_batch(
        signal[None], target, [(0, n_dates - 2)], (0, n_dates - 2),
        universe_mask=mask, ic_min_stocks=5,
    )[0]
    assert summary.valid_ic_days <= 2
    ok, reasons = evaluate_quality_gates(summary, gates)
    assert not ok
    assert "valid_ic_days_below_minimum" in reasons


def test_gates_reject_extremely_sparse_cross_section():
    gates = _gates(min_effective_stocks=10)
    n_stocks, n_dates = 50, 40
    target = np.random.default_rng(3).normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = np.random.default_rng(4).normal(size=(n_stocks, n_dates))
    # Only five stocks are selectable: effective stock count ~5 < 10.
    mask = np.zeros(signal.shape, dtype=bool)
    mask[:5] = True
    summary = summarize_signal_quality_batch(
        signal[None], target, [(0, n_dates - 2)], (0, n_dates - 2),
        universe_mask=mask, ic_min_stocks=3,
    )[0]
    assert summary.effective_stocks < 10
    ok, reasons = evaluate_quality_gates(summary, gates)
    assert not ok
    assert "effective_stocks_below_minimum" in reasons


def test_gates_reject_sign_instability():
    # One aligned window, three inverted windows: sign stability 0.25 < 0.5.
    gates = _gates(min_sign_stability=0.5)
    n_stocks, n_dates = 12, 60
    target = np.random.default_rng(5).normal(0.001, 0.01, size=(n_stocks, n_dates))
    aligned = target + np.random.default_rng(6).normal(0, 0.001, size=target.shape)
    inverted = -aligned
    signal = np.concatenate([aligned[:, :12], inverted[:, 12:24],
                             inverted[:, 24:36], inverted[:, 36:48],
                             inverted[:, 48:60]], axis=1)
    mask = np.ones(signal.shape, dtype=bool)
    windows = [(0, 12), (12, 24), (24, 36), (36, 48)]
    summary = summarize_signal_quality_batch(
        signal[None], target, windows, (0, 58),
        universe_mask=mask, ic_min_stocks=3,
    )[0]
    ok, reasons = evaluate_quality_gates(summary, gates)
    assert not ok
    assert "sign_instability" in reasons


def test_gates_reject_low_window_q25():
    gates = _gates(min_val_window_q25=0.0)
    n_stocks, n_dates = 12, 60
    target = np.random.default_rng(5).normal(0.001, 0.01, size=(n_stocks, n_dates))
    aligned = target + np.random.default_rng(6).normal(0, 0.001, size=target.shape)
    signal = np.concatenate([aligned[:, :30], -aligned[:, 30:60]], axis=1)
    mask = np.ones(signal.shape, dtype=bool)
    windows = [(0, 15), (15, 30), (30, 45), (45, 58)]
    summary = summarize_signal_quality_batch(
        signal[None], target, windows, (0, 58),
        universe_mask=mask, ic_min_stocks=3,
    )[0]
    ok, reasons = evaluate_quality_gates(summary, gates)
    assert not ok
    assert "val_window_q25_below_minimum" in reasons


def test_gates_pass_healthy_signal():
    gates = _gates()
    n_stocks, n_dates = 20, 60
    rng = np.random.default_rng(9)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.0005, size=target.shape)
    mask = np.ones(signal.shape, dtype=bool)
    windows = [(0, 15), (15, 30), (30, 45), (45, 58)]
    summary = summarize_signal_quality_batch(
        signal[None], target, windows, (0, 58),
        universe_mask=mask, ic_min_stocks=5,
    )[0]
    ok, reasons = evaluate_quality_gates(summary, gates)
    assert ok, reasons
    assert summary.valid_ic_days >= 50
    assert summary.effective_stocks == pytest.approx(20.0)
    assert summary.sign_stability == 1.0
    assert summary.window_icir_q25 is not None


def test_summary_reports_robust_icir_and_eff_n():
    n_stocks, n_dates = 15, 50
    rng = np.random.default_rng(12)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.0005, size=target.shape)
    mask = np.ones(signal.shape, dtype=bool)
    summary = summarize_signal_quality_batch(
        signal[None], target, [(0, n_dates - 2)], (0, n_dates - 2),
        universe_mask=mask, ic_min_stocks=5,
    )[0]
    assert summary.ic_mean > 0.0
    assert summary.icir_hac is not None and np.isfinite(summary.icir_hac)
    assert 0 < summary.effective_n <= summary.valid_ic_days
    assert summary.activity > 0.5
    assert summary.coverage == pytest.approx(1.0)


# --- complexity billing -----------------------------------------------------


def _feat(name: str) -> int:
    return FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name)


def _op(name: str) -> int:
    return FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index(name)


def _tokens(parts: list[str]) -> "object":
    """Decode a postfix formula given as token names."""
    tokens = []
    for part in parts:
        if part.startswith("F:"):
            tokens.append(_feat(part[2:]))
        else:
            tokens.append(_op(part))
    return decode(tokens, FORMULA_VOCAB)


def test_complexity_metrics_on_known_ast():
    ast = _tokens(["F:RET_1", "F:RET_5", "ADD"])
    assert node_count(ast) == 3
    assert max_depth(ast) == 2  # root + two leaves
    assert longest_window(ast) == 0
    assert operation_cost(ast) == 1


def test_longest_window_tracks_operator_windows():
    ma60 = _tokens(["F:RET_1", "MA60"])
    ma5 = _tokens(["F:RET_1", "MA5"])
    assert longest_window(ma60) == 60
    assert longest_window(ma5) == 5
    assert operation_cost(ma60) == 2  # windowed ops cost more than arithmetic
    assert complexity_bill(ma60) > complexity_bill(ma5)


def test_complexity_bill_is_monotone():
    bare = _tokens(["F:RET_1"])  # bill exactly 1.0
    small = _tokens(["F:RET_1", "F:RET_5", "ADD"])  # 3 nodes, no window
    deep = _tokens(["F:RET_1", "F:RET_5", "ADD", "F:RET_10", "ADD"])
    wide = _tokens(["F:RET_1", "MA60"])
    assert complexity_bill(deep) > complexity_bill(small)
    assert complexity_bill(wide) > complexity_bill(bare)
    assert complexity_bill(small) >= 1.0


def test_bare_feature_bill_is_one():
    ast = _tokens(["F:RET_1"])
    assert node_count(ast) == 1
    assert complexity_bill(ast) == 1.0


# --- scorer integration -----------------------------------------------------


def _bt(**kwargs) -> BacktestConfig:
    values = dict(
        initial_capital=100000.0,
        top_n=3,
        single_weight_cap=0.5,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )
    values.update(kwargs)
    return BacktestConfig(**values)


def _reward(**kwargs) -> RewardConfig:
    values = dict(ic_min_stocks=3, min_val_reward=-1e9, min_val_icir=-1e9)
    values.update(kwargs)
    return RewardConfig(**values)


def _healthy(n_stocks=12, n_dates=40, seed=21):
    rng = np.random.default_rng(seed)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.0005, size=target.shape)
    return signal, target, np.ones(target.shape, dtype=bool)


def test_scorer_rejects_degenerate_classes():
    scorer = CandidateScorer(_bt(), _reward())
    windows = [(20, 30), (30, 38)]
    spec = CandidateSpec("d", "d", "test", (1,))

    # All-zero signal: near-constant rejection (existing) fires first.
    signal, target, mask = _healthy()
    zero = np.zeros_like(signal)
    score = scorer.score(spec, zero, target, windows, universe_mask=mask)
    assert not score.eligible
    assert "constant_or_near_constant_signal" in score.rejection_reasons

    # Two-valid-day signal with distinct values: the hard IC-days gate
    # rejects it (a constant two-day signal would be caught by the
    # near-constant rejection instead, which is equally a rejection).
    signal, target, mask = _healthy()
    sparse = np.full_like(signal, np.nan)
    sparse[:, :2] = np.tile(
        np.arange(signal.shape[0], dtype=float)[:, None], (1, 2)
    )
    score = scorer.score(spec, sparse, target, windows, universe_mask=mask)
    assert not score.eligible
    assert "valid_ic_days_below_minimum" in score.rejection_reasons

    # Extremely sparse cross-section (2 of 12 eligible, min_stocks 3):
    # effective stock count below the IC requirement.
    signal, target, mask = _healthy()
    mask = np.zeros_like(mask)
    mask[:2] = True
    score = scorer.score(spec, signal, target, windows, universe_mask=mask)
    assert not score.eligible
    assert "effective_stocks_below_minimum" in score.rejection_reasons


def test_scorer_passes_healthy_signal_with_hard_gates():
    scorer = CandidateScorer(_bt(), _reward())
    signal, target, mask = _healthy()
    windows = [(20, 30), (30, 38)]
    score = scorer.score(
        CandidateSpec("h", "h", "test", (1,)),
        signal, target, windows, universe_mask=mask,
    )
    assert score.eligible
    assert score.complexity_cost == pytest.approx(1.0)
    # v15 (contract docs/p11_reward_v15_contract.md §5.2/§6.3): the bare
    # factor's bill (1.0) sits inside the complexity_free_bill free zone
    # (default 3.0), so the scorer charges nothing.
    assert score.complexity_penalty == pytest.approx(0.0)


def test_scorer_complexity_gate_rejects_pathological_formula():
    signal, target, mask = _healthy()
    windows = [(20, 30), (30, 38)]
    # A long windowed formula: RET_1 MA60 RET_5 MA60 ADD RET_10 ADD.
    tokens = tuple(
        [_feat("RET_1"), _op("MA60"), _feat("RET_5"), _op("MA60"), _op("ADD"),
         _feat("RET_10"), _op("ADD")]
    )
    bill = complexity_bill(decode(list(tokens), FORMULA_VOCAB))
    scorer = CandidateScorer(_bt(), _reward(max_complexity=max(1.0, bill - 0.5)))
    score = scorer.score(
        CandidateSpec("c", "c", "test", tokens),
        signal, target, windows, universe_mask=mask,
    )
    assert not score.eligible
    assert "complexity_above_maximum" in score.rejection_reasons


def test_rank_ic_permutation_invariance():
    rng = np.random.default_rng(0)
    n_stocks, n_dates = 10, 30
    target = rng.normal(size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.001, size=target.shape)
    mask = np.ones(target.shape, dtype=bool)
    base = rank_ic_series(signal[None], target, 3, universe_mask=mask)
    order = rng.permutation(n_stocks)
    permuted = rank_ic_series(
        signal[order][None], target[order], 3, universe_mask=mask[order]
    )
    assert np.allclose(base, permuted, equal_nan=True)
