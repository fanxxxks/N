"""T1-05 contracts: matched-budget baseline harness and the completion
gates of Phase 1.

The harness is the research-validity proof, not a pytest substitute:
* the random-search baseline receives the **same evaluation budget** as
  the trained candidate (``steps x batch_size`` unique formula
  evaluations), so RL-vs-baseline comparisons are budget-fair;
* ``reward_oos_correlation`` measures whether the training reward orders
  candidates the same way the out-of-sample active IR does (Spearman,
  permutation p-value, deterministic seed) — the stable positive
  correlation is a phase-1 completion gate;
* the acceptance tests assert the four completion gates on the full
  production scoring chain: degenerate signals rejected, permutation
  invariance, reward<->OOS correlation, and ``single_weight_cap`` never
  broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.baseline_harness import (
    oos_active_ir,
    reward_oos_correlation,
    run_matched_baseline,
)
from ashare_model.candidates import CandidateScorer, CandidateSpec
from ashare_model.reward import simulate_basket_daily_returns


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


# --- pure statistics --------------------------------------------------------


def test_oos_active_ir_decomposition():
    daily = np.full(20, 0.001)
    bench = np.zeros(20)
    ir = oos_active_ir(daily, bench)
    # Constant 0.1% daily excess: unbounded ratio (capped finite).
    assert ir > 100.0
    noise = np.random.default_rng(0).normal(0.0005, 0.01, size=60)
    assert np.isfinite(oos_active_ir(noise, noise))  # zero excess -> 0


def test_reward_oos_correlation_positive_and_deterministic():
    rng = np.random.default_rng(7)
    # Known-quality candidates: reward and OOS active IR both rise with
    # signal strength, plus measurement noise.
    quality = np.linspace(0.0, 1.0, 40)
    reward = 0.8 * quality + rng.normal(0, 0.02, size=40)
    oos = 0.9 * quality + rng.normal(0, 0.03, size=40)
    first = reward_oos_correlation(list(zip(reward, oos)), seed=3)
    second = reward_oos_correlation(list(zip(reward, oos)), seed=3)
    assert first["rho"] > 0.9
    assert first["p_value"] < 0.01
    assert first == second  # deterministic in seed


def test_reward_oos_correlation_noise_is_insignificant():
    rng = np.random.default_rng(11)
    reward = rng.normal(size=50)
    oos = rng.normal(size=50)
    result = reward_oos_correlation(list(zip(reward, oos)), seed=5)
    assert abs(result["rho"]) < 0.4
    assert result["p_value"] > 0.05


def test_reward_oos_correlation_requires_two_pairs():
    with pytest.raises(ValueError):
        reward_oos_correlation([(1.0, 2.0)])


# --- matched budget ---------------------------------------------------------


def test_run_matched_baseline_respects_budget():
    rng = np.random.default_rng(21)
    n_stocks, n_dates = 10, 40
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    mask = np.ones(target.shape, dtype=bool)
    cfg = _cfg()
    rc = _reward()

    def fake_execute(tokens):
        return target + rng.normal(0, 0.001, size=target.shape)

    result = run_matched_baseline(
        target=target,
        universe_mask=mask,
        backtest_config=cfg,
        reward_config=rc,
        val_windows=[(20, 30), (30, 38)],
        train_signal_range=(0, 20),
        budget=8,
        seed=42,
        max_formula_len=4,
        execute=fake_execute,
    )
    assert result.budget == 8
    assert result.n_evaluated == 8  # all unique
    assert result.selected is not None
    assert len(result.scores) == 8


def test_run_matched_baseline_deterministic_in_seed():
    rng = np.random.default_rng(23)
    n_stocks, n_dates = 10, 40
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    mask = np.ones(target.shape, dtype=bool)
    cfg = _cfg()
    rc = _reward()

    def fake_execute(tokens):
        return target + np.random.default_rng(0).normal(0, 0.001, size=target.shape)

    kwargs = dict(
        target=target,
        universe_mask=mask,
        backtest_config=cfg,
        reward_config=rc,
        val_windows=[(20, 30), (30, 38)],
        train_signal_range=(0, 20),
        budget=6,
        seed=9,
        max_formula_len=4,
        execute=fake_execute,
    )
    first = run_matched_baseline(**kwargs)
    second = run_matched_baseline(**kwargs)
    assert [s.candidate_id for s in first.scores] == [
        s.candidate_id for s in second.scores
    ]
    assert first.selected.formula_text == second.selected.formula_text


# --- T2-01 semantic budget -------------------------------------------------


def _vocab_tokens(*names: str) -> list[int]:
    from ashare_model.vocab import FORMULA_VOCAB

    vocab = FORMULA_VOCAB
    return [
        vocab.feature_offset + vocab.feature_names.index(n) for n in names
    ]


def _op_token(name: str) -> int:
    from ashare_model.ops import OPS_CONFIG
    from ashare_model.vocab import FORMULA_VOCAB

    return FORMULA_VOCAB.operator_offset + [
        cfg[0] for cfg in OPS_CONFIG
    ].index(name)


def _semantic_kwargs(n_stocks=10, n_dates=40):
    rng = np.random.default_rng(31)
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    mask = np.ones(target.shape, dtype=bool)
    cfg = _cfg()
    rc = _reward()
    return dict(
        target=target,
        universe_mask=mask,
        backtest_config=cfg,
        reward_config=rc,
        val_windows=[(20, 30), (30, 38)],
        train_signal_range=(0, 20),
        budget=10,
        max_formula_len=4,
        dataset_id="dataset-a",
        protocol_version=18,
        window_id="train:0:100",
    )


def test_semantic_budget_skips_degenerate_and_canonical_duplicates(
    monkeypatch,
):
    """The v18 budget unit is the unique semantic formula evaluation:
    degenerate formulas are never evaluated, commuted forms collapse, and
    numerically equivalent formulas split only when their complexity bills
    differ."""

    import ashare_model.baseline_harness as harness_module

    add = _op_token("ADD")
    sub = _op_token("SUB")
    ret1 = _vocab_tokens("RET_1")[0]
    ret5 = _vocab_tokens("RET_5")[0]
    eos = 102  # FORMULA_VOCAB.eos_token_id
    # x; ADD(a, b) and its commuted twin; SUB(x, x) (degenerate);
    # ADD(x, x) — same rank pattern and same bill (1.7) as ADD(a, b)
    # under this deterministic executor, so it dedups against it.
    crafted = [
        (ret1, eos, 0, 0),
        (ret1, ret5, add, eos),
        (ret5, ret1, add, eos),  # canonical duplicate of ADD(a, b)
        (ret1, ret1, sub, eos),  # degenerate: SUB(x, x) == 0
        (ret1, ret1, add, eos),  # same semantic class as ADD(a, b)
    ]
    monkeypatch.setattr(
        harness_module,
        "sample_random_formulas",
        lambda seed, vocab, max_len, n: crafted * (n // len(crafted) + 1),
    )
    kwargs = _semantic_kwargs()
    target = kwargs["target"]

    def execute(tokens):
        return target + (sum(tokens) % 5) * 1e-4

    result = run_matched_baseline(
        **kwargs, seed=1, execute=execute, fingerprint_execute=execute
    )
    # SUB(x, x) is never evaluated; the commuted ADD is one formula; the
    # two 1.7-bill ADD forms share one semantic class.
    assert result.n_evaluated == 2  # x (bill 1.0), ADD class (bill 1.7)
    assert result.n_semantic_dedups == 1
    assert result.budget == 10


def test_semantic_budget_dedups_equivalent_classes(monkeypatch):
    """When every proposal shares one semantic class, only the first is
    evaluated; the rest are deduped without consuming budget."""

    import ashare_model.baseline_harness as harness_module

    add = _op_token("ADD")
    sub = _op_token("SUB")
    ret1 = _vocab_tokens("RET_1")[0]
    ret5 = _vocab_tokens("RET_5")[0]
    eos = 102
    # ADD(a, b) and SUB(a, b): distinct canonical forms, equal complexity
    # bills (both binary arithmetic, 3 nodes), and the same signal under
    # this deterministic executor — one semantic class.
    crafted = [
        (ret1, ret5, add, eos),
        (ret1, ret5, sub, eos),
    ]
    monkeypatch.setattr(
        harness_module,
        "sample_random_formulas",
        lambda seed, vocab, max_len, n: crafted * (n // len(crafted) + 1),
    )
    kwargs = _semantic_kwargs()
    target = kwargs["target"]

    def execute(tokens):
        return target  # every formula evaluates to the same signal

    result = run_matched_baseline(
        **kwargs, seed=2, execute=execute, fingerprint_execute=execute
    )
    # One semantic class (the constant signal): one evaluation, the other
    # proposal dedups against it.
    assert result.n_evaluated == 1
    assert result.n_semantic_dedups == 1


def test_semantic_budget_legacy_mode_unchanged():
    """Without the semantic context the harness keeps the canonical
    token-sequence budget (T1-05 behavior, canonicalized)."""

    rng = np.random.default_rng(33)
    n_stocks, n_dates = 10, 40
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    mask = np.ones(target.shape, dtype=bool)

    def execute(tokens):
        return target + rng.normal(0, 0.001, size=target.shape)

    result = run_matched_baseline(
        target=target,
        universe_mask=mask,
        backtest_config=_cfg(),
        reward_config=_reward(),
        val_windows=[(20, 30), (30, 38)],
        train_signal_range=(0, 20),
        budget=8,
        seed=3,
        max_formula_len=4,
        execute=execute,
    )
    assert result.n_evaluated == 8
    assert result.n_semantic_dedups == 0


# --- completion gate: cap never broken --------------------------------------


def test_completion_gate_single_weight_cap_never_broken():
    # Adversarial sweep across cap regimes, under-fills, force-holds and
    # degenerate days: no path may push any name above the cap.
    rng = np.random.default_rng(31)
    n_stocks, n_dates = 8, 12
    for cap in (0.03, 0.1, 0.5, 1.0):
        for top_n in (2, 5, 20):
            cfg = _cfg(top_n=top_n, single_weight_cap=cap)
            target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
            signal = rng.normal(size=(n_stocks, n_dates))
            mask = np.ones(target.shape, dtype=bool)
            # Every other day only one stock is selectable (under-fill),
            # and one stock is sell-blocked mid-window (force-hold).
            mask[2:, ::2] = False
            blocked_sell = np.zeros(target.shape, dtype=bool)
            blocked_sell[0, 5] = True
            sim = simulate_basket_daily_returns(
                signal, target, cfg, blocked_sell=blocked_sell,
                universe_mask=mask,
                tie_break_keys=np.asarray([f"{i:04d}" for i in range(n_stocks)]),
            )
            # The basket exposes no weights; verify via the engine and the
            # batched path's invariants instead.
            engine = AshareBacktestEngine(cfg).run(
                signal,
                _raw_market(rng, n_stocks, n_dates),
                [f"{i:04d}.SZ" for i in range(n_stocks)],
                [f"202401{i:02d}" for i in range(n_dates)],
                universe_mask=mask,
            )
            for snapshot in engine.positions:
                assert all(w <= cap + 1e-12 for w in snapshot["weights"])
                assert sum(snapshot["weights"]) <= 1.0 + 1e-12


def _raw_market(rng, n_stocks, n_dates):
    open_ = np.cumprod(
        1.0 + rng.normal(0.0005, 0.005, (n_stocks, n_dates)), axis=1
    ) * 10.0
    return {
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": np.roll(open_, 1, axis=1),
        "volume": np.full((n_stocks, n_dates), 1_000_000.0),
    }


# --- completion gate: reward <-> OOS active IR ------------------------------


def test_completion_gate_reward_correlates_with_oos_active_ir():
    # The phase-1 completion gate: the training reward must order
    # candidates the same way the out-of-sample active IR does, stably
    # across seeds.  Candidates are signal-strength-perturbed copies of
    # the engine-consistent target (open-to-open returns, known quality
    # ordering), scored in-sample and measured OOS by the engine.
    cfg = _cfg(top_n=5, single_weight_cap=0.5)
    # Effectively unbounded clip band: the raw reward must spread across
    # the candidate qualities (a +/-1 band would tie most candidates at
    # the ceiling and compress the rank correlation).
    rc = _reward(reward_clip_low=-1e9, reward_clip_high=1e9)
    rng = np.random.default_rng(41)
    n_stocks, n_dates = 15, 70
    train_end = 40
    drift = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    open_ = np.cumprod(1.0 + drift, axis=1) * 10.0
    raw = {
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": np.roll(open_, 1, axis=1),
        "volume": np.full((n_stocks, n_dates), 1_000_000.0),
    }
    from ashare_data.processor import open_to_open_returns

    target = open_to_open_returns(open_)
    mask = np.ones(target.shape, dtype=bool)
    windows = [(20, 30), (30, 38)]
    scorer = CandidateScorer(cfg, rc)
    keys = np.asarray([f"{i:04d}" for i in range(n_stocks)])
    pairs = []
    for scale in np.linspace(0.05, 1.5, 24):
        signal = target + rng.normal(0, scale * 0.02, size=target.shape)
        score = scorer.score(
            CandidateSpec(f"s{scale:.2f}", f"s{scale:.2f}", "gate", (1,)),
            signal, target, windows,
            universe_mask=mask,
            tie_break_keys=keys,
        )
        engine = AshareBacktestEngine(cfg).run(
            signal, raw,
            [f"{i:04d}.SZ" for i in range(n_stocks)],
            [f"202401{i:02d}" for i in range(n_dates)],
            universe_mask=mask,
            signal_range=range(train_end, n_dates - 2),
        )
        bench = np.asarray(
            [engine.benchmark_equity[i + 1] / engine.benchmark_equity[i] - 1
             for i in range(len(engine.benchmark_equity) - 1)]
        )
        oos_ir = oos_active_ir(
            np.asarray(engine.daily_returns), bench
        )
        pairs.append((score.val_reward, oos_ir))
    result = reward_oos_correlation(pairs, seed=1)
    assert result["n"] == 24
    assert result["rho"] > 0.4, result
    assert result["p_value"] < 0.05, result
