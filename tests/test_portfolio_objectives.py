"""T1-04 contracts: reward/backtest universe alignment, portfolio
objectives and Pareto selection.

* The reward basket selects on the **same** signal-date AND entry-date
  eligibility as the engine (``universe_mask[:, t] & universe_mask[:, t+1]``)
  — a stock that leaves the universe between signal and execution can
  never be bought by the reward, and the basket reproduces the engine
  day-for-day on aligned inputs.
* The reward's primary term is the portfolio **active IR** (gross basket
  returns minus the equal-weight benchmark, effective-n shrunk,
  annualized), with the exact cost drag and the complexity bill
  subtracted; IC stays the auxiliary reported/gated metric.
* Portfolio objectives — active IR, risk exposure, turnover, capacity —
  are recorded per candidate; capacity is gated
  (``capacity_above_maximum``) when dollar-volume data is available.
* Selection prefers constrained/Pareto ranking over the fragile scalar:
  eligibility (hard constraints) first, then non-dominated fronts on the
  four objectives with IC as the deterministic tie-break.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine, equal_weight_benchmark_returns
from ashare_model.candidates import (
    CandidateScore,
    CandidateScorer,
    CandidateSelector,
    CandidateSpec,
)
from ashare_model.pareto import nondominated_fronts
from ashare_model.reward import (
    batched_basket_rewards,
    formula_reward,
    simulate_basket_daily_returns,
)
from tests.conftest import with_price_semantics


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=100000.0,
        top_n=3,
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
        ic_min_stocks=3,
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


def _market(n_stocks=6, n_dates=12, seed=17):
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = [f"202401{i:02d}" for i in range(n_dates)]
    open_ = np.cumprod(
        1.0 + rng.normal(0.0005, 0.005, (n_stocks, n_dates)), axis=1
    ) * 10.0
    pre_close = np.roll(open_, 1, axis=1)
    pre_close[:, 0] = open_[:, 0]
    raw = with_price_semantics({
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": pre_close,
        "volume": np.full((n_stocks, n_dates), 1_000_000.0),
        "close": open_ * 1.005,
    })
    target = open_to_open_returns(open_)
    signal = target + rng.normal(0, 0.001, size=target.shape)
    return codes, dates, raw, signal, target


def _lenient(**kwargs):
    values = dict(
        min_valid_ic_days=2,
        min_effective_stocks=2,
        min_coverage=0.0,
        min_activity=0.0,
        min_sign_stability=0.0,
        min_val_window_q25=-1e9,
    )
    values.update(kwargs)
    return values


# --- alignment: signal-date AND entry-date eligibility ---------------------


def test_basket_aligns_entry_day_universe_like_engine():
    codes, dates, raw, signal, target = _market()
    # Stock 0 carries the top signal at t=0 but leaves the universe at
    # t+1=1: the engine refuses to buy it at the entry day; the basket
    # must too (the pre-T1-04 basket bought on the signal-date mask only).
    signal = signal.copy()
    signal[0] = 1e3
    signal[1] = 1e2
    signal[2:] = np.random.default_rng(3).normal(size=signal[2:].shape)
    mask = np.ones(signal.shape, dtype=bool)
    mask[0, 1:] = False
    cfg = _cfg(top_n=1, single_weight_cap=1.0)
    engine = AshareBacktestEngine(cfg).run(
        signal, raw, codes, dates, universe_mask=mask
    )
    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask
    )
    # P35 (v4): the engine executes whole lots against a cash-capped
    # affordability loop while the basket is the continuous-weight
    # limit, so the daily nets agree only to the execution friction
    # (one-lot grain + fee financing, order 1e-4 per session — the same
    # bound the golden parity contract uses).
    assert sim.daily_net_returns == pytest.approx(
        np.asarray(engine.daily_returns), abs=5e-4
    )
    # The engine buys the second-best stock instead of the exited one.
    assert engine.positions[0]["ts_codes"] == [codes[1]]


def test_basket_engine_parity_on_entry_day_exits():
    # A stock that exits between the signal date and the entry date must
    # never be bought by either path, day for day.
    codes, dates, raw, signal, target = _market(n_stocks=4, n_dates=10)
    mask = np.ones(signal.shape, dtype=bool)
    mask[2, 3:5] = False  # stock 2 ineligible exactly on entry days 3,4
    cfg = _cfg(top_n=2, single_weight_cap=0.5)
    engine = AshareBacktestEngine(cfg).run(
        signal, raw, codes, dates, universe_mask=mask
    )
    sim = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    # P35 (v4): daily nets agree to the whole-lot execution friction
    # (see test_basket_aligns_entry_day_universe_like_engine).  Turnover
    # also drifts apart by one round of that friction: the engine feeds
    # the constructor its ACHIEVED (lot-rounded) book as the next prev
    # while the basket compounds the continuous weights, so the next
    # session's weight deltas differ by the carried lot residual.
    assert sim.daily_net_returns == pytest.approx(
        np.asarray(engine.daily_returns), abs=5e-4
    )
    assert sim.turnover == pytest.approx(np.asarray(engine.turnover), abs=5e-2)


# --- reward v13: active IR as the primary term ------------------------------


def test_reward_is_active_ir_minus_cost_minus_complexity():
    rng = np.random.default_rng(5)
    n_stocks, n_dates = 10, 40
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.0005, size=target.shape)
    mask = np.ones(target.shape, dtype=bool)
    cfg = _cfg(top_n=3, single_weight_cap=0.5)
    rc = _reward(complexity_penalty=0.0)
    signal_range = (0, n_dates - 2)

    from ashare_model.signal_quality import robust_icir

    sim = simulate_basket_daily_returns(
        signal, target, cfg, signal_range=signal_range, universe_mask=mask
    )
    bench = np.asarray(
        equal_weight_benchmark_returns(
            target, list(range(signal_range[0], signal_range[1])), mask
        )
    )
    active = sim.daily_gross_returns - bench
    expected_ir = robust_icir(active, rc.ic_hac_max_lags) * np.sqrt(252)
    expected_cost = float(sim.daily_cost_fractions.mean()) * 252
    expected = float(
        np.clip(
            expected_ir - expected_cost,
            rc.reward_clip_low,
            rc.reward_clip_high,
        )
    )
    reward = formula_reward(signal, target, cfg, rc, universe_mask=mask)
    assert reward == pytest.approx(expected, abs=1e-9)


def test_active_ir_orders_signals_by_portfolio_quality():
    rng = np.random.default_rng(9)
    n_stocks, n_dates = 12, 40
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    mask = np.ones(target.shape, dtype=bool)
    cfg = _cfg(top_n=3, single_weight_cap=0.5)
    rc = _reward(complexity_penalty=0.0, cost_weight=0.0)
    strong = target + rng.normal(0, 0.0002, size=target.shape)
    weak = rng.normal(0, 1.0, size=target.shape)
    strong_r = formula_reward(strong, target, cfg, rc, universe_mask=mask)
    weak_r = formula_reward(weak, target, cfg, rc, universe_mask=mask)
    assert strong_r > weak_r


def test_batched_objectives_recorded_per_candidate():
    rng = np.random.default_rng(11)
    n_stocks, n_dates = 8, 30
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signals = np.stack(
        [
            target + rng.normal(0, 0.0005, size=target.shape),
            rng.normal(size=target.shape),
        ]
    )
    mask = np.ones(target.shape, dtype=bool)
    cfg = _cfg(top_n=2, single_weight_cap=0.5)
    rc = _reward()
    windows = [(10, 20), (20, 28)]
    out = batched_basket_rewards(
        signals, target, cfg, rc, windows, universe_mask=mask
    )
    assert len(out) == 5
    rewards, val_rewards, icir, val_icir, objectives = out
    assert objectives.shape == (2, 8)
    # Train columns: active IR, exposure, turnover, capacity.
    assert np.isfinite(objectives[:, 0]).all()
    assert objectives[:, 1].min() >= 0.0  # vol is non-negative
    assert objectives[:, 2].min() >= 0.0  # turnover is non-negative
    assert np.isnan(objectives[:, 3]).all()  # no adv provided
    # Val columns (median over windows) are finite for the strong signal.
    assert np.isfinite(objectives[0, 4])
    # The strong signal's val active IR beats the noise signal's.
    assert objectives[0, 4] > objectives[1, 4]


def test_scorer_records_portfolio_objectives():
    rng = np.random.default_rng(13)
    n_stocks, n_dates = 10, 40
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.0005, size=target.shape)
    mask = np.ones(target.shape, dtype=bool)
    scorer = CandidateScorer(_cfg(top_n=3), _reward())
    windows = [(20, 30), (30, 38)]
    score = scorer.score(
        CandidateSpec("o", "o", "test", (1,)),
        signal, target, windows, universe_mask=mask,
    )
    assert np.isfinite(score.active_ir)
    assert score.risk_exposure >= 0.0
    assert score.average_turnover >= 0.0
    assert score.capacity_utilization == 0.0  # no adv: unmeasured
    assert score.eligible


def test_capacity_gate_rejects_illiquid_positions():
    rng = np.random.default_rng(15)
    n_stocks, n_dates = 10, 40
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0, 0.0005, size=target.shape)
    mask = np.ones(target.shape, dtype=bool)
    scorer = CandidateScorer(
        _cfg(top_n=3),
        _reward(max_capacity_utilization=0.05),
    )
    windows = [(20, 30), (30, 38)]
    # Tiny dollar volume: any position is a large fraction of the day's
    # volume, so the capacity gate must reject.
    adv = np.full(target.shape, 1_000.0)
    score = scorer.score(
        CandidateSpec("c", "c", "test", (1,)),
        signal, target, windows, universe_mask=mask, adv=adv,
    )
    assert not score.eligible
    assert "capacity_above_maximum" in score.rejection_reasons
    # Plenty of volume: the same signal passes.
    adv = np.full(target.shape, 1e9)
    ok = scorer.score(
        CandidateSpec("c", "c", "test", (1,)),
        signal, target, windows, universe_mask=mask, adv=adv,
    )
    assert ok.eligible


# --- Pareto selection -------------------------------------------------------


def _score(id_, active_ir, exposure, turnover, capacity, val_icir):
    return CandidateScore(
        tokens=(1,),
        candidate_id=id_,
        formula_text=id_,
        source="test",
        direction=1,
        val_reward=0.0,
        val_icir=val_icir,
        train_reward=0.0,
        train_icir=0.0,
        complexity_penalty=0.0,
        eligible=True,
        rejection_reasons=(),
        complexity_cost=1.0,
        active_ir=active_ir,
        risk_exposure=exposure,
        average_turnover=turnover,
        capacity_utilization=capacity,
    )


PARETO = (
    ("active_ir", 1),
    ("risk_exposure", -1),
    ("average_turnover", -1),
    ("capacity_utilization", -1),
)


def test_nondominated_fronts_basic():
    # A strictly dominates B and C; B dominates C.
    points = [
        {"a": 2.0, "b": 2.0},
        {"a": 3.0, "b": 3.0},
        {"a": 1.0, "b": 0.5},
    ]
    fronts = nondominated_fronts(points, [("a", 1), ("b", 1)])
    assert fronts[0] == [points[1]]
    assert fronts[1] == [points[0]]
    assert fronts[2] == [points[2]]


def test_nondominated_fronts_handles_ties():
    points = [
        {"a": 2.0, "b": 2.0},
        {"a": 2.0, "b": 2.0},
        {"a": 3.0, "b": 3.0},
    ]
    fronts = nondominated_fronts(points, [("a", 1), ("b", 1)])
    assert len(fronts[0]) == 1
    assert fronts[0][0]["a"] == 3.0
    # The two identical points never dominate each other: same front.
    assert fronts[1] == [points[0], points[1]]


def test_selector_pareto_mode_prefers_non_dominated():
    dominated = _score("dom", active_ir=0.2, exposure=0.2, turnover=0.3,
                       capacity=0.1, val_icir=0.5)
    frontier = _score("front", active_ir=0.25, exposure=0.1, turnover=0.1,
                      capacity=0.05, val_icir=0.2)
    result = CandidateSelector().select(
        [dominated, frontier], pareto_objectives=PARETO
    )
    # The dominated candidate has the higher scalar ICIR but is worse on
    # every portfolio objective: Pareto selection must pick the frontier.
    assert result.selected == frontier


def test_selector_pareto_mode_respects_eligibility():
    rejected = CandidateScore(
        tokens=(1,),
        candidate_id="rej",
        formula_text="rej",
        source="test",
        direction=1,
        val_reward=1.0,
        val_icir=0.9,
        train_reward=1.0,
        train_icir=0.9,
        complexity_penalty=0.0,
        eligible=False,
        rejection_reasons=("val_icir_below_minimum",),
        complexity_cost=1.0,
        active_ir=0.9,
        risk_exposure=0.0,
        average_turnover=0.0,
        capacity_utilization=0.0,
    )
    ok = _score("ok", active_ir=0.2, exposure=0.2, turnover=0.2,
                capacity=0.1, val_icir=0.1)
    result = CandidateSelector().select(
        [rejected, ok], pareto_objectives=PARETO
    )
    assert result.selected == ok
    assert result.best_rejected == rejected


def test_selector_legacy_mode_unchanged():
    a = _score("a", active_ir=0.1, exposure=0.9, turnover=0.9,
               capacity=0.9, val_icir=0.4)
    b = _score("b", active_ir=0.9, exposure=0.1, turnover=0.1,
               capacity=0.1, val_icir=0.5)
    result = CandidateSelector().select([a, b])
    assert result.selected == b  # scalar val_reward/val_icir order


# --- P1-1: incrementally maintained first front -----------------------------

def _synthetic(idx, rng, *, eligible=True):
    """A synthetic measurable eligible candidate with a unique key."""

    return CandidateScore(
        tokens=(idx,),
        candidate_id=f"synthetic:{idx}",
        formula_text=f"f{idx}",
        source="test",
        direction=1,
        val_reward=float(rng.normal(0.0, 1.0)),
        val_icir=float(rng.normal(0.0, 1.0)),
        train_reward=float(rng.normal(0.0, 1.0)),
        train_icir=float(rng.normal(0.0, 1.0)),
        complexity_penalty=0.0,
        complexity_cost=1.0,
        eligible=eligible,
        rejection_reasons=() if eligible else ("val_icir_below_minimum",),
        active_ir=float(rng.uniform(-2.0, 2.0)),
        risk_exposure=float(rng.uniform(0.1, 3.0)),
        average_turnover=float(rng.uniform(0.1, 3.0)),
        capacity_utilization=float(rng.uniform(0.1, 3.0)),
    )


def test_incremental_front_matches_one_shot_sort_on_random_batches():
    """P1-1 equivalence pin: 2000 random candidates arriving in 8 batches
    (the RL arrival pattern).  The incrementally maintained first front
    and the selection winner must equal the one-shot full sort over the
    whole accumulated set — the incremental path exists to remove the
    per-step quadratic re-sort, not to change any elite decision."""

    from ashare_model.pareto import IncrementalNonDominatedFront

    rng = np.random.default_rng(20260921)
    scores = [_synthetic(i, rng) for i in range(2000)]
    batches = [scores[i : i + 250] for i in range(0, len(scores), 250)]
    assert len(batches) == 8

    front = IncrementalNonDominatedFront(PARETO)
    for batch in batches:
        for score in batch:
            front.add(score.deterministic_key, score)

    one_shot = nondominated_fronts(scores, PARETO)[0]
    assert {s.deterministic_key for s in front.members()} == {
        s.deterministic_key for s in one_shot
    }
    winner = min(front.members(), key=CandidateSelector._pareto_tie_key)
    assert winner == CandidateSelector().select(
        scores, pareto_objectives=PARETO
    ).selected


def test_incremental_front_classifies_unmeasurable_like_one_shot():
    """NaN objectives are never admitted to the front (the one-shot sort
    sends them to the last front); a pool of ONLY unmeasurable items
    keeps the front empty so the selector-side tie-break path decides."""

    from ashare_model.pareto import IncrementalNonDominatedFront

    rng = np.random.default_rng(5)
    measurable = [_synthetic(i, rng) for i in range(20)]
    unmeasurable = [
        _synthetic(100 + i, rng) for i in range(5)
    ]
    unmeasurable = [
        score.__class__(**{**score.__dict__, "active_ir": float("nan")})
        for score in unmeasurable
    ]

    front = IncrementalNonDominatedFront(PARETO)
    for score in measurable + unmeasurable:
        status = front.add(score.deterministic_key, score)
        if score in unmeasurable:
            assert status == "unmeasurable"
    one_shot = nondominated_fronts(measurable + unmeasurable, PARETO)
    assert len(one_shot[-1]) == len(unmeasurable)
    assert {s.deterministic_key for s in front.members()} == {
        s.deterministic_key for s in one_shot[0]
    }

    empty_front = IncrementalNonDominatedFront(PARETO)
    for score in unmeasurable:
        assert empty_front.add(score.deterministic_key, score) == "unmeasurable"
    assert empty_front.members() == []
    assert len(nondominated_fronts(unmeasurable, PARETO)) == 1


def test_incremental_front_replacement_sets_dirty():
    """Re-adding a key with a changed item asks the caller for a full
    rebuild (the incremental structure cannot replay a front member
    leaving); a fresh key never does."""

    from ashare_model.pareto import IncrementalNonDominatedFront

    front = IncrementalNonDominatedFront(PARETO)
    a = _synthetic(1, np.random.default_rng(1))
    b = _synthetic(1, np.random.default_rng(2))
    front.add(a.deterministic_key, a)
    assert not front.dirty
    front.add(b.deterministic_key, b)
    assert front.dirty
    front.reset()
    assert not front.dirty and len(front) == 0


def test_incremental_front_comparisons_follow_batch_not_pool():
    """P1-1 performance pin, by comparison count instead of wall-clock
    (CI-stable): with a totally ordered arrival stream the first front
    stays a single member, so the domination comparisons per batch stay
    proportional to the batch size — not to the accumulated pool the
    one-shot sort re-scans quadratically on every step."""

    from ashare_model.pareto import IncrementalNonDominatedFront

    rng = np.random.default_rng(7)
    objectives = [("a", 1), ("b", 1)]
    front = IncrementalNonDominatedFront(objectives)
    total = 0
    for _ in range(40):  # 40 batches x 250 = 10,000 accumulated
        batch = [
            {"a": float(rng.uniform(0.0, 1.0)), "b": 0.5}
            for _ in range(250)
        ]
        before = front.dominance_comparisons
        for idx, item in enumerate(batch):
            front.add((total, idx), item)
        total += len(batch)
        # Front size is 1 throughout: each arrival costs at most one
        # domination check plus one removal check.
        assert len(front) == 1
        assert front.dominance_comparisons - before <= 3 * len(batch) + 3
    assert total == 10_000
    # Total work stayed linear in the arrivals although the pool grew to
    # 10k — the per-step cost is O(batch x front), never O(pool^2).
    assert front.dominance_comparisons <= 3 * total + 40 * 3
