"""T1-02 contracts: no-signal semantics and permutation invariance.

The weight-construction contract shared by the backtest engine, the
reward basket and the paper-trading sim:

* a cross-section without sufficient dispersion (fewer than two distinct
  selectable values) triggers **no rebalance**: the previous book is held,
  with zero turnover and zero cost — never an arbitrary churn;
* an under-filled day (fewer than ``top_n`` selectable stocks) invests
  only what is selectable and **keeps the rest in cash** — weights are
  never renormalized upward;
* ``single_weight_cap`` is a hard per-name ceiling that no rebalance path
  may ever break (no renormalization after force-holds either);
* stock-row order never changes any measurement: selection ties resolve by
  a stable key (``ts_code``), and rank IC / costs / benchmarks are
  permutation-invariant by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_data.processor import has_cross_sectional_dispersion, open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.candidates import CandidateScorer, CandidateSpec
from ashare_model.reward import simulate_basket_daily_returns, simulate_basket_daily_returns_batch


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=100000.0,
        top_n=2,
        single_weight_cap=0.5,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _market(n_stocks=5, n_dates=8):
    """Deterministic market: stock i has signal ``n_stocks - i`` (distinct
    per row), all eligible, constant opens."""
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = [f"202401{i:02d}" for i in range(n_dates)]
    open_ = np.full((n_stocks, n_dates), 10.0)
    raw = {
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": open_.copy(),
        "volume": np.full((n_stocks, n_dates), 1_000_000.0),
    }
    signal = np.tile(
        np.arange(n_stocks, 0, -1, dtype=float)[:, None], (1, n_dates)
    )
    target = open_to_open_returns(open_)
    return codes, dates, raw, signal, target


def _permute(rows: list[np.ndarray | list], rng: np.random.Generator):
    order = rng.permutation(len(rows[0]))
    permuted = []
    for arr in rows:
        arr = np.asarray(arr)
        permuted.append(arr[order] if arr.ndim >= 1 else arr)
    return order, permuted


# --- dispersion helper ------------------------------------------------------


def test_dispersion_helper_contract():
    assert has_cross_sectional_dispersion(np.array([])) is False
    assert has_cross_sectional_dispersion(np.array([1.0])) is False
    assert has_cross_sectional_dispersion(np.array([1.0, 1.0])) is False
    assert has_cross_sectional_dispersion(np.array([0.0, 0.0, 0.0])) is False
    assert has_cross_sectional_dispersion(np.array([1.0, 2.0])) is True
    assert has_cross_sectional_dispersion(np.array([1.0, 2.0, 2.0])) is True
    # Non-finite values never count as distinct (selectable sets are finite
    # by construction, but the helper must be robust anyway).
    assert has_cross_sectional_dispersion(np.array([1.0, np.nan])) is False
    assert has_cross_sectional_dispersion(np.array([1.0, np.inf])) is False
    # Threshold is configurable.
    assert has_cross_sectional_dispersion(np.array([1.0, 2.0, 3.0]), min_distinct=3) is True
    assert has_cross_sectional_dispersion(np.array([1.0, 2.0, 2.0]), min_distinct=3) is False


# --- engine: no-signal semantics -------------------------------------------


def test_engine_degenerate_cross_section_holds_book():
    codes, dates, raw, signal, _ = _market(n_stocks=3)
    flat = np.full_like(signal, 2.5)  # every selectable value equal
    flat[0, 3:] = 5.0  # dispersion returns from day 3
    cfg = _cfg(top_n=1, single_weight_cap=1.0)
    result = AshareBacktestEngine(cfg).run(
        flat, raw, codes, dates, universe_mask=np.ones(flat.shape, dtype=bool)
    )
    # Days 0-2: no rebalance (flat book -> cash), zero turnover, zero cost.
    assert result.positions[0]["ts_codes"] == []  # nothing bought on day 0
    assert result.turnover[0] == pytest.approx(0.0, abs=1e-15)
    assert result.turnover[1] == pytest.approx(0.0, abs=1e-15)
    assert result.daily_returns[0] == pytest.approx(0.0, abs=1e-15)
    # Day 3: dispersion returns; the book trades again.
    assert result.positions[3]["ts_codes"] == [codes[0]]
    assert result.turnover[3] == pytest.approx(1.0, abs=1e-12)


def test_engine_holds_existing_book_on_degenerate_day():
    codes, dates, raw, signal, _ = _market(n_stocks=3)
    flat = signal.copy()
    flat[:, 2] = 7.0  # day 2 degenerates after a position exists
    cfg = _cfg(top_n=1, single_weight_cap=1.0)
    result = AshareBacktestEngine(cfg).run(
        flat, raw, codes, dates, universe_mask=np.ones(flat.shape, dtype=bool)
    )
    # Day 0 bought stock 0 (weight 1.0); the degenerate day 2 holds it:
    # zero turnover, zero cost, return = held position's target return.
    assert result.positions[0]["ts_codes"] == [codes[0]]
    assert result.positions[2]["ts_codes"] == [codes[0]]
    assert result.turnover[2] == pytest.approx(0.0, abs=1e-15)
    target = open_to_open_returns(raw["open"])
    assert result.daily_returns[2] == pytest.approx(
        float(target[0, 2]), abs=1e-15
    )


def test_engine_underfilled_keeps_cash_and_never_breaks_cap():
    codes, dates, raw, signal, _ = _market(n_stocks=4)
    cfg = _cfg(top_n=5, single_weight_cap=0.1)
    mask = np.ones((4, len(dates)), dtype=bool)
    mask[2:] = False  # only two selectable stocks vs top_n=5
    result = AshareBacktestEngine(cfg).run(
        signal, raw, codes, dates, universe_mask=mask
    )
    for snapshot in result.positions:
        assert len(snapshot["weights"]) <= 2
        assert all(weight <= 0.1 + 1e-12 for weight in snapshot["weights"])
        assert sum(snapshot["weights"]) <= 1.0 + 1e-12
        # Under-filled day: exactly min(1/top_n, cap) per name, cash rest.
        assert sum(snapshot["weights"]) == pytest.approx(
            2 * min(1.0 / 5, 0.1), abs=1e-12
        )


def test_engine_force_hold_never_renormalizes_above_cap():
    codes, dates, raw, signal, _ = _market(n_stocks=3)
    cfg = _cfg(top_n=2, single_weight_cap=0.4)
    mask = np.ones((3, len(dates)), dtype=bool)
    # Stock 0 exits after day 1 and is sell-blocked on its exit execution
    # day (t+1 = 3): the position must be force-held at 0.4, and the book
    # must NOT renormalize the remaining names above the cap.
    mask[0, 2:] = False
    open_ = raw["open"].copy()
    high = raw["high"].copy()
    low = raw["low"].copy()
    pre_close = raw["pre_close"].copy()
    high[0, 3] = low[0, 3] = open_[0, 3] = pre_close[0, 3] * 0.9  # limit-down open
    raw = {"open": open_, "high": high, "low": low, "pre_close": pre_close,
           "volume": raw["volume"]}
    result = AshareBacktestEngine(cfg).run(
        signal, raw, codes, dates, universe_mask=mask
    )
    for snapshot in result.positions:
        assert all(weight <= 0.4 + 1e-12 for weight in snapshot["weights"])
        assert sum(snapshot["weights"]) <= 1.0 + 1e-12


def test_engine_permutation_invariance():
    rng = np.random.default_rng(1234)
    codes, dates, raw, signal, _ = _market(n_stocks=6, n_dates=10)
    # Inject exact ties at the selection boundary so the tie-break is
    # actually exercised (not just assumed).
    signal[2, :] = signal[1, :]
    signal[4, :] = signal[3, :]
    cfg = _cfg(top_n=2, single_weight_cap=0.5)
    base = AshareBacktestEngine(cfg).run(
        signal, raw, codes, dates, universe_mask=np.ones(signal.shape, dtype=bool)
    )
    for seed in range(5):
        order, permuted = _permute(
            [signal, raw["open"], raw["high"], raw["low"], raw["pre_close"],
             raw["volume"], codes],
            np.random.default_rng(seed),
        )
        p_signal, p_open, p_high, p_low, p_pre, p_vol, p_codes = permuted
        p_raw = {"open": p_open, "high": p_high, "low": p_low,
                 "pre_close": p_pre, "volume": p_vol}
        other = AshareBacktestEngine(cfg).run(
            p_signal, p_raw, list(p_codes), dates,
            universe_mask=np.ones(p_signal.shape, dtype=bool),
        )
        assert other.daily_returns == pytest.approx(base.daily_returns, abs=1e-12)
        assert other.turnover == pytest.approx(base.turnover, abs=1e-12)
        assert other.benchmark_equity == pytest.approx(base.benchmark_equity, abs=1e-12)
        # Position *content* is identical (ts_codes are stable keys); the
        # snapshot list order follows the row order, which is cosmetic.
        for base_snap, other_snap in zip(base.positions, other.positions):
            assert sorted(zip(base_snap["ts_codes"], base_snap["weights"])) == sorted(
                zip(other_snap["ts_codes"], other_snap["weights"])
            )


# --- reward basket: no-signal semantics ------------------------------------


def test_basket_degenerate_cross_section_holds_book():
    _, _, _, signal, target = _market(n_stocks=3)
    flat = np.full_like(signal, 2.5)
    flat[0, 3:] = 5.0
    cfg = _cfg(top_n=1, single_weight_cap=1.0)
    sim = simulate_basket_daily_returns(
        flat, target, cfg, universe_mask=np.ones(flat.shape, dtype=bool)
    )
    assert sim.turnover[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.turnover[1] == pytest.approx(0.0, abs=1e-15)
    assert sim.daily_net_returns[0] == pytest.approx(0.0, abs=1e-15)
    assert sim.daily_cost_fractions[0] == pytest.approx(0.0, abs=1e-15)
    # Period 3 is the first dispersed signal day: the basket trades again.
    assert sim.turnover[3] == pytest.approx(1.0, abs=1e-12)


def test_basket_underfilled_keeps_cash_and_cap_like_engine():
    codes, dates, raw, signal, target = _market(n_stocks=4)
    cfg = _cfg(top_n=5, single_weight_cap=0.1, commission_rate=0.0,
               min_commission=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0,
               slippage_rate=0.0)
    mask = np.ones((4, len(dates)), dtype=bool)
    mask[2:] = False
    sim = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    # First period: 2 names at min(1/5, 0.1) each, 0.8 stays in cash.
    assert sim.daily_gross_returns[0] == pytest.approx(
        2 * 0.1 * float(target[0, 0]), abs=1e-12
    )
    assert sim.turnover[0] == pytest.approx(2 * 0.1, abs=1e-12)
    engine = AshareBacktestEngine(cfg).run(
        signal, raw, codes, dates, universe_mask=mask
    )
    assert sim.daily_gross_returns == pytest.approx(
        np.asarray(engine.daily_returns), abs=1e-12
    )
    assert sim.turnover == pytest.approx(np.asarray(engine.turnover), abs=1e-12)


def test_basket_force_hold_never_renormalizes_above_cap():
    codes, _, _, signal, target = _market(n_stocks=3)
    cfg = _cfg(top_n=2, single_weight_cap=0.4)
    mask = np.ones((3, 8), dtype=bool)
    mask[0, 2:] = False
    # The exit execution day is the first day whose entry is outside the
    # universe (T1-04 alignment: signal day 1 executes at day 2).
    blocked_sell = np.zeros((3, 8), dtype=bool)
    blocked_sell[0, 2] = True
    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask, blocked_sell=blocked_sell
    )
    # P3 contract section 3: the blocked 0.4 position consumes budget and
    # only the fresh buy may shrink.  The surviving 0.4 incumbent cannot be
    # sold merely to make room, so the exact target is [0.4, 0.4, 0.2].
    np.testing.assert_array_equal(
        sim.target_weights[1], np.array([0.4, 0.4, 0.2])
    )
    assert sim.turnover[1] == 0.2
    assert sim.daily_gross_returns[1] == (
        0.4 * float(target[0, 1])
        + 0.4 * float(target[1, 1])
        + 0.2 * float(target[2, 1])
    )


def test_basket_permutation_invariance_with_tie_break_keys():
    rng = np.random.default_rng(99)
    codes, _, _, signal, target = _market(n_stocks=6, n_dates=10)
    signal[2, :] = signal[1, :]  # exact ties exercise the key tie-break
    cfg = _cfg(top_n=2, single_weight_cap=0.5)
    keys = np.asarray(codes)
    base = simulate_basket_daily_returns(
        signal, target, cfg,
        universe_mask=np.ones(signal.shape, dtype=bool),
        tie_break_keys=keys,
    )
    for seed in range(5):
        order, permuted = _permute(
            [signal, target, keys], np.random.default_rng(seed)
        )
        p_signal, p_target, p_keys = permuted
        other = simulate_basket_daily_returns(
            p_signal, p_target, cfg,
            universe_mask=np.ones(p_signal.shape, dtype=bool),
            tie_break_keys=p_keys,
        )
        assert other.daily_net_returns == pytest.approx(base.daily_net_returns, abs=1e-12)
        assert other.turnover == pytest.approx(base.turnover, abs=1e-12)
        assert other.daily_cost_fractions == pytest.approx(
            base.daily_cost_fractions, abs=1e-12
        )


def test_basket_batch_permutation_invariance_with_keys():
    rng = np.random.default_rng(5)
    codes, _, _, signal, target = _market(n_stocks=6, n_dates=10)
    signal[3, :] = signal[2, :]
    cfg = _cfg(top_n=2, single_weight_cap=0.5)
    keys = np.asarray(codes)
    stacked = np.stack([signal, -signal])
    base = simulate_basket_daily_returns_batch(
        stacked, target, cfg,
        universe_mask=np.ones(target.shape, dtype=bool),
        tie_break_keys=keys,
    )
    order, permuted = _permute(
        [signal, target, keys], np.random.default_rng(3)
    )
    p_signal, p_target, p_keys = permuted
    other = simulate_basket_daily_returns_batch(
        np.stack([p_signal, -p_signal]), p_target, cfg,
        universe_mask=np.ones(p_target.shape, dtype=bool),
        tie_break_keys=p_keys,
    )
    assert other.daily_net_returns == pytest.approx(base.daily_net_returns, abs=1e-12)
    assert other.turnover == pytest.approx(base.turnover, abs=1e-12)


# --- scorer: permutation invariance ----------------------------------------


def test_scorer_permutation_invariance():
    rng = np.random.default_rng(21)
    n_stocks, n_dates = 8, 24
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = target + rng.normal(0.0, 0.001, size=target.shape)
    mask = np.ones(target.shape, dtype=bool)
    cfg = _cfg(top_n=2, single_weight_cap=0.5)
    reward_cfg = RewardConfig(ic_min_stocks=3)
    scorer = CandidateScorer(cfg, reward_cfg)
    windows = [(6, 12), (12, 18)]
    keys = np.asarray([f"{i:06d}.SZ" for i in range(n_stocks)])

    def score(signal, target, mask, keys):
        spec = CandidateSpec(candidate_id="p", formula_text="x", source="test")
        return scorer.score(
            spec, signal, target, windows,
            universe_mask=mask, tie_break_keys=keys,
        )

    base = score(signal, target, mask, keys)
    for seed in range(4):
        order, permuted = _permute(
            [signal, target, mask, keys], np.random.default_rng(seed)
        )
        p_signal, p_target, p_mask, p_keys = permuted
        other = score(p_signal, p_target, p_mask, p_keys)
        assert other.val_reward == pytest.approx(base.val_reward, abs=1e-12)
        assert other.val_icir == pytest.approx(base.val_icir, abs=1e-12)
        assert other.train_reward == pytest.approx(base.train_reward, abs=1e-12)
        assert other.train_icir == pytest.approx(base.train_icir, abs=1e-12)
        assert other.direction == base.direction
        assert other.rejection_reasons == base.rejection_reasons
        assert other.eligible == base.eligible
