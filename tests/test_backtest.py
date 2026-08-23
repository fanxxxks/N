from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, DataConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine, equal_weight_benchmark_returns
from ashare_model.data_loader import AshareDataLoader
from ashare_model.vocab import FORMULA_VOCAB


def _engine_inputs(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, __import__("ashare_data.config", fromlist=["ModelConfig"]).ModelConfig())
    loader.load_data()
    factors = loader.factor_tensor[0].numpy()
    raw = {k: v.numpy() for k, v in loader.raw_data_cache.items()}
    return factors, raw, loader.ts_codes, loader.dates


def test_run_returns_expected_result(populated_db: DataConfig):
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    result = AshareBacktestEngine(BacktestConfig(top_n=2, train_end_date="2024-02-01")).run(
        factors, raw, ts_codes, dates,
        np.ones((len(ts_codes), len(dates)), dtype=bool),
    )
    assert len(result.daily_returns) == len(dates) - 2
    assert len(result.equity_curve) == len(dates) - 1
    assert "sortino" in result.metrics
    assert result.equity_curve[-1] > 0


def test_run_validates_factor_shape(populated_db: DataConfig):
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    engine = AshareBacktestEngine(BacktestConfig())
    with pytest.raises(ValueError):
        engine.run(
            factors.reshape(1, -1), raw, ts_codes, dates,
            np.ones((len(ts_codes), len(dates)), dtype=bool),
        )


def test_open_to_open_returns():
    open_ = np.array([[10.0, 11.0, 12.0, 13.0]])
    ret = AshareBacktestEngine._open_to_open_returns(open_)
    assert ret[0, 0] == pytest.approx((12.0 - 11.0) / 11.0)
    assert ret[0, -1] == 0.0 and ret[0, -2] == 0.0


def test_blocked_mask_suspension_and_limits(populated_db: DataConfig):
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    engine = AshareBacktestEngine(BacktestConfig())
    open_ = np.ones((1, 3))
    high = np.ones((1, 3))
    low = np.ones((1, 3))
    pre_close = np.ones((1, 3))
    volume = np.ones((1, 3))

    open_[0, 1] = 0.0
    blocked = engine._blocked_mask(1, open_, high, low, pre_close, volume, ["000001.SZ"], "buy")
    assert blocked[0]

    open_[:] = 1.1
    high[:] = 1.1
    low[:] = 1.1
    pre_close[:] = 1.0
    blocked_buy = engine._blocked_mask(1, open_, high, low, pre_close, volume, ["000001.SZ"], "buy")
    assert blocked_buy[0]

    open_[:] = 0.9
    high[:] = 0.9
    low[:] = 0.9
    blocked_sell = engine._blocked_mask(1, open_, high, low, pre_close, volume, ["000001.SZ"], "sell")
    assert blocked_sell[0]


def test_engine_uses_board_rates_not_current_st_names():
    # A one-word +5.2% open is inside the 10% main-board band: the
    # historical engine must buy it, whatever the stock's current name is
    # (there is no dated ST history, so names can never rewrite the past).
    codes = ["000001.SZ", "600000.SH"]
    dates = ["20240102", "20240103", "20240104", "20240105"]
    open_ = np.full((2, 4), 10.0)
    high = np.full((2, 4), 10.0)
    low = np.full((2, 4), 10.0)
    pre_close = np.full((2, 4), 10.0)
    volume = np.full((2, 4), 1_000_000.0)
    open_[0, 1] = high[0, 1] = low[0, 1] = 10.52  # one-word entry-day open
    signal = np.array([[5.0] * 4, [1.0] * 4])
    result = AshareBacktestEngine(
        BacktestConfig(top_n=1, single_weight_cap=1.0)
    ).run(signal, {"open": open_, "high": high, "low": low,
                   "pre_close": pre_close, "volume": volume}, codes, dates,
          np.ones((len(codes), len(dates)), dtype=bool))
    assert result.positions[0]["ts_codes"] == ["000001.SZ"]


def test_run_includes_benchmark_positions_and_aligned_dates(populated_db: DataConfig):
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    result = AshareBacktestEngine(BacktestConfig(top_n=2)).run(
        factors, raw, ts_codes, dates,
        np.ones((len(ts_codes), len(dates)), dtype=bool),
    )
    # Equal-weight benchmark computed on the same open-to-open basis.
    assert result.benchmark_equity is not None
    assert len(result.benchmark_equity) == len(result.equity_curve)
    # Dates and equity curves align 1:1 for plotting.
    assert len(result.dates) == len(result.equity_curve)
    # Position snapshots for the dashboard selection tab.
    assert result.positions
    assert set(result.positions[0]) == {
        "signal_date",
        "entry_date",
        "exit_date",
        "ts_codes",
        "weights",
    }
    assert result.positions[0]["entry_date"] == dates[1]
    assert result.positions[0]["exit_date"] == dates[2]


def test_initial_capital_from_config():
    engine = AshareBacktestEngine(BacktestConfig(initial_capital=50000.0))
    assert engine.initial_capital == 50000.0


def test_run_excludes_high_signal_outside_pit_universe():
    dates = ["20240102", "20240103", "20240104"]
    codes = ["000001.SZ", "600000.SH"]
    factors = np.array([[100.0, 100.0, 100.0], [1.0, 1.0, 1.0]])
    raw = {
        "open": np.ones((2, 3)),
        "high": np.full((2, 3), 1.1),
        "low": np.full((2, 3), 0.9),
        "pre_close": np.ones((2, 3)),
        "volume": np.ones((2, 3)),
    }
    universe_mask = np.array(
        [[False, False, True], [True, True, True]], dtype=bool
    )
    result = AshareBacktestEngine(BacktestConfig(top_n=1)).run(
        factors,
        raw,
        codes,
        dates,
        universe_mask=universe_mask,
    )
    assert result.positions[0]["ts_codes"] == ["600000.SH"]


# --- PIT universe mask in execution -----------------------------------------


def _masked_market(n_stocks=4, n_dates=8):
    """Constant-open market where stock 0 has the top signal; a stock's
    eligibility is controlled per test through the returned mask."""
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
    return codes, dates, raw, signal


def _run_engine(signal, raw, codes, dates, mask, **kwargs):
    return AshareBacktestEngine(BacktestConfig(**kwargs)).run(
        signal, raw, codes, dates, universe_mask=mask
    )


def test_engine_extreme_future_member_changes_nothing_pre_join():
    # The future member (row -1) carries extreme finite signal values
    # before its join day: with the mask in place neither the positions nor
    # the daily returns, turnover, costs or the benchmark curve may change.
    codes, dates, raw, signal = _masked_market()
    join_day = 3
    extreme = signal.copy()
    extreme[-1, :join_day] = 1e9
    mask = np.ones((4, 8), dtype=bool)
    mask[-1, :join_day] = False
    cfg = dict(top_n=1, single_weight_cap=1.0)
    mild_result = _run_engine(signal, raw, codes, dates, mask, **cfg)
    extreme_result = _run_engine(extreme, raw, codes, dates, mask, **cfg)
    assert extreme_result.daily_returns == mild_result.daily_returns
    assert extreme_result.turnover == mild_result.turnover
    assert extreme_result.benchmark_equity == mild_result.benchmark_equity
    assert extreme_result.positions == mild_result.positions
    # Costs are the deterministic gap between the gross position return and
    # the net daily return: identical because the positions are identical.
    target = open_to_open_returns(raw["open"])

    def _costs(result):
        out = []
        for t, snap in enumerate(result.positions):
            weights = dict(zip(snap["ts_codes"], snap["weights"]))
            gross = sum(
                w * target[codes.index(c), t] for c, w in weights.items()
            )
            out.append(gross - result.daily_returns[t])
        return out

    assert _costs(mild_result) == _costs(extreme_result)
    # Sanity: with an all-eligible mask the extreme values do change the outcome.
    open_result = _run_engine(extreme, raw, codes, dates, np.ones((4, 8), dtype=bool), **cfg)
    assert open_result.daily_returns != mild_result.daily_returns


def test_engine_signal_date_and_entry_date_alignment():
    # Position eligibility = universe_mask[:, t] AND universe_mask[:, t+1]:
    # the signal date decides observation, the entry date decides the buy.
    codes, dates, raw, signal = _masked_market(n_stocks=3, n_dates=7)
    mask = np.ones((3, 7), dtype=bool)
    # Stock 0: ineligible at t=0; eligible at t=1 (signal and entry);
    # eligible at t=2 but INELIGIBLE at its entry t+1=3; ineligible at t=3.
    mask[0] = [False, True, True, False, True, True, True]
    result = _run_engine(signal, raw, codes, dates, mask, top_n=1, single_weight_cap=1.0)
    names = [p["ts_codes"] for p in result.positions]
    assert names[0] == [codes[1]]  # t=0: stock 0 not signal-date eligible
    assert names[1] == [codes[0]]  # t=1: eligible at both t and t+1
    assert names[2] == [codes[1]]  # t=2: stock 0 ineligible at entry t+1=3
    assert names[3] == [codes[1]]  # t=3: stock 0 ineligible at the signal date


def test_engine_exit_day_sells_position_with_cost():
    codes, dates, raw, signal = _masked_market(n_stocks=3, n_dates=8)
    exit_day = 3
    mask = np.ones((3, 8), dtype=bool)
    mask[0, exit_day:] = False  # stock 0 exits the universe
    result = _run_engine(signal, raw, codes, dates, mask, top_n=1, single_weight_cap=1.0)
    names = [p["ts_codes"] for p in result.positions]
    # Stock 0 leads every day it can still be held: its last holdable
    # signal day is exit_day-1 (the entry at exit_day is already outside
    # the universe), so the position exits exactly there through the
    # ordinary sell path — full sell+buy turnover and the sell cost leg —
    # never silently dropped.
    assert names[0] == [codes[0]]
    assert names[1] == [codes[0]]
    assert names[exit_day - 1] == [codes[1]]
    assert names[exit_day] == [codes[1]]
    assert result.turnover[exit_day - 1] == pytest.approx(2.0)
    # Zero targets: the exit day's net return is exactly the rebalance cost.
    assert result.daily_returns[exit_day - 1] < 0.0
    assert result.daily_returns[1] == pytest.approx(0.0, abs=1e-15)


def test_engine_day_without_eligible_stocks_is_stable():
    codes, dates, raw, signal = _masked_market(n_stocks=3, n_dates=8)
    mask = np.ones((3, 8), dtype=bool)
    mask[:, 0] = False
    mask[:, 1] = False
    result = _run_engine(signal, raw, codes, dates, mask, top_n=1, single_weight_cap=1.0)
    target = open_to_open_returns(raw["open"])
    bench = equal_weight_benchmark_returns(
        target, list(range(6)), mask
    )
    # Both the benchmark and the (flat-book) strategy earn exactly zero on
    # the empty days, with no NaN or extreme values anywhere.
    assert bench[0] == 0.0 and bench[1] == 0.0
    assert result.daily_returns[0] == 0.0
    assert result.daily_returns[1] == 0.0
    assert all(np.isfinite(r) for r in result.daily_returns)
    assert all(np.isfinite(r) for r in result.benchmark_equity)


def test_engine_mask_shape_mismatch_fails():
    codes, dates, raw, signal = _masked_market(n_stocks=3, n_dates=6)
    bad = np.ones((2, 6), dtype=bool)
    with pytest.raises(ValueError, match="universe_mask shape"):
        _run_engine(signal, raw, codes, dates, bad)


def test_equal_weight_benchmark_requires_valid_targets():
    n_stocks, n_dates = 4, 6
    target = np.zeros((n_stocks, n_dates))
    target[:, 0] = 0.01
    target[0, 1] = np.nan  # invalid target for one stock on day 1
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    returns = equal_weight_benchmark_returns(target, [0, 1, 2], mask)
    assert returns[0] == pytest.approx(0.01)
    # Day 1 averages only the finite-target cells, not the NaN cell.
    assert returns[1] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="universe_mask shape"):
        equal_weight_benchmark_returns(
            target, [0, 1], np.ones((3, 6), dtype=bool)
        )
