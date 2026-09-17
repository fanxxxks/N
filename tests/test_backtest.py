from __future__ import annotations

import json

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, DataConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import (
    AshareBacktestEngine,
    equal_weight_benchmark_returns,
    load_formal_strategy,
)
from ashare_model.data_loader import AshareDataLoader
from ashare_model.vocab import FORMULA_VOCAB
from ashare_model.run_store import RunStore
from tests.conftest import write_current_strategy


def test_load_formal_strategy_reconstructs_exact_runspec(tmp_path):
    payload = write_current_strategy(tmp_path, dataset_id="ds-backtest")
    loaded, spec = load_formal_strategy(
        tmp_path / "best_ashare_strategy.json", RunStore(tmp_path)
    )
    assert loaded == payload
    assert spec.spec_id == payload["spec_id"]


def test_load_formal_strategy_rejects_legacy_strategy(tmp_path):
    path = tmp_path / "strategy.json"
    path.write_text(
        json.dumps({"formula": [1], "formula_text": "RET_1"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="legacy"):
        load_formal_strategy(path, RunStore(tmp_path))


def _engine_inputs(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, __import__("ashare_data.config", fromlist=["ModelConfig"]).ModelConfig())
    loader.load_data()
    factors = loader.factor_tensor[0].numpy()
    raw = {k: v.numpy() for k, v in loader.raw_data_cache.items()}
    return factors, raw, loader.ts_codes, loader.dates


def _with_price_semantics(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Attach the p19 cache keys to a synthetic execution-price dict:
    adjusted label basis mirrors the raw prices (no corporate actions)
    and the exact constraints are unconstrained by default."""
    out = dict(raw)
    out.setdefault("adjusted_open", np.asarray(raw["open"], dtype=np.float64).copy())
    n, t = np.asarray(raw["open"]).shape
    out.setdefault("up_limit", np.full((n, t), 1e12))
    out.setdefault("down_limit", np.full((n, t), 1e-9))
    out.setdefault("suspended", np.zeros((n, t)))
    return out


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


def test_engine_exact_constraints_block_execution(populated_db: DataConfig):
    """The engine consumes the exact constraint matrices: a suspension row
    blocks both sides and an open at the published up_limit blocks the
    fresh buy (p19 §2.4)."""
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    n, t = factors.shape
    signal = np.zeros((n, t))
    signal[0, :] = 3.0
    signal[1, :] = 2.0
    signal[2, :] = 1.0
    # Both blocks land on the FIRST entry day (day 1), so the blocked
    # names are fresh entrants (a held survivor would legitimately keep
    # its slot — blocking only prevents new buys).
    raw["suspended"][0, 1] = 1.0  # stock 0 suspended on day 1
    raw["up_limit"][1, 1] = raw["open"][1, 1]  # stock 1 opens at its limit
    result = AshareBacktestEngine(BacktestConfig(top_n=2, single_weight_cap=0.5)).run(
        signal, raw, ts_codes, dates,
        np.ones((n, t), dtype=bool),
    )
    # The first entry finds both top-ranked names blocked: the only
    # selectable cross-section is a single stock, which is dispersion-less
    # (T1-02), so the book holds instead of entering either blocked name.
    assert set(result.positions[0]["ts_codes"]) == set()

    # With only the suspension blocking, the other two names trade.
    factors2, raw2, ts_codes2, dates2 = _engine_inputs(populated_db)
    raw2["suspended"][0, 1] = 1.0
    result2 = AshareBacktestEngine(
        BacktestConfig(top_n=2, single_weight_cap=0.5)
    ).run(
        signal, raw2, ts_codes2, dates2,
        np.ones((n, t), dtype=bool),
    )
    entered2 = set(result2.positions[0]["ts_codes"])
    assert ts_codes2[0] not in entered2
    assert ts_codes2[1] in entered2
    assert ts_codes2[2] in entered2


def test_engine_missing_limit_record_blocks_entry(populated_db: DataConfig):
    """A NaN limit cell is fail-closed: the stock cannot be newly bought
    that day (no approximate fallback, p19 §2.4)."""
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    n, t = factors.shape
    # NaN limit record on the first entry day: the stock cannot be newly
    # bought THAT day (later days keep their own exact semantics).
    raw["up_limit"][0, 1] = np.nan
    raw["down_limit"][0, 1] = np.nan
    signal = np.zeros((n, t))
    signal[0, :] = 3.0
    signal[1, :] = 2.0
    result = AshareBacktestEngine(BacktestConfig(top_n=1, single_weight_cap=1.0)).run(
        signal, raw, ts_codes, dates,
        np.ones((n, t), dtype=bool),
    )
    assert ts_codes[0] not in result.positions[0]["ts_codes"]
    assert result.positions[0]["ts_codes"] == [ts_codes[1]]


def test_engine_exact_limit_table_not_current_names(populated_db: DataConfig):
    # The dated limit row carries the band in force that day: a near-limit
    # one-word open below the published up_limit is buyable, whatever the
    # stock's current display name is (names never rewrite the past).
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    n, t = factors.shape
    day = 2
    raw["open"][0, day] = raw["open"][0, day] * 1.052
    raw["high"][0, day] = raw["open"][0, day]
    raw["low"][0, day] = raw["open"][0, day]
    raw["up_limit"][0, day] = raw["open"][0, day - 1] * 1.10
    signal = np.zeros((n, t))
    signal[0, :] = 5.0
    signal[1, :] = 1.0
    result = AshareBacktestEngine(
        BacktestConfig(top_n=1, single_weight_cap=1.0)
    ).run(
        signal, raw, ts_codes, dates,
        np.ones((n, t), dtype=bool),
    )
    assert result.positions[0]["ts_codes"] == [ts_codes[0]]


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
    # Position snapshots for the dashboard selection tab.  P35 adds the
    # share-anchored book: each snapshot carries the raw share counts
    # (contract docs/p35_share_accounting_draft.md §2/§4).
    assert result.positions
    assert set(result.positions[0]) == {
        "signal_date",
        "entry_date",
        "exit_date",
        "ts_codes",
        "weights",
        "shares",
    }
    assert result.positions[0]["entry_date"] == dates[1]
    assert result.positions[0]["exit_date"] == dates[2]


def test_initial_capital_from_config():
    engine = AshareBacktestEngine(BacktestConfig(initial_capital=50000.0))
    assert engine.initial_capital == 50000.0


def test_run_excludes_high_signal_outside_pit_universe():
    # A stock outside the universe never enters the selection, whatever its
    # signal; with two selectable stocks (a dispersed cross-section) the
    # higher of the two wins.  A single-selectable day would be a
    # dispersion-less cross-section and hold instead (T1-02).
    dates = ["20240102", "20240103", "20240104"]
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    factors = np.array(
        [[100.0, 100.0, 100.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    )
    raw = _with_price_semantics({
        "open": np.ones((3, 3)),
        "high": np.full((3, 3), 1.1),
        "low": np.full((3, 3), 0.9),
        "pre_close": np.ones((3, 3)),
        "volume": np.ones((3, 3)),
    })
    universe_mask = np.array(
        [[False, False, True], [True, True, True], [True, True, True]],
        dtype=bool,
    )
    result = AshareBacktestEngine(BacktestConfig(top_n=1)).run(
        factors,
        raw,
        codes,
        dates,
        universe_mask=universe_mask,
    )
    assert result.positions[0]["ts_codes"] == ["300001.SZ"]


# --- PIT universe mask in execution -----------------------------------------


def _masked_market(n_stocks=4, n_dates=8):
    """Constant-open market where stock 0 has the top signal; a stock's
    eligibility is controlled per test through the returned mask."""
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = [f"202401{i:02d}" for i in range(n_dates)]
    open_ = np.full((n_stocks, n_dates), 10.0)
    raw = _with_price_semantics({
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": open_.copy(),
        "volume": np.full((n_stocks, n_dates), 1_000_000.0),
    })
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
    # P35 (p35 §2): the turnover is weight-difference based on the DRIFTED
    # book.  Hand-derived: the t=0 entry buys floor(100000/10/100)*100
    # = 10000 shares unfundable once its 76.0 buy fee (0.00076 * 100000)
    # lands; one proportional affordability pass floors the order to 9900
    # shares for a 75.24 fee, so the drifted book weight is 99000/99924.76.
    # The exit-day rebalance sells that entire weight and buys the fresh
    # top name at 1.0 — the full-exit + fresh-buy double leg.
    assert result.turnover[exit_day - 1] == pytest.approx(99000.0 / 99924.76 + 1.0)
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


# --- busted-book metrics robustness -----------------------------------------


def test_metrics_clamp_total_loss_beyond_capital():
    # A book that lost more than its capital (costs on a collapsed book)
    # compounds below zero.  Annualization must clamp the total return to
    # -100% instead of raising TypeError on the complex (1 + total) ** x.
    returns = [-0.5, -1.005, 0.1]  # the -1.005 day flips equity negative
    equity = [1.0]
    for ret in returns:
        equity.append(equity[-1] * (1.0 + ret))
    assert equity[-1] < 0.0
    metrics = AshareBacktestEngine._metrics(returns, equity)
    assert metrics["total_return"] == -1.0
    assert metrics["annual_return"] == -1.0
    assert metrics["max_drawdown"] <= 1.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_metrics_clamp_drawdown_into_unit_interval():
    # Drawdown is a fraction of peak equity: negative equity must not
    # report a drawdown above 100%.
    returns = [-0.9, -3.0]
    equity = [1.0, 0.1, -0.2]
    metrics = AshareBacktestEngine._metrics(returns, equity)
    assert metrics["max_drawdown"] == 1.0


def test_metrics_normal_books_unchanged():
    returns = [0.01, -0.005, 0.02]
    equity = [1.0]
    for ret in returns:
        equity.append(equity[-1] * (1.0 + ret))
    metrics = AshareBacktestEngine._metrics(returns, equity)
    assert metrics["total_return"] == pytest.approx(equity[-1] - 1.0)
    assert metrics["max_drawdown"] < 1.0


def test_engine_survives_busted_book_end_to_end():
    # End-to-end reproduction of the original crash: the exit open gaps to
    # ~0, the gross day is a -100% return and costs push the net day below
    # -100%, so the compounded equity goes negative.  The metrics must stay
    # finite instead of crashing on complex annualization.  Two stocks with
    # a dispersed cross-section so the engine actually trades (a degenerate
    # single-stock market would hold under the T1-02 no-signal contract).
    codes = ["000001.SZ", "600000.SH"]
    dates = ["20240102", "20240103", "20240104", "20240105"]
    open_ = np.array([[10.0, 10.0, 1e-9, 1e-9], [10.0, 10.0, 10.0, 10.0]])
    raw = _with_price_semantics({
        "open": open_,
        "high": open_.copy(),
        "low": open_.copy(),
        "pre_close": open_.copy(),
        "volume": np.ones((2, 4)),
    })
    # The busted open is a missing bar, not a limit state: keep the exact
    # constraints unconstrained so the day blocks via the dead-bar rule.
    raw["up_limit"][:] = 1e12
    result = AshareBacktestEngine(
        BacktestConfig(top_n=1, single_weight_cap=1.0)
    ).run(
        np.array([[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]),
        raw,
        codes,
        dates,
        universe_mask=np.ones((2, 4), dtype=bool),
    )
    assert all(np.isfinite(value) for value in result.metrics.values())
    # P35 (p35 §2): the share-based book can never lose more than the
    # holding value, so equity stays positive and the -100% clamp is
    # unreachable for a single-position wipeout.  Hand-derived: t=0 buys
    # 9900 shares at 10 (fee 75.24, weight 99000/99924.76); the
    # entry->exit session wipes the holding at -100% o2o, leaving the
    # 924.76 cash book; t=1 re-buys the still-top-ranked name at the 1e-9
    # open where the board lot is negligible — the affordability loop
    # spends the cash down to the min-commission floor, a fee of
    # 5 + 0.00051 * 919.29 = 5.4688, so the final equity is
    # 924.76 - 5.4688 = 919.2911714 / 100000.
    assert result.metrics["total_return"] == pytest.approx(
        -0.9908070882859237, rel=1e-9
    )
    # (0.0091929...) ** (252/2) underflows to 0, so the annualized figure
    # is exactly the -1.0 floor.
    assert result.metrics["annual_return"] == -1.0
