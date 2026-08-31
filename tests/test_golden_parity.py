"""T3-02 contracts: unified execution spec — vectorized engine vs whole-lot
matcher golden parity.

The fast vectorized research engine (``AshareBacktestEngine``) and the
paper-trading whole-lot matcher (``SimBroker``) execute the **same
signals** through the golden harness (``ashare_trading.golden``):

* **Lot-free mode is exact**: the matcher with ``lot_size=0`` (continuous
  shares, engine-aligned funding) reproduces the vectorized engine's
  equity curve day for day — identical to 1e-9 relative.
* **Lot mode diverges only by recorded integer lots and cash residuals**:
  the per-day residual ``(Delta_lot - Delta_free)`` decomposes exactly
  into the recorded ``rounding`` PnL (integer-lot quantity differences),
  ``fee_diff`` (fee financing) and ``carry`` (sub-lot cash), and the
  bookkeeping identity ``sum(residual) == lot_equity(T) - free_equity(T)``
  holds exactly.  When every target notional is a whole lot the residual
  is zero — lot mode is then exactly equal too.
* **Coverage**: suspension, limit up/down, T+1, delisting, adjusted
  (qfq) prices, cash, fees and sell-blocked exits are covered as golden
  scenarios below; the matcher's own skip reasons stay the authoritative
  record (``missing_bar``, ``suspended``, ``limit_up``, ``limit_down``,
  ``no_position``, ``no_available_position``, ``insufficient_cash``).
* **Stress**: cost/slippage multipliers 0.5x/1x/2x, one-day delayed
  execution, missing bars, extreme limit moves and capital scales 1e5 /
  1e7 / 1e9 all keep the parity contract exact in lot-free mode.
* **Capstone**: optimizer target weights (T3-01) executed through the
  spec keep the bookkeeping identity and the executed book respects the
  optimizer's caps within lot tolerance.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_trading.golden import EXECUTION_SPEC_VERSION, GoldenParity
from ashare_portfolio.execution_spec import execution_provenance
from ashare_portfolio.optimizer import (
    PortfolioConstraints,
    PortfolioObjective,
    PortfolioOptimizer,
)
from tests.conftest import make_bars

DEFAULT_CODES = ["000001.SZ", "600000.SH", "300001.SZ"]

COST_CFG = dict(
    initial_capital=1000000.0,
    top_n=2,
    single_weight_cap=0.5,
    commission_rate=0.00025,
    min_commission=5.0,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.00001,
    slippage_rate=0.0005,
)


def _cfg(**kwargs) -> BacktestConfig:
    values = dict(COST_CFG)
    values.update(kwargs)
    return BacktestConfig(**values)


def _raw_from_bars(bars, ts_codes, dates) -> dict[str, np.ndarray]:
    """The engine's raw-cache dict reconstructed from a bars frame."""
    n, t = len(ts_codes), len(dates)
    out = {}
    for name in ("open", "high", "low", "pre_close", "volume"):
        matrix = np.zeros((n, t), dtype=np.float64)
        for i, code in enumerate(ts_codes):
            rows = bars[bars["ts_code"] == code].sort_values("trade_date")
            matrix[i, :] = rows[name].to_numpy(dtype=np.float64)
        out[name] = matrix
    return out


def _constant_rank_signal(n_stocks: int, n_dates: int) -> np.ndarray:
    """Stock 0 always tops, then 1, 2 — deterministic selection."""
    signal = np.zeros((n_stocks, n_dates), dtype=np.float64)
    for i in range(n_stocks):
        signal[i, :] = 1.0 - 0.25 * i
    return signal


def _market(n_dates: int = 20, ts_codes: list[str] | None = None):
    ts_codes = list(ts_codes or DEFAULT_CODES)
    dates, _, bars = make_bars(n_dates, ts_codes)
    raw = _raw_from_bars(bars, ts_codes, dates)
    signal = _constant_rank_signal(len(ts_codes), n_dates)
    mask = np.ones((len(ts_codes), n_dates), dtype=bool)
    return signal, raw, ts_codes, dates, mask


def _run(
    signal,
    raw,
    ts_codes,
    dates,
    mask,
    *,
    lot_size: int = 100,
    delay: int = 1,
    cfg: BacktestConfig | None = None,
):
    harness = GoldenParity(
        cfg or _cfg(), lot_size=lot_size, execution_delay=delay
    )
    report = harness.run(signal, raw, ts_codes, dates, mask)
    return harness, report


# --- structure -------------------------------------------------------------


def test_execution_spec_is_versioned():
    assert EXECUTION_SPEC_VERSION == 2


def test_external_weights_without_constructor_provenance_are_rejected():
    signal, raw, ts_codes, dates, mask = _market(n_dates=10)
    weights = [np.asarray([0.5, 0.5, 0.0], dtype=np.float64)]
    with pytest.raises(ValueError, match="constructor provenance"):
        GoldenParity(_cfg(), lot_size=0).run(
            signal,
            raw,
            ts_codes,
            dates,
            mask,
            target_weights=weights,
        )


# --- canonical parity -------------------------------------------------------


def test_lot_free_matches_engine_exactly():
    signal, raw, ts_codes, dates, mask = _market()
    cfg = _cfg()
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=0, cfg=cfg)
    assert report.records
    assert report.execution_version == EXECUTION_SPEC_VERSION
    assert dict(report.constructor_provenance) == execution_provenance(cfg)
    assert report.weights_source == "portfolio_constructor"
    for free, lot_free in zip(report.free_equity, report.lot_free_equity):
        assert free == pytest.approx(lot_free, rel=1e-9)
    # The weight-based free path reproduces the engine's own daily returns.
    assert report.engine_daily_returns is not None
    for free_ret, engine_ret in zip(
        [r.free_return for r in report.records], report.engine_daily_returns
    ):
        assert free_ret == pytest.approx(engine_ret, rel=1e-9)
    harness = GoldenParity(_cfg(), lot_size=0)
    harness.verify(report)  # must not raise


def test_whole_lot_scenario_is_exact_in_lot_mode():
    # Constant 10-yuan prices and 1e5 capital with top_n=2: every target
    # notional is an exact whole lot, so lot mode == lot-free == engine
    # with zero recorded residual.  Zero fees isolate the lot-rounding
    # claim (fee financing is a separate cash residual covered by
    # test_fees_charged_in_both_paths).
    dates, ts_codes, bars = make_bars(12, DEFAULT_CODES)
    rows = []
    for code in ts_codes:
        for date in dates:
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "pre_close": 10.0,
                    "volume": 1_000_000.0,
                    "close": 10.0,
                }
            )
    import pandas as pd

    bars = pd.DataFrame(rows)
    raw = _raw_from_bars(bars, ts_codes, dates)
    signal = _constant_rank_signal(3, len(dates))
    mask = np.ones((3, len(dates)), dtype=bool)
    _, report = _run(
        signal, raw, ts_codes, dates, mask, lot_size=100,
        cfg=_cfg(
            top_n=2,
            single_weight_cap=0.5,
            initial_capital=100000.0,
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            slippage_rate=0.0,
        ),
    )
    for free, lot in zip(report.free_equity, report.lot_equity):
        assert free == pytest.approx(lot, rel=1e-12)
    for record in report.records:
        assert record.residual == pytest.approx(0.0, abs=1e-9)
        assert record.rounding == pytest.approx(0.0, abs=1e-9)


def test_lot_residual_decomposes_into_recorded_rounding_and_cash():
    signal, raw, ts_codes, dates, mask = _market()
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    # Every recorded residual is the sum of the recorded attribution.
    for record in report.records:
        assert record.residual == pytest.approx(
            record.rounding + record.fee_diff + record.carry, abs=1e-6
        )
    # The bookkeeping identity: accumulated residuals equal the total
    # lot-mode divergence from the free path.
    total_residual = sum(r.residual for r in report.records)
    assert report.lot_equity[-1] - report.free_equity[-1] == pytest.approx(
        total_residual, rel=1e-9
    )
    # Drifting prices mean fractional targets: some rounding was recorded.
    assert any(abs(r.rounding) > 1e-6 for r in report.records)


def test_deterministic_report():
    signal, raw, ts_codes, dates, mask = _market()
    _, a = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    _, b = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    assert a.free_equity == b.free_equity
    assert a.lot_equity == b.lot_equity
    assert [r.fills for r in a.records] == [r.fills for r in b.records]


# --- coverage: market frictions --------------------------------------------


def test_suspension_blocks_buy_in_both_paths():
    # Stock 0 is already held and remains in the target Top-2 when it is
    # suspended.  P3 contract section 2 only prohibits a buy-blocked name
    # from becoming a *new* holding; it does not create a needless exit.
    signal, raw, ts_codes, dates, mask = _market()
    suspension_day = 5
    raw["volume"][0, suspension_day] = 0.0
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    GoldenParity(_cfg(), lot_size=100).verify(report)
    entry_record = report.records[suspension_day - 1]
    assert all(f.ts_code != "000001.SZ" for f in entry_record.fills)
    assert entry_record.fills == ()
    np.testing.assert_array_equal(
        report.target_weights[suspension_day - 1],
        report.target_weights[suspension_day - 2],
    )


def test_limit_up_open_blocks_buy():
    # One-word +10% open on the entry day: the engine's selection excludes
    # the name — no BUY may fill in either path, exact parity.  (A sell of
    # a previously held position is legitimate on the limit-up day.)
    signal, raw, ts_codes, dates, mask = _market()
    limit_day = 6
    raw["open"][0, limit_day] = raw["pre_close"][0, limit_day] * 1.10
    raw["high"][0, limit_day] = raw["open"][0, limit_day]
    raw["low"][0, limit_day] = raw["open"][0, limit_day]
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    GoldenParity(_cfg(), lot_size=100).verify(report)
    entry_record = report.records[limit_day - 1]
    assert all(
        not (f.ts_code == "000001.SZ" and f.side == "buy") for f in entry_record.fills
    )


def test_limit_down_exit_is_force_held_in_both_paths():
    # Stock 0 leads the signal until day 6 then falls out of the top-2;
    # its exit day opens at a one-word -10%.  The engine force-holds the
    # position (sell blocked) and the matcher never sees a sell order for
    # it — the position persists in both paths and parity holds.
    signal, raw, ts_codes, dates, mask = _market(n_dates=14)
    crash_day = 7
    signal[0, 6:] = 0.0  # stock 0 drops out of the selection
    signal[2, 6:] = 0.9
    raw["open"][0, crash_day] = raw["pre_close"][0, crash_day] * 0.90
    raw["high"][0, crash_day] = raw["open"][0, crash_day]
    raw["low"][0, crash_day] = raw["open"][0, crash_day]
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    GoldenParity(_cfg(), lot_size=100).verify(report)
    # The executed weight for stock 0 stays positive on the crash day and
    # no sell fill for it appears (the exit is deferred, never dropped).
    crash_signal_idx = crash_day - 1
    assert report.target_weights[crash_signal_idx][0] > 0.0
    crash_fills = report.records[crash_signal_idx].fills
    assert all(
        not (f.ts_code == "000001.SZ" and f.side == "sell") for f in crash_fills
    )


def test_delisting_force_holds_position():
    # The top stock's bars end mid-window (delisting): the engine sees a
    # suspended cell and force-holds; the matcher skips with missing_bar.
    # Both keep the position and parity holds.
    signal, raw, ts_codes, dates, mask = _market(n_dates=16)
    last_bar = 9
    raw["open"][0, last_bar:] = np.nan
    raw["high"][0, last_bar:] = np.nan
    raw["low"][0, last_bar:] = np.nan
    raw["pre_close"][0, last_bar:] = np.nan
    raw["volume"][0, last_bar:] = np.nan
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    GoldenParity(_cfg(), lot_size=100).verify(report)
    # After the delisting day no FILL may mention the delisted name (the
    # position is force-held; attempted liquidations are skipped by the
    # authoritative matcher, never filled).
    for record in report.records[last_bar - 1 :]:
        assert all(
            not (f.ts_code == "000001.SZ" and f.status == "filled")
            for f in record.fills
        )


def test_adjusted_price_series_parity():
    # 复权: both engines consume the same qfq-adjusted series; a 2:1
    # split-like price jump must not break the parity contract.
    signal, raw, ts_codes, dates, mask = _market()
    split_day = 6
    for name in ("open", "high", "low", "pre_close"):
        raw[name][:, split_day:] *= 0.5
    raw["volume"][:, split_day:] *= 2.0
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=0)
    GoldenParity(_cfg(), lot_size=0).verify(report)


def test_fees_charged_in_both_paths():
    # Nonzero commission/stamp/transfer/slippage: lot-free stays exact and
    # the lot-mode fee financing shows up in the recorded fee_diff.
    signal, raw, ts_codes, dates, mask = _market()
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    GoldenParity(_cfg(), lot_size=100).verify(report)
    assert any(abs(r.fee_diff) > 1e-6 for r in report.records)


def test_insufficient_cash_recorded_in_lot_mode():
    # A small account with the 5-yuan commission floor: lot-mode buys are
    # affordability-capped, the skip/under-fill is recorded, cash never
    # goes negative, and the identity still holds.
    signal, raw, ts_codes, dates, mask = _market(n_dates=14)
    _, report = _run(
        signal, raw, ts_codes, dates, mask, lot_size=100,
        cfg=_cfg(initial_capital=20000.0),
    )
    GoldenParity(_cfg(initial_capital=20000.0), lot_size=100).verify(report)
    assert any(abs(r.carry) > 1e-6 for r in report.records)


def test_t_plus_one_flow_buys_then_sells_next_day():
    # A buy fills with zero available quantity (T+1); the exit on the
    # following execution day sells through the ordinary path — both
    # engines agree day for day.  The selection flips on the second
    # signal day so the sell is actually generated.
    signal, raw, ts_codes, dates, mask = _market(n_dates=10)
    signal[0, :] = 1.0
    signal[1, 0] = 0.9
    signal[1, 1:] = 0.0  # stock 1 drops out after the first signal day
    signal[2, :] = 0.8
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    GoldenParity(_cfg(), lot_size=100).verify(report)
    # First entry day: buys only (nothing held yet).
    first_fills = report.records[0].fills
    assert any(f.side == "buy" and f.status == "filled" for f in first_fills)
    assert all(f.side != "sell" for f in first_fills)
    # Second entry day: yesterday's buy is sellable (T+1 satisfied) and
    # the exit sell fills.
    second_fills = report.records[1].fills
    assert any(f.side == "sell" and f.status == "filled" for f in second_fills)


# --- stress and capacity ----------------------------------------------------


@pytest.mark.parametrize("multiplier", [0.5, 1.0, 2.0])
def test_cost_multiplier_grid_keeps_exact_parity(multiplier: float):
    signal, raw, ts_codes, dates, mask = _market()
    cfg = _cfg(
        commission_rate=0.00025 * multiplier,
        slippage_rate=0.0005 * multiplier,
        stamp_tax_rate=0.0005 * multiplier,
    )
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=0, cfg=cfg)
    GoldenParity(cfg, lot_size=0).verify(report)
    for free, lot_free in zip(report.free_equity, report.lot_free_equity):
        assert free == pytest.approx(lot_free, rel=1e-9)


def test_one_day_delayed_execution_parity():
    # Signals observed at t execute at t+2 instead of t+1: both engines
    # shift identically and parity stays exact.
    signal, raw, ts_codes, dates, mask = _market(n_dates=22)
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=0, delay=2)
    GoldenParity(_cfg(), lot_size=0, execution_delay=2).verify(report)
    assert report.execution_delay == 2
    assert len(report.records) == len(dates) - 3


def test_missing_bar_parity():
    # NaN opens are missing bars: the matcher drops the row and skips with
    # missing_bar; the engine's zero-cell rule blocks the same day.  No
    # fill in either path, exact parity.
    signal, raw, ts_codes, dates, mask = _market()
    missing_day = 8
    raw["open"][1, missing_day] = np.nan
    raw["high"][1, missing_day] = np.nan
    raw["low"][1, missing_day] = np.nan
    raw["pre_close"][1, missing_day] = np.nan
    raw["volume"][1, missing_day] = np.nan
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=100)
    GoldenParity(_cfg(), lot_size=100).verify(report)
    entry_record = report.records[missing_day - 1]
    assert all(
        not (f.ts_code == "600000.SH" and f.status == "filled")
        for f in entry_record.fills
    )


def test_extreme_limit_moves_parity():
    # A full one-word ±10% sequence: limit-up days block buys, limit-down
    # days block sells; every day stays in exact parity.
    signal, raw, ts_codes, dates, mask = _market(n_dates=16)
    for day in range(3, 12):
        stock = day % 3
        up = day % 2 == 0
        raw["open"][stock, day] = raw["pre_close"][stock, day] * (1.10 if up else 0.90)
        raw["high"][stock, day] = raw["open"][stock, day]
        raw["low"][stock, day] = raw["open"][stock, day]
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=0)
    GoldenParity(_cfg(), lot_size=0).verify(report)


@pytest.mark.parametrize("capital", [1e5, 1e7, 1e9])
def test_capital_scale_grid_keeps_exact_parity(capital: float):
    signal, raw, ts_codes, dates, mask = _market(n_dates=14)
    cfg = _cfg(initial_capital=capital)
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=0, cfg=cfg)
    GoldenParity(cfg, lot_size=0).verify(report)
    for free, lot_free in zip(report.free_equity, report.lot_free_equity):
        assert free == pytest.approx(lot_free, rel=1e-9)


@pytest.mark.parametrize("lot_size", [100, 1000])
def test_lot_size_sensitivity_recorded(lot_size: int):
    signal, raw, ts_codes, dates, mask = _market(n_dates=14)
    cfg = _cfg(initial_capital=1e6)
    _, report = _run(signal, raw, ts_codes, dates, mask, lot_size=lot_size, cfg=cfg)
    GoldenParity(cfg, lot_size=lot_size).verify(report)


# --- capstone: optimizer -> spec -> executed book ---------------------------


def _optimizer_weights(signal, raw, ts_codes, dates, mask):
    """T3-01 optimizer targets on a late signal column (no lookahead)."""
    n, t = signal.shape
    t_signal = t - 3
    industries = np.asarray([0, 1, 0], dtype=np.int64)
    adv = raw["volume"][:, t_signal] * raw["open"][:, t_signal]
    constraints = PortfolioConstraints(
        single_weight_cap=0.5,
        industry_cap=0.6,
        adv_participation=0.1,
        max_positions=2,
    )
    optimizer = PortfolioOptimizer(
        constraints,
        PortfolioObjective(risk_aversion=1.0, turnover_cost=0.0, impact_cost=0.0),
    )
    cov = np.eye(n) * 0.01
    solution = optimizer.solve(
        signal[:, t_signal],
        np.zeros(n),
        capital=1e6,
        cov=cov,
        industries=industries,
        adv=adv,
    )
    assert solution.status == "optimal"
    return [solution.weights], industries, adv


def test_optimizer_weights_execute_with_constraints():
    signal, raw, ts_codes, dates, mask = _market(n_dates=10)
    weights, industries, _ = _optimizer_weights(signal, raw, ts_codes, dates, mask)
    cfg = _cfg(top_n=2, single_weight_cap=0.5, initial_capital=1e6)
    harness = GoldenParity(cfg, lot_size=100)
    report = harness.run(
        signal,
        raw,
        ts_codes,
        dates,
        mask,
        target_weights=weights,
        target_weights_provenance=execution_provenance(cfg),
    )
    harness.verify(report)
    # The executed book (fills) respects the optimizer's caps within lot
    # tolerance: single-name cap and industry cap.
    w = np.asarray(report.target_weights[0], dtype=np.float64)
    assert w.max() <= 0.5 + 1e-4
    for group in np.unique(industries):
        assert w[industries == group].sum() <= 0.6 + 1e-4
    assert np.count_nonzero(w > 0) <= 2


def test_optimizer_weights_lot_free_matches_mirror():
    signal, raw, ts_codes, dates, mask = _market(n_dates=10)
    weights, _, _ = _optimizer_weights(signal, raw, ts_codes, dates, mask)
    cfg = _cfg(top_n=2, single_weight_cap=0.5, initial_capital=1e6)
    harness = GoldenParity(cfg, lot_size=0)
    report = harness.run(
        signal,
        raw,
        ts_codes,
        dates,
        mask,
        target_weights=weights,
        target_weights_provenance=execution_provenance(cfg),
    )
    harness.verify(report)
    for free, lot_free in zip(report.free_equity, report.lot_free_equity):
        assert free == pytest.approx(lot_free, rel=1e-9)
