from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, DataConfig
from ashare_model.backtest import AshareBacktestEngine
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
        factors, raw, ts_codes, dates
    )
    assert len(result.daily_returns) == len(dates) - 1
    assert len(result.equity_curve) == len(dates)
    assert "sortino" in result.metrics
    assert result.equity_curve[-1] > 0


def test_run_validates_factor_shape(populated_db: DataConfig):
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    engine = AshareBacktestEngine(BacktestConfig())
    with pytest.raises(ValueError):
        engine.run(factors.reshape(1, -1), raw, ts_codes, dates)


def test_open_to_open_returns():
    open_ = np.array([[10.0, 11.0, 12.0, 13.0]])
    ret = AshareBacktestEngine._open_to_open_returns(open_)
    assert ret[0, 0] == pytest.approx((12.0 - 11.0) / 11.0)
    assert ret[0, -1] == 0.0 and ret[0, -2] == 0.0


def test_limit_rate():
    assert AshareBacktestEngine._limit_rate("000001.SZ", "平安银行") == 0.10
    assert AshareBacktestEngine._limit_rate("300001.SZ", "") == 0.20
    assert AshareBacktestEngine._limit_rate("688001.SH", "") == 0.20
    assert AshareBacktestEngine._limit_rate("000001.SZ", "ST 风险") == 0.05


def test_blocked_mask_suspension_and_limits(populated_db: DataConfig):
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    engine = AshareBacktestEngine(BacktestConfig())
    open_ = np.ones((1, 3))
    high = np.ones((1, 3))
    low = np.ones((1, 3))
    pre_close = np.ones((1, 3))
    volume = np.ones((1, 3))
    names = {"000001.SZ": "平安银行"}

    open_[0, 1] = 0.0
    blocked = engine._blocked_mask(1, open_, high, low, pre_close, volume, ["000001.SZ"], names, "buy")
    assert blocked[0]

    open_[:] = 1.1
    high[:] = 1.1
    low[:] = 1.1
    pre_close[:] = 1.0
    blocked_buy = engine._blocked_mask(1, open_, high, low, pre_close, volume, ["000001.SZ"], names, "buy")
    assert blocked_buy[0]

    open_[:] = 0.9
    high[:] = 0.9
    low[:] = 0.9
    blocked_sell = engine._blocked_mask(1, open_, high, low, pre_close, volume, ["000001.SZ"], names, "sell")
    assert blocked_sell[0]


def test_run_includes_benchmark_positions_and_aligned_dates(populated_db: DataConfig):
    factors, raw, ts_codes, dates = _engine_inputs(populated_db)
    result = AshareBacktestEngine(BacktestConfig(top_n=2)).run(
        factors, raw, ts_codes, dates
    )
    # Equal-weight benchmark computed on the same open-to-open basis.
    assert result.benchmark_equity is not None
    assert len(result.benchmark_equity) == len(result.equity_curve)
    # Dates and equity curves align 1:1 for plotting.
    assert len(result.dates) == len(result.equity_curve)
    # Position snapshots for the dashboard selection tab.
    assert result.positions
    assert set(result.positions[0]) == {"signal_date", "exec_date", "ts_codes", "weights"}


def test_initial_capital_from_config():
    engine = AshareBacktestEngine(BacktestConfig(initial_capital=50000.0))
    assert engine.initial_capital == 50000.0
