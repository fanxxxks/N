from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_model.factors import AshareFactorEngine, compute_factor_tensor
from ashare_model.vocab import FEATURE_NAMES


def test_compute_factor_tensor_shape_and_finite(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(bars, ts_codes, dates)
    assert tensor.shape == (len(FEATURE_NAMES), len(ts_codes), len(dates))
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_compute_factor_tensor_with_empty_fundamentals(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(bars, ts_codes, dates, {"000001.SZ": {}})
    assert tensor.shape == (len(FEATURE_NAMES), len(ts_codes), len(dates))


def test_factor_engine_helpers(bars_data):
    dates, ts_codes, bars = bars_data
    engine = AshareFactorEngine()
    close = engine._pivot(bars, ts_codes, dates, "close")
    returns = engine._returns(close)
    assert returns.shape == close.shape
    # The first day has no prior close: the honest value is NaN (the
    # cross-sectional standardizer later maps it to the neutral 0).
    assert returns.iloc[:, 0].isna().all()
    assert returns.iloc[0, 1] == pytest.approx(close.iloc[0, 1] / close.iloc[0, 0] - 1.0)
    shifted = engine._shift_ratio(close, 5)
    assert shifted.shape == close.shape


def test_factor_tensor_is_cross_sectionally_standardized(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(bars, ts_codes, dates)
    first_factor = tensor[0]
    assert np.nanmax(np.abs(first_factor)) <= 5.0 + 1e-6


def test_missing_returns_are_neutral_in_tensor(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(bars, ts_codes, dates)
    ret1 = tensor[FEATURE_NAMES.index("RET_1")]
    # The first date column has no prior close; after standardization it must
    # be the neutral 0 rather than a biased raw 0 imputation.
    assert np.allclose(ret1[:, 0], 0.0)
    assert np.isfinite(tensor).all()


def test_limit_event_features_detect_one_word_moves():
    dates = ["20240101", "20240102", "20240103"]
    codes = ["000001.SZ", "600000.SH"]
    rows = [
        {"ts_code": "000001.SZ", "trade_date": "20240101", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "pre_close": 9.9, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "000001.SZ", "trade_date": "20240102", "open": 11.0, "high": 11.0, "low": 11.0, "close": 11.0, "pre_close": 10.0, "volume": 1e6, "amount": 1.1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "open": 11.1, "high": 11.2, "low": 11.0, "close": 11.1, "pre_close": 11.0, "volume": 1e6, "amount": 1.1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "600000.SH", "trade_date": "20240101", "open": 8.0, "high": 8.2, "low": 7.9, "close": 8.0, "pre_close": 7.9, "volume": 1e6, "amount": 8e6, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "600000.SH", "trade_date": "20240102", "open": 8.1, "high": 8.3, "low": 8.0, "close": 8.2, "pre_close": 8.0, "volume": 1e6, "amount": 8e6, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "600000.SH", "trade_date": "20240103", "open": 8.2, "high": 8.4, "low": 8.1, "close": 8.3, "pre_close": 8.2, "volume": 1e6, "amount": 8e6, "turnover_rate": 1.0, "adj_factor": 1.0},
    ]
    bars = pd.DataFrame(rows)
    tensor = compute_factor_tensor(bars, codes, dates)
    up = tensor[FEATURE_NAMES.index("LIMIT_UP_EVENT")]
    # 000001.SZ locked limit-up (10%) on day 2: it must outrank the normal
    # stock in the standardized cross-section.
    assert up[0, 1] > 0
    assert up[1, 1] < up[0, 1]
    assert np.isfinite(up).all()


def test_fundamentals_applied_only_to_last_date(bars_data):
    dates, ts_codes, bars = bars_data
    fundamentals = {"000001.SZ": {"PE_TTM": 15.0}, "600000.SH": {"PE_TTM": 8.0}}
    tensor = compute_factor_tensor(bars, ts_codes, dates, fundamentals)
    pe = tensor[FEATURE_NAMES.index("PE_TTM")]
    # The snapshot is informative on the final date only (a non-constant
    # cross-section); every earlier date stays neutral (no lookahead).
    assert not np.allclose(pe[:, -1], 0.0)
    assert np.allclose(pe[:, :-1], 0.0)
