from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import DataConfig
from ashare_data.processor import (
    filter_universe,
    is_valid_a_share_code,
    long_factor_frame,
    normalize_daily_bars,
    open_to_open_returns,
    pivot_wide,
    winsorize_cross_section,
)


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "trade_date": "20240101", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100.0, "amount": 1010.0, "turnover_rate": 1.0, "adj_factor": 1.0},
            {"ts_code": "000001.SZ", "trade_date": "20240102", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "volume": 110.0, "amount": 1122.0, "turnover_rate": 1.1, "adj_factor": 1.0},
        ]
    )


def test_normalize_daily_bars_empty_and_missing_required():
    assert normalize_daily_bars(None).empty
    assert normalize_daily_bars(pd.DataFrame()).empty
    assert normalize_daily_bars(pd.DataFrame({"x": [1]})).empty


def test_normalize_daily_bars_fills_pre_close_and_turnover():
    df = pd.DataFrame(
        [
            {"ts_code": "A", "trade_date": "20240101", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100.0, "amount": 1010.0},
            {"ts_code": "A", "trade_date": "20240102", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "volume": 110.0, "amount": 1122.0},
        ]
    )
    out = normalize_daily_bars(df)
    assert out.iloc[0]["pre_close"] == pytest.approx(10.0)
    assert out.iloc[1]["pre_close"] == pytest.approx(10.1)
    # A missing turnover column cannot be reconstructed from OHLCV (it needs
    # the float share count), so it must stay NaN instead of a fake ~1.0
    # constant; downstream standardization treats NaN as neutral.
    assert out["turnover_rate"].isna().all()


def test_normalize_daily_bars_keeps_provided_turnover():
    df = pd.DataFrame(
        [
            {"ts_code": "A", "trade_date": "20240101", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100.0, "amount": 1010.0, "turnover_rate": 1.2},
            {"ts_code": "A", "trade_date": "20240102", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "volume": 110.0, "amount": 1122.0, "turnover_rate": 1.3},
        ]
    )
    out = normalize_daily_bars(df)
    assert out["turnover_rate"].tolist() == pytest.approx([1.2, 1.3])


def test_normalize_daily_bars_drops_invalid_and_deduplicates():
    df = _bars()
    df.loc[1, "close"] = np.nan
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    out = normalize_daily_bars(df)
    assert len(out) == 1
    assert out.iloc[0]["trade_date"] == "20240101"


def test_filter_universe_empty():
    assert filter_universe(pd.DataFrame(), pd.DataFrame(), None, DataConfig()) == []


def test_filter_universe_filters_st_and_thresholds():
    stocks = pd.DataFrame(
        [
            {"ts_code": "000001.SZ", "name": "平安银行", "is_st": False},
            {"ts_code": "000002.SZ", "name": "ST 风险", "is_st": False},
            {"ts_code": "600000.SH", "name": "浦发银行", "is_st": True},
        ]
    )
    bars = _bars()
    bars = pd.concat(
        [
            bars,
            pd.DataFrame(
                [
                    {"ts_code": "000002.SZ", "trade_date": "20240101", "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 100.0, "amount": 110.0, "turnover_rate": 1.0, "adj_factor": 1.0},
                    {"ts_code": "600000.SH", "trade_date": "20240101", "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 100.0, "amount": 110.0, "turnover_rate": 1.0, "adj_factor": 1.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    config = DataConfig(min_listed_days=1, min_price=1.0, max_price=10000.0, min_amount=1.0)
    codes = filter_universe(stocks, bars, None, config)
    assert "000002.SZ" not in codes
    assert "600000.SH" not in codes
    assert "000001.SZ" in codes


def test_filter_universe_applies_constituents_intersection():
    bars = _bars()
    config = DataConfig(min_listed_days=1, min_price=1.0, max_price=10000.0, min_amount=1.0)
    codes = filter_universe(None, bars, ["000001.SZ"], config)
    assert codes == ["000001.SZ"]


def test_pivot_wide_shape_and_reindex():
    bars = _bars()
    wide = pivot_wide(bars, ["000001.SZ", "600000.SH"], ["20240101", "20240102"])
    assert wide.shape == (2, 2)
    assert pd.isna(wide.loc["600000.SH", "20240101"])


def test_winsorize_cross_section_clips_and_handles_all_nan():
    wide = pd.DataFrame(
        [[0.0, 1.0], [10.0, 2.0]],
        index=["A", "B"],
        columns=["d1", "d2"],
    )
    out = winsorize_cross_section(wide)
    assert out.shape == wide.shape
    assert np.isfinite(out.to_numpy()).all()

    all_nan = pd.DataFrame([[np.nan], [np.nan]], index=["A", "B"], columns=["d"])
    out_nan = winsorize_cross_section(all_nan)
    assert (out_nan == 0.0).all().all()


def test_long_factor_frame_flattens_and_handles_nonfinite():
    arr = np.zeros((2, 2, 2), dtype=np.float64)
    arr[0, 0, 0] = np.inf
    frame = long_factor_frame(arr, ["A", "B"], ["d1", "d2"], ["f0", "f1"])
    assert frame.shape == (8, 4)
    assert pd.isna(frame.loc[(frame["factor_name"] == "f0") & (frame["ts_code"] == "A") & (frame["trade_date"] == "d1"), "value"].iloc[0])


def test_is_valid_a_share_code():
    assert is_valid_a_share_code("000001.SZ")
    assert is_valid_a_share_code("600000.SH")
    assert is_valid_a_share_code("688001.SH")
    assert is_valid_a_share_code("300001.SZ")
    assert is_valid_a_share_code("830001.BJ")
    # Index symbols must be rejected (they used to leak into the universe).
    assert not is_valid_a_share_code("000300.SZ")
    assert not is_valid_a_share_code("000905.SZ")
    assert not is_valid_a_share_code("000852.SZ")
    assert not is_valid_a_share_code("000300.SH")
    # B-shares and malformed codes are rejected.
    assert not is_valid_a_share_code("900901.SH")
    assert not is_valid_a_share_code("12345")
    assert not is_valid_a_share_code("600000")
    assert not is_valid_a_share_code("600000.XX")
    assert not is_valid_a_share_code(600000)


def test_open_to_open_returns_masks_suspension():
    # The second day is suspended (open == 0 after padding): the surrounding
    # returns must be masked instead of producing 1e10-scale fake returns.
    open_ = np.array([[10.0, 0.0, 12.0, 13.0, 14.0]])
    ret = open_to_open_returns(open_)
    assert np.isfinite(ret).all()
    assert np.abs(ret).max() < 1.0
    assert (ret[0, -2:] == 0.0).all()

    open2 = np.array([[10.0, 11.0, 12.0, 13.0]])
    ret2 = open_to_open_returns(open2)
    assert ret2[0, 0] == pytest.approx((12.0 - 11.0) / 11.0)
    assert ret2[0, -1] == 0.0 and ret2[0, -2] == 0.0


def test_winsorize_constant_cross_section_maps_to_zero():
    wide = pd.DataFrame([[1.0], [1.0], [1.0]], index=["A", "B", "C"], columns=["d"])
    out = winsorize_cross_section(wide)
    assert (out == 0.0).all().all()

    event_like = pd.DataFrame({"d": [0.0, 0.0, 1.0, 1.0]}, index=list("ABCD"))
    out2 = winsorize_cross_section(event_like)
    assert np.isfinite(out2.to_numpy()).all()
    assert out2["d"].abs().max() <= 5.0
