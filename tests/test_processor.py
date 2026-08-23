from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import DataConfig
from ashare_data.processor import (
    blocked_components,
    encode_industry_frame,
    is_valid_a_share_code,
    limit_rate,
    normalize_daily_bars,
    open_to_open_returns,
    pivot_wide,
    tradability_blocked,
    tradability_blocked_matrix,
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
    out = winsorize_cross_section(wide, np.ones(wide.shape, dtype=bool))
    assert out.shape == wide.shape
    assert np.isfinite(out.to_numpy()).all()

    all_nan = pd.DataFrame([[np.nan], [np.nan]], index=["A", "B"], columns=["d"])
    out_nan = winsorize_cross_section(all_nan, np.ones(all_nan.shape, dtype=bool))
    assert (out_nan == 0.0).all().all()


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
    out = winsorize_cross_section(wide, np.ones(wide.shape, dtype=bool))
    assert (out == 0.0).all().all()

    event_like = pd.DataFrame({"d": [0.0, 0.0, 1.0, 1.0]}, index=list("ABCD"))
    out2 = winsorize_cross_section(event_like, np.ones(event_like.shape, dtype=bool))
    assert np.isfinite(out2.to_numpy()).all()
    assert out2["d"].abs().max() <= 5.0


# --- PIT universe mask in cross-sectional standardization --------------------


def test_winsorize_reference_set_is_eligible_only():
    # An ineligible stock with extreme values must not move the quantile
    # points / median / MAD used to standardize the eligible stocks.
    wide = pd.DataFrame(
        {"d1": [1.0, 2.0, 3.0, 1e6], "d2": [10.0, 10.0, 10.0, 1e6]},
        index=["A", "B", "C", "FUTURE"],
    )
    eligible = np.array(
        [[True, True], [True, True], [True, True], [False, False]]
    )
    out = winsorize_cross_section(wide, eligible=eligible)
    baseline = winsorize_cross_section(wide.iloc[:3], np.ones(wide.iloc[:3].shape, dtype=bool))
    for day in ("d1", "d2"):
        assert np.allclose(out.loc[["A", "B", "C"], day], baseline[day])
    # The ineligible row keeps a bounded, finite transform of its own value
    # (own history is never zeroed) but contributed to no reference stat.
    assert np.isfinite(out.to_numpy()).all()
    assert np.abs(out.to_numpy()).max() <= 5.0


def test_winsorize_column_without_eligible_cells_is_neutral():
    wide = pd.DataFrame(
        {"d1": [1.0, 2.0, 3.0], "d2": [5.0, 5.0, 5.0]},
        index=list("ABC"),
    )
    eligible = np.array([[True, False], [True, False], [True, False]])
    out = winsorize_cross_section(wide, eligible=eligible)
    # Day 1 keeps its eligible reference set; day 2 has none -> stable
    # neutral 0, no NaN spread and no extreme values.
    assert (out["d2"] == 0.0).all()
    assert np.isfinite(out.to_numpy()).all()


def test_winsorize_rejects_mask_shape_mismatch():
    wide = pd.DataFrame([[1.0, 2.0]], index=["A"], columns=["d1", "d2"])
    with pytest.raises(ValueError, match="eligible shape"):
        winsorize_cross_section(wide, eligible=np.ones((2, 2), dtype=bool))


# --- tradability masks (shared by the backtest engine and the reward path) ---


def test_limit_rate_by_board_and_explicit_st_flag():
    assert limit_rate("000001.SZ") == 0.10
    assert limit_rate("600000.SH") == 0.10
    assert limit_rate("300001.SZ") == 0.20
    assert limit_rate("301001.SZ") == 0.20
    assert limit_rate("688001.SH") == 0.20
    assert limit_rate("689001.SH") == 0.20
    assert limit_rate("830001.BJ") == 0.10
    # The 5% band requires an explicit as-of flag: a name is display data
    # and can never imply ST status for any date.
    assert limit_rate("000001.SZ", is_st=True) == 0.05
    assert limit_rate("300001.SZ", is_st=True) == 0.05


def test_tradability_blocked_suspension_variants():
    ts_codes = ["000001.SZ"]
    ones = np.ones((1, 1))
    for col, value in (
        ("open", 0.0),
        ("volume", 0.0),
        ("pre_close", 0.0),
    ):
        kwargs = dict(
            open_col=ones.copy(),
            high_col=ones.copy(),
            low_col=ones.copy(),
            pre_close_col=ones.copy(),
            volume_col=ones.copy(),
        )
        kwargs[col + "_col"] = np.full((1, 1), value)
        assert tradability_blocked(**kwargs, ts_codes=ts_codes, side="buy")[0]
        assert tradability_blocked(**kwargs, ts_codes=ts_codes, side="sell")[0]


def test_tradability_blocked_limit_moves_are_side_specific():
    ts_codes = ["000001.SZ"]

    def blocked(open_, side, rate=0.10):
        price = np.asarray([[open_]], dtype=np.float64)
        return tradability_blocked(
            price, price, price, np.full((1, 1), 1.0), np.ones((1, 1)),
            ts_codes, side,
        )[0]

    # One-word +10% open: buy side blocked, sell side free.
    assert blocked(1.10, "buy")
    assert not blocked(1.10, "sell")
    # One-word -10% open: sell side blocked, buy side free (buying at the
    # limit-down board is legal).
    assert blocked(0.90, "sell")
    assert not blocked(0.90, "buy")
    # Not one-word (high above the open): the limit-up is breakable -> free.
    price = np.asarray([[1.10]], dtype=np.float64)
    assert not tradability_blocked(
        price, np.full((1, 1), 1.11), price, np.full((1, 1), 1.0),
        np.ones((1, 1)), ts_codes, "buy",
    )[0]


def test_tradability_st_mask_controls_5pct_band():
    # A one-word +5.2% / -5.2% open is inside the 10% board band but
    # outside the ST 5% band: only the explicit as-of st_mask applies the
    # 5% band, exactly what same-day execution needs.
    ts_codes = ["000001.SZ", "600000.SH"]
    price = np.asarray([[1.052], [1.0]], dtype=np.float64)
    pc = np.ones((2, 1))
    vol = np.ones((2, 1))
    assert not tradability_blocked(
        price, price, price, pc, vol, ts_codes, "buy"
    )[0]
    assert tradability_blocked(
        price, price, price, pc, vol, ts_codes, "buy",
        st_mask=np.asarray([True, False]),
    )[0]
    assert not tradability_blocked(
        price, price, price, pc, vol, ts_codes, "buy",
        st_mask=np.asarray([True, False]),
    )[1]

    down = np.asarray([[0.948], [1.0]], dtype=np.float64)
    assert not tradability_blocked(
        down, down, down, pc, vol, ts_codes, "sell"
    )[0]
    assert tradability_blocked(
        down, down, down, pc, vol, ts_codes, "sell",
        st_mask=np.asarray([True, False]),
    )[0]


def test_tradability_rejects_mismatched_st_mask():
    price = np.asarray([[1.052], [1.052]], dtype=np.float64)
    ones = np.ones((2, 1))
    with pytest.raises(ValueError, match="st_mask"):
        tradability_blocked(
            price, price, price, ones, ones, ["000001.SZ", "600000.SH"], "buy",
            st_mask=np.asarray([True]),
        )


def test_tradability_blocked_matrix_uses_board_rates_only():
    # The full-window matrix is historical by construction: a +5.2%
    # one-word open on a main-board stock is not a limit-up, regardless of
    # what the stock's current name says.
    open_ = np.full((1, 3), 1.052)
    ones = np.ones((1, 3))
    buy = tradability_blocked_matrix(
        open_, open_, open_, ones, ones, ["000001.SZ"], "buy"
    )
    assert not buy.any()
    sell = tradability_blocked_matrix(
        np.full((1, 3), 0.948), np.full((1, 3), 0.948), np.full((1, 3), 0.948),
        ones, ones, ["000001.SZ"], "sell",
    )
    assert not sell.any()


def test_blocked_components_decomposes_suspension_and_limit():
    ts_codes = ["000001.SZ"]
    one_word_up = np.asarray([[1.10]], dtype=np.float64)
    ones = np.ones((1, 1))
    suspended, limit = blocked_components(
        one_word_up, one_word_up, one_word_up, ones, ones, ts_codes, "buy"
    )
    assert not suspended[0] and limit[0]
    suspended, limit = blocked_components(
        one_word_up, one_word_up, one_word_up, ones, ones, ts_codes, "sell"
    )
    assert not suspended[0] and not limit[0]

    suspended, limit = blocked_components(
        np.zeros((1, 1)), ones, ones, ones, ones, ts_codes, "buy"
    )
    assert suspended[0] and not limit[0]


def test_tradability_blocked_matrix_matches_per_day_columns():
    rng = np.random.default_rng(31)
    ts_codes = [f"{i:06d}.SZ" for i in range(1, 6)]
    open_ = np.abs(rng.normal(10.0, 0.5, (5, 8)))
    pre_close = np.roll(open_, 1, axis=1)
    pre_close[:, 0] = open_[:, 0]
    high = open_ * 1.03
    low = open_ * 0.97
    volume = np.abs(rng.normal(1e6, 1e5, (5, 8)))
    # A forced suspension and two one-word limit opens.
    open_[0, 3] = 0.0
    open_[1, 4] = pre_close[1, 4] * 1.10
    high[1, 4] = low[1, 4] = open_[1, 4]
    open_[2, 5] = pre_close[2, 5] * 0.90
    high[2, 5] = low[2, 5] = open_[2, 5]

    for side in ("buy", "sell"):
        matrix = tradability_blocked_matrix(
            open_, high, low, pre_close, volume, ts_codes, side
        )
        for day in range(8):
            per_day = tradability_blocked(
                open_[:, day], high[:, day], low[:, day],
                pre_close[:, day], volume[:, day], ts_codes, side,
            )
            assert (matrix[:, day] == per_day).all()
    buy = tradability_blocked_matrix(open_, high, low, pre_close, volume, ts_codes, "buy")
    sell = tradability_blocked_matrix(open_, high, low, pre_close, volume, ts_codes, "sell")
    assert buy[0, 3] and sell[0, 3]  # suspension blocks both sides
    assert buy[1, 4] and not sell[1, 4]  # one-word limit-up blocks buys only
    assert sell[2, 5] and not buy[2, 5]  # one-word limit-down blocks sells only


def test_tradability_rejects_bad_inputs():
    with pytest.raises(ValueError, match="share one shape"):
        tradability_blocked_matrix(
            np.ones((2, 3)), np.ones((2, 3)), np.ones((2, 3)),
            np.ones((2, 3)), np.ones((2, 4)), ["A", "B"], "buy",
        )
    with pytest.raises(ValueError, match="expected \\[stock x date\\]"):
        tradability_blocked_matrix(
            np.ones(3), np.ones(3), np.ones(3), np.ones(3), np.ones(3),
            ["A", "B", "C"], "buy",
        )
    with pytest.raises(ValueError, match="unknown side"):
        tradability_blocked(
            np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1)),
            np.ones((2, 1)), np.ones((2, 1)), ["A", "B"], "hold",
        )


# --- industry-code encoding for the VM's group neutralization ----------------


def test_encode_industry_frame_maps_codes_and_keeps_nan():
    frame = pd.DataFrame(
        {
            "d1": ["801010", "801010", "801030", None],
            "d2": ["801010", None, "801030", None],
        },
        index=["A", "B", "C", "D"],
    )
    out = encode_industry_frame(frame)
    assert out.shape == (4, 2)
    assert out.dtype == np.float32
    # Same code -> same id; different code -> different id; NaN stays NaN.
    assert out[0, 0] == out[1, 0]
    assert out[0, 0] != out[2, 0]
    assert np.isnan(out[3, 0])
    assert np.isnan(out[1, 1])
    # Ids are dense: 0..n-1 across the mapped cells.
    mapped = out[np.isfinite(out)]
    assert set(mapped.tolist()) == set(range(int(np.nanmax(out)) + 1))


def test_encode_industry_frame_all_nan_stays_all_nan():
    frame = pd.DataFrame(
        np.nan, index=["A", "B"], columns=["d1", "d2"], dtype=object
    )
    out = encode_industry_frame(frame)
    assert np.isnan(out).all()
