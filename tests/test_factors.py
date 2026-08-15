from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from ashare_data.capital_flow import EXTERNAL_FACTOR_NAMES
from ashare_data.fundamentals import FUNDAMENTAL_PIT_NAMES
from ashare_model.factors import (
    BAR_COLUMNS,
    FACTOR_REGISTRY,
    FAMILIES,
    NEUTRAL_FEATURE_NAMES,
    AshareFactorEngine,
    _factor_amount_share,
    _factor_atr,
    _factor_macd,
    _factor_rsi,
    _returns,
    _rolling_capm,
    _rolling_max,
    _shift_ratio,
    compute_factor_tensor,
)
from ashare_model.vocab import FEATURE_NAMES


def test_compute_factor_tensor_shape_and_finite(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(bars, ts_codes, dates)
    assert tensor.shape == (len(FEATURE_NAMES), len(ts_codes), len(dates))
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_compute_factor_tensor_without_pit_fundamentals(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(bars, ts_codes, dates, pit_fundamentals={})
    assert tensor.shape == (len(FEATURE_NAMES), len(ts_codes), len(dates))


def test_factor_engine_helpers(bars_data):
    dates, ts_codes, bars = bars_data
    engine = AshareFactorEngine()
    close = engine._pivot(bars, ts_codes, dates, "close")
    returns = _returns(close)
    assert returns.shape == close.shape
    # The first day has no prior close: the honest value is NaN (the
    # cross-sectional standardizer later maps it to the neutral 0).
    assert returns.iloc[:, 0].isna().all()
    assert returns.iloc[0, 1] == pytest.approx(close.iloc[0, 1] / close.iloc[0, 0] - 1.0)
    shifted = _shift_ratio(close, 5)
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


def test_pit_fundamentals_injected_and_missing_stay_neutral(bars_data):
    dates, ts_codes, bars = bars_data
    # A hand-built PIT frame for one feature: informative only on the dates
    # where the builder placed values; every other feature/date is neutral.
    pe_frame = pd.DataFrame(np.nan, index=ts_codes, columns=dates)
    pe_frame.loc["000001.SZ", dates[5]] = 15.0
    pe_frame.loc["600000.SH", dates[5]] = 8.0
    tensor = compute_factor_tensor(bars, ts_codes, dates, pit_fundamentals={"PE_TTM": pe_frame})
    pe = tensor[FEATURE_NAMES.index("PE_TTM")]
    assert pe[0, 5] > 0
    assert np.allclose(pe[:, :5], 0.0)
    assert np.allclose(pe[:, 6:], 0.0)
    # An unprovided fundamental feature stays fully neutral.
    pb = tensor[FEATURE_NAMES.index("PB")]
    assert np.allclose(pb, 0.0)


def test_registry_covers_every_vocabulary_feature():
    registered = set(FACTOR_REGISTRY)
    declared = (
        set(FUNDAMENTAL_PIT_NAMES)
        | set(EXTERNAL_FACTOR_NAMES)
        | set(NEUTRAL_FEATURE_NAMES)
    )
    # Every vocabulary feature is either a registered local factor or a
    # declared PIT-fundamental/external/neutral one; the sets never overlap.
    assert registered.isdisjoint(declared)
    assert set(FUNDAMENTAL_PIT_NAMES).isdisjoint(EXTERNAL_FACTOR_NAMES)
    assert set(FUNDAMENTAL_PIT_NAMES).isdisjoint(NEUTRAL_FEATURE_NAMES)
    assert set(EXTERNAL_FACTOR_NAMES).isdisjoint(NEUTRAL_FEATURE_NAMES)
    assert registered | declared == set(FEATURE_NAMES)
    assert registered.issubset(FEATURE_NAMES)


def test_registry_metadata_is_valid():
    for name, (spec, fn) in FACTOR_REGISTRY.items():
        assert spec.name == name
        assert spec.family in FAMILIES
        assert spec.warmup >= 1
        assert spec.description
        assert all(col in BAR_COLUMNS for col in spec.required_columns)
        assert callable(fn)


def test_fundamental_and_neutral_families_declared():
    # PIT fundamentals come from the point-in-time pipeline, the external
    # factors from the capital-flow pipeline, and NORTHBOUND_CHG stays
    # neutral (its daily feed stopped in Aug 2024); all metadata-driven.
    assert FUNDAMENTAL_PIT_NAMES
    assert set(EXTERNAL_FACTOR_NAMES) == {"MARGIN_BALANCE_CHG", "INDUSTRY_MOMENTUM"}
    assert NEUTRAL_FEATURE_NAMES == {"NORTHBOUND_CHG"}
    assert "MARKET_CAP" in FACTOR_REGISTRY  # local daily-bar size proxy


def test_extra_frames_injected_and_missing_stay_neutral(bars_data):
    dates, ts_codes, bars = bars_data
    margin_frame = pd.DataFrame(np.nan, index=ts_codes, columns=dates)
    margin_frame.loc["000001.SZ", dates[5]] = 0.5
    margin_frame.loc["600000.SH", dates[5]] = -0.5
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, extra_frames={"MARGIN_BALANCE_CHG": margin_frame}
    )
    margin = tensor[FEATURE_NAMES.index("MARGIN_BALANCE_CHG")]
    assert margin[0, 5] > 0
    assert margin[1, 5] < 0
    assert np.allclose(margin[:, :5], 0.0)
    assert np.allclose(margin[:, 6:], 0.0)
    # Unprovided external and neutral features stay fully neutral.
    industry = tensor[FEATURE_NAMES.index("INDUSTRY_MOMENTUM")]
    assert np.allclose(industry, 0.0)


# Golden checksum of the first-generation 34-feature tensor on the
# deterministic ``make_bars`` fixture.  It pins the exact legacy values so
# any refactor that silently changes an existing factor fails this test.
# The checksum was re-captured once, deliberately, when MARKET_CAP gained
# real data (float market cap = amount/turnover_rate) in the PIT pipeline
# phase; all other v1 features kept their pre-refactor values.
_GOLDEN_TENSOR_SHA256 = "f7a480bc9c20d416d369914ddce7c57197d06a2e133af480e5afe545c343e79b"


def test_factor_tensor_matches_pre_refactor_golden_values(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(bars, ts_codes, dates)
    legacy = np.ascontiguousarray(tensor[:34])
    assert hashlib.sha256(legacy.tobytes()).hexdigest() == _GOLDEN_TENSOR_SHA256
    assert legacy.sum() == pytest.approx(199.34854, abs=1e-3)
    ret1 = tensor[FEATURE_NAMES.index("RET_1")]
    assert ret1[0, 10] == pytest.approx(-0.00039433446, abs=1e-6)
    assert tensor[FEATURE_NAMES.index("SKEW_20")][0, 30] == pytest.approx(-0.004940729, abs=1e-6)


def test_engine_computes_subset_of_features(bars_data):
    dates, ts_codes, bars = bars_data
    engine = AshareFactorEngine(feature_names=["RET_1", "PE_TTM", "NORTHBOUND_CHG"])
    tensor = engine.compute_factor_tensor(bars, ts_codes, dates)
    assert tensor.shape == (3, len(ts_codes), len(dates))
    # Unbacked features stay neutral 0 everywhere; the local factor works.
    assert np.allclose(tensor[1], 0.0)
    assert np.allclose(tensor[2], 0.0)
    assert tensor[0][0, 1] != 0.0


def _factor_fn(name):
    return FACTOR_REGISTRY[name][1]


def _context_for(bars: pd.DataFrame, dates: list[str], codes: list[str], names: list[str]):
    engine = AshareFactorEngine(feature_names=names)
    close = engine._pivot(bars, codes, dates, "close")
    return engine._build_context(bars, codes, dates, close)


def _rows_to_bars(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_overnight_and_intraday_decomposition_values():
    dates = ["20240101", "20240102", "20240103"]
    codes = ["000001.SZ"]
    rows = [
        {"ts_code": "000001.SZ", "trade_date": "20240101", "open": 10.0, "high": 10.6, "low": 9.9, "close": 10.5, "pre_close": 9.9, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "000001.SZ", "trade_date": "20240102", "open": 11.0, "high": 11.6, "low": 10.9, "close": 11.5, "pre_close": 10.5, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "open": 12.0, "high": 12.6, "low": 11.9, "close": 12.5, "pre_close": 11.5, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
    ]
    bars = _rows_to_bars(rows)
    ctx = _context_for(bars, dates, codes, ["OVERNIGHT_RET", "INTRADAY_RET"])
    overnight = _factor_fn("OVERNIGHT_RET")(ctx)
    intraday = _factor_fn("INTRADAY_RET")(ctx)
    # With an official pre_close the overnight return is defined from day 1.
    assert overnight.iloc[0, 0] == pytest.approx(10.0 / 9.9 - 1.0)
    assert overnight.iloc[0, 1] == pytest.approx(11.0 / 10.5 - 1.0)
    assert overnight.iloc[0, 2] == pytest.approx(12.0 / 11.5 - 1.0)
    assert intraday.iloc[0, 1] == pytest.approx(11.5 / 11.0 - 1.0)
    assert intraday.iloc[0, 2] == pytest.approx(12.5 / 12.0 - 1.0)


def test_amount_share_sums_to_one():
    amount = pd.DataFrame(
        {"d1": [30.0, 70.0], "d2": [50.0, 50.0], "d3": [np.nan, np.nan]},
        index=["A", "B"],
    )
    share = _factor_amount_share(amount)
    assert share.loc["A", "d1"] == pytest.approx(0.3)
    assert share.loc["B", "d1"] == pytest.approx(0.7)
    assert share.loc["A", "d2"] == pytest.approx(0.5)
    # A day with no valid amount anywhere stays NaN (neutral), not 1/n.
    assert share[["d3"]].isna().all().all()


def test_illiq_matches_amihud_definition():
    dates = ["20240101", "20240102", "20240103"]
    codes = ["000001.SZ"]
    rows = [
        {"ts_code": "000001.SZ", "trade_date": d, "open": 10.0, "high": 10.0, "low": 10.0,
         "close": c, "pre_close": 10.0, "volume": 1e6, "amount": amt, "turnover_rate": 1.0, "adj_factor": 1.0}
        for d, c, amt in zip(dates, [10.0, 11.0, 11.1], [1e6, 1e6, 1e6])
    ]
    ctx = _context_for(_rows_to_bars(rows), dates, codes, ["ILLIQ_20"])
    illiq = _factor_fn("ILLIQ_20")(ctx)
    # Day 2: |0.1| / 1e6; day 3: mean(|0.1|, |0.1/11|) / 1e6.
    assert illiq.iloc[0, 1] == pytest.approx(0.1 / 1e6)
    assert illiq.iloc[0, 2] == pytest.approx((0.1 + 0.1 / 11.0) / 2 / 1e6)


def test_max20_and_high52w_values():
    dates = [f"202401{i:02d}" for i in range(1, 6)]
    codes = ["000001.SZ"]
    closes = [10.0, 11.0, 9.0, 12.0, 10.0]
    rows = [
        {"ts_code": "000001.SZ", "trade_date": d, "open": c, "high": c, "low": c,
         "close": c, "pre_close": c, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0}
        for d, c in zip(dates, closes)
    ]
    ctx = _context_for(_rows_to_bars(rows), dates, codes, ["MAX_20", "HIGH_52W"])
    max_ret = _factor_fn("MAX_20")(ctx)
    dist = _factor_fn("HIGH_52W")(ctx)
    # Daily returns: [NaN, 0.1, -0.1818, 0.3333, -0.1667]; trailing max.
    assert max_ret.iloc[0, 3] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert max_ret.iloc[0, 4] == pytest.approx(1.0 / 3.0, abs=1e-4)
    # Expanding 52-week high: day 5 close 10 vs max 12 -> -1/6.
    assert dist.iloc[0, 4] == pytest.approx(10.0 / 12.0 - 1.0)


def test_rolling_capm_recovers_known_betas():
    # Stock A = 2*mkt noise-free, stock B = 0.5*mkt.  The internal market is
    # their equal-weight mean (1.25*mkt), so the recovered betas are 1.6 and
    # 0.4 with zero idiosyncratic volatility and R-squared 1.
    n = 30
    m = np.sin(np.linspace(0.1, 2.0, n)) * 0.01
    ret_a = 2.0 * m
    ret_b = 0.5 * m
    close = pd.DataFrame(
        {
            f"d{i}": [
                10.0 * np.prod(1.0 + ret_a[: i + 1]),
                10.0 * np.prod(1.0 + ret_b[: i + 1]),
            ]
            for i in range(n)
        },
        index=["A", "B"],
    )
    beta, ivol, rsq = _rolling_capm(close, window=60, min_periods=2)
    assert beta.loc["A"].iloc[-1] == pytest.approx(1.6, abs=1e-4)
    assert beta.loc["B"].iloc[-1] == pytest.approx(0.4, abs=1e-4)
    assert ivol.loc["A"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert rsq.loc["A"].iloc[-1] == pytest.approx(1.0, abs=1e-4)
    assert rsq.loc["B"].iloc[-1] == pytest.approx(1.0, abs=1e-4)
    # The leading positions lack enough history and stay NaN (neutral).
    assert np.isnan(beta.loc["A"].iloc[0])


def test_rsi_extremes():
    dates = [f"d{i}" for i in range(20)]
    up = pd.DataFrame({"d": [10.0 + i for i in range(20)]}).T
    up.columns = dates
    down = pd.DataFrame({"d": [30.0 - i for i in range(20)]}).T
    down.columns = dates
    rsi_up = _factor_rsi(up, 14)
    rsi_down = _factor_rsi(down, 14)
    assert rsi_up.iloc[0, -1] == pytest.approx(100.0)
    assert rsi_down.iloc[0, -1] == pytest.approx(0.0)
    # The warm-up days before 14 observations stay NaN (neutral).
    assert np.isnan(rsi_up.iloc[0, 12])


def test_atr_matches_definition():
    dates = [f"d{i}" for i in range(5)]
    high = pd.DataFrame([[12.0] * 5], index=["A"], columns=dates)
    low = pd.DataFrame([[10.0] * 5], index=["A"], columns=dates)
    pre_close = pd.DataFrame([[11.0] * 5], index=["A"], columns=dates)
    atr = _factor_atr(high, low, pre_close, 14)
    # TR = max(high-low, |high-prev|, |low-prev|) = max(2, 1, 1) = 2.
    assert atr.iloc[0, -1] == pytest.approx(2.0)


def test_macd_of_constant_close_is_zero():
    dates = [f"d{i}" for i in range(30)]
    close = pd.DataFrame([[10.0] * 30], index=["A"], columns=dates)
    dif, dea = _factor_macd(close)
    assert dif.iloc[0, -1] == pytest.approx(0.0, abs=1e-9)
    assert dea.iloc[0, -1] == pytest.approx(0.0, abs=1e-9)


def test_suspension_days_and_list_age_count_correctly():
    dates = [f"d{i}" for i in range(5)]
    close = pd.DataFrame(
        [[np.nan, 10.0, np.nan, np.nan, 10.0]], index=["A"], columns=dates
    )
    ctx = _context_for(
        pd.DataFrame({"ts_code": ["A"] * 5, "trade_date": dates, "close": close.iloc[0].tolist(),
                      "open": 1.0, "high": 1.0, "low": 1.0, "pre_close": 1.0,
                      "volume": 1.0, "amount": 1.0, "turnover_rate": 1.0, "adj_factor": 1.0}),
        dates,
        ["A"],
        ["SUSPEND_DAYS_60", "LIST_AGE"],
    )
    susp = _factor_fn("SUSPEND_DAYS_60")(ctx)
    age = _factor_fn("LIST_AGE")(ctx)
    # Trailing (expanding here) count of missing days: 1,1,2,3,3.
    assert susp.iloc[0].tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]
    # Listed days so far: 0,1,1,1,2.
    assert age.iloc[0].tolist() == [0.0, 1.0, 1.0, 1.0, 2.0]


def test_new_factors_have_no_lookahead(bars_data):
    # Perturbing the last day must not change any new factor on earlier
    # dates (all window/EMA recursions are trailing-only).
    dates, ts_codes, bars = bars_data
    names = [
        "OVERNIGHT_RET", "INTRADAY_RET", "ILLIQ_20", "AMOUNT_SHARE", "MAX_20",
        "HIGH_52W", "BETA_60", "IVOL_60", "RSQ_60", "BIAS_20", "RSI_14",
        "ATR_14", "MACD_DIF", "MACD_DEA", "SUSPEND_DAYS_60", "LIST_AGE",
    ]
    engine = AshareFactorEngine(feature_names=names)
    base = engine.compute_factor_tensor(bars, ts_codes, dates)
    shocked = bars.copy()
    last = dates[-1]
    shocked.loc[shocked["trade_date"] == last, "close"] *= 5.0
    shocked.loc[shocked["trade_date"] == last, "high"] *= 5.0
    shocked.loc[shocked["trade_date"] == last, "low"] *= 5.0
    after = engine.compute_factor_tensor(shocked, ts_codes, dates)
    assert np.allclose(base[:, :, :-1], after[:, :, :-1])
