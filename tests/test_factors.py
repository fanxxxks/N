from __future__ import annotations

import hashlib
import warnings

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
    FactorContext,
    _factor_amount_share,
    _factor_atr,
    _factor_macd,
    _factor_rsi,
    _industry_demean,
    _limit_break,
    _limit_streak,
    _limit_up_count,
    _returns,
    _rolling_capm,
    _rolling_max,
    _shift_ratio,
    _turnover_smoothed,
    _turnover_vol,
    compute_factor_tensor,
)
from ashare_model.vocab import FEATURE_NAMES


def test_compute_factor_tensor_shape_and_finite(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool)
    )
    assert tensor.shape == (len(FEATURE_NAMES), len(ts_codes), len(dates))
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_compute_factor_tensor_without_pit_fundamentals(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool),
        pit_fundamentals={},
    )
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
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool)
    )
    first_factor = tensor[0]
    assert np.nanmax(np.abs(first_factor)) <= 5.0 + 1e-6


def test_missing_returns_are_neutral_in_tensor(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool)
    )
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
    tensor = compute_factor_tensor(
        bars, codes, dates, np.ones((len(codes), len(dates)), dtype=bool)
    )
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
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool),
        pit_fundamentals={"PE_TTM": pe_frame},
    )
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
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool),
        extra_frames={"MARGIN_BALANCE_CHG": margin_frame},
    )
    margin = tensor[FEATURE_NAMES.index("MARGIN_BALANCE_CHG")]
    assert margin[0, 5] > 0
    assert margin[1, 5] < 0
    assert np.allclose(margin[:, :5], 0.0)
    assert np.allclose(margin[:, 6:], 0.0)
    # Unprovided external and neutral features stay fully neutral.
    industry = tensor[FEATURE_NAMES.index("INDUSTRY_MOMENTUM")]
    assert np.allclose(industry, 0.0)


# Golden checksum of the first-generation feature block on the deterministic
# ``make_bars`` fixture.  It pins the exact legacy values so any refactor
# that silently changes an existing factor fails this test.  The checksum
# was re-captured twice, deliberately: once when MARKET_CAP gained real data
# (float market cap = amount/turnover_rate) in the PIT pipeline phase, and
# once when the v3 generation retired RET_20 (an exact duplicate of
# MOMENTUM_20, kept resolvable through FEATURE_ALIASES) - all other v1
# features keep their pre-refactor values.
_GOLDEN_TENSOR_SHA256 = "0eabd18dc9f100bdc9f9ac94c72688bf28424e83fc7955ce63edd3d4d43b061f"


def test_factor_tensor_matches_pre_refactor_golden_values(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool)
    )
    legacy = np.ascontiguousarray(tensor[:33])
    assert hashlib.sha256(legacy.tobytes()).hexdigest() == _GOLDEN_TENSOR_SHA256
    assert legacy.sum() == pytest.approx(199.50612, abs=1e-3)
    ret1 = tensor[FEATURE_NAMES.index("RET_1")]
    assert ret1[0, 10] == pytest.approx(-0.00039433446, abs=1e-6)
    assert tensor[FEATURE_NAMES.index("SKEW_20")][0, 30] == pytest.approx(-0.004940729, abs=1e-6)
    # The retired duplicate is gone from the live vocabulary and registry;
    # MOMENTUM_20 keeps the identical 20-day return semantics in its place.
    assert "RET_20" not in FEATURE_NAMES
    assert "RET_20" not in FACTOR_REGISTRY
    assert "MOMENTUM_20" in FACTOR_REGISTRY


def test_engine_computes_subset_of_features(bars_data):
    dates, ts_codes, bars = bars_data
    engine = AshareFactorEngine(feature_names=["RET_1", "PE_TTM", "NORTHBOUND_CHG"])
    tensor = engine.compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool)
    )
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
    eligible = pd.DataFrame(True, index=close.index, columns=close.columns)
    return engine._build_context(bars, codes, dates, close, eligible)


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
    eligible = pd.DataFrame(True, index=amount.index, columns=amount.columns)
    share = _factor_amount_share(amount, eligible)
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
    beta, ivol, rsq = _rolling_capm(
        close, np.ones(close.shape, dtype=bool), window=60, min_periods=2
    )
    assert beta.loc["A"].iloc[-1] == pytest.approx(1.6, abs=1e-4)
    assert beta.loc["B"].iloc[-1] == pytest.approx(0.4, abs=1e-4)
    assert ivol.loc["A"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert rsq.loc["A"].iloc[-1] == pytest.approx(1.0, abs=1e-4)
    assert rsq.loc["B"].iloc[-1] == pytest.approx(1.0, abs=1e-4)
    # The leading positions lack enough history and stay NaN (neutral).
    assert np.isnan(beta.loc["A"].iloc[0])


def test_rolling_capm_aligns_market_window_to_stock_suspension_gaps():
    """The regression pairs are the stock's own trading days.

    A suspended stock must not accumulate market returns from its
    suspension sessions: the market window sums are masked per stock, so a
    stock suspended through a market rally regresses on the rally-free
    subset of sessions exactly like an explicit aligned regression.  The
    reference recomputes every (stock, date) cell with plain Python sums
    over the trailing positional window, pairing the stock's return with
    the market return of the same session only.
    """
    rng = np.random.default_rng(7)
    n = 60
    ret_a = rng.normal(0.0, 0.01, n)
    ret_b = rng.normal(0.0, 0.01, n)
    ret_c = rng.normal(0.0, 0.02, n)
    # C is suspended on scattered sessions; while it is suspended the
    # market rallies hard — the exact regime that biases beta when the
    # market window is not aligned to the stock's own sessions.
    suspended = np.zeros(n, dtype=bool)
    suspended[10:20] = True
    suspended[35:40] = True
    ret_a = ret_a + np.where(suspended, 0.04, 0.0)
    ret_b = ret_b + np.where(suspended, 0.04, 0.0)

    def build_close(rets: np.ndarray, gaps: np.ndarray) -> np.ndarray:
        # Missing bars stay NaN exactly like pivot_wide leaves them.
        out = np.full(n, np.nan)
        last = 10.0
        for i in range(n):
            if gaps[i]:
                continue
            last *= 1.0 + rets[i]
            out[i] = last
        return out

    dates = [f"d{i}" for i in range(n)]
    close = pd.DataFrame(
        {
            "A": build_close(ret_a, np.zeros(n, dtype=bool)),
            "B": build_close(ret_b, np.zeros(n, dtype=bool)),
            "C": build_close(ret_c, suspended),
        },
        index=dates,
    ).T
    eligible = np.ones(close.shape, dtype=bool)

    # --- explicit aligned reference ------------------------------------
    r = close.pct_change(fill_method=None, axis=1)
    valid = r.notna().to_numpy()
    r_arr = r.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        market = np.nan_to_num(
            np.nanmean(np.where(valid, r_arr, np.nan), axis=0), nan=0.0
        )
    window, min_periods = 30, 10
    ref = {"beta": np.full((3, n), np.nan),
           "ivol": np.full((3, n), np.nan),
           "rsq": np.full((3, n), np.nan)}
    for s in range(3):
        for t in range(n):
            lo = max(0, t - window + 1)
            pairs = [
                (market[pos], r_arr[s, pos])
                for pos in range(lo, t + 1)
                if valid[s, pos]
            ]
            m = len(pairs)
            if m < min_periods:
                continue
            x = np.array([p[0] for p in pairs])
            y = np.array([p[1] for p in pairs])
            sx, sy = x.sum(), y.sum()
            cov = (x * y).sum() - sx * sy / m
            var_m = (x * x).sum() - sx * sx / m
            var_r = (y * y).sum() - sy * sy / m
            if var_m <= 1e-12 or var_r <= 1e-12:
                continue
            beta = cov / var_m
            ref["beta"][s, t] = beta
            ref["rsq"][s, t] = cov * cov / (var_r * var_m)
            ref["ivol"][s, t] = np.sqrt(max(var_r - beta * cov, 0.0))

    beta, ivol, rsq = _rolling_capm(
        close, eligible, window=window, min_periods=min_periods
    )
    np.testing.assert_allclose(beta.to_numpy(), ref["beta"], atol=1e-10)
    np.testing.assert_allclose(ivol.to_numpy(), ref["ivol"], atol=1e-10)
    np.testing.assert_allclose(rsq.to_numpy(), ref["rsq"], atol=1e-10)


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
        "RET_120", "REVERSAL_60", "REVERSAL_120",
        "TURNOVER_MA5", "TURNOVER_MA20", "TURNOVER_STD20",
        "LIMIT_STREAK", "LIMIT_UP_CNT_20", "LIMIT_BREAK",
        "IND_REL_RET_5", "IND_REL_RET_20", "IND_REL_VOL_20", "IND_REL_TURNOVER",
    ]
    industry = pd.DataFrame(np.nan, index=ts_codes, columns=dates, dtype=object)
    industry.loc["000001.SZ"] = "801780"
    industry.loc["600000.SH"] = "801780"
    industry.loc["300001.SZ"] = "801880"
    engine = AshareFactorEngine(feature_names=names)
    base = engine.compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool),
        industry_frame=industry,
    )
    shocked = bars.copy()
    last = dates[-1]
    shocked.loc[shocked["trade_date"] == last, "close"] *= 5.0
    shocked.loc[shocked["trade_date"] == last, "high"] *= 5.0
    shocked.loc[shocked["trade_date"] == last, "low"] *= 5.0
    after = engine.compute_factor_tensor(
        shocked, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool),
        industry_frame=industry,
    )
    assert np.allclose(base[:, :, :-1], after[:, :, :-1])


# --- factor P0 additions (v3 generation) -------------------------------------


def _empty_ctx(index: list[str], columns: list[str]) -> FactorContext:
    empty = pd.DataFrame(np.nan, index=index, columns=columns)
    return FactorContext(
        ts_codes=list(index),
        dates=list(columns),
        close=empty.copy(),
        open=empty.copy(),
        high=empty.copy(),
        low=empty.copy(),
        pre_close=empty.copy(),
        volume=empty.copy(),
        amount=empty.copy(),
        turnover=empty.copy(),
        eligible=pd.DataFrame(True, index=index, columns=columns),
    )


def test_industry_demean_subtracts_industry_means():
    frame = pd.DataFrame(
        {"d1": [1.0, 2.0, 3.0, 10.0], "d2": [4.0, 6.0, 8.0, 20.0]},
        index=["A", "B", "C", "D"],
    )
    ctx = _empty_ctx(list(frame.index), list(frame.columns))
    ctx.industry = pd.DataFrame(
        {"d1": ["X", "X", "Y", np.nan], "d2": ["X", "X", "Y", np.nan]},
        index=frame.index,
        columns=frame.columns,
        dtype=object,
    )
    out = _industry_demean(ctx, frame)
    # X mean d1 = 1.5 -> A: -0.5, B: +0.5; a single-member industry has no
    # cross-industry dispersion and demeans to exactly 0; unmapped stocks
    # stay NaN (neutral).
    assert out.loc["A", "d1"] == pytest.approx(-0.5)
    assert out.loc["B", "d1"] == pytest.approx(0.5)
    assert out.loc["A", "d2"] == pytest.approx(-1.0)
    assert out.loc["C", "d1"] == pytest.approx(0.0)
    assert np.isnan(out.loc["D", "d1"])


def test_industry_demean_without_frame_is_all_neutral():
    frame = pd.DataFrame([[1.0, 2.0]], index=["A"], columns=["d1", "d2"])
    out = _industry_demean(_empty_ctx(["A"], ["d1", "d2"]), frame)
    assert out.isna().all().all()


def test_industry_relative_factors_neutral_without_frame(bars_data):
    dates, ts_codes, bars = bars_data
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool)
    )
    for name in ("IND_REL_RET_5", "IND_REL_RET_20", "IND_REL_VOL_20", "IND_REL_TURNOVER"):
        assert np.allclose(tensor[FEATURE_NAMES.index(name)], 0.0), name


def test_industry_relative_tensor_demeaned(bars_data):
    dates, ts_codes, bars = bars_data
    industry = pd.DataFrame(np.nan, index=ts_codes, columns=dates, dtype=object)
    industry.loc["000001.SZ"] = "801780"
    industry.loc["600000.SH"] = "801780"
    industry.loc["300001.SZ"] = "801880"
    tensor = compute_factor_tensor(
        bars, ts_codes, dates, np.ones((len(ts_codes), len(dates)), dtype=bool),
        industry_frame=industry,
    )
    # The two same-industry stocks have different raw returns, so their
    # demeaned values are exact opposites; the single-member industry
    # demeans to exactly zero and stays neutral after standardization.
    for name in ("IND_REL_RET_5", "IND_REL_RET_20", "IND_REL_VOL_20"):
        row = tensor[FEATURE_NAMES.index(name)]
        assert np.allclose(row[0], -row[1]), name
        assert np.allclose(row[2], 0.0), name
    # The fixture's turnover is identical across stocks: the demeaned
    # turnover is zero everywhere by construction.
    assert np.allclose(tensor[FEATURE_NAMES.index("IND_REL_TURNOVER")], 0.0)


def test_turnover_smoothed_means_available_days_and_masks_missing():
    turnover = pd.DataFrame(
        {"d1": [1.0, 2.0], "d2": [3.0, np.nan], "d3": [np.nan, 4.0], "d4": [5.0, 6.0]},
        index=["A", "B"],
    )
    ctx = _empty_ctx(["A", "B"], ["d1", "d2", "d3", "d4"])
    ctx.turnover = turnover
    ma = _turnover_smoothed(ctx, 5)
    # Expanding mean over the available trailing values.
    assert ma.loc["A", "d2"] == pytest.approx(2.0)
    assert ma.loc["A", "d4"] == pytest.approx(3.0)  # mean(1, 3, 5)
    assert ma.loc["B", "d4"] == pytest.approx(4.0)  # mean(2, 4, 6)
    # A missing current turnover never fabricates a smoothed value.
    assert np.isnan(ma.loc["A", "d3"])
    assert np.isnan(ma.loc["B", "d2"])


def test_turnover_vol_std_and_missing_mask():
    turnover = pd.DataFrame(
        {"d1": [1.0, np.nan], "d2": [3.0, 2.0], "d3": [np.nan, 4.0], "d4": [5.0, 6.0]},
        index=["A", "B"],
    )
    ctx = _empty_ctx(["A", "B"], ["d1", "d2", "d3", "d4"])
    ctx.turnover = turnover
    vol = _turnover_vol(ctx, 20)
    # Sample std (ddof=1) over the available trailing values; a single
    # observation and a missing current day stay NaN.
    assert np.isnan(vol.loc["A", "d1"])
    assert vol.loc["A", "d2"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert np.isnan(vol.loc["A", "d3"])
    assert vol.loc["B", "d4"] == pytest.approx(np.std([2.0, 4.0, 6.0], ddof=1))


def _limit_rows(dates: list[str], closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    highs = highs or closes
    lows = lows or closes
    rows = []
    for i, d in enumerate(dates):
        pre = closes[i - 1] if i else closes[0] / 1.1
        rows.append(
            {
                "ts_code": "000001.SZ",
                "trade_date": d,
                "open": closes[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "pre_close": pre,
                "volume": 1e6,
                "amount": 1e7,
                "turnover_rate": 1.0,
                "adj_factor": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_limit_streak_counts_consecutive_one_word_days():
    dates = [f"d{i}" for i in range(6)]
    # 10% board: 4 consecutive locked limit-ups, a normal day, a normal day.
    closes = [11.0, 12.1, 13.31, 14.64, 15.5, 16.0]
    highs = [11.0, 12.1, 13.31, 14.64, 15.6, 16.1]
    lows = [11.0, 12.1, 13.31, 14.64, 15.0, 15.9]
    ctx = _context_for(_limit_rows(dates, closes, highs, lows), dates, ["000001.SZ"], ["LIMIT_STREAK"])
    streak = _factor_fn("LIMIT_STREAK")(ctx)
    assert streak.iloc[0].tolist() == [1.0, 2.0, 3.0, 4.0, 0.0, 0.0]


def test_limit_up_count_20_accumulates_events():
    dates = [f"d{i}" for i in range(6)]
    closes = [11.0, 12.1, 13.31, 14.64, 15.5, 16.0]
    highs = [11.0, 12.1, 13.31, 14.64, 15.6, 16.1]
    lows = [11.0, 12.1, 13.31, 14.64, 15.0, 15.9]
    ctx = _context_for(_limit_rows(dates, closes, highs, lows), dates, ["000001.SZ"], ["LIMIT_UP_CNT_20"])
    count = _factor_fn("LIMIT_UP_CNT_20")(ctx)
    assert count.iloc[0].tolist() == [1.0, 2.0, 3.0, 4.0, 4.0, 4.0]


def test_limit_break_detects_touch_without_lock():
    dates = ["d0", "d1", "d2", "d3"]
    rows = [
        # d0: normal day, never touched the limit.
        {"ts_code": "000001.SZ", "trade_date": "d0", "open": 9.9, "high": 10.2, "low": 9.9,
         "close": 10.0, "pre_close": 9.9, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        # d1: touched the +10% limit (11.0) intraday but closed below: a break.
        {"ts_code": "000001.SZ", "trade_date": "d1", "open": 10.2, "high": 11.0, "low": 10.2,
         "close": 10.5, "pre_close": 10.0, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        # d2: sealed limit-up close (opened below the limit, closed locked):
        # touched, but not a break even though the day was not one-word.
        {"ts_code": "000001.SZ", "trade_date": "d2", "open": 11.0, "high": 11.55, "low": 11.0,
         "close": 11.55, "pre_close": 10.5, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        # d3: normal day after the lock.
        {"ts_code": "000001.SZ", "trade_date": "d3", "open": 11.0, "high": 11.2, "low": 10.9,
         "close": 11.0, "pre_close": 11.55, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
    ]
    ctx = _context_for(pd.DataFrame(rows), dates, ["000001.SZ"], ["LIMIT_BREAK"])
    breaks = _factor_fn("LIMIT_BREAK")(ctx)
    assert breaks.iloc[0].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_limit_break_uses_board_specific_limit_rate():
    # A 20%-board stock (300001.SZ) touching +20% without locking is a
    # break, while +9% on a 10%-board stock below its limit is not.
    dates = ["d0", "d1"]
    rows = [
        {"ts_code": "300001.SZ", "trade_date": "d0", "open": 10.0, "high": 10.2, "low": 9.9,
         "close": 10.0, "pre_close": 9.9, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "300001.SZ", "trade_date": "d1", "open": 11.0, "high": 12.0, "low": 10.9,
         "close": 11.5, "pre_close": 10.0, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "600000.SH", "trade_date": "d0", "open": 10.0, "high": 10.2, "low": 9.9,
         "close": 10.0, "pre_close": 9.9, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
        {"ts_code": "600000.SH", "trade_date": "d1", "open": 10.5, "high": 10.99, "low": 10.5,
         "close": 10.9, "pre_close": 10.0, "volume": 1e6, "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0},
    ]
    bars = pd.DataFrame(rows)
    codes = ["300001.SZ", "600000.SH"]
    tensor = compute_factor_tensor(
        bars, codes, dates, np.ones((len(codes), len(dates)), dtype=bool)
    )
    breaks = tensor[FEATURE_NAMES.index("LIMIT_BREAK")]
    # 300001 (+15%, touched its 20% limit) is a break; 600000 (+9%, below
    # its 10% limit) is not.
    assert breaks[0, 1] > 0
    assert breaks[1, 1] <= 0


def test_medium_term_reversal_and_ret120_values():
    n = 130
    dates = pd.bdate_range("2024-01-01", periods=n).strftime("%Y%m%d").tolist()
    closes = [10.0 + 0.05 * i for i in range(n)]
    rows = [
        {"ts_code": "000001.SZ", "trade_date": d, "open": c, "high": c, "low": c,
         "close": c, "pre_close": closes[i - 1] if i else c, "volume": 1e6,
         "amount": 1e7, "turnover_rate": 1.0, "adj_factor": 1.0}
        for i, (d, c) in enumerate(zip(dates, closes))
    ]
    ctx = _context_for(pd.DataFrame(rows), dates, ["000001.SZ"],
                       ["RET_120", "REVERSAL_60", "REVERSAL_120"])
    ret120 = _factor_fn("RET_120")(ctx)
    rev60 = _factor_fn("REVERSAL_60")(ctx)
    rev120 = _factor_fn("REVERSAL_120")(ctx)
    # RET_120 needs 120 prior closes: defined from day 120 onward.
    assert np.isnan(ret120.iloc[0, 119])
    assert ret120.iloc[0, 120] == pytest.approx(closes[120] / closes[0] - 1.0)
    assert rev60.iloc[0, 120] == pytest.approx(-(closes[120] / closes[60] - 1.0))
    assert rev120.iloc[0, 120] == pytest.approx(-(closes[120] / closes[0] - 1.0))
    # Trailing-only: perturbing the last close leaves earlier cells intact.
    assert np.isnan(rev60.iloc[0, 59])


# --- PIT universe mask in the factor layer ------------------------------------


def _universe_mask(codes: list[str], dates: list[str], join_day: int) -> np.ndarray:
    """All eligible except the future member (``300001.SZ``), which joins
    on ``join_day``."""
    mask = np.ones((len(codes), len(dates)), dtype=bool)
    mask[codes.index("300001.SZ"), :join_day] = False
    return mask


def _join_bars(
    codes: list[str],
    join_day: int,
    *,
    pre_join_amount: float = 1e7,
    join_day_amount: float | None = None,
    pre_join_close_jump: bool = False,
    n_dates: int = 40,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Deterministic bars where ``300001.SZ`` is the future member.

    Eligible stocks trade with distinct amounts ((i+1)*1e7); the future
    member carries ``pre_join_amount`` before its join day, ``join_day_amount``
    exactly on the join day (``None`` keeps the pre-join amount), and 1e7
    afterwards.  ``pre_join_close_jump`` plants a huge close on the day
    before the join.
    """

    dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y%m%d").tolist()
    rows = []
    for code in codes:
        closes = []
        for i in range(n_dates):
            if code == "300001.SZ":
                closes.append(1e5 if (pre_join_close_jump and i == join_day - 1) else 20.0 + 0.1 * i)
            else:
                closes.append(10.0 + 0.1 * i + (0.5 if code == "000001.SZ" else 0.0))
        for i, d in enumerate(dates):
            close = closes[i]
            pre = closes[i - 1] if i else close * 0.99
            if code == "300001.SZ":
                if i < join_day:
                    amount = pre_join_amount
                elif i == join_day and join_day_amount is not None:
                    amount = join_day_amount
                else:
                    amount = 1e7
            else:
                amount = (codes.index(code) + 1) * 1e7
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": d,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "pre_close": pre,
                    "volume": 1e6,
                    "amount": amount,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
            )
    return codes, dates, pd.DataFrame(rows)


def test_future_member_extreme_does_not_move_eligible_factors():
    # The spec contract: give the future member extreme pre-join values and
    # the eligible stocks' winsorized factor block (winsorize quantiles,
    # CAPM market, amount-share denominator, industry-relative demean) must
    # not change at all before the join day.
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    join_day = 20
    _, dates, bars_extreme = _join_bars(
        codes, join_day, pre_join_amount=1e12, pre_join_close_jump=True
    )
    _, _, bars_mild = _join_bars(codes, join_day)
    mask = _universe_mask(codes, dates, join_day)
    # One shared industry so the industry-relative factors are live: their
    # industry means must also exclude the future member before the join.
    industry = pd.DataFrame(np.nan, index=codes, columns=dates, dtype=object)
    industry.loc[:, :] = "801780"
    t_ext = compute_factor_tensor(
        bars_extreme, codes, dates, industry_frame=industry, universe_mask=mask
    )
    t_mild = compute_factor_tensor(
        bars_mild, codes, dates, industry_frame=industry, universe_mask=mask
    )
    # The eligible stocks' whole pre-join factor block is bit-identical.
    assert np.array_equal(t_ext[:, :2, :join_day], t_mild[:, :2, :join_day])
    # The future member's own pre-join history is preserved (not zeroed):
    # its values differ between the two scenarios.
    assert not np.array_equal(t_ext[:, 2, :join_day], t_mild[:, 2, :join_day])


def test_join_day_extreme_legitimately_affects_the_cross_section():
    # On the join day the future member is eligible: its extreme amount is
    # part of the reference set and legitimately moves the eligible stocks'
    # standardized values; before the join day nothing changes.
    codes = ["000001.SZ", "600000.SH", "300001.SZ", "000002.SZ", "600001.SH"]
    join_day = 20
    _, dates, bars_mild = _join_bars(codes, join_day, join_day_amount=5e7)
    _, _, bars_extreme = _join_bars(codes, join_day, join_day_amount=1e13)
    mask = _universe_mask(codes, dates, join_day)
    t_mild = compute_factor_tensor(bars_mild, codes, dates, universe_mask=mask)
    t_ext = compute_factor_tensor(bars_extreme, codes, dates, universe_mask=mask)
    assert np.array_equal(t_mild[:, :, :join_day], t_ext[:, :, :join_day])
    amt = FEATURE_NAMES.index("AMOUNT_SHARE")
    f_row = codes.index("300001.SZ")
    eligible_rows = [i for i in range(len(codes)) if i != f_row]
    # The join-day extreme shifts the eligible reference statistics, so the
    # eligible stocks' standardized shares differ between the scenarios...
    assert not np.allclose(
        t_mild[amt, eligible_rows, join_day],
        t_ext[amt, eligible_rows, join_day],
    )
    # ...and the future member itself is now the dominant share.
    assert t_ext[amt, f_row, join_day] > max(t_ext[amt, eligible_rows, join_day])


def test_join_day_momentum_uses_pre_join_history():
    # The mask never zeroes a stock's own bar history: the future member's
    # join-day momentum is the standardized form of its true close ratio,
    # identical to the unmasked computation (the join-day reference set is
    # the same either way).
    codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    join_day = 25
    _, dates, bars = _join_bars(codes, join_day)
    mask = _universe_mask(codes, dates, join_day)
    engine = AshareFactorEngine()
    masked = engine.compute_factor_tensor(bars, codes, dates, universe_mask=mask)
    unmasked = engine.compute_factor_tensor(
        bars, codes, dates, np.ones((len(codes), len(dates)), dtype=bool)
    )
    f_row = codes.index("300001.SZ")
    for name in ("RET_5", "MOMENTUM_20"):
        idx = FEATURE_NAMES.index(name)
        assert np.allclose(
            masked[idx, f_row, join_day], unmasked[idx, f_row, join_day]
        )
    ctx = _context_for(bars, dates, codes, ["RET_5"])
    raw_ret5 = _factor_fn("RET_5")(ctx)
    expected = raw_ret5.loc["300001.SZ", dates[join_day]]
    expected_ratio = bars.loc[
        (bars["ts_code"] == "300001.SZ") & (bars["trade_date"] == dates[join_day]),
        "close",
    ].iloc[0] / bars.loc[
        (bars["ts_code"] == "300001.SZ") & (bars["trade_date"] == dates[join_day - 5]),
        "close",
    ].iloc[0] - 1.0
    assert expected == pytest.approx(expected_ratio)
    assert masked[FEATURE_NAMES.index("RET_5"), f_row, join_day] != 0.0


def test_engine_rejects_universe_mask_shape_mismatch(bars_data):
    dates, ts_codes, bars = bars_data
    bad = np.ones((len(ts_codes), len(dates) + 1), dtype=bool)
    with pytest.raises(ValueError, match="universe_mask shape"):
        compute_factor_tensor(bars, ts_codes, dates, universe_mask=bad)


def test_amount_share_denominator_excludes_ineligible():
    amount = pd.DataFrame(
        {"d1": [30.0, 70.0, 1e9], "d2": [50.0, 50.0, 1e9]},
        index=["A", "B", "FUTURE"],
    )
    eligible = pd.DataFrame(
        [[True, True], [True, True], [False, False]],
        index=["A", "B", "FUTURE"],
        columns=["d1", "d2"],
    )
    share = _factor_amount_share(amount, eligible)
    # The denominator counts only eligible amounts: the extreme ineligible
    # amount cannot dilute the eligible shares.
    assert share.loc["A", "d1"] == pytest.approx(0.3)
    assert share.loc["B", "d1"] == pytest.approx(0.7)
    assert share.loc["A", "d2"] == pytest.approx(0.5)
    assert share.loc["B", "d2"] == pytest.approx(0.5)


def test_rolling_capm_market_excludes_ineligible_stocks():
    # Stock A = 2*mkt, stock B = 0.5*mkt, and a future member with extreme
    # returns.  With the future member masked out the equal-weight market
    # uses only A/B, so the recovered betas stay 1.6 and 0.4 exactly as in
    # the unmasked two-stock reference test.
    n = 30
    m = np.sin(np.linspace(0.1, 2.0, n)) * 0.01
    ret_a = 2.0 * m
    ret_b = 0.5 * m
    ret_f = np.full(n, 5.0)  # extreme future-member returns
    close = pd.DataFrame(
        {
            f"d{i}": [
                10.0 * np.prod(1.0 + ret_a[: i + 1]),
                10.0 * np.prod(1.0 + ret_b[: i + 1]),
                10.0 * np.prod(1.0 + ret_f[: i + 1]),
            ]
            for i in range(n)
        },
        index=["A", "B", "FUTURE"],
    )
    eligible = np.ones((3, n), dtype=bool)
    eligible[2] = False
    beta, ivol, rsq = _rolling_capm(close, window=60, min_periods=2, eligible=eligible)
    assert beta.loc["A"].iloc[-1] == pytest.approx(1.6, abs=1e-4)
    assert beta.loc["B"].iloc[-1] == pytest.approx(0.4, abs=1e-4)
    assert ivol.loc["A"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert rsq.loc["A"].iloc[-1] == pytest.approx(1.0, abs=1e-4)
    # With an all-eligible mask the extreme future member distorts the
    # market factor and the recovered betas shift.
    beta_all, _, _ = _rolling_capm(
        close, np.ones(close.shape, dtype=bool), window=60, min_periods=2
    )
    assert abs(beta_all.loc["A"].iloc[-1] - 1.6) > 1e-3


def test_industry_demean_reference_is_eligible_only():
    frame = pd.DataFrame(
        {"d1": [1.0, 3.0, 100.0], "d2": [2.0, 4.0, 200.0]},
        index=["A", "B", "FUTURE"],
    )
    ctx = _empty_ctx(["A", "B", "FUTURE"], ["d1", "d2"])
    ctx.industry = pd.DataFrame(
        {"d1": ["X", "X", "X"], "d2": ["X", "X", "X"]},
        index=["A", "B", "FUTURE"],
        dtype=object,
    )
    ctx.eligible = pd.DataFrame(
        [[True, True], [True, True], [False, False]],
        index=["A", "B", "FUTURE"],
        columns=["d1", "d2"],
    )
    out = _industry_demean(ctx, frame)
    # The industry mean counts only eligible members: mean(A, B) = 2.0 on
    # d1 and 3.0 on d2, so the extreme ineligible member cannot shift it.
    assert out.loc["A", "d1"] == pytest.approx(-1.0)
    assert out.loc["B", "d1"] == pytest.approx(1.0)
    assert out.loc["A", "d2"] == pytest.approx(-1.0)
    assert out.loc["B", "d2"] == pytest.approx(1.0)
