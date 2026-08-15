"""A-share cross-sectional factor computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_data.processor import normalize_daily_bars, pivot_wide, winsorize_cross_section

from .vocab import FEATURE_NAMES

# Fundamental feature names that can be fed from point-in-time snapshots.
FUNDAMENTAL_NAMES = (
    "PE_TTM",
    "PB",
    "PS_TTM",
    "ROE",
    "ROA",
    "GROSS_MARGIN",
    "NET_MARGIN",
    "REVENUE_YOY",
    "PROFIT_YOY",
    "DEBT_RATIO",
    "MARKET_CAP",
    "DIVIDEND_YIELD",
)

# Features declared in the vocabulary but not computable from the local data
# (they need northbound / margin / industry-history sources).  They stay
# neutral (missing) instead of being fabricated.
_NEUTRAL_FEATURES = {
    "NORTHBOUND_CHG",
    "MARGIN_BALANCE_CHG",
    "INDUSTRY_MOMENTUM",
}

_CHINEXT_PREFIXES = {"300", "301", "688", "689"}


def _limit_rate_per_stock(ts_codes: list[str]) -> np.ndarray:
    """Board daily limit rate per stock (ST-adjusted rates need stock names,
    which the factor engine does not have; ST stocks are excluded upstream)."""
    rates = []
    for code in ts_codes:
        symbol = code.split(".", 1)[0]
        rates.append(0.20 if symbol[:3] in _CHINEXT_PREFIXES else 0.10)
    return np.asarray(rates, dtype=float)


def _limit_events(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    pre_close: pd.DataFrame,
    ts_codes: list[str],
    direction: str,
) -> pd.DataFrame:
    """One-word limit move events (1.0 where the close is locked at the limit)."""
    rates = _limit_rate_per_stock(ts_codes)
    one_word = np.isclose(close.to_numpy(), high.to_numpy()) & np.isclose(
        close.to_numpy(), low.to_numpy()
    )
    with np.errstate(all="ignore"):
        change = close.to_numpy() / pre_close.to_numpy() - 1.0
    if direction == "up":
        event = one_word & (change >= rates[:, None] - 0.005)
    else:
        event = one_word & (change <= -rates[:, None] + 0.005)
    event = np.nan_to_num(event, nan=False).astype(float)
    return pd.DataFrame(event, index=close.index, columns=close.columns)


class AshareFactorEngine:
    """Compute standardized price-volume and fundamental factor tensors."""

    def __init__(self, feature_names: list[str] | tuple[str, ...] | None = None):
        self.feature_names = list(feature_names or FEATURE_NAMES)

    @staticmethod
    def _pivot(
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        value_col: str,
    ) -> pd.DataFrame:
        return pivot_wide(bars, ts_codes, dates, value_col)

    @staticmethod
    def _returns(close: pd.DataFrame) -> pd.DataFrame:
        # NaN for the first day and for missing days: downstream
        # cross-sectional standardization maps missing to the neutral 0.
        return close.pct_change(fill_method=None, axis=1).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _rolling_std(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        return returns.T.rolling(window, min_periods=1).std().T

    @staticmethod
    def _rolling_mean(values: pd.DataFrame, window: int) -> pd.DataFrame:
        return values.T.rolling(window, min_periods=1).mean().T

    @staticmethod
    def _rolling_skew(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        return returns.T.rolling(window, min_periods=3).skew().T

    @staticmethod
    def _rolling_kurt(returns: pd.DataFrame, window: int) -> pd.DataFrame:
        return returns.T.rolling(window, min_periods=4).kurt().T

    @staticmethod
    def _shift_ratio(close: pd.DataFrame, periods: int) -> pd.DataFrame:
        shifted = close.shift(periods, axis=1)
        return (close / shifted - 1.0).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _turnover(bars: pd.DataFrame, ts_codes: list[str], dates: list[str]) -> pd.DataFrame:
        # Turnover needs the float share count, which the local data does not
        # carry; a missing column stays NaN (neutral) rather than a fake
        # constant ~1.0 proxy.
        turnover = pivot_wide(bars, ts_codes, dates, "turnover_rate")
        return turnover.replace([np.inf, -np.inf], np.nan)

    def compute_factor_tensor(
        self,
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        fundamentals: dict[str, dict[str, float | None]] | None = None,
    ) -> np.ndarray:
        """Return a ``[feature, stock, date]`` float32 tensor.

        ``fundamentals`` maps ts_code -> field snapshot.  Snapshots are
        point-in-time only for the final date, so they are injected into the
        last date column exclusively; no future information is fabricated.
        """

        bars = normalize_daily_bars(bars)
        dates = sorted(dates)
        close = self._pivot(bars, ts_codes, dates, "close")
        open_ = self._pivot(bars, ts_codes, dates, "open")
        high = self._pivot(bars, ts_codes, dates, "high")
        low = self._pivot(bars, ts_codes, dates, "low")
        pre_close = self._pivot(bars, ts_codes, dates, "pre_close")
        volume = self._pivot(bars, ts_codes, dates, "volume")
        amount = self._pivot(bars, ts_codes, dates, "amount")
        turnover = self._turnover(bars, ts_codes, dates)

        returns = self._returns(close)
        log_volume = np.log1p(volume.clip(lower=0))

        raw_factors: dict[str, pd.DataFrame] = {
            "RET_1": returns,
            "RET_5": self._shift_ratio(close, 5),
            "RET_10": self._shift_ratio(close, 10),
            "RET_20": self._shift_ratio(close, 20),
            "VOL_20": self._rolling_std(returns, 20),
            "VOL_60": self._rolling_std(returns, 60),
            "TURNOVER": turnover,
            "TURNOVER_CHG": turnover.pct_change(fill_method=None, axis=1).replace(
                [np.inf, -np.inf], np.nan
            ),
            "VOLUME_RATIO": volume / (self._rolling_mean(volume, 20) + 1e-9) - 1.0,
            "VOLUME_IMPACT": log_volume - self._rolling_mean(log_volume, 20),
            "AMPLITUDE": (high - low) / (close.shift(1) + 1e-9),
            "CLOSE_POSITION": (close - low) / (high - low + 1e-9),
            "MOMENTUM_20": self._shift_ratio(close, 20),
            "MOMENTUM_60": self._shift_ratio(close, 60),
            "REVERSAL_5": -self._shift_ratio(close, 5),
            "SKEW_20": self._rolling_skew(returns, 20),
            "KURT_20": self._rolling_kurt(returns, 20),
            "LIMIT_UP_EVENT": _limit_events(close, high, low, pre_close, ts_codes, "up"),
            "LIMIT_DOWN_EVENT": _limit_events(close, high, low, pre_close, ts_codes, "down"),
        }

        # Features without a local data source stay missing/neutral.
        empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for name in self.feature_names:
            if name not in raw_factors:
                raw_factors[name] = empty.copy()

        # Point-in-time fundamentals: apply the snapshot to the final date
        # column only.  All other dates stay missing (neutral).
        fundamentals = fundamentals or {}
        if fundamentals and dates:
            last_date = dates[-1]
            for name in FUNDAMENTAL_NAMES:
                if name not in self.feature_names:
                    continue
                column = pd.Series(np.nan, index=close.index, dtype=float)
                for ts_code, snapshot in fundamentals.items():
                    if ts_code in close.index:
                        value = snapshot.get(name) if isinstance(snapshot, dict) else None
                        if value is not None and np.isfinite(float(value)):
                            column[ts_code] = float(value)
                frame = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
                frame[last_date] = column
                raw_factors[name] = frame

        standardized = []
        for name in self.feature_names:
            frame = raw_factors.get(name, empty)
            if isinstance(frame, pd.Series):
                frame = frame.to_frame().T
            frame = frame.reindex(index=ts_codes, columns=dates)
            frame = winsorize_cross_section(frame)
            # After cross-sectional standardization 0 is the neutral value.
            standardized.append(frame.fillna(0.0).values.astype(np.float32))

        return np.stack(standardized, axis=0)


def compute_factor_tensor(
    bars: pd.DataFrame,
    ts_codes: list[str],
    dates: list[str],
    fundamentals: dict[str, dict[str, float | None]] | None = None,
) -> np.ndarray:
    return AshareFactorEngine().compute_factor_tensor(bars, ts_codes, dates, fundamentals)
