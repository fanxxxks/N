"""A-share cross-sectional factor computation.

Every factor is registered in :data:`FACTOR_REGISTRY` with declarative
metadata (family, required daily-bar columns, warmup, description) and a
pure function over the shared :class:`FactorContext`.  The engine pivots
exactly the columns the requested factors need, computes each registered
factor once, then cross-sectionally standardizes the stack.  Features with
no local data source are declared separately (fundamental point-in-time
snapshots, external capital-flow sources) and stay neutral instead of being
fabricated.

Adding a factor therefore means adding one registry entry plus tests; the
metadata drives diagnostics and family-level ablation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
NEUTRAL_FEATURE_NAMES = {
    "NORTHBOUND_CHG",
    "MARGIN_BALANCE_CHG",
    "INDUSTRY_MOMENTUM",
}

# Factor family labels: the metadata fields used by diagnostics and
# family-level ablation experiments.
FAMILIES = (
    "momentum",
    "volatility",
    "distribution",
    "volume",
    "event",
    "fundamental",
    "external",
)

# Daily-bar columns a factor may draw from.  Registry metadata is validated
# against this set in tests so a typo can never hide a missing pivot.
BAR_COLUMNS = (
    "close",
    "open",
    "high",
    "low",
    "pre_close",
    "volume",
    "amount",
    "turnover_rate",
)

_CHINEXT_PREFIXES = {"300", "301", "688", "689"}


@dataclass(frozen=True)
class FactorSpec:
    """Declarative metadata for one registered factor."""

    name: str
    family: str
    required_columns: tuple[str, ...] = ()
    warmup: int = 1
    description: str = ""


@dataclass
class FactorContext:
    """Pivoted ``[stock x date]`` inputs shared by all factor functions.

    Only the columns required by the requested factors are pivoted; the
    others are empty frames with the correct index/columns so a factor that
    misdeclares its requirements fails loudly instead of reading stale data.
    """

    ts_codes: list[str]
    dates: list[str]
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    pre_close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    turnover: pd.DataFrame


def _returns(close: pd.DataFrame) -> pd.DataFrame:
    # NaN for the first day and for missing days: downstream
    # cross-sectional standardization maps missing to the neutral 0.
    return close.pct_change(fill_method=None, axis=1).replace([np.inf, -np.inf], np.nan)


def _shift_ratio(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    shifted = close.shift(periods, axis=1)
    return (close / shifted - 1.0).replace([np.inf, -np.inf], np.nan)


def _rolling_std(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.T.rolling(window, min_periods=1).std().T


def _rolling_mean(values: pd.DataFrame, window: int) -> pd.DataFrame:
    return values.T.rolling(window, min_periods=1).mean().T


def _rolling_skew(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.T.rolling(window, min_periods=3).skew().T


def _rolling_kurt(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.T.rolling(window, min_periods=4).kurt().T


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


def _limit_rate_per_stock(ts_codes: list[str]) -> np.ndarray:
    """Board daily limit rate per stock (ST-adjusted rates need stock names,
    which the factor engine does not have; ST stocks are excluded upstream)."""
    rates = []
    for code in ts_codes:
        symbol = code.split(".", 1)[0]
        rates.append(0.20 if symbol[:3] in _CHINEXT_PREFIXES else 0.10)
    return np.asarray(rates, dtype=float)


def _make_registry() -> dict[str, tuple[FactorSpec, Callable[[FactorContext], pd.DataFrame]]]:
    """Build the ordered factor registry (insertion order = computation order)."""

    registry: dict[str, tuple[FactorSpec, Callable[[FactorContext], pd.DataFrame]]] = {}

    def add(
        name: str,
        family: str,
        required: tuple[str, ...],
        warmup: int,
        description: str,
        fn: Callable[[FactorContext], pd.DataFrame],
    ) -> None:
        registry[name] = (
            FactorSpec(name, family, tuple(required), warmup, description),
            fn,
        )

    add("RET_1", "momentum", ("close",), 2, "1-day close-to-close return", lambda ctx: _returns(ctx.close))
    add("RET_5", "momentum", ("close",), 6, "5-day close-to-close return", lambda ctx: _shift_ratio(ctx.close, 5))
    add("RET_10", "momentum", ("close",), 11, "10-day close-to-close return", lambda ctx: _shift_ratio(ctx.close, 10))
    add("RET_20", "momentum", ("close",), 21, "20-day close-to-close return", lambda ctx: _shift_ratio(ctx.close, 20))
    add("VOL_20", "volatility", ("close",), 20, "20-day return volatility", lambda ctx: _rolling_std(_returns(ctx.close), 20))
    add("VOL_60", "volatility", ("close",), 60, "60-day return volatility", lambda ctx: _rolling_std(_returns(ctx.close), 60))
    add(
        "TURNOVER",
        "volume",
        ("turnover_rate",),
        1,
        "Daily turnover rate (volume / float shares); missing stays neutral",
        lambda ctx: ctx.turnover.replace([np.inf, -np.inf], np.nan),
    )
    add(
        "TURNOVER_CHG",
        "volume",
        ("turnover_rate",),
        2,
        "1-day change in turnover rate",
        lambda ctx: ctx.turnover.pct_change(fill_method=None, axis=1).replace([np.inf, -np.inf], np.nan),
    )
    add(
        "VOLUME_RATIO",
        "volume",
        ("volume",),
        20,
        "Volume relative to its 20-day mean minus 1",
        lambda ctx: ctx.volume / (_rolling_mean(ctx.volume, 20) + 1e-9) - 1.0,
    )
    add(
        "VOLUME_IMPACT",
        "volume",
        ("volume",),
        20,
        "log-volume deviation from its 20-day mean",
        lambda ctx: np.log1p(ctx.volume.clip(lower=0)) - _rolling_mean(np.log1p(ctx.volume.clip(lower=0)), 20),
    )
    add(
        "AMPLITUDE",
        "volume",
        ("high", "low", "close"),
        2,
        "Intraday amplitude (high-low)/prev close",
        lambda ctx: (ctx.high - ctx.low) / (ctx.close.shift(1) + 1e-9),
    )
    add(
        "CLOSE_POSITION",
        "volume",
        ("high", "low", "close"),
        1,
        "Close position within the intraday range",
        lambda ctx: (ctx.close - ctx.low) / (ctx.high - ctx.low + 1e-9),
    )
    add("MOMENTUM_20", "momentum", ("close",), 21, "20-day momentum (same as RET_20)", lambda ctx: _shift_ratio(ctx.close, 20))
    add("MOMENTUM_60", "momentum", ("close",), 61, "60-day momentum", lambda ctx: _shift_ratio(ctx.close, 60))
    add("REVERSAL_5", "momentum", ("close",), 6, "Negative 5-day return (short-term reversal)", lambda ctx: -_shift_ratio(ctx.close, 5))
    add("SKEW_20", "distribution", ("close",), 20, "20-day return skewness", lambda ctx: _rolling_skew(_returns(ctx.close), 20))
    add("KURT_20", "distribution", ("close",), 20, "20-day return excess kurtosis", lambda ctx: _rolling_kurt(_returns(ctx.close), 20))
    add(
        "LIMIT_UP_EVENT",
        "event",
        ("close", "high", "low", "pre_close"),
        2,
        "One-word limit-up close (1.0 on the event day)",
        lambda ctx: _limit_events(ctx.close, ctx.high, ctx.low, ctx.pre_close, ctx.ts_codes, "up"),
    )
    add(
        "LIMIT_DOWN_EVENT",
        "event",
        ("close", "high", "low", "pre_close"),
        2,
        "One-word limit-down close (1.0 on the event day)",
        lambda ctx: _limit_events(ctx.close, ctx.high, ctx.low, ctx.pre_close, ctx.ts_codes, "down"),
    )
    return registry


FACTOR_REGISTRY = _make_registry()


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

    def _build_context(
        self,
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        close: pd.DataFrame,
    ) -> FactorContext:
        """Pivot exactly the columns the requested registered factors need."""

        needed: set[str] = set()
        for name in self.feature_names:
            entry = FACTOR_REGISTRY.get(name)
            if entry is not None:
                needed.update(entry[0].required_columns)
        empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        frames: dict[str, pd.DataFrame] = {col: empty.copy() for col in BAR_COLUMNS}
        for col in needed:
            frames[col] = self._pivot(bars, ts_codes, dates, col)
        return FactorContext(
            ts_codes=list(ts_codes),
            dates=list(dates),
            close=frames["close"],
            open=frames["open"],
            high=frames["high"],
            low=frames["low"],
            pre_close=frames["pre_close"],
            volume=frames["volume"],
            amount=frames["amount"],
            turnover=frames["turnover_rate"],
        )

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
        context = self._build_context(bars, ts_codes, dates, close)

        empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        raw_factors: dict[str, pd.DataFrame] = {}
        for name in self.feature_names:
            entry = FACTOR_REGISTRY.get(name)
            raw_factors[name] = entry[1](context) if entry is not None else empty.copy()

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
