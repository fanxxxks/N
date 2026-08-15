"""A-share cross-sectional factor computation.

Every factor is registered in :data:`FACTOR_REGISTRY` with declarative
metadata (family, required daily-bar columns, warmup, description) and a
pure function over the shared :class:`FactorContext`.  The engine pivots
exactly the columns the requested factors need, computes each registered
factor once, then cross-sectionally standardizes the stack.  Features with
no local data source are declared separately (point-in-time fundamentals
from :mod:`ashare_data.fundamentals`, external capital-flow sources) and
stay neutral instead of being fabricated.

Adding a factor therefore means adding one registry entry plus tests; the
metadata drives diagnostics and family-level ablation experiments.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ashare_data.processor import normalize_daily_bars, pivot_wide, winsorize_cross_section

from .vocab import FEATURE_NAMES

# Features with a local data source outside the daily bars: the PIT
# fundamentals and the capital-flow factors are injected as pre-built
# frames by the loader.  Only NORTHBOUND_CHG has no free historical feed
# (daily disclosure stopped in Aug 2024) and stays neutral.
NEUTRAL_FEATURE_NAMES = {
    "NORTHBOUND_CHG",
}

# Factor family labels: the metadata fields used by diagnostics and
# family-level ablation experiments.
FAMILIES = (
    "momentum",
    "volatility",
    "distribution",
    "volume",
    "event",
    "intraday",
    "liquidity",
    "lottery",
    "risk",
    "technical",
    "microstructure",
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
    ``_cache`` lets related factors (e.g. the CAPM triple) share one
    computation without duplicating work.
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
    _cache: dict | None = None

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}


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


def _rolling_max(values: pd.DataFrame, window: int) -> pd.DataFrame:
    return values.T.rolling(window, min_periods=1).max().T


def _safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return (numerator / denominator - 1.0).replace([np.inf, -np.inf], np.nan)


def _rolling_capm(
    close: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling CAPM regression of each stock on the equal-weight market.

    Returns ``(beta, ivol, rsq)`` ``[stock x date]`` frames over trailing
    windows (expanding before ``window``, at least ``min_periods`` valid
    days, and a non-degenerate market/stock variance).  Prefix-sum
    vectorization keeps it cheap on the full cross-section.  The market is
    the cross-sectional mean return per date, so no external index data is
    needed.
    """

    r = _returns(close)
    arr = r.to_numpy(dtype=float)  # [S, T], NaN on missing days
    s, t = arr.shape
    with warnings.catch_warnings():
        # All-NaN columns (e.g. the first date) legitimately produce an
        # empty-slice warning; they map to a zero market return below.
        warnings.simplefilter("ignore", RuntimeWarning)
        mkt = np.nanmean(arr, axis=0)  # [T] equal-weight market return
    mkt = np.nan_to_num(mkt, nan=0.0)
    valid = ~np.isnan(arr)
    r0 = np.nan_to_num(arr, nan=0.0)
    m0 = np.nan_to_num(mkt, nan=0.0)
    rm = r0 * m0
    r2 = r0 * r0
    m2 = m0 * m0

    def prefix(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([np.zeros((s, 1)), np.cumsum(mat, axis=1)], axis=1)

    pv = prefix(valid.astype(float))
    pr = prefix(r0)
    prm = prefix(rm)
    pr2 = prefix(r2)
    pm = np.concatenate([np.zeros(1), np.cumsum(m0)])
    pm2 = np.concatenate([np.zeros(1), np.cumsum(m2)])

    idx = np.arange(t)
    start = np.maximum(idx - window + 1, 0)
    end = idx + 1
    n = pv[:, end] - pv[:, start]
    sr = pr[:, end] - pr[:, start]
    srm = prm[:, end] - prm[:, start]
    sr2 = pr2[:, end] - pr2[:, start]
    sm = pm[end] - pm[start]
    sm2 = pm2[end] - pm2[start]

    with np.errstate(all="ignore"):
        cov = srm - sr * sm / n
        var_m = sm2 - sm * sm / n
        var_r = sr2 - sr * sr / n
        beta = cov / var_m
        resid_var = var_r - beta * cov
        rsq = cov * cov / (var_r * var_m)
    ivol = np.sqrt(np.clip(resid_var, 0.0, None))

    good = (n >= min_periods) & (var_m > 1e-12) & (var_r > 1e-12)
    beta[~good] = np.nan
    ivol[~good] = np.nan
    rsq[~good] = np.nan
    index, columns = close.index, close.columns
    return (
        pd.DataFrame(beta, index=index, columns=columns),
        pd.DataFrame(ivol, index=index, columns=columns),
        pd.DataFrame(rsq, index=index, columns=columns),
    )


def _factor_amount_share(amount: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional share of the day's total turnover amount.

    Days where no stock has a valid amount keep NaN (neutral).
    """

    denom = amount.fillna(0.0).sum(axis=0).replace(0.0, np.nan)
    return (amount / denom).replace([np.inf, -np.inf], np.nan)


def _factor_capm(ctx: FactorContext, which: int) -> pd.DataFrame:
    """Shared CAPM triple; ``which`` picks beta (0), ivol (1) or rsq (2)."""
    triple = ctx._cache.get("capm60")
    if triple is None:
        triple = _rolling_capm(ctx.close, window=60, min_periods=20)
        ctx._cache["capm60"] = triple
    return triple[which]


def _ewm_mean(
    frame: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Exponential moving average along the time axis (pandas 3.0 dropped
    the ``axis`` kwarg from ``ewm``, so the frame is transposed instead)."""

    return frame.T.ewm(**kwargs).mean().T


def _factor_rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder-smoothed RSI in [0, 100]; no-movement days stay NaN (neutral)."""

    delta = close.diff(axis=1).replace([np.inf, -np.inf], np.nan)
    up = delta.clip(lower=0).fillna(0.0)
    down = (-delta.clip(upper=0)).fillna(0.0)
    avg_up = _ewm_mean(up, alpha=1.0 / window, adjust=False, min_periods=window)
    avg_down = _ewm_mean(down, alpha=1.0 / window, adjust=False, min_periods=window)
    with np.errstate(all="ignore"):
        rsi = 100.0 * avg_up / (avg_up + avg_down)
    return rsi.replace([np.inf, -np.inf], np.nan)


def _factor_atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    pre_close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Wilder-smoothed average true range; missing days contribute 0 so the
    average decays over suspensions instead of leaking future information."""

    tr = np.maximum.reduce(
        [
            (high - low).to_numpy(dtype=float),
            np.abs(high.to_numpy(dtype=float) - pre_close.to_numpy(dtype=float)),
            np.abs(low.to_numpy(dtype=float) - pre_close.to_numpy(dtype=float)),
        ]
    )
    tr = pd.DataFrame(tr, index=high.index, columns=high.columns)
    tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _ewm_mean(tr, alpha=1.0 / window, adjust=False)


def _factor_macd(
    close: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MACD DIF (EMA fast - EMA slow) and DEA (EMA of DIF).

    Missing days are forward-filled for the EMA recursion (causal); rows
    without any history stay NaN.
    """

    filled = close.ffill(axis=1)
    ema_fast = _ewm_mean(filled, span=fast, adjust=False)
    ema_slow = _ewm_mean(filled, span=slow, adjust=False)
    dif = ema_fast - ema_slow
    dea = _ewm_mean(dif, span=signal, adjust=False)
    return dif, dea


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

    # Intraday decomposition.  GAP (open/pre_close - 1) is the same quantity
    # as OVERNIGHT_RET, so it is not a separate feature.
    add(
        "OVERNIGHT_RET",
        "intraday",
        ("open", "pre_close"),
        2,
        "Overnight return open/prev close - 1",
        lambda ctx: _safe_ratio(ctx.open, ctx.pre_close),
    )
    add(
        "INTRADAY_RET",
        "intraday",
        ("open", "close"),
        1,
        "Intraday return close/open - 1",
        lambda ctx: _safe_ratio(ctx.close, ctx.open),
    )

    # Liquidity.  Free-float turnover is already covered by TURNOVER, so
    # only the Amihud measure and the amount share are added.
    add(
        "ILLIQ_20",
        "liquidity",
        ("close", "amount"),
        20,
        "Amihud illiquidity: 20-day mean of |return|/amount",
        lambda ctx: _rolling_mean(
            (_returns(ctx.close).abs() / ctx.amount).replace([np.inf, -np.inf], np.nan), 20
        ),
    )
    add(
        "AMOUNT_SHARE",
        "liquidity",
        ("amount",),
        1,
        "Share of the day's total cross-sectional turnover amount",
        lambda ctx: _factor_amount_share(ctx.amount),
    )

    # Lottery / anchoring.
    add(
        "MAX_20",
        "lottery",
        ("close",),
        20,
        "Maximum daily return over the trailing 20 days",
        lambda ctx: _rolling_max(_returns(ctx.close), 20),
    )
    add(
        "HIGH_52W",
        "momentum",
        ("close",),
        250,
        "Distance to the 52-week (250-day) high: close/max250 - 1",
        lambda ctx: _safe_ratio(ctx.close, _rolling_max(ctx.close, 250)),
    )

    # Market-model risk family (one shared rolling CAPM computation).
    add("BETA_60", "risk", ("close",), 20, "Rolling CAPM beta vs equal-weight market (60d)", lambda ctx: _factor_capm(ctx, 0))
    add("IVOL_60", "risk", ("close",), 20, "Idiosyncratic volatility from rolling CAPM (60d)", lambda ctx: _factor_capm(ctx, 1))
    add("RSQ_60", "risk", ("close",), 20, "R-squared from rolling CAPM (60d)", lambda ctx: _factor_capm(ctx, 2))

    # Technical.  BIAS is the moving-average distance; MACD keeps DIF/DEA
    # (HIST is expressible as DIF SUB DEA).
    add(
        "BIAS_20",
        "technical",
        ("close",),
        20,
        "20-day moving-average distance close/MA20 - 1",
        lambda ctx: _safe_ratio(ctx.close, _rolling_mean(ctx.close, 20)),
    )
    add("RSI_14", "technical", ("close",), 14, "Wilder RSI over 14 days (0-100)", lambda ctx: _factor_rsi(ctx.close, 14))
    add("ATR_14", "technical", ("high", "low", "pre_close"), 14, "Average true range, Wilder-smoothed 14 days", lambda ctx: _factor_atr(ctx.high, ctx.low, ctx.pre_close, 14))
    add(
        "MACD_DIF",
        "technical",
        ("close",),
        26,
        "MACD DIF line (EMA12 - EMA26)",
        lambda ctx: _factor_macd(ctx.close)[0],
    )
    add(
        "MACD_DEA",
        "technical",
        ("close",),
        26,
        "MACD DEA signal line (EMA9 of DIF)",
        lambda ctx: _factor_macd(ctx.close)[1],
    )

    # Microstructure.
    add(
        "SUSPEND_DAYS_60",
        "microstructure",
        ("close",),
        60,
        "Number of missing (suspended) days in the trailing 60",
        lambda ctx: ctx.close.isna().astype(float).T.rolling(60, min_periods=1).sum().T,
    )
    add(
        "LIST_AGE",
        "microstructure",
        ("close",),
        1,
        "Trading days since first available bar (listing age proxy)",
        lambda ctx: (~ctx.close.isna()).cumsum(axis=1).astype(float),
    )

    # Size: float market-cap proxy reconstructed from daily bars
    # (amount / turnover_rate); turnover_rate is the float-share turnover,
    # so the ratio is the float market cap in yuan.  No share-count
    # history is needed and the value is available for every trading day.
    add(
        "MARKET_CAP",
        "fundamental",
        ("amount", "turnover_rate"),
        1,
        "Float market cap proxy = amount / turnover_rate (yuan)",
        lambda ctx: (ctx.amount / ctx.turnover).replace([np.inf, -np.inf], np.nan),
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
        pit_fundamentals: dict[str, pd.DataFrame] | None = None,
        extra_frames: dict[str, pd.DataFrame] | None = None,
    ) -> np.ndarray:
        """Return a ``[feature, stock, date]`` float32 tensor.

        ``pit_fundamentals`` maps a PIT feature name to its
        disclosure-aligned ``[stock x date]`` frame (built by
        :func:`ashare_data.fundamentals.build_pit_frames`); ``extra_frames``
        carries the other externally fed features (margin/industry, built by
        :func:`ashare_data.capital_flow.build_capital_frames`).  Frames are
        injected verbatim; the no-lookahead guarantees live in the
        builders.  Missing names/frames stay neutral (0) after
        standardization.
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

        # Point-in-time fundamentals and capital-flow factors: inject the
        # pre-built frames; everything else stays neutral.
        injected: dict[str, pd.DataFrame] = {}
        injected.update(pit_fundamentals or {})
        injected.update(extra_frames or {})
        for name, frame in injected.items():
            if name not in self.feature_names:
                continue
            if frame is not None and not frame.empty:
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
    pit_fundamentals: dict[str, pd.DataFrame] | None = None,
    extra_frames: dict[str, pd.DataFrame] | None = None,
) -> np.ndarray:
    return AshareFactorEngine().compute_factor_tensor(
        bars, ts_codes, dates, pit_fundamentals, extra_frames
    )
