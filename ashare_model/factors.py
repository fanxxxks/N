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

from ashare_data.processor import (
    limit_rate,
    normalize_daily_bars,
    pivot_wide,
    winsorize_cross_section,
)

from .vocab import FEATURE_NAMES

# Features with a local data source outside the daily bars: the PIT
# fundamentals and the capital-flow factors are injected as pre-built
# frames by the loader.  Only NORTHBOUND_CHG has no free historical feed
# (daily disclosure stopped in Aug 2024) and stays neutral.
NEUTRAL_FEATURE_NAMES = {
    "NORTHBOUND_CHG",
}

# P9 (docs/p9_factor_family_contract.md §5.3): version of the factor
# computation semantics.  v1 = the sparse-safe standardization for the
# event family (P9 §5.3, t2 finding F1) plus the P9 feature additions.
# Recorded in run artifacts via the runspec version registry.
FACTOR_COMPUTE_VERSION = 1

# Sparse event features standardize on the sparse-safe path: the 1%/99%
# winsorize quantiles flatten any day whose event incidence is below the
# quantile into an all-zero cross-section (t2 finding F1 — LIMIT_* were
# non-degenerate on as few as 8 of 2826 sessions), so these features skip
# the winsorize and use a plain mean/std standardization with a floored
# standard deviation instead.  Days without events legitimately keep a
# degenerate (all-neutral) cross-section; event days keep their signal.
SPARSE_EVENT_FEATURES = {
    "LIMIT_UP_EVENT",
    "LIMIT_DOWN_EVENT",
    "LIMIT_STREAK",
    "LIMIT_UP_CNT_20",
    "LIMIT_BREAK",
    "LIMIT_UP_CNT_5",
    "LIMIT_DOWN_STREAK",
    "LIMIT_BREAK_5",
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
    "industry",
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
    computation without duplicating work.  ``industry`` carries the Shenwan
    first-level industry code per ``[stock x date]`` (a current-snapshot
    mapping repeated over dates, NaN for unmapped stocks); it is ``None``
    when no membership data is available, in which case the industry-relative
    factors stay neutral instead of fabricating groupings.  ``eligible``
    carries the PIT universe eligibility mask per ``[stock x date]`` (bool);
    every cross-sectional reference statistic — winsorize quantiles, the
    CAPM equal-weight market return, the amount-share denominator and the
    industry means — uses only eligible & finite cells, while a stock's own
    time-series history is never masked.  It is ``None`` when no mask is
    supplied (all finite cells are reference cells).
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
    eligible: pd.DataFrame
    industry: pd.DataFrame | None = None
    _cache: dict | None = None

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}


def _returns(close: pd.DataFrame) -> pd.DataFrame:
    # NaN for the first day and for missing days: downstream
    # cross-sectional standardization maps missing to the neutral 0.
    return close.pct_change(fill_method=None, axis=1).replace([np.inf, -np.inf], np.nan)


def _shift_ratio(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    # Pure numpy path: pandas' axis-1 shift of a wide frame produces a
    # fragmented block layout that emits PerformanceWarnings on every
    # subsequent operation.  Rebuilding the frame from numpy is exactly
    # the same shift (first ``periods`` columns become NaN) and always
    # yields a consolidated frame.
    values = close.to_numpy(dtype=float)
    shifted = np.roll(values, periods, axis=1)
    if periods > 0:
        shifted[:, :periods] = np.nan
    with np.errstate(all="ignore"):
        out = values / shifted - 1.0
    return pd.DataFrame(
        np.where(np.isfinite(out), out, np.nan),
        index=close.index,
        columns=close.columns,
    )


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
    eligible: np.ndarray,
    window: int = 60,
    min_periods: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rolling CAPM regression of each stock on the equal-weight market.

    Returns ``(beta, ivol, rsq)`` ``[stock x date]`` frames over trailing
    windows (expanding before ``window``, at least ``min_periods`` valid
    days, and a non-degenerate market/stock variance).  Prefix-sum
    vectorization keeps it cheap on the full cross-section.  The market is
    the equal-weight mean return per date over the *eligible and finite*
    cross-section (``eligible`` is the mandatory aligned ``[stock x date]``
    bool mask), so stocks outside the PIT universe can never move the
    market factor — each stock's own trailing window still uses only its
    own history, and the regression pairs are restricted to the stock's
    own valid sessions: market returns from sessions the stock did not
    trade (suspension gaps) never enter its window sums, so a stock
    suspended through a rally regresses on the sessions it actually
    observed.
    """

    r = _returns(close)
    arr = r.to_numpy(dtype=float)  # [S, T], NaN on missing days
    s, t = arr.shape
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != arr.shape:
        raise ValueError(
            f"eligible shape {eligible.shape} does not match "
            f"close shape {arr.shape}"
        )
    ref = ~np.isnan(arr) & eligible
    with warnings.catch_warnings():
        # All-NaN columns (e.g. the first date, or a date whose eligible
        # cross-section is empty) legitimately produce an empty-slice
        # warning; they map to a zero market return below.
        warnings.simplefilter("ignore", RuntimeWarning)
        mkt = np.nanmean(np.where(ref, arr, np.nan), axis=0)  # [T]
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
    # Market sums are accumulated per stock over the stock's own valid
    # days: a suspended stock's regression window must not include market
    # returns from sessions it did not trade.  n, sr, srm are already
    # stock-valid-masked (r0 is 0 on missing days), so masking the market
    # the same way keeps every window statistic on one denominator.
    m_stock = np.where(valid, m0, 0.0)
    pm = prefix(m_stock)
    pm2 = prefix(m_stock * m_stock)

    idx = np.arange(t)
    start = np.maximum(idx - window + 1, 0)
    end = idx + 1
    n = pv[:, end] - pv[:, start]
    sr = pr[:, end] - pr[:, start]
    srm = prm[:, end] - prm[:, start]
    sr2 = pr2[:, end] - pr2[:, start]
    sm = pm[:, end] - pm[:, start]
    sm2 = pm2[:, end] - pm2[:, start]

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


def _factor_amount_share(
    amount: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-sectional share of the day's total turnover amount.

    The per-date denominator sums only the eligible and finite amounts
    (``eligible`` is the mandatory aligned bool frame), so stocks outside the
    PIT universe cannot dilute the share.  Days with no eligible amount keep
    NaN (neutral).
    """

    reference = amount.where(eligible)
    denom = reference.fillna(0.0).sum(axis=0).replace(0.0, np.nan)
    return (amount / denom).replace([np.inf, -np.inf], np.nan)


def _factor_capm(ctx: FactorContext, which: int) -> pd.DataFrame:
    """Shared CAPM triple; ``which`` picks beta (0), ivol (1) or rsq (2)."""
    triple = ctx._cache.get("capm60")
    if triple is None:
        triple = _rolling_capm(
            ctx.close,
            ctx.eligible.to_numpy(dtype=bool),
            window=60,
            min_periods=20,
        )
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


def _limit_event_frame(ctx: FactorContext, direction: str) -> pd.DataFrame:
    """Cached one-word limit-move frame (computed once per direction)."""

    key = f"limit_{direction}"
    cached = ctx._cache.get(key)
    if cached is None:
        cached = _limit_events(
            ctx.close, ctx.high, ctx.low, ctx.pre_close, ctx.ts_codes, direction
        )
        ctx._cache[key] = cached
    return cached


def _streak_frame(event: np.ndarray, index, columns) -> pd.DataFrame:
    """Consecutive non-zero event days ending today (0 on non-event days).

    Trailing-only recursion: the streak at day t depends solely on days
    <= t, so no future information can leak in.
    """

    out = np.zeros_like(event)
    for t in range(event.shape[1]):
        if t == 0:
            out[:, t] = event[:, t]
        else:
            out[:, t] = np.where(event[:, t] > 0.0, out[:, t - 1] + 1.0, 0.0)
    return pd.DataFrame(out, index=index, columns=columns)


def _limit_streak(ctx: FactorContext) -> pd.DataFrame:
    """Consecutive one-word limit-up days ending today (0 on non-event days)."""

    event = _limit_event_frame(ctx, "up").to_numpy(dtype=float)
    return _streak_frame(event, ctx.close.index, ctx.close.columns)


def _limit_down_streak(ctx: FactorContext) -> pd.DataFrame:
    """Consecutive one-word limit-down days ending today (P9 §5.3)."""

    event = _limit_event_frame(ctx, "down").to_numpy(dtype=float)
    return _streak_frame(event, ctx.close.index, ctx.close.columns)


def _limit_up_count(ctx: FactorContext, window: int = 20) -> pd.DataFrame:
    """Number of one-word limit-up days in the trailing ``window`` days."""

    event = _limit_event_frame(ctx, "up")
    return event.T.rolling(window, min_periods=1).sum().T


def _limit_break(ctx: FactorContext) -> pd.DataFrame:
    """Limit-up touch without a sealed limit-up close (intraday break, 炸板).

    The high touched the board's limit price but the close fell back below
    the limit band: 1.0 on such days, 0.0 otherwise.  A close at the limit
    counts as sealed even when the day was not one-word (e.g. opened below
    the limit and closed locked).
    """

    rates = _limit_rate_per_stock(ctx.ts_codes)[:, None]
    with np.errstate(all="ignore"):
        limit_price = ctx.pre_close.to_numpy(dtype=float) * (1.0 + rates)
        touched = ctx.high.to_numpy(dtype=float) >= limit_price - 1e-9
        change = (
            ctx.close.to_numpy(dtype=float) / ctx.pre_close.to_numpy(dtype=float)
            - 1.0
        )
        sealed = change >= rates - 0.005
    touched = np.nan_to_num(touched, nan=False)
    sealed = np.nan_to_num(sealed, nan=False)
    return pd.DataFrame(
        (touched & ~sealed).astype(float),
        index=ctx.close.index,
        columns=ctx.close.columns,
    )


def _turnover_smoothed(ctx: FactorContext, window: int) -> pd.DataFrame:
    """Trailing mean of the turnover rate, neutral where turnover is missing.

    The daily turnover is missing whenever the float share count is
    unknown (never fabricated); the smoothed value is therefore only
    reported on days where the current turnover exists, and averages the
    available trailing values.
    """

    raw = ctx.turnover.replace([np.inf, -np.inf], np.nan)
    smoothed = raw.T.rolling(window, min_periods=1).mean().T
    return smoothed.where(raw.notna())


def _turnover_vol(ctx: FactorContext, window: int = 20) -> pd.DataFrame:
    """Trailing standard deviation of the turnover rate (same missingness rule)."""

    raw = ctx.turnover.replace([np.inf, -np.inf], np.nan)
    vol = raw.T.rolling(window, min_periods=2).std().T
    return vol.where(raw.notna())


def _amount_share_cached(ctx: FactorContext) -> pd.DataFrame:
    """The AMOUNT_SHARE cross-section, computed once per context."""

    cached = ctx._cache.get("amount_share")
    if cached is None:
        cached = _factor_amount_share(ctx.amount, ctx.eligible)
        ctx._cache["amount_share"] = cached
    return cached


def _rolling_corr(
    a: pd.DataFrame,
    b: pd.DataFrame,
    window: int = 20,
    min_periods: int = 10,
) -> pd.DataFrame:
    """Trailing-window Pearson correlation of two ``[stock x date]`` frames.

    NaN-aware and strictly backward-looking: prefix sums restricted to the
    pairs where both series are finite (the same technique as
    :func:`_rolling_capm`), so suspension gaps never fabricate correlation.
    Windows with fewer than ``min_periods`` valid pairs, or with a
    degenerate variance, yield NaN (neutral).
    """

    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    if av.shape != bv.shape:
        raise ValueError(f"rolling-corr shape mismatch: {av.shape} vs {bv.shape}")
    valid = np.isfinite(av) & np.isfinite(bv)
    r = np.where(valid, av, 0.0)
    s = np.where(valid, bv, 0.0)
    n_stocks, n_dates = av.shape

    def prefix(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([np.zeros((n_stocks, 1)), np.cumsum(mat, axis=1)], axis=1)

    pv = prefix(valid.astype(float))
    pr = prefix(r)
    ps = prefix(s)
    prs = prefix(r * s)
    pr2 = prefix(r * r)
    ps2 = prefix(s * s)

    idx = np.arange(n_dates)
    start = np.maximum(idx - window + 1, 0)
    end = idx + 1
    n = pv[:, end] - pv[:, start]
    sr = pr[:, end] - pr[:, start]
    ss = ps[:, end] - ps[:, start]
    srs = prs[:, end] - prs[:, start]
    sr2 = pr2[:, end] - pr2[:, start]
    ss2 = ps2[:, end] - ps2[:, start]

    with np.errstate(all="ignore"):
        cov = n * srs - sr * ss
        var_r = n * sr2 - sr * sr
        var_s = n * ss2 - ss * ss
        denom = np.sqrt(var_r * var_s)
        corr = np.where(
            (n >= min_periods) & (denom > 1e-12), cov / denom, np.nan
        )
    return pd.DataFrame(
        np.where(np.isfinite(corr), corr, np.nan),
        index=a.index,
        columns=a.columns,
    )


def _crowd_ratio(raw: pd.DataFrame, window: int) -> pd.DataFrame:
    """Level of ``raw`` relative to its own trailing mean (P9 §5.4 crowding).

    NaN stays NaN (no fabrication from missing history); a zero trailing
    mean is guarded like every other division in this module.
    """

    return (raw / (_rolling_mean(raw, window) + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    )


def _crowd_turnover(ctx: FactorContext) -> pd.DataFrame:
    """Turnover crowding: turnover rate relative to its own 60-day mean.

    Missingness follows the TURNOVER convention (``_turnover_smoothed``):
    the ratio exists only on days where today's turnover rate exists.
    """

    raw = ctx.turnover.replace([np.inf, -np.inf], np.nan)
    return _crowd_ratio(raw, 60).where(raw.notna())


def _sparse_safe_standardize(wide: pd.DataFrame, eligible) -> pd.DataFrame:
    """Mean/std standardization for sparse event features (P9 §5.3).

    The reference statistics (mean, standard deviation) are computed only
    over the eligible and finite cells of each date column, exactly like
    :func:`ashare_data.processor.winsorize_cross_section` — but the
    quantile winsorize is skipped, so a day with rare events keeps its
    non-degenerate cross-section instead of being clipped into an
    all-zero column (t2 finding F1).  The standard deviation is floored
    so a no-event day (constant cross-section) maps to the neutral 0
    instead of exploding; missing values stay NaN for the caller's
    neutral fill.
    """

    arr = wide.to_numpy(dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != arr.shape:
        raise ValueError(
            f"eligible shape {eligible.shape} does not match wide shape {arr.shape}"
        )
    ref = eligible & np.isfinite(arr)
    has_ref = ref.any(axis=0)
    reference = np.where(ref, arr, np.nan)
    reference[:, ~has_ref] = 0.0
    with np.errstate(all="ignore"):
        mean = np.nanmean(reference, axis=0)
        std = np.nanstd(reference, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1e-9)
    out = (arr - mean) / std
    out[:, ~has_ref] = 0.0
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def _industry_demean(ctx: FactorContext, frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract the per-industry cross-sectional mean of ``frame``.

    Membership comes from the (current-snapshot) Shenwan mapping carried in
    ``ctx.industry``; stocks without an industry stay NaN (neutral after
    standardization), and no industry is fabricated when the frame is
    absent.  Industry means are computed only over the *eligible and
    finite* members (``ctx.eligible``, the PIT universe mask), so an
    ineligible stock can never shift its industry's mean — its own raw
    value is still demeaned with that mean so its history stays observable.
    A stock's own NaN never contaminates its industry's mean.
    """

    if ctx.industry is None or ctx.industry.empty:
        return pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    codes = ctx.industry.reindex(frame.index).iloc[:, 0]
    if int(codes.notna().sum()) == 0:
        # No usable membership at all: stay neutral rather than asking
        # groupby to demean an empty set of groups.
        return pd.DataFrame(np.nan, index=frame.index, columns=frame.columns)
    reference = frame.where(
        ctx.eligible.reindex(index=frame.index, columns=frame.columns)
    )
    means = reference.groupby(codes, dropna=True).transform("mean")
    return frame - means


def _limit_rate_per_stock(ts_codes: list[str]) -> np.ndarray:
    """Board daily limit rate per stock: factors are historical by
    construction and there is no dated ST status, so the shared rule is
    evaluated without any ST flag."""
    return np.asarray([limit_rate(code) for code in ts_codes], dtype=float)


def _factor_macd_cached(ctx: FactorContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MACD pair computed once per context (DIF and DEA share one recursion)."""

    cached = ctx._cache.get("macd")
    if cached is None:
        cached = _factor_macd(ctx.close)
        ctx._cache["macd"] = cached
    return cached


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
        "TURNOVER_MA5",
        "volume",
        ("turnover_rate",),
        5,
        "5-day mean turnover rate (neutral where turnover is missing)",
        lambda ctx: _turnover_smoothed(ctx, 5),
    )
    add(
        "TURNOVER_MA20",
        "volume",
        ("turnover_rate",),
        20,
        "20-day mean turnover rate (neutral where turnover is missing)",
        lambda ctx: _turnover_smoothed(ctx, 20),
    )
    add(
        "TURNOVER_STD20",
        "volume",
        ("turnover_rate",),
        20,
        "20-day turnover-rate volatility (neutral where turnover is missing)",
        lambda ctx: _turnover_vol(ctx, 20),
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
    add("MOMENTUM_20", "momentum", ("close",), 21, "20-day momentum", lambda ctx: _shift_ratio(ctx.close, 20))
    add("MOMENTUM_60", "momentum", ("close",), 61, "60-day momentum", lambda ctx: _shift_ratio(ctx.close, 60))
    add("RET_120", "momentum", ("close",), 121, "120-day close-to-close return", lambda ctx: _shift_ratio(ctx.close, 120))
    add("REVERSAL_60", "momentum", ("close",), 61, "Negative 60-day return (medium-term reversal)", lambda ctx: -_shift_ratio(ctx.close, 60))
    add("REVERSAL_120", "momentum", ("close",), 121, "Negative 120-day return (medium-term reversal)", lambda ctx: -_shift_ratio(ctx.close, 120))
    add("REVERSAL_5", "momentum", ("close",), 6, "Negative 5-day return (short-term reversal)", lambda ctx: -_shift_ratio(ctx.close, 5))
    add("SKEW_20", "distribution", ("close",), 20, "20-day return skewness", lambda ctx: _rolling_skew(_returns(ctx.close), 20))
    add("KURT_20", "distribution", ("close",), 20, "20-day return excess kurtosis", lambda ctx: _rolling_kurt(_returns(ctx.close), 20))
    add(
        "LIMIT_UP_EVENT",
        "event",
        ("close", "high", "low", "pre_close"),
        2,
        "One-word limit-up close (1.0 on the event day)",
        lambda ctx: _limit_event_frame(ctx, "up"),
    )
    add(
        "LIMIT_DOWN_EVENT",
        "event",
        ("close", "high", "low", "pre_close"),
        2,
        "One-word limit-down close (1.0 on the event day)",
        lambda ctx: _limit_event_frame(ctx, "down"),
    )
    add(
        "LIMIT_STREAK",
        "event",
        ("close", "high", "low", "pre_close"),
        2,
        "Consecutive one-word limit-up days ending today",
        _limit_streak,
    )
    add(
        "LIMIT_UP_CNT_20",
        "event",
        ("close", "high", "low", "pre_close"),
        20,
        "One-word limit-up days in the trailing 20 days",
        _limit_up_count,
    )
    add(
        "LIMIT_BREAK",
        "event",
        ("close", "high", "low", "pre_close"),
        2,
        "Limit-up touch without a limit-up close (intraday break)",
        _limit_break,
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
        lambda ctx: _factor_amount_share(ctx.amount, ctx.eligible),
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
        lambda ctx: _factor_macd_cached(ctx)[0],
    )
    add(
        "MACD_DEA",
        "technical",
        ("close",),
        26,
        "MACD DEA signal line (EMA9 of DIF)",
        lambda ctx: _factor_macd_cached(ctx)[1],
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

    # Industry-relative versions of the strongest families.  The industry
    # code frame is supplied by the loader (Shenwan snapshot); without it
    # these factors stay neutral.  Raw values are demeaned per industry
    # *before* the cross-sectional standardization.
    add(
        "IND_REL_RET_5",
        "industry",
        ("close",),
        6,
        "5-day return minus its Shenwan-industry mean",
        lambda ctx: _industry_demean(ctx, _shift_ratio(ctx.close, 5)),
    )
    add(
        "IND_REL_RET_20",
        "industry",
        ("close",),
        21,
        "20-day return minus its Shenwan-industry mean",
        lambda ctx: _industry_demean(ctx, _shift_ratio(ctx.close, 20)),
    )
    add(
        "IND_REL_VOL_20",
        "industry",
        ("close",),
        20,
        "20-day volatility minus its Shenwan-industry mean",
        lambda ctx: _industry_demean(ctx, _rolling_std(_returns(ctx.close), 20)),
    )
    add(
        "IND_REL_TURNOVER",
        "industry",
        ("turnover_rate",),
        1,
        "Turnover rate minus its Shenwan-industry mean",
        lambda ctx: _industry_demean(
            ctx, ctx.turnover.replace([np.inf, -np.inf], np.nan)
        ),
    )

    # --- P9 additions (docs/p9_factor_family_contract.md §5) ---------------
    # Family ①: industry-residualized medium-horizon momentum/reversal.
    # (Market residualization is already implicit in the per-date
    # cross-sectional standardization; only the industry dimension is new.)
    add(
        "IND_REL_RET_60",
        "industry",
        ("close",),
        61,
        "60-day return minus its Shenwan-industry mean",
        lambda ctx: _industry_demean(ctx, _shift_ratio(ctx.close, 60)),
    )
    add(
        "IND_REL_RET_120",
        "industry",
        ("close",),
        121,
        "120-day return minus its Shenwan-industry mean",
        lambda ctx: _industry_demean(ctx, _shift_ratio(ctx.close, 120)),
    )
    # Family ②: liquidity shock, volume shrinkage, price-volume divergence.
    add(
        "LIQ_SHOCK_20",
        "liquidity",
        ("amount",),
        20,
        "One-day amount-share shock vs its 20-day baseline (share / MA20 - 1)",
        lambda ctx: _crowd_ratio(_amount_share_cached(ctx), 20),
    )
    add(
        "VOLUME_SHRINK_5_20",
        "volume",
        ("volume",),
        20,
        "Volume shrinkage: MA5(volume) / MA20(volume) - 1",
        lambda ctx: (
            _rolling_mean(ctx.volume, 5) / (_rolling_mean(ctx.volume, 20) + 1e-9)
            - 1.0
        ).replace([np.inf, -np.inf], np.nan),
    )
    add(
        "PV_DIV_20",
        "volume",
        ("close", "volume"),
        20,
        "Price-volume divergence: 20-day correlation of daily returns and log-volume changes",
        lambda ctx: _rolling_corr(
            _returns(ctx.close),
            np.log1p(ctx.volume.clip(lower=0)).diff(axis=1),
            window=20,
            min_periods=10,
        ),
    )
    # Family ③: limit-event conditioning (requires the P9 sparse-safe
    # standardization path; see SPARSE_EVENT_FEATURES).
    add(
        "LIMIT_UP_CNT_5",
        "event",
        ("close", "high", "low", "pre_close"),
        5,
        "One-word limit-up days in the trailing 5 days",
        lambda ctx: _limit_up_count(ctx, 5),
    )
    add(
        "LIMIT_DOWN_STREAK",
        "event",
        ("close", "high", "low", "pre_close"),
        2,
        "Consecutive one-word limit-down days ending today",
        _limit_down_streak,
    )
    add(
        "LIMIT_BREAK_5",
        "event",
        ("close", "high", "low", "pre_close"),
        5,
        "Limit-up breaks (intraday touch without a sealed close) in the trailing 5 days",
        lambda ctx: _limit_break(ctx).T.rolling(5, min_periods=1).sum().T,
    )
    # Family ④: per-stock crowding (market breadth is deliberately OUT of
    # the vocabulary — a per-date constant cannot survive cross-sectional
    # standardization; P9 §5.4).
    add(
        "CROWD_TURNOVER_60",
        "liquidity",
        ("turnover_rate",),
        60,
        "Turnover crowding: turnover rate relative to its own 60-day mean",
        _crowd_turnover,
    )
    add(
        "CROWD_AMOUNT_60",
        "liquidity",
        ("amount",),
        60,
        "Amount-share crowding: today's market share relative to its own 60-day mean",
        lambda ctx: _crowd_ratio(_amount_share_cached(ctx), 60),
    )
    # MARGIN_CROWD_60 is injected via build_capital_frames (external feed,
    # like MARGIN_BALANCE_CHG) and is therefore not registered here.
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
        eligible_frame: pd.DataFrame,
        industry_frame: pd.DataFrame | None = None,
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
            industry=industry_frame,
            eligible=eligible_frame,
        )

    def compute_factor_tensor(
        self,
        bars: pd.DataFrame,
        ts_codes: list[str],
        dates: list[str],
        universe_mask: np.ndarray,
        pit_fundamentals: dict[str, pd.DataFrame] | None = None,
        extra_frames: dict[str, pd.DataFrame] | None = None,
        industry_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Return a ``[feature, stock, date]`` float32 tensor.

        ``pit_fundamentals`` maps a PIT feature name to its
        disclosure-aligned ``[stock x date]`` frame (built by
        :func:`ashare_data.fundamentals.build_pit_frames`); ``extra_frames``
        carries the other externally fed features (margin/industry, built by
        :func:`ashare_data.capital_flow.build_capital_frames`);
        ``industry_frame`` is the ``[stock x date]`` Shenwan first-level
        industry code frame feeding the industry-relative factors (built by
        :func:`ashare_data.capital_flow.build_industry_member_frame`).
        Frames are injected verbatim; the no-lookahead guarantees live in
        the builders.  Missing names/frames stay neutral (0) after
        standardization.

        ``universe_mask`` is the mandatory ``[stock x date]`` bool PIT
        eligibility mask aligned with ``(ts_codes, dates)``.  Every
        stock-axis reference statistic (winsorize quantiles / median / MAD,
        the CAPM equal-weight market return, the amount-share denominator,
        the industry means) is computed only over eligible & finite cells,
        so stocks outside the universe cannot change eligible stocks'
        factor values.  A stock's own bar history is never masked: e.g. the
        join-day momentum still reads its pre-join prices.  A mask whose
        shape does not match ``(ts_codes, dates)`` raises ``ValueError``.
        """

        bars = normalize_daily_bars(bars)
        date_order = sorted(range(len(dates)), key=dates.__getitem__)
        dates = [dates[i] for i in date_order]
        universe_mask = np.asarray(universe_mask, dtype=bool)
        if universe_mask.shape != (len(ts_codes), len(dates)):
            raise ValueError(
                f"universe_mask shape {universe_mask.shape} does not match "
                f"(stock={len(ts_codes)}, date={len(dates)})"
            )
        universe_mask = universe_mask[:, date_order]
        eligible_frame = pd.DataFrame(
            universe_mask, index=ts_codes, columns=dates
        )
        close = self._pivot(bars, ts_codes, dates, "close")
        context = self._build_context(
            bars, ts_codes, dates, close, eligible_frame, industry_frame
        )

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
            if name in SPARSE_EVENT_FEATURES:
                # P9 §5.3: sparse event features skip the quantile
                # winsorize (t2 finding F1) and standardize on the plain
                # mean/std of the eligible cross-section with a floored
                # standard deviation.
                frame = _sparse_safe_standardize(frame, eligible=universe_mask)
            else:
                frame = winsorize_cross_section(frame, eligible=universe_mask)
            # After cross-sectional standardization 0 is the neutral value.
            standardized.append(frame.fillna(0.0).values.astype(np.float32))

        return np.stack(standardized, axis=0)


def compute_factor_tensor(
    bars: pd.DataFrame,
    ts_codes: list[str],
    dates: list[str],
    universe_mask: np.ndarray,
    pit_fundamentals: dict[str, pd.DataFrame] | None = None,
    extra_frames: dict[str, pd.DataFrame] | None = None,
    industry_frame: pd.DataFrame | None = None,
) -> np.ndarray:
    return AshareFactorEngine().compute_factor_tensor(
        bars, ts_codes, dates, universe_mask, pit_fundamentals, extra_frames,
        industry_frame,
    )


def ablate_factors(
    tensor: np.ndarray,
    excluded_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Zero the rows of the excluded features (family ablation).

    The tensor keeps its full ``[feature, stock, date]`` shape so formula
    token ids stay valid; the excluded features simply become neutral (0).
    """

    out = np.asarray(tensor, dtype=np.float32).copy()
    for name in excluded_names:
        if name in FEATURE_NAMES:
            out[FEATURE_NAMES.index(name)] = 0.0
    return out
