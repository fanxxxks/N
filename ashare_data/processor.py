"""Data cleaning, universe filtering, and cross-sectional preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Shanghai main board / STAR market (incl. CDR) code prefixes.
_SH_PREFIXES = {"600", "601", "603", "605", "688", "689"}
# Shenzhen main board / SME-merged / ChiNext code prefixes.
_SZ_PREFIXES = {"000", "001", "002", "003", "300", "301"}
# Beijing Stock Exchange prefixes (43x/8xx/92x families).
_BJ_PREFIXES = {
    "430",
    "830",
    "831",
    "832",
    "833",
    "834",
    "835",
    "836",
    "837",
    "838",
    "839",
    "870",
    "871",
    "872",
    "873",
    "874",
    "875",
    "876",
    "877",
    "878",
    "879",
    "920",
}

# CSI/SZSE index symbols that share the 000xxx space with real SZ stocks
# (000300 = CSI300, 000905 = CSI500, 000852 = CSI1000, ...).  A pure prefix
# check cannot tell them apart from listed stocks, so they are blocked here
# and, downstream, by intersection with the actual stock list.
_KNOWN_INDEX_SYMBOLS = {
    "000016",
    "000300",
    "000688",
    "000852",
    "000902",
    "000903",
    "000904",
    "000905",
    "000906",
    "000907",
    "000922",
}


def is_valid_a_share_code(ts_code: str) -> bool:
    """Return True when ``ts_code`` looks like a real A-share stock code.

    Index codes (e.g. ``000300.SZ``, ``000905.SZ``), B-shares
    (``900xxx.SH``) and other non-stock instruments are rejected.  The
    validator is applied everywhere codes enter the universe so that index
    symbols can never be stored or traded as stocks.
    """

    if not isinstance(ts_code, str):
        return False
    ts_code = ts_code.strip()
    if "." not in ts_code:
        return False
    symbol, suffix = ts_code.split(".", 1)
    if len(symbol) != 6 or not symbol.isdigit():
        return False
    if symbol in _KNOWN_INDEX_SYMBOLS:
        return False
    prefix = symbol[:3]
    if suffix == "SH":
        return prefix in _SH_PREFIXES
    if suffix == "SZ":
        return prefix in _SZ_PREFIXES
    if suffix == "BJ":
        return prefix in _BJ_PREFIXES
    return False


# Board daily-limit prefixes (ST stocks trade under 5% via an explicit
# as-of flag — see :func:`limit_rate`).
_CHINEXT_LIMIT_PREFIXES = {"300", "301", "688", "689"}


def limit_rate(ts_code: str, *, is_st: bool = False) -> float:
    """Daily price-limit rate of one stock.

    ST stocks trade under a 5% band.  There is no dated ST history, so the
    caller must supply the as-of ``is_st`` flag explicitly: the current
    ``stocks.is_st`` snapshot is only valid for same-day execution and must
    never be projected onto historical dates.  Stock names are display data
    and never influence the rate.  ChiNext / STAR markets trade under 20%;
    every other board 10%.  Single code path shared by the factor engine,
    the backtest / reward tradability masks and the paper-trading matcher.
    """

    if is_st:
        return 0.05
    prefix = ts_code.split(".")[0][:3]
    if prefix in _CHINEXT_LIMIT_PREFIXES:
        return 0.20
    return 0.10


def _blocked_parts(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    pc: np.ndarray,
    v: np.ndarray,
    rates: np.ndarray,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``(suspended, limit)`` mask pair: one implementation of the exchange
    rule shared by the per-date column and full-matrix paths.

    ``o``/``h``/``l``/``pc``/``v`` share one shape (a single date column or
    the full ``[stock x date]`` matrix); ``rates`` is broadcastable to it.
    Missing cells are 0 (the raw-cache convention), which marks suspension.
    The buy side blocks one-word limit-up opens (cannot buy at the board);
    the sell side blocks one-word limit-down opens (cannot sell at the
    board).
    """

    suspended = (o <= 0) | (v <= 0) | (pc <= 0)
    one_word = np.isclose(o, h) & np.isclose(o, l)
    change = np.zeros_like(o)
    valid = pc > 0
    change[valid] = o[valid] / pc[valid] - 1.0
    if side == "buy":
        limit = one_word & (change >= rates - 0.005)
    elif side == "sell":
        limit = one_word & (change <= -rates + 0.005)
    else:
        raise ValueError(f"unknown side {side!r}; expected 'buy' or 'sell'")
    return suspended, limit


def _limit_rates(ts_codes: list[str], st_mask: np.ndarray | None) -> np.ndarray:
    """Per-stock daily limit rates for one date column.

    ``st_mask`` optionally marks the stocks that are ST as of that exact
    date; without dated ST history the caller leaves it None and only the
    board rates apply.
    """

    if st_mask is not None:
        st = np.asarray(st_mask, dtype=bool)
        if st.shape != (len(ts_codes),):
            raise ValueError(
                f"st_mask shape {st.shape} does not match {len(ts_codes)} ts_codes"
            )
        return np.asarray(
            [limit_rate(code, is_st=bool(flag)) for code, flag in zip(ts_codes, st)],
            dtype=np.float64,
        )
    return np.asarray([limit_rate(code) for code in ts_codes], dtype=np.float64)


def blocked_components(
    open_col: np.ndarray,
    high_col: np.ndarray,
    low_col: np.ndarray,
    pre_close_col: np.ndarray,
    volume_col: np.ndarray,
    ts_codes: list[str],
    side: str,
    *,
    st_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(suspended, limit)`` per-stock bool masks for one date column.

    The exact rule behind :func:`tradability_blocked`, split so callers
    that need distinct reasons (the paper-trading matcher) do not
    re-derive them.  ``st_mask`` optionally supplies per-stock as-of ST
    flags for same-day execution; historical paths leave it None so a
    current ``*ST`` name can never rewrite the past.
    """

    o = np.asarray(open_col, dtype=np.float64).reshape(-1, 1)
    h = np.asarray(high_col, dtype=np.float64).reshape(-1, 1)
    l = np.asarray(low_col, dtype=np.float64).reshape(-1, 1)
    pc = np.asarray(pre_close_col, dtype=np.float64).reshape(-1, 1)
    v = np.asarray(volume_col, dtype=np.float64).reshape(-1, 1)
    if not (o.shape == h.shape == l.shape == pc.shape == v.shape):
        raise ValueError("open/high/low/pre_close/volume columns must share one shape")
    if o.shape[0] != len(ts_codes):
        raise ValueError(f"{len(ts_codes)} ts_codes but {o.shape[0]} rows")
    rates = _limit_rates(ts_codes, st_mask)[:, None]
    suspended, limit = _blocked_parts(o, h, l, pc, v, rates, side)
    return suspended[:, 0], limit[:, 0]


def tradability_blocked(
    open_col: np.ndarray,
    high_col: np.ndarray,
    low_col: np.ndarray,
    pre_close_col: np.ndarray,
    volume_col: np.ndarray,
    ts_codes: list[str],
    side: str,
    *,
    st_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Buy/sell blocked mask for one date column (``[stock]`` bool).

    Same semantics as :func:`tradability_blocked_matrix`, vectorized per
    column; the backtest engine calls this per execution day.  ``st_mask``
    optionally marks the stocks that are ST as of that exact execution
    date (the only valid use of the current snapshot); historical callers
    leave it None, so limit rates come from the board alone and stock
    names never influence the result.
    """

    suspended, limit = blocked_components(
        open_col,
        high_col,
        low_col,
        pre_close_col,
        volume_col,
        ts_codes,
        side,
        st_mask=st_mask,
    )
    return suspended | limit


def tradability_blocked_matrix(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    pre_close: np.ndarray,
    volume: np.ndarray,
    ts_codes: list[str],
    side: str,
) -> np.ndarray:
    """Buy/sell blocked mask over the full ``[stock x date]`` window.

    The reward path precomputes this matrix once and shares it across the
    whole formula batch; per date it is the exact rule the backtest engine
    applies through :func:`tradability_blocked`.  The matrix is historical
    by construction: there is no dated ST status, so only board limit
    rates apply and stock names never influence the result.
    """

    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    pc = np.asarray(pre_close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    if not (o.shape == h.shape == l.shape == pc.shape == v.shape):
        raise ValueError("open/high/low/pre_close/volume must share one shape")
    if o.ndim != 2 or o.shape[0] != len(ts_codes):
        raise ValueError(f"expected [stock x date] matrices for {len(ts_codes)} ts_codes")
    rates = np.asarray([limit_rate(code) for code in ts_codes], dtype=np.float64)[:, None]
    suspended, limit = _blocked_parts(o, h, l, pc, v, rates, side)
    return suspended | limit


def encode_industry_frame(frame: pd.DataFrame) -> np.ndarray:
    """Encode a ``[stock x date]`` industry-code frame into dense ids.

    Object/string Shenwan codes are factorized to dense integer ids for the
    VM's group-neutralization operator; unmapped cells (NaN) stay NaN so
    the neutralization excludes them instead of fabricating a group.
    """

    arr = frame.to_numpy(dtype=object)
    codes, _ = pd.factorize(arr.ravel(), use_na_sentinel=True)
    out = codes.reshape(arr.shape).astype(np.float32)
    out[out < 0] = np.nan
    return out


def open_to_open_returns(open_: np.ndarray) -> np.ndarray:
    """Compute ``open[t+2] / open[t+1] - 1`` per stock with missing-data mask.

    The entry price is the day-``t+1`` open and the exit price the day-``t+2``
    open, matching the backtest's next-open execution model.  Positions where
    either price is missing (suspension / not listed) yield 0.0 so they can
    never fabricate huge fake returns; the final two columns have no future
    and are zeroed.
    """

    open_ = np.asarray(open_, dtype=np.float64)
    t1 = np.roll(open_, -1, axis=1)
    t2 = np.roll(open_, -2, axis=1)
    ret = np.zeros_like(open_)
    valid = (t1 > 0) & (t2 > 0)
    ret[valid] = (t2[valid] - t1[valid]) / t1[valid]
    ret[:, -2:] = 0.0
    return ret


def normalize_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean daily-bar frame with the columns used downstream."""

    if df is None or df.empty:
        return pd.DataFrame()

    required = {
        "ts_code": "ts_code",
        "trade_date": "trade_date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "amount": "amount",
    }
    optional = {
        "turnover_rate": "turnover_rate",
        "adj_factor": "adj_factor",
    }

    df = df.copy()
    for col, target in required.items():
        if col not in df.columns:
            if target in df.columns:
                df[col] = df[target]
            else:
                return pd.DataFrame()

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    if "pre_close" in df.columns:
        numeric_cols.append("pre_close")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in optional.values():
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["trade_date", "open", "close"])
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    ]
    if "pre_close" not in df.columns:
        df["pre_close"] = df.groupby("ts_code")["close"].shift(1)
    df["pre_close"] = df.groupby("ts_code")["pre_close"].ffill()
    df.loc[df["pre_close"].isna(), "pre_close"] = df["open"]
    df["volume"] = df["volume"].fillna(0.0)
    df["amount"] = df["amount"].fillna(0.0)
    df["adj_factor"] = df["adj_factor"].ffill().bfill()
    # Turnover rate cannot be reconstructed from OHLCV (it needs the float
    # share count), so a missing column stays missing instead of pretending
    # to be ~1.0.  Downstream standardization treats it as neutral.
    df["turnover_rate"] = df["turnover_rate"].replace([np.inf, -np.inf], np.nan)

    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"], keep="last"
    )
    return df


def pivot_wide(
    bars: pd.DataFrame,
    ts_codes: list[str],
    dates: list[str],
    value_col: str = "close",
) -> pd.DataFrame:
    """Pivot long daily bars into a ``[stock x date]`` matrix."""

    bars = normalize_daily_bars(bars)
    if bars.empty or not ts_codes:
        return pd.DataFrame(index=ts_codes, columns=dates, dtype=float)
    bars = bars[bars["ts_code"].isin(ts_codes)]
    bars = bars[bars["trade_date"].isin(dates)]
    wide = bars.pivot_table(index="ts_code", columns="trade_date", values=value_col, aggfunc="last")
    return wide.reindex(index=ts_codes, columns=dates)


def winsorize_cross_section(
    wide: pd.DataFrame,
    eligible: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Winsorize then robust z-score each date cross-section.

    The input is ``[stock x date]``.  The quantile points, median and MAD of
    each date column are computed only over the *eligible and finite*
    reference cells: ``eligible`` is the mandatory ``[stock x date]`` bool
    mask aligned with ``wide`` (the PIT universe mask), so stocks outside
    the universe can never shift the reference statistics.  The same
    clip/standardize transform is then applied to every cell, keeping each
    stock's own history on one per-date scale (a future member's pre-join
    values stay observable for time-series use instead of being zeroed);
    only the reference set decides the statistics.  Standardization uses
    only the current date column, so it does not leak future information.
    Missing values stay NaN (the caller converts them to the neutral 0); a
    degenerate column with zero MAD — or with no reference cell at all —
    maps to 0 instead of exploding into ±clip spikes.
    """

    arr = wide.to_numpy(dtype=float).copy()
    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != arr.shape:
        raise ValueError(
            f"eligible shape {eligible.shape} does not match "
            f"wide shape {arr.shape}"
        )
    ref = eligible & np.isfinite(arr)
    has_ref = ref.any(axis=0)
    # Reference matrix for the quantile points: non-reference cells are NaN
    # so they cannot contribute; empty columns are zeroed up front so every
    # reduce below stays finite.
    reference = np.where(ref, arr, np.nan)
    reference[:, ~has_ref] = 0.0
    with np.errstate(all="ignore"):
        lower = np.nanquantile(reference, lower_q, axis=0)
        upper = np.nanquantile(reference, upper_q, axis=0)
        clipped = np.clip(arr, lower, upper)
        # Median/MAD input over the reference cells only; empty columns are
        # zeroed like the quantile matrix so every reduce stays warning-free.
        center = np.where(ref, clipped, np.nan)
        center[:, ~has_ref] = 0.0
        median = np.nanmedian(center, axis=0)
        mad = np.nanmedian(np.abs(center - median), axis=0)
        mad = np.where(mad == 0, 1.0, mad)
    result = (clipped - median) / mad
    result = np.clip(result, -clip, clip)
    result[:, ~has_ref] = 0.0
    return pd.DataFrame(result, index=wide.index, columns=wide.columns)
