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


# Board daily-limit prefixes (ST stocks get 5% via the name check).
_CHINEXT_LIMIT_PREFIXES = {"300", "301", "688", "689"}


def limit_rate(ts_code: str, name: str = "") -> float:
    """Daily price-limit rate of one stock.

    ST stocks trade under a 5% band (detected by name, so pass the stock
    name when available); ChiNext / STAR markets trade under 20%; every
    other board 10%.  Single code path shared by the factor engine (which
    has no names because ST stocks are excluded upstream) and the backtest /
    reward tradability masks.
    """

    if "ST" in name.upper():
        return 0.05
    prefix = ts_code.split(".")[0][:3]
    if prefix in _CHINEXT_LIMIT_PREFIXES:
        return 0.20
    return 0.10


def _blocked_columns(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    pc: np.ndarray,
    v: np.ndarray,
    rates: np.ndarray,
    side: str,
) -> np.ndarray:
    """Blocked mask core: suspended stocks plus one-word limit moves.

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
    return suspended | limit


def tradability_blocked(
    open_col: np.ndarray,
    high_col: np.ndarray,
    low_col: np.ndarray,
    pre_close_col: np.ndarray,
    volume_col: np.ndarray,
    ts_codes: list[str],
    stock_names: dict[str, str],
    side: str,
) -> np.ndarray:
    """Buy/sell blocked mask for one date column (``[stock]`` bool).

    Same semantics as :func:`tradability_blocked_matrix`, vectorized per
    column; the backtest engine calls this per execution day.
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
    rates = np.asarray(
        [limit_rate(c, stock_names.get(c, "")) for c in ts_codes], dtype=np.float64
    )[:, None]
    return _blocked_columns(o, h, l, pc, v, rates, side)[:, 0]


def tradability_blocked_matrix(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    pre_close: np.ndarray,
    volume: np.ndarray,
    ts_codes: list[str],
    stock_names: dict[str, str],
    side: str,
) -> np.ndarray:
    """Buy/sell blocked mask over the full ``[stock x date]`` window.

    The reward path precomputes this matrix once and shares it across the
    whole formula batch; per date it is the exact rule the backtest engine
    applies through :func:`tradability_blocked`.
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
    rates = np.asarray(
        [limit_rate(c, stock_names.get(c, "")) for c in ts_codes], dtype=np.float64
    )[:, None]
    return _blocked_columns(o, h, l, pc, v, rates, side)


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


def filter_universe(
    stocks: pd.DataFrame,
    bars: pd.DataFrame,
    constituents: list[str] | None,
    config,
) -> list[str]:
    """Build the A-share universe with the objective's basic filters.

    Filters applied per stock (on the full history):
    * minimum number of listed trading days;
    * last close within [min_price, max_price];
    * last-day turnover amount >= min_amount;
    * exclude ST stocks (by name or ``is_st`` flag);
    * optional intersection with an index constituent list.
    """

    if bars is None or bars.empty:
        return []
    bars = normalize_daily_bars(bars)
    if bars.empty:
        return []

    if constituents is not None:
        cons = {str(c) for c in constituents}
    else:
        cons = None

    if stocks is not None and not stocks.empty:
        stocks = stocks.copy()
        stocks["name"] = stocks.get("name", "").fillna("").astype(str)
        st_codes = set(
            stocks.loc[stocks["name"].str.contains("ST", case=False, na=False), "ts_code"]
        )
        if "is_st" in stocks.columns:
            st_codes.update(
                stocks.loc[
                    pd.to_numeric(stocks["is_st"], errors="coerce").fillna(0) != 0,
                    "ts_code",
                ]
            )
    else:
        st_codes = set()

    # Bars are sorted by (ts_code, trade_date), so "last" is the latest row.
    agg = bars.groupby("ts_code", as_index=False, sort=False).agg(
        listed_days=("trade_date", "nunique"),
        last_close=("close", "last"),
        last_amount=("amount", "last"),
    )
    agg = agg[
        (agg["listed_days"] >= config.min_listed_days)
        & (agg["last_close"] >= config.min_price)
        & (agg["last_close"] <= config.max_price)
        & (agg["last_amount"] >= config.min_amount)
    ]

    codes = {str(c) for c in agg["ts_code"]}
    codes.difference_update(st_codes)
    if cons is not None:
        if cons:
            codes.intersection_update(cons)
    return sorted(codes)


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
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Winsorize then robust z-score each date cross-section.

    The input is ``[stock x date]``. Standardization uses only the values in
    the current date column, so it does not leak future information.  Missing
    values stay NaN (the caller converts them to the neutral 0); a degenerate
    cross-section with zero MAD is normalized by 1.0 so constant columns map
    to 0 instead of exploding into ±clip spikes.
    """

    arr = wide.to_numpy(dtype=float).copy()
    all_nan = np.isnan(arr).all(axis=0)
    arr[:, all_nan] = 0.0
    with np.errstate(all="ignore"):
        lower = np.nanquantile(arr, lower_q, axis=0)
        upper = np.nanquantile(arr, upper_q, axis=0)
        clipped = np.clip(arr, lower, upper)
        median = np.nanmedian(clipped, axis=0)
        mad = np.nanmedian(np.abs(clipped - median), axis=0)
        mad = np.where(mad == 0, 1.0, mad)
    result = (clipped - median) / mad
    result = np.clip(result, -clip, clip)
    result[:, all_nan] = 0.0
    return pd.DataFrame(result, index=wide.index, columns=wide.columns)


def long_factor_frame(
    factor_tensor: np.ndarray,
    ts_codes: list[str],
    dates: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    """Flatten a ``[feature, stock, date]`` tensor into a long DataFrame."""

    records: list[dict[str, object]] = []
    n_features, n_stocks, n_dates = factor_tensor.shape
    for f, feature in enumerate(feature_names):
        for s, ts_code in enumerate(ts_codes):
            for t, date in enumerate(dates):
                value = factor_tensor[f, s, t]
                if not np.isfinite(value):
                    value = np.nan
                records.append(
                    {
                        "ts_code": ts_code,
                        "trade_date": date,
                        "factor_name": feature,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(records)
