"""Margin-balance and Shenwan-industry auxiliary factors.

Sources (all free via AkShare):
* Margin financing: the SSE/SZSE per-date cross-section feeds
  (``stock_margin_detail_sse`` / ``stock_margin_detail_szse``) publish the
  full market for a given trading date, so a backfill costs one request
  per exchange per trading day.  Published after the close of the as-of
  date: the factor column stores the as-of value, and the engine only
  trades the signal at the next open, so no lookahead.
* Shenwan first-level industries: the industry index daily closes
  (``index_hist_sw``) and the CURRENT member snapshot
  (``index_component_sw``).  Membership is a current snapshot, the same
  survivorship caveat as the index constituents.

:func:`build_capital_frames` turns the stored tables into
``[stock x date]`` frames for the three vocabulary features:
``MARGIN_BALANCE_CHG`` (20-day financing-balance change),
``MARGIN_CROWD_60`` (P9 §5.4: balance relative to its own 60-day mean) and
``INDUSTRY_MOMENTUM`` (20-day industry index return mapped to members).
:func:`build_industry_member_frame` exposes the raw membership snapshot as
a ``[stock x date]`` industry-code frame for the industry-relative factors.
Stocks with no data stay neutral, never fabricated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .db import AshareDB, sql_quoted_list
from .fundamentals import _cached_or_fetch, _read_cached, _write_cached

# Vocabulary feature names produced by this module.
EXTERNAL_FACTOR_NAMES = ("MARGIN_BALANCE_CHG", "INDUSTRY_MOMENTUM", "MARGIN_CROWD_60")

_MARGIN_WINDOW = 20
_MARGIN_CROWD_WINDOW = 60
_INDUSTRY_WINDOW = 20
# Margin data is final shortly after publication; dates older than this
# many days are never refetched.
_MARGIN_STALE_DAYS = 30


def _capital_cache_path(config, name: str) -> Any:
    return config.parquet_dir / "capital" / f"{name}.parquet"


def sync_capital_flow(
    client,
    db: AshareDB,
    config,
    calendar_dates: list[str],
) -> dict[str, Any]:
    """Sync margin balances (whole market, by trading date) and Shenwan
    industry indexes/members (whole market, by industry).

    Degrades per date/industry like the other syncs: failures are logged
    and skipped, never fatal.  Offline mode skips entirely.
    """

    if client.offline:
        return {
            "margin_rows": 0,
            "margin_dates": 0,
            "industries": 0,
            "industry_rows": 0,
            "capital_failures": 0,
        }

    today = datetime.now().strftime("%Y%m%d")
    start = config.start_date.replace("-", "")
    dates = [d for d in calendar_dates if start <= d <= today]
    margin_rows = 0
    failures = 0
    for date in dates:
        path = _capital_cache_path(config, f"margin_{date}")
        age_days = (datetime.now() - datetime.strptime(date, "%Y%m%d")).days
        settled = age_days > _MARGIN_STALE_DAYS
        df = _read_cached(path) if settled else pd.DataFrame()
        if df.empty:
            parts = []
            for exchange in ("sse", "szse"):
                try:
                    part = client.get_margin_detail(exchange, date)
                    if part is not None and not part.empty:
                        parts.append(part)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Margin {exchange} {date} failed: {exc}")
                    failures += 1
            if parts:
                df = pd.concat(parts, ignore_index=True).drop_duplicates(
                    subset=["ts_code"], keep="last"
                )
                _write_cached(path, df)
        if df is None or df.empty:
            continue
        db.upsert_margin(df.to_dict("records"), config)
        margin_rows += len(df)

    industries = client.get_sw_industry_list()
    industry_rows = 0
    industry_count = 0
    for row in industries.itertuples():
        code = str(row.index_code)
        name = str(row.industry_name)
        try:
            history = _cached_or_fetch(
                config,
                f"sw_hist_{code}",
                lambda: client.get_sw_index_history(code),
            )
            components = _cached_or_fetch(
                config,
                f"sw_comp_{code}",
                lambda: client.get_sw_components(code),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Shenwan industry {code} failed: {exc}")
            failures += 1
            continue
        if history is not None and not history.empty:
            history = history.copy()
            history["industry_name"] = name
            db.upsert_sw_index(history.to_dict("records"), config)
            industry_rows += len(history)
        if components is not None and not components.empty:
            db.replace_sw_members(code, components["ts_code"].astype(str).tolist(), config)
        industry_count += 1

    logger.info(
        f"Capital flow synced: {len(dates)} margin dates ({margin_rows} rows), "
        f"{industry_count} industries ({industry_rows} index rows)"
    )
    return {
        "margin_rows": margin_rows,
        "margin_dates": len(dates),
        "industries": industry_count,
        "industry_rows": industry_rows,
        "capital_failures": failures,
    }


def _industry_map(config) -> dict[str, str]:
    """``ts_code -> Shenwan first-level industry index code`` (snapshot)."""

    with AshareDB(config.duckdb_path, read_only=True) as db:
        members = db.query(
            f"SELECT index_code, ts_code FROM {config.sw_member_table}"
        )
    return {
        str(code): str(index)
        for code, index in zip(members["ts_code"], members["index_code"])
    }


def build_industry_member_frame(
    config,
    ts_codes: list[str],
    dates: list[str],
) -> pd.DataFrame:
    """Build the ``[stock x date]`` Shenwan first-level industry code frame.

    Membership is a current snapshot repeated over all dates (the same
    survivorship caveat as ``INDUSTRY_MOMENTUM``); unmapped stocks stay
    NaN so the industry-relative factors never fabricate a grouping.  Any
    failure (e.g. the table does not exist yet) degrades to an all-NaN
    frame, keeping the loader independent of this data source.
    """

    frame = pd.DataFrame(np.nan, index=ts_codes, columns=dates, dtype=object)
    if not ts_codes:
        return frame
    try:
        industry_map = _industry_map(config)
    except Exception as exc:  # noqa: BLE001 - table may not exist yet.
        logger.warning(f"Industry membership frame unavailable: {exc}")
        return frame
    for code in ts_codes:
        industry = industry_map.get(code)
        if industry is not None:
            frame.loc[code] = industry
    return frame


def _margin_crowd(wide: pd.DataFrame, window: int = _MARGIN_CROWD_WINDOW) -> pd.DataFrame:
    """Margin-balance crowding: the balance relative to its own trailing mean
    (P9 §5.4).  Strict warmup: the first ``window - 1`` sessions stay NaN
    (no fabricated baseline from partial history); missing values stay NaN.
    """

    base = wide.replace([np.inf, -np.inf], np.nan)
    baseline = base.T.rolling(window, min_periods=window).mean().T
    return (base / baseline).replace([np.inf, -np.inf], np.nan)


def build_capital_frames(
    config,
    ts_codes: list[str],
    dates: list[str],
) -> dict[str, pd.DataFrame]:
    """Build ``[stock x date]`` frames for the capital-flow features.

    Any failure (e.g. the tables do not exist yet) degrades to neutral
    frames, keeping the loader independent of this data source.
    """

    frames = {
        name: pd.DataFrame(np.nan, index=ts_codes, columns=dates)
        for name in EXTERNAL_FACTOR_NAMES
    }
    if not ts_codes:
        return frames
    quoted = sql_quoted_list(ts_codes)
    try:
        with AshareDB(config.duckdb_path, read_only=True) as db:
            margin = db.query(
                f"SELECT ts_code, trade_date, rzye FROM {config.margin_table} "
                f"WHERE ts_code IN ({quoted})"
            )
            sw_index = db.query(
                f"SELECT index_code, trade_date, close FROM {config.sw_index_table}"
            )
        industry_map = _industry_map(config)
    except Exception as exc:  # noqa: BLE001 - tables may not exist yet.
        logger.warning(f"Capital-flow frames unavailable: {exc}")
        return frames

    if not margin.empty:
        wide = (
            margin.pivot_table(
                index="ts_code", columns="trade_date", values="rzye", aggfunc="last"
            )
            .reindex(index=ts_codes, columns=dates)
        )
        with np.errstate(all="ignore"):
            change = wide / wide.shift(_MARGIN_WINDOW, axis=1) - 1.0
        frames["MARGIN_BALANCE_CHG"] = change.replace([np.inf, -np.inf], np.nan)
        frames["MARGIN_CROWD_60"] = _margin_crowd(wide, _MARGIN_CROWD_WINDOW)

    if industry_map and not sw_index.empty:
        index_wide = sw_index.pivot_table(
            index="index_code", columns="trade_date", values="close", aggfunc="last"
        ).reindex(columns=dates)
        with np.errstate(all="ignore"):
            industry_ret = index_wide / index_wide.shift(_INDUSTRY_WINDOW, axis=1) - 1.0
        for position, code in enumerate(ts_codes):
            industry = industry_map.get(code)
            if industry is not None and industry in industry_ret.index:
                frames["INDUSTRY_MOMENTUM"].iloc[position] = industry_ret.loc[
                    industry
                ].to_numpy(dtype=float)

    return frames
