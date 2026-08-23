"""Pure helpers for the point-in-time universe import.

The network/DB orchestration lives in ``scripts/import_pit_universe.py``;
everything here is side-effect-free and unit-tested (the script itself
had zero test coverage before this split).
"""

from __future__ import annotations

import pandas as pd


def merge_list_dates(stocks: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    """Merge fetched listing dates into the stocks table (vectorized).

    A fetched non-null ``list_date`` wins for overlapping codes (the
    exchange batch metadata is authoritative for listed stocks); a fetched
    NaN never overwrites an existing value; codes absent from the table
    are appended with neutral display defaults (delisted members
    backfilled by the provider); codes absent from ``fetched`` are
    untouched.  Single O(n) merge — the previous implementation
    concatenated one row at a time inside a per-row loop.
    """

    merged = stocks.merge(
        fetched[["ts_code", "list_date"]],
        on="ts_code",
        how="outer",
        suffixes=("", "_new"),
    )
    merged["list_date"] = merged["list_date_new"].fillna(merged["list_date"])
    merged = merged.drop(columns=["list_date_new"])
    if "name" in merged.columns:
        merged["name"] = merged["name"].fillna(merged["ts_code"])
    if "is_st" in merged.columns:
        merged["is_st"] = merged["is_st"].fillna(False)
    return merged


def bs_code_to_ts_code(code: str) -> str:
    """BaoStock ``sh.600000`` / ``sz.000001`` -> Tushare-style code."""

    market, symbol = code.split(".")
    suffix = "SH" if market == "sh" else "SZ" if market == "sz" else None
    if suffix is None:
        raise ValueError(f"unexpected baostock code format: {code!r}")
    return f"{symbol}.{suffix}"


def month_end_samples(sessions: list[str]) -> list[str]:
    """One sample per calendar month plus the final session.

    The 12 months before the first session are included too, so the first
    dataset session is covered by a prior adjustment snapshot (an interval
    can only begin at a provider update date).
    """

    samples: dict[str, str] = {}
    for session in sessions:
        samples[session[:6]] = session
    pre_year = int(sessions[0][:4]) - 1
    for month in range(1, 13):
        key = f"{pre_year}{month:02d}28"
        samples[key] = key
    ordered = [samples[key] for key in sorted(samples)]
    if ordered and ordered[-1] != sessions[-1]:
        ordered.append(sessions[-1])
    return ordered


def timeline_to_intervals(
    dates: list[str], present: list[bool]
) -> list[tuple[str, str]]:
    """Compress a sampled membership timeline into half-open intervals.

    ``dates`` carries the effective provider update date of each sample;
    a run of True samples becomes ``[first_update, next_update)`` and the
    trailing open run ends at ``99991231``.
    """

    intervals: list[tuple[str, str]] = []
    start: str | None = None
    for date, is_member in zip(dates, present):
        if is_member and start is None:
            start = date
        elif not is_member and start is not None:
            intervals.append((start, date))
            start = None
    if start is not None:
        intervals.append((start, "99991231"))
    return intervals
