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


# --- D6 (p17 §6.2): the single financial-statement as-of read ------------------

from .config import resolve_v2_duckdb_path  # noqa: E402
from .db import AshareDB, sql_quoted_list  # noqa: E402

_FINANCIAL_STATEMENTS = {
    "income": "income_statement_pit",
    "balance": "balance_sheet_pit",
    "cashflow": "cashflow_statement_pit",
}

_FINANCIAL_VALUE_COLUMNS = {
    "income": [
        "total_revenue", "revenue", "oper_cost", "total_profit",
        "n_income", "n_income_attr_p",
    ],
    "balance": [
        "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
        "total_hldr_eqy_inc_min_int",
    ],
    "cashflow": [
        "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
        "c_cash_equ_end_period",
    ],
}

_ASOF_OPEN_END = "99991231"


def asof_winner_intervals(
    config,
    statement: str,
    ts_codes: list[str],
    *,
    report_type: str = "1",
    comp_type: str = "1",
) -> pd.DataFrame:
    """The single as-of read over one financial version table (p17 §6.2).

    For every business key ``K = (ts_code, end_date, report_type,
    comp_type)`` the *candidate* rows are those with
    ``available_date <= s`` and the *winner* at ``s`` is the maximum by
    ``(f_ann_date, ann_date, available_date, ingest_id)`` — a NULL
    ``f_ann_date`` falls back to ``ann_date`` (which is NOT NULL).  This
    function resolves that per-session winner into per-version winner
    intervals: one row per visible segment, with ``winner_from`` /
    ``winner_to`` (half-open; the trailing segment ends at the open
    sentinel).  ``update_flag`` is never filtered (调整前 rows stay
    auditable), null values stay NULL (never 0), and the three statements
    are read independently — callers must never merge their visibility.

    The reader is the ONLY as-of implementation for the version tables;
    consumers call it, never re-derive (p17 §6.2 rule 5).
    """

    if statement not in _FINANCIAL_STATEMENTS:
        raise ValueError(f"unknown financial statement {statement!r}")
    table = _FINANCIAL_STATEMENTS[statement]
    value_cols = ", ".join(_FINANCIAL_VALUE_COLUMNS[statement])
    v2_path = resolve_v2_duckdb_path(config)
    quoted = sql_quoted_list(ts_codes) if ts_codes else "NULL"
    with AshareDB(v2_path, read_only=True) as db:
        rows = db.query(
            f"SELECT ts_code, end_date, ann_date, f_ann_date, available_date, "
            f"report_type, comp_type, update_flag, ingest_id, {value_cols} "
            f"FROM {table} "
            f"WHERE ts_code IN ({quoted}) AND report_type = '{report_type}' "
            f"AND comp_type = '{comp_type}'"
        )
    empty = rows.iloc[0:0].copy()
    empty["winner_from"] = pd.Series(dtype=object)
    empty["winner_to"] = pd.Series(dtype=object)
    if rows.empty:
        return empty

    rows = rows.copy()
    # NULL f_ann_date falls back to ann_date for the ranking (ann_date is
    # NOT NULL by schema); the emitted columns keep the source values.
    rows["rank_f_ann"] = rows["f_ann_date"].fillna(rows["ann_date"]).astype(str)
    rows = rows.sort_values(
        ["ts_code", "end_date", "rank_f_ann", "ann_date", "available_date",
         "ingest_id"],
        ascending=[True, True, False, False, False, False],
    )
    rows = rows.drop(columns=["rank_f_ann"])

    segments: list[pd.DataFrame] = []
    for _key, group in rows.groupby(["ts_code", "end_date"], sort=False):
        boundary: str | None = None
        for row in group.itertuples(index=False):
            available = str(row.available_date)
            if boundary is not None and available >= boundary:
                continue  # a higher-ranked version already covers this day
            winner_to = boundary if boundary is not None else _ASOF_OPEN_END
            segment = pd.DataFrame([row]).copy()
            segment["winner_from"] = available
            segment["winner_to"] = winner_to
            segments.append(segment)
            boundary = available
    if not segments:
        return empty
    return pd.concat(segments, ignore_index=True)
