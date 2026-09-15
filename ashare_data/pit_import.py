"""Pure helpers for the point-in-time universe import.

The network/DB orchestration lives in ``scripts/import_pit_universe.py``;
everything here is side-effect-free and unit-tested (the script itself
had zero test coverage before this split).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import pandas as pd

from .universe import (
    DateAxisFormatError,
    _normalize_date,
    _valid_date,
    is_canonical_date,
)


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


# --- B01 (p29 §3): import-boundary date canon (I1/I2), the month-end
# --- snapshot as-of rule (I5) and the four-time semantics declaration (I3/I4).

__all__ = [
    "DateAxisFormatError",
    "SourceTimeSemantics",
    "DAILY_PRICE_FAMILY_TIME_SEMANTICS",
    "INDEX_WEIGHT_TIME_SEMANTICS",
    "asof_winner_intervals",
    "assert_comparable_date_axes",
    "bs_code_to_ts_code",
    "merge_list_dates",
    "month_end_samples",
    "month_end_snapshot_intervals",
    "normalize_import_date",
    "timeline_to_intervals",
    "validate_canonical_date_column",
    "validate_time_semantics",
]

#: B01 I4: revision-history completeness is declared, never assumed from
#: the presence of ``ann_date``.  ``unrestorable``/incomplete families must
#: carry their Tier limit note (p2 A/B/C semantics preserved).
_REVISION_SEMANTICS = ("versioned_rows", "single_final", "unrestorable")


def normalize_import_date(value: object) -> str:
    """The I1 explicit record-and-fix primitive: one date value to the
    canonical compact ``YYYYMMDD`` form.  ISO strings normalize (the
    caller must record the fix — the B01 report's format audit does);
    anything that is not a real calendar date raises.  NULL/NaN maps to
    ``""`` (absence is not a format violation)."""

    normalized = _normalize_date(value)
    if not normalized:
        return ""
    if not _valid_date(normalized):
        raise DateAxisFormatError(
            f"not a real calendar date in YYYYMMDD form: {value!r}"
        )
    return normalized


def validate_canonical_date_column(
    values: list[object], *, table: str, field: str
) -> list[str]:
    """The I2 machine assertion over one date column: every non-null
    value must ALREADY be canonical compact ``YYYYMMDD``.  Offenders are
    named and the check fails closed — mixed formats never pass through
    into a comparison (p29 §1.1: ISO rows silently vanished from the
    mask/presence alignment).  Returns the values unchanged when clean;
    the ISO→compact fix primitive is :func:`normalize_import_date`."""

    offenders = [value for value in values if not is_canonical_date(value)]
    if offenders:
        raise DateAxisFormatError(
            f"{table}.{field} must be canonical YYYYMMDD before any "
            f"interval comparison (B01 I1/I2); "
            f"{len(offenders)} offender(s), examples: {offenders[:3]}"
        )
    return [normalize_import_date(value) for value in values]


def assert_comparable_date_axes(*axes: list[object], field: str = "date_axis") -> None:
    """I2: every axis participating in an interval comparison must be
    100% canonical (the machine assertion p29 §3/I2 requires).  Fails
    closed before the comparison happens."""

    for position, axis in enumerate(axes):
        validate_canonical_date_column(
            list(axis), table=f"{field}[{position}]", field=field,
        )


def month_end_snapshot_intervals(
    snapshot_dates: list[str], open_sessions: list[str]
) -> list[tuple[str, str, str]]:
    """The I5 machine rule: month-end snapshots → daily visible intervals.

    Preregistered rule (p29 §3/I5): a month-end snapshot becomes visible
    at the **first open session strictly after the snapshot date** (a
    non-trading month end is therefore deferred to the next open
    session — the same deterministic treatment for both cases), and
    stays visible until the next visible snapshot's first session
    (half-open ``[visible_from, visible_to)``; the trailing interval
    ends at the ``99991231`` open sentinel).  A snapshot with no
    following open session produces no interval (it is never visible
    within the calendar horizon).  Both axes must be canonical (I2) and
    the snapshots strictly ascending/unique.
    """

    snaps = list(snapshot_dates)
    sessions = list(open_sessions)
    assert_comparable_date_axes(snaps, sessions, field="month_end_snapshot")
    if len(set(snaps)) != len(snaps) or snaps != sorted(snaps):
        raise ValueError(
            "snapshot_dates must be strictly ascending and unique: "
            f"{snaps[:5]}"
        )
    # The bisect lookups below PRECONDITION a strictly ascending session
    # axis — unordered/duplicated sessions fail closed instead of silently
    # mis-interval-ing (B01 review F3).
    if len(set(sessions)) != len(sessions) or sessions != sorted(sessions):
        raise ValueError(
            "open_sessions must be strictly ascending and unique: "
            f"{sessions[:5]}"
        )
    visible_from: dict[str, str] = {}
    for snap in snaps:
        position = bisect.bisect_right(sessions, snap)
        if position >= len(sessions):
            continue  # never visible within the calendar horizon
        visible_from[snap] = sessions[position]
    ordered = [snap for snap in snaps if snap in visible_from]
    intervals: list[tuple[str, str, str]] = []
    for index, snap in enumerate(ordered):
        visible_to = (
            visible_from[ordered[index + 1]]
            if index + 1 < len(ordered)
            else "99991231"
        )
        intervals.append((snap, visible_from[snap], visible_to))
    return intervals


@dataclass(frozen=True)
class SourceTimeSemantics:
    """B01 I3: the machine declaration of one source family's four time
    semantics — (a) economic valid time, (b) announcement/availability
    time, (c) actual collection time (the single ``source_ingest_runs``
    ledger), (d) revision versions.  ``revision_history_complete=False``
    requires a non-empty ``tier_limit_note`` (I4: an unrestorable
    revision history keeps its p2 Tier limit; ``ann_date`` presence never
    implies a complete restatement history)."""

    family: str
    tables: tuple[str, ...]
    economic_time: str
    availability_time: str | None
    collection_authority: str
    revision_semantics: str
    revision_history_complete: bool
    tier_limit_note: str
    visibility_rule: str


def validate_time_semantics(decl: SourceTimeSemantics) -> None:
    """Fail-closed validation of one four-time declaration (I3/I4)."""

    if not isinstance(decl, SourceTimeSemantics):
        raise ValueError("declaration must be a SourceTimeSemantics")
    if not decl.family or not decl.tables or not decl.economic_time:
        raise ValueError(
            f"declaration {decl.family!r}: family/tables/economic_time are "
            "required"
        )
    if not decl.visibility_rule:
        raise ValueError(
            f"declaration {decl.family!r}: the visibility rule must be stated"
        )
    if decl.collection_authority != "source_ingest_runs":
        raise ValueError(
            f"declaration {decl.family!r}: the collection-time authority is "
            f"the source_ingest_runs ledger (B01 I3), got "
            f"{decl.collection_authority!r}"
        )
    if decl.revision_semantics not in _REVISION_SEMANTICS:
        raise ValueError(
            f"declaration {decl.family!r}: unknown revision semantics "
            f"{decl.revision_semantics!r}; expected one of "
            f"{_REVISION_SEMANTICS}"
        )
    if not decl.revision_history_complete and not decl.tier_limit_note:
        raise ValueError(
            f"declaration {decl.family!r}: an incomplete revision history "
            "must carry its Tier limit note (B01 I4)"
        )


#: The daily price family (bars / adjustment / basics / execution
#: constraints).  Cell-complete: absence of a row is an enumerated gap,
#: never a "zero event" (p29 I8 applies to event-style families only).
DAILY_PRICE_FAMILY_TIME_SEMANTICS = SourceTimeSemantics(
    family="daily_price",
    tables=(
        "daily_bar_raw", "daily_adj_factor", "daily_basic_pit",
        "daily_price_limit", "daily_suspension",
    ),
    economic_time="trade_date (the session the row describes)",
    availability_time=(
        "p17 §6.1: daily-family bars of T are available at the next open "
        "session after T; price-limit rows of T at T's own session (p19 "
        "§2.4); a suspension record is available at its own session"
    ),
    collection_authority="source_ingest_runs",
    revision_semantics="single_final",
    revision_history_complete=True,
    tier_limit_note="",
    visibility_rule=(
        "the loader pivots bars/basics by trade_date and constraints onto "
        "the same session axis; a missing price-limit record is fail-closed "
        "degraded (p19 §2.4), never filled"
    ),
)

#: The month-end index-weight snapshot family.  Visibility follows the
#: I5 as-of rule (:func:`month_end_snapshot_intervals`); the writer's
#: ``available_date`` (p17 §6.1 month_end rule) must agree with it.
INDEX_WEIGHT_TIME_SEMANTICS = SourceTimeSemantics(
    family="index_weight",
    tables=("index_weight_pit",),
    economic_time="trade_date (the month-end snapshot date)",
    availability_time=(
        "available_date (writer-computed: the first open session after the "
        "month end, p17 §6.1 'month_end' rule)"
    ),
    collection_authority="source_ingest_runs",
    revision_semantics="single_final",
    revision_history_complete=True,
    tier_limit_note="",
    visibility_rule=(
        "a month-end snapshot is visible [first open session after the "
        "snapshot, next snapshot's first session) — "
        "month_end_snapshot_intervals; the store's available_date must "
        "match the rule (B01 I5 machine contract)"
    ),
)
