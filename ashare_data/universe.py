"""Point-in-time universe domain model, contract and production-data gate.

The production universe has three independent sources of truth:

* index membership intervals from ``constituents``;
* listing dates from ``stocks.list_date``;
* trading sessions from ``trade_calendar`` rows whose ``is_open`` is true.

Current constituent snapshots are useful for deciding which symbols to sync,
but they are not historical intervals and are never accepted by this module.
The only fallback is an explicit, in-process development opt-in; environment
variables, pytest detection and call-stack inspection are deliberately absent.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import IntFlag
from numbers import Integral
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DataConfig
from .db import AshareDB
from .processor import is_valid_a_share_code


MEMBERSHIP_COLUMNS = ("index_code", "ts_code", "in_date", "out_date")
STOCK_COLUMNS = ("ts_code", "list_date")
CALENDAR_COLUMNS = ("trade_date", "is_open")
OPEN_ENDED_OUT_DATE = "99991231"


class UniverseContractError(ValueError):
    """The database cannot support a point-in-time production run."""


class UniverseDevelopmentFallbackWarning(UserWarning):
    """An explicitly requested non-production universe fallback is active."""


class UniverseReason(IntFlag):
    """Auditable reasons for a stock/date universe decision."""

    NOT_MEMBER = 1 << 0
    NOT_YET_LISTED = 1 << 1
    LISTING_AGE_INSUFFICIENT = 1 << 2
    STATUS_UNKNOWN = 1 << 3
    MISSING_BAR = 1 << 4


@dataclass(frozen=True)
class UniversePolicy:
    """Pure eligibility policy applied to point-in-time source records."""

    index_codes: tuple[str, ...]
    min_listed_sessions: int
    membership_end_inclusive: bool


@dataclass(frozen=True)
class UniverseMask:
    """Immutable ``[stock, signal-date]`` eligibility and audit reasons."""

    eligible: np.ndarray
    reasons: np.ndarray

    def __post_init__(self) -> None:
        eligible = np.array(self.eligible, dtype=np.bool_, copy=True)
        reasons = np.array(self.reasons, dtype=np.uint16, copy=True)
        if eligible.ndim != 2:
            raise UniverseContractError("eligible must be a two-dimensional array")
        if reasons.ndim != 2:
            raise UniverseContractError("reasons must be a two-dimensional array")
        if eligible.shape != reasons.shape:
            raise UniverseContractError(
                "eligible and reasons must have the same [stock, date] shape"
            )
        eligible.setflags(write=False)
        reasons.setflags(write=False)
        object.__setattr__(self, "eligible", eligible)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class UniverseContractStatus:
    """Machine-assertable provenance for the resolved universe."""

    mode: str
    strict: bool
    degraded: bool
    membership_source: str
    session_source: str
    warnings: tuple[str, ...]
    constituent_rows: int
    stock_rows: int
    open_sessions: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ResolvedUniverse:
    """Validated membership/listing data and the authoritative session axis."""

    constituents: pd.DataFrame
    stocks: pd.DataFrame
    sessions: list[str]
    codes: list[str]
    status: UniverseContractStatus


def dev_fallback_membership_records(
    index_code: str, contract: ResolvedUniverse
) -> list[dict]:
    """All-period membership records for the explicit development fallback.

    Single definition shared by the data loader and the production gates
    (arch-review F4): every configured code is recorded as a member of
    ``index_code`` from the first session, open-ended.  The fallback is
    in-memory only, never persisted, and its degraded provenance stays
    visible through ``UniverseContractStatus.degraded``.
    """

    return [
        {
            "index_code": str(index_code),
            "ts_code": code,
            "in_date": contract.sessions[0],
            "out_date": OPEN_ENDED_OUT_DATE,
        }
        for code in contract.codes
    ]


def _normalize_date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().replace("-", "")
    return text


def _valid_date(value: object) -> bool:
    text = _normalize_date(value)
    if len(text) != 8 or not text.isdigit():
        return False
    try:
        datetime.strptime(text, "%Y%m%d")
    except ValueError:
        return False
    return True


def _require_date(value: object, field: str) -> str:
    normalized = _normalize_date(value)
    if not _valid_date(normalized):
        raise UniverseContractError(
            f"{field} must be a valid date in YYYYMMDD form: {value!r}"
        )
    return normalized


def _normalize_date_axis(values: Iterable[object], field: str) -> list[str]:
    try:
        raw_values = list(values)
    except TypeError as exc:
        raise UniverseContractError(f"{field} must be an iterable of dates") from exc
    normalized = [
        _require_date(value, f"{field}[{position}]")
        for position, value in enumerate(raw_values)
    ]
    if len(set(normalized)) != len(normalized):
        raise UniverseContractError(f"{field} contains duplicate normalized dates")
    return normalized


def _validate_policy(policy: UniversePolicy) -> tuple[str, ...]:
    if not isinstance(policy, UniversePolicy):
        raise UniverseContractError("policy must be a UniversePolicy")
    if not isinstance(policy.index_codes, tuple) or not policy.index_codes:
        raise UniverseContractError("policy.index_codes must be a non-empty tuple")
    index_codes = tuple(str(code).strip() for code in policy.index_codes)
    if any(not code for code in index_codes):
        raise UniverseContractError("policy.index_codes cannot contain blank values")
    if len(set(index_codes)) != len(index_codes):
        raise UniverseContractError("policy.index_codes cannot contain duplicates")
    if (
        isinstance(policy.min_listed_sessions, bool)
        or not isinstance(policy.min_listed_sessions, Integral)
        or policy.min_listed_sessions < 0
    ):
        raise UniverseContractError(
            "policy.min_listed_sessions must be a non-negative integer"
        )
    if not isinstance(policy.membership_end_inclusive, bool):
        raise UniverseContractError(
            "policy.membership_end_inclusive must be a boolean"
        )
    return index_codes


def _normalize_codes(ts_codes: Iterable[str]) -> list[str]:
    try:
        codes = [str(code).strip() for code in ts_codes]
    except TypeError as exc:
        raise UniverseContractError("ts_codes must be an iterable") from exc
    if any(not code for code in codes):
        raise UniverseContractError("ts_codes cannot contain blank values")
    if len(set(codes)) != len(codes):
        raise UniverseContractError("ts_codes cannot contain duplicates")
    return codes


def _exclusive_membership_end(out_date: str, inclusive: bool) -> str:
    if not inclusive or out_date == OPEN_ENDED_OUT_DATE:
        return out_date
    return (
        datetime.strptime(out_date, "%Y%m%d") + timedelta(days=1)
    ).strftime("%Y%m%d")


def _normalize_membership_intervals(
    constituents: Iterable[Mapping[str, object]] | pd.DataFrame,
    *,
    end_inclusive: bool,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    if isinstance(constituents, pd.DataFrame):
        missing = _missing_columns(constituents, MEMBERSHIP_COLUMNS)
        if missing:
            raise UniverseContractError(
                "constituents missing required columns: " + ", ".join(missing)
            )
        raw_rows: list[object] = constituents.to_dict("records")
    else:
        try:
            raw_rows = list(constituents)
        except TypeError as exc:
            raise UniverseContractError(
                "constituents must be a DataFrame or iterable of mappings"
            ) from exc

    intervals: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for position, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise UniverseContractError(
                f"constituents[{position}] must be a mapping"
            )
        missing = [column for column in MEMBERSHIP_COLUMNS if column not in raw_row]
        if missing:
            raise UniverseContractError(
                f"constituents[{position}] missing required fields: "
                + ", ".join(missing)
            )
        index_code = str(raw_row["index_code"]).strip()
        ts_code = str(raw_row["ts_code"]).strip()
        if not index_code or not ts_code:
            raise UniverseContractError(
                f"constituents[{position}] index_code and ts_code cannot be blank"
            )
        in_date = _require_date(
            raw_row["in_date"], f"constituents[{position}].in_date"
        )
        provider_out_date = _require_date(
            raw_row["out_date"], f"constituents[{position}].out_date"
        )
        if end_inclusive:
            invalid = provider_out_date < in_date
        else:
            invalid = provider_out_date <= in_date
        if invalid:
            interval_kind = "inclusive" if end_inclusive else "half-open"
            raise UniverseContractError(
                f"constituents[{position}] has an invalid {interval_kind} interval: "
                f"{in_date} to {provider_out_date}"
            )
        out_date = _exclusive_membership_end(
            provider_out_date, end_inclusive
        )
        intervals.setdefault((index_code, ts_code), []).append(
            (in_date, out_date)
        )

    for (index_code, ts_code), values in intervals.items():
        values.sort()
        previous_out: str | None = None
        for in_date, out_date in values:
            if previous_out is not None and in_date < previous_out:
                raise UniverseContractError(
                    "duplicate or overlapping constituent intervals for "
                    f"({index_code}, {ts_code}); {in_date} < {previous_out}"
                )
            previous_out = out_date
    return intervals


def _normalize_bar_presence(
    bar_presence: object,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    raw = np.asarray(bar_presence)
    if raw.shape != expected_shape:
        raise UniverseContractError(
            "bar_presence must have [stock, date] shape "
            f"{expected_shape}, got {raw.shape}"
        )
    if raw.dtype != np.bool_:
        try:
            boolean_like = np.isin(raw, (0, 1)).all()
        except TypeError as exc:
            raise UniverseContractError(
                "bar_presence values must be boolean or zero/one"
            ) from exc
        if not boolean_like:
            raise UniverseContractError(
                "bar_presence values must be boolean or zero/one"
            )
    return raw.astype(np.bool_, copy=False)


def build_universe_mask(
    ts_codes: Iterable[str],
    signal_dates: Iterable[object],
    open_sessions: Iterable[object],
    constituents: Iterable[Mapping[str, object]] | pd.DataFrame,
    list_dates: Mapping[str, object],
    bar_presence: object,
    policy: UniversePolicy,
) -> UniverseMask:
    """Build a pure, auditable point-in-time eligibility mask.

    Membership is evaluated as the union of ``policy.index_codes``.  Provider
    inclusive end dates are converted once to the module's canonical half-open
    form.  ``STATUS_UNKNOWN`` records the absence of dated ST history and is
    non-blocking; malformed supplied listing dates are rejected.
    """

    index_codes = set(_validate_policy(policy))
    codes = _normalize_codes(ts_codes)
    dates = _normalize_date_axis(signal_dates, "signal_dates")
    sessions = _normalize_date_axis(open_sessions, "open_sessions")
    if not sessions:
        raise UniverseContractError("open_sessions cannot be empty")
    session_set = set(sessions)
    non_sessions = [date for date in dates if date not in session_set]
    if non_sessions:
        raise UniverseContractError(
            "signal_dates contains date(s) absent from open_sessions: "
            + ", ".join(non_sessions[:3])
        )
    sorted_sessions = np.asarray(sorted(sessions), dtype="U8")
    signal_axis = np.asarray(dates, dtype="U8")
    presence = _normalize_bar_presence(
        bar_presence, (len(codes), len(dates))
    )
    if not isinstance(list_dates, Mapping):
        raise UniverseContractError("list_dates must be a mapping by ts_code")

    all_intervals = _normalize_membership_intervals(
        constituents,
        end_inclusive=policy.membership_end_inclusive,
    )
    selected_intervals: dict[str, list[tuple[str, str]]] = {}
    for (index_code, ts_code), values in all_intervals.items():
        if index_code in index_codes:
            selected_intervals.setdefault(ts_code, []).extend(values)

    # No dated ST-status source exists yet.  The current ``stocks.is_st``
    # snapshot cannot prove any historical cell's status, regardless of
    # whether its current value is true or false.  Keep that uncertainty as a
    # non-blocking audit bit until a proper status-history source is added.
    reasons = np.full(
        (len(codes), len(dates)),
        np.uint16(UniverseReason.STATUS_UNKNOWN),
        dtype=np.uint16,
    )
    not_member_value = np.uint16(UniverseReason.NOT_MEMBER)
    not_yet_listed_value = np.uint16(UniverseReason.NOT_YET_LISTED)
    insufficient_age_value = np.uint16(
        UniverseReason.LISTING_AGE_INSUFFICIENT
    )
    missing_bar_value = np.uint16(UniverseReason.MISSING_BAR)

    for row, code in enumerate(codes):
        member = np.zeros(len(dates), dtype=np.bool_)
        for in_date, out_date in selected_intervals.get(code, ()):
            member |= (signal_axis >= in_date) & (signal_axis < out_date)
        reasons[row, ~member] |= not_member_value

        raw_list_date = list_dates.get(code)
        normalized_list_date = _normalize_date(raw_list_date)
        if normalized_list_date:
            list_date = _require_date(
                normalized_list_date, f"list_dates[{code!r}]"
            )
            not_yet_listed = signal_axis < list_date
            reasons[row, not_yet_listed] |= not_yet_listed_value
            listed_session_start = int(
                np.searchsorted(sorted_sessions, list_date, side="left")
            )
            session_positions = np.searchsorted(
                sorted_sessions, signal_axis, side="right"
            )
            listed_session_age = session_positions - listed_session_start
            insufficient_age = (
                ~not_yet_listed
                & (listed_session_age < int(policy.min_listed_sessions))
            )
            reasons[row, insufficient_age] |= insufficient_age_value

    reasons[~presence] |= missing_bar_value
    blocking_reasons = np.uint16(
        UniverseReason.NOT_MEMBER
        | UniverseReason.NOT_YET_LISTED
        | UniverseReason.LISTING_AGE_INSUFFICIENT
        | UniverseReason.MISSING_BAR
    )
    eligible = (reasons & blocking_reasons) == 0
    return UniverseMask(eligible=eligible, reasons=reasons)


def _missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [column for column in required if column not in frame.columns]


def membership_interval_issues(frame: pd.DataFrame) -> list[str]:
    """Return structural/date/overlap errors for constituent intervals.

    Overlap is checked per ``(index_code, ts_code)``.  Therefore the same
    stock may belong to different indices concurrently, and adjacent
    half-open intervals ``[a,b)`` and ``[b,c)`` are valid.
    """

    missing = _missing_columns(frame, MEMBERSHIP_COLUMNS)
    if missing:
        return [
            "constituents missing required columns: " + ", ".join(missing)
        ]
    if frame.empty:
        return ["constituents contains no historical membership intervals"]

    work = frame.loc[:, MEMBERSHIP_COLUMNS].copy()
    issues: list[str] = []
    for column in MEMBERSHIP_COLUMNS:
        blank = work[column].isna() | work[column].astype(str).str.strip().eq("")
        if blank.any():
            issues.append(
                f"constituents.{column} has {int(blank.sum())} missing value(s)"
            )
    if issues:
        return issues

    work["index_code"] = work["index_code"].astype(str).str.strip()
    work["ts_code"] = work["ts_code"].astype(str).str.strip()
    for column in ("in_date", "out_date"):
        invalid = ~work[column].map(_valid_date)
        if invalid.any():
            examples = ", ".join(work.loc[invalid, column].astype(str).head(3))
            issues.append(
                f"constituents.{column} has invalid YYYYMMDD value(s): {examples}"
            )
        work[column] = work[column].map(_normalize_date)
    if issues:
        return issues

    backwards = work["out_date"] <= work["in_date"]
    if backwards.any():
        row = work.loc[backwards].iloc[0]
        issues.append(
            "constituent interval must be non-empty and half-open: "
            f"({row.index_code}, {row.ts_code}) "
            f"[{row.in_date}, {row.out_date})"
        )

    for (index_code, ts_code), group in work.groupby(
        ["index_code", "ts_code"], sort=True
    ):
        ordered = group.sort_values(["in_date", "out_date"])
        previous_out: str | None = None
        for row in ordered.itertuples(index=False):
            if previous_out is not None and row.in_date < previous_out:
                issues.append(
                    "overlapping constituent intervals for "
                    f"({index_code}, {ts_code}); {row.in_date} < {previous_out}"
                )
                break
            previous_out = max(previous_out or row.out_date, row.out_date)
    return issues


def validate_membership_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize valid intervals or raise a clear persistence error."""

    issues = membership_interval_issues(frame)
    if issues:
        raise UniverseContractError("; ".join(issues))
    normalized = frame.loc[:, MEMBERSHIP_COLUMNS].copy()
    for column in ("index_code", "ts_code"):
        normalized[column] = normalized[column].astype(str).str.strip()
    for column in ("in_date", "out_date"):
        normalized[column] = normalized[column].map(_normalize_date)
    return normalized


def _read_table(db: AshareDB, table: str) -> tuple[pd.DataFrame, str | None]:
    try:
        return db.query(f"SELECT * FROM {table}"), None
    except Exception as exc:  # noqa: BLE001 - converted to a contract error.
        return pd.DataFrame(), f"required table {table} is unavailable: {exc}"


def _snapshot_shaped(
    constituents: pd.DataFrame,
    sessions: list[str],
) -> bool:
    """Detect the legacy 'current snapshot stretched over history' shape."""

    if len(constituents) < 2 or not sessions:
        return False
    if _missing_columns(constituents, MEMBERSHIP_COLUMNS):
        return False
    in_dates = constituents["in_date"].map(_normalize_date)
    out_dates = constituents["out_date"].map(_normalize_date)
    return (
        in_dates.nunique(dropna=False) == 1
        and out_dates.nunique(dropna=False) == 1
        and out_dates.iloc[0] == OPEN_ENDED_OUT_DATE
        and in_dates.iloc[0] <= sessions[0]
    )


def _resolve_calendar(
    calendar: pd.DataFrame,
    table_error: str | None,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    if table_error:
        return [], [table_error]
    missing = _missing_columns(calendar, CALENDAR_COLUMNS)
    if missing:
        return [], ["trade_calendar missing required columns: " + ", ".join(missing)]
    open_rows = calendar.loc[calendar["is_open"].fillna(False).astype(bool)].copy()
    invalid = ~open_rows["trade_date"].map(_valid_date)
    if invalid.any():
        issues.append("trade_calendar.is_open=True contains invalid trade_date values")
    sessions = sorted(
        {
            _normalize_date(value)
            for value in open_rows.loc[~invalid, "trade_date"].tolist()
        }
    )
    if not sessions:
        issues.append("trade_calendar has no rows with is_open=True")
    return sessions, issues


def resolve_universe_contract(
    config: DataConfig,
    *,
    allow_development_fallback: bool = False,
) -> ResolvedUniverse:
    """Resolve the PIT universe, strict unless explicitly opted into dev mode."""

    with AshareDB(config.duckdb_path, read_only=True) as db:
        constituents, constituent_table_error = _read_table(
            db, config.constituents_table
        )
        stocks, stock_table_error = _read_table(db, config.stocks_table)
        calendar, calendar_table_error = _read_table(db, config.calendar_table)

    sessions, calendar_issues = _resolve_calendar(calendar, calendar_table_error)
    if calendar_issues:
        raise UniverseContractError(
            "production universe contract violation: " + "; ".join(calendar_issues)
        )

    issues: list[str] = []
    if constituent_table_error:
        issues.append(constituent_table_error)
    else:
        issues.extend(membership_interval_issues(constituents))

    if stock_table_error:
        issues.append(stock_table_error)
    else:
        missing = _missing_columns(stocks, STOCK_COLUMNS)
        if missing:
            issues.append("stocks missing required columns: " + ", ".join(missing))

    configured = {str(code) for code in config.index_codes}
    usable_constituents = constituents.copy()
    if not _missing_columns(usable_constituents, MEMBERSHIP_COLUMNS):
        for column in ("in_date", "out_date"):
            usable_constituents[column] = usable_constituents[column].map(
                _normalize_date
            )
        usable_constituents = usable_constituents[
            usable_constituents["index_code"].astype(str).isin(configured)
        ].copy()
        present = set(usable_constituents["index_code"].astype(str))
        missing_indices = sorted(configured - present)
        if missing_indices:
            issues.append(
                "constituents missing configured index_code interval(s): "
                + ", ".join(missing_indices)
            )
        if _snapshot_shaped(usable_constituents, sessions):
            issues.append(
                "constituents looks like a current snapshot stretched across "
                "the full history; current snapshots are not PIT intervals"
            )

    member_codes = (
        {
            str(code)
            for code in usable_constituents.get("ts_code", pd.Series(dtype=str))
            if is_valid_a_share_code(str(code))
        }
        if not usable_constituents.empty
        else set()
    )

    usable_stocks = stocks.copy()
    if not _missing_columns(usable_stocks, STOCK_COLUMNS):
        usable_stocks["ts_code"] = usable_stocks["ts_code"].astype(str)
        usable_stocks["list_date"] = usable_stocks["list_date"].map(_normalize_date)
        stock_codes = set(usable_stocks["ts_code"])
        missing_stock_rows = sorted(member_codes - stock_codes)
        if missing_stock_rows:
            issues.append(
                "constituent ts_code missing from stocks: "
                + ", ".join(missing_stock_rows[:5])
            )
        relevant = usable_stocks[usable_stocks["ts_code"].isin(member_codes)]
        invalid_list = ~relevant["list_date"].map(_valid_date)
        if invalid_list.any():
            examples = ", ".join(
                relevant.loc[invalid_list, "ts_code"].astype(str).head(5)
            )
            issues.append(
                "stocks.list_date missing or invalid for constituent ts_code(s): "
                + examples
            )

        membership_for_list = usable_constituents[
            usable_constituents["ts_code"].astype(str).isin(member_codes)
        ]
        if not membership_for_list.empty and not invalid_list.any():
            listed = relevant.set_index("ts_code")["list_date"].to_dict()
            before_listing = membership_for_list.apply(
                lambda row: listed.get(str(row["ts_code"]), OPEN_ENDED_OUT_DATE)
                > str(row["in_date"]),
                axis=1,
            )
            if before_listing.any():
                row = membership_for_list.loc[before_listing].iloc[0]
                issues.append(
                    "constituent interval begins before stocks.list_date: "
                    f"{row['ts_code']} in_date={row['in_date']} "
                    f"list_date={listed.get(str(row['ts_code']))}"
                )

    if issues and not allow_development_fallback:
        raise UniverseContractError(
            "production universe contract violation: " + "; ".join(dict.fromkeys(issues))
        )

    degraded = bool(issues)
    warning_messages = tuple(dict.fromkeys(issues)) if degraded else ()
    if degraded:
        warning = (
            "development universe fallback enabled; using an all-period "
            "membership that is forbidden for production: "
            + "; ".join(warning_messages)
        )
        warnings.warn(warning, UniverseDevelopmentFallbackWarning, stacklevel=2)
        fallback_codes = member_codes
        if not fallback_codes and not _missing_columns(usable_stocks, ("ts_code",)):
            fallback_codes = {
                str(code)
                for code in usable_stocks["ts_code"]
                if is_valid_a_share_code(str(code))
            }
        codes = sorted(fallback_codes)
    else:
        codes = sorted(member_codes)

    status = UniverseContractStatus(
        mode="development_fallback" if degraded else "strict",
        strict=not degraded,
        degraded=degraded,
        membership_source=(
            "development_all_period" if degraded else "constituents_pit_intervals"
        ),
        session_source=f"{config.calendar_table}.is_open=True",
        warnings=warning_messages,
        constituent_rows=len(usable_constituents),
        stock_rows=len(usable_stocks),
        open_sessions=len(sessions),
    )
    return ResolvedUniverse(
        constituents=usable_constituents,
        stocks=usable_stocks,
        sessions=sessions,
        codes=codes,
        status=status,
    )


def member_bar_coverage(db: AshareDB, config: DataConfig) -> pd.DataFrame:
    """Per-membership-interval daily-bar coverage (survivorship audit).

    Returns one row per ``constituents`` interval with ``bars`` (daily-bar
    rows inside the interval), ``sessions`` (open calendar sessions in the
    same span, 0 for intervals entirely beyond the data), and
    ``coverage = bars / sessions`` (NaN when ``sessions`` is 0).

    Interval semantics are the module's canonical **half-open**
    ``[in_date, out_date)`` form (provider inclusive end dates are already
    converted), and both counts are capped to the daily-bar horizon
    ``[min(trade_date), max(trade_date)]``: sessions outside the data can
    never fabricate a zero-bar interval (e.g. an interval entirely before
    the first synced bar is not auditable with the current data).  An
    interval that *overlaps* the horizon but has open sessions and zero
    bars is the signature of a historical member that was never synced —
    the current-snapshot sync universe silently dropped delisted members
    and the PIT mask then marked them MISSING_BAR, biasing every
    historical backtest optimistically.  The production gate (G7 in
    ``ashare_data.gates``) fails on those intervals.
    """

    horizon = db.query(
        f"SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx "
        f"FROM {config.daily_table}"
    ).iloc[0]
    if pd.isna(horizon["mn"]):
        frame = db.query(
            f"SELECT index_code, ts_code, in_date, out_date, "
            f"0 AS bars, 0 AS sessions FROM {config.constituents_table}"
        )
        frame["coverage"] = np.nan
        return frame
    min_bar = str(horizon["mn"])
    max_bar = str(horizon["mx"])
    sql = f"""
        SELECT c.index_code, c.ts_code, c.in_date, c.out_date,
               COUNT(d.trade_date) AS bars,
               (SELECT COUNT(*)
                FROM {config.calendar_table} k
                WHERE k.is_open = true
                  AND k.trade_date >= GREATEST(c.in_date, '{min_bar}')
                  AND k.trade_date < LEAST(c.out_date, '{max_bar}')
               ) AS sessions
        FROM {config.constituents_table} c
        LEFT JOIN {config.daily_table} d
          ON d.ts_code = c.ts_code
         AND d.trade_date >= GREATEST(c.in_date, '{min_bar}')
         AND d.trade_date < LEAST(c.out_date, '{max_bar}')
        GROUP BY c.index_code, c.ts_code, c.in_date, c.out_date
        ORDER BY c.ts_code, c.in_date
    """
    frame = db.query(sql)
    frame["coverage"] = frame["bars"].astype(float) / frame["sessions"].replace(0, np.nan)
    return frame


def require_production_universe(config: DataConfig) -> UniverseContractStatus:
    """Non-bypassable helper for formal training/evaluation entry points."""
    return resolve_universe_contract(
        config, allow_development_fallback=False
    ).status
