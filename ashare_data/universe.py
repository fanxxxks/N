"""Point-in-time universe contract and production-data gate.

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
from dataclasses import asdict, dataclass
from datetime import datetime
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

    def membership_mask(
        self,
        ts_codes: Iterable[str],
        dates: Iterable[str],
    ) -> np.ndarray:
        """Return ``[stock, session]`` eligibility under half-open intervals."""

        codes = [str(code) for code in ts_codes]
        session_dates = [_normalize_date(date) for date in dates]
        mask = np.zeros((len(codes), len(session_dates)), dtype=bool)
        allowed = set(self.codes)

        if self.status.degraded:
            for row, code in enumerate(codes):
                if code in allowed:
                    mask[row, :] = True
            return mask

        listed = {
            str(row.ts_code): str(row.list_date)
            for row in self.stocks.itertuples(index=False)
        }
        intervals: dict[str, list[tuple[str, str]]] = {}
        for row in self.constituents.itertuples(index=False):
            intervals.setdefault(str(row.ts_code), []).append(
                (str(row.in_date), str(row.out_date))
            )
        for row, code in enumerate(codes):
            list_date = listed.get(code)
            if not list_date:
                continue
            for col, date in enumerate(session_dates):
                if date < list_date:
                    continue
                mask[row, col] = any(
                    in_date <= date < out_date
                    for in_date, out_date in intervals.get(code, ())
                )
        return mask


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


def require_production_universe(config: DataConfig) -> UniverseContractStatus:
    """Non-bypassable helper for formal training/evaluation entry points."""

    return resolve_universe_contract(
        config, allow_development_fallback=False
    ).status
