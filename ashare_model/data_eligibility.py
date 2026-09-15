"""Per-formula-dependency data eligibility gate (D7; p17 §12).

The P8 ``FACTOR_SET_QUALIFIED`` data check: resolve which p17 §12 data
sources a RunSpec's factor set actually depends on — via the feature
registry's ``pit_level``/``family``, the single per-feature source
authority — and check every required source against the p17 preregistered
thresholds (the constants live in ``ashare_data.gates``, the data-gate
authority).

* **Per-formula-dependency, not global** — a RunSpec is rejected only
  when a source it depends on fails.  A global "all optional sources
  must be fresh" gate is a preregistered NON-goal (p17 §12): unrequired
  sources never affect a verdict and are reported as not evaluated.
* **Not computable never passes** — a missing store/table/manifest or an
  empty source fails its check with an explicit not-computable reason
  (p17 §12: 一切源 来源不可计算一律 fail 不得 pass).
* **Lag semantics** — 滞后 = 最近会话 − 该源最新 available_session, in
  open-session positions of the unique trade calendar (the production
  ``trade_calendar``; p17 §6.1 forbids a second calendar).  The §6.1
  mapping per family: daily-family bars of T are available at the next
  open session after T; price-limit rows of T at T's own session (09:00
  pre-open); margin rows carry the writer-computed T+1
  ``available_date`` (consumed, not re-derived); industry-index
  freshness is the D5 gate's session count.  Data available beyond the
  completed window has lag 0.
* **Reuse, no second implementation** — freshness/coverage inputs come
  from the existing authorities: the D5 industry gates
  (``capital_flow.industry_index_freshness`` /
  ``industry_membership_gate``), the v2 dataset manifest
  (``manifest.latest_manifest`` / ``verify_dataset`` over
  ``TUSHARE_V2_CONFIG``), and the unique calendar.  The financial
  report-period coverage reuses ``pit_import``'s per-table core-column
  semantics (p17 §5.3).

Dependency mapping (RunSpec level + registry level):

* RunSpec level, always required — DAILY (the daily price family feeds
  prices/labels/the universe for every formal run) and PRICE_LIMIT (the
  D4 execution constraints consume ``daily_price_limit`` for every
  formal run) plus the v2 DATASET_MANIFEST identity check;
* ``PIT_DAILY`` features → DAILY; ``PIT_FUNDAMENTAL`` → FINANCIAL;
  ``PIT_CAPITAL`` → MARGIN; ``SNAPSHOT`` industry features →
  INDUSTRY_MEMBERSHIP + INDUSTRY_INDEX; ``NEUTRAL`` → nothing.

Lifecycle wiring (consuming this verdict at the P8
FACTOR_SET_QUALIFIED transition) is E1b/t21's scope; this module
delivers the resolution + checks only.  Dependency direction (§9):
ashare_model consumes ashare_data authorities; no CLI/Web imports.
"""

from __future__ import annotations

import bisect
import calendar as _calendar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import ashare_data.gates as gates
from ashare_data.capital_flow import (
    _membership_intervals,
    _membership_rows,
    INDUSTRY_MEMBERSHIP_TIME_SEMANTICS,
    MARGIN_DETAIL_TIME_SEMANTICS,
    industry_index_freshness,
    industry_membership_gate,
)
from ashare_data.config import resolve_v2_duckdb_path
from ashare_data.db import AshareDB, sql_quoted_list
from ashare_data.fundamentals import FINANCIAL_VERSION_TIME_SEMANTICS
from ashare_data.gates import (
    DAILY_ELIGIBILITY_MAX_LAG_SESSIONS,
    FINANCIAL_REPORT_COVERAGE_MIN,
    INDUSTRY_INDEX_MAX_LAG_SESSIONS,
    INDUSTRY_MEMBERSHIP_MIN_COVERAGE,
    MARGIN_MAX_LAG_SESSIONS,
    PRICE_LIMIT_MAX_LAG_SESSIONS,
)
from ashare_data.manifest import (
    TUSHARE_V2_CONFIG,
    latest_manifest,
    verify_dataset,
)
from ashare_data.pit_import import (
    DAILY_PRICE_FAMILY_TIME_SEMANTICS,
    INDEX_WEIGHT_TIME_SEMANTICS,
    SourceTimeSemantics,
    _FINANCIAL_VALUE_COLUMNS,
    assert_comparable_date_axes,
    month_end_snapshot_intervals,
)
from ashare_data.universe import (
    DateAxisFormatError,
    UniversePolicy,
    expected_observable_units,
    is_canonical_date,
    resolve_universe_contract,
)
from ashare_model.feature_registry import PitLevel, family_of, pit_level_of
from ashare_model.readiness import (
    STATUS_FAIL,
    STATUS_NOT_APPLICABLE,
    STATUS_PASS,
    STATUS_UNKNOWN,
    aggregate_status,
)
from ashare_model.vocab import FORMULA_VOCAB

_SHANGHAI = ZoneInfo("Asia/Shanghai")

#: p17 §12 source families (verdict keys; names = the p17 tables they
#: check, except the industry-index closes and the financial version
#: tables, which span the sw-index table and the three statement tables).
SOURCE_DAILY = "daily_bar_raw"
SOURCE_PRICE_LIMIT = "daily_price_limit"
SOURCE_MARGIN = "margin_detail_pit"
SOURCE_INDUSTRY_MEMBERSHIP = "industry_membership_pit"
SOURCE_INDUSTRY_INDEX = "industry_index"
SOURCE_FINANCIAL = "financial_version_tables"
SOURCE_DATASET_MANIFEST = "dataset_manifest"

#: The daily price family checked as one unit (p17 §13 pulls
#: daily / daily_basic / adj_factor per trade_date together; the family
#: fails closed on its laggard member).
_DAILY_FAMILY_TABLES = ("daily_bar_raw", "daily_adj_factor", "daily_basic_pit")

#: The financial core columns whose report-period coverage §12 measures
#: (p17 §5.3; the same per-table sets the D6 as-of authority uses —
#: single implementation, no second column list).
_FINANCIAL_TABLES = {
    "income_statement_pit": _FINANCIAL_VALUE_COLUMNS["income"],
    "balance_sheet_pit": _FINANCIAL_VALUE_COLUMNS["balance"],
    "cashflow_statement_pit": _FINANCIAL_VALUE_COLUMNS["cashflow"],
}

_ALL_SOURCES = (
    SOURCE_DAILY,
    SOURCE_PRICE_LIMIT,
    SOURCE_MARGIN,
    SOURCE_INDUSTRY_MEMBERSHIP,
    SOURCE_INDUSTRY_INDEX,
    SOURCE_FINANCIAL,
)


class DataEligibilityError(Exception):
    """D7 contract violation (fail-closed input validation)."""


@dataclass(frozen=True)
class SourceCheck:
    """One p17 §12 source's eligibility outcome."""

    source: str
    required: bool
    ok: bool
    detail: str


@dataclass(frozen=True)
class DataEligibilityVerdict:
    """The RunSpec-grained eligibility verdict (the caller — the P8
    FACTOR_SET_QUALIFIED transition, t21 — treats ``ok=False`` plus
    ``reasons`` as the fail-closed rejection)."""

    spec_id: str
    ok: bool
    checks: dict[str, SourceCheck] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def required_source_families(feature_names: Iterable[str]) -> frozenset[str]:
    """The p17 §12 sources the given features depend on (registry
    mapping; RunSpec-level sources are added by the check itself).

    Unknown feature names raise — silently mapping an unknown feature to
    the daily family would be fail-open (AGENTS §4.3).
    """

    valid = set(FORMULA_VOCAB.feature_names)
    families: set[str] = set()
    for name in feature_names:
        if name not in valid:
            raise DataEligibilityError(
                f"unknown feature {name!r}: the factor set must come from "
                "the vocabulary the feature registry describes"
            )
        level = pit_level_of(name)
        if level is PitLevel.PIT_DAILY:
            families.add(SOURCE_DAILY)
        elif level is PitLevel.PIT_FUNDAMENTAL:
            families.add(SOURCE_FINANCIAL)
        elif level is PitLevel.PIT_CAPITAL:
            families.add(SOURCE_MARGIN)
        elif level is PitLevel.SNAPSHOT:
            if family_of(name) == "industry" or name == "INDUSTRY_MOMENTUM":
                families.update(
                    {SOURCE_INDUSTRY_MEMBERSHIP, SOURCE_INDUSTRY_INDEX}
                )
            else:  # pragma: no cover - defensive: an unmapped SNAPSHOT family
                raise DataEligibilityError(
                    f"feature {name!r} has SNAPSHOT level outside the known "
                    "industry families; the dependency mapping must be "
                    "extended fail-closed instead of guessing"
                )
        elif level is PitLevel.NEUTRAL:
            continue
        else:  # pragma: no cover - defensive: enum exhaustion
            raise DataEligibilityError(
                f"unmapped pit level {level!r} for feature {name!r}"
            )
    return frozenset(families)


def _shanghai_today() -> str:
    return datetime.now(_SHANGHAI).date().strftime("%Y%m%d")


def _is_missing(value) -> bool:
    """DuckDB scalar absence: NULL or NaN (NaN != NaN)."""

    return value is None or value != value


def _open_sessions_before(config, evaluation_day: str) -> list[str] | None:
    """Open sessions strictly before ``evaluation_day`` from the unique
    production trade calendar (ascending); ``None`` = not computable."""

    try:
        with AshareDB(config.duckdb_path, read_only=True) as db:
            rows = db.query(
                f"SELECT trade_date FROM {config.calendar_table} "
                "WHERE is_open=TRUE AND trade_date < ? ORDER BY trade_date",
                [evaluation_day],
            )
    except Exception:  # noqa: BLE001 - missing db/calendar is not computable
        return None
    if rows.empty:
        return []
    return [str(value) for value in rows["trade_date"].tolist()]


def _lag_sessions(
    axis: list[str], source_latest: str, *, same_day: bool
) -> int:
    """p17 §12 lag in open-session positions: the distance from the
    source's available_session back to the most recent completed open
    session (the axis tail).  ``same_day`` selects the §6.1 mapping
    (price-limit: T's own session; otherwise the next open session after
    T).  An available_session beyond the axis means the data is ahead of
    the completed window: lag 0."""

    if same_day:
        index = bisect.bisect_left(axis, source_latest)
    else:
        index = bisect.bisect_right(axis, source_latest)
    if index >= len(axis):
        return 0
    return len(axis) - 1 - index


def _max_column(db: AshareDB, table: str, column: str) -> str | None:
    """MAX(column) of ``table``; ``None`` when the table is missing or
    empty (the caller reports not-computable)."""

    try:
        row = db.query(f"SELECT MAX({column}) AS mx FROM {table}").iloc[0]
    except Exception:  # noqa: BLE001 - a missing table is a report, not a crash
        return None
    value = row["mx"]
    if _is_missing(value):
        return None
    return str(value)


def _quarter_ends(start: str, end: str) -> list[str]:
    """Quarter-end dates (YYYYMMDD) within ``[start, end]`` — the report
    periods the formal scope's fundamental reads depend on."""

    start_year, end_year = int(start[:4]) - 1, int(end[:4])
    ends = []
    for year in range(start_year, end_year + 1):
        for month_day in ("0331", "0630", "0930", "1231"):
            date = f"{year}{month_day}"
            if start <= date <= end:
                ends.append(date)
    return ends


def _universe_list_dates(
    config, universe_codes: list[str]
) -> dict[str, str | None]:
    """``list_date`` per universe code from the production ``stocks``
    table (the single listing-date source).  Unknown/missing values map
    to ``None`` and keep ALL periods in the expected denominator
    (conservative — a coverage gap is never silently excused)."""

    try:
        with AshareDB(config.duckdb_path, read_only=True) as db:
            placeholders = ", ".join("?" for _ in universe_codes)
            rows = db.query(
                f"SELECT ts_code, list_date FROM {config.stocks_table} "
                f"WHERE ts_code IN ({placeholders})",
                list(universe_codes),
            )
    except Exception:  # noqa: BLE001 - missing table: stay conservative
        return {code: None for code in universe_codes}
    dates = {
        str(row.ts_code): str(row.list_date)
        for row in rows.itertuples(index=False)
        if not _is_missing(row.list_date)
    }
    return {code: dates.get(code) for code in universe_codes}


#: The quarter-end month/day suffixes a report period must carry (p17
#: §5.3; B01 R11: non-quarter-end rows are quality findings, never
#: report periods).
_QUARTER_END_MONTH_DAYS = ("0331", "0630", "0930", "1231")


@dataclass(frozen=True)
class _FinancialCoverage:
    """Rich result of one :func:`_financial_coverage` pass (the worst
    table decides the family ratio; the counters feed the B01 report's
    revision / quality sub-items and the real gap list — same row pass,
    no second scan)."""

    ratio: float | None
    detail: str
    expected_units: int
    covered_units: int
    gap_count: int
    gap_examples: tuple[tuple[str, str], ...]
    versioned_keys: int
    total_keys: int
    non_quarter_end_rows: int


def _financial_coverage(
    db: AshareDB,
    universe_codes: list[str],
    quarter_ends: list[str],
    listing_dates: dict[str, str | None] | None = None,
    *,
    gap_example_limit: int = 20,
) -> _FinancialCoverage:
    """Report-period coverage of the three financial version tables
    (p17 §12: 当前正式 scope 报告期覆盖 ≥ 90%).  Expected = universe codes
    × quarter-end periods in the RunSpec window (extended one year back
    for the YoY/TTM lookbacks the fundamental features consume), with
    PRE-LISTING quarters excluded per the stock's ``list_date`` (t73
    correctness fix: no reports exist before listing — those quarters
    are a source gap, not a coverage failure; data-engineer's sampling
    confirmed every sampled gap was pre-listing).  Post-listing gaps
    stay expected (delisting/suspension periods are honestly missing).
    Covered = keys with a version row whose every core column is
    non-null.  The worst table decides the family verdict.  Ratio
    ``None`` = not computable (the detail carries the reason).  The B01
    counters (revision versions per business key, non-quarter-end rows
    per erratum #4, and the real expected-minus-covered gap list) come
    from the same pass — no second scan."""

    listing_dates = listing_dates or {}
    per_table: list[tuple[str, float]] = []
    excluded_pre_listing = 0
    expected_units = covered_units = gap_count = 0
    gap_examples: list[tuple[str, str]] = []
    versioned_keys = total_keys = non_quarter_end_rows = 0
    for table, core_columns in _FINANCIAL_TABLES.items():
        try:
            rows = db.query(
                f"SELECT ts_code, end_date, {', '.join(core_columns)} "
                f"FROM {table}"
            )
        except Exception as exc:  # noqa: BLE001 - missing table: not computable
            return _FinancialCoverage(
                None, f"{table} not readable ({type(exc).__name__})",
                0, 0, 0, (), 0, 0, 0,
            )
        covered: set[tuple[str, str]] = set()
        key_counts: dict[tuple[str, str], int] = {}
        if not rows.empty:
            for row in rows.itertuples(index=False):
                values = [getattr(row, column) for column in core_columns]
                key = (str(row.ts_code), str(row.end_date))
                key_counts[key] = key_counts.get(key, 0) + 1
                if str(row.end_date)[4:] not in _QUARTER_END_MONTH_DAYS:
                    non_quarter_end_rows += 1
                if not any(_is_missing(value) for value in values):
                    covered.add(key)
        total_keys += len(key_counts)
        versioned_keys += sum(1 for count in key_counts.values() if count > 1)
        expected = {
            (code, period)
            for code in universe_codes
            for period in quarter_ends
            if listing_dates.get(code) in (None, "")
            or period >= listing_dates.get(code, "")
        }
        excluded_pre_listing += sum(
            1
            for code in universe_codes
            for period in quarter_ends
            if listing_dates.get(code)
            and period < listing_dates.get(code, "")
        )
        if not expected:
            continue  # no report periods in scope: nothing required here
        expected_units += len(expected)
        covered_units += len(covered & expected)
        table_gaps = expected - covered
        gap_count += len(table_gaps)
        gap_examples.extend(sorted(table_gaps))
        ratio = len(covered & expected) / len(expected)
        per_table.append((table, ratio))
    gap_examples = gap_examples[:gap_example_limit]
    if not per_table:
        return _FinancialCoverage(
            None, "no report periods in the formal scope",
            expected_units, covered_units, gap_count, tuple(gap_examples),
            versioned_keys, total_keys, non_quarter_end_rows,
        )
    worst_table, worst_ratio = min(per_table, key=lambda entry: entry[1])
    detail = "; ".join(f"{table}={ratio:.2%}" for table, ratio in per_table)
    detail += (
        f" (worst: {worst_table}; pre-listing periods excluded from "
        f"expected: {excluded_pre_listing}; gaps: {gap_count}; "
        f"versioned keys: {versioned_keys}/{total_keys}; "
        f"non-quarter-end rows: {non_quarter_end_rows})"
    )
    return _FinancialCoverage(
        worst_ratio, detail, expected_units, covered_units, gap_count,
        tuple(gap_examples), versioned_keys, total_keys,
        non_quarter_end_rows,
    )


def _dataset_manifest_check(v2_db: AshareDB) -> tuple[bool | None, str]:
    """The A02 composite-identity integrity check shared by the D7
    verdict and the B01 report (single implementation — p29 §2.3
    non-goal 7).  ``True``/``False`` = verified / failed (the detail
    carries the reason); ``None`` = not computable."""

    try:
        manifest = latest_manifest(v2_db, TUSHARE_V2_CONFIG)
    except Exception as exc:  # noqa: BLE001 - no manifest table yet
        return None, (
            "not computable: the v2 store carries no dataset manifest "
            "(identity cannot be established; "
            f"{type(exc).__name__}: {exc})"
        )
    report = verify_dataset(
        v2_db, TUSHARE_V2_CONFIG, expected=manifest, verify="fast",
    )
    if not report.ok:
        failed = [
            f"{entry.table}: {entry.detail}"
            for entry in report.tables.values()
            if not entry.ok
        ]
        return False, "manifest verification failed: " + "; ".join(failed)
    return True, f"manifest {manifest.dataset_id[:12]}… verified (fast)"


def check_runspec_data_eligibility(
    spec,
    feature_names: Iterable[str],
    universe_codes: list[str],
    config,
    *,
    evaluation_day: str | None = None,
) -> DataEligibilityVerdict:
    """Run the p17 §12 checks for the sources ``spec``'s factor set needs.

    ``feature_names`` are vocabulary feature names (the factor set's
    resolved features); ``universe_codes`` are the PIT-eligible codes the
    DATA_QUALIFIED stage established (the financial coverage universe);
    ``evaluation_day`` defaults to the real Shanghai clock (tests inject
    it, mirroring the p16 seam)."""

    if not universe_codes:
        raise DataEligibilityError(
            "universe_codes must be non-empty: the DATA_QUALIFIED stage's "
            "universe is the financial coverage denominator"
        )
    families = required_source_families(feature_names)
    # RunSpec level: every formal run consumes the daily price family
    # (prices/labels/universe) and the D4 execution limit constraints;
    # the manifest integrity check is identity, not a feature dependency.
    required = set(families) | {
        SOURCE_DAILY,
        SOURCE_PRICE_LIMIT,
        SOURCE_DATASET_MANIFEST,
    }

    day = evaluation_day or _shanghai_today()
    axis = _open_sessions_before(config, day)
    try:
        v2_db: AshareDB | None = AshareDB(
            resolve_v2_duckdb_path(config), read_only=True
        )
    except Exception as exc:  # noqa: BLE001 - missing store: report, don't crash
        v2_db = None
        v2_error = f"{type(exc).__name__}: {exc}"

    checks: dict[str, SourceCheck] = {}
    reasons: list[str] = []

    def fail(source: str, detail: str) -> SourceCheck:
        return SourceCheck(source, True, False, detail)

    try:
        if axis is None:
            for source in (SOURCE_DAILY, SOURCE_PRICE_LIMIT, SOURCE_MARGIN):
                if source in required:
                    checks[source] = fail(
                        source,
                        "not computable: the trade calendar at "
                        f"{config.duckdb_path} is unavailable (no completed "
                        "open-session reference)",
                    )

        if v2_db is None:
            for source in _ALL_SOURCES:
                if source in required and source not in checks:
                    checks[source] = fail(
                        source,
                        f"not computable: v2 store unreadable ({v2_error})",
                    )

        if axis is not None and v2_db is not None:
            def lag_check(source: str, tables, column: str, *,
                          same_day: bool, tolerance: int) -> SourceCheck:
                lags: list[str] = []
                worst = 0
                for table in tables:
                    latest = _max_column(v2_db, table, column)
                    if latest is None:
                        return fail(
                            source,
                            f"not computable: {table} is missing or empty "
                            "(一切源 来源不可计算一律 fail)",
                        )
                    lag = _lag_sessions(axis, latest, same_day=same_day)
                    lags.append(f"{table}={lag}")
                    worst = max(worst, lag)
                if worst > tolerance:
                    return fail(
                        source,
                        f"lag {worst} > {tolerance} sessions "
                        f"({'; '.join(lags)}; today={day})",
                    )
                return SourceCheck(
                    source, True, True,
                    f"lag {worst} <= {tolerance} sessions "
                    f"({'; '.join(lags)}; today={day})",
                )

            checks[SOURCE_DAILY] = lag_check(
                SOURCE_DAILY, _DAILY_FAMILY_TABLES, "trade_date",
                same_day=False, tolerance=DAILY_ELIGIBILITY_MAX_LAG_SESSIONS,
            ) if SOURCE_DAILY in required else SourceCheck(
                SOURCE_DAILY, False, True,
                "not required by this factor set (not evaluated; p17 §12 "
                "non-goal: no global freshness gate)",
            )
            checks[SOURCE_PRICE_LIMIT] = lag_check(
                SOURCE_PRICE_LIMIT, (SOURCE_PRICE_LIMIT,), "trade_date",
                same_day=True, tolerance=PRICE_LIMIT_MAX_LAG_SESSIONS,
            ) if SOURCE_PRICE_LIMIT in required else SourceCheck(
                SOURCE_PRICE_LIMIT, False, True,
                "not required by this factor set (not evaluated; p17 §12 "
                "non-goal: no global freshness gate)",
            )
            checks[SOURCE_MARGIN] = lag_check(
                SOURCE_MARGIN, (SOURCE_MARGIN,), "available_date",
                same_day=True, tolerance=MARGIN_MAX_LAG_SESSIONS,
            ) if SOURCE_MARGIN in required else SourceCheck(
                SOURCE_MARGIN, False, True,
                "not required by this factor set (not evaluated; p17 §12 "
                "non-goal: no global freshness gate)",
            )

            if SOURCE_INDUSTRY_MEMBERSHIP in required:
                gate = industry_membership_gate(
                    config,
                    axis,
                    level=1,
                    coverage_threshold=INDUSTRY_MEMBERSHIP_MIN_COVERAGE,
                )
                if (
                    gate["coverage"] < INDUSTRY_MEMBERSHIP_MIN_COVERAGE
                    or gate["overlap_violations"] > 0
                    or not gate["availability_complete"]
                ):
                    checks[SOURCE_INDUSTRY_MEMBERSHIP] = fail(
                        SOURCE_INDUSTRY_MEMBERSHIP,
                        f"coverage {gate['coverage']:.2%} < "
                        f"{INDUSTRY_MEMBERSHIP_MIN_COVERAGE:.0%}, "
                        f"overlap_violations={gate['overlap_violations']}, "
                        f"availability_complete="
                        f"{gate['availability_complete']} (p17 §12/§15.3)",
                    )
                else:
                    checks[SOURCE_INDUSTRY_MEMBERSHIP] = SourceCheck(
                        SOURCE_INDUSTRY_MEMBERSHIP, True, True,
                        f"coverage {gate['coverage']:.2%} >= "
                        f"{INDUSTRY_MEMBERSHIP_MIN_COVERAGE:.0%}, overlap 0",
                    )

            if SOURCE_INDUSTRY_INDEX in required:
                lag = industry_index_freshness(config, axis)
                if lag is None:
                    checks[SOURCE_INDUSTRY_INDEX] = fail(
                        SOURCE_INDUSTRY_INDEX,
                        "not computable: the Shenwan index source has no "
                        "rows (D5 industry_index_freshness)",
                    )
                elif lag > INDUSTRY_INDEX_MAX_LAG_SESSIONS:
                    checks[SOURCE_INDUSTRY_INDEX] = fail(
                        SOURCE_INDUSTRY_INDEX,
                        f"lag {lag} > {INDUSTRY_INDEX_MAX_LAG_SESSIONS} "
                        "sessions (p17 §12)",
                    )
                else:
                    checks[SOURCE_INDUSTRY_INDEX] = SourceCheck(
                        SOURCE_INDUSTRY_INDEX, True, True,
                        f"lag {lag} <= "
                        f"{INDUSTRY_INDEX_MAX_LAG_SESSIONS} sessions",
                    )

            if SOURCE_FINANCIAL in required:
                window_start = (
                    f"{int(spec.dev_window.start[:4]) - 1}"
                    f"{spec.dev_window.start[4:]}"
                )
                quarter_ends = _quarter_ends(
                    window_start, spec.dev_window.end
                )
                # t73: pre-listing quarters are excluded from the expected
                # denominator (a mid-quarter-listed stock cannot have
                # reports before its listing — a source gap, not a
                # coverage failure; data-engineer's sampling confirmed it).
                listing_dates = _universe_list_dates(
                    config, list(universe_codes)
                )
                coverage = _financial_coverage(
                    v2_db, list(universe_codes), quarter_ends,
                    listing_dates=listing_dates,
                )
                if coverage.ratio is None:
                    checks[SOURCE_FINANCIAL] = fail(
                        SOURCE_FINANCIAL,
                        f"not computable: {coverage.detail}",
                    )
                elif coverage.ratio < FINANCIAL_REPORT_COVERAGE_MIN:
                    checks[SOURCE_FINANCIAL] = fail(
                        SOURCE_FINANCIAL,
                        f"report-period coverage {coverage.ratio:.2%} < "
                        f"{FINANCIAL_REPORT_COVERAGE_MIN:.0%} "
                        f"({coverage.detail}; "
                        f"periods {quarter_ends[0]}..{quarter_ends[-1]})",
                    )
                else:
                    checks[SOURCE_FINANCIAL] = SourceCheck(
                        SOURCE_FINANCIAL, True, True,
                        f"report-period coverage {coverage.ratio:.2%} >= "
                        f"{FINANCIAL_REPORT_COVERAGE_MIN:.0%} "
                        f"({coverage.detail})",
                    )

        if SOURCE_DATASET_MANIFEST in required:
            if v2_db is None:
                checks[SOURCE_DATASET_MANIFEST] = fail(
                    SOURCE_DATASET_MANIFEST,
                    f"not computable: v2 store unreadable ({v2_error})",
                )
            else:
                ok, detail = _dataset_manifest_check(v2_db)
                if ok is True:
                    checks[SOURCE_DATASET_MANIFEST] = SourceCheck(
                        SOURCE_DATASET_MANIFEST, True, True, detail,
                    )
                else:
                    checks[SOURCE_DATASET_MANIFEST] = fail(
                        SOURCE_DATASET_MANIFEST, detail,
                    )

        for source in _ALL_SOURCES:
            if source not in checks:
                checks[source] = SourceCheck(
                    source, source in required, True,
                    "not required by this factor set (not evaluated; p17 "
                    "§12 non-goal: no global freshness gate)",
                )

        reasons = [
            f"source {source}: {checks[source].detail}"
            for source, check in checks.items()
            if check.required and not check.ok
        ]
    finally:
        if v2_db is not None:
            try:
                v2_db.close()
            except Exception:  # noqa: BLE001 - close is best effort
                pass

    return DataEligibilityVerdict(
        spec_id=spec.spec_id,
        ok=not reasons,
        checks=checks,
        reasons=reasons,
    )


# --- B01 (p29): the per-family data-quality report ------------------------------
#
# The D7 verdict above stays the p17 §12 RunSpec gate, unchanged (p29
# §2.3 non-goal 4).  The report below is the B01 machine product: per
# source family it emits coverage / lag / revision / quality sub-items,
# the four states (pass/fail/unknown/not_applicable — N/A never counts
# as a pass, I9), and the REAL gap lists (table/field/code/date).  Every
# threshold is consumed from ``ashare_data.gates`` at call time (I12);
# every conclusion is labelled with its numbering space (I11); the
# report is strictly read-only (I14) and carries deliberately NO global
# ok/healthy boolean (the F06 lesson).

# v2 (B01 re-review F8): the repair round added payload structure
# (quality.suspension_endpoint / ingest_batches_seen,
# coverage.pre_family_sessions, the events revision sub-item and the
# denominator_rule wording) — the additive-fields-still-bump convention
# applies to the standalone report version, so v1 and v2 payloads are
# distinguishable.  Standalone (not spec-bound) per the
# READINESS_REPORT_VERSION precedent — see the v1 rationale below.
DATA_QUALITY_REPORT_VERSION = 2

SOURCE_INDEX_WEIGHT = "index_weight_pit"
SOURCE_EVENTS_AUDIT = "events_audit"

#: I11: the numbering spaces (AGENTS §5.4 keeps the data gate and the
#: promotion spaces independent; the report adds the informational
#: ones).  Every family's conclusions carry their space label.
SPACE_D7 = "D7 (ashare_model.data_eligibility, p17 §12)"
SPACE_SNAPSHOT = "A02 snapshot identity (ashare_data.manifest)"
SPACE_GATES = "G1-G8 (ashare_data.gates)"
SPACE_REPORT = "B01 report space (informational; no gate binding)"

_FAMILY_SPACES = {
    SOURCE_DAILY: SPACE_D7,
    SOURCE_PRICE_LIMIT: SPACE_D7,
    SOURCE_MARGIN: SPACE_D7,
    SOURCE_INDUSTRY_MEMBERSHIP: SPACE_D7,
    SOURCE_INDUSTRY_INDEX: SPACE_D7,
    SOURCE_FINANCIAL: SPACE_D7,
    SOURCE_DATASET_MANIFEST: SPACE_SNAPSHOT,
    SOURCE_INDEX_WEIGHT: SPACE_REPORT,
    SOURCE_EVENTS_AUDIT: SPACE_REPORT,
}

QUALITY_FAMILIES = tuple(_FAMILY_SPACES)

#: I12: every threshold the report consumes, named by its single
#: authority.  The VALUES are read from the gates module at call time —
#: the report never carries a second copy of a number.
_THRESHOLD_SOURCES = {
    "daily_max_lag_sessions": (
        "ashare_data.gates.DAILY_ELIGIBILITY_MAX_LAG_SESSIONS"
    ),
    "price_limit_max_lag_sessions": (
        "ashare_data.gates.PRICE_LIMIT_MAX_LAG_SESSIONS"
    ),
    "margin_max_lag_sessions": "ashare_data.gates.MARGIN_MAX_LAG_SESSIONS",
    "industry_index_max_lag_sessions": (
        "ashare_data.gates.INDUSTRY_INDEX_MAX_LAG_SESSIONS"
    ),
    "industry_membership_min_coverage": (
        "ashare_data.gates.INDUSTRY_MEMBERSHIP_MIN_COVERAGE"
    ),
    "financial_report_coverage_min": (
        "ashare_data.gates.FINANCIAL_REPORT_COVERAGE_MIN"
    ),
    "freshness_tolerance_sessions": (
        "ashare_data.gates.FRESHNESS_TOLERANCE_SESSIONS"
    ),
}

#: I8: the event-style reconciliation runs per collection date — a
#: zero-row day is only "zero event" when a succeeded ingest batch
#: proves it.  The ledger's ``endpoint`` column records the V2 TABLE
#: name (the writer passes the table to the batch recorder, and the
#: store validates the endpoint against ``V2_TABLES``) — the Tushare API
#: name (``suspend_d``) is NOT the key (B01 review F1: an API-name
#: constant made the zero-event branch unreachable on the real ledger).
SUSPENSION_INGEST_ENDPOINT = "daily_suspension"

#: The informational events family (I10): the F3 tables are NOT in the
#: p17 §12 verdict set (p26 ruling 4) — the family reports honestly and
#: never blocks a price baseline.
_EVENT_TABLES = (
    "earnings_forecast_pit",
    "earnings_express_pit",
    "repurchase_event_pit",
    "holder_number_pit",
)

#: The informational events family's I3/I4 declaration (B01 review F6):
#: event rows are single-version (source_row_hash keyed) and no
#: restatement chain is restorable from the source, so the revision
#: history is honestly unrestorable and the informational Tier limit
#: travels with every report.
EVENTS_AUDIT_TIME_SEMANTICS = SourceTimeSemantics(
    family="events_audit",
    tables=_EVENT_TABLES,
    economic_time="ann_date / end_date (the announcement or report period)",
    availability_time="available_date (writer-computed per p17 §6.1)",
    collection_authority="source_ingest_runs",
    revision_semantics="unrestorable",
    revision_history_complete=False,
    tier_limit_note=(
        "event rows are single-version and the source exposes no "
        "restatement chain; the informational Tier limit is preserved "
        "and F3 expansion must re-check coverage separately (p29 §1.7)"
    ),
    visibility_rule=(
        "visible from available_date; the family is informational "
        "(not in the p17 §12 verdict set, p26 ruling 4) — it reports "
        "lag/staleness only and never blocks a price baseline (I10)"
    ),
)


@dataclass(frozen=True)
class DataGap:
    """One entry of a real gap list (p29 I9: concrete table/field/code/
    date; ``ts_code`` is None for whole-market/date-level gaps)."""

    table: str
    field: str
    ts_code: str | None
    date: str
    kind: str
    board: str = "unknown"
    detail: str = ""


@dataclass(frozen=True)
class SourceQualityCheck:
    """One source family's B01 outcome: the four sub-items + four states.

    ``status`` uses the readiness four-state vocabulary; unknown = the
    check could not be computed (never a pass); not_applicable = the
    research does not depend on this source (I10).  ``coverage`` /
    ``lag`` / ``revision`` / ``quality`` are family-shaped dicts with
    exact counts; ``gaps`` carries exact counts with examples capped."""

    source: str
    required: bool
    status: str
    coverage: dict
    lag: dict
    revision: dict
    quality: dict
    gaps: tuple[DataGap, ...]
    gate_space: str
    detail: str


@dataclass(frozen=True)
class DataQualityReport:
    """The B01 per-family data-quality report (p29 §2.2 item 5)."""

    report_version: int
    evaluation_day: str
    primary_db: str
    v2_db: str
    families: dict[str, SourceQualityCheck]
    spaces: dict[str, str]
    mixed_format_findings: tuple[dict, ...]
    threshold_sources: dict[str, str]


def board_of(ts_code: str) -> str:
    """Board classification for gap lists (reporting only — the formal
    constraint semantics stay the exact ``daily_price_limit`` table per
    p19 §2.4; the retired board-rate inference is NOT re-introduced)."""

    try:
        symbol, suffix = str(ts_code).strip().split(".", 1)
    except ValueError:
        return "unknown"
    if len(symbol) != 6 or not symbol.isdigit():
        return "unknown"
    prefix = symbol[:3]
    if suffix == "SH":
        return "star" if prefix in ("688", "689") else "sse_main"
    if suffix == "SZ":
        return "chinext" if prefix in ("300", "301") else "szse_main"
    if suffix == "BJ":
        return "bse"
    return "unknown"


def _mixed_format_findings(
    db: AshareDB, table: str, fields: Iterable[str], note: str = "",
) -> list[dict]:
    """The I1 read-only format audit over one table's date columns: exact
    non-canonical counts plus capped examples.  A missing table is a
    family check's business, not the audit's."""

    out: list[dict] = []
    for field_name in fields:
        try:
            frame = db.query(
                f"SELECT {field_name} AS v FROM {table} "
                f"WHERE {field_name} IS NOT NULL"
            )
        except Exception:  # noqa: BLE001 - missing table: skip silently here
            continue
        if frame.empty:
            continue
        bad = [v for v in frame["v"].tolist() if not is_canonical_date(v)]
        if bad:
            examples: list[str] = []
            for value in bad:
                text = str(value)
                if text not in examples:
                    examples.append(text)
                if len(examples) >= 5:
                    break
            out.append({
                "table": table,
                "field": field_name,
                "count": len(bad),
                "examples": examples,
                "note": note,
            })
    return out


def build_data_quality_report(
    config,
    *,
    evaluation_day: str | None = None,
    required_families: Iterable[str] | None = None,
    feature_names: Iterable[str] | None = None,
    universe_codes: list[str] | None = None,
    gap_example_limit: int = 20,
    suspension_endpoint: str = SUSPENSION_INGEST_ENDPOINT,
) -> DataQualityReport:
    """The per-family data-quality report (p29 §2.2 item 5, I1-I14).

    ``feature_names`` (RunSpec-grained) or ``required_families`` select
    the families the research depends on; every other family is reported
    ``not_applicable`` with its reason (I10 — no global gate).  With
    neither argument the report is the full audit (all families evaluated
    from the data, none required).  ``evaluation_day`` defaults to the
    real Shanghai clock (tests inject it).  Strictly read-only (I14).
    """

    day = evaluation_day or _shanghai_today()
    if feature_names is not None:
        required = set(required_source_families(feature_names)) | {
            SOURCE_DAILY,
            SOURCE_PRICE_LIMIT,
            SOURCE_DATASET_MANIFEST,
        }
    elif required_families is not None:
        unknown_families = sorted(
            set(required_families) - set(QUALITY_FAMILIES)
        )
        if unknown_families:
            raise DataEligibilityError(
                f"unknown source family {unknown_families[0]!r}; the p29 "
                "family set is closed"
            )
        required = set(required_families)
    else:
        required = set()  # audit mode: evaluate everything, require nothing
    evaluate = set(QUALITY_FAMILIES) if not required else required

    axis = _open_sessions_before(config, day)

    # -- I14: strictly read-only database handles --------------------------
    primary_db: AshareDB | None = None
    try:
        primary_db = AshareDB(config.duckdb_path, read_only=True)
    except Exception:  # noqa: BLE001 - missing store: report, not crash
        primary_db = None
    v2_db: AshareDB | None = None
    v2_error: str | None = None
    try:
        v2_db = AshareDB(resolve_v2_duckdb_path(config), read_only=True)
    except Exception as exc:  # noqa: BLE001
        v2_error = f"{type(exc).__name__}: {exc}"

    contract = None
    contract_error: str | None = None
    try:
        contract = resolve_universe_contract(config)
    except Exception as exc:  # noqa: BLE001 - the denominator is then unknown
        contract_error = f"{type(exc).__name__}: {exc}"

    # -- I1: the read-only mixed-format audit over the consumed columns ----
    findings: list[dict] = []
    if primary_db is not None:
        findings.extend(_mixed_format_findings(
            primary_db, config.daily_table, ("trade_date",),
            note=(
                "legacy main-DB daily_bar is read by G1-G8 ("
                f"{SPACE_GATES}); judgement zero-change per p29 §2.3 "
                "non-goal 2 — the G space owns the verdict"
            ),
        ))
        findings.extend(_mixed_format_findings(
            primary_db, config.calendar_table, ("trade_date",),
            note="the open-session axis every interval comparison aligns to",
        ))
        findings.extend(_mixed_format_findings(
            primary_db, config.stocks_table, ("list_date",),
        ))
        findings.extend(_mixed_format_findings(
            primary_db, config.constituents_table,
            ("in_date", "out_date"),
        ))
    if v2_db is not None:
        for table, fields in (
            ("daily_bar_raw", ("trade_date",)),
            ("daily_adj_factor", ("trade_date",)),
            ("daily_basic_pit", ("trade_date",)),
            ("daily_price_limit", ("trade_date",)),
            ("daily_suspension", ("trade_date",)),
            ("margin_detail_pit", ("trade_date", "available_date")),
            ("industry_membership_pit",
             ("in_date", "out_date", "available_date")),
            ("index_weight_pit", ("trade_date", "available_date")),
        ):
            findings.extend(_mixed_format_findings(v2_db, table, fields))
        for table in _EVENT_TABLES:
            findings.extend(_mixed_format_findings(
                v2_db, table, ("ann_date", "end_date", "available_date"),
            ))

    def _findings_for(table: str) -> list[dict]:
        return [f for f in findings if f["table"] == table]

    # -- I2: a mixed calendar axis blocks every comparison that aligns to it
    axis_block: str | None = None
    if axis is None:
        axis_block = (
            "not computable: the trade calendar is unavailable (no "
            "completed open-session reference)"
        )
    else:
        try:
            assert_comparable_date_axes(axis, field="open_session_axis")
        except DateAxisFormatError as exc:
            axis_block = (
                "not computable: mixed-format date axis blocked the "
                f"comparison (I2 fail-closed): {exc}"
            )

    _horizon_memo: dict = {}

    def _horizon():
        """(mn, mx, window dates) of the v2 collection horizon."""

        if "value" not in _horizon_memo:
            if v2_db is None:
                _horizon_memo["value"] = None
            else:
                row = v2_db.query(
                    "SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx "
                    "FROM daily_bar_raw"
                ).iloc[0]
                if pd.isna(row["mn"]):
                    _horizon_memo["value"] = None
                else:
                    mn, mx = str(row["mn"]), str(row["mx"])
                    window = (
                        [d for d in contract.sessions if mn <= d <= mx]
                        if contract is not None
                        else []
                    )
                    _horizon_memo["value"] = (mn, mx, window)
        return _horizon_memo["value"]

    def _universe():
        if contract is None:
            return None, (
                "not computable: the PIT universe is unresolvable "
                f"({contract_error})"
            )
        if contract.status.degraded:
            return None, (
                "not computable: the universe contract resolved only via "
                "the development fallback; an audit denominator must be "
                "strict (p29 I6)"
            )
        codes = (
            list(universe_codes) if universe_codes is not None
            else list(contract.codes)
        )
        membership_records = contract.constituents.to_dict("records")
        list_dates = {
            str(row["ts_code"]): row["list_date"]
            for row in contract.stocks.to_dict("records")
        }
        policy = UniversePolicy(
            index_codes=tuple(str(code) for code in config.index_codes),
            min_listed_sessions=config.min_listed_sessions,
            membership_end_inclusive=False,
        )
        return (codes, membership_records, list_dates, policy), None

    def _bar_panel(codes: list[str], dates_for_mask: list[str]) -> np.ndarray:
        """The v2 daily-bar presence matrix over the horizon window (the
        same panel shape the G6 mask build consumes)."""

        quoted = sql_quoted_list(codes)
        bars = v2_db.query(
            f"SELECT ts_code, trade_date FROM daily_bar_raw "
            f"WHERE ts_code IN ({quoted})"
        )
        if bars.empty:
            return np.zeros((len(codes), len(dates_for_mask)), dtype=bool)
        panel = bars.assign(_p=1.0).pivot_table(
            index="ts_code", columns="trade_date", values="_p",
            aggfunc="last",
        ).reindex(index=codes, columns=dates_for_mask)
        return panel.notna().to_numpy(dtype=bool)

    def _check(source, required_flag, status, coverage, lag, revision,
               quality, gaps, detail) -> SourceQualityCheck:
        return SourceQualityCheck(
            source, required_flag, status, coverage, lag, revision,
            quality, tuple(gaps), _FAMILY_SPACES[source], detail,
        )

    def _not_computable(source: str, reason: str) -> SourceQualityCheck:
        return _check(
            source, source in required, STATUS_UNKNOWN, {}, {}, {}, {},
            [], reason,
        )

    def _not_required(source: str) -> SourceQualityCheck:
        return _check(
            source, False, STATUS_NOT_APPLICABLE, {}, {}, {}, {}, [],
            "not required by this factor set (not evaluated; p17 §12 "
            "non-goal: no global freshness gate)",
        )

    # -- families -----------------------------------------------------------

    def build_daily() -> SourceQualityCheck:
        gaps: list[DataGap] = []
        quality: dict = {"mixed_format_count": 0, "invalid_adj_factor": 0}
        family_tables = (config.daily_table,) + _DAILY_FAMILY_TABLES
        quality["mixed_format_count"] = sum(
            f["count"] for f in findings if f["table"] in family_tables
        )
        # v2 family-table findings block the comparison (I2); the main-DB
        # daily_bar finding is G-space — counted above, never blocking.
        for table in _DAILY_FAMILY_TABLES:
            if _findings_for(table):
                return _not_computable(
                    SOURCE_DAILY,
                    "not computable: mixed-format date axis blocked the "
                    f"comparison (I2 fail-closed): {table}",
                )
        if axis_block:
            return _not_computable(SOURCE_DAILY, axis_block)
        if v2_db is None:
            return _not_computable(
                SOURCE_DAILY,
                f"not computable: v2 store unreadable ({v2_error})",
            )
        universe, universe_error = _universe()
        if universe_error is not None:
            return _not_computable(SOURCE_DAILY, universe_error)
        codes, membership_records, list_dates, policy = universe
        horizon = _horizon()
        if horizon is None:
            return _not_computable(
                SOURCE_DAILY,
                "not computable: daily_bar_raw is missing or empty "
                "(一切源 来源不可计算一律 fail)",
            )
        _mn, _mx, dates_for_mask = horizon
        lags: dict[str, int] = {}
        worst = 0
        for table in _DAILY_FAMILY_TABLES:
            latest = _max_column(v2_db, table, "trade_date")
            if latest is None:
                return _not_computable(
                    SOURCE_DAILY,
                    f"not computable: {table} is missing or empty "
                    "(一切源 来源不可计算一律 fail)",
                )
            lag = _lag_sessions(axis, latest, same_day=False)
            lags[table] = lag
            worst = max(worst, lag)
        tolerance = gates.DAILY_ELIGIBILITY_MAX_LAG_SESSIONS
        presence = _bar_panel(codes, dates_for_mask)
        expected = expected_observable_units(
            codes, dates_for_mask, contract.sessions,
            membership_records, list_dates, policy,
        )
        expected_units = int(expected.sum())
        covered_units = int((expected & presence).sum())
        missing = expected & ~presence
        gap_count = int(missing.sum())
        if gap_count:
            if gap_count <= 100_000:
                for row, col in np.argwhere(missing)[:gap_example_limit]:
                    gaps.append(DataGap(
                        "daily_bar_raw", "trade_date",
                        codes[int(row)], dates_for_mask[int(col)],
                        "missing_bar",
                    ))
            else:
                # exact count stands; a bounded scan still yields examples
                found = 0
                for row in range(missing.shape[0]):
                    for col in np.flatnonzero(missing[row]):
                        if found >= gap_example_limit:
                            break
                        gaps.append(DataGap(
                            "daily_bar_raw", "trade_date", codes[row],
                            dates_for_mask[int(col)], "missing_bar",
                        ))
                        found += 1
                    if found >= gap_example_limit:
                        break
        broken = v2_db.query(
            "SELECT ts_code, trade_date FROM daily_adj_factor "
            "WHERE adj_factor IS NULL OR adj_factor <= 0 "
            "ORDER BY ts_code, trade_date"
        )
        quality["invalid_adj_factor"] = int(len(broken))
        for entry in broken.head(gap_example_limit).itertuples(index=False):
            gaps.append(DataGap(
                "daily_adj_factor", "adj_factor", str(entry.ts_code),
                str(entry.trade_date), "invalid_adj_factor",
                board_of(str(entry.ts_code)),
                "non-positive/NULL adjustment factor: the series is "
                "fail-closed degraded, never filled (p19 §2.1)",
            ))
        coverage = {
            "ratio": (
                covered_units / expected_units if expected_units else None
            ),
            "expected_units": expected_units,
            "covered_units": covered_units,
            "gap_count": gap_count,
            "denominator_rule": (
                "listed ∧ target PIT universe ∧ observable "
                "(build_universe_mask; p29 I6) over the v2 collection "
                "horizon"
            ),
            "window": (
                [dates_for_mask[0], dates_for_mask[-1]]
                if dates_for_mask else []
            ),
        }
        if quality["invalid_adj_factor"] or worst > tolerance:
            status = STATUS_FAIL
        else:
            status = STATUS_PASS
        # The lag direction keys on the LAG, never on the overall status
        # (B01 re-review F7): a family failing on adj-factor defects while
        # fresh must not print the reverse-false "lag 0 > 3".
        detail = (
            f"lag {worst} "
            f"{'<=' if worst <= tolerance else '>'} {tolerance} sessions; "
            f"coverage {covered_units}/{expected_units} collected "
            f"(gaps: missing_bar={gap_count}, "
            f"invalid_adj_factor={quality['invalid_adj_factor']}); a "
            "missing bar is a listed gap, not automatically a defect "
            "(suspensions legitimately have no bar)"
        )
        revision = {
            "semantics": DAILY_PRICE_FAMILY_TIME_SEMANTICS.revision_semantics,
            "revision_history_complete": (
                DAILY_PRICE_FAMILY_TIME_SEMANTICS.revision_history_complete
            ),
            "tier_limit_note": (
                DAILY_PRICE_FAMILY_TIME_SEMANTICS.tier_limit_note
            ),
        }
        return _check(
            SOURCE_DAILY, SOURCE_DAILY in required, status, coverage,
            {
                "worst_sessions": worst,
                "tolerance": tolerance,
                "per_table": lags,
            },
            revision, quality, gaps, detail,
        )

    def build_price_limit() -> SourceQualityCheck:
        gaps: list[DataGap] = []
        if axis_block:
            return _not_computable(SOURCE_PRICE_LIMIT, axis_block)
        if v2_db is None:
            return _not_computable(
                SOURCE_PRICE_LIMIT,
                f"not computable: v2 store unreadable ({v2_error})",
            )
        hits = _findings_for("daily_price_limit") + _findings_for(
            "daily_suspension"
        )
        if hits:
            return _not_computable(
                SOURCE_PRICE_LIMIT,
                "not computable: mixed-format date axis blocked the "
                f"comparison (I2 fail-closed): {hits[0]['table']}."
                f"{hits[0]['field']}",
            )
        latest = _max_column(v2_db, "daily_price_limit", "trade_date")
        if latest is None:
            return _not_computable(
                SOURCE_PRICE_LIMIT,
                "not computable: daily_price_limit is missing or empty "
                "(一切源 来源不可计算一律 fail)",
            )
        tolerance = gates.PRICE_LIMIT_MAX_LAG_SESSIONS
        lag = _lag_sessions(axis, latest, same_day=True)
        # I7: a constraint record must accompany every bar cell — the SQL
        # anti-join IS the real per-code-per-day gap list (no hardcoded
        # board rates anywhere near the evidence).  Deterministic order:
        # reproducible evidence samples (B01 review F5).
        gap_frame = v2_db.query(
            "SELECT b.ts_code AS ts_code, b.trade_date AS trade_date "
            "FROM daily_bar_raw b "
            "LEFT JOIN daily_price_limit p "
            "ON p.ts_code = b.ts_code AND p.trade_date = b.trade_date "
            "WHERE p.ts_code IS NULL "
            "ORDER BY b.ts_code, b.trade_date"
        )
        gap_count = int(len(gap_frame))
        for entry in gap_frame.head(gap_example_limit).itertuples(index=False):
            gaps.append(DataGap(
                "daily_price_limit", "trade_date", str(entry.ts_code),
                str(entry.trade_date), "missing_price_limit",
                board_of(str(entry.ts_code)),
                "missing daily_price_limit record: fail-closed for the "
                "cell's execution evidence (p19 §2.4)",
            ))
        coverage = {
            "gap_count": gap_count,
            "denominator_rule": (
                "bar-present cells (execution constraints must accompany "
                "every bar; p19 §2.4)"
            ),
        }
        # -- I8: suspension is event-style; zero rows are only "zero
        # event" when a succeeded ingest batch proves the day was collected.
        zero_event_dates: list[str] = []
        missed_dates: list[str] = []
        evidence: list[dict] = []
        suspension_status = STATUS_UNKNOWN
        suspension_note = ""
        try:
            susp = v2_db.query(
                "SELECT trade_date, COUNT(*) AS rows_n FROM daily_suspension "
                "GROUP BY trade_date"
            )
            counts = {
                str(row.trade_date): int(row.rows_n)
                for row in susp.itertuples(index=False)
            }
            runs = v2_db.query(
                "SELECT ingest_id, parameters_json, status "
                "FROM source_ingest_runs "
                "WHERE endpoint = ?",
                [suspension_endpoint],
            )
            run_list = [
                (str(r.ingest_id), str(r.parameters_json), str(r.status))
                for r in runs.itertuples(index=False)
            ]
            ledger_ok = True
        except Exception:  # noqa: BLE001 - table/ledger missing: unprovable
            counts, run_list, ledger_ok = {}, [], False
        if ledger_ok:
            horizon = _horizon()
            window = horizon[2] if horizon is not None else []
            for date in window:
                rows_n = counts.get(date, 0)
                matching = [
                    (ingest_id, run_status)
                    for ingest_id, params, run_status in run_list
                    if date in params
                ]
                succeeded = next(
                    (
                        (ingest_id, run_status)
                        for ingest_id, run_status in matching
                        if run_status == "succeeded"
                    ),
                    None,
                )
                if rows_n == 0 and succeeded is not None:
                    zero_event_dates.append(date)
                    evidence.append({
                        "date": date,
                        "ingest_id": succeeded[0],
                        "status": succeeded[1],
                    })
                elif rows_n == 0:
                    missed_dates.append(date)
                    if matching:
                        evidence.append({
                            "date": date,
                            "ingest_id": matching[0][0],
                            "status": matching[0][1],
                        })
                    gaps.append(DataGap(
                        "daily_suspension", "trade_date", None, date,
                        "missed_collection",
                        detail=(
                            "no suspension rows and no succeeded ingest "
                            "batch proves the zero (I8): missed/failed, "
                            "never a zero event"
                        ),
                    ))
            suspension_status = (
                STATUS_FAIL if missed_dates else STATUS_PASS
            )
        else:
            suspension_note = (
                "not computable: the source_ingest_runs ledger is "
                "unavailable — zero suspension rows cannot be proven to "
                "be zero events (I8)"
            )
        quality = {
            "zero_event_dates": tuple(zero_event_dates),
            "missed_collection_dates": tuple(missed_dates),
            "ingest_evidence": tuple(evidence),
            "suspension_reconciliation": (
                "ledger" if ledger_ok else "unavailable"
            ),
            # Diagnosability: a 0-batch reconciliation is visible in the
            # payload instead of silently reading as "no evidence".
            "suspension_endpoint": suspension_endpoint,
            "ingest_batches_seen": len(run_list),
        }
        statuses = [STATUS_PASS if lag <= tolerance else STATUS_FAIL]
        if gap_count:
            statuses.append(STATUS_FAIL)
        statuses.append(suspension_status)
        status = aggregate_status(statuses)
        detail = (
            f"lag {lag} "
            f"{'<=' if lag <= tolerance else '>'} {tolerance} sessions; "
            f"missing price-limit records: {gap_count}; "
            f"suspension zero-event days: {len(zero_event_dates)}, "
            f"missed-collection days: {len(missed_dates)} "
            f"(endpoint {suspension_endpoint}, "
            f"batches seen {len(run_list) if ledger_ok else 'n/a'})"
            + (f"; {suspension_note}" if suspension_note else "")
        )
        revision = {
            "semantics": DAILY_PRICE_FAMILY_TIME_SEMANTICS.revision_semantics,
            "revision_history_complete": (
                DAILY_PRICE_FAMILY_TIME_SEMANTICS.revision_history_complete
            ),
            "tier_limit_note": (
                DAILY_PRICE_FAMILY_TIME_SEMANTICS.tier_limit_note
            ),
        }
        return _check(
            SOURCE_PRICE_LIMIT, SOURCE_PRICE_LIMIT in required, status,
            coverage,
            {
                "worst_sessions": lag,
                "tolerance": tolerance,
                "same_day": True,
            },
            revision, quality, gaps, detail,
        )

    def build_margin() -> SourceQualityCheck:
        if axis_block:
            return _not_computable(SOURCE_MARGIN, axis_block)
        if v2_db is None:
            return _not_computable(
                SOURCE_MARGIN,
                f"not computable: v2 store unreadable ({v2_error})",
            )
        if _findings_for("margin_detail_pit"):
            return _not_computable(
                SOURCE_MARGIN,
                "not computable: mixed-format date axis blocked the "
                "comparison (I2 fail-closed): margin_detail_pit",
            )
        latest = _max_column(v2_db, "margin_detail_pit", "available_date")
        if latest is None:
            return _not_computable(
                SOURCE_MARGIN,
                "not computable: margin_detail_pit is missing or empty "
                "(一切源 来源不可计算一律 fail)",
            )
        tolerance = gates.MARGIN_MAX_LAG_SESSIONS
        lag = _lag_sessions(axis, latest, same_day=True)
        horizon = _horizon()
        window = horizon[2] if horizon is not None else []
        rows = v2_db.query(
            "SELECT trade_date, COUNT(*) AS rows_n FROM margin_detail_pit "
            "GROUP BY trade_date ORDER BY trade_date"
        )
        dates_with_rows = {
            str(row.trade_date) for row in rows.itertuples(index=False)
        }
        # The whole-market family's active span starts at its first
        # collected session: earlier sessions are the pre-family era (out
        # of the denominator, never fabricated into a coverage failure),
        # while sessions WITHOUT rows inside the span are defects — a
        # whole-market publication has no suspension-like zero excuse
        # (B01 review F4), so they fail the family.
        earliest = min(dates_with_rows) if dates_with_rows else None
        span = [d for d in window if earliest is None or d >= earliest]
        pre_family_sessions = len(window) - len(span)
        missing_dates = [d for d in span if d not in dates_with_rows]
        gaps = [
            DataGap(
                "margin_detail_pit", "trade_date", None, date,
                "missing_margin_date",
                detail=(
                    "whole-market family: the session carries no rows "
                    "(no zero-event excuse, B01 review F4)"
                ),
            )
            for date in missing_dates[:gap_example_limit]
        ]
        covered_units = len(span) - len(missing_dates)
        coverage = {
            "ratio": covered_units / len(span) if span else None,
            "expected_units": len(span),
            "covered_units": covered_units,
            "gap_count": len(missing_dates),
            "pre_family_sessions": pre_family_sessions,
            "denominator_rule": (
                "open sessions from the family's first collected session "
                "through the collection horizon (whole-market per-date "
                "publication)"
            ),
        }
        if lag > tolerance or missing_dates:
            status = STATUS_FAIL
        else:
            status = STATUS_PASS
        detail = (
            f"lag {lag} "
            f"{'<=' if lag <= tolerance else '>'} {tolerance} sessions; "
            f"sessions with rows: {covered_units}/{len(span)} "
            f"(missing: {len(missing_dates)}, pre-family sessions out of "
            f"the denominator: {pre_family_sessions}; writer-computed T+1 "
            "available_date consumed as-is)"
        )
        revision = {
            "semantics": MARGIN_DETAIL_TIME_SEMANTICS.revision_semantics,
            "revision_history_complete": (
                MARGIN_DETAIL_TIME_SEMANTICS.revision_history_complete
            ),
            "tier_limit_note": MARGIN_DETAIL_TIME_SEMANTICS.tier_limit_note,
        }
        return _check(
            SOURCE_MARGIN, SOURCE_MARGIN in required, status, coverage,
            {
                "worst_sessions": lag,
                "tolerance": tolerance,
                "same_day": True,
            },
            revision, {}, gaps, detail,
        )

    def build_industry_membership() -> SourceQualityCheck:
        if axis_block:
            return _not_computable(SOURCE_INDUSTRY_MEMBERSHIP, axis_block)
        try:
            rows = _membership_rows(config, 1)
        except Exception as exc:  # noqa: BLE001 - missing source reports
            return _not_computable(
                SOURCE_INDUSTRY_MEMBERSHIP,
                "not computable: industry_membership_pit unreadable "
                f"({type(exc).__name__}: {exc})",
            )
        gate = industry_membership_gate(
            config, axis, level=1,
            coverage_threshold=gates.INDUSTRY_MEMBERSHIP_MIN_COVERAGE,
        )
        quality = {
            "overlap_violations": gate["overlap_violations"],
            "availability_complete": gate["availability_complete"],
            "official_reconciliation": gate["official_reconciliation"],
            "mixed_format_count": sum(
                f["count"] for f in _findings_for("industry_membership_pit")
            ),
        }
        coverage = {
            "ratio": gate["coverage"],
            "denominator_rule": (
                "member-interval span sessions with a PIT-visible "
                "industry (p17 §12/§15.3)"
            ),
        }
        gaps: list[DataGap] = []
        if gate["overlap_violations"]:
            intervals = _membership_intervals(rows)
            for code, values in intervals.items():
                for date in axis:
                    covering = sum(
                        1 for in_date, out_date, _a, _i in values
                        if in_date <= date < out_date
                    )
                    if covering > 1 and len(gaps) < gap_example_limit:
                        gaps.append(DataGap(
                            "industry_membership_pit", "in_date", code,
                            date, "industry_overlap",
                            detail=(
                                "two valid industries on one day "
                                "(p17 §14.3 red line)"
                            ),
                        ))
        if not gate["availability_complete"]:
            nulls = rows[rows["available_date"].isna()]
            for entry in nulls.head(gap_example_limit).itertuples(index=False):
                gaps.append(DataGap(
                    "industry_membership_pit", "available_date",
                    str(entry.ts_code), str(entry.in_date),
                    "missing_availability",
                    detail="expanded row without a non-null available_date",
                ))
        if rows.empty:
            status = STATUS_FAIL
            detail = "industry membership covers nothing (empty table)"
        elif (
            gate["coverage"] < gates.INDUSTRY_MEMBERSHIP_MIN_COVERAGE
            or gate["overlap_violations"] > 0
            or not gate["availability_complete"]
        ):
            status = STATUS_FAIL
            detail = (
                f"coverage {gate['coverage']:.2%}, overlap "
                f"{gate['overlap_violations']}, availability_complete="
                f"{gate['availability_complete']} (p17 §12/§15.3)"
            )
        else:
            status = STATUS_PASS
            detail = (
                f"coverage {gate['coverage']:.2%} >= threshold, overlap 0, "
                "availability complete"
            )
        revision = {
            "semantics": (
                INDUSTRY_MEMBERSHIP_TIME_SEMANTICS.revision_semantics
            ),
            "revision_history_complete": (
                INDUSTRY_MEMBERSHIP_TIME_SEMANTICS.revision_history_complete
            ),
            "tier_limit_note": (
                INDUSTRY_MEMBERSHIP_TIME_SEMANTICS.tier_limit_note
            ),
        }
        return _check(
            SOURCE_INDUSTRY_MEMBERSHIP,
            SOURCE_INDUSTRY_MEMBERSHIP in required, status, coverage,
            {}, revision, quality, gaps, detail,
        )

    def build_industry_index() -> SourceQualityCheck:
        if axis_block:
            return _not_computable(SOURCE_INDUSTRY_INDEX, axis_block)
        lag = industry_index_freshness(config, axis)
        if lag is None:
            return _not_computable(
                SOURCE_INDUSTRY_INDEX,
                "not computable: the Shenwan index source has no rows "
                "(D5 industry_index_freshness)",
            )
        tolerance = gates.INDUSTRY_INDEX_MAX_LAG_SESSIONS
        status = STATUS_PASS if lag <= tolerance else STATUS_FAIL
        detail = (
            f"lag {lag} {'<=' if status == STATUS_PASS else '>'} "
            f"{tolerance} sessions (D5)"
        )
        return _check(
            SOURCE_INDUSTRY_INDEX, SOURCE_INDUSTRY_INDEX in required,
            status, {}, {"worst_sessions": lag, "tolerance": tolerance},
            {}, {}, [], detail,
        )

    def build_financial() -> SourceQualityCheck:
        if axis_block:
            return _not_computable(SOURCE_FINANCIAL, axis_block)
        if v2_db is None:
            return _not_computable(
                SOURCE_FINANCIAL,
                f"not computable: v2 store unreadable ({v2_error})",
            )
        statement_findings = (
            _findings_for("income_statement_pit")
            + _findings_for("balance_sheet_pit")
            + _findings_for("cashflow_statement_pit")
        )
        if statement_findings:
            return _not_computable(
                SOURCE_FINANCIAL,
                "not computable: mixed-format date axis blocked the "
                "comparison (I2 fail-closed): "
                f"{statement_findings[0]['table']}."
                f"{statement_findings[0]['field']}",
            )
        universe, universe_error = _universe()
        if universe_error is not None:
            return _not_computable(SOURCE_FINANCIAL, universe_error)
        codes = universe[0]
        horizon = _horizon()
        if horizon is None:
            return _not_computable(
                SOURCE_FINANCIAL,
                "not computable: the v2 collection horizon is empty",
            )
        mn, mx, _window = horizon
        window_start = f"{int(mn[:4]) - 1}{mn[4:]}"
        quarter_ends = _quarter_ends(window_start, mx)
        listing_dates = _universe_list_dates(config, list(codes))
        cv = _financial_coverage(
            v2_db, list(codes), quarter_ends,
            listing_dates=listing_dates,
            gap_example_limit=gap_example_limit,
        )
        if cv.ratio is None:
            return _not_computable(
                SOURCE_FINANCIAL, f"not computable: {cv.detail}",
            )
        gaps = [
            DataGap(
                "financial_version_tables", "end_date", code, period,
                "missing_report_period",
                detail=(
                    "post-listing report period without a fully "
                    "non-null version row"
                ),
            )
            for code, period in cv.gap_examples
        ]
        coverage = {
            "ratio": cv.ratio,
            "expected_units": cv.expected_units,
            "covered_units": cv.covered_units,
            "gap_count": cv.gap_count,
            "denominator_rule": (
                "universe ∧ listed ∧ report period (pre-listing "
                "excluded; p17 §12 + t73)"
            ),
        }
        revision = {
            "semantics": FINANCIAL_VERSION_TIME_SEMANTICS.revision_semantics,
            "revision_history_complete": (
                FINANCIAL_VERSION_TIME_SEMANTICS.revision_history_complete
            ),
            "tier_limit_note": (
                FINANCIAL_VERSION_TIME_SEMANTICS.tier_limit_note
            ),
            "versioned_keys": cv.versioned_keys,
            "total_keys": cv.total_keys,
        }
        quality = {
            "non_quarter_end_rows": cv.non_quarter_end_rows,
            "mixed_format_count": sum(
                f["count"] for f in statement_findings
            ),
        }
        status = (
            STATUS_PASS
            if cv.ratio >= gates.FINANCIAL_REPORT_COVERAGE_MIN
            else STATUS_FAIL
        )
        return _check(
            SOURCE_FINANCIAL, SOURCE_FINANCIAL in required, status,
            coverage, {}, revision, quality, gaps, cv.detail,
        )

    def build_manifest() -> SourceQualityCheck:
        if v2_db is None:
            return _not_computable(
                SOURCE_DATASET_MANIFEST,
                f"not computable: v2 store unreadable ({v2_error})",
            )
        ok, detail = _dataset_manifest_check(v2_db)
        if ok is None:
            return _not_computable(SOURCE_DATASET_MANIFEST, detail)
        status = STATUS_PASS if ok else STATUS_FAIL
        return _check(
            SOURCE_DATASET_MANIFEST, SOURCE_DATASET_MANIFEST in required,
            status, {}, {}, {}, {}, [], detail,
        )

    def build_index_weight() -> SourceQualityCheck:
        if axis_block:
            return _not_computable(SOURCE_INDEX_WEIGHT, axis_block)
        if v2_db is None:
            return _not_computable(
                SOURCE_INDEX_WEIGHT,
                f"not computable: v2 store unreadable ({v2_error})",
            )
        if _findings_for("index_weight_pit"):
            return _not_computable(
                SOURCE_INDEX_WEIGHT,
                "not computable: mixed-format date axis blocked the "
                "comparison (I2 fail-closed): index_weight_pit",
            )
        try:
            rows = v2_db.query(
                "SELECT index_code, con_code, trade_date, available_date "
                "FROM index_weight_pit"
            )
        except Exception as exc:  # noqa: BLE001 - table may not exist
            return _not_computable(
                SOURCE_INDEX_WEIGHT,
                f"not computable: index_weight_pit is missing "
                f"({type(exc).__name__}: {exc})",
            )
        if rows.empty:
            return _not_computable(
                SOURCE_INDEX_WEIGHT,
                "not computable: index_weight_pit is empty",
            )
        horizon = _horizon()
        mx = horizon[1] if horizon is not None else ""
        snapshots = sorted({str(v) for v in rows["trade_date"].tolist()})
        intervals = month_end_snapshot_intervals(snapshots, axis)
        expected_from = {
            snap: visible_from for snap, visible_from, _to in intervals
        }
        gaps: list[DataGap] = []
        violation_count = 0
        for entry in rows.itertuples(index=False):
            snap = str(entry.trade_date)
            expected_visible = expected_from.get(snap)
            if expected_visible is None:
                continue  # snapshot beyond the calendar horizon
            if str(entry.available_date) != expected_visible:
                violation_count += 1
                if len(gaps) < gap_example_limit:
                    gaps.append(DataGap(
                        "index_weight_pit", "available_date",
                        str(entry.con_code), snap,
                        "visibility_rule_violation",
                        detail=(
                            f"available_date {entry.available_date} != the "
                            f"I5 rule {expected_visible}"
                        ),
                    ))
        snap_months = {snap[:6] for snap in snapshots}
        expected_months = 0
        if horizon is not None:
            mn = horizon[0]
            year, month = int(mn[:4]), int(mn[4:6])
            end_year, end_month = int(mx[:4]), int(mx[4:6])
            while (year, month) <= (end_year, end_month):
                last_day = _calendar.monthrange(year, month)[1]
                if int(f"{year}{month:02d}{last_day:02d}") <= int(mx):
                    expected_months += 1
                month += 1
                if month == 13:
                    year, month = year + 1, 1
        coverage = {
            "months_with_snapshot": len(snap_months),
            "expected_months_in_horizon": expected_months,
            "ratio": None,  # month granularity is descriptive; violations decide
            "denominator_rule": (
                "calendar months whose month-end falls inside the "
                "collection horizon"
            ),
        }
        status = STATUS_FAIL if violation_count else STATUS_PASS
        detail = (
            f"I5 visibility-rule violations: {violation_count} "
            "(the store's available_date must equal the first open "
            f"session after the month end); snapshots: {len(snapshots)}"
        )
        revision = {
            "semantics": INDEX_WEIGHT_TIME_SEMANTICS.revision_semantics,
            "revision_history_complete": (
                INDEX_WEIGHT_TIME_SEMANTICS.revision_history_complete
            ),
            "tier_limit_note": INDEX_WEIGHT_TIME_SEMANTICS.tier_limit_note,
        }
        quality = {"mixed_format_count": sum(
            f["count"] for f in _findings_for("index_weight_pit")
        )}
        return _check(
            SOURCE_INDEX_WEIGHT, SOURCE_INDEX_WEIGHT in required, status,
            coverage, {}, revision, quality, gaps, detail,
        )

    def build_events() -> SourceQualityCheck:
        if axis_block:
            return _not_computable(SOURCE_EVENTS_AUDIT, axis_block)
        if v2_db is None:
            return _not_computable(
                SOURCE_EVENTS_AUDIT,
                f"not computable: v2 store unreadable ({v2_error})",
            )
        horizon = _horizon()
        mx = horizon[1] if horizon is not None else ""
        gaps: list[DataGap] = []
        lags: dict[str, int] = {}
        missing: list[str] = []
        for table in _EVENT_TABLES:
            latest = _max_column(v2_db, table, "available_date")
            if latest is None:
                missing.append(table)
                continue
            lag = _lag_sessions(axis, latest, same_day=False)
            lags[table] = lag
            if lag > gates.FRESHNESS_TOLERANCE_SESSIONS:
                gaps.append(DataGap(
                    table, "available_date", None, latest, "stale_source",
                    detail=(
                        f"lag {lag} > "
                        f"{gates.FRESHNESS_TOLERANCE_SESSIONS} sessions"
                    ),
                ))
        if len(missing) == len(_EVENT_TABLES):
            return _not_computable(
                SOURCE_EVENTS_AUDIT,
                "not computable: none of the event tables exist (the F3 "
                "expectation-event domain is not admitted; p26 ruling 4) "
                "— informational, never a global blocker (I10)",
            )
        if missing:
            return _not_computable(
                SOURCE_EVENTS_AUDIT,
                "not computable: event tables missing "
                f"({', '.join(missing)}); partial event coverage cannot "
                "be graded fail-closed (I14)",
            )
        holder_latest = _max_column(v2_db, "holder_number_pit", "end_date")
        if holder_latest is not None and mx and holder_latest < mx:
            gaps.append(DataGap(
                "holder_number_pit", "end_date", None, holder_latest,
                "stale_source",
                detail=(
                    "holder_number history ends before the collection "
                    "horizon; F3 expansion must re-check this source "
                    "separately (p29 §1.7) — informational, not "
                    "blocking (I10)"
                ),
            ))
        status = (
            STATUS_FAIL
            if any(gap.kind == "stale_source" for gap in gaps)
            else STATUS_PASS
        )
        detail = (
            "event-family audit (informational; not in the p17 §12 "
            f"verdict set): lags {lags}; four-subitem boundary — this "
            "informational family reports lag/staleness only (coverage "
            "and quality live in the stale_source gaps; the revision "
            "semantics come from its unrestorable declaration)"
        )
        revision = {
            "semantics": EVENTS_AUDIT_TIME_SEMANTICS.revision_semantics,
            "revision_history_complete": (
                EVENTS_AUDIT_TIME_SEMANTICS.revision_history_complete
            ),
            "tier_limit_note": EVENTS_AUDIT_TIME_SEMANTICS.tier_limit_note,
        }
        return _check(
            SOURCE_EVENTS_AUDIT, SOURCE_EVENTS_AUDIT in required, status,
            {}, {"per_table": lags}, revision, {}, gaps, detail,
        )

    builders = {
        SOURCE_DAILY: build_daily,
        SOURCE_PRICE_LIMIT: build_price_limit,
        SOURCE_MARGIN: build_margin,
        SOURCE_INDUSTRY_MEMBERSHIP: build_industry_membership,
        SOURCE_INDUSTRY_INDEX: build_industry_index,
        SOURCE_FINANCIAL: build_financial,
        SOURCE_DATASET_MANIFEST: build_manifest,
        SOURCE_INDEX_WEIGHT: build_index_weight,
        SOURCE_EVENTS_AUDIT: build_events,
    }
    families: dict[str, SourceQualityCheck] = {}
    try:
        for family in QUALITY_FAMILIES:
            if family in evaluate:
                families[family] = builders[family]()
            else:
                families[family] = _not_required(family)
    finally:
        for db_handle in (primary_db, v2_db):
            if db_handle is not None:
                try:
                    db_handle.close()
                except Exception:  # noqa: BLE001 - close is best effort
                    pass

    return DataQualityReport(
        report_version=DATA_QUALITY_REPORT_VERSION,
        evaluation_day=day,
        primary_db=str(config.duckdb_path),
        v2_db=str(resolve_v2_duckdb_path(config)),
        families=families,
        spaces=dict(_FAMILY_SPACES),
        mixed_format_findings=tuple(findings),
        threshold_sources=dict(_THRESHOLD_SOURCES),
    )
