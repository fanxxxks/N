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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable
from zoneinfo import ZoneInfo

from ashare_data.capital_flow import (
    industry_index_freshness,
    industry_membership_gate,
)
from ashare_data.config import resolve_v2_duckdb_path
from ashare_data.db import AshareDB
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
from ashare_data.pit_import import _FINANCIAL_VALUE_COLUMNS
from ashare_model.feature_registry import PitLevel, family_of, pit_level_of
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


def _financial_coverage(
    db: AshareDB,
    universe_codes: list[str],
    quarter_ends: list[str],
) -> tuple[float | None, str]:
    """Report-period coverage of the three financial version tables
    (p17 §12: 当前正式 scope 报告期覆盖 ≥ 90%).  Expected = universe codes
    × quarter-end periods in the RunSpec window (extended one year back
    for the YoY/TTM lookbacks the fundamental features consume); covered
    = keys with a version row whose every core column is non-null.  The
    worst table decides the family verdict.  Ratio ``None`` = not
    computable (the detail carries the reason)."""

    per_table: list[tuple[str, float]] = []
    for table, core_columns in _FINANCIAL_TABLES.items():
        try:
            rows = db.query(
                f"SELECT ts_code, end_date, {', '.join(core_columns)} "
                f"FROM {table}"
            )
        except Exception as exc:  # noqa: BLE001 - missing table: not computable
            return None, f"{table} not readable ({type(exc).__name__})"
        covered: set[tuple[str, str]] = set()
        if not rows.empty:
            for row in rows.itertuples(index=False):
                values = [getattr(row, column) for column in core_columns]
                if not any(_is_missing(value) for value in values):
                    covered.add((str(row.ts_code), str(row.end_date)))
        expected = {
            (code, period) for code in universe_codes for period in quarter_ends
        }
        if not expected:
            continue  # no report periods in scope: nothing required here
        ratio = len(covered & expected) / len(expected)
        per_table.append((table, ratio))
    if not per_table:
        return None, "no report periods in the formal scope"
    worst_table, worst_ratio = min(per_table, key=lambda entry: entry[1])
    detail = "; ".join(f"{table}={ratio:.2%}" for table, ratio in per_table)
    return worst_ratio, f"{detail} (worst: {worst_table})"


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
                ratio, detail = _financial_coverage(
                    v2_db, list(universe_codes), quarter_ends
                )
                if ratio is None:
                    checks[SOURCE_FINANCIAL] = fail(
                        SOURCE_FINANCIAL,
                        f"not computable: {detail}",
                    )
                elif ratio < FINANCIAL_REPORT_COVERAGE_MIN:
                    checks[SOURCE_FINANCIAL] = fail(
                        SOURCE_FINANCIAL,
                        f"report-period coverage {ratio:.2%} < "
                        f"{FINANCIAL_REPORT_COVERAGE_MIN:.0%} ({detail}; "
                        f"periods {quarter_ends[0]}..{quarter_ends[-1]})",
                    )
                else:
                    checks[SOURCE_FINANCIAL] = SourceCheck(
                        SOURCE_FINANCIAL, True, True,
                        f"report-period coverage {ratio:.2%} >= "
                        f"{FINANCIAL_REPORT_COVERAGE_MIN:.0%} ({detail})",
                    )

        if SOURCE_DATASET_MANIFEST in required:
            if v2_db is None:
                checks[SOURCE_DATASET_MANIFEST] = fail(
                    SOURCE_DATASET_MANIFEST,
                    f"not computable: v2 store unreadable ({v2_error})",
                )
            else:
                try:
                    manifest = latest_manifest(v2_db, TUSHARE_V2_CONFIG)
                except Exception as exc:  # noqa: BLE001 - no manifest table yet
                    manifest = None
                    manifest_error = f"{type(exc).__name__}: {exc}"
                else:
                    manifest_error = None
                if manifest is None:
                    checks[SOURCE_DATASET_MANIFEST] = fail(
                        SOURCE_DATASET_MANIFEST,
                        "not computable: the v2 store carries no dataset "
                        "manifest (identity cannot be established"
                        + (f"; {manifest_error})" if manifest_error else ")"),
                    )
                else:
                    report = verify_dataset(
                        v2_db, TUSHARE_V2_CONFIG, expected=manifest,
                        verify="fast",
                    )
                    if not report.ok:
                        failed = [
                            f"{entry.table}: {entry.detail}"
                            for entry in report.tables.values()
                            if not entry.ok
                        ]
                        checks[SOURCE_DATASET_MANIFEST] = fail(
                            SOURCE_DATASET_MANIFEST,
                            "manifest verification failed: "
                            + "; ".join(failed),
                        )
                    else:
                        checks[SOURCE_DATASET_MANIFEST] = SourceCheck(
                            SOURCE_DATASET_MANIFEST, True, True,
                            f"manifest {manifest.dataset_id[:12]}… "
                            "verified (fast)",
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
