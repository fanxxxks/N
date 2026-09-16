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

from .config import resolve_v2_duckdb_path
from .db import AshareDB, sql_quoted_list
from .fundamentals import _cached_or_fetch, _read_cached, _write_cached
from .gates import FRESHNESS_TOLERANCE_SESSIONS
from .pit_import import SourceTimeSemantics
from .universe import OPEN_ENDED_OUT_DATE

#: B01 I3 (p29): the margin family's four-time semantics.  The T+1
#: ``available_date`` is writer-computed and consumed as-is — the report
#: never re-derives it (p17 §6.1; t22 fix).
MARGIN_DETAIL_TIME_SEMANTICS = SourceTimeSemantics(
    family="margin_detail",
    tables=("margin_detail_pit",),
    economic_time="trade_date (the session the margin detail describes)",
    availability_time=(
        "available_date (writer-computed T+1; consumed, never re-derived)"
    ),
    collection_authority="source_ingest_runs",
    revision_semantics="single_final",
    revision_history_complete=True,
    tier_limit_note="",
    visibility_rule=(
        "the value visible at session s is the detail whose available_date "
        "== s (the previous session's detail, t22 in-place fix)"
    ),
)

#: B01 I3 (p29): the industry membership family's four-time semantics
#: (expanded rows per p17 勘误 #2; one valid industry per stock/level/day).
INDUSTRY_MEMBERSHIP_TIME_SEMANTICS = SourceTimeSemantics(
    family="industry_membership",
    tables=("industry_membership_pit",),
    economic_time="[in_date, out_date) half-open membership interval",
    availability_time="available_date (p17 §6.1)",
    collection_authority="source_ingest_runs",
    revision_semantics="single_final",
    revision_history_complete=True,
    tier_limit_note="",
    visibility_rule=(
        "an assignment is visible from max(in_date, available_date) until "
        "out_date; overlapping intervals raise IndustryMembershipError "
        "(p17 §14.3 fail-closed)"
    ),
)

# Vocabulary feature names produced by this module.
EXTERNAL_FACTOR_NAMES = (
    "MARGIN_BALANCE_CHG",
    "INDUSTRY_MOMENTUM",
    "MARGIN_CROWD_60",
    # F2 batch (p24, user-approved): the margin-crowding cross-sectional
    # intensity factors (the raw ratios; the xcs-z applied loader-side).
    "MARGIN_BALANCE_TO_CIRC_MV",
    "MARGIN_BUY_PRESSURE_5",
)

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


class IndustryMembershipError(ValueError):
    """The membership intervals violate p17 §14.3: the same stock at the
    same level has more than one valid industry on some day (overlap).
    Fail-closed — never silently merged."""


def _membership_rows(config, level: int) -> pd.DataFrame:
    """Expanded membership contract rows of one level from the v2 PIT table
    (p17 §5.2 table 8, 勘误 #2).  A missing source propagates as a duckdb
    error for the caller to degrade on (Tier C auxiliary semantics)."""

    v2_path = resolve_v2_duckdb_path(config)
    with AshareDB(v2_path, read_only=True) as db:
        return db.query(
            "SELECT ts_code, index_code, in_date, out_date, is_new, "
            "industry_name, available_date FROM industry_membership_pit "
            f"WHERE level = {int(level)} ORDER BY ts_code, in_date"
        )


def _membership_intervals(
    rows: pd.DataFrame,
) -> dict[str, list[tuple[str, str, str, str]]]:
    """``ts_code -> sorted (in_date, out_date, available_date, index_code)``
    half-open intervals; an empty ``out_date`` becomes the open-ended
    sentinel (still a member, p17 §5.2 table 8)."""

    intervals: dict[str, list[tuple[str, str, str, str]]] = {}
    for row in rows.itertuples(index=False):
        out_date = (
            str(row.out_date)
            if pd.notna(row.out_date) and str(row.out_date)
            else OPEN_ENDED_OUT_DATE
        )
        available_date = (
            str(row.available_date) if pd.notna(row.available_date) else ""
        )
        intervals.setdefault(str(row.ts_code), []).append(
            (
                str(row.in_date),
                out_date,
                available_date,
                str(row.index_code),
            )
        )
    for values in intervals.values():
        values.sort(key=lambda item: (item[0], item[1]))
    return intervals


def build_industry_member_frame(
    config,
    ts_codes: list[str],
    dates: list[str],
    *,
    level: int = 1,
) -> pd.DataFrame:
    """Build the ``[stock x date]`` Shenwan industry-code frame from the
    v2 ``industry_membership_pit`` intervals (p17, D5 原位切换).

    Semantics (p17 §6.3/§14.3, single implementation):

    * an assignment is valid on the half-open interval
      ``[in_date, out_date)``; ``out_date`` NULL means still a member;
    * it becomes *visible* only from ``available_date`` (p17 §6.1: the
      session after ``in_date``) — the inclusion day itself carries no
      industry yet, and nothing exists before inclusion;
    * the same stock at the same level has at most one valid industry
      per day — overlapping intervals raise
      :class:`IndustryMembershipError` (fail-closed, never silently
      merged);
    * the default level is Shenwan L1 (the historical IND_REL grouping);
      L2/L3 rows never leak into it.

    Unmapped / not-yet-visible cells stay NaN so the industry-relative
    factors never fabricate a grouping.  A missing v2 source degrades to
    an all-NaN frame with a warning (Tier C auxiliary semantics); only
    the data-integrity violation (overlap) fails closed.
    """

    frame = pd.DataFrame(np.nan, index=ts_codes, columns=dates, dtype=object)
    if not ts_codes:
        return frame
    try:
        rows = _membership_rows(config, level)
    except Exception as exc:  # noqa: BLE001 - source may not exist yet.
        logger.warning(f"Industry membership frame unavailable: {exc}")
        return frame
    intervals = _membership_intervals(rows)

    # Fail-closed uniqueness sweep (p17 §14.3): interval overlap means
    # some day carries two valid industries for one stock/level.
    for code, values in intervals.items():
        previous_out: str | None = None
        for in_date, out_date, _available, _index_code in values:
            if previous_out is not None and in_date < previous_out:
                raise IndustryMembershipError(
                    f"overlapping industry_membership_pit intervals for "
                    f"({code}, level={level}): {in_date} < {previous_out} "
                    "(p17 §14.3: one valid industry per stock/level/day)"
                )
            previous_out = (
                out_date if previous_out is None else max(previous_out, out_date)
            )

    # Fill: each interval contributes one contiguous visible column slice
    # [max(in_date, available_date), out_date) clipped to the window.
    date_index = {date: position for position, date in enumerate(dates)}
    known_codes = set(ts_codes)
    for code, values in intervals.items():
        if code not in known_codes:
            continue
        row = frame.index.get_loc(code)
        for in_date, out_date, available_date, index_code in values:
            start = max(in_date, available_date) if available_date else in_date
            start_idx = date_index.get(start)
            if start_idx is None:
                inside = [d for d in dates if start <= d < out_date]
                if not inside:
                    continue  # interval entirely outside the window
                start_idx = date_index[inside[0]]
            end_idx = min(date_index.get(out_date, len(dates)), len(dates))
            if start_idx < end_idx:
                frame.iloc[row, start_idx:end_idx] = index_code
    return frame


def load_industry_taxonomy(config) -> pd.DataFrame:
    """The v2 ``industry_taxonomy`` rows (index_code -> generation/name),
    the single traceability path for the 2021 taxonomy switch."""

    v2_path = resolve_v2_duckdb_path(config)
    with AshareDB(v2_path, read_only=True) as db:
        return db.query(
            "SELECT index_code, industry_name, level, industry_code, src "
            "FROM industry_taxonomy ORDER BY index_code"
        )


def industry_index_freshness(
    config,
    dates: list[str],
    tolerance: int = FRESHNESS_TOLERANCE_SESSIONS,
) -> int | None:
    """Open-session lag of the Shenwan industry-index close source (p17
    §12: 滞后 = 最近会话 − 该源最新会话, tolerance 3).  ``None`` when the
    source has no rows (unusable — reported by the caller); the lag count
    otherwise.  ``lag > tolerance`` marks the source stale for the D7
    qualification gate (the RunSpec rejection wiring is t19's)."""

    try:
        with AshareDB(config.duckdb_path, read_only=True) as db:
            latest = db.query(
                f"SELECT MAX(trade_date) AS latest FROM {config.sw_index_table}"
            )
    except Exception:  # noqa: BLE001 - table may not exist yet.
        return None
    latest_value = latest.iloc[0]["latest"] if not latest.empty else None
    if latest_value is None or pd.isna(latest_value):
        return None
    latest_date = str(latest_value)
    return len([date for date in dates if date > latest_date])


def industry_membership_gate(
    config,
    dates: list[str],
    *,
    level: int = 1,
    coverage_threshold: float = 0.98,
) -> dict[str, object]:
    """Machine-checkable D5 verification gate (p17 §7/§12 C→B thresholds).

    * ``coverage`` — PIT-visible industry sessions over the interval span
      sessions of every member stock (p17 §12: 有效区间覆盖 ≥ 98%);
    * ``overlap_violations`` — (stock, day) cells with two interval-valid
      industries (must be 0; p17 §15.3 red line);
    * ``availability_complete`` — every expanded row carries a non-null
      ``available_date`` (p17 §6.1);
    * ``official_reconciliation`` — the third pre-registered sub-gate is
      an external reconciliation against official SW announcements: it is
      NOT verifiable in-repo, so the gate can never upgrade the source to
      Tier B here (p17 §7: 未验证维持 C) — the honest gap travels with
      the result instead of being silently upgraded.
    """

    result: dict[str, object] = {
        "coverage": 0.0,
        "overlap_violations": 0,
        "availability_complete": False,
        "official_reconciliation": "not_verifiable_in_repo",
        "coverage_threshold": coverage_threshold,
        "upgrade_to_b": False,
    }
    try:
        rows = _membership_rows(config, level)
    except Exception as exc:  # noqa: BLE001 - missing source reports, not raises.
        logger.warning(f"industry membership gate unavailable: {exc}")
        return result
    if rows.empty:
        # An empty membership table is "complete but covers nothing".
        result["availability_complete"] = True
        return result
    result["availability_complete"] = not bool(rows["available_date"].isna().any())

    intervals = _membership_intervals(rows)
    span_total = 0
    visible_total = 0
    overlaps = 0
    for _code, values in intervals.items():
        covered = sorted(
            {
                date
                for in_date, out_date, _available, _index in values
                for date in dates
                if in_date <= date < out_date
            }
        )
        span_total += len(covered)
        overlaps += sum(
            1
            for date in covered
            if sum(
                1
                for in_date, out_date, _available, _index in values
                if in_date <= date < out_date
            )
            > 1
        )
        visible_total += sum(
            1
            for date in covered
            if any(
                in_date <= date < out_date and available_date and available_date <= date
                for in_date, out_date, available_date, _index in values
            )
        )
    result["overlap_violations"] = overlaps
    result["coverage"] = (visible_total / span_total) if span_total else 0.0
    result["upgrade_to_b"] = bool(
        result["coverage"] >= coverage_threshold
        and overlaps == 0
        and result["availability_complete"]
        and result["official_reconciliation"] == "verified"
    )
    return result


def _margin_crowd(wide: pd.DataFrame, window: int = _MARGIN_CROWD_WINDOW) -> pd.DataFrame:
    """Margin-balance crowding: the balance relative to its own trailing mean
    (P9 §5.4).  Strict warmup: the first ``window - 1`` sessions stay NaN
    (no fabricated baseline from partial history); missing values stay NaN.
    """

    base = wide.replace([np.inf, -np.inf], np.nan)
    baseline = base.T.rolling(window, min_periods=window).mean().T
    return (base / baseline).replace([np.inf, -np.inf], np.nan)


def _canonical_sw_code(code: str) -> str:
    """One canonical Shenwan index-code form (``801010``) for the
    membership/index-close join: the v2 membership carries ``.SI``-suffixed
    codes while the legacy close table carries bare codes (p17 §14; the
    t18 INDUSTRY_MOMENTUM coverage-0.000 finding)."""

    return str(code)[:-3] if str(code).endswith(".SI") else str(code)


def _canonicalize_sw_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`_canonical_sw_code` to every finite cell of a
    ``[stock x date]`` industry-code frame (vectorized per cell)."""

    if frame is None or frame.empty:
        return frame
    arr = frame.to_numpy(dtype=object)
    flat = [
        _canonical_sw_code(cell) if isinstance(cell, str) and cell else cell
        for cell in arr.ravel()
    ]
    return pd.DataFrame(
        np.asarray(flat, dtype=object).reshape(arr.shape),
        index=frame.index,
        columns=frame.columns,
    )


def build_capital_frames(
    config,
    ts_codes: list[str],
    dates: list[str],
) -> dict[str, pd.DataFrame]:
    """Build ``[stock x date]`` frames for the capital-flow features.

    Margin semantics read the v2 ``margin_detail_pit`` source (p17 §5.2
    table 18) pivoted by ``available_date`` — the p17 §6.1 T+1 visibility:
    the value visible at session *s* is the detail whose
    ``available_date == s`` (the previous session's margin detail), never
    the detail of *s* itself (t22 in-place fix; the legacy
    ``margin_balance`` trade-date pivot carried no availability lag).

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
    # B04 (p33 I3): the window bounds are pushed into the SQL reads.  The
    # pivots below `reindex(columns=dates)` anyway — rows outside
    # [min(dates), max(dates)] were read and then dropped, so filtering
    # them at the source is output-equivalent (the shift/rolling windows
    # operate on the reindexed frame's window columns only).
    bounds = (min(dates), max(dates)) if dates else None
    # The pivots' `aggfunc="last"` tie-breaking rests on the sources'
    # PRIMARY KEY (ts_code, trade_date): duplicate (code, key) rows cannot
    # occur in the production tables (p17 §4.1 DDL authority), and the
    # filter column below is a member of that key — a pivot group can
    # never be partially filtered (dual-review F6 note).

    def _window_clause(column: str, *, keyword: str = "AND") -> str:
        return (
            f"{keyword} {column} BETWEEN ? AND ?"
            if bounds is not None
            else ""
        )

    def _window_params() -> list[str]:
        return list(bounds) if bounds is not None else []

    try:
        v2_path = resolve_v2_duckdb_path(config)
        with AshareDB(v2_path, read_only=True) as v2db:
            margin = v2db.query(
                "SELECT ts_code, available_date, rzye, rzmre FROM margin_detail_pit "
                f"WHERE ts_code IN ({quoted}) AND available_date IS NOT NULL "
                f"{_window_clause('available_date')}",
                _window_params(),
            )
    except Exception as exc:  # noqa: BLE001 - tables may not exist yet.
        # The v2 margin source is expected to be absent in pre-D3 /
        # fixture databases — the same expected-absent convention as the
        # family-⑤ as-of frames (logger.debug, not a warning).
        logger.debug(f"Margin frames unavailable: {exc}")
        margin = pd.DataFrame()
    try:
        # F2 denominators (p24 §2): the session-t float market cap from
        # daily_basic_pit — trade_date-keyed (the D4-era daily-basic
        # convention: the session-t basic values are known at the
        # session-t close; no available_date column exists).
        v2_path = resolve_v2_duckdb_path(config)
        with AshareDB(v2_path, read_only=True) as v2db:
            basics = v2db.query(
                "SELECT ts_code, trade_date, circ_mv FROM daily_basic_pit "
                f"WHERE ts_code IN ({quoted}) "
                f"{_window_clause('trade_date')}",
                _window_params(),
            )
    except Exception as exc:  # noqa: BLE001 - tables may not exist yet.
        logger.debug(f" daily_basic circ_mv unavailable: {exc}")
        basics = pd.DataFrame()
    try:
        with AshareDB(config.duckdb_path, read_only=True) as db:
            sw_index = db.query(
                f"SELECT index_code, trade_date, close FROM {config.sw_index_table} "
                f"{_window_clause('trade_date', keyword='WHERE')}",
                _window_params(),
            )
    except Exception as exc:  # noqa: BLE001 - tables may not exist yet.
        logger.warning(f"Industry index closes unavailable: {exc}")
        sw_index = pd.DataFrame()
    # D5: the membership dimension comes from the v2 PIT intervals; the
    # industry index closes stay on the legacy source (no v2 SW-close
    # table).  The legacy close table carries bare codes (``801010``)
    # while the v2 membership carries ``801010.SI`` — canonicalize both
    # sides to one form so the per-date membership join cannot silently
    # match nothing (t18 finding: INDUSTRY_MOMENTUM coverage 0.000).
    industry_frame = build_industry_member_frame(config, ts_codes, dates)
    industry_frame = _canonicalize_sw_codes(industry_frame)
    if not sw_index.empty:
        sw_index["index_code"] = sw_index["index_code"].map(_canonical_sw_code)

    if not margin.empty:
        wide = (
            margin.pivot_table(
                index="ts_code", columns="available_date", values="rzye",
                aggfunc="last",
            )
            .reindex(index=ts_codes, columns=dates)
        )
        with np.errstate(all="ignore"):
            change = wide / wide.shift(_MARGIN_WINDOW, axis=1) - 1.0
        frames["MARGIN_BALANCE_CHG"] = change.replace([np.inf, -np.inf], np.nan)
        frames["MARGIN_CROWD_60"] = _margin_crowd(wide, _MARGIN_CROWD_WINDOW)

        # F2 batch (p24, user-approved): the margin-crowding cross-sectional
        # intensity factors — the RAW ratios (the per-date cross-sectional
        # z is applied loader-side in ashare_model, the dependency-correct
        # layer for the single xcs-z implementation; p24 §2 formulas).
        if not basics.empty:
            circ_wide = (
                basics.pivot_table(
                    index="ts_code", columns="trade_date", values="circ_mv",
                    aggfunc="last",
                )
                .reindex(index=ts_codes, columns=dates)
            )
            balance_ratio = (wide / circ_wide).replace(
                [np.inf, -np.inf], np.nan
            )
            frames["MARGIN_BALANCE_TO_CIRC_MV"] = balance_ratio
            buy_wide = (
                margin.pivot_table(
                    index="ts_code", columns="available_date", values="rzmre",
                    aggfunc="last",
                )
                .reindex(index=ts_codes, columns=dates)
            )
            # The five most recent VISIBLE sessions at each column: the
            # rolling 5-column sum over the available_date pivots (the T+1
            # visibility makes column k carry the session-k detail).  The
            # transpose pattern: the newer pandas rolling() has no axis.
            buy_5 = buy_wide.T.rolling(5, min_periods=5).sum().T
            pressure_ratio = (buy_5 / circ_wide).replace(
                [np.inf, -np.inf], np.nan
            )
            frames["MARGIN_BUY_PRESSURE_5"] = pressure_ratio

    if not sw_index.empty and not industry_frame.empty:
        index_wide = sw_index.pivot_table(
            index="index_code", columns="trade_date", values="close", aggfunc="last"
        ).reindex(columns=dates)
        with np.errstate(all="ignore"):
            industry_ret = index_wide / index_wide.shift(_INDUSTRY_WINDOW, axis=1) - 1.0
        momentum = pd.DataFrame(
            np.nan, index=industry_frame.index, columns=industry_frame.columns
        )
        # Per-date membership: each industry's return series is applied
        # exactly where the PIT frame assigns that industry on that date.
        for industry in industry_ret.index:
            momentum = momentum.mask(
                industry_frame == industry, industry_ret.loc[industry], axis=1
            )
        frames["INDUSTRY_MOMENTUM"] = momentum

    return frames
