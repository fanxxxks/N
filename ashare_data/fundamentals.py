"""Point-in-time fundamental pipeline (quarterly, disclosure-season aligned).

Sources (all free via AkShare):
* Eastmoney 业绩报表 (``stock_yjbb_em``, whole market per quarter): the
  primary source -- cumulative year-to-date EPS/BVPS/ROE/gross margin/
  revenue/profit/YoY growth.  The endpoint's own announcement column
  carries restatement dates (each quarter is re-published in the following
  annual-report season), so the point-in-time date is set to the END of
  the quarter's statutory disclosure season (Q1 -> 04-30, H1 -> 08-31,
  Q3 -> 10-31, annual -> 04-30 next year): conservative, never visible
  early, and independent of any per-stock announcement feed.
* Sina financial indicators (``stock_financial_analysis_indicator``, per
  stock): supplements ROA and the debt ratio; the disclosure date comes
  from the matching earnings-table row (rows without a match are dropped,
  never guessed).
* Eastmoney dividend records (``stock_fhps_detail_em``, per stock): the
  reported cash dividend yield, aligned to the exact ex-dividend date
  (``除权除息日``), kept in its own ``dividend_announce`` column so
  merging sources never moves a field's visibility date.

Everything is normalized to canonical rows keyed by
``(ts_code, report_date)`` and stored in the ``fundamental_pit`` table.
:func:`build_pit_frames` turns the rows into disclosure-aligned
``[stock x date]`` frames: a value is only visible from its disclosure
date onwards and fills forward until the next one, so no future
information can leak into a training window.  Restatements are not
tracked: the latest synced value is applied from the disclosure date.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from .db import AshareDB, sql_quoted_list
from .processor import is_valid_a_share_code

# Feature names produced by the point-in-time pipeline (MARKET_CAP is a
# local daily-bar factor, not a PIT field).  The four family-⑤ features
# (P13, docs/p13_fundamental_fields_contract.md §5.3) consume the cash-flow
# statement and balance-sheet fields synced alongside the earnings table.
FUNDAMENTAL_PIT_NAMES = (
    "PE_TTM",
    "PB",
    "PS_TTM",
    "ROE",
    "ROA",
    "GROSS_MARGIN",
    "NET_MARGIN",
    "REVENUE_YOY",
    "PROFIT_YOY",
    "DEBT_RATIO",
    "DIVIDEND_YIELD",
    "CASHFLOW_QUALITY",
    "ACCRUALS",
    "ASSET_GROWTH",
    "EARNINGS_ACCEL",
)

# Level fields that fill forward from their announcement date unchanged.
_LEVEL_COLUMNS = {
    "ROE": "roe",
    "ROA": "roa",
    "GROSS_MARGIN": "gross_margin",
    "NET_MARGIN": "net_margin",
    "REVENUE_YOY": "revenue_yoy",
    "PROFIT_YOY": "profit_yoy",
    "DEBT_RATIO": "debt_ratio",
    "DIVIDEND_YIELD": "dividend_yield",
}

# Rows synced from the bulk earnings endpoint carry these columns.
_EARNINGS_COLUMNS = [
    "ts_code",
    "report_date",
    "announce_date",
    "eps_cum",
    "bvps",
    "roe",
    "gross_margin",
    "net_margin",
    "revenue_cum",
    "profit_cum",
    "revenue_yoy",
    "profit_yoy",
]

_STALE_AFTER_DAYS = 45
# A quarter's report is final only after the announcement season ends;
# refresh while the quarter is younger than this many days.
_ANNOUNCING_WINDOW_DAYS = 120

# Quarter-end month-day keys, oldest first (P13 §5.3 report-period
# arithmetic for ASSET_GROWTH / EARNINGS_ACCEL).
_QUARTER_ENDS = ("0331", "0630", "0930", "1231")


def _shift_report_date(report_date: str, quarters: int) -> str:
    """The quarter-end date ``quarters`` report periods before
    ``report_date`` (exact period arithmetic: a missing report in between
    stays missing instead of silently misaligning the pair)."""

    ordinal = int(report_date[:4]) * 4 + _QUARTER_ENDS.index(report_date[4:])
    ordinal -= quarters
    year, index = divmod(ordinal, 4)
    return f"{year:04d}{_QUARTER_ENDS[index]}"


def _quarter_ends(start_year: int, end_inclusive: str) -> list[str]:
    """Quarter-end dates from ``start_year`` Q1 through ``end_inclusive``."""

    today = int(end_inclusive)
    quarters: list[str] = []
    year = start_year
    while True:
        batch = [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]
        if int(batch[0]) > today:
            break
        quarters.extend(q for q in batch if int(q) <= today)
        year += 1
        if year > 2100:  # pragma: no cover - safety bound.
            break
    return quarters


def _fundamental_cache_path(config, name: str) -> Path:
    return config.parquet_dir / "fundamental" / f"{name}.parquet"


class FundamentalCacheMiss(Exception):
    """A fundamental cache file exists but is unreadable (IP-05).

    Typed expected-miss marker: callers refetch instead of treating the
    corruption as an empty fundamentals history."""


def _read_cached(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        # IP-05: an unreadable cache is a typed expected-miss, never a
        # silent empty frame indistinguishable from "no data".
        raise FundamentalCacheMiss(f"{path}: {exc}") from exc


def _write_cached(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    shutil.move(str(tmp), str(path))


def _cached_or_fetch(
    config,
    name: str,
    fetcher: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    path = _fundamental_cache_path(config, name)
    stale = True
    if path.exists():
        try:
            age = datetime.now().timestamp() - path.stat().st_mtime
            stale = age > _STALE_AFTER_DAYS * 86400
        except OSError:
            stale = True
    if stale:
        df = fetcher()
        if not df.empty:
            _write_cached(path, df)
        return df
    try:
        return _read_cached(path)
    except FundamentalCacheMiss as exc:
        # Typed expected-miss (IP-05): a corrupt cache drives the same
        # refetch path as staleness instead of permanently masquerading
        # as an empty fundamentals history.
        logger.warning(f"fundamental cache unreadable, refetching: {exc}")
        df = fetcher()
        if not df.empty:
            _write_cached(path, df)
        return df


def _single_periods(rows: pd.DataFrame, col: str) -> list[tuple[str, float]]:
    """Convert cumulative YTD values to single-period values.

    ``rows`` must be sorted by ``report_date``; within one fiscal year each
    report contributes ``cum - previous cum`` (the first report of the year
    is already a single period).  Non-finite rows are skipped, which makes
    the window check in :func:`_ttm` reject any series with a missing
    quarter instead of silently misaligning it.
    """

    singles: list[tuple[str, float]] = []
    prev_year: str | None = None
    prev_cum: float | None = None
    for row in rows.itertuples():
        value = getattr(row, col)
        if not np.isfinite(value):
            continue
        year = row.report_date[:4]
        if year == prev_year and prev_cum is not None:
            single = value - prev_cum
        else:
            single = value
        singles.append((row.report_date, float(single)))
        prev_year, prev_cum = year, value
    return singles


def _ttm(
    rows: pd.DataFrame,
    col: str,
) -> tuple[list[str], list[str], list[float]]:
    """Trailing-four-quarter sums with report and announce dates.

    Returns ``(report_dates, announce_dates, values)``; a value is NaN
    unless the four trailing reports are complete and cover exactly three
    consecutive quarters (a missing quarter must yield neutral, not a
    shifted window).  Returning the report dates lets callers align
    several TTM series on the same reports.
    """

    singles = _single_periods(rows, col)
    reports: list[str] = []
    announces: list[str] = []
    values: list[float] = []
    for i, (report_date, single) in enumerate(singles):
        announce = rows.loc[rows["report_date"] == report_date, "announce_date"].iloc[0]
        reports.append(report_date)
        announces.append(announce)
        if i < 3:
            values.append(np.nan)
            continue
        window = singles[i - 3 : i + 1]
        span = (
            datetime.strptime(window[-1][0], "%Y%m%d")
            - datetime.strptime(window[0][0], "%Y%m%d")
        ).days
        if not 265 <= span <= 285:
            values.append(np.nan)
        else:
            values.append(float(sum(w[1] for w in window)))
    return reports, announces, values


def _ffill_from_announcements(
    dates: list[str],
    announces: list[str],
    values: list[float],
) -> np.ndarray:
    """Effective value per date: the latest announcement at or before it.

    Pairs with a missing/invalid announce date or a non-finite value are
    dropped (they can never become visible).
    """

    out = np.full(len(dates), np.nan)
    pairs = [
        (a, v)
        for a, v in zip(announces, values)
        if a is not None
        and str(a).isdigit()
        and v is not None
        and np.isfinite(v)
    ]
    if not pairs:
        return out
    pairs.sort(key=lambda pair: int(pair[0]))
    announce_ints = np.asarray([int(a) for a, _ in pairs])
    value_arr = np.asarray([v for _, v in pairs], dtype=float)
    date_ints = np.asarray([int(d) for d in dates])
    idx = np.searchsorted(announce_ints, date_ints, side="right") - 1
    valid = idx >= 0
    out[valid] = value_arr[idx[valid]]
    return out


def _fetch_fundamental_rows(
    db: AshareDB,
    config,
    ts_codes: list[str],
) -> pd.DataFrame:
    if not ts_codes:
        return pd.DataFrame()
    quoted = sql_quoted_list(ts_codes)
    return db.query(
        f"""
        SELECT ts_code, report_date, announce_date, dividend_announce,
               eps_cum, bvps, roe, roa, gross_margin, net_margin,
               revenue_cum, profit_cum, revenue_yoy, profit_yoy,
               debt_ratio, dividend_yield,
               net_operate_cash_flow, total_assets
        FROM {config.fundamentals_table}
        WHERE ts_code IN ({quoted})
        """
    )


def build_pit_frames(
    config,
    ts_codes: list[str],
    dates: list[str],
    close: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build announce-date-aligned ``[stock x date]`` frames for the PIT
    fundamental features.

    ``close`` is the ``[stock x date]`` close matrix used to convert
    per-share/level fields into price ratios (PE/PB/PS).  Any failure
    (e.g. the table does not exist yet) degrades to an empty mapping so
    the loader keeps working on a bare daily-bar database.
    """

    frames = {
        name: pd.DataFrame(np.nan, index=ts_codes, columns=dates)
        for name in FUNDAMENTAL_PIT_NAMES
    }
    try:
        with AshareDB(config.duckdb_path, read_only=True) as db:
            rows = _fetch_fundamental_rows(db, config, ts_codes)
    except Exception as exc:  # noqa: BLE001 - table may not exist yet.
        logger.warning(f"Fundamental PIT frames unavailable: {exc}")
        return frames
    if rows.empty:
        return frames

    # Group once: the per-stock loop below would otherwise re-scan the
    # whole table for every stock.
    grouped = {
        str(key): sub for key, sub in rows.groupby("ts_code", sort=False)
    }
    row_positions = {code: i for i, code in enumerate(ts_codes)}
    for code in ts_codes:
        sub = grouped.get(code)
        if sub is None or sub.empty:
            continue
        stock_row = row_positions[code]
        announces = sub["announce_date"].astype(str).tolist()
        div_announces = sub["dividend_announce"].astype(str).tolist()

        # Level fields: fill forward from their own disclosure date
        # (dividends use the exact ex-dividend date, everything else the
        # disclosure-season date).
        for name, column in _LEVEL_COLUMNS.items():
            values = pd.to_numeric(sub[column], errors="coerce").tolist()
            dates_source = div_announces if name == "DIVIDEND_YIELD" else announces
            frames[name].iloc[stock_row] = _ffill_from_announcements(
                dates, dates_source, values
            )

        # TTM conversions and price ratios.
        sub_sorted = sub.sort_values("report_date")
        _, eps_ann, eps_ttm = _ttm(sub_sorted, "eps_cum")
        rev_reports, rev_ann, rev_ttm = _ttm(sub_sorted, "revenue_cum")
        prof_reports, _, prof_ttm = _ttm(sub_sorted, "profit_cum")

        close_row = close.loc[code].to_numpy(dtype=float) if code in close.index else None
        if close_row is None:
            continue

        eps_series = _ffill_from_announcements(dates, eps_ann, eps_ttm)
        with np.errstate(all="ignore"):
            frames["PE_TTM"].iloc[stock_row] = np.where(
                eps_series > 0, close_row / eps_series, np.nan
            )

        bvps_series = _ffill_from_announcements(
            dates, announces, pd.to_numeric(sub["bvps"], errors="coerce").tolist()
        )
        with np.errstate(all="ignore"):
            frames["PB"].iloc[stock_row] = np.where(
                bvps_series > 0, close_row / bvps_series, np.nan
            )

        # Profit margin on the trailing four quarters, joined on the report
        # date so a missing quarter in either column cannot misalign the two
        # TTM series.
        revenue_ttm = dict(zip(rev_reports, rev_ttm))
        profit_ttm_by_report = dict(zip(prof_reports, prof_ttm))
        announce_by_report = dict(zip(rev_reports, rev_ann))
        margin_ann, margin_vals = [], []
        for report, rev in revenue_ttm.items():
            prof = profit_ttm_by_report.get(report)
            if np.isfinite(rev) and np.isfinite(prof) and rev > 0 and prof > 0:
                margin_ann.append(announce_by_report[report])
                margin_vals.append(prof / rev)
        margin_series = _ffill_from_announcements(dates, margin_ann, margin_vals)
        pe_row = frames["PE_TTM"].iloc[stock_row].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            frames["PS_TTM"].iloc[stock_row] = np.where(
                margin_series > 0, pe_row * margin_series, np.nan
            )

        # --- P13 family ⑤ (docs/p13_fundamental_fields_contract.md §5.3) ---
        # Cash-flow quality, accruals, asset growth and earnings
        # acceleration.  Every TTM conversion goes through _ttm and every
        # visibility through _ffill_from_announcements (the single PIT
        # path); ratios join strictly on the report period (never
        # misaligned), and a ratio is visible from the announce date of its
        # latest contributing report.  Missing/incomplete inputs stay NaN
        # (AGENTS §5.3) -- never a fabricated neutral value.
        prof_ttm_by_report = dict(zip(prof_reports, prof_ttm))
        cfo_reports, _, cfo_ttm = _ttm(sub_sorted, "net_operate_cash_flow")
        cfo_ttm_by_report = dict(zip(cfo_reports, cfo_ttm))
        ta_by_report = {
            str(rep): ta
            for rep, ta in zip(
                sub_sorted["report_date"],
                pd.to_numeric(sub_sorted["total_assets"], errors="coerce"),
            )
        }
        announce_by_report = {
            str(rep): (str(ann) if pd.notna(ann) else None)
            for rep, ann in zip(sub["report_date"], sub["announce_date"])
        }

        def _g_of(report: str) -> float | None:
            """TTM profit growth vs the report four periods earlier."""
            profit = prof_ttm_by_report.get(report)
            profit_prev = prof_ttm_by_report.get(_shift_report_date(report, 4))
            if (
                profit is None
                or not np.isfinite(profit)
                or profit_prev is None
                or not np.isfinite(profit_prev)
                or profit_prev <= 0
            ):
                return None
            return profit / profit_prev - 1

        quality_ann, quality_vals = [], []
        accrual_ann, accrual_vals = [], []
        growth_ann, growth_vals = [], []
        accel_ann, accel_vals = [], []
        for report in prof_reports:
            announce = announce_by_report.get(report)
            profit = prof_ttm_by_report.get(report)
            cfo = cfo_ttm_by_report.get(report)
            assets = ta_by_report.get(report)
            if announce is None:
                continue  # without its master date a row is never visible
            if (
                profit is not None
                and np.isfinite(profit)
                and profit > 0
                and cfo is not None
                and np.isfinite(cfo)
            ):
                quality_ann.append(announce)
                quality_vals.append(cfo / profit)
            if (
                profit is not None
                and np.isfinite(profit)
                and cfo is not None
                and np.isfinite(cfo)
                and assets is not None
                and np.isfinite(assets)
                and assets > 0
            ):
                accrual_ann.append(announce)
                accrual_vals.append((profit - cfo) / assets)
            assets_prev = ta_by_report.get(_shift_report_date(report, 4))
            announce_prev = announce_by_report.get(_shift_report_date(report, 4))
            if (
                announce_prev is not None
                and assets is not None
                and np.isfinite(assets)
                and assets > 0
                and assets_prev is not None
                and np.isfinite(assets_prev)
                and assets_prev > 0
            ):
                growth_ann.append(max(announce, announce_prev))
                growth_vals.append(assets / assets_prev - 1)
            growth_t = _g_of(report)
            growth_prev = _g_of(_shift_report_date(report, 1))
            if (
                growth_t is not None
                and growth_prev is not None
                and np.isfinite(growth_t)
                and np.isfinite(growth_prev)
            ):
                accel_ann.append(announce)
                accel_vals.append(growth_t - growth_prev)

        frames["CASHFLOW_QUALITY"].iloc[stock_row] = _ffill_from_announcements(
            dates, quality_ann, quality_vals
        )
        frames["ACCRUALS"].iloc[stock_row] = _ffill_from_announcements(
            dates, accrual_ann, accrual_vals
        )
        frames["ASSET_GROWTH"].iloc[stock_row] = _ffill_from_announcements(
            dates, growth_ann, growth_vals
        )
        frames["EARNINGS_ACCEL"].iloc[stock_row] = _ffill_from_announcements(
            dates, accel_ann, accel_vals
        )

    return frames


# --- P2-01: fundamentals-table scope ------------------------------------------
#
# The ``fundamental_pit`` table only ever holds rows whose ts_code is a
# valid A-share code AND belongs to the persisted stock scope: the
# ``stocks`` table (current snapshot + PIT backfills), the PIT
# ``constituents`` members and the cached daily-bar codes.  Codes outside
# the scope (BJ/NEEQ instruments without bars, B-shares, index symbols)
# can never be consumed by the loader, and their presence would mislabel
# the table as broader than the pipeline's actual data reach.


def persisted_scope_codes(db: AshareDB, config) -> set[str]:
    """The persisted stock scope: stocks ∪ PIT constituents ∪ cached bars.

    Only valid A-share codes count; a missing table (fresh database) is
    treated as an empty member, never an error.
    """

    codes: set[str] = set()
    for table in (config.stocks_table, config.constituents_table):
        try:
            df = db.query(f"SELECT DISTINCT ts_code FROM {table}")
            codes |= {str(c) for c in df["ts_code"]}
        except Exception:  # noqa: BLE001 - table may not exist yet.
            continue
    daily_dir = config.parquet_dir / "daily"
    if daily_dir.exists():
        codes |= {path.stem for path in daily_dir.glob("*.parquet")}
    return {code for code in codes if is_valid_a_share_code(code)}


def fundamental_scope_stats(db: AshareDB, config) -> dict[str, object]:
    """Scope audit of the fundamentals table: how many rows/codes lie
    inside and outside the persisted stock scope (P2-01)."""

    scope = persisted_scope_codes(db, config)
    try:
        rows = db.query(
            f"SELECT ts_code, count(*) AS n FROM {config.fundamentals_table} "
            "GROUP BY ts_code"
        )
    except Exception:  # noqa: BLE001 - table may not exist yet.
        return {
            "total_rows": 0,
            "in_scope_rows": 0,
            "out_of_scope_rows": 0,
            "out_of_scope_codes": [],
            "out_of_scope_codes_count": 0,
            "scope_codes_count": len(scope),
        }
    total = 0
    in_scope = 0
    out_by_code: dict[str, int] = {}
    for row in rows.itertuples():
        code = str(row.ts_code)
        count = int(row.n)
        total += count
        if code in scope:
            in_scope += count
        else:
            out_by_code[code] = out_by_code.get(code, 0) + count
    return {
        "total_rows": total,
        "in_scope_rows": in_scope,
        "out_of_scope_rows": total - in_scope,
        "out_of_scope_codes": sorted(out_by_code),
        "out_of_scope_codes_count": len(out_by_code),
        "scope_codes_count": len(scope),
    }


def purge_out_of_scope_fundamentals(db: AshareDB, config) -> dict[str, int]:
    """Delete every fundamentals row outside the persisted stock scope.

    Migration for pre-P2 tables (P2-01); idempotent — a second purge
    removes nothing.  Returns measured before/after counts.
    """

    scope = persisted_scope_codes(db, config)
    before = int(
        db.query(f"SELECT count(*) AS n FROM {config.fundamentals_table}")["n"].iloc[0]
    )
    if scope:
        db.execute(
            f"DELETE FROM {config.fundamentals_table} "
            f"WHERE ts_code NOT IN ({sql_quoted_list(sorted(scope))})"
        )
    else:
        db.execute(f"DELETE FROM {config.fundamentals_table}")
    after = int(
        db.query(f"SELECT count(*) AS n FROM {config.fundamentals_table}")["n"].iloc[0]
    )
    return {"removed_rows": before - after, "rows_before": before, "rows_after": after}


def sync_fundamentals(
    client,
    db: AshareDB,
    config,
    universe: list[str],
) -> dict[str, Any]:
    """Sync quarterly fundamentals for the given universe.

    Bulk earnings rows are restricted to ``universe`` before they are
    cached or stored (P2-01 scope contract: the table only holds rows for
    codes the pipeline can consume).  Degrades per stock/quarter like the
    daily-bar sync: failures are logged and skipped, never fatal.  Offline
    mode skips entirely.
    """

    if client.offline:
        return {
            "fundamental_quarters": 0,
            "fundamental_rows": 0,
            "fundamental_supplements": 0,
            "fundamental_failures": 0,
        }

    today = datetime.now().strftime("%Y%m%d")
    quarters = _quarter_ends(int(config.start_date[:4]), today)
    total_rows = 0
    failures = 0
    # P2-01 scope: the bulk endpoint covers the whole market incl. BJ/NEEQ
    # instruments the pipeline can never consume; only the synced universe
    # may be stored (cache included).
    universe_set = set(universe)
    for quarter in quarters:
        quarter_date = datetime.strptime(quarter, "%Y%m%d").date()
        announcing = (datetime.now().date() - quarter_date).days < _ANNOUNCING_WINDOW_DAYS
        path = _fundamental_cache_path(config, f"earnings_{quarter}")
        df = pd.DataFrame() if announcing else _read_cached(path)
        if df.empty:
            try:
                df = client.get_earnings_report(quarter)
                if not df.empty:
                    df = df[df["ts_code"].isin(universe_set)]
                    _write_cached(path, df)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Earnings report failed for {quarter}: {exc}")
                failures += 1
                continue
        if df.empty:
            continue
        # The universe filter applies to the cache path too: pre-P2-01
        # caches hold whole-market rows, and reading them unfiltered would
        # resurrect purged out-of-scope codes (t20 window-② incident).
        df = df[df["ts_code"].isin(universe_set)]
        rows = [{c: row[c] for c in _EARNINGS_COLUMNS} for row in df.to_dict("records")]
        db.upsert_fundamentals(rows, config)
        total_rows += len(rows)
    logger.info(f"Earnings reports synced: {len(quarters)} quarters, {total_rows} rows")

    # Announcement-date master for the Sina supplement: only rows whose
    # report period is already in the earnings table get a PIT date.
    announce_map: dict[tuple[str, str], str] = {}
    try:
        meta = db.query(
            f"SELECT ts_code, report_date, announce_date FROM {config.fundamentals_table}"
        )
        announce_map = {
            (str(r.ts_code), str(r.report_date)): str(r.announce_date)
            for r in meta.itertuples()
        }
    except Exception:  # noqa: BLE001
        announce_map = {}

    # P13 bulk supplements (docs/p13_fundamental_fields_contract.md §5.2):
    # the cash-flow statement and the balance sheet mirror the earnings
    # path -- whole-market per report period, universe-filtered before
    # caching/storage, and joined to the earnings announce master.  Rows
    # without a master match are dropped: a disclosure date is never
    # guessed (the Sina-supplement discipline, generalized).
    for section, fetcher in (
        ("cash_flow", client.get_cash_flow_report),
        ("balance_sheet", client.get_balance_sheet),
    ):
        written = 0
        for quarter in quarters:
            quarter_date = datetime.strptime(quarter, "%Y%m%d").date()
            announcing = (
                datetime.now().date() - quarter_date
            ).days < _ANNOUNCING_WINDOW_DAYS
            path = _fundamental_cache_path(config, f"{section}_{quarter}")
            df = pd.DataFrame() if announcing else _read_cached(path)
            if df.empty:
                try:
                    df = fetcher(quarter)
                    if not df.empty:
                        df = df[df["ts_code"].isin(universe_set)]
                        _write_cached(path, df)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"{section} report failed for {quarter}: {exc}")
                    failures += 1
                    continue
            if df.empty:
                continue
            # Same cache-path discipline as the earnings loop (t20
            # window-② incident): filter before the master join.
            df = df[df["ts_code"].isin(universe_set)]
            rows: list[dict[str, Any]] = []
            for row in df.to_dict("records"):
                announce = announce_map.get(
                    (str(row["ts_code"]), str(row["report_date"]))
                )
                if announce is None:
                    continue  # no master match -> dropped, never guessed
                payload: dict[str, Any] = {
                    "ts_code": str(row["ts_code"]),
                    "report_date": str(row["report_date"]),
                    "announce_date": announce,
                }
                if "net_operate_cash_flow" in row:
                    payload["net_operate_cash_flow"] = row["net_operate_cash_flow"]
                if "total_assets" in row:
                    payload["total_assets"] = row["total_assets"]
                rows.append(payload)
            if rows:
                db.upsert_fundamentals(rows, config)
                written += len(rows)
        logger.info(
            f"{section} reports synced: {len(quarters)} quarters, {written} rows"
        )

    supplements = 0
    for code in universe:
        try:
            indicator = _cached_or_fetch(
                config,
                f"{code}_indicator",
                lambda: client.get_financial_indicator(
                    code, int(config.start_date[:4])
                ),
            )
            dividend = _cached_or_fetch(
                config, f"{code}_dividend", lambda: client.get_dividend_detail(code)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Fundamental supplement failed for {code}: {exc}")
            failures += 1
            continue
        rows: list[dict[str, Any]] = []
        if not indicator.empty:
            for row in indicator.to_dict("records"):
                announce = announce_map.get((str(row["ts_code"]), str(row["report_date"])))
                if announce is None:
                    continue
                rows.append(
                    {
                        "ts_code": str(row["ts_code"]),
                        "report_date": str(row["report_date"]),
                        "announce_date": announce,
                        "roa": row.get("roa"),
                        "debt_ratio": row.get("debt_ratio"),
                    }
                )
        if not dividend.empty:
            for row in dividend.to_dict("records"):
                rows.append(
                    {
                        "ts_code": str(row["ts_code"]),
                        "report_date": str(row["report_date"]),
                        "dividend_announce": str(row["announce_date"]),
                        "dividend_yield": row.get("dividend_yield"),
                    }
                )
        if rows:
            db.upsert_fundamentals(rows, config)
            supplements += 1

    logger.info(f"Fundamental supplements synced for {supplements} stocks")
    return {
        "fundamental_quarters": len(quarters),
        "fundamental_rows": total_rows,
        "fundamental_supplements": supplements,
        "fundamental_failures": failures,
    }
