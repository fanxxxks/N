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

# Feature names produced by the point-in-time pipeline (MARKET_CAP is a
# local daily-bar factor, not a PIT field).
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


def _read_cached(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read fundamental cache {path}: {exc}")
        return pd.DataFrame()


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
    return _read_cached(path)


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
               debt_ratio, dividend_yield
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

    return frames


def sync_fundamentals(
    client,
    db: AshareDB,
    config,
    universe: list[str],
) -> dict[str, Any]:
    """Sync quarterly fundamentals for the given universe.

    Degrades per stock/quarter like the daily-bar sync: failures are
    logged and skipped, never fatal.  Offline mode skips entirely.
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
    for quarter in quarters:
        quarter_date = datetime.strptime(quarter, "%Y%m%d").date()
        announcing = (datetime.now().date() - quarter_date).days < _ANNOUNCING_WINDOW_DAYS
        path = _fundamental_cache_path(config, f"earnings_{quarter}")
        df = pd.DataFrame() if announcing else _read_cached(path)
        if df.empty:
            try:
                df = client.get_earnings_report(quarter)
                if not df.empty:
                    _write_cached(path, df)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Earnings report failed for {quarter}: {exc}")
                failures += 1
                continue
        if df.empty:
            continue
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
