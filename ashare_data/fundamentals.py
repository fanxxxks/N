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
from .pit_import import asof_winner_intervals
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
    "NET_MARGIN_TREND_4Q",
    "LEVERAGE_TREND_4Q",
    "CAPEX_INTENSITY",
    "CFO_TO_DEBT",
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


_FAMILY5_NAMES = ("CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL")

# F1 batch (p23 v5, the user-approved set confirmed by the captain's
# correction): the four slow-fundamental quality factors computed from the
# same winner-resolved as-of timelines (the net-margin trend, the leverage
# trend, capex intensity, CFO/debt coverage).
_F1_NAMES = (
    "NET_MARGIN_TREND_4Q",
    "LEVERAGE_TREND_4Q",
    "CAPEX_INTENSITY",
    "CFO_TO_DEBT",
)


def _piecewise_lookup(
    timeline: list[tuple[str, str, float]], session: str
) -> float | None:
    """The value whose ``[from, to)`` segment contains ``session``."""

    for start, end, value in timeline:
        if start <= session < end:
            return value
    return None


def _merge_timelines(
    timelines: list[list[tuple[str, str, float]]],
    combine: Callable[[list[float | None]], float | None],
) -> list[tuple[str, str, float]]:
    """Merge piecewise timelines into one, evaluating ``combine`` per
    segment.  A segment boundary from any input splits the output; the
    combine callback receives every input's value at the segment start
    (``None`` where that input has no covering version) and returns the
    merged value or ``None`` (invisible — missing input stays NaN)."""

    boundaries: set[str] = set()
    for timeline in timelines:
        for start, _end, _value in timeline:
            boundaries.add(start)
    ordered = sorted(boundaries)
    out: list[tuple[str, str, float]] = []
    for position, start in enumerate(ordered):
        end = (
            ordered[position + 1]
            if position + 1 < len(ordered)
            else _ASOF_OPEN_END_FUNDAMENTALS
        )
        values = [_piecewise_lookup(timeline, start) for timeline in timelines]
        value = combine(values)
        if value is None or not np.isfinite(value):
            continue
        if out and out[-1][1] == start and out[-1][2] == value:
            out[-1] = (out[-1][0], end, value)  # merge equal neighbours
            continue
        out.append((start, end, value))
    return out


_ASOF_OPEN_END_FUNDAMENTALS = "99991231"


def _family5_asof_frames(
    config,
    ts_codes: list[str],
    dates: list[str],
) -> dict[str, pd.DataFrame] | None:
    """The four family-⑤ features from the v2 financial version tables
    (p17 §6.2 as-of; D6 原位切换).  Returns ``None`` when the v2 statements
    are unavailable (the caller degrades those four to NaN — the legacy
    ``fundamental_pit`` table is never a fallback)."""

    try:
        income = asof_winner_intervals(config, "income", ts_codes)
        balance = asof_winner_intervals(config, "balance", ts_codes)
        cashflow = asof_winner_intervals(config, "cashflow", ts_codes)
    except Exception as exc:  # noqa: BLE001 - missing v2 source: expected in
        # tests / pre-D6 databases; the four features degrade to NaN (Tier B
        # auxiliary, documented behaviour — never a legacy fallback).
        logger.debug(f"family-⑤ as-of frames unavailable: {exc}")
        return None

    frames = {
        name: pd.DataFrame(np.nan, index=ts_codes, columns=dates)
        for name in _FAMILY5_NAMES + _F1_NAMES
    }
    date_index = {date: position for position, date in enumerate(dates)}

    def _per_stock_intervals(rows: pd.DataFrame, value_col: str):
        intervals: dict[str, dict[str, list[tuple[str, str, float]]]] = {}
        for row in rows.itertuples(index=False):
            value = getattr(row, value_col)
            timeline = intervals.setdefault(str(row.ts_code), {}).setdefault(
                str(row.end_date), []
            )
            timeline.append(
                (
                    str(row.winner_from),
                    str(row.winner_to),
                    float(value) if pd.notna(value) else np.nan,
                )
            )
        for stock in intervals.values():
            for timeline in stock.values():
                timeline.sort(key=lambda item: item[0])
        return intervals

    # Pre-group by ts_code once: O(N) total instead of O(N×M) per-code
    # boolean mask scans (t74 perf: 24.7s → 0.56s on 415K×5542 synthetic;
    # py-spy double-confirmed take/reindex_indexer hot path).
    income_grouped = (
        dict(tuple(income.groupby("ts_code", sort=False)))
        if not income.empty else {}
    )
    balance_grouped = (
        dict(tuple(balance.groupby("ts_code", sort=False)))
        if not balance.empty else {}
    )
    cashflow_grouped = (
        dict(tuple(cashflow.groupby("ts_code", sort=False)))
        if not cashflow.empty else {}
    )
    code_index = {code: i for i, code in enumerate(ts_codes)}

    _empty = pd.DataFrame()
    for code in ts_codes:
        income_stock = income_grouped.get(code, _empty)
        balance_stock = balance_grouped.get(code, _empty)
        cashflow_stock = cashflow_grouped.get(code, _empty)
        if income_stock.empty and balance_stock.empty and cashflow_stock.empty:
            # Nothing to compose for any feature (the F1 leverage change
            # composes from balance alone — a stock with balance rows but
            # no income rows still gets it; its income-TTM-derived frames
            # degrade to NaN inside the loop).
            continue
        stock_row = code_index.get(code)
        if stock_row is None:
            continue

        income_cum = _per_stock_intervals(income_stock, "n_income_attr_p").get(code, {})
        revenue_cum = _per_stock_intervals(income_stock, "revenue").get(code, {})
        inv_cum = _per_stock_intervals(cashflow_stock, "n_cashflow_inv_act").get(code, {})
        cashflow_cum = _per_stock_intervals(cashflow_stock, "n_cashflow_act").get(code, {})
        assets_level = _per_stock_intervals(balance_stock, "total_assets").get(code, {})
        liab_level = _per_stock_intervals(balance_stock, "total_liab").get(code, {})

        def _cum_maps(end_date: str, cum: dict[str, list[tuple[str, str, float]]]):
            return [
                cum.get(_shift_report_date(end_date, back), [])
                for back in range(4, -1, -1)
            ]

        def _profit_ttm(end_date: str) -> list[tuple[str, str, float]]:
            maps = _cum_maps(end_date, income_cum)

            def combine(values):
                return _ttm_at(maps, end_date, values and values[-1] and _segment_session(values))

            return _merge_timelines(maps, combine)

        def _segment_session(values):
            return None

        del _segment_session
        profit_ttm_cache: dict[str, list[tuple[str, str, float]]] = {}
        _cashflow_ttm_cache: dict[str, list[tuple[str, str, float]]] = {}

        def _profit_ttm_timeline(end_date: str) -> list[tuple[str, str, float]]:
            if end_date in profit_ttm_cache:
                return profit_ttm_cache[end_date]
            maps = _cum_maps(end_date, income_cum)

            def combine(values):
                # values: [cum(E-4), cum(E-3), cum(E-2), cum(E-1), cum(E)]
                cums = list(values)
                singles: list[float] = []
                for position in range(1, 5):
                    current = cums[position]
                    previous = cums[position - 1]
                    quarter_end = _shift_report_date(end_date, 4 - position)
                    if current is None or not np.isfinite(current):
                        return None
                    if quarter_end[4:] == "0331" or previous is None or not np.isfinite(
                        previous
                    ):
                        if quarter_end[4:] != "0331":
                            return None
                        singles.append(float(current))
                    else:
                        singles.append(float(current) - float(previous))
                if len(singles) != 4:
                    return None
                # The quarter identity comes from exact period arithmetic
                # (_shift_report_date), so a missing quarter yields None
                # above and the 265-285-day span check of the legacy path
                # is satisfied by construction.
                return float(sum(singles))

            timeline = _merge_timelines(maps, combine)
            profit_ttm_cache[end_date] = timeline
            return timeline

        def _cashflow_ttm_timeline(end_date: str) -> list[tuple[str, str, float]]:
            if end_date in _cashflow_ttm_cache:
                return _cashflow_ttm_cache[end_date]
            maps = _cum_maps(end_date, cashflow_cum)

            def combine(values):
                cums = list(values)
                singles: list[float] = []
                for position in range(1, 5):
                    current = cums[position]
                    previous = cums[position - 1]
                    quarter_end = _shift_report_date(end_date, 4 - position)
                    if current is None or not np.isfinite(current):
                        return None
                    if quarter_end[4:] == "0331" or previous is None or not np.isfinite(
                        previous
                    ):
                        if quarter_end[4:] != "0331":
                            return None
                        singles.append(float(current))
                    else:
                        singles.append(float(current) - float(previous))
                if len(singles) != 4:
                    return None
                return float(sum(singles))

            timeline = _merge_timelines(maps, combine)
            _cashflow_ttm_cache[end_date] = timeline
            return timeline

        # F1 batch (p23 v4, user-approved via the captain's delegated
        # ruling): the flow-TTM machinery is shared across every
        # cumulative-flow column (§9.1 dedup — one combine, four users).
        revenue_ttm_cache: dict[str, list[tuple[str, str, float]]] = {}
        inv_ttm_cache: dict[str, list[tuple[str, str, float]]] = {}

        def _flow_ttm_timeline_for(cum_map, cache):
            def timeline_for(end_date: str) -> list[tuple[str, str, float]]:
                if end_date in cache:
                    return cache[end_date]
                maps = _cum_maps(end_date, cum_map)

                def combine(values):
                    cums = list(values)
                    singles: list[float] = []
                    for position in range(1, 5):
                        current = cums[position]
                        previous = cums[position - 1]
                        quarter_end = _shift_report_date(end_date, 4 - position)
                        if current is None or not np.isfinite(current):
                            return None
                        if quarter_end[4:] == "0331" or previous is None or not np.isfinite(
                            previous
                        ):
                            if quarter_end[4:] != "0331":
                                return None
                            singles.append(float(current))
                        else:
                            singles.append(float(current) - float(previous))
                    if len(singles) != 4:
                        return None
                    return float(sum(singles))

                timeline = _merge_timelines(maps, combine)
                cache[end_date] = timeline
                return timeline

            return timeline_for

        _revenue_ttm_timeline = _flow_ttm_timeline_for(revenue_cum, revenue_ttm_cache)
        _inv_ttm_timeline = _flow_ttm_timeline_for(inv_cum, inv_ttm_cache)

        def _piecewise_fill(frame: pd.DataFrame, timeline: list[tuple[str, str, float]]):
            date_ints = np.asarray([int(d) for d in dates])
            for start, end, value in timeline:
                if not np.isfinite(value):
                    continue
                start_idx = int(np.searchsorted(date_ints, int(start), side="left"))
                end_idx = int(np.searchsorted(date_ints, int(end), side="left"))
                if end_idx > start_idx:
                    frame.iloc[stock_row, start_idx:end_idx] = value

        for end_date in sorted(
            set(income_cum) | set(cashflow_cum) | set(assets_level)
        ):
            # Erratum #4: non-quarter-end end_dates (real irregular
            # reporting periods like 20170131) do not participate in the
            # quarterly cumulative TTM period arithmetic — they cannot be
            # interpreted as quarter-end cumulative reports.  Skip them;
            # the related features stay NaN for those periods.
            if end_date[4:] not in _QUARTER_ENDS:
                continue
            profit_tl = _profit_ttm_timeline(end_date)
            cashflow_tl = _cashflow_ttm_timeline(end_date)
            assets_tl = assets_level.get(end_date, [])

            quality = _merge_timelines(
                [profit_tl, cashflow_tl],
                combine=lambda values: (
                    values[1] / values[0]
                    if values[0] is not None
                    and np.isfinite(values[0])
                    and values[0] > 0
                    and values[1] is not None
                    and np.isfinite(values[1])
                    else None
                ),
            )
            _piecewise_fill(frames["CASHFLOW_QUALITY"], quality)

            accrual = _merge_timelines(
                [profit_tl, cashflow_tl, assets_tl],
                combine=lambda values: (
                    (values[0] - values[1]) / values[2]
                    if values[0] is not None
                    and np.isfinite(values[0])
                    and values[1] is not None
                    and np.isfinite(values[1])
                    and values[2] is not None
                    and np.isfinite(values[2])
                    and values[2] > 0
                    else None
                ),
            )
            _piecewise_fill(frames["ACCRUALS"], accrual)

            prev_end = _shift_report_date(end_date, 4)
            assets_prev_tl = assets_level.get(prev_end, [])
            growth = _merge_timelines(
                [assets_tl, assets_prev_tl],
                combine=lambda values: (
                    values[0] / values[1] - 1
                    if values[0] is not None
                    and np.isfinite(values[0])
                    and values[0] > 0
                    and values[1] is not None
                    and np.isfinite(values[1])
                    and values[1] > 0
                    else None
                ),
            )
            _piecewise_fill(frames["ASSET_GROWTH"], growth)

            def _profit_growth_ratio(later: str, earlier: str) -> list[tuple[str, str, float]]:
                return _merge_timelines(
                    [_profit_ttm_timeline(later), _profit_ttm_timeline(earlier)],
                    combine=lambda values: (
                        values[0] / values[1] - 1
                        if values[0] is not None
                        and np.isfinite(values[0])
                        and values[1] is not None
                        and np.isfinite(values[1])
                        and values[1] > 0
                        else None
                    ),
                )

            accel = _merge_timelines(
                [
                    _profit_growth_ratio(end_date, _shift_report_date(end_date, 4)),
                    _profit_growth_ratio(
                        _shift_report_date(end_date, 1),
                        _shift_report_date(end_date, 5),
                    ),
                ],
                combine=lambda values: (
                    values[0] - values[1]
                    if values[0] is not None
                    and np.isfinite(values[0])
                    and values[1] is not None
                    and np.isfinite(values[1])
                    else None
                ),
            )
            _piecewise_fill(frames["EARNINGS_ACCEL"], accel)

            # F1 batch (p23 v5, the user-approved set confirmed by the
            # captain's correction): the four slow-fundamental quality
            # factors — the net-margin 4-report-period trend, the leverage
            # trend, capex intensity, and CFO/debt coverage — composed from
            # the same winner-resolved as-of timelines (p23 §2 formulas).
            def _div_guarded(a_tl, b_tl):
                return _merge_timelines(
                    [a_tl, b_tl],
                    combine=lambda values: (
                        values[0] / values[1]
                        if values[0] is not None
                        and np.isfinite(values[0])
                        and values[1] is not None
                        and np.isfinite(values[1])
                        and values[1] > 0
                        else None
                    ),
                )

            def _sub_guarded(a_tl, b_tl):
                return _merge_timelines(
                    [a_tl, b_tl],
                    combine=lambda values: (
                        values[0] - values[1]
                        if values[0] is not None
                        and np.isfinite(values[0])
                        and values[1] is not None
                        and np.isfinite(values[1])
                        else None
                    ),
                )

            prev_end_f1 = _shift_report_date(end_date, 4)
            assets_tl_f1 = assets_level.get(end_date, [])
            assets_prev_tl_f1 = assets_level.get(prev_end_f1, [])

            def _nm_ratio(e: str):
                return _div_guarded(
                    _profit_ttm_timeline(e), _revenue_ttm_timeline(e)
                )

            nm_now = _nm_ratio(end_date)
            nm_prev = _nm_ratio(prev_end_f1)
            _piecewise_fill(
                frames["NET_MARGIN_TREND_4Q"], _sub_guarded(nm_now, nm_prev)
            )

            liab_tl = liab_level.get(end_date, [])
            liab_prev_tl = liab_level.get(prev_end_f1, [])
            lev_now = _div_guarded(liab_tl, assets_tl_f1)
            lev_prev = _div_guarded(liab_prev_tl, assets_prev_tl_f1)
            _piecewise_fill(
                frames["LEVERAGE_TREND_4Q"], _sub_guarded(lev_now, lev_prev)
            )

            capex_tl = _merge_timelines(
                [_inv_ttm_timeline(end_date), assets_tl_f1],
                combine=lambda values: (
                    -values[0] / values[1]
                    if values[0] is not None
                    and np.isfinite(values[0])
                    and values[1] is not None
                    and np.isfinite(values[1])
                    and values[1] > 0
                    else None
                ),
            )
            _piecewise_fill(frames["CAPEX_INTENSITY"], capex_tl)

            cfo_debt_tl = _div_guarded(_cashflow_ttm_timeline(end_date), liab_tl)
            _piecewise_fill(frames["CFO_TO_DEBT"], cfo_debt_tl)

    return frames


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

    D6 (p17 §6.2): the four family-⑤ features (CASHFLOW_QUALITY /
    ACCRUALS / ASSET_GROWTH / EARNINGS_ACCEL) read the three v2 financial
    version tables through the single as-of implementation
    (:func:`ashare_data.pit_import.asof_winner_intervals`) — revision
    timing exact, missing values NaN, the legacy ``fundamental_pit``
    table never a fallback.  The remaining eleven features keep their
    legacy source until their v2 columns exist (documented boundary).
    """

    frames = {
        name: pd.DataFrame(np.nan, index=ts_codes, columns=dates)
        for name in FUNDAMENTAL_PIT_NAMES
    }
    # Family-⑤ from the v2 version tables (原位切换, p17 §6.2): missing
    # source leaves those four NaN — never a legacy fallback.
    family5 = _family5_asof_frames(config, ts_codes, dates)
    if family5 is not None:
        frames.update(family5)
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

        # Family-⑤ features come from the v2 as-of source (see above).

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
