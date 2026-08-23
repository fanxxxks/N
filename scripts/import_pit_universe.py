"""Import point-in-time universe data into the local DuckDB.

Two independent repairs for the phase-9 production gates:

1. ``--constituents`` (default): historical index-membership intervals.
   BaoStock exposes per-date constituent queries for 沪深300
   (``query_hs300_stocks``) and 中证500 (``query_zz500_stocks``).  The
   script samples membership at every month-end open session from the
   first session to the last daily bar, then compresses each stock's
   membership timeline into half-open intervals ``[in_date, out_date)``
   (re-entries become multiple intervals; the last open interval ends at
   ``99991231``).  Interval boundaries therefore carry the provider's
   real membership information at month granularity.  Index codes without
   a provider mapping (e.g. 000852.SH / CSI1000, which has no free
   historical source) are rejected instead of being fabricated.

2. ``--list-dates`` (default): refresh ``stocks.list_date`` from the
   exchange batch metadata via the repo's own AkShare client, and fill
   any member stock missing from the table (e.g. delisted members) from
   the SSE/SZSE delisting tables.

Run both repairs, then verify with ``scripts/check_production_gates.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Scripts live one level below the package root: make `ashare_*` importable
# regardless of the caller's sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from ashare_data.akshare_client import AkShareClient, normalize_listing_date
from ashare_data.config import load_config, make_data_config
from ashare_data.db import AshareDB
from ashare_data.processor import is_valid_a_share_code
from ashare_data.universe import membership_interval_issues

# BaoStock per-date constituent queries per configured index code.
BAOSTOCK_QUERIES = {
    "000300.SH": "query_hs300_stocks",
    "000905.SH": "query_zz500_stocks",
}


def _root() -> Path:
    return _ROOT


def _load_config():
    raw = load_config(None, project_root=_root())
    return make_data_config(raw, _root())


def _sessions(config) -> list[str]:
    with AshareDB(config.duckdb_path, read_only=True) as db:
        calendar = db.query(
            f"SELECT trade_date FROM {config.calendar_table} "
            "WHERE is_open=true ORDER BY trade_date"
        )["trade_date"].astype(str).tolist()
        daily_max = db.query(
            f"SELECT MAX(trade_date) AS mx FROM {config.daily_table}"
        ).iloc[0]["mx"]
    if not calendar:
        raise SystemExit("trade_calendar has no open sessions")
    sessions = [s for s in calendar if s <= str(daily_max)]
    if not sessions:
        raise SystemExit("no calendar sessions overlap the daily bars")
    return sessions


def _month_end_samples(sessions: list[str]) -> list[str]:
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


def _bs_code_to_ts_code(code: str) -> str:
    market, symbol = code.split(".")
    suffix = "SH" if market == "sh" else "SZ" if market == "sz" else None
    if suffix is None:
        raise ValueError(f"unexpected baostock code format: {code!r}")
    return f"{symbol}.{suffix}"


def _snapshot_members(bs, query_name: str, date: str) -> tuple[str, set[str]]:
    """Constituents as of ``date`` plus the provider's snapshot update date.

    BaoStock answers with the most recent adjustment snapshot on or before
    ``date`` and reports its ``updateDate`` (the effective adjustment date),
    so interval boundaries can use the provider's exact dates instead of
    the sampling dates.
    """
    query_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    rs = getattr(bs, query_name)(query_date)
    if rs.error_code != "0":
        raise RuntimeError(
            f"baostock {query_name}({query_date}) failed: "
            f"{rs.error_code} {rs.error_msg}"
        )
    code_idx = rs.fields.index("code")
    update_idx = rs.fields.index("updateDate")
    members = set()
    update_dates: list[str] = []
    while rs.next():
        row = rs.get_row_data()
        update_dates.append(str(row[update_idx]).replace("-", ""))
        code = str(row[code_idx])
        try:
            ts_code = _bs_code_to_ts_code(code)
        except ValueError:
            continue
        if is_valid_a_share_code(ts_code):
            members.add(ts_code)
    if not members or not update_dates:
        raise RuntimeError(
            f"baostock {query_name}({query_date}) returned no members"
        )
    return max(update_dates), members


def _timeline_to_intervals(
    dates: list[str], present: list[bool]
) -> list[tuple[str, str]]:
    """Compress a sampled membership timeline into half-open intervals."""
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


def import_calendar(config) -> int:
    """Backfill the session calendar with the provider's full history.

    Listing age is measured in open sessions, so the calendar must contain
    sessions BEFORE the dataset window; without them every stock fails the
    configured 60-session age gate during the first ~60 dataset sessions.
    """

    client = AkShareClient(config)
    dates = client.get_trade_calendar()
    rows = [{"trade_date": d, "is_open": True} for d in dates]
    with AshareDB(config.duckdb_path) as db:
        db.upsert_calendar(rows, config)
    print(f"calendar sessions: {len(dates)} ({dates[0]} .. {dates[-1]})")
    return len(dates)


def import_constituents(config) -> dict[str, int]:
    """Rebuild the constituents table from BaoStock monthly snapshots."""
    import baostock as bs

    for index_code in config.index_codes:
        if index_code not in BAOSTOCK_QUERIES:
            raise SystemExit(
                f"no historical-constituent source for {index_code}; remove it "
                "from config index_codes or add a provider mapping"
            )
    sessions = _sessions(config)
    samples = _month_end_samples(sessions)
    print(f"sampling membership at {len(samples)} month-end sessions "
          f"({samples[0]} .. {samples[-1]})")

    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"baostock login failed: {lg.error_msg}")
    try:
        rows: list[dict[str, str]] = []
        for index_code in config.index_codes:
            query_name = BAOSTOCK_QUERIES[index_code]
            timelines: dict[str, list[tuple[str, bool]]] = {}
            for date in samples:
                update, members = _snapshot_members(bs, query_name, date)
                for code in sorted(set(timelines) | members):
                    entry = (update, code in members)
                    history = timelines.setdefault(code, [])
                    if not history or history[-1] != entry:
                        history.append(entry)
            interval_rows = 0
            for code, present in timelines.items():
                dates = [d for d, _ in present]
                flags = [f for _, f in present]
                for in_date, out_date in _timeline_to_intervals(dates, flags):
                    rows.append(
                        {
                            "index_code": index_code,
                            "ts_code": code,
                            "in_date": in_date,
                            "out_date": out_date,
                        }
                    )
                    interval_rows += 1
            print(f"{index_code}: {len(timelines)} member codes, "
                  f"{interval_rows} intervals")
    finally:
        bs.logout()

    frame = pd.DataFrame(rows)
    issues = membership_interval_issues(frame)
    if issues:
        raise SystemExit("imported intervals failed validation: " + "; ".join(issues))
    with AshareDB(config.duckdb_path) as db:
        # create_schema also migrates the legacy two-column primary key to
        # (index_code, ts_code, in_date), which the upsert path requires.
        db.create_schema(config)
        # Replace (not merge) the whole table: leftover snapshot rows would
        # overlap the imported intervals and fail the contract.
        db.execute(f"DELETE FROM {config.constituents_table}")
        db.upsert_constituents(frame.to_dict("records"), config)
    print(f"constituents table rebuilt: {len(rows)} intervals")
    return {
        index_code: len(frame[frame["index_code"] == index_code])
        for index_code in config.index_codes
    }


def _all_stock_list_dates() -> pd.DataFrame:
    """Listing dates for every stock ever (BaoStock ``query_stock_basic``).

    The provider's stock-basic table covers listed and delisted codes with
    their real ``ipoDate``, so absorbed/terminated members of past index
    snapshots can be backfilled without fabrication.
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"baostock login failed: {lg.error_msg}")
    try:
        rs = bs.query_stock_basic()
        if rs.error_code != "0":
            raise RuntimeError(
                f"baostock query_stock_basic failed: {rs.error_code} {rs.error_msg}"
            )
        code_idx = rs.fields.index("code")
        ipo_idx = rs.fields.index("ipoDate")
        rows = []
        while rs.next():
            row = rs.get_row_data()
            code = str(row[code_idx])
            try:
                ts_code = _bs_code_to_ts_code(code)
            except ValueError:
                continue
            if not is_valid_a_share_code(ts_code):
                continue
            ipo = str(row[ipo_idx]).replace("-", "")
            rows.append({"ts_code": ts_code, "list_date": ipo or None})
    finally:
        bs.logout()
    return pd.DataFrame(rows).dropna(subset=["list_date"])


def import_list_dates(config) -> int:
    """Fill stocks.list_date for every stock (listed + delisted)."""
    client = AkShareClient(config)
    listed = client.get_stock_list()
    listed_dates = pd.DataFrame(
        {
            "ts_code": listed["ts_code"].astype(str),
            "list_date": listed["list_date"].map(normalize_listing_date),
        }
    ).dropna(subset=["list_date"])
    # BaoStock's all-stock table is the backstop for codes the exchange
    # batch metadata no longer lists (delisted / absorbed members).
    all_dates = _all_stock_list_dates()
    fetched = pd.concat([listed_dates, all_dates], ignore_index=True)
    fetched = fetched.drop_duplicates(subset=["ts_code"], keep="first")

    with AshareDB(config.duckdb_path) as db:
        stocks = db.query(f"SELECT * FROM {config.stocks_table}")
        merged = stocks.copy()
        merged["list_date"] = merged["list_date"].astype(object)
        for _, row in fetched.iterrows():
            code = row["ts_code"]
            if code in set(merged["ts_code"].astype(str)):
                mask = merged["ts_code"].astype(str) == code
                merged.loc[mask, "list_date"] = row["list_date"]
            else:
                merged = pd.concat(
                    [
                        merged,
                        pd.DataFrame(
                            [
                                {
                                    "ts_code": code,
                                    "name": code,
                                    "industry": None,
                                    "list_date": row["list_date"],
                                    "is_st": False,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
        db.upsert_stocks(merged.to_dict("records"), config)
    filled = int(merged["list_date"].notna().sum())
    print(f"stocks table: {len(merged)} rows, {filled} with list_date")
    return filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-calendar", action="store_true")
    parser.add_argument("--skip-list-dates", action="store_true")
    parser.add_argument("--skip-constituents", action="store_true")
    args = parser.parse_args()

    config = _load_config()
    if not args.skip_calendar:
        import_calendar(config)
    if not args.skip_list_dates:
        import_list_dates(config)
    if not args.skip_constituents:
        import_constituents(config)
    print("import complete; run scripts/check_production_gates.py to verify")


if __name__ == "__main__":
    sys.exit(main())
