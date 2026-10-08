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

Safety (P1 review follow-up): the constituents rebuild is a full-table
replace.  ``--dry-run`` runs every validation and prints the would-be
delete/insert counts without touching the database, and the destructive
run first dumps the rows it is about to delete to a JSON backup under
``--backup-dir`` (default ``data/backups/``, matching the existing
``data/backup_<date>_<tag>`` convention) and aborts when that backup
cannot be written (fail-closed).

Run both repairs, then verify with ``scripts/check_production_gates.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
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
from ashare_data.pit_import import (
    bs_code_to_ts_code,
    merge_list_dates,
    month_end_samples,
    timeline_to_intervals,
)
from ashare_data.processor import is_valid_a_share_code
from ashare_data.universe import membership_interval_issues

# BaoStock per-date constituent queries per configured index code.
BAOSTOCK_QUERIES = {
    "000300.SH": "query_hs300_stocks",
    "000905.SH": "query_zz500_stocks",
}


class BackupError(RuntimeError):
    """A pre-delete backup could not be written; the destructive step
    must not run (fail-closed)."""


def _resolve_backup_dir(raw: str | Path | None) -> Path:
    """Backup root: ``data/backups/`` under the project root by default.

    Mirrors the existing convention of ``data/backup_<date>_<tag>``
    directories that hold whole-DB copies made around migrations; this
    script keeps its row-level JSON dumps in a dedicated subdirectory.
    """
    if raw is None:
        return _ROOT / "data" / "backups"
    path = Path(raw)
    return path if path.is_absolute() else _ROOT / path


def _json_records(df) -> list[dict]:
    """DataFrame rows as JSON-safe dicts (NULL/NaN become ``None``)."""
    if df is None or len(df) == 0:
        return []
    cleaned = df.astype(object).where(pd.notna(df), None)
    return cleaned.to_dict("records")


def write_backup_json(path: Path, payload: dict) -> None:
    """Atomically persist ``payload`` and verify it reads back.

    Any failure raises :class:`BackupError` so callers abort BEFORE the
    destructive statement runs (fail-closed).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(text):
            raise BackupError(f"backup verification mismatch for {path}")
    except BackupError:
        raise
    except (OSError, ValueError) as exc:  # noqa: BLE001 - fail closed
        raise BackupError(f"backup write failed for {path}: {exc}") from exc


def _table_rows(db, table: str) -> list[dict]:
    """Every row of ``table`` (empty when the table does not exist yet)."""
    try:
        return _json_records(db.query(f"SELECT * FROM {table}"))
    except Exception:  # noqa: BLE001 - table may not exist yet.
        return []


def rebuild_constituents_table(
    db,
    config,
    records: list[dict],
    *,
    dry_run: bool = False,
    backup_dir: Path | None = None,
) -> dict[str, object]:
    """Apply — or, with ``dry_run``, only measure — the constituents
    full-table rebuild (P1 review follow-up).

    Contract under test:

    * ``dry_run`` never issues a write (it only counts what a real run
      would delete and insert).
    * The real run writes the rows the ``DELETE`` is about to remove to
      a JSON backup first (:func:`write_backup_json`); if that backup
      cannot be written the function raises :class:`BackupError` and no
      ``DELETE`` is ever issued.
    """
    table = config.constituents_table
    try:
        existing_rows = int(
            db.query(f"SELECT count(*) AS n FROM {table}")["n"].iloc[0]
        )
    except Exception:  # noqa: BLE001 - table may not exist yet.
        existing_rows = 0
    if dry_run:
        print(
            f"dry-run: constituents rebuild would delete {existing_rows} "
            f"existing rows and insert {len(records)} intervals; "
            "database untouched"
        )
        return {
            "dry_run": 1,
            "existing_rows": existing_rows,
            "inserted_rows": len(records),
        }
    # SQL-injection hardening: verbatim statement for the registered
    # default table (any other configured name fails closed).  Checked
    # BEFORE the backup so a rejected table name writes nothing.
    if table != "constituents":
        raise ValueError(
            f"no literal constituents purge statement for table {table!r}"
        )
    backup_root = _resolve_backup_dir(backup_dir)
    backup_path = backup_root / (
        "constituents_backup_"
        + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    doomed = _table_rows(db, table)
    write_backup_json(
        backup_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "table": table,
            "row_count": len(doomed),
            "rows": doomed,
        },
    )
    print(f"backup: {len(doomed)} constituents rows -> {backup_path}")
    # create_schema also migrates the legacy two-column primary key to
    # (index_code, ts_code, in_date), which the upsert path requires.
    db.create_schema(config)
    # Replace (not merge) the whole table: leftover snapshot rows would
    # overlap the imported intervals and fail the contract.
    db.execute("DELETE FROM constituents")
    db.upsert_constituents(records, config)
    return {
        "dry_run": 0,
        "existing_rows": existing_rows,
        "inserted_rows": len(records),
        "backed_up_rows": len(doomed),
        "backup_path": str(backup_path),
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
            ts_code = bs_code_to_ts_code(code)
        except ValueError:
            continue
        if is_valid_a_share_code(ts_code):
            members.add(ts_code)
    if not members or not update_dates:
        raise RuntimeError(
            f"baostock {query_name}({query_date}) returned no members"
        )
    return max(update_dates), members


def import_calendar(config, dry_run: bool = False) -> int:
    """Backfill the session calendar with the provider's full history.

    Listing age is measured in open sessions, so the calendar must contain
    sessions BEFORE the dataset window; without them every stock fails the
    configured 60-session age gate during the first ~60 dataset sessions.
    """

    client = AkShareClient(config)
    dates = client.get_trade_calendar()
    if dry_run:
        print(
            f"dry-run: would upsert {len(dates)} calendar sessions "
            f"({dates[0]} .. {dates[-1]}); database untouched"
        )
        return len(dates)
    rows = [{"trade_date": d, "is_open": True} for d in dates]
    with AshareDB(config.duckdb_path) as db:
        db.upsert_calendar(rows, config)
    print(f"calendar sessions: {len(dates)} ({dates[0]} .. {dates[-1]})")
    return len(dates)


def import_constituents(
    config,
    dry_run: bool = False,
    backup_dir: Path | None = None,
) -> dict[str, int]:
    """Rebuild the constituents table from BaoStock monthly snapshots.

    ``dry_run`` stops short of every database write (validation and the
    per-index statistics still run); otherwise the rows removed by the
    full-table replace are backed up first (fail-closed, P1).
    """
    import baostock as bs

    for index_code in config.index_codes:
        if index_code not in BAOSTOCK_QUERIES:
            raise SystemExit(
                f"no historical-constituent source for {index_code}; remove it "
                "from config index_codes or add a provider mapping"
            )
    sessions = _sessions(config)
    samples = month_end_samples(sessions)
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
                for in_date, out_date in timeline_to_intervals(dates, flags):
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
    records = frame.to_dict("records")
    try:
        # read-only in dry-run: any accidental write raises immediately.
        with AshareDB(config.duckdb_path, read_only=dry_run) as db:
            outcome = rebuild_constituents_table(
                db, config, records, dry_run=dry_run, backup_dir=backup_dir
            )
    except BackupError as exc:
        raise SystemExit(
            f"constituents rebuild aborted — backup failed: {exc}"
        ) from exc
    if dry_run:
        print("constituents rebuild: dry run complete; database untouched")
    else:
        print(
            f"constituents table rebuilt: {len(rows)} intervals "
            f"(backup: {outcome['backed_up_rows']} rows -> "
            f"{outcome['backup_path']})"
        )
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
                ts_code = bs_code_to_ts_code(code)
            except ValueError:
                continue
            if not is_valid_a_share_code(ts_code):
                continue
            ipo = str(row[ipo_idx]).replace("-", "")
            rows.append({"ts_code": ts_code, "list_date": ipo or None})
    finally:
        bs.logout()
    return pd.DataFrame(rows).dropna(subset=["list_date"])


def import_list_dates(config, dry_run: bool = False) -> int:
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

    # read_only under --dry-run keeps the no-write guarantee at the
    # connection level, matching import_constituents' double guard.
    with AshareDB(config.duckdb_path, read_only=dry_run) as db:
        stocks = db.query(f"SELECT * FROM {config.stocks_table}")
        # One vectorized merge replaces the previous per-row concat +
        # in-loop membership scan (O(n^2)); upsert_stocks then merges the
        # result without deleting anything.
        merged = merge_list_dates(stocks, fetched)
        if not dry_run:
            db.upsert_stocks(merged.to_dict("records"), config)
    filled = int(merged["list_date"].notna().sum())
    label = "dry-run: would refresh stocks table" if dry_run else "stocks table"
    print(f"{label}: {len(merged)} rows, {filled} with list_date")
    return filled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-calendar", action="store_true")
    parser.add_argument("--skip-list-dates", action="store_true")
    parser.add_argument("--skip-constituents", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every validation and print the would-be delete/insert "
        "counts without writing anything to the database",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="directory for the pre-delete JSON backup of the "
        "constituents table (default: data/backups/ under the project "
        "root); the rebuild aborts when the backup cannot be written",
    )
    args = parser.parse_args()

    config = _load_config()
    backup_dir = _resolve_backup_dir(args.backup_dir)
    if not args.skip_calendar:
        import_calendar(config, dry_run=args.dry_run)
    if not args.skip_list_dates:
        import_list_dates(config, dry_run=args.dry_run)
    if not args.skip_constituents:
        import_constituents(
            config, dry_run=args.dry_run, backup_dir=backup_dir
        )
    print("import complete; run scripts/check_production_gates.py to verify")


if __name__ == "__main__":
    sys.exit(main())
