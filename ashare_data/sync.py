"""AkShare data synchronisation entry point.

Usage:
    python -m ashare_data.sync [--config config/ashare_config.yaml]
                              [--offline] [--limit N]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from .akshare_client import (
    AkShareClient,
    AkShareUnavailable,
    normalize_stock_metadata,
)
from .config import (
    DataConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_sim_config,
)
from .db import AshareDB, sql_quoted_list
from .processor import is_valid_a_share_code, normalize_daily_bars
from .universe import member_bar_coverage
from ashare_logging import export_log_txt, setup_run_logging


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_data_config(config_path: str | Path | None) -> DataConfig:
    root = _project_root()
    raw = load_config(config_path, project_root=root)
    return make_data_config(raw, root)


def _parquet_cache_path(config: DataConfig, ts_code: str) -> Path:
    return config.parquet_dir / "daily" / f"{ts_code}.parquet"


def _read_cached_bars(config: DataConfig, ts_code: str) -> pd.DataFrame:
    path = _parquet_cache_path(config, ts_code)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read parquet cache {path}: {exc}")
        return pd.DataFrame()


def _write_cached_bars(config: DataConfig, ts_code: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path = _parquet_cache_path(config, ts_code)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    shutil.move(str(tmp), str(path))


def _calendar_rows(dates: list[str]) -> list[dict[str, Any]]:
    return [{"trade_date": d, "is_open": True} for d in dates]


def _purge_stale_daily_rows(
    db: AshareDB,
    config: DataConfig,
    universe: list[str],
    failures: list[str],
) -> int:
    """Delete daily-bar rows for codes outside the synced universe.

    Only used on full (unlimited) syncs so a ``--limit`` run never destroys
    previously synced data; codes that failed to fetch are kept so a flaky
    network does not erase history.
    """

    if not universe:
        return 0
    keep = set(universe) | set(failures)
    quoted = sql_quoted_list(sorted(keep))
    before = db.query(f"SELECT COUNT(*) AS n FROM {config.daily_table}").iloc[0]["n"]
    db.execute(
        f"DELETE FROM {config.daily_table} WHERE ts_code NOT IN ({quoted})"
    )
    after = db.query(f"SELECT COUNT(*) AS n FROM {config.daily_table}").iloc[0]["n"]
    removed = int(before) - int(after)
    if removed:
        logger.info(f"Purged {removed} stale daily-bar rows outside the universe")
    return removed


def _purge_stale_parquet_files(config: DataConfig, keep: set[str]) -> int:
    """Delete parquet cache files for codes outside the synced universe."""
    daily_dir = config.parquet_dir / "daily"
    if not daily_dir.exists():
        return 0
    removed = 0
    for path in daily_dir.glob("*.parquet"):
        if path.stem not in keep:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
    if removed:
        logger.info(f"Removed {removed} stale parquet cache files")
    return removed


def _stock_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    return normalize_stock_metadata(df).to_dict("records")


def _bar_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    df = normalize_daily_bars(df)
    if df.empty:
        return []
    cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "turnover_rate",
        "adj_factor",
    ]
    return df[cols].to_dict("records")


def _augment_sync_universe(
    db: AshareDB,
    config: DataConfig,
    base_codes: set[str],
) -> list[str]:
    """Extend the snapshot-derived sync codes with every historical PIT
    member and every cached-bar code (survivorship-aware union).

    The current constituent snapshot only names today's members: syncing
    exactly that set leaves delisted/historical members without bars and
    ``build_universe_mask`` silently marks them MISSING_BAR, biasing every
    historical backtest optimistically (H3).  Codes appearing in the PIT
    ``constituents`` table and codes with a local bar cache therefore join
    the sync universe unconditionally — only the valid A-share code check
    applies, never the current stock-list intersection, or a delisted
    member would be dropped again.
    """

    pit = {
        str(code)
        for code in db.query(
            f"SELECT DISTINCT ts_code FROM {config.constituents_table}"
        )["ts_code"]
    }
    daily_dir = config.parquet_dir / "daily"
    cached = (
        {path.stem for path in daily_dir.glob("*.parquet")}
        if daily_dir.exists()
        else set()
    )
    union = (
        base_codes
        | {c for c in pit if is_valid_a_share_code(c)}
        | {c for c in cached if is_valid_a_share_code(c)}
    )
    return sorted(union)


def sync_all(
    config_path: str | Path | None = None,
    offline: bool | None = None,
    limit: int | None = None,
    sync_fundamentals: bool | None = None,
    sync_capital_flow: bool | None = None,
) -> dict[str, Any]:
    """Synchronize calendar, stocks, constituents, daily bars and (unless
    disabled) quarterly fundamentals plus margin/Shenwan-industry data."""

    config = _load_data_config(config_path)
    if sync_fundamentals is not None:
        config.sync_fundamentals = sync_fundamentals
    if sync_capital_flow is not None:
        config.sync_capital_flow = sync_capital_flow
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.parquet_dir.mkdir(parents=True, exist_ok=True)

    client = AkShareClient(config, offline=offline)
    db = AshareDB(config.duckdb_path)
    try:
        db.create_schema(config)

        dates = client.get_trade_calendar()
        # Calendar is a snapshot for the configured range: replace it so
        # stale rows (e.g. a previous business-day fallback) disappear.
        db.execute(f"DELETE FROM {config.calendar_table}")
        db.upsert_calendar(_calendar_rows(dates), config)
        logger.info(f"Trade calendar synced: {len(dates)} days")
        # Latest trade date up to today: future entries in the calendar must
        # not mark every cache as permanently stale.
        today = pd.Timestamp.now().strftime("%Y%m%d")
        refresh_cutoff = max(
            (d for d in dates if d <= today), default=dates[-1]
        )

        stocks_df = client.get_stock_list()
        db.upsert_stocks(_stock_rows(stocks_df), config)
        logger.info(f"Stock list synced: {len(stocks_df)} stocks")

        all_constituents: set[str] = set()
        for index_code in config.index_codes:
            codes = client.get_constituents(index_code)
            all_constituents.update(codes)
            logger.info(f"Current constituent snapshot for {index_code}: {len(codes)}")
        if all_constituents:
            logger.warning(
                "Current constituent snapshots are only one of three sync-universe "
                "sources (plus PIT constituents and the local bar cache); no PIT "
                "membership intervals were written by this sync"
            )

        # Defensive validation: only real A-share stock codes may enter the
        # trading universe (index symbols, B-shares etc. are dropped).  The
        # stock list from the exchange is the authoritative membership check
        # for *snapshot* codes, which also catches index codes that share the
        # 000xxx space with real SZ stocks.
        stock_codes = (
            {str(c) for c in stocks_df["ts_code"].astype(str)}
            if stocks_df is not None and not stocks_df.empty
            else set()
        )
        candidate_universe = sorted(
            {c for c in all_constituents if is_valid_a_share_code(c)}
        )
        if not candidate_universe:
            # Fallback when no constituent data is available: the validated
            # stock list itself.
            candidate_universe = sorted(
                {c for c in stock_codes if is_valid_a_share_code(c)}
            )
        if stock_codes:
            base_codes = set(candidate_universe) & stock_codes
        else:
            base_codes = set(candidate_universe)
        # Survivorship-aware union: historical PIT members and cached-bar
        # codes join even when absent from the current snapshot/stock list
        # (delisted members), so their bars are backfilled and never purged.
        universe = _augment_sync_universe(db, config, base_codes)
        if limit:
            universe = universe[:limit]

        start = config.start_date.replace("-", "")
        end = config.end_date.replace("-", "")
        total_rows = 0
        failures: list[str] = []

        for ts_code in universe:
            df = _read_cached_bars(config, ts_code)
            stale = False
            if not df.empty and "trade_date" in df.columns:
                max_cached = str(df["trade_date"].max())
                if max_cached < refresh_cutoff:
                    stale = True
            if df.empty or "trade_date" not in df.columns or stale:
                if stale:
                    logger.info(
                        f"Cache stale for {ts_code} ({max_cached} < {refresh_cutoff}); "
                        "refreshing"
                    )
                try:
                    df = client.get_daily_bar(ts_code, start_date=start, end_date=end)
                    if df is not None and not df.empty:
                        _write_cached_bars(config, ts_code, df)
                except AkShareUnavailable as exc:
                    logger.warning(f"Daily bar fetch failed for {ts_code}: {exc}")
                    failures.append(ts_code)
                    continue
            if df is None or df.empty:
                failures.append(ts_code)
                continue
            rows = _bar_rows(df)
            if rows:
                db.upsert_daily(rows, config)
                total_rows += len(rows)

        purged = 0
        purged_parquet = 0
        if not limit:
            purged = _purge_stale_daily_rows(db, config, universe, failures)
            purged_parquet = _purge_stale_parquet_files(
                config, set(universe) | set(failures)
            )

        fundamental_stats = {
            "fundamental_quarters": 0,
            "fundamental_rows": 0,
            "fundamental_supplements": 0,
            "fundamental_failures": 0,
        }
        if config.sync_fundamentals and not offline:
            from .fundamentals import sync_fundamentals

            fundamental_stats = sync_fundamentals(client, db, config, universe)

        capital_stats = {
            "margin_rows": 0,
            "margin_dates": 0,
            "industries": 0,
            "industry_rows": 0,
            "capital_failures": 0,
        }
        if config.sync_capital_flow and not offline:
            from .capital_flow import sync_capital_flow

            capital_stats = sync_capital_flow(client, db, config, dates)

        logger.info(
            f"Daily bars synced: {total_rows} rows, failures: {len(failures)}, "
            f"purged: {purged}"
        )
        # Observability for the survivorship-aware union: a falling count
        # after a constituent-table refresh means PIT members lack bars.
        logger.info(
            "Sync universe: snapshot-validated base + PIT members + cached "
            f"codes = {len(universe)} codes (snapshot symbols: "
            f"{len(all_constituents)})"
        )
        return {
            "calendar_days": len(dates),
            "stocks": len(stocks_df),
            "universe": len(universe),
            "constituent_snapshot_symbols": len(all_constituents),
            "pit_constituent_rows_written": 0,
            "daily_rows": total_rows,
            "failures": failures,
            "purged_rows": purged,
            "purged_parquet": purged_parquet,
            **fundamental_stats,
            **capital_stats,
        }
    finally:
        db.close()


def backfill_member_bars(
    config: DataConfig,
    client: AkShareClient | None = None,
) -> dict[str, Any]:
    """Backfill daily bars for historical PIT members with zero-bar
    intervals (delisted / merged / never-synced members).

    The zero-bar audit comes from :func:`ashare_data.universe.member_bar_coverage`
    (half-open intervals, capped to the daily-bar horizon): every member
    whose interval overlaps the data window without a single bar is
    fetched through the client (the Tencent fallback serves delisted
    history) and upserted into both the daily table and the parquet cache.

    When the main fetch still leaves an interval's audited span bare — the
    member was suspended through the whole span (e.g. 宏源证券 absorbed
    2015, 中银绒业's long restructuring suspension) — the Baostock record
    supplements the official suspension rows (flat price, zero volume),
    which the pipeline's tradability layer treats as blocked, so presence
    never fabricates a tradeable member.

    Idempotent: codes that already have bars are untouched, and a re-run
    audits the current state.  Returns per-code statistics plus the
    re-audited remaining zero-bar interval count.
    """

    with AshareDB(config.duckdb_path, read_only=True) as db:
        coverage = member_bar_coverage(db, config)
    observed = coverage[coverage["sessions"] > 0]
    zero_bar = observed[observed["bars"] == 0]
    intervals = zero_bar.loc[:, ["index_code", "ts_code", "in_date", "out_date"]]

    client = client or AkShareClient(config)
    start = config.start_date.replace("-", "")
    end = config.end_date.replace("-", "")
    result: dict[str, Any] = {
        "audited_zero_bar_codes": int(zero_bar["ts_code"].nunique()),
        "fetched_codes": 0,
        "daily_rows": 0,
        "failures": [],
    }
    baostock_fallback = getattr(client, "_get_daily_bar_baostock", None)
    db = AshareDB(config.duckdb_path)
    try:
        for _, interval in intervals.iterrows():
            code = str(interval["ts_code"])
            span_start = max(str(interval["in_date"]), start)
            span_end = min(str(interval["out_date"]), end)
            frames: list[pd.DataFrame] = []
            df = client.get_daily_bar(code, start_date=start, end_date=end)
            if df is not None and not df.empty:
                frames.append(df)
            # Suspension-span supplement: if the interval's audited span
            # still has no bars, the Baostock record (which includes the
            # official suspension rows) fills it.
            have = db.query(
                f"SELECT COUNT(*) AS n FROM {config.daily_table} "
                f"WHERE ts_code = ? AND trade_date >= ? AND trade_date < ?",
                [code, span_start, span_end],
            ).iloc[0]["n"]
            if have == 0 and baostock_fallback is not None:
                bs_df = baostock_fallback(code, span_start, span_end)
                if bs_df is not None and not bs_df.empty:
                    frames.append(bs_df)
            if not frames:
                result["failures"].append(code)
                continue
            combined = pd.concat(frames, ignore_index=True)
            _write_cached_bars(config, code, combined)
            rows = _bar_rows(combined)
            if rows:
                db.upsert_daily(rows, config)
                result["daily_rows"] += len(rows)
            result["fetched_codes"] += 1
    finally:
        db.close()

    with AshareDB(config.duckdb_path, read_only=True) as db:
        after = member_bar_coverage(db, config)
    remaining = after[(after["sessions"] > 0) & (after["bars"] == 0)]
    result["remaining_zero_bar_intervals"] = int(len(remaining))
    return result


def main() -> None:
    setup_run_logging(run_name="sync")
    parser = argparse.ArgumentParser(description="Sync A-share data via AkShare")
    parser.add_argument("--config", default=None, help="Path to ashare_config.yaml")
    parser.add_argument("--offline", action="store_true", help="Use local fixtures only")
    parser.add_argument("--limit", type=int, default=None, help="Limit universe size")
    parser.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="Skip the quarterly fundamental sync",
    )
    parser.add_argument(
        "--no-capital-flow",
        action="store_true",
        help="Skip the margin/Shenwan-industry sync",
    )
    args = parser.parse_args()
    try:
        result = sync_all(
            args.config,
            offline=args.offline,
            limit=args.limit,
            sync_fundamentals=False if args.no_fundamentals else None,
            sync_capital_flow=False if args.no_capital_flow else None,
        )
        logger.success(f"Sync complete: {result}")
    finally:
        export_log_txt(run_name="sync")


if __name__ == "__main__":
    main()
