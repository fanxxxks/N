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

from .akshare_client import AkShareClient, AkShareUnavailable
from .config import (
    DataConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_sim_config,
)
from .db import AshareDB
from .processor import is_valid_a_share_code, normalize_daily_bars
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
    quoted = ",".join(f"'{c}'" for c in sorted(keep))
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
    if df is None or df.empty:
        return []
    df = df.copy()
    for col in ("ts_code", "name", "industry", "list_date"):
        if col not in df.columns:
            df[col] = None
    if "is_st" not in df.columns:
        df["is_st"] = df["name"].astype(str).str.contains("ST", na=False)
    return df[["ts_code", "name", "industry", "list_date", "is_st"]].to_dict("records")


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


def sync_all(
    config_path: str | Path | None = None,
    offline: bool | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Synchronize calendar, stocks, constituents, and daily bars."""

    config = _load_data_config(config_path)
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

        constituent_rows: list[dict[str, Any]] = []
        all_constituents: set[str] = set()
        for index_code in config.index_codes:
            codes = client.get_constituents(index_code)
            constituent_rows.extend(
                {
                    "index_code": index_code,
                    "ts_code": code,
                    "in_date": config.start_date.replace("-", ""),
                    "out_date": "99991231",
                }
                for code in codes
            )
            all_constituents.update(codes)
            logger.info(f"Constituents for {index_code}: {len(codes)}")
        if constituent_rows:
            # Constituent data is a current snapshot: replace the table so
            # stale memberships from previous runs disappear.
            db.execute(f"DELETE FROM {config.constituents_table}")
            db.upsert_constituents(constituent_rows, config)

        # Defensive validation: only real A-share stock codes may enter the
        # trading universe (index symbols, B-shares etc. are dropped).  The
        # stock list from the exchange is the authoritative membership check,
        # which also catches index codes that share the 000xxx space with
        # real SZ stocks.
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
            universe = [c for c in candidate_universe if c in stock_codes]
        else:
            universe = candidate_universe
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

        logger.info(
            f"Daily bars synced: {total_rows} rows, failures: {len(failures)}, "
            f"purged: {purged}"
        )
        return {
            "calendar_days": len(dates),
            "stocks": len(stocks_df),
            "universe": len(universe),
            "daily_rows": total_rows,
            "failures": failures,
            "purged_rows": purged,
            "purged_parquet": purged_parquet,
        }
    finally:
        db.close()


def main() -> None:
    setup_run_logging(run_name="sync")
    parser = argparse.ArgumentParser(description="Sync A-share data via AkShare")
    parser.add_argument("--config", default=None, help="Path to ashare_config.yaml")
    parser.add_argument("--offline", action="store_true", help="Use local fixtures only")
    parser.add_argument("--limit", type=int, default=None, help="Limit universe size")
    args = parser.parse_args()
    try:
        result = sync_all(args.config, offline=args.offline, limit=args.limit)
        logger.success(f"Sync complete: {result}")
    finally:
        export_log_txt(run_name="sync")


if __name__ == "__main__":
    main()
