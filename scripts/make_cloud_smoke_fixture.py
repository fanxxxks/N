#!/usr/bin/env python3
"""Create a published synthetic database PAIR for CLI functional validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def create_fixture(directory):
    import numpy as np
    import pandas as pd
    import yaml
    from ashare_data.config import DataConfig
    from ashare_data.db import AshareDB
    from ashare_data.tushare_store import TushareStore
    from ashare_data.manifest import TUSHARE_V2_CONFIG, publish_snapshot
    from ashare_data.processor import is_valid_a_share_code
    from ashare_model.runspec import publish_composite_snapshot
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    dates = pd.bdate_range("2022-01-03", periods=340).strftime("%Y%m%d").tolist()
    calendar = pd.bdate_range(end=dates[-1], periods=430).strftime("%Y%m%d").tolist()
    codes = [f"{i:06d}.SZ" for i in range(1, 100) if is_valid_a_share_code(f"{i:06d}.SZ")][:32]
    config = DataConfig(data_dir=directory, duckdb_path=directory / "ashare.duckdb",
        v2_duckdb_path=directory / "ashare_v2.duckdb", index_codes=["000300.SH"])
    rng = np.random.default_rng(71)
    rows = []
    for i, code in enumerate(codes):
        price = 4 + i / 8
        for t, day in enumerate(dates):
            previous = price
            price *= 1 + (i - 15) * 0.00012 + rng.normal(0, 0.005)
            rows.append(dict(ts_code=code, trade_date=day, open=price, high=price * 1.01,
                low=price * 0.99, close=price * (1 + rng.normal(0, 0.001)), pre_close=previous,
                volume=1000000.0 + rng.uniform(0, 200000), amount=price * 1100000,
                turnover_rate=1 + rng.uniform(0, 0.3), adj_factor=1.0))
    bars = pd.DataFrame(rows)
    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
        db.upsert_stocks([dict(ts_code=c, name=c, industry=None, list_date="20100101", is_st=False) for c in codes], config)
        db.upsert_calendar([dict(trade_date=d, is_open=True) for d in calendar], config)
        db.upsert_constituents([dict(index_code="000300.SH", ts_code=c, in_date="20100101", out_date="20990101") for c in codes], config)
        db.upsert_daily(rows, config)
    with AshareDB(config.v2_duckdb_path) as db:
        TushareStore(db, directory, session_resolver=lambda d: d, code_commit="synthetic", sdk_version="synthetic").create_schema()
        frames = {
            "daily_bar_raw": bars.rename(columns={"volume": "vol"}).drop(columns=["turnover_rate", "adj_factor"]),
            "daily_adj_factor": bars[["ts_code", "trade_date"]].assign(adj_factor=1.0),
            "daily_price_limit": bars[["ts_code", "trade_date", "pre_close"]].assign(up_limit=bars.open * 2, down_limit=bars.open * 0.5),
            "daily_basic_pit": bars[["ts_code", "trade_date", "turnover_rate"]].assign(circ_mv=100000.0),
        }
        for table, frame in frames.items():
            hashes = frame.astype(str).agg("|".join, axis=1).map(lambda s: hashlib.sha256(s.encode()).hexdigest())
            frame = frame.assign(ingest_id="synthetic", source_row_hash=hashes)
            db.register("_fixture_rows", frame)
            columns = ",".join(frame.columns)
            db.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM _fixture_rows")
        publish_snapshot(db, TUSHARE_V2_CONFIG, source_versions={"fixture": "synthetic-v1"})
    with AshareDB(config.duckdb_path) as a, AshareDB(config.v2_duckdb_path, read_only=True) as b:
        snapshot = publish_composite_snapshot(a, config, b)
    settings = yaml.safe_load((ROOT / "config/cloud_four_search.yaml").read_text(encoding="utf-8"))
    settings.update(primary=str(config.duckdb_path), v2=str(config.v2_duckdb_path),
        output=str(directory.parent / (directory.name + "-runs")), dataset_id=snapshot.composite_id,
        budget=8, threads=1, time_limit=180, device="cpu", min_eligible=24,
        index_codes=["000300.SH"], horizon=1, synthetic_fixture=True, minimum_validation_excess=-1e9,
        windows=[dict(name="synthetic", start="2022-01-03", train_end=pd.Timestamp(dates[219]).strftime("%Y-%m-%d"),
            validation_end=pd.Timestamp(dates[279]).strftime("%Y-%m-%d"), test_end=pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"))])
    settings["feature_names"] = ["RET_1", "MOMENTUM_20", "REVERSAL_5", "VOL_20",
        "TURNOVER", "AMPLITUDE", "CLOSE_POSITION", "OVERNIGHT_RET", "ILLIQ_20"]
    settings["model"].update(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, num_loops=1,
        batch_size=8, max_formula_len=6, validation_splits=2, dropout=0.0)
    # Deliberately permissive artificial-data floors. These never enter the
    # real cloud config, and the identity declares this synthetic experiment.
    settings["reward"] = dict(ic_min_stocks=10, min_val_reward=-1e9, min_val_icir=-1e9,
        min_valid_ic_days=2, min_effective_stocks=10, min_coverage=0.0, min_activity=0.0,
        min_sign_stability=0.0, min_val_window_q25=-1e9)
    path = directory / "smoke.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(create_fixture(args.output))
