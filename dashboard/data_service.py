"""Dashboard data access helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ashare_data.config import DataConfig, SimConfig
from ashare_data.db import AshareDB


def load_backtest_result(data_config: DataConfig) -> dict:
    path = data_config.data_dir / "backtest_result.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_sim_state(sim_config: SimConfig) -> dict:
    if not Path(sim_config.state_path).exists():
        return {}
    return json.loads(Path(sim_config.state_path).read_text(encoding="utf-8"))


def load_data_status(data_config: DataConfig) -> dict:
    if not Path(data_config.duckdb_path).exists():
        return {"ready": False, "reason": "DuckDB file not found"}
    try:
        # Read-only: the dashboard must work while a sync/sim run holds the
        # single-writer lock on the database file.
        with AshareDB(data_config.duckdb_path, read_only=True) as db:
            stocks = db.query(f"SELECT COUNT(*) AS n FROM {data_config.stocks_table}")
            daily = db.query(
                f"SELECT COUNT(*) AS n, MAX(trade_date) AS last_date FROM {data_config.daily_table}"
            )
            return {
                "ready": True,
                "stocks": int(stocks.iloc[0]["n"]),
                "daily_rows": int(daily.iloc[0]["n"]),
                "last_trade_date": str(daily.iloc[0]["last_date"]),
            }
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "reason": str(exc)}
