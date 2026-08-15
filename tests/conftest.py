"""Shared pytest fixtures and session-level run logging."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ashare_data.config import DataConfig
from ashare_data.db import AshareDB
from ashare_logging import export_log_txt, setup_run_logging


DEFAULT_CODES = ["000001.SZ", "600000.SH", "300001.SZ"]


def make_bars(
    n_dates: int = 40,
    ts_codes: list[str] | None = None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    ts_codes = ts_codes or DEFAULT_CODES
    dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y%m%d").tolist()
    rows = []
    for code in ts_codes:
        for i, date in enumerate(dates):
            close = 10.0 + i * 0.1 + (0.5 if code == "000001.SZ" else 0.0)
            open_ = close * 0.99
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": open_,
                    "high": close * 1.02,
                    "low": open_ * 0.98,
                    "close": close,
                    "pre_close": close * 0.99,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * close,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
            )
    return dates, ts_codes, pd.DataFrame(rows)


@pytest.fixture(scope="session", autouse=True)
def _run_logging():
    setup_run_logging(run_name="pytest")
    yield
    project_root = Path(__file__).resolve().parents[1]
    export_log_txt(path=project_root / "logs" / "pytest.txt")


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        min_listed_days=1,
        min_amount=1.0,
    )


@pytest.fixture
def bars_data():
    return make_bars()


@pytest.fixture
def populated_db(data_config: DataConfig, bars_data) -> DataConfig:
    dates, ts_codes, bars = bars_data
    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    data_config.parquet_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {
                    "ts_code": c,
                    "name": c,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for c in ts_codes
            ],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in dates],
            data_config,
        )
        db.upsert_constituents(
            [
                {
                    "index_code": "000300.SH",
                    "ts_code": c,
                    "in_date": "20240101",
                    "out_date": "99991231",
                }
                for c in ts_codes
            ],
            data_config,
        )
    return data_config
