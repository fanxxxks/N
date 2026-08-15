from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data import config as config_module
from ashare_data.akshare_client import AkShareClient
from ashare_data.capital_flow import (
    EXTERNAL_FACTOR_NAMES,
    build_capital_frames,
    sync_capital_flow,
)
from ashare_data.db import AshareDB


def _data_config(tmp_path: Path) -> config_module.DataConfig:
    return config_module.DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )


def _dates(n: int = 30) -> list[str]:
    return pd.bdate_range("2024-01-01", periods=n).strftime("%Y%m%d").tolist()


def test_client_offline_margin_normalization(tmp_path: Path):
    cfg = config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "margin_sse_20240102.json").write_text(
        '[{"信用交易日期": "2024-01-02", "标的证券代码": "600000", "融资余额": 1.5e9,'
        ' "融资买入额": 1.0e8}]',
        encoding="utf-8",
    )
    (fx / "margin_szse_20240102.json").write_text(
        '[{"证券代码": "000001", "融资余额": 2.0e9, "融资买入额": 2.0e8},'
        ' {"证券代码": "900901", "融资余额": 1.0e8}]',
        encoding="utf-8",
    )
    client = AkShareClient(cfg, offline=True, fixture_dir=fx)
    sse = client.get_margin_detail("sse", "20240102")
    assert sse.iloc[0]["ts_code"] == "600000.SH"
    assert sse.iloc[0]["rzye"] == pytest.approx(1.5e9)
    szse = client.get_margin_detail("szse", "20240102")
    # B-share dropped by the validator.
    assert szse["ts_code"].tolist() == ["000001.SZ"]
    assert szse.iloc[0]["trade_date"] == "20240102"


def test_client_offline_sw_industry_normalization(tmp_path: Path):
    cfg = config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "sw_industries.json").write_text(
        '[{"行业代码": "801010.SI", "行业名称": "农林牧渔"}]', encoding="utf-8"
    )
    (fx / "sw_index_801010.json").write_text(
        '[{"代码": "801010", "日期": "2024-01-02", "收盘": 3000.0}]',
        encoding="utf-8",
    )
    (fx / "sw_components_801010.json").write_text(
        '[{"证券代码": "000505"}, {"证券代码": "900901"}]', encoding="utf-8"
    )
    client = AkShareClient(cfg, offline=True, fixture_dir=fx)
    industries = client.get_sw_industry_list()
    assert industries.iloc[0]["index_code"] == "801010"
    assert industries.iloc[0]["industry_name"] == "农林牧渔"
    hist = client.get_sw_index_history("801010")
    assert hist.iloc[0]["trade_date"] == "20240102"
    assert hist.iloc[0]["close"] == pytest.approx(3000.0)
    comp = client.get_sw_components("801010")
    assert comp["ts_code"].tolist() == ["000505.SZ"]


def test_build_capital_frames_margin_and_industry(tmp_path: Path):
    cfg = _data_config(tmp_path)
    dates = _dates(30)
    codes = ["000001.SZ", "600000.SH", "000505.SZ"]
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        # Margin balance: 000001 grows 10% per day; 600000 constant.
        margin_rows = []
        for i, d in enumerate(dates):
            margin_rows.append({"ts_code": "000001.SZ", "trade_date": d, "rzye": 100.0 * 1.001**i})
            margin_rows.append({"ts_code": "600000.SH", "trade_date": d, "rzye": 50.0})
        db.upsert_margin(margin_rows, cfg)
        # Industry: 801010 index grows 1% per day; members 000001/000505.
        index_rows = [
            {"index_code": "801010", "industry_name": "农业", "trade_date": d, "close": 1000.0 * 1.01**i}
            for i, d in enumerate(dates)
        ]
        db.upsert_sw_index(index_rows, cfg)
        db.replace_sw_members("801010", ["000001.SZ", "000505.SZ"], cfg)

    frames = build_capital_frames(cfg, codes, dates)
    assert set(frames) == set(EXTERNAL_FACTOR_NAMES)

    margin = frames["MARGIN_BALANCE_CHG"]
    # 20-day change of the growing balance: 1.001^20 - 1 for 000001; 0 for
    # the constant stock; the first 20 days have no base -> NaN.
    assert margin.loc["000001.SZ", dates[20]] == pytest.approx(1.001**20 - 1.0)
    assert margin.loc["600000.SH", dates[20]] == pytest.approx(0.0)
    assert np.isnan(margin.loc["000001.SZ", dates[19]])

    industry = frames["INDUSTRY_MOMENTUM"]
    # Every member inherits the industry index 20-day return; non-members
    # stay neutral.
    assert industry.loc["000001.SZ", dates[20]] == pytest.approx(1.01**20 - 1.0)
    assert industry.loc["000505.SZ", dates[20]] == pytest.approx(1.01**20 - 1.0)
    assert np.isnan(industry.loc["600000.SH", dates[20]])
    assert np.isnan(industry.loc["000001.SZ", dates[19]])


def test_build_capital_frames_degrades_without_tables(tmp_path: Path):
    cfg = _data_config(tmp_path)
    frames = build_capital_frames(cfg, ["000001.SZ"], _dates(5))
    assert set(frames) == set(EXTERNAL_FACTOR_NAMES)
    assert frames["MARGIN_BALANCE_CHG"].isna().all().all()


def test_sync_capital_flow_offline_skips(tmp_path: Path):
    cfg = _data_config(tmp_path)
    client = AkShareClient(cfg, offline=True, fixture_dir=Path("tests/fixtures_missing"))
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        result = sync_capital_flow(client, db, cfg, _dates(5))
    assert result == {
        "margin_rows": 0,
        "margin_dates": 0,
        "industries": 0,
        "industry_rows": 0,
        "capital_failures": 0,
    }
