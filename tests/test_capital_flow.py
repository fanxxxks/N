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
    build_industry_member_frame,
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
        # Industry: 801010 index grows 1% per day; members 000001/000505.
        index_rows = [
            {"index_code": "801010", "industry_name": "农业", "trade_date": d, "close": 1000.0 * 1.01**i}
            for i, d in enumerate(dates)
        ]
        db.upsert_sw_index(index_rows, cfg)
    # Margin: t22 — the frames read the v2 margin_detail_pit source with
    # p17 §6.1 T+1 availability (pivot by available_date).  000001 grows
    # 0.1% per day; 600000 constant; the detail of trade_date dates[i]
    # becomes visible at dates[i+1].
    from tests.conftest import populate_v2_industry_tables, populate_v2_margin_table

    v2_path = cfg.duckdb_path.parent / "ashare_v2.duckdb"
    margin_rows = []
    for i, d in enumerate(dates[:-1]):
        margin_rows.append({
            "ts_code": "000001.SZ", "trade_date": d,
            "available_date": dates[i + 1], "rzye": 100.0 * 1.001**i,
        })
        margin_rows.append({
            "ts_code": "600000.SH", "trade_date": d,
            "available_date": dates[i + 1], "rzye": 50.0,
        })
    populate_v2_margin_table(v2_path, margin_rows)
    # D5: membership comes from the v2 PIT intervals (in_date before the
    # window, so the whole window is visible).
    populate_v2_industry_tables(
        v2_path,
        membership_rows=[
            {
                "ts_code": code,
                "index_code": "801010",
                "level": 1,
                "in_date": "20200101",
                "out_date": None,
                "is_new": "0",
                "industry_name": "农业",
                "available_date": "20200102",
            }
            for code in ("000001.SZ", "000505.SZ")
        ],
        taxonomy_rows=[
            {
                "index_code": "801010",
                "industry_name": "农业",
                "level": "L1",
                "industry_code": "SW2020-110000",
                "src": "sw2020",
            }
        ],
    )

    frames = build_capital_frames(cfg, codes, dates)
    assert set(frames) == set(EXTERNAL_FACTOR_NAMES)

    margin = frames["MARGIN_BALANCE_CHG"]
    # 20-session change of the visible balance at dates[21]: the visible
    # value is the detail of dates[20] (available at dates[21]) and the
    # base is the detail of dates[0] (available at dates[1]):
    # 1.001^20 - 1 for 000001; 0 for the constant stock.
    assert margin.loc["000001.SZ", dates[21]] == pytest.approx(1.001**20 - 1.0)
    assert margin.loc["600000.SH", dates[21]] == pytest.approx(0.0)
    # The detail of dates[20] is NOT visible before dates[21]: the change
    # at dates[20] consumes the base available at dates[0]... the first
    # visible observation is at dates[1], so the 20-session change is
    # defined from dates[21] on and NaN before it.
    assert np.isnan(margin.loc["000001.SZ", dates[20]])

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


def test_build_industry_member_frame_maps_pit_members_only(tmp_path: Path):
    # D5 (p17, 勘误 #2): the frame reads the v2 membership intervals, not
    # the legacy snapshot.  The interval/PIT-visibility/uniqueness
    # contracts live in tests/test_industry_pit.py; here the mapping-only
    # behavior is re-anchored on the PIT source (AGENTS §10.1 case 2).
    from tests.conftest import populate_v2_industry_tables

    cfg = _data_config(tmp_path)
    dates = _dates(5)
    codes = ["000001.SZ", "600000.SH", "000505.SZ"]
    populate_v2_industry_tables(
        cfg.duckdb_path.parent / "ashare_v2.duckdb",
        membership_rows=[
            {
                "ts_code": code,
                "index_code": "801010",
                "level": 1,
                "in_date": "20200101",
                "out_date": None,
                "is_new": "0",
                "industry_name": "农业",
                "available_date": "20200102",
            }
            for code in ("000001.SZ", "000505.SZ")
        ],
    )
    frame = build_industry_member_frame(cfg, codes, dates)
    assert frame.shape == (len(codes), len(dates))
    # Members carry their industry code on every date; unmapped stocks and
    # unknown codes stay NaN.
    assert (frame.loc["000001.SZ"] == "801010").all()
    assert (frame.loc["000505.SZ"] == "801010").all()
    assert frame.loc["600000.SH"].isna().all()


def test_build_industry_member_frame_degrades_without_tables(tmp_path: Path):
    cfg = _data_config(tmp_path)
    frame = build_industry_member_frame(cfg, ["000001.SZ"], _dates(5))
    assert frame.isna().all().all()


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


def test_f2_frames_raw_ratios(tmp_path: Path):
    """The two F2 frames carry the RAW ratios (the balance-to-float and
    the 5-day buy pressure); the per-date xcs-z is applied loader-side
    (p24 §2 — the dependency-correct split)."""

    from tests.conftest import populate_v2_margin_table, _V2_PRICE_DDL

    cfg = _data_config(tmp_path)
    dates = _dates(8)
    codes = ("000001.SZ", "600000.SH")
    v2_path = cfg.duckdb_path.parent / "ashare_v2.duckdb"
    margin_rows = []
    for i, d in enumerate(dates[:-1]):
        margin_rows.append({
            "ts_code": "000001.SZ", "trade_date": d,
            "available_date": dates[i + 1], "rzye": 100.0 + 10.0 * i,
            "rzmre": 50.0 + 5.0 * i,
        })
        margin_rows.append({
            "ts_code": "600000.SH", "trade_date": d,
            "available_date": dates[i + 1], "rzye": 300.0,
            "rzmre": 150.0,
        })
    populate_v2_margin_table(v2_path, margin_rows)
    # The basic rows (the circ_mv denominators) written directly: the
    # trade_date-keyed daily_basic_pit (the D4-era convention).
    with AshareDB(v2_path) as db:
        for ddl in _V2_PRICE_DDL.values():
            if "daily_basic_pit" in ddl:
                db.execute(ddl)
        db.execute("DELETE FROM daily_basic_pit")
        for code in codes:
            for d in dates:
                db.execute(
                    "INSERT INTO daily_basic_pit (ts_code, trade_date, "
                    "turnover_rate, circ_mv) VALUES (?, ?, ?, ?)",
                    [code, d, 1.0, 1000.0],
                )
    frames = build_capital_frames(cfg, list(codes), dates)
    for name in ("MARGIN_BALANCE_TO_CIRC_MV", "MARGIN_BUY_PRESSURE_5"):
        assert name in frames, name
        assert not frames[name].isna().all().all(), name
    # The raw ratios at the last visible date (dates[-2]'s detail visible
    # at dates[-1]): A: 160/1000 = 0.16; B: 300/1000 = 0.3.
    assert frames["MARGIN_BALANCE_TO_CIRC_MV"].loc["000001.SZ", dates[-1]] == pytest.approx(0.16)
    assert frames["MARGIN_BALANCE_TO_CIRC_MV"].loc["600000.SH", dates[-1]] == pytest.approx(0.3)
    # The 5-session buy sum for A: the rzmre visible at the last five
    # columns (the trade_dates dates[-6..-2], the values 50+5*i).
    a5 = sum(50.0 + 5.0 * i for i in range(len(dates) - 6, len(dates) - 1))
    assert frames["MARGIN_BUY_PRESSURE_5"].loc["000001.SZ", dates[-1]] == pytest.approx(a5 / 1000.0)
    b5 = 150.0 * 5
    assert frames["MARGIN_BUY_PRESSURE_5"].loc["600000.SH", dates[-1]] == pytest.approx(b5 / 1000.0)
