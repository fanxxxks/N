from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data import config as config_module
from ashare_data.akshare_client import AkShareClient
from ashare_data.db import AshareDB
from ashare_data.fundamentals import (
    FUNDAMENTAL_PIT_NAMES,
    _ffill_from_announcements,
    _quarter_ends,
    _single_periods,
    _ttm,
    build_pit_frames,
    sync_fundamentals,
)


def _data_config(tmp_path: Path) -> config_module.DataConfig:
    return config_module.DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )


def _row(ts_code, report_date, announce_date, **fields):
    row = {
        "ts_code": ts_code,
        "report_date": report_date,
        "announce_date": announce_date,
    }
    row.update(fields)
    return row


def _populate(db: AshareDB, cfg, rows: list[dict]) -> None:
    db.create_schema(cfg)
    db.upsert_fundamentals(rows, cfg)


def test_quarter_ends_range():
    assert _quarter_ends(2023, "20241231") == [
        "20230331", "20230630", "20230930", "20231231",
        "20240331", "20240630", "20240930", "20241231",
    ]
    assert _quarter_ends(2024, "20240215") == []
    assert _quarter_ends(2024, "20240331") == ["20240331"]


def test_single_periods_convert_cumulative_to_quarterly():
    rows = pd.DataFrame(
        [
            _row("000001.SZ", "20230331", "20230420", eps_cum=0.5),
            _row("000001.SZ", "20230630", "20230725", eps_cum=1.1),
            _row("000001.SZ", "20230930", "20231020", eps_cum=1.6),
            _row("000001.SZ", "20231231", "20240315", eps_cum=2.0),
            _row("000001.SZ", "20240331", "20240425", eps_cum=0.4),
        ]
    )
    singles = _single_periods(rows, "eps_cum")
    assert [v for _, v in singles] == pytest.approx([0.5, 0.6, 0.5, 0.4, 0.4])


def test_ttm_sums_four_trailing_quarters():
    rows = pd.DataFrame(
        [
            _row("000001.SZ", "20230331", "20230420", eps_cum=0.5),
            _row("000001.SZ", "20230630", "20230725", eps_cum=1.1),
            _row("000001.SZ", "20230930", "20231020", eps_cum=1.6),
            _row("000001.SZ", "20231231", "20240315", eps_cum=2.0),
            _row("000001.SZ", "20240331", "20240425", eps_cum=0.4),
        ]
    )
    _, announces, ttm = _ttm(rows, "eps_cum")
    # First three reports have no trailing four quarters: neutral.
    assert np.isnan(ttm[0]) and np.isnan(ttm[1]) and np.isnan(ttm[2])
    # Annual 2023: Q1+Q2+Q3+Q4 = 2.0; Q1 2024: 0.4 + (2.0-1.6) + (1.6-1.1)
    # + (1.1-0.5) = 1.9.
    assert ttm[3] == pytest.approx(2.0)
    assert ttm[4] == pytest.approx(1.9)
    assert announces[3] == "20240315"
    assert announces[4] == "20240425"


def test_ttm_missing_quarter_is_neutral_not_shifted():
    # Q3 is missing: the annual row must NOT borrow from a stale window.
    rows = pd.DataFrame(
        [
            _row("000001.SZ", "20230331", "20230420", eps_cum=0.5),
            _row("000001.SZ", "20230630", "20230725", eps_cum=1.1),
            _row("000001.SZ", "20231231", "20240315", eps_cum=2.0),
            _row("000001.SZ", "20240331", "20240425", eps_cum=0.4),
        ]
    )
    _, _, ttm = _ttm(rows, "eps_cum")
    assert np.isnan(ttm[2])  # annual: only three singles available
    assert np.isnan(ttm[3])  # Q1'24: window spans a year gap -> neutral


def test_ffill_from_announcements_no_lookahead():
    dates = [f"202401{i:02d}" for i in range(1, 11)]
    values = _ffill_from_announcements(
        dates, ["20240105", "20240108"], [10.0, 12.0]
    )
    assert np.isnan(values[0:4]).all()  # before the first announcement
    assert values[4] == 10.0 and values[5] == 10.0 and values[6] == 10.0
    assert values[7] == 12.0 and values[9] == 12.0


def test_build_pit_frames_aligns_to_announcement_dates(tmp_path: Path):
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ", "600000.SH"]
    dates = pd.bdate_range("2024-01-01", periods=60).strftime("%Y%m%d").tolist()
    rows = [
        _row("000001.SZ", "20230930", "20240115", roe=9.0, bvps=8.0),
        _row("000001.SZ", "20231231", "20240220", roe=11.0, bvps=10.0),
    ]
    with AshareDB(cfg.duckdb_path) as db:
        _populate(db, cfg, rows)

    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)
    assert set(frames) == set(FUNDAMENTAL_PIT_NAMES)

    roe = frames["ROE"].loc["000001.SZ"]
    d1, d2 = dates.index("20240115"), dates.index("20240220")
    assert np.isnan(roe.iloc[:d1]).all()
    assert (roe.iloc[d1:d2] == 9.0).all()
    assert (roe.iloc[d2:] == 11.0).all()
    # The stock without rows stays neutral everywhere.
    assert frames["ROE"].loc["600000.SH"].isna().all()

    # PB = close / bvps, aligned to the same announcement dates.
    pb = frames["PB"].loc["000001.SZ"]
    assert pb.iloc[d1] == pytest.approx(20.0 / 8.0)
    assert pb.iloc[d2] == pytest.approx(20.0 / 10.0)
    assert np.isnan(pb.iloc[:d1]).all()


def test_build_pit_frames_pe_ps_guards(tmp_path: Path):
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2024-01-01", "2024-09-30").strftime("%Y%m%d").tolist()
    rows = [
        _row("000001.SZ", "20230331", "20240401", eps_cum=0.5, revenue_cum=100.0, profit_cum=10.0),
        _row("000001.SZ", "20230630", "20240501", eps_cum=1.1, revenue_cum=220.0, profit_cum=22.0),
        _row("000001.SZ", "20230930", "20240601", eps_cum=1.6, revenue_cum=330.0, profit_cum=33.0),
        _row("000001.SZ", "20231231", "20240701", eps_cum=2.0, revenue_cum=440.0, profit_cum=44.0),
        # Negative TTM earnings: PE/PS must go neutral, not negative.
        _row("000001.SZ", "20240331", "20240801", eps_cum=-3.0, revenue_cum=10.0, profit_cum=-5.0),
    ]
    with AshareDB(cfg.duckdb_path) as db:
        _populate(db, cfg, rows)

    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)
    pe = frames["PE_TTM"].loc["000001.SZ"]
    d = dates.index("20240701")
    # TTM EPS at the annual report = 2.0 -> PE 10.
    assert pe.iloc[d] == pytest.approx(10.0)
    # After the negative Q1'24 report the trailing window is contaminated:
    # the four-quarter span check fails (Q1'24 vs Q2'23) -> neutral.
    assert np.isnan(pe.iloc[-1])
    # PS = PE x trailing profit/revenue margin (10%): 20 / 2.0 * 0.1 = 1.0.
    ps = frames["PS_TTM"].loc["000001.SZ"]
    assert ps.iloc[d] == pytest.approx(1.0)


def test_dividend_yield_aligns_to_ex_date(tmp_path: Path):
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2024-01-01", periods=30).strftime("%Y%m%d").tolist()
    rows = [
        _row("000001.SZ", "20231231", None, dividend_announce="20240110", dividend_yield=0.035),
    ]
    with AshareDB(cfg.duckdb_path) as db:
        _populate(db, cfg, rows)
    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)
    dy = frames["DIVIDEND_YIELD"].loc["000001.SZ"]
    d = dates.index("20240110")
    assert np.isnan(dy.iloc[:d]).all()
    assert (dy.iloc[d:] == 0.035).all()


def test_build_pit_frames_without_table_degrades(tmp_path: Path):
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2024-01-01", periods=5).strftime("%Y%m%d").tolist()
    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)  # no DB file at all
    assert set(frames) == set(FUNDAMENTAL_PIT_NAMES)
    assert frames["ROE"].isna().all().all()


def test_upsert_fundamentals_coalesces_nulls(tmp_path: Path):
    cfg = _data_config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        _populate(
            db,
            cfg,
            [_row("000001.SZ", "20231231", "20240315", eps_cum=2.0, roe=11.0)],
        )
        # A dividend-only row for the same period must not wipe the earnings
        # and carries its own ex-dividend date.
        db.upsert_fundamentals(
            [
                _row(
                    "000001.SZ",
                    "20231231",
                    None,
                    dividend_announce="20240110",
                    dividend_yield=0.035,
                )
            ],
            cfg,
        )
        out = db.query(
            f"SELECT * FROM {cfg.fundamentals_table} WHERE ts_code='000001.SZ'"
        )
        row = out.iloc[0]
        assert row["eps_cum"] == pytest.approx(2.0)
        assert row["roe"] == pytest.approx(11.0)
        assert row["dividend_yield"] == pytest.approx(0.035)
        # Each source keeps its own disclosure date: merging can never move
        # any field's point-in-time visibility.
        assert row["announce_date"] == "20240315"
        assert row["dividend_announce"] == "20240110"


def test_sync_fundamentals_offline_skips(tmp_path: Path):
    cfg = _data_config(tmp_path)
    client = AkShareClient(cfg, offline=True, fixture_dir=Path("tests/fixtures_missing"))
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        result = sync_fundamentals(client, db, cfg, ["000001.SZ"])
    assert result == {
        "fundamental_quarters": 0,
        "fundamental_rows": 0,
        "fundamental_supplements": 0,
        "fundamental_failures": 0,
    }


def test_akshare_client_offline_earnings_normalization(tmp_path: Path):
    cfg = config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    fx = tmp_path / "fx"
    fx.mkdir()
    # The endpoint's own 最新公告日期 is deliberately ignored: it carries
    # restatement dates, so the client sets the disclosure-season end.
    (fx / "earnings_20240331.json").write_text(
        '[{"股票代码": "000001", "每股收益": 0.5, "营业总收入-营业总收入": 100.0,'
        ' "营业总收入-同比增长": 12.5, "净利润-净利润": 20.0, "净利润-同比增长": 8.0,'
        ' "每股净资产": 8.0, "净资产收益率": 6.25, "销售毛利率": 30.0,'
        ' "最新公告日期": "2024-04-20"},'
        ' {"股票代码": "900901", "每股收益": 0.5, "营业总收入-营业总收入": 100.0,'
        ' "净利润-净利润": 20.0, "最新公告日期": "2024-04-20"}]',
        encoding="utf-8",
    )
    (fx / "earnings_20231231.json").write_text(
        '[{"股票代码": "000001", "每股收益": 2.0, "营业总收入-营业总收入": 400.0,'
        ' "净利润-净利润": 40.0}]',
        encoding="utf-8",
    )
    client = AkShareClient(cfg, offline=True, fixture_dir=fx)
    df = client.get_earnings_report("20240331")
    assert df["ts_code"].tolist() == ["000001.SZ"]  # B-share dropped
    assert df.iloc[0]["report_date"] == "20240331"
    # Q1 season end, not the restatement date from the endpoint column.
    assert df.iloc[0]["announce_date"] == "20240430"
    assert df.iloc[0]["net_margin"] == pytest.approx(20.0)
    assert df.iloc[0]["revenue_yoy"] == pytest.approx(12.5)
    annual = client.get_earnings_report("20231231")
    # The annual season ends on April 30 of the NEXT year.
    assert annual.iloc[0]["announce_date"] == "20240430"


def test_akshare_client_offline_indicator_and_dividend(tmp_path: Path):
    cfg = config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "financial_000001.SZ.json").write_text(
        '[{"日期": "2023-12-31", "总资产净利润率(%)": 1.1, "资产负债率(%)": 90.0}]',
        encoding="utf-8",
    )
    (fx / "dividend_000001.SZ.json").write_text(
        '[{"报告期": "2023-12-31", "现金分红-股息率": 3.5, "除权除息日": "2024-05-20"}]',
        encoding="utf-8",
    )
    client = AkShareClient(cfg, offline=True, fixture_dir=fx)
    ind = client.get_financial_indicator("000001.SZ", 2023)
    assert ind.iloc[0]["report_date"] == "20231231"
    assert ind.iloc[0]["roa"] == pytest.approx(1.1)
    assert ind.iloc[0]["debt_ratio"] == pytest.approx(90.0)
    div = client.get_dividend_detail("000001.SZ")
    assert div.iloc[0]["announce_date"] == "20240520"
    assert div.iloc[0]["dividend_yield"] == pytest.approx(0.035)
