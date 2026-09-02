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


# --- P13 (C line, t14): cash-flow statement + total-assets fields -------------
#
# Contract: docs/p13_fundamental_fields_contract.md §5.1-§5.4, §7 (RED list).
# Every expected value below is derived independently from the pre-registered
# formulas (§5.3), never from the implementation's output (AGENTS §10.1).

_FAMILY5 = (
    "CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL",
)

# Synthetic cumulative (YTD) fundamentals for one stock.  TTM singles
# derived from these rows (independently computed):
#   profit singles: 2021: 6,8,10,18 | 2022: 8,10,12,20 | 2023: 10,14,15,21 | 2024Q1: 18
#   TTM profit:  Q4'21=42 Q1'22=44 Q2'22=46 Q3'22=48 Q4'22=50
#                Q1'23=52 Q2'23=56 Q3'23=59 Q4'23=60 Q1'24=68
#   cfo singles:    2021: 12,14,12,26 | 2022: 16,16,18,24 | 2023: 18,18,20,24 | 2024Q1: 24
#   TTM cfo:     Q4'21=64 Q1'22=68 Q2'22=70 Q3'22=76 Q4'22=74
#                Q1'23=76 Q2'23=78 Q3'23=80 Q4'23=80 Q1'24=86
_CUM_ROWS = [
    # (report_date, announce_date, profit_cum, cfo_cum, total_assets)
    ("20210331", "20210430", 6.0, 12.0, 820.0),
    ("20210630", "20210831", 14.0, 26.0, 830.0),
    ("20210930", "20211031", 24.0, 38.0, 840.0),
    ("20211231", "20220430", 42.0, 64.0, 850.0),
    ("20220331", "20220430", 8.0, 16.0, 900.0),
    ("20220630", "20220831", 18.0, 32.0, 920.0),
    ("20220930", "20221031", 30.0, 50.0, 950.0),
    ("20221231", "20230430", 50.0, 74.0, 1000.0),
    ("20230331", "20230430", 10.0, 18.0, 1000.0),
    ("20230630", "20230831", 24.0, 36.0, 1050.0),
    ("20230930", "20231031", 39.0, 56.0, 1100.0),
    ("20231231", "20240430", 60.0, 80.0, 1200.0),
    ("20240331", "20240430", 18.0, 24.0, 1250.0),
]


def _fundamental_columns(db: AshareDB, cfg) -> dict[str, str]:
    cols = db.query(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [cfg.fundamentals_table],
    )
    return {str(r.column_name): str(r.data_type) for r in cols.itertuples()}


def _populate_cumulative(db: AshareDB, cfg, rows=None) -> None:
    rows = rows if rows is not None else _CUM_ROWS
    _populate(
        db,
        cfg,
        [
            _row(
                "000001.SZ",
                rep,
                ann,
                profit_cum=p,
                net_operate_cash_flow=c,
                total_assets=ta,
            )
            for rep, ann, p, c, ta in rows
        ],
    )


def _first_visible(dates: list[str], announce: str) -> str:
    return next(d for d in dates if d >= announce)


def test_fundamental_schema_has_cash_flow_and_total_assets(tmp_path: Path):
    """RED-1 (§5.1): the schema carries both new DOUBLE columns and the
    migration is idempotent."""
    cfg = _data_config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        cols = _fundamental_columns(db, cfg)
        assert cols.get("net_operate_cash_flow") == "DOUBLE"
        assert cols.get("total_assets") == "DOUBLE"
        # Idempotent: re-running the migration neither fails nor duplicates.
        db.create_schema(cfg)
        assert _fundamental_columns(db, cfg) == cols


def test_fundamental_schema_migrates_existing_table_additively(tmp_path: Path):
    """RED-1 (§5.1/§4.6): a pre-P13 table survives create_schema with the
    new columns added and NULL until backfilled; existing rows keep their
    values."""
    cfg = _data_config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.execute(
            f"""
            CREATE TABLE {cfg.fundamentals_table} (
                ts_code VARCHAR,
                report_date VARCHAR,
                announce_date VARCHAR,
                dividend_announce VARCHAR,
                eps_cum DOUBLE,
                bvps DOUBLE,
                roe DOUBLE,
                roa DOUBLE,
                gross_margin DOUBLE,
                net_margin DOUBLE,
                revenue_cum DOUBLE,
                profit_cum DOUBLE,
                revenue_yoy DOUBLE,
                profit_yoy DOUBLE,
                debt_ratio DOUBLE,
                dividend_yield DOUBLE,
                PRIMARY KEY (ts_code, report_date)
            )
            """
        )
        # Pre-migration row: raw insert in the 16-column pre-P13 shape
        # (create_schema always precedes upserts in the real sync path).
        db.execute(
            f"INSERT INTO {cfg.fundamentals_table} VALUES "
            f"('000001.SZ', '20231231', '20240315', NULL, NULL, NULL, NULL, "
            f"NULL, NULL, NULL, NULL, 60.0, NULL, NULL, NULL, NULL)"
        )
        db.create_schema(cfg)  # additive migration path
        cols = _fundamental_columns(db, cfg)
        assert cols.get("net_operate_cash_flow") == "DOUBLE"
        assert cols.get("total_assets") == "DOUBLE"
        row = db.query(
            f"SELECT * FROM {cfg.fundamentals_table} WHERE ts_code='000001.SZ'"
        ).iloc[0]
        assert row["profit_cum"] == pytest.approx(60.0)
        # NULL doubles surface as NaN through the pandas bridge.
        assert pd.isna(row["net_operate_cash_flow"])
        assert pd.isna(row["total_assets"])


def test_upsert_fundamentals_new_fields_coalesce(tmp_path: Path):
    """Invariant 3 (§4.3): partial upserts merge per column; no existing
    value is wiped and merging sources never moves a visibility date."""
    cfg = _data_config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        _populate(
            db,
            cfg,
            [_row("000001.SZ", "20231231", "20240315", profit_cum=60.0)],
        )
        db.upsert_fundamentals(
            [_row("000001.SZ", "20231231", None, net_operate_cash_flow=74.0)],
            cfg,
        )
        db.upsert_fundamentals(
            [_row("000001.SZ", "20231231", None, total_assets=1000.0)], cfg
        )
        row = db.query(
            f"SELECT * FROM {cfg.fundamentals_table} WHERE ts_code='000001.SZ'"
        ).iloc[0]
        assert row["profit_cum"] == pytest.approx(60.0)
        assert row["net_operate_cash_flow"] == pytest.approx(74.0)
        assert row["total_assets"] == pytest.approx(1000.0)
        # The cash-flow-only row must not move the earnings visibility date.
        assert row["announce_date"] == "20240315"


class _StubFundamentalClient:
    """Minimal client stub for the P13 bulk cash-flow / balance-sheet path."""

    offline = False

    # Canonical earnings columns (mirrors the real client's return shape).
    _EARNINGS_COLS = [
        "ts_code", "report_date", "announce_date", "eps_cum", "bvps",
        "roe", "gross_margin", "net_margin", "revenue_cum", "profit_cum",
        "revenue_yoy", "profit_yoy",
    ]

    def __init__(self, earnings, cash_flow, balance_sheet):
        self._earnings = earnings
        self._cash_flow = cash_flow
        self._balance_sheet = balance_sheet

    def get_earnings_report(self, quarter):
        df = self._earnings.get(quarter, pd.DataFrame())
        if df.empty:
            return df
        return df.reindex(columns=self._EARNINGS_COLS)

    def get_cash_flow_report(self, quarter):
        return self._cash_flow.get(quarter, pd.DataFrame())

    def get_balance_sheet(self, quarter):
        return self._balance_sheet.get(quarter, pd.DataFrame())

    def get_financial_indicator(self, code, year):
        return pd.DataFrame()

    def get_dividend_detail(self, code):
        return pd.DataFrame()


def test_sync_cash_flow_and_balance_sheet_join_announce_master(tmp_path: Path):
    """RED-2 (§5.2): bulk cash-flow/balance-sheet rows are universe-filtered
    and joined to the earnings announce master; rows without a master match
    are dropped (fail-closed — a disclosure date is never guessed)."""
    cfg = _data_config(tmp_path)
    universe = ["000001.SZ", "000002.SZ"]
    earnings = {
        "20231231": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20231231",
                    "announce_date": "20240430",
                },
                {
                    "ts_code": "000002.SZ",
                    "report_date": "20231231",
                    "announce_date": "20240430",
                },
                {
                    "ts_code": "600000.SH",
                    "report_date": "20231231",
                    "announce_date": "20240430",
                },
            ]
        )
    }
    cash_flow = {
        "20231231": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20231231",
                    "net_operate_cash_flow": 74.0,
                },
                # Out-of-universe code: must never be stored.
                {
                    "ts_code": "600000.SH",
                    "report_date": "20231231",
                    "net_operate_cash_flow": 5.0,
                },
                # No earnings master row -> dropped, date never guessed.
                {
                    "ts_code": "000002.SZ",
                    "report_date": "20220930",
                    "net_operate_cash_flow": 3.0,
                },
            ]
        )
    }
    balance_sheet = {
        "20231231": pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": "20231231",
                    "total_assets": 1000.0,
                }
            ]
        )
    }
    client = _StubFundamentalClient(earnings, cash_flow, balance_sheet)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        sync_fundamentals(client, db, cfg, universe)
        rows = db.query(f"SELECT * FROM {cfg.fundamentals_table}")
    assert "600000.SH" not in set(rows["ts_code"])  # universe filter
    by_code = {code: sub for code, sub in rows.groupby("ts_code")}
    r1 = by_code["000001.SZ"].iloc[0]
    assert r1["net_operate_cash_flow"] == pytest.approx(74.0)
    assert r1["total_assets"] == pytest.approx(1000.0)
    assert r1["announce_date"] == "20240430"  # joined from the master
    # 000002.SZ keeps exactly its earnings row; the master-less cash-flow
    # row for 20220930 was dropped entirely.
    assert len(by_code["000002.SZ"]) == 1
    assert by_code["000002.SZ"].iloc[0]["report_date"] == "20231231"
    assert np.isnan(by_code["000002.SZ"].iloc[0]["net_operate_cash_flow"])


def test_build_pit_frames_family5_golden(tmp_path: Path):
    """RED-4 (§5.3): the four family-⑤ formulas, value by value, on
    synthetic fundamentals.  Visibility = the later announce of the
    numerator/denominator pair; announcement-season ties resolve to the
    later report."""
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2022-01-03", "2024-07-31").strftime("%Y%m%d").tolist()
    with AshareDB(cfg.duckdb_path) as db:
        _populate_cumulative(db, cfg)

    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)
    for name in _FAMILY5:
        assert name in frames, name

    def visible(announce: str) -> str:
        return _first_visible(dates, announce)

    def check(name: str, announce: str, expected: float) -> None:
        got = frames[name].loc["000001.SZ", visible(announce)]
        assert got == pytest.approx(expected), (name, announce, got)

    # Before any four-quarter TTM is visible: neutral, never fabricated.
    pre = dates[dates.index(visible("20220430")) - 1]
    for name in _FAMILY5:
        assert np.isnan(frames[name].loc["000001.SZ", pre]), name

    # 20220430: season tie (Q4'21 annual + Q1'22) -> the later report wins.
    check("CASHFLOW_QUALITY", "20220430", 68 / 44)
    check("ACCRUALS", "20220430", (44 - 68) / 900)
    # ASSET_GROWTH at Q1'22: t-4 report period is Q1'21 (present).
    check("ASSET_GROWTH", "20220430", 900 / 820 - 1)
    # EARNINGS_ACCEL needs g at t-1 with a complete TTM at t-5: NaN.
    assert np.isnan(
        frames["EARNINGS_ACCEL"].loc["000001.SZ", visible("20220430")]
    )

    check("CASHFLOW_QUALITY", "20220831", 70 / 46)
    check("ACCRUALS", "20220831", (46 - 70) / 920)
    check("ASSET_GROWTH", "20220831", 920 / 830 - 1)

    check("CASHFLOW_QUALITY", "20221031", 76 / 48)
    check("ACCRUALS", "20221031", (48 - 76) / 950)
    check("ASSET_GROWTH", "20221031", 950 / 840 - 1)

    # 20230430: season tie (Q4'22 + Q1'23) -> Q1'23 values win.
    check("CASHFLOW_QUALITY", "20230430", 76 / 52)
    check("ACCRUALS", "20230430", (52 - 76) / 1000)
    check("ASSET_GROWTH", "20230430", 1000 / 900 - 1)
    # g(Q1'23)=52/44-1, g(Q4'22)=50/42-1 -> acceleration.
    check("EARNINGS_ACCEL", "20230430", (52 / 44 - 1) - (50 / 42 - 1))

    check("CASHFLOW_QUALITY", "20230831", 78 / 56)
    check("ACCRUALS", "20230831", (56 - 78) / 1050)
    check("ASSET_GROWTH", "20230831", 1050 / 920 - 1)
    check("EARNINGS_ACCEL", "20230831", (56 / 46 - 1) - (52 / 44 - 1))

    check("CASHFLOW_QUALITY", "20231031", 80 / 59)
    check("ACCRUALS", "20231031", (59 - 80) / 1100)
    check("ASSET_GROWTH", "20231031", 1100 / 950 - 1)
    check("EARNINGS_ACCEL", "20231031", (59 / 48 - 1) - (56 / 46 - 1))

    # 20240430: season tie (Q4'23 + Q1'24) -> Q1'24 values win, and the
    # frame fills forward to the end of the range.
    check("CASHFLOW_QUALITY", "20240430", 86 / 68)
    check("ACCRUALS", "20240430", (68 - 86) / 1250)
    check("ASSET_GROWTH", "20240430", 1250 / 1000 - 1)
    check("EARNINGS_ACCEL", "20240430", (68 / 52 - 1) - (60 / 50 - 1))
    last = dates[-1]
    assert frames["CASHFLOW_QUALITY"].loc["000001.SZ", last] == pytest.approx(
        86 / 68
    )
    assert frames["ASSET_GROWTH"].loc["000001.SZ", last] == pytest.approx(0.25)


def test_build_pit_frames_family5_no_future_leak(tmp_path: Path):
    """RED-3 (§7): property scan — a family-⑤ frame may only change value
    on the first trading date at or after a statutory announcement."""
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2022-01-03", "2024-07-31").strftime("%Y%m%d").tolist()
    with AshareDB(cfg.duckdb_path) as db:
        _populate_cumulative(db, cfg)
    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)

    visible_dates = {
        _first_visible(dates, ann) for _, ann, _, _, _ in _CUM_ROWS
    }
    for name in _FAMILY5:
        series = frames[name].loc["000001.SZ"]
        changes: list[str] = []
        prev = np.nan
        for date, value in series.items():
            changed = (np.isnan(prev) != np.isnan(value)) or (
                not np.isnan(prev)
                and not np.isnan(value)
                and value != prev
            )
            if changed:
                changes.append(date)
            prev = value
        assert set(changes) <= visible_dates, (name, set(changes) - visible_dates)


def test_build_pit_frames_family5_missing_quarter_stays_nan(tmp_path: Path):
    """RED-4 NaN propagation (§5.3): a missing report breaks every TTM
    window -> the affected features stay NaN (never a shifted window)."""
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2023-01-02", "2024-07-31").strftime("%Y%m%d").tolist()
    rows = [r for r in _CUM_ROWS if r[0] != "20230630"]  # drop Q2'23
    rows = [r for r in rows if r[0] >= "20230331"]
    with AshareDB(cfg.duckdb_path) as db:
        _populate_cumulative(db, cfg, rows)
    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)
    for name in ("CASHFLOW_QUALITY", "ACCRUALS"):
        assert frames[name].loc["000001.SZ"].isna().all(), name


def test_build_pit_frames_family5_guards(tmp_path: Path):
    """RED-4 guards (§5.3): non-positive TTM profit / non-positive
    total_assets must produce NaN, never a signed garbage ratio."""
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2022-01-03", "2024-07-31").strftime("%Y%m%d").tolist()
    rows = [
        # 2022: loss year -> TTM profit negative.
        ("20220331", "20220430", -50.0, 20.0, 0.0),
        ("20220630", "20220831", -80.0, 40.0, 0.0),
        ("20220930", "20221031", -90.0, 60.0, 0.0),
        ("20221231", "20230430", -120.0, 80.0, 0.0),
        # 2023: recovery to positive TTM.
        ("20230331", "20230430", 5.0, 10.0, 1000.0),
        ("20230630", "20230831", 12.0, 18.0, 1050.0),
        ("20230930", "20231031", 21.0, 27.0, 1100.0),
        ("20231231", "20240430", 40.0, 34.0, 1200.0),
    ]
    with AshareDB(cfg.duckdb_path) as db:
        _populate_cumulative(db, cfg, rows)
    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)

    def series(name):
        return frames[name].loc["000001.SZ"]

    # total_assets <= 0 through 2022: ACCRUALS/ASSET_GROWTH stay NaN there.
    v2022 = dates.index(_first_visible(dates, "20220430"))
    v2022_end = dates.index(_first_visible(dates, "20230430")) - 1
    assert series("ACCRUALS").iloc[v2022:v2022_end].isna().all()
    assert series("ASSET_GROWTH").iloc[v2022:v2022_end].isna().all()
    # CASHFLOW_QUALITY while TTM profit <= 0 (the whole 2022 window): NaN.
    assert series("CASHFLOW_QUALITY").iloc[v2022:v2022_end].isna().all()
    # After the recovery TTM is visible the ratios become finite again.
    post = dates.index(_first_visible(dates, "20240430"))
    assert series("CASHFLOW_QUALITY").iloc[post] == pytest.approx(34 / 40)
    assert series("ACCRUALS").iloc[post] == pytest.approx((40 - 34) / 1200)
    # EARNINGS_ACCEL: g needs TTM profit at t-4 > 0; with the 2022 loss year
    # the denominator is non-positive (or incomplete) for every t -> NaN.
    assert series("EARNINGS_ACCEL").isna().all()


def test_build_pit_frames_family5_unbacked_rows_stay_nan(tmp_path: Path):
    """RED-7 (§7, in-scope portion): rows that predate the backfill (new
    fields NULL) yield all-NaN frames — fail-closed transition state, no
    neutral values fabricated (AGENTS §5.3)."""
    cfg = _data_config(tmp_path)
    codes = ["000001.SZ"]
    dates = pd.bdate_range("2023-01-02", "2024-07-31").strftime("%Y%m%d").tolist()
    with AshareDB(cfg.duckdb_path) as db:
        _populate(
            db,
            cfg,
            [_row("000001.SZ", "20231231", "20240430", profit_cum=60.0)],
        )
    close = pd.DataFrame(20.0, index=codes, columns=dates)
    frames = build_pit_frames(cfg, codes, dates, close)
    for name in _FAMILY5:
        assert name in frames, name
        assert frames[name].loc["000001.SZ"].isna().all(), name


def test_balance_sheet_endpoint_uses_stock_zcfz_em(monkeypatch):
    """P13 amendment RED (t20 window-② incident, A①): the balance-sheet
    bulk endpoint in akshare 1.18.91 is ``stock_zcfz_em`` — t14 originally
    called the nonexistent ``stock_zcfzb_em``, which the network-mocked
    RED suite could not catch (AGENTS §2.2 step-5 lesson: extend
    verification to the real integration seam).  This contract test pins
    the attribute name without touching the network."""
    import akshare

    from ashare_data.akshare_client import AkShareClient

    calls: dict[str, str] = {}

    def fake_zcfz(date: str):
        calls["stock_zcfz_em"] = date
        return pd.DataFrame([{"股票代码": "000001", "资产-总资产": 1200.0}])

    monkeypatch.setattr(akshare, "stock_zcfz_em", fake_zcfz, raising=False)
    # The wrong attribute must stay absent: calling it must fail loudly.
    monkeypatch.delattr(akshare, "stock_zcfzb_em", raising=False)

    cfg = config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    client = AkShareClient(cfg)  # online path; the patched module answers
    df = client.get_balance_sheet("20240331")
    assert calls == {"stock_zcfz_em": "20240331"}
    assert df.iloc[0]["ts_code"] == "000001.SZ"
    assert df.iloc[0]["total_assets"] == 1200.0
    # Statutory season-end anchor, not the endpoint's announcement column.
    assert df.iloc[0]["announce_date"] == "20240430"


def test_sync_cache_path_filters_out_of_scope_codes(tmp_path: Path):
    """P13 amendment RED (t20 window-② incident, A②): the cache-read path
    must apply the same universe filter as the fetch path — the pre-P2-01
    whole-market caches otherwise resurrect purged out-of-scope rows
    (root cause of the 116,613-row regression on 2026-09-02).  The chain
    reproduced here is the real one: an unfiltered earnings cache read
    writes the out-of-scope code, whose row then becomes its own announce
    master and admits the cash-flow row too."""
    cfg = _data_config(tmp_path)
    universe = ["000001.SZ"]  # 600000.SH is deliberately out of scope
    cache_dir = cfg.parquet_dir / "fundamental"
    cache_dir.mkdir(parents=True)
    # A stale whole-market earnings cache, like the pre-P2-01 artifacts
    # (full client column shape, whole-market code coverage).
    earnings_cache = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_date": "20231231",
                "announce_date": "20240430",
                "profit_cum": 60.0,
            },
            {
                "ts_code": "600000.SH",
                "report_date": "20231231",
                "announce_date": "20240430",
                "profit_cum": 500.0,
            },
        ]
    )
    earnings_cache = earnings_cache.reindex(
        columns=[
            "ts_code", "report_date", "announce_date", "eps_cum", "bvps",
            "roe", "gross_margin", "net_margin", "revenue_cum", "profit_cum",
            "revenue_yoy", "profit_yoy",
        ]
    )
    earnings_cache.to_parquet(cache_dir / "earnings_20231231.parquet", index=False)
    # A whole-market cash-flow cache for the same period.
    pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "report_date": "20231231",
                "announce_date": "20240430",
                "net_operate_cash_flow": 74.0,
            },
            {
                "ts_code": "600000.SH",
                "report_date": "20231231",
                "announce_date": "20240430",
                "net_operate_cash_flow": 99.0,
            },
        ]
    ).to_parquet(cache_dir / "cash_flow_20231231.parquet", index=False)
    client = _StubFundamentalClient({}, {}, {})
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        sync_fundamentals(client, db, cfg, universe)
        rows = db.query(f"SELECT * FROM {cfg.fundamentals_table}")
    assert set(rows["ts_code"]) == {"000001.SZ"}
    r = rows[rows["ts_code"] == "000001.SZ"].iloc[0]
    assert r["net_operate_cash_flow"] == pytest.approx(74.0)
