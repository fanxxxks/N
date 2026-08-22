from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ashare_data import config as data_config_module
from ashare_data.akshare_client import (
    AkShareClient,
    AkShareUnavailable,
    _call_with_timeout,
    _retry,
    _symbol_from_ts_code,
    normalize_listing_date,
    normalize_stock_metadata,
)


def test_symbol_from_ts_code():
    assert _symbol_from_ts_code("000001.SZ") == "000001"
    assert _symbol_from_ts_code("600000.SH") == "600000"


def test_stock_metadata_normalizes_listing_dates_without_fabrication():
    frame = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "A",
                "list_date": "1991-04-03",
                "is_st": False,
            },
            {
                "ts_code": "600000.SH",
                "name": "ST B",
                "list_date": "invalid",
            },
        ]
    )
    normalized = normalize_stock_metadata(frame)

    assert normalized.iloc[0]["list_date"] == "19910403"
    assert pd.isna(normalized.iloc[1]["list_date"])
    assert normalized["is_st"].tolist() == [False, True]
    assert normalize_listing_date(None) is None


def test_online_stock_list_merges_exchange_listing_dates(
    monkeypatch: pytest.MonkeyPatch,
):
    import akshare as ak

    monkeypatch.setattr(
        ak,
        "stock_info_a_code_name",
        lambda: pd.DataFrame(
            {
                "code": ["000001", "600000", "688001", "830001"],
                "name": ["深市", "沪市", "科创", "北证"],
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "stock_info_sh_name_code",
        lambda symbol: pd.DataFrame(
            {
                "证券代码": ["600000" if symbol == "主板A股" else "688001"],
                "证券简称": ["沪市" if symbol == "主板A股" else "科创"],
                "上市日期": ["1999-11-10" if symbol == "主板A股" else "2020-01-01"],
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "stock_info_sz_name_code",
        lambda symbol: pd.DataFrame(
            {
                "A股代码": ["000001"],
                "A股简称": ["深市"],
                "A股上市日期": ["1991-04-03"],
                "所属行业": ["金融"],
            }
        ),
    )
    monkeypatch.setattr(
        ak,
        "stock_info_bj_name_code",
        lambda: pd.DataFrame(
            {
                "证券代码": ["830001"],
                "证券简称": ["北证"],
                "上市日期": ["2022-01-04"],
                "所属行业": ["制造"],
            }
        ),
    )
    client = AkShareClient(data_config_module.DataConfig(), offline=False)
    monkeypatch.setattr(client, "_fetch", lambda fn, **_kwargs: fn())

    stocks = client.get_stock_list().set_index("ts_code")
    assert stocks.loc["000001.SZ", "list_date"] == "19910403"
    assert stocks.loc["600000.SH", "list_date"] == "19991110"
    assert stocks.loc["688001.SH", "list_date"] == "20200101"
    assert stocks.loc["830001.BJ", "list_date"] == "20220104"


def test_call_with_timeout_bounds_a_hung_call():
    import threading
    import time

    blocker = threading.Event()
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _call_with_timeout(lambda: blocker.wait(60), timeout=0.2)
    assert time.monotonic() - started < 10


def test_call_with_timeout_returns_value_and_forwards_errors():
    assert _call_with_timeout(lambda: 42, timeout=1.0) == 42
    with pytest.raises(RuntimeError):
        _call_with_timeout(
            lambda: (_ for _ in ()).throw(RuntimeError("boom")), timeout=1.0
        )


def test_retry_raises_after_exhaustion(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ashare_data.akshare_client.time.sleep", lambda _: None)
    with pytest.raises(AkShareUnavailable):
        _retry(lambda: (_ for _ in ()).throw(RuntimeError("boom")), retries=2, delay=0)


def test_offline_calendar_fallback():
    cfg = data_config_module.DataConfig(start_date="2024-01-01", end_date="2024-01-05")
    client = AkShareClient(cfg, offline=True, fixture_dir=Path("tests/fixtures_missing"))
    dates = client.get_trade_calendar()
    assert dates
    assert all(len(d) == 8 for d in dates)


def test_online_calendar_retains_history_needed_for_listing_age(
    monkeypatch: pytest.MonkeyPatch,
):
    import akshare as ak

    monkeypatch.setattr(
        ak,
        "tool_trade_date_hist_sina",
        lambda: pd.DataFrame(
            {"trade_date": ["2020-01-02", "2024-01-02", "2025-01-02"]}
        ),
    )
    config = data_config_module.DataConfig(
        start_date="2024-01-01", end_date="2024-12-31"
    )
    client = AkShareClient(config, offline=False)
    monkeypatch.setattr(client, "_fetch", lambda fn, **_kwargs: fn())

    assert client.get_trade_calendar() == ["20200102", "20240102"]


def test_offline_fixtures(tmp_path: Path):
    cfg = data_config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "calendar.json").write_text(
        '[{"trade_date": 20240102}, {"trade_date": 20240103}]',
        encoding="utf-8",
    )
    (fixture_dir / "stocks.json").write_text(
        '[{"ts_code": "000001.SZ", "name": "A", "industry": null, "list_date": "20200101", "is_st": false}]',
        encoding="utf-8",
    )
    (fixture_dir / "constituents_000300.SH.json").write_text(
        '[{"ts_code": "000001.SZ"}, {"ts_code": "600000.SH"}]',
        encoding="utf-8",
    )
    (fixture_dir / "daily_000001.SZ.json").write_text(
        '[{"ts_code": "000001.SZ", "trade_date": "20240102", "open": 1.0}]',
        encoding="utf-8",
    )

    client = AkShareClient(cfg, offline=True, fixture_dir=fixture_dir)
    assert client.get_trade_calendar() == ["20240102", "20240103"]
    assert client.get_stock_list().iloc[0]["ts_code"] == "000001.SZ"
    assert client.get_constituents("000300.SH") == ["000001.SZ", "600000.SH"]
    daily = client.get_daily_bar("000001.SZ")
    assert len(daily) == 1


def test_offline_defaults_for_missing_fixtures():
    cfg = data_config_module.DataConfig(start_date="2024-01-01", end_date="2024-01-05")
    client = AkShareClient(cfg, offline=True, fixture_dir=Path("tests/fixtures_missing"))
    stocks = client.get_stock_list()
    assert set(stocks["ts_code"]) == {"000001.SZ", "600000.SH"}
    assert client.get_constituents("000300.SH") == ["000001.SZ", "600000.SH"]
    assert client.get_daily_bar("000001.SZ").empty


def test_constituents_filters_index_and_b_share_codes(tmp_path: Path):
    from ashare_data.processor import is_valid_a_share_code

    cfg = data_config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "constituents_000300.SH.json").write_text(
        '[{"ts_code": "000300.SZ"}, {"ts_code": "000001.SZ"},'
        ' {"ts_code": "900901.SH"}, {"ts_code": "600000.SH"}]',
        encoding="utf-8",
    )
    client = AkShareClient(cfg, offline=True, fixture_dir=fx)
    codes = client.get_constituents("000300.SH")
    assert codes == ["000001.SZ", "600000.SH"]
    assert all(is_valid_a_share_code(c) for c in codes)


def test_stock_list_filters_invalid_codes(tmp_path: Path):
    cfg = data_config_module.DataConfig(start_date="2024-01-01", end_date="2024-12-31")
    fx = tmp_path / "fx"
    fx.mkdir()
    (fx / "stocks.json").write_text(
        '[{"ts_code": "000001.SZ", "name": "A", "industry": null, "list_date": null, "is_st": false},'
        ' {"ts_code": "900901.SH", "name": "B股", "industry": null, "list_date": null, "is_st": false}]',
        encoding="utf-8",
    )
    client = AkShareClient(cfg, offline=True, fixture_dir=fx)
    stocks = client.get_stock_list()
    assert stocks["ts_code"].tolist() == ["000001.SZ"]
