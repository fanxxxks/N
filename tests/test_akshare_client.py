from __future__ import annotations

from pathlib import Path

import pytest

from ashare_data import config as data_config_module
from ashare_data.akshare_client import (
    AkShareClient,
    AkShareUnavailable,
    _call_with_timeout,
    _retry,
    _symbol_from_ts_code,
)


def test_symbol_from_ts_code():
    assert _symbol_from_ts_code("000001.SZ") == "000001"
    assert _symbol_from_ts_code("600000.SH") == "600000"


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
