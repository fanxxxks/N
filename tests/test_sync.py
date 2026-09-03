from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

import ashare_data.sync as sync
from ashare_data.config import DataConfig
from ashare_data.db import AshareDB


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )


def test_parquet_cache_path(tmp_path: Path):
    cfg = _config(tmp_path)
    assert sync._parquet_cache_path(cfg, "000001.SZ") == tmp_path / "parquet" / "daily" / "000001.SZ.parquet"


def test_parquet_cache_roundtrip(tmp_path: Path):
    cfg = _config(tmp_path)
    df = pd.DataFrame({"trade_date": ["20240101"], "close": [10.0]})
    sync._write_cached_bars(cfg, "000001.SZ", df)
    out = sync._read_cached_bars(cfg, "000001.SZ")
    assert out.equals(df)


def test_read_cached_bars_missing_and_corrupt(tmp_path: Path):
    cfg = _config(tmp_path)
    assert sync._read_cached_bars(cfg, "missing").empty
    path = sync._parquet_cache_path(cfg, "bad")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a parquet", encoding="utf-8")
    assert sync._read_cached_bars(cfg, "bad").empty


def test_calendar_stock_and_bar_rows():
    assert sync._calendar_rows(["20240101"]) == [{"trade_date": "20240101", "is_open": True}]
    stocks = pd.DataFrame(
        [{"ts_code": "000001.SZ", "name": "ST 风险", "industry": None, "list_date": "2020-01-02"}]
    )
    rows = sync._stock_rows(stocks)
    assert rows[0]["is_st"] is True
    assert rows[0]["list_date"] == "20200102"

    bars = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "trade_date": "20240101",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1000.0,
            }
        ]
    )
    bar_rows = sync._bar_rows(bars)
    assert len(bar_rows) == 1
    assert bar_rows[0]["pre_close"] == 10.0


def test_sync_all_offline(tmp_path: Path):
    cfg_path = tmp_path / "ashare_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "duckdb_path": str(tmp_path / "data" / "ashare.duckdb"),
                "parquet_dir": str(tmp_path / "data" / "parquet"),
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
        ),
        encoding="utf-8",
    )
    result = sync.sync_all(cfg_path, offline=True, limit=3)
    assert result["calendar_days"] == 4
    assert result["stocks"] == 2
    # 300001.SZ appears in the constituents fixture but not in the stock
    # list fixture: the authoritative stock-list intersection removes it, so
    # it is neither synced nor attempted.
    assert result["universe"] == 2
    assert result["constituent_snapshot_symbols"] >= 2
    assert result["pit_constituent_rows_written"] == 0
    assert result["daily_rows"] > 0
    assert result["failures"] == []

    with AshareDB(tmp_path / "data" / "ashare.duckdb") as db:
        assert db.query("SELECT COUNT(*) AS n FROM daily_bar").iloc[0]["n"] > 0
        codes = set(db.query("SELECT DISTINCT ts_code FROM daily_bar")["ts_code"])
        assert codes == {"000001.SZ", "600000.SH"}
        assert db.query("SELECT COUNT(*) AS n FROM constituents").iloc[0]["n"] == 0


def test_augment_sync_universe_includes_pit_and_cached_codes(tmp_path: Path):
    """H3: historical PIT members and cached-bar codes join the sync
    universe even when absent from the current snapshot/stock list, while
    invalid codes are still filtered out."""
    cfg = _config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_constituents(
            [
                # Historical (delisted) member: absent from today's snapshot.
                {"index_code": "000300.SH", "ts_code": "600999.SH", "in_date": "20150101", "out_date": "20201231"},
                # Invalid code shares the 000xxx index space: filtered.
                {"index_code": "000300.SH", "ts_code": "000300.SZ", "in_date": "20150101", "out_date": "99991231"},
            ],
            cfg,
        )
    daily_dir = cfg.parquet_dir / "daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "601888.SH.parquet").write_bytes(b"x")  # cached, delisted
    (daily_dir / "not_a_code.parquet").write_bytes(b"x")
    with AshareDB(cfg.duckdb_path) as db:
        out = sync._augment_sync_universe(db, cfg, {"000001.SZ"})
    assert "000001.SZ" in out
    assert "600999.SH" in out  # PIT historical member survives
    assert "601888.SH" in out  # cached-bar code survives
    assert "000300.SZ" not in out
    assert "not_a_code" not in out


def test_sync_all_includes_pit_members_and_cached_codes(tmp_path: Path):
    """End-to-end: a routine sync keeps syncing codes that only exist as
    PIT members or cached bars (delisted stocks), instead of dropping
    them via the current snapshot/stock-list intersection."""
    cfg_path = tmp_path / "ashare_config.yaml"
    data_dir = tmp_path / "data"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data_dir),
                "duckdb_path": str(data_dir / "ashare.duckdb"),
                "parquet_dir": str(data_dir / "parquet"),
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
        ),
        encoding="utf-8",
    )
    cfg = _config(data_dir)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "600999.SH", "in_date": "20150101", "out_date": "20201231"},
            ],
            cfg,
        )
    (data_dir / "parquet" / "daily").mkdir(parents=True)
    (data_dir / "parquet" / "daily" / "601888.SH.parquet").write_bytes(b"x")
    result = sync.sync_all(cfg_path, offline=True)
    # The two snapshot-validated fixture codes plus the PIT member and the
    # cached code: both delisted-style codes stay in the sync universe.
    assert result["universe"] == 4


def test_purge_stale_daily_rows(tmp_path: Path):
    cfg = _config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_daily(
            [
                {
                    "ts_code": c,
                    "trade_date": "20240102",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "pre_close": 9.9,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
                for c in ("000001.SZ", "600000.SH", "000300.SZ")
            ],
            cfg,
        )
        removed = sync._purge_stale_daily_rows(
            db, cfg, ["000001.SZ"], failures=["600000.SH"]
        )
        assert removed == 1  # only the index code row is removed
        remaining = db.query("SELECT ts_code FROM daily_bar")["ts_code"].tolist()
        assert set(remaining) == {"000001.SZ", "600000.SH"}


def _seed_backfill_db(tmp_path: Path) -> DataConfig:
    """Fixture DB: one live member with bars, one historical member with a
    zero-bar interval overlapping the horizon (the delisted signature)."""
    cfg = _config(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in ("20240102", "20240103")],
            cfg,
        )
        db.upsert_stocks(
            [
                {"ts_code": "000001.SZ", "name": "A", "industry": None, "list_date": "20200101", "is_st": False},
                {"ts_code": "600999.SH", "name": "D", "industry": None, "list_date": "20200101", "is_st": False},
            ],
            cfg,
        )
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20240101", "out_date": "99991231"},
                {"index_code": "000300.SH", "ts_code": "600999.SH", "in_date": "20240101", "out_date": "20241231"},
            ],
            cfg,
        )
        db.upsert_daily(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": d,
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "pre_close": 9.9,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
                for d in ("20240102", "20240103")
            ],
            cfg,
        )
    return cfg


class _FakeClient:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.calls: list[str] = []

    def get_daily_bar(self, ts_code, start_date, end_date):
        self.calls.append(ts_code)
        return self.frames.get(ts_code, pd.DataFrame())


def test_backfill_member_bars_fetches_zero_bar_members(tmp_path: Path):
    """T0-04: a zero-bar historical member is audited, fetched and
    upserted (daily table + parquet cache), and the re-audit is clean."""

    cfg = _seed_backfill_db(tmp_path)
    bars = pd.DataFrame(
        [
            {
                "ts_code": "600999.SH",
                "trade_date": d,
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.05,
                "volume": 10.0,
                "amount": 100.0,
            }
            for d in ("20240102", "20240103")
        ]
    )
    client = _FakeClient({"600999.SH": bars})
    result = sync.backfill_member_bars(cfg, client=client)
    assert client.calls == ["600999.SH"]  # only the zero-bar member
    assert result["fetched_codes"] == 1
    assert result["daily_rows"] == 2
    assert result["failures"] == []
    assert result["remaining_zero_bar_intervals"] == 0
    # The bars landed in the daily table and the parquet cache.
    with AshareDB(cfg.duckdb_path) as db:
        rows = db.query(
            "SELECT COUNT(*) AS n FROM daily_bar WHERE ts_code='600999.SH'"
        ).iloc[0]["n"]
        assert rows == 2
    cached = sync._read_cached_bars(cfg, "600999.SH")
    assert len(cached) == 2
    # A re-run audits the now-covered member and touches nothing.
    result2 = sync.backfill_member_bars(cfg, client=client)
    assert result2["audited_zero_bar_codes"] == 0
    assert result2["fetched_codes"] == 0
    assert result2["remaining_zero_bar_intervals"] == 0


def test_backfill_member_bars_reports_unfetchable_codes(tmp_path: Path):
    cfg = _seed_backfill_db(tmp_path)
    client = _FakeClient({})  # the delisted member is unfetchable
    result = sync.backfill_member_bars(cfg, client=client)
    assert result["audited_zero_bar_codes"] == 1
    assert result["fetched_codes"] == 0
    assert result["failures"] == ["600999.SH"]
    assert result["remaining_zero_bar_intervals"] == 1


class _SuspendedSpanClient(_FakeClient):
    """A client whose main fetch resumes after the interval (long
    suspension), plus a Baostock supplement that carries the official
    suspension rows (flat price, zero volume)."""

    def _get_daily_bar_baostock(self, ts_code, start_date, end_date):
        self.calls.append(f"bs:{ts_code}")
        return pd.DataFrame(
            [
                {
                    "ts_code": ts_code,
                    "trade_date": "20240102",
                    "open": 4.64,
                    "high": 4.64,
                    "low": 4.64,
                    "close": 4.64,
                    "pre_close": 4.64,
                    "volume": 0.0,
                    "amount": 0.0,
                    "turnover_rate": None,
                }
            ]
        )


def test_backfill_member_bars_supplements_suspended_span(tmp_path: Path):
    """T0-04: when the main fetch resumes only after the interval (the
    member was suspended through the whole audited span), the Baostock
    supplement supplies the official suspension rows so the interval is
    no longer zero-bar (presence with blocked trading, volume zero)."""

    cfg = _seed_backfill_db(tmp_path)
    # Main fetch returns data only after the interval (like 中银绒业's
    # post-suspension resume): the audited span stays bare without the
    # Baostock supplement.
    late = pd.DataFrame(
        [
            {
                "ts_code": "600999.SH",
                "trade_date": "20240301",
                "open": 5.0,
                "high": 5.1,
                "low": 4.9,
                "close": 5.05,
                "volume": 100.0,
                "amount": 500.0,
            }
        ]
    )
    client = _SuspendedSpanClient({"600999.SH": late})
    result = sync.backfill_member_bars(cfg, client=client)
    assert "600999.SH" in client.calls
    assert "bs:600999.SH" in client.calls  # supplement was invoked
    assert result["fetched_codes"] == 1
    assert result["failures"] == []
    assert result["remaining_zero_bar_intervals"] == 0
    with AshareDB(cfg.duckdb_path) as db:
        rows = db.query(
            "SELECT trade_date, volume FROM daily_bar "
            "WHERE ts_code='600999.SH' ORDER BY trade_date"
        )
        assert len(rows) == 2
        # The suspension row has zero volume: present but untradeable.
        assert rows.iloc[0]["volume"] == 0.0


# ---------------------------------------------------------------------------
# IP-11 (03-F-07 / 03-F-08): sync log identity header, fixed summary key set,
# and the loud process-exit guard.  Everything here is fixture-based — no
# real (stateful) sync runs in tests.
# ---------------------------------------------------------------------------


def _offline_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "ashare_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(tmp_path / "data"),
                "duckdb_path": str(tmp_path / "data" / "ashare.duckdb"),
                "parquet_dir": str(tmp_path / "data" / "parquet"),
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def _log_messages(text: str) -> list[str]:
    """Message part of each formatted memory-sink line (C2 format)."""
    return [
        line.split(" | ", 3)[3]
        for line in text.splitlines()
        if line.count(" | ") >= 3
    ]


def test_sync_summary_fixed_key_set(tmp_path: Path):
    """IP-11: the tail summary dict carries a fixed key set, including the
    cross-entry ``duplicates``/``suppressed`` counters (0 while the sync
    path produces no such events) so runs stay directly comparable."""
    result = sync.sync_all(_offline_config(tmp_path), offline=True, limit=3)
    assert set(result) == {
        "calendar_days",
        "stocks",
        "universe",
        "constituent_snapshot_symbols",
        "pit_constituent_rows_written",
        "daily_rows",
        "failures",
        "purged_rows",
        "purged_parquet",
        "dataset_id",
        "duplicates",
        "suppressed",
        "fundamental_quarters",
        "fundamental_rows",
        "fundamental_supplements",
        "fundamental_failures",
        "margin_rows",
        "margin_dates",
        "industries",
        "industry_rows",
        "capital_failures",
    }
    assert result["duplicates"] == 0
    assert result["suppressed"] == 0


def test_sync_all_emits_identity_header_first(tmp_path: Path):
    """IP-11: the first content line after logging setup carries the
    identity quadruple (run_id / git commit / config sha256 / versions)."""
    from ashare_logging import get_log_text, setup_run_logging

    setup_run_logging(
        log_dir=tmp_path / "logs", run_name="syncidentity", reset=True
    )
    sync.sync_all(_offline_config(tmp_path), offline=True, limit=3)
    messages = _log_messages(get_log_text())
    header_idx = next(
        i for i, m in enumerate(messages) if m.startswith("run identity: ")
    )
    # The header is the first content line (only the pipeline's own setup
    # line precedes it).
    assert header_idx == 1, messages[: header_idx + 1]
    header = messages[header_idx]
    match = re.fullmatch(
        r"run identity: run_id=[0-9a-f]{32} "
        r"git_commit=([0-9a-f]{40}|unknown) "
        r"config_sha256=[0-9a-f]{64} versions=(\{.*\})",
        header,
    )
    assert match, header
    versions = json.loads(match.group(2))
    assert set(versions) == {"akshare", "duckdb", "sync_tool"}


def test_exit_guard_noop_without_survivors():
    # Main thread only: the guard is a silent no-op.
    sync._guard_process_exit(timeout=0.1)


def test_exit_guard_joins_threads_that_finish_in_time():
    done = threading.Event()

    def worker():
        time.sleep(0.05)
        done.set()

    thread = threading.Thread(target=worker, name="guard-finishes")
    thread.start()
    sync._guard_process_exit(timeout=10.0)
    assert done.is_set()
    assert not thread.is_alive()


def test_exit_guard_forces_loud_exit_after_timeout(monkeypatch, tmp_path):
    """F-08: a surviving non-daemon thread must never hang the interpreter
    silently — its stack is logged at ERROR and the exit is forced loudly
    (never a silent ``os._exit`` swallow)."""
    from ashare_logging import get_log_text, setup_run_logging

    release = threading.Event()
    started = threading.Event()

    def hang():
        started.set()
        release.wait(30)

    thread = threading.Thread(target=hang, name="guard-hangs")
    thread.start()
    assert started.wait(5)
    setup_run_logging(log_dir=tmp_path, run_name="guard", reset=True)
    forced: list[int] = []
    monkeypatch.setattr(sync.os, "_exit", lambda code=0: forced.append(code))
    try:
        sync._guard_process_exit(timeout=0.2)
    finally:
        release.set()
        thread.join(5)
    assert forced == [3]
    text = get_log_text()
    assert "guard-hangs" in text  # survivor named in the ERROR report
    assert "forcing process exit" in text
    assert "in hang" in text  # the survivor's stack is the evidence
