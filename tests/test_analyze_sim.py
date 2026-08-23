"""Tests for scripts/analyze_sim.py (sim trade analysis).

The report must derive every number from inputs: no hardcoded initial
capital and no hardcoded final cash.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_sim import (
    StateSnapshot,
    TradeSummary,
    load_state_snapshot,
    load_trades,
    main,
    render_report,
    summarize_trades,
)


def make_trade(
    side="buy",
    amount=10000.0,
    commission=5.0,
    stamp=0.0,
    transfer=0.1,
    slippage=5.0,
) -> dict:
    return {
        "side": side,
        "amount": amount,
        "commission": commission,
        "stamp_tax": stamp,
        "transfer_fee": transfer,
        "slippage": slippage,
    }


def write_trades(trades_dir: Path, day: str, rows: list) -> Path:
    trades_dir.mkdir(parents=True, exist_ok=True)
    path = trades_dir / f"{day}.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_load_trades_reads_only_non_empty_files(tmp_path: Path):
    write_trades(tmp_path, "20240102", [make_trade()])
    write_trades(tmp_path, "20240103", [])
    (tmp_path / "20240104.json").write_text("not json", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("[]", encoding="utf-8")
    days = load_trades(tmp_path)
    assert days == [("20240102", [make_trade()])]


def test_summarize_trades_aggregates():
    days = [
        (
            "20240102",
            [
                make_trade("buy", amount=10000.0, commission=5.0),
                make_trade("sell", amount=8000.0, commission=4.0, stamp=4.0),
                make_trade("buy", amount=2000.0, commission=5.0),
            ],
        ),
    ]
    summary = summarize_trades(days)
    assert summary.days == 1
    assert summary.trades == 3
    assert summary.buy_amount == 12000.0
    assert summary.sell_amount == 8000.0
    assert summary.total_amount == 20000.0
    assert summary.commission == 14.0
    assert summary.stamp_tax == 4.0
    assert summary.transfer_fee == pytest.approx(0.3)
    assert summary.slippage == 15.0
    assert summary.total_fees == pytest.approx(33.3)
    assert summary.fees_by_year == {"2024": pytest.approx(33.3)}


def test_summarize_trades_treats_missing_fee_fields_as_zero():
    summary = summarize_trades([("20240102", [{"side": "buy", "amount": 100.0}])])
    assert summary.total_fees == 0.0
    assert summary.buy_amount == 100.0


def test_load_state_snapshot_reads_values(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "initial_capital": 100000.0,
                "cash": 5946.37,
                "equity_history": [
                    {"trade_date": "20240102", "equity": 99990.0},
                    {"trade_date": "20240103", "equity": 5946.37},
                ],
            }
        ),
        encoding="utf-8",
    )
    snapshot = load_state_snapshot(state_path, fallback_initial_capital=1.0)
    assert snapshot == StateSnapshot(100000.0, 5946.37, 5946.37)


def test_load_state_snapshot_missing_or_malformed_returns_none(tmp_path: Path):
    assert load_state_snapshot(tmp_path / "absent.json", 1.0) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert load_state_snapshot(bad, 1.0) is None


def _summary(buy=12000.0, sell=8000.0, fees=33.3) -> TradeSummary:
    return TradeSummary(
        days=1,
        trades=3,
        buy_amount=buy,
        sell_amount=sell,
        total_amount=buy + sell,
        commission=fees,
        stamp_tax=0.0,
        transfer_fee=0.0,
        slippage=0.0,
        fees_by_year={"2024": fees},
    )


def test_report_uses_state_values_and_reconstruction(tmp_path: Path):
    state = StateSnapshot(100000.0, 95966.7, 95966.7)
    report = render_report(_summary(), state, 12345.0, tmp_path / "state.json")
    assert "initial 100,000 -> final cash 95,966.70" in report
    assert "final equity (state equity_history tail): 95,966.70" in report
    assert "matches state cash 95,966.70" in report
    assert "WARNING" not in report
    assert "12345" not in report  # fallback not used when state exists


def test_report_scales_with_initial_capital_not_hardcoded(tmp_path: Path):
    report = render_report(
        _summary(fees=1000.0), StateSnapshot(200000.0, 187000.0, None),
        0.0, tmp_path / "state.json",
    )
    assert "initial 200,000 -> final cash 187,000.00" in report
    assert "cumulative fee drag = 0.5% of initial capital" in report
    assert "100,000" not in report


def test_report_without_state_declares_final_cash_unknown(tmp_path: Path):
    report = render_report(_summary(), None, 100000.0, tmp_path / "absent.json")
    assert "final cash unknown" in report
    assert "state file not found" in report
    assert "WARNING" not in report


def test_report_warns_when_reconstruction_differs_from_state(tmp_path: Path):
    report = render_report(_summary(), StateSnapshot(100000.0, 123.45, None),
                           100000.0, tmp_path / "state.json")
    assert "WARNING: trade-file reconstruction differs from state cash" in report


def test_main_end_to_end(tmp_path: Path, monkeypatch, capsys):
    import scripts.analyze_sim as analyze

    monkeypatch.setattr(analyze, "REPO_ROOT", tmp_path)
    # load_config raises on a missing baseline; provide a minimal one.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "ashare_config.yaml").write_text(
        "data_dir: data\n"
        "duckdb_path: data/ashare.duckdb\n"
        "parquet_dir: data/parquet\n",
        encoding="utf-8",
    )
    write_trades(
        tmp_path / "data" / "sim_trades",
        "20240102",
        [make_trade("buy", amount=10000.0, commission=5.0, transfer=0.1, slippage=5.0),
         make_trade("sell", amount=8000.0, commission=4.0, stamp=4.0, transfer=0.1, slippage=4.0)],
    )
    state_dir = tmp_path / "data"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "sim_portfolio_state.json").write_text(
        json.dumps({"initial_capital": 100000.0, "cash": 97977.8,
                    "equity_history": [{"trade_date": "20240102", "equity": 97977.8}]}),
        encoding="utf-8",
    )
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "final cash 97,977.80" in out
    assert "matches state cash 97,977.80" in out