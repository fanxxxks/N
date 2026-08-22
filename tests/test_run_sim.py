from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig, SimConfig
from ashare_data.db import AshareDB
from ashare_model.data_loader import AshareDataLoader
from ashare_trading.run_sim import SimulationRunner
from tests.conftest import make_bars


def _write_sim_db(data_config: DataConfig, dates, ts_codes, bars):
    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {"ts_code": code, "name": code, "industry": None, "list_date": "20200101", "is_st": False}
                for code in ts_codes
            ],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": date, "is_open": True} for date in dates],
            data_config,
        )
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": code, "in_date": f"2020010{i + 1}", "out_date": "99991231"}
                for i, code in enumerate(ts_codes)
            ],
            data_config,
        )


def _make_runner(
    tmp_path: Path,
    n_dates: int = 40,
    top_n: int = 2,
    max_positions: int | None = None,
) -> tuple[SimulationRunner, list[str]]:
    """A runner over ``n_dates`` synthetic bars with all paths under tmp_path."""

    dates, ts_codes, bars = make_bars(n_dates)
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
    )
    _write_sim_db(data_config, dates, ts_codes, bars)
    model_config = ModelConfig(max_formula_len=6)
    loader = AshareDataLoader(data_config, model_config)
    loader.load_data(ts_codes=ts_codes, dates=dates)
    (data_config.data_dir / "best_ashare_strategy.json").write_text(
        json.dumps({"formula": [1], "formula_text": "RET_1"}), encoding="utf-8"
    )

    position_count = max_positions if max_positions is not None else top_n
    sim_config = SimConfig(
        initial_capital=100000.0,
        max_positions=position_count,
        single_weight_cap=0.5,
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
        stop_signal_path=tmp_path / "STOP",
        progress_path=tmp_path / "progress.json",
    )
    runner = SimulationRunner(
        data_config,
        model_config,
        BacktestConfig(
            initial_capital=sim_config.initial_capital,
            top_n=position_count,
            single_weight_cap=sim_config.single_weight_cap,
            train_end_date="2024-01-20",
        ),
        sim_config,
        loader,
    )
    return runner, dates


def _snapshot(runner: SimulationRunner) -> dict:
    """Everything a replayed run could pollute, compared as plain data."""

    return {
        "cash": runner.portfolio.cash,
        "trade_count": runner.portfolio.trade_count,
        "positions": {k: asdict(v) for k, v in runner.portfolio.positions.items()},
        "equity_history": runner.portfolio.equity_history,
        "last_exec_date": runner.portfolio.last_exec_date,
        "orders": {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(runner.sim_config.orders_dir.glob("*.json"))
        },
        "trades": {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(runner.sim_config.trades_dir.glob("*.json"))
        },
    }


def test_simulation_runner_end_to_end(tmp_path: Path):
    runner, _ = _make_runner(tmp_path)
    result = runner.run()
    assert isinstance(result["final_cash"], float)
    assert isinstance(result["trade_count"], int)
    assert result["equity_history"]
    assert list(runner.sim_config.orders_dir.glob("*.json"))
    assert list(runner.sim_config.trades_dir.glob("*.json"))
    assert runner.sim_config.state_path.exists()

    # Persisted orders must carry their final execution status (filled /
    # skipped), not the pre-execution "pending" placeholder.
    order_files = sorted(runner.sim_config.orders_dir.glob("*.json"))
    assert order_files
    payload = json.loads(order_files[-1].read_text(encoding="utf-8"))
    if payload:
        assert all(o.get("status") != "pending" for o in payload)


def test_simulation_runner_respects_max_positions(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, max_positions=1, top_n=3)
    runner.run()
    assert len(runner.portfolio.positions) <= 1


def test_simulation_runner_stop_signal(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, n_dates=8)
    stop = tmp_path / "STOP"
    stop.write_text("STOP", encoding="utf-8")
    result = runner.run()
    assert result["equity_history"] == []
    assert stop.read_text(encoding="utf-8") == "STOPPED"
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "stopped"


def test_simulation_runner_writes_progress(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=8)
    runner.run()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["phase"] == "finished"
    assert progress["current_date"] == dates[-1]


def test_simulation_runner_resume_matches_full_run(tmp_path: Path):
    """Golden test: one full pass == two passes (first chunk + --resume)."""

    full_runner, dates = _make_runner(tmp_path, n_dates=30)
    full_runner.run()
    full_snapshot = _snapshot(full_runner)

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    resume_runner, split_dates = _make_runner(split_dir, n_dates=30)
    assert split_dates == dates
    # First chunk: signals through dates[15], executions through dates[16].
    resume_runner.run(end_date=dates[15])
    assert resume_runner.portfolio.last_exec_date == dates[16]
    resume_runner.run(resume=True)
    resume_snapshot = _snapshot(resume_runner)

    assert resume_snapshot == full_snapshot


def test_simulation_runner_plain_run_on_history_fails(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=12)
    runner.run(end_date=dates[6])
    with pytest.raises(ValueError, match="resume"):
        runner.run()


def test_simulation_runner_resume_rejects_start_before_watermark(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=12)
    runner.run(end_date=dates[6])
    with pytest.raises(ValueError, match="precedes"):
        runner.run(start_date=dates[2], resume=True)


def test_simulation_runner_resume_requires_watermark(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, n_dates=12)
    # Fresh state has no last_exec_date; --resume without any history is an error.
    with pytest.raises(ValueError, match="last_exec_date"):
        runner.run(resume=True)


def test_simulation_runner_reset_allows_clean_replay(tmp_path: Path):
    runner, dates = _make_runner(tmp_path, n_dates=12)
    runner.run(end_date=dates[6])
    runner.portfolio.reset()
    result = runner.run()
    history_dates = [h["trade_date"] for h in result["equity_history"]]
    assert len(history_dates) == len(dates) - 1
    assert len(history_dates) == len(set(history_dates))
    assert runner.portfolio.last_exec_date == dates[-1]


def test_simulation_runner_honors_artifact_direction(tmp_path: Path):
    runner, _ = _make_runner(tmp_path, n_dates=12)
    # Record a negative direction in the strategy artifact: the runner must
    # use it verbatim instead of inferring from the training window.
    path = runner.data_config.data_dir / "best_ashare_strategy.json"
    path.write_text(
        json.dumps(
            {
                "formula": [1],
                "formula_text": "RET_1",
                "direction": -1,
            }
        ),
        encoding="utf-8",
    )
    runner.load_formula()
    assert runner.direction == -1
    assert runner._has_recorded_direction
    runner.run()
    # A direction of -1 flips the signal: the selected paper orders must
    # differ from a +1 replay of the same formula (fresh state each side).
    flipped_orders = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(runner.sim_config.orders_dir.glob("*.json"))
    }
    runner.portfolio.reset()
    runner.direction = 1
    runner._has_recorded_direction = True
    runner.run()
    replay_orders = {
        p.name: p.read_text(encoding="utf-8")
        for p in sorted(runner.sim_config.orders_dir.glob("*.json"))
    }
    assert flipped_orders != replay_orders
