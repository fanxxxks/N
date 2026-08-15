from __future__ import annotations

import json
from pathlib import Path

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


def test_simulation_runner_end_to_end(tmp_path: Path):
    dates, ts_codes, bars = make_bars(8)
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    _write_sim_db(data_config, dates, ts_codes, bars)
    model_config = ModelConfig(max_formula_len=6)
    loader = AshareDataLoader(data_config, model_config)
    loader.load_data(ts_codes=ts_codes, dates=dates)

    formula_path = data_config.data_dir / "best_ashare_strategy.json"
    formula_path.write_text(json.dumps({"formula": [1], "formula_text": "RET_1"}), encoding="utf-8")

    sim_config = SimConfig(
        initial_capital=100000.0,
        max_positions=2,
        single_weight_cap=0.5,
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
        stop_signal_path=tmp_path / "STOP",
    )
    runner = SimulationRunner(
        data_config,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-01-20"),
        sim_config,
        loader,
    )
    result = runner.run()
    assert isinstance(result["final_cash"], float)
    assert isinstance(result["trade_count"], int)
    assert result["equity_history"]
    assert list(tmp_path.glob("orders/*.json"))
    assert list(tmp_path.glob("trades/*.json"))
    assert (tmp_path / "state.json").exists()

    # Persisted orders must carry their final execution status (filled /
    # skipped), not the pre-execution "pending" placeholder.
    order_files = sorted(tmp_path.glob("orders/*.json"))
    assert order_files
    payload = json.loads(order_files[-1].read_text(encoding="utf-8"))
    if payload:
        assert all(o.get("status") != "pending" for o in payload)


def test_simulation_runner_respects_max_positions(tmp_path: Path):
    dates, ts_codes, bars = make_bars(8)
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    _write_sim_db(data_config, dates, ts_codes, bars)
    model_config = ModelConfig(max_formula_len=6)
    loader = AshareDataLoader(data_config, model_config)
    loader.load_data(ts_codes=ts_codes, dates=dates)
    (data_config.data_dir / "best_ashare_strategy.json").write_text(
        json.dumps({"formula": [1]}), encoding="utf-8"
    )
    sim_config = SimConfig(
        initial_capital=100000.0,
        max_positions=1,
        single_weight_cap=1.0,
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
        stop_signal_path=tmp_path / "STOP",
    )
    runner = SimulationRunner(
        data_config,
        model_config,
        BacktestConfig(top_n=3, train_end_date="2024-01-20"),
        sim_config,
        loader,
    )
    runner.run()
    assert len(runner.portfolio.positions) <= 1


def test_simulation_runner_stop_signal(tmp_path: Path):
    dates, ts_codes, bars = make_bars(8)
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    _write_sim_db(data_config, dates, ts_codes, bars)
    model_config = ModelConfig(max_formula_len=6)
    loader = AshareDataLoader(data_config, model_config)
    loader.load_data(ts_codes=ts_codes, dates=dates)
    (data_config.data_dir / "best_ashare_strategy.json").write_text(
        json.dumps({"formula": [1]}), encoding="utf-8"
    )

    sim_config = SimConfig(
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
        stop_signal_path=tmp_path / "STOP",
    )
    runner = SimulationRunner(
        data_config,
        model_config,
        BacktestConfig(top_n=2),
        sim_config,
        loader,
    )
    stop = tmp_path / "STOP"
    stop.write_text("STOP", encoding="utf-8")
    result = runner.run()
    assert result["equity_history"] == []
    assert stop.read_text(encoding="utf-8") == "STOPPED"
