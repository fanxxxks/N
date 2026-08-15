from __future__ import annotations

import json
from pathlib import Path
import importlib

from ashare_data.config import DataConfig, SimConfig
from dashboard.data_service import (
    load_backtest_result,
    load_data_status,
    load_sim_state,
)
from dashboard.visualizer import equity_figure, factor_bar_figure


def test_load_backtest_result_missing_and_present(tmp_path: Path):
    cfg = DataConfig(data_dir=tmp_path)
    assert load_backtest_result(cfg) == {}
    payload = {"metrics": {"sharpe": 1.2}}
    (tmp_path / "backtest_result.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_backtest_result(cfg) == payload


def test_load_sim_state(tmp_path: Path):
    cfg = SimConfig(state_path=tmp_path / "state.json")
    assert load_sim_state(cfg) == {}
    payload = {"cash": 100.0, "positions": {}}
    (tmp_path / "state.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_sim_state(cfg)["cash"] == 100.0


def test_load_data_status_missing_and_ready(tmp_path: Path, populated_db: DataConfig):
    missing = DataConfig(duckdb_path=tmp_path / "missing.duckdb")
    assert load_data_status(missing)["ready"] is False
    status = load_data_status(populated_db)
    assert status["ready"] is True
    assert status["stocks"] == 3


def test_visualizers_build_figures():
    fig1 = equity_figure(["d1", "d2"], [1.0, 1.1], "Equity")
    fig2 = factor_bar_figure(["A", "B"], [1, 2], "Factors")
    assert len(fig1.data) == 1
    assert len(fig2.data) == 1


def test_dashboard_app_imports():
    app = importlib.import_module("dashboard.app")
    assert callable(app.main)
