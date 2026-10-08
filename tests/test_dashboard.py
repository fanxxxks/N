from __future__ import annotations

import json
from pathlib import Path
import importlib
import subprocess
import sys

from ashare_data.config import DataConfig, SimConfig
from dashboard.data_service import (
    load_backtest_result,
    load_data_status,
    load_sim_state,
)
from dashboard.visualizer import equity_figure

REPO_ROOT = Path(__file__).resolve().parents[1]


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
    assert len(fig1.data) == 1


def test_dashboard_app_imports():
    app = importlib.import_module("dashboard.app")
    assert callable(app.main)


# --- P1-3: launch-mode guards -------------------------------------------------


def test_dashboard_app_bare_script_execution_exits_with_guidance():
    """`python dashboard/app.py` has no parent package for the relative
    imports; it must exit with a pointer to `streamlit run` instead of
    the cryptic "attempted relative import" ImportError."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "dashboard" / "app.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "attempted relative import" not in combined
    assert "streamlit run dashboard/app.py" in combined


def test_dashboard_app_module_run_without_streamlit_exits_with_guidance():
    """`python -m dashboard.app` runs Streamlit's bare mode (no server,
    nothing renders); it must exit with the same guidance."""
    proc = subprocess.run(
        [sys.executable, "-m", "dashboard.app"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "attempted relative import" not in combined
    assert "streamlit run dashboard/app.py" in combined


def test_dashboard_app_streamlit_script_mode_imports_resolve():
    """`streamlit run dashboard/app.py` executes the file as a bare
    __main__ script: no parent package, the script's directory on
    sys.path, streamlit already imported by the server — reproduced here
    exactly. The PEP 366 bootstrap must make the package-relative
    imports resolve and let main() complete instead of raising
    ImportError."""
    code = (
        "import sys, runpy\n"
        "sys.path.insert(0, 'dashboard')\n"
        "import streamlit\n"
        "runpy.run_path('dashboard/app.py', run_name='__main__')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    combined = proc.stderr + proc.stdout
    assert "attempted relative import" not in combined
    assert "ImportError" not in combined
