from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ashare_data.config import (
    DataConfig,
    ModelConfig,
    BacktestConfig,
    RewardConfig,
    SimConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_reward_config,
    make_sim_config,
)


def test_load_config_missing_file_returns_empty(tmp_path: Path):
    assert load_config(tmp_path / "nope.yaml", project_root=tmp_path) == {}


def test_load_config_resolves_relative_paths(tmp_path: Path):
    cfg_path = tmp_path / "ashare_config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": "data",
                "duckdb_path": "data/db.duckdb",
                "parquet_dir": "data/parquet",
            }
        ),
        encoding="utf-8",
    )
    raw = load_config(cfg_path, project_root=tmp_path)
    assert str(raw["data_dir"]).startswith(str(tmp_path))
    assert raw["duckdb_path"].endswith("db.duckdb")


def test_load_config_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "ashare_config.yaml"
    cfg_path.write_text(yaml.safe_dump({"data_dir": "ignored"}), encoding="utf-8")
    monkeypatch.setenv("ASHARE_DATA_DIR", str(tmp_path / "env-data"))
    raw = load_config(cfg_path, project_root=tmp_path)
    assert raw["data_dir"] == str(tmp_path / "env-data")


def test_load_config_merges_runtime_overrides(tmp_path: Path):
    cfg_path = tmp_path / "config" / "ashare_config.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "sim": {"max_positions": 30, "initial_capital": 100000.0},
                "backtest": {"top_n": 5, "commission_rate": 0.00025},
            }
        ),
        encoding="utf-8",
    )
    (cfg_path.parent / "runtime_overrides.yaml").write_text(
        yaml.safe_dump(
            {
                "sim": {"max_positions": 10},
                "backtest": {"commission_rate": 0.001},
            }
        ),
        encoding="utf-8",
    )
    raw = load_config(cfg_path, project_root=tmp_path)
    # Overridden keys win, untouched keys keep the baseline (deep merge).
    assert raw["sim"]["max_positions"] == 10
    assert raw["sim"]["initial_capital"] == 100000.0
    assert raw["backtest"]["top_n"] == 5
    assert raw["backtest"]["commission_rate"] == 0.001


def test_load_config_ignores_missing_overrides(tmp_path: Path):
    cfg_path = tmp_path / "config" / "ashare_config.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump({"sim": {"max_positions": 30}}), encoding="utf-8")
    raw = load_config(cfg_path, project_root=tmp_path)
    assert raw["sim"]["max_positions"] == 30


def test_make_data_config_applies_defaults_and_paths(tmp_path: Path):
    cfg = make_data_config({}, tmp_path)
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.duckdb_path == tmp_path / "data" / "ashare.duckdb"
    assert cfg.start_date == DataConfig().start_date


def test_make_model_and_backtest_config_nested(tmp_path: Path):
    raw = {
        "model": {"d_model": 32, "max_formula_len": 7},
        "backtest": {"top_n": 5},
    }
    model = make_model_config(raw)
    backtest = make_backtest_config(raw)
    assert model.d_model == 32
    assert model.max_formula_len == 7
    assert model.num_layers == ModelConfig().num_layers
    assert backtest.top_n == 5
    assert backtest.commission_rate == BacktestConfig().commission_rate


def test_make_reward_config_nested_with_defaults(tmp_path: Path):
    raw = {"reward": {"bad_reward": -5.0, "turnover_threshold": 0.8}}
    reward = make_reward_config(raw)
    assert reward.bad_reward == -5.0
    assert reward.turnover_threshold == 0.8
    assert reward.reward_clip_low == RewardConfig().reward_clip_low
    assert reward.turnover_penalty == RewardConfig().turnover_penalty


def test_make_sim_config_resolves_absolute_paths(tmp_path: Path):
    sim = make_sim_config(
        {
            "sim": {
                "state_path": "state.json",
                "orders_dir": "orders",
                "trades_dir": "trades",
                "stop_signal_path": "STOP",
            }
        },
        tmp_path,
    )
    assert sim.state_path == tmp_path / "state.json"
    assert sim.orders_dir == tmp_path / "orders"
    assert sim.trades_dir == tmp_path / "trades"
    assert sim.stop_signal_path == tmp_path / "STOP"
