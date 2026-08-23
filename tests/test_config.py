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
    FoldConfig,
    ProtocolConfig,
    TierConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_reward_config,
    make_sim_config,
    validate_baseline_signals,
    validate_folds,
)


def test_load_config_missing_file_raises(tmp_path: Path):
    # A missing baseline fails fast instead of silently degrading to
    # defaults (the web API catches it in its defensive wrapper).
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", project_root=tmp_path)


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
    assert cfg.min_listed_sessions == 60


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
    raw = {"reward": {"bad_reward": -5.0, "complexity_penalty": 0.4}}
    reward = make_reward_config(raw)
    assert reward.bad_reward == -5.0
    assert reward.complexity_penalty == 0.4
    assert reward.reward_clip_low == RewardConfig().reward_clip_low
    assert reward.cost_weight == RewardConfig().cost_weight
    assert reward.min_val_reward == RewardConfig().min_val_reward
    assert reward.min_val_icir == RewardConfig().min_val_icir
    assert reward.ic_min_stocks == RewardConfig().ic_min_stocks


def test_make_model_config_v3_training_fields(tmp_path: Path):
    model = make_model_config({})
    assert model.validation_fraction == ModelConfig().validation_fraction == 0.35
    assert model.validation_splits == ModelConfig().validation_splits == 4
    assert model.entropy_coef == ModelConfig().entropy_coef == 0.01
    assert model.advantage_clip == ModelConfig().advantage_clip == 10.0
    assert model.collapse_warn_fraction == ModelConfig().collapse_warn_fraction
    raw = {"model": {"validation_splits": 2, "entropy_coef": 0.05}}
    overridden = make_model_config(raw)
    assert overridden.validation_splits == 2
    assert overridden.entropy_coef == 0.05


def test_make_protocol_config_random_search_fields(tmp_path: Path):
    proto = make_protocol_config({})
    assert proto.screening.steps == 150
    assert proto.random_samples == 4096
    assert proto.random_seed == 1234
    overridden = make_protocol_config(
        {"protocol": {"random_samples": 0, "random_seed": 99}}
    )
    assert overridden.random_samples == 0
    assert overridden.random_seed == 99
    with pytest.raises(ValueError, match="random_samples"):
        make_protocol_config({"protocol": {"random_samples": -1}})


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


# --- protocol ---------------------------------------------------------------


def test_make_protocol_config_defaults():
    proto = make_protocol_config({})
    assert proto.frequency == "daily"
    assert proto.horizon == 1
    assert proto.seeds == [42, 7, 2024]
    assert len(proto.folds) == 5
    assert proto.folds[0] == FoldConfig("2020-12-31", "2021-12-31")
    assert proto.baseline_signals == [
        "REVERSAL_5",
        "RSQ_60",
        "ILLIQ_20",
        "OVERNIGHT_RET",
        "MOMENTUM_20",
        "ROE",
        "TURNOVER",
    ]
    assert proto.screening == TierConfig(150, 256)
    assert proto.confirmation == TierConfig(200, 512)


def test_make_protocol_config_nested_overrides():
    proto = make_protocol_config(
        {
            "protocol": {
                "seeds": [1, 2],
                "folds": [
                    {"train_end": "2022-06-30", "test_end": "2022-12-31"},
                    {"train_end": "2022-12-31", "test_end": "2023-06-30"},
                ],
                "screening": {"steps": 10, "batch_size": 512},
            }
        }
    )
    assert proto.seeds == [1, 2]
    assert [f.test_end for f in proto.folds] == ["2022-12-31", "2023-06-30"]
    assert proto.screening.steps == 10
    assert proto.screening.batch_size == 512
    # Untouched nested defaults survive.
    assert proto.confirmation == TierConfig(200, 512)
    assert proto.frequency == "daily"


def test_validate_folds_rejects_backward_or_zero_length_fold():
    with pytest.raises(ValueError, match="must be after"):
        validate_folds([FoldConfig("2021-12-31", "2021-12-31")])
    with pytest.raises(ValueError, match="must be after"):
        validate_folds([FoldConfig("2022-06-30", "2021-12-31")])


def test_validate_folds_rejects_unsorted_and_overlapping():
    with pytest.raises(ValueError, match="strictly after"):
        validate_folds(
            [
                FoldConfig("2022-12-31", "2023-12-31"),
                FoldConfig("2021-12-31", "2022-12-31"),
            ]
        )
    with pytest.raises(ValueError, match="overlaps"):
        validate_folds(
            [
                FoldConfig("2021-12-31", "2022-12-31"),
                FoldConfig("2022-06-30", "2023-12-31"),
            ]
        )


def test_make_protocol_config_validates_folds_and_seeds():
    with pytest.raises(ValueError, match="must be after"):
        make_protocol_config(
            {"protocol": {"folds": [{"train_end": "2022-12-31", "test_end": "2022-06-30"}]}}
        )
    with pytest.raises(ValueError, match="seeds"):
        make_protocol_config({"protocol": {"seeds": []}})
    with pytest.raises(ValueError, match="horizon"):
        make_protocol_config({"protocol": {"horizon": 0}})


def test_validate_baseline_signals_unknown_name_raises():
    with pytest.raises(ValueError, match="NOT_A_FACTOR"):
        validate_baseline_signals(["MOMENTUM_20", "NOT_A_FACTOR"], ["MOMENTUM_20", "ROE"])
    assert validate_baseline_signals(["ROE", "MOMENTUM_20"], ["MOMENTUM_20", "ROE"]) == [
        "ROE",
        "MOMENTUM_20",
    ]
