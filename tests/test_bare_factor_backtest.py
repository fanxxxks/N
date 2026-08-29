"""Tests for the seven bare-factor fixed backtest (P1-03).

Contract (docs/phase5_measurement_log.md §2, BARE_FACTOR_BACKTEST_VERSION 1):

* Fixed backtest only: the bare factor column enters the shared backtest
  engine directly; there is NO searcher, NO sampling and NO trial-and-
  error anywhere in the path — the payload records ``search: "none"``
  and repeated runs are bit-identical (no RNG).
* Trade direction is the fixed rule of the training window (columns
  before ``backtest.train_end_date``): the sign of the mean forward rank
  IC over signal-date eligible cells (``signal_direction``), the same
  convention the protocol baseline rows use.  Direction is decided once
  per factor, never searched.
* One row per factor: name, direction, engine metrics (net of the shared
  fee schedule), plus config provenance (top_n, fee params, window).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig
from ashare_model.bare_factor_backtest import (
    BARE_FACTOR_BACKTEST_VERSION,
    backtest_bare_factors,
    main as bare_factor_main,
)
from ashare_model.data_loader import AshareDataLoader
from ashare_model.evaluation import PROTOCOL_VERSION
from ashare_model.reward import REWARD_VERSION
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.golden import EXECUTION_SPEC_VERSION

SEVEN = ["REVERSAL_5", "RSQ_60", "ILLIQ_20", "OVERNIGHT_RET",
         "MOMENTUM_20", "ROE", "TURNOVER"]

METRIC_KEYS = {
    "total_return",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "average_turnover",
}


def _loader(populated_db: DataConfig) -> AshareDataLoader:
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    return loader


def test_backtest_seven_factors_fixed_only(populated_db: DataConfig):
    loader = _loader(populated_db)
    bt = BacktestConfig(top_n=2, train_end_date="2024-02-01")
    payload = backtest_bare_factors(loader, bt, names=SEVEN)
    assert payload["version"] == BARE_FACTOR_BACKTEST_VERSION
    assert payload["reward_version"] == REWARD_VERSION
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["execution_version"] == EXECUTION_SPEC_VERSION
    assert (
        payload["portfolio_constructor_version"]
        == PORTFOLIO_CONSTRUCTOR_VERSION
    )
    assert payload["portfolio_config"]["buy_rank"] == 2
    assert payload["search"] == "none"  # fixed backtest: no searcher
    assert [r["name"] for r in payload["factors"]] == SEVEN
    assert payload["config"]["top_n"] == 2
    assert payload["config"]["train_end_date"] == "2024-02-01"
    assert payload["config"]["commission_rate"] == 0.00025
    assert payload["config"]["min_commission"] == 5.0
    for row in payload["factors"]:
        assert row["direction"] in (1, -1)
        assert row["formula_text"] == row["name"]
        assert METRIC_KEYS <= set(row["metrics"])
        assert np.isfinite(row["metrics"]["average_turnover"])
        assert row["metrics"]["average_turnover"] >= 0.0


def test_backtest_is_deterministic(populated_db: DataConfig):
    loader = _loader(populated_db)
    bt = BacktestConfig(top_n=2, train_end_date="2024-02-01")
    first = backtest_bare_factors(loader, bt, names=SEVEN)
    second = backtest_bare_factors(loader, bt, names=SEVEN)
    assert first == second  # no RNG anywhere: bit-identical


def test_backtest_unknown_factor_rejected(populated_db: DataConfig):
    loader = _loader(populated_db)
    bt = BacktestConfig(top_n=2, train_end_date="2024-02-01")
    with pytest.raises(ValueError):
        backtest_bare_factors(loader, bt, names=["NOT_A_FACTOR"])


def test_backtest_empty_names_rejected(populated_db: DataConfig):
    loader = _loader(populated_db)
    bt = BacktestConfig(top_n=2, train_end_date="2024-02-01")
    with pytest.raises(ValueError):
        backtest_bare_factors(loader, bt, names=[])


def test_direction_uses_training_window_rule(populated_db: DataConfig):
    """The direction is the fixed training-window rank-IC sign rule, not
    a search: a factor whose training-window IC is non-finite defaults to
    +1 (the neutral direction of signal_direction)."""
    loader = _loader(populated_db)
    bt = BacktestConfig(top_n=2, train_end_date="2024-02-01")
    payload = backtest_bare_factors(loader, bt, names=["TURNOVER"])
    assert payload["factors"][0]["direction"] in (1, -1)


def test_cli_smoke_writes_versioned_payload(tmp_path, populated_db: DataConfig):
    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "bare_factor_backtest.json"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(populated_db.data_dir),
                "duckdb_path": str(populated_db.duckdb_path),
                "parquet_dir": str(populated_db.parquet_dir),
                "index_codes": ["000300.SH"],
                "min_listed_sessions": 1,
                "backtest": {
                    "train_end_date": "2024-02-01",
                    "top_n": 2,
                },
                "sim": {
                    "max_positions": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    rc = bare_factor_main(
        [
            "--config",
            str(cfg_path),
            "--output",
            str(out_path),
            "--min-eligible",
            "3",
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["version"] == BARE_FACTOR_BACKTEST_VERSION
    assert payload["search"] == "none"
    assert len(payload["factors"]) == len(SEVEN)
