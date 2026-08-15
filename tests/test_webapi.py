from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from webapi import service


def _make_root(tmp_path: Path) -> Path:
    """A minimal project root with a config baseline (no DuckDB needed)."""

    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "config" / "ashare_config.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data",
                "duckdb_path": "data/ashare.duckdb",
                "parquet_dir": "data/parquet",
                "backtest": {
                    "initial_capital": 100000.0,
                    "top_n": 30,
                    "commission_rate": 0.00025,
                    "min_commission": 5.0,
                    "stamp_tax_rate": 0.0005,
                    "transfer_fee_rate": 0.00001,
                    "slippage_rate": 0.0005,
                },
                "sim": {
                    "initial_capital": 100000.0,
                    "max_positions": 30,
                    "state_path": "data/sim_portfolio_state.json",
                    "orders_dir": "data/sim_orders",
                    "trades_dir": "data/sim_trades",
                    "stop_signal_path": "STOP_SIGNAL",
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_sim_config_patch_writes_overrides_and_effective(tmp_path: Path):
    root = _make_root(tmp_path)
    result = service.write_sim_config(
        service.SimConfigPatch(
            initial_capital=200000.0,
            max_positions=10,
            commission_rate=0.0003,
            stamp_tax_rate=0.001,
        ),
        root=root,
    )
    assert result["ok"]
    assert result["effective"]["initial_capital"] == 200000.0
    assert result["effective"]["max_positions"] == 10
    assert result["effective"]["commission_rate"] == 0.0003
    assert result["effective"]["stamp_tax_rate"] == 0.001
    # Untouched keys fall back to the YAML baseline.
    assert result["effective"]["min_commission"] == 5.0
    assert result["effective"]["slippage_rate"] == 0.0005

    # The override file is written next to the baseline; the baseline itself
    # is never touched.
    overrides = yaml.safe_load(
        (tmp_path / "config" / "runtime_overrides.yaml").read_text(encoding="utf-8")
    )
    assert overrides["sim"]["initial_capital"] == 200000.0
    assert overrides["backtest"]["commission_rate"] == 0.0003
    baseline = yaml.safe_load(
        (tmp_path / "config" / "ashare_config.yaml").read_text(encoding="utf-8")
    )
    assert baseline["sim"]["initial_capital"] == 100000.0


def test_sim_config_none_removes_override_key(tmp_path: Path):
    root = _make_root(tmp_path)
    service.write_sim_config(
        service.SimConfigPatch(initial_capital=300000.0, slippage_rate=0.01),
        root=root,
    )
    service.write_sim_config(
        service.SimConfigPatch(initial_capital=None, slippage_rate=None),
        root=root,
    )
    result = service.get_sim_config(root=root)
    assert result["effective"]["initial_capital"] == 100000.0
    assert result["effective"]["slippage_rate"] == 0.0005
    assert result["overrides"] == {}


def test_sim_config_pending_reset_flags_initial_capital(tmp_path: Path):
    root = _make_root(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sim_portfolio_state.json").write_text(
        json.dumps(
            {
                "initial_capital": 100000.0,
                "cash": 100000.0,
                "trade_count": 0,
                "positions": {},
                "equity_history": [],
            }
        ),
        encoding="utf-8",
    )
    assert service.get_sim_config(root=root)["pending_reset"] is False

    service.write_sim_config(
        service.SimConfigPatch(initial_capital=250000.0), root=root
    )
    assert service.get_sim_config(root=root)["pending_reset"] is True


def test_sim_config_patch_validation(tmp_path: Path):
    root = _make_root(tmp_path)
    with pytest.raises(ValueError):
        service.SimConfigPatch(initial_capital=-5.0)
    with pytest.raises(ValueError):
        service.SimConfigPatch(max_positions=0)
    with pytest.raises(ValueError):
        service.SimConfigPatch(commission_rate=1.5)
    with pytest.raises(ValueError):
        service.SimConfigPatch(min_commission=-1.0)
    assert service.get_sim_config(root=root)["overrides"] == {}
