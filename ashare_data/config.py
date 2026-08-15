"""Configuration dataclasses and loader for AlphaGPT A-share modules."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _as_path(value: Any, base: Path | None = None) -> Path:
    if value is None:
        return base or Path(".")
    path = Path(str(value))
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/ashare.duckdb")
    parquet_dir: Path = Path("data/parquet")
    calendar_table: str = "trade_calendar"
    stocks_table: str = "stocks"
    daily_table: str = "daily_bar"
    constituents_table: str = "constituents"
    factor_table: str = "factor_cache"
    fundamentals_table: str = "fundamental_pit"
    sync_fundamentals: bool = True
    start_date: str = "2015-01-01"
    end_date: str = "2026-12-31"
    adjust: str = "qfq"
    index_codes: list[str] = field(
        default_factory=lambda: ["000300.SH", "000905.SH", "000852.SH"]
    )
    index_names: list[str] = field(
        default_factory=lambda: ["沪深300", "中证500", "中证1000"]
    )
    min_listed_days: int = 60
    min_price: float = 1.0
    max_price: float = 10000.0
    min_amount: float = 100000.0
    request_retries: int = 3
    request_timeout: int = 20
    daily_provider: str = "auto"


@dataclass
class ModelConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    num_loops: int = 3
    dropout: float = 0.1
    batch_size: int = 4096
    train_steps: int = 1000
    max_formula_len: int = 12
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    value_loss_weight: float = 0.5
    reward_clip_low: float = -3.0
    reward_clip_high: float = 5.0
    feature_names: list[str] | None = None


@dataclass
class BacktestConfig:
    start_date: str = "2015-01-01"
    end_date: str = "2026-12-31"
    train_end_date: str = "2023-12-31"
    initial_capital: float = 100000.0
    top_n: int = 30
    single_weight_cap: float = 0.05
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.0005
    benchmark: str = "全市场等权"
    reward_clip_low: float = -3.0
    reward_clip_high: float = 5.0


@dataclass
class SimConfig:
    initial_capital: float = 100000.0
    max_positions: int = 30
    single_weight_cap: float = 0.05
    benchmark: str = "全市场等权"
    stop_signal_path: Path = Path("STOP_SIGNAL")
    state_path: Path = Path("data/sim_portfolio_state.json")
    orders_dir: Path = Path("data/sim_orders")
    trades_dir: Path = Path("data/sim_trades")
    progress_path: Path = Path("data/sim_progress.json")


def _resolve_paths(
    data: dict[str, Any], base: Path
) -> dict[str, Any]:
    """Resolve relative path values under a project base directory."""

    for key in ("data_dir", "duckdb_path", "parquet_dir"):
        if key in data and data[key] is not None:
            data[key] = str(_as_path(data[key], base))
    return data


RUNTIME_OVERRIDES_FILENAME = "runtime_overrides.yaml"


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge ``patch`` over ``base``: dicts recurse, scalars/lists replace."""

    merged = dict(base)
    for key, value in patch.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None = None,
    project_root: str | Path | None = None,
    overrides_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load YAML config, merge the global runtime overrides file, then
    optional .env overrides.

    The runtime overrides file (default: ``runtime_overrides.yaml`` next to
    the config file) is the single source of web-UI config edits such as
    initial capital, position count and the shared fee model. It is merged
    on top of the YAML baseline for every entry point (run_sim, backtest,
    train), so the effective configuration is identical everywhere.

    Paths are resolved relative to the directory containing the YAML file.
    """

    if project_root is None:
        project_root = Path.cwd()
    project_root = Path(project_root)

    if path is None:
        path = project_root / "config" / "ashare_config.yaml"
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        return {}

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    if overrides_path is None:
        overrides_path = config_path.parent / RUNTIME_OVERRIDES_FILENAME
    overrides_path = Path(overrides_path)
    if overrides_path.exists():
        overrides_raw = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
        if isinstance(overrides_raw, dict):
            raw = _deep_merge(raw, overrides_raw)

    raw = _resolve_paths(raw, project_root)

    env_path = config_path.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    env_map = {
        "data_dir": os.getenv("ASHARE_DATA_DIR"),
        "duckdb_path": os.getenv("ASHARE_DUCKDB_PATH"),
        "parquet_dir": os.getenv("ASHARE_PARQUET_DIR"),
    }
    for key, value in env_map.items():
        if value:
            raw[key] = str(_as_path(value, project_root))

    return raw


def make_data_config(raw: dict[str, Any], project_root: Path) -> DataConfig:
    defaults = DataConfig()
    data = {
        k: raw.get(k, getattr(defaults, k))
        for k in DataConfig.__dataclass_fields__
    }
    for key in ("data_dir", "duckdb_path", "parquet_dir"):
        data[key] = _as_path(data[key], project_root)
    return DataConfig(**data)


def make_model_config(raw: dict[str, Any]) -> ModelConfig:
    defaults = ModelConfig()
    model_raw = raw.get("model", {}) or {}
    data = {
        k: model_raw.get(k, getattr(defaults, k))
        for k in ModelConfig.__dataclass_fields__
    }
    return ModelConfig(**data)


def make_backtest_config(raw: dict[str, Any]) -> BacktestConfig:
    defaults = BacktestConfig()
    bt_raw = raw.get("backtest", {}) or {}
    data = {
        k: bt_raw.get(k, getattr(defaults, k))
        for k in BacktestConfig.__dataclass_fields__
    }
    return BacktestConfig(**data)


def make_sim_config(raw: dict[str, Any], project_root: Path) -> SimConfig:
    defaults = SimConfig()
    sim_raw = raw.get("sim", {}) or {}
    data = {
        k: sim_raw.get(k, getattr(defaults, k))
        for k in SimConfig.__dataclass_fields__
    }
    if not Path(str(data.get("stop_signal_path", "") or "")).is_absolute():
        data["stop_signal_path"] = project_root / str(data.get("stop_signal_path", defaults.stop_signal_path))
    if not Path(str(data.get("state_path", "") or "")).is_absolute():
        data["state_path"] = project_root / str(data.get("state_path", defaults.state_path))
    if not Path(str(data.get("orders_dir", "") or "")).is_absolute():
        data["orders_dir"] = project_root / str(data.get("orders_dir", defaults.orders_dir))
    if not Path(str(data.get("trades_dir", "") or "")).is_absolute():
        data["trades_dir"] = project_root / str(data.get("trades_dir", defaults.trades_dir))
    if not Path(str(data.get("progress_path", "") or "")).is_absolute():
        data["progress_path"] = project_root / str(data.get("progress_path", defaults.progress_path))
    return SimConfig(**data)
