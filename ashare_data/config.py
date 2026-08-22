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
    margin_table: str = "margin_balance"
    sw_index_table: str = "sw_industry_index"
    sw_member_table: str = "sw_industry_member"
    sync_fundamentals: bool = True
    sync_capital_flow: bool = True
    start_date: str = "2015-01-01"
    end_date: str = "2026-12-31"
    adjust: str = "qfq"
    index_codes: list[str] = field(
        default_factory=lambda: ["000300.SH", "000905.SH", "000852.SH"]
    )
    index_names: list[str] = field(
        default_factory=lambda: ["沪深300", "中证500", "中证1000"]
    )
    min_listed_sessions: int = 60
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
    # Tail of the training window held out for out-of-sample best-formula
    # selection (0.35: longer than the v3 default so each sub-window holds
    # enough cross-sections for a stable rank-ICIR).
    validation_fraction: float = 0.35
    value_loss_weight: float = 0.5
    # Number of independent sub-windows the validation tail is split into;
    # best-formula selection uses the median reward across them.
    validation_splits: int = 4
    # Exploration bonus: policy loss subtracts entropy_coef * mean(entropy).
    entropy_coef: float = 0.01
    # Advantage normalization is clipped to [-advantage_clip, advantage_clip]
    # so a degenerate reward spread cannot explode the policy gradient.
    advantage_clip: float = 10.0
    # Policy-collapse monitoring: a warning fires after this many consecutive
    # steps whose unique-formula fraction falls below ``collapse_warn_fraction``
    # (mode collapse shows up as the batch re-sampling the same formulas).
    collapse_warn_fraction: float = 0.95
    collapse_warn_steps: int = 10
    feature_names: list[str] | None = None


@dataclass
class RewardConfig:
    """Reward-scoring constants (single source: ashare_model.reward).

    Semantic changes to the reward implementation bump
    ``ashare_model.reward.REWARD_VERSION``; these values only tune the
    current version (v6: direction-adjusted rank-ICIR minus the annualized
    mean of exact daily execution costs).
    """

    reward_clip_low: float = -1.0
    reward_clip_high: float = 1.0
    # Assigned by the trainer to invalid/constant formulas only; it sits
    # below ``reward_clip_low`` so unusable formulas stay distinguishable.
    bad_reward: float = -2.0
    # Multiplier on exact annualized daily execution costs (1.0 = honest cost).
    cost_weight: float = 1.0
    # Subtracted from the reward of formulas without any operator (bare
    # single-factor copies), nudging the policy towards combinations.  Kept
    # small: the v3 value (0.2) pushed every admissible signal below the
    # validation floor and made the floor unreachable in practice.
    complexity_penalty: float = 0.02
    # Cost-adjusted validation floor: training saves no artifact unless the
    # best validation reward reaches it (0.0 = at least zero net signal).
    min_val_reward: float = 0.0
    # Signal-quality gate: the candidate's validation-window rank-ICIR must
    # reach this value before anything is saved.  Guards against low-IC
    # low-turnover formulas (e.g. quarterly fundamental copies) that pass
    # the cost-adjusted floor on turnover alone.  Grounded in the factor
    # diagnostics: the useful families score ICIR >= 0.11 while the dead
    # ones sit below 0.03.
    min_val_icir: float = 0.05
    # Minimum finite cross-section per date for a rank-IC observation.
    ic_min_stocks: int = 10


@dataclass
class FoldConfig:
    """One walk-forward fold, anchored to absolute dates (YYYY-MM-DD).

    Train on data up to and including ``train_end``; the out-of-sample test
    window is (``train_end``, ``test_end``].  Absolute anchors keep folds
    stable as the database grows.
    """

    train_end: str
    test_end: str


@dataclass
class TierConfig:
    """Search budget of one protocol tier (steps x samples-per-step)."""

    steps: int = 150
    batch_size: int = 256


@dataclass
class ProtocolConfig:
    """Evaluation-protocol constants (single source: ashare_model.evaluation).

    Semantic changes to the protocol implementation bump
    ``ashare_model.evaluation.PROTOCOL_VERSION``; these values only tune the
    current version.  ``frequency``/``horizon`` are record-only for now: no
    rebalance-calendar mechanism exists yet (weekly/multi-period deferred to
    a later phase), but they are written into protocol artifacts so future
    runs can be told apart.
    """

    frequency: str = "daily"
    horizon: int = 1
    seeds: list[int] = field(default_factory=lambda: [42, 7, 2024])
    folds: list[FoldConfig] = field(
        default_factory=lambda: [
            FoldConfig("2020-12-31", "2021-12-31"),
            FoldConfig("2021-12-31", "2022-12-31"),
            FoldConfig("2022-12-31", "2023-12-31"),
            FoldConfig("2023-12-31", "2024-12-31"),
            FoldConfig("2024-12-31", "2025-12-31"),
        ]
    )
    # Single-factor baselines (momentum / quality / liquidity / risk
    # features from the vocabulary, chosen by their diagnostic ICIR).
    # Names are validated against the vocab when the protocol is built.
    baseline_signals: list[str] = field(
        default_factory=lambda: [
            "REVERSAL_5",
            "RSQ_60",
            "ILLIQ_20",
            "OVERNIGHT_RET",
            "MOMENTUM_20",
            "ROE",
            "TURNOVER",
        ]
    )
    screening: TierConfig = field(default_factory=TierConfig)
    confirmation: TierConfig = field(
        default_factory=lambda: TierConfig(steps=200, batch_size=512)
    )
    # Random-search baseline: uniformly sampled structurally-valid formulas,
    # scored with the same reward path, one best per fold.  Separates
    # "the RL search is ineffective" from "the reward is uninformative".
    # 0 disables the baseline.
    random_samples: int = 4096
    random_seed: int = 1234


def validate_folds(folds: list[FoldConfig]) -> list[FoldConfig]:
    """Return ``folds`` after checking structure.

    Every fold must point forward (``test_end > train_end``); folds must be
    strictly increasing and non-overlapping (a test window never reaches
    into the next train window).  Date strings compare lexicographically,
    which is exact for ``YYYY-MM-DD``.
    """

    for i, fold in enumerate(folds):
        if fold.test_end <= fold.train_end:
            raise ValueError(
                f"fold {i}: test_end {fold.test_end} must be after "
                f"train_end {fold.train_end}"
            )
        if i > 0:
            prev = folds[i - 1]
            if fold.train_end <= prev.train_end or fold.test_end <= prev.test_end:
                raise ValueError(
                    f"fold {i} ({fold.train_end} -> {fold.test_end}) is not "
                    f"strictly after fold {i - 1} "
                    f"({prev.train_end} -> {prev.test_end})"
                )
            if fold.train_end < prev.test_end:
                raise ValueError(
                    f"fold {i} train window overlaps fold {i - 1} test window "
                    f"({fold.train_end} < {prev.test_end})"
                )
    return folds


def validate_baseline_signals(names: list[str], allowed) -> list[str]:
    """Return ``names`` after checking every one exists in ``allowed``."""

    allowed_set = set(allowed)
    unknown = [name for name in names if name not in allowed_set]
    if unknown:
        raise ValueError(
            f"baseline signals not in the vocabulary: {', '.join(unknown)}"
        )
    return names


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


def make_reward_config(raw: dict[str, Any]) -> RewardConfig:
    defaults = RewardConfig()
    reward_raw = raw.get("reward", {}) or {}
    data = {
        k: reward_raw.get(k, getattr(defaults, k))
        for k in RewardConfig.__dataclass_fields__
    }
    return RewardConfig(**data)


def _make_tier(tier_raw: dict[str, Any] | None, defaults: TierConfig) -> TierConfig:
    tier_raw = tier_raw or {}
    return TierConfig(
        steps=int(tier_raw.get("steps", defaults.steps)),
        batch_size=int(tier_raw.get("batch_size", defaults.batch_size)),
    )


def make_protocol_config(raw: dict[str, Any]) -> ProtocolConfig:
    defaults = ProtocolConfig()
    proto_raw = raw.get("protocol", {}) or {}

    folds_raw = proto_raw.get("folds")
    if folds_raw is None:
        folds = defaults.folds
    else:
        folds = [FoldConfig(**dict(f)) for f in folds_raw]
    folds = validate_folds(folds)

    cfg = ProtocolConfig(
        frequency=str(proto_raw.get("frequency", defaults.frequency)),
        horizon=int(proto_raw.get("horizon", defaults.horizon)),
        seeds=[int(s) for s in proto_raw.get("seeds", defaults.seeds)],
        folds=folds,
        baseline_signals=[
            str(name)
            for name in proto_raw.get("baseline_signals", defaults.baseline_signals)
        ],
        screening=_make_tier(proto_raw.get("screening"), defaults.screening),
        confirmation=_make_tier(proto_raw.get("confirmation"), defaults.confirmation),
        random_samples=int(
            proto_raw.get("random_samples", defaults.random_samples)
        ),
        random_seed=int(proto_raw.get("random_seed", defaults.random_seed)),
    )
    if not cfg.seeds:
        raise ValueError("protocol.seeds must not be empty")
    if cfg.horizon < 1:
        raise ValueError("protocol.horizon must be a positive integer")
    if cfg.random_samples < 0:
        raise ValueError("protocol.random_samples must be >= 0")
    return cfg


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
