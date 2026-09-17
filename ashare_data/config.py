"""Configuration dataclasses and loader for AlphaGPT A-share modules."""

from __future__ import annotations

import json
import os
import sys
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


class ConfigPathError(ValueError):
    """A configured path cannot be resolved unambiguously."""


class InputPathMissing(ConfigPathError):
    """A required input path does not exist (fail-closed before any run)."""


#: Path keys of :class:`DataConfig` that form the runtime path surface.
PATH_KEYS: tuple[str, ...] = (
    "data_dir",
    "duckdb_path",
    "v2_duckdb_path",
    "parquet_dir",
)

#: Required inputs: a formal run must refuse to start when one is absent.
REQUIRED_INPUT_KEYS: tuple[str, ...] = ("duckdb_path", "v2_duckdb_path")

_ENV_PATH_KEYS: dict[str, str] = {
    "data_dir": "ASHARE_DATA_DIR",
    "duckdb_path": "ASHARE_DUCKDB_PATH",
    "parquet_dir": "ASHARE_PARQUET_DIR",
    "v2_duckdb_path": "ASHARE_V2_DUCKDB_PATH",
}

_PROJECT_MARKERS: tuple[str, ...] = (".git", "AGENTS.md")


def infer_project_root(config_path: str | Path) -> Path:
    """Deterministic project root of a config file (A01 C2, cwd-independent).

    Resolution order:

    1. the nearest ancestor directory holding ``.git`` or ``AGENTS.md``;
    2. the parent of a ``config`` / ``configs`` directory;
    3. the config file's own directory.

    The result never depends on ``Path.cwd()``, so a relative path inside a
    shared config resolves identically from any working directory.
    """

    path = Path(config_path).resolve()
    for candidate in (path.parent, *path.parent.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate
    if path.parent.name in ("config", "configs"):
        return path.parent.parent
    return path.parent


@dataclass(frozen=True)
class ResolvedPaths:
    """The single resolved runtime-path surface (A01 C1/C5/C7).

    ``sources`` records the winning layer of every key so the effective
    configuration is auditable instead of a merged mystery.
    """

    data_dir: Path
    duckdb_path: Path
    v2_duckdb_path: Path
    parquet_dir: Path
    sources: dict[str, str]
    missing_inputs: tuple[str, ...] = ()

    def effective(self) -> dict[str, dict[str, Any]]:
        """Effective config block: value + winning layer + existence."""

        return {
            key: {
                "path": str(getattr(self, key)),
                "source": self.sources.get(key, "default"),
                "exists": Path(getattr(self, key)).exists(),
            }
            for key in PATH_KEYS
        }


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("data/ashare.duckdb")
    # v2 Tushare/PIT price source (p17 four-layer architecture; p19 §2: the
    # loader's formal price input).  A relative value resolves against the
    # parent directory of ``duckdb_path`` (the shared data directory), so
    # the default sits next to the legacy database; YAML overrides may use
    # absolute paths.  The universe/calendar/stocks/fundamental sources
    # stay on ``duckdb_path`` (the production calendar remains the single
    # authority, p17 §6.1).
    v2_duckdb_path: Path = Path("ashare_v2.duckdb")
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
    # Defaults are calibrated to the smallest documented machine (6 GB
    # VRAM / 16 GB RAM): batch 4096 reproducibly crashed step 0 with a
    # c10.dll access violation on torch 2.11.0+cu128 / Windows, and the
    # search converges within ~50 steps at batch 256 (~1.7 s/step after
    # the cold start), so the protocol screening tier (150 x 256) is the
    # default workload.  Larger batches remain configurable for bigger
    # machines.
    batch_size: int = 256
    train_steps: int = 150
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
    # Default formula searcher (T2-03 admission verdict: RL was NOT admitted
    # — best-so-far area 0.899 vs GP 0.975 / random 0.938 / TPE 0.916 median
    # under identical unique-semantic-evaluation budgets across 5 independent
    # seeds).  "gp" (strongly-typed GP, DEAP) is the production default;
    # "rl" (REINFORCE policy) stays an experimental opt-in; "tpe" and
    # "random" are formal alternative/baseline backends — all billed in
    # unique semantic evaluations.  YAML may override this default via
    # model.searcher.
    searcher: str = "gp"
    # Formal RL runs imitate a baseline elite archive before REINFORCE.
    # ``random`` is retained only as an explicit admission/benchmark arm.
    rl_initialization: str = "imitation"
    imitation_epochs: int = 20
    imitation_batch_size: int = 32
    imitation_learning_rate: float = 1e-3


@dataclass
class RewardConfig:
    """Reward-scoring constants (single source: ashare_model.reward).

    Semantic changes to the reward implementation bump
    ``ashare_model.reward.REWARD_VERSION``; these values only tune the
    current version (v15: portfolio active IR minus exact annualized
    execution costs, scored against a +/-10 clip band; IC/ICIR are
    auxiliary reported/gated statistics; robust HAC-shrunk ICIR, hard
    signal-quality gates and AST complexity billing carried over from
    v12; contract ``docs/p11_reward_v15_contract.md``).

    v15 changes (contract §5): the clip band widens from +/-1.0 to
    +/-10.0 and ``bad_reward`` moves from -2.0 to -20.0 so the sentinel
    stays outside the band; the complexity penalty becomes a two-segment
    non-monotonic shape (free zone at or below ``complexity_free_bill``,
    then linear in the excess).  v14 artifacts are not comparable with
    v15 rewards.
    """

    # v14: +/-1.0.  v15 widens the band to +/-10.0 (contract §5.1): the
    # annualized active IR of usable factor families reaches ~1.75+, so
    # the v14 band clipped nearly every candidate onto a 0.98 ceiling
    # platform and destroyed reward discrimination.
    reward_clip_low: float = -10.0
    reward_clip_high: float = 10.0
    # Assigned by the trainer to invalid/constant formulas only; it sits
    # below ``reward_clip_low`` so unusable formulas stay distinguishable.
    # v14: -2.0; v15 moves it to -20.0 to keep the sentinel outside the
    # widened band (invariant: ``bad_reward < reward_clip_low``).
    bad_reward: float = -20.0
    # Multiplier on exact annualized daily execution costs (1.0 = honest cost).
    cost_weight: float = 1.0
    # Complexity billing (T1-03; v15 redefinition, contract §5.2): the
    # scorer charges nothing at or below ``complexity_free_bill`` and
    # ``complexity_penalty * (bill - complexity_free_bill)`` above it,
    # where the bill combines AST node count, depth, longest operator
    # window and operation cost (bare single-factor copies bill exactly
    # 1.0).  ``complexity_penalty`` is the *excess* slope (v14 semantics:
    # a full slope ``0.02 * bill`` starting at bill=1.0; v14 value 0.02).
    # Both intentional relaxations relative to v14 are disclosed here:
    # (a) the free zone ``bill <= 3.0`` pays nothing (v14 billed 0.02 at
    # bill=1.0); (b) bills in (3.0, 5.0) pay less than v14 (bill=4: 0.05
    # vs 0.08).  High-complexity pressure is not weaker: bill=5 pays 0.10
    # (equal to v14) and bill=25 pays 1.10 vs v14's 0.50.
    complexity_penalty: float = 0.05
    # v15 (contract §5.2): complexity bills at or below this threshold pay
    # no penalty (free zone); above it the excess is billed at
    # ``complexity_penalty`` per unit.  The field is absent in v14.
    complexity_free_bill: float = 3.0
    # Hard complexity ceiling: formulas whose complexity bill exceeds this
    # are rejected outright (``complexity_above_maximum``).
    max_complexity: float = 25.0
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
    # Robust IC statistics (T1-03): the ICIR used in rewards and gates is
    # shrunk by the effective sample size under autocorrelation
    # (Newey-West HAC, ``ic_hac_max_lags`` lags; None = Andrews rule).
    ic_hac_max_lags: int | None = 10
    # Block bootstrap of the mean IC (deterministic seed): the sign-
    # stability probability and CI recorded in quality summaries.
    ic_boot_n: int = 500
    ic_boot_block: int = 5
    # Hard quality gates (T1-03, ``None`` disables a gate).  The defaults
    # reject the degenerate classes — all-zero (activity), two-valid-day
    # (valid-IC days) and extremely sparse (effective stocks vs
    # ``ic_min_stocks``) signals — while passing healthy cross-sections.
    min_valid_ic_days: int | None = 8
    min_effective_stocks: int | None = None  # None -> ic_min_stocks
    min_coverage: float | None = 0.2
    min_activity: float | None = 0.05
    min_sign_stability: float | None = 0.5
    min_val_window_q25: float | None = 0.0
    # T1-04 capacity gate: a position larger than this fraction of the
    # execution-day dollar volume is rejected (``capacity_above_maximum``).
    # Only measurable when the caller supplies dollar-volume data (adv);
    # None disables the gate.
    max_capacity_utilization: float | None = 0.25


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
    current version. P3 makes ``frequency``/``horizon`` executable: they
    resolve the shared rebalance calendar and causal research target.
    """

    frequency: str = "daily"
    horizon: int = 1
    # P6: research domain (one of short_price_volume /
    # medium_cross_section / slow_fundamental, or the reserved
    # compatible semantic "unified").  Domain runs resolve their default
    # frequency/horizon/baseline ladder from the versioned registry
    # (docs/p6_research_domain_contract.md §3).
    domain: str = "unified"
    # p21 §8 (APPROVED 2026-09-06): optional preregistered tier-set
    # restriction for domain runs (None = no restriction, byte-identical
    # to pre-p21 behavior; "A" / "AB" mirror the P2 §6 collections).
    # Unified never combines with a tier set.  The tier semantics (the
    # feature intersection minus deprecated names) live in
    # ashare_model.research_domain — this layer only carries the config
    # grammar, validated below.
    tier_set: str | None = None
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
    # T1-05: when True the baseline receives the trained candidate's exact
    # evaluation budget (tier.steps * tier.batch_size unique evaluations)
    # per fold, making the RL-vs-baseline comparison budget-fair;
    # ``random_samples`` then only applies when this is False.
    random_match_budget: bool = True
    # T2-02/T2-03 baseline ladder: strongly-typed GP (DEAP) and TPE
    # (Optuna) join the random baseline with the same matched budget per
    # fold; False disables either.  The admission experiment (T2-03)
    # decides whether RL stays the default searcher.
    gp_enabled: bool = True
    gp_seed: int = 1235
    tpe_enabled: bool = True
    tpe_seed: int = 1236


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
    rebalance_frequency: str = "daily"
    target_horizon: int = 1
    initial_capital: float = 100000.0
    top_n: int = 30
    # P3 constructor fields. ``None`` ranks preserve legacy programmatic
    # callers: buy_rank=top_n and sell_rank=buy_rank (no implicit buffer).
    portfolio_method: str = "equal_weight"
    buy_rank: int | None = None
    sell_rank: int | None = None
    min_trade_amount: float | None = None
    turnover_budget: float | None = None
    target_weight_change_threshold: float = 0.0
    single_weight_cap: float = 0.05
    commission_rate: float = 0.00025
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.0005
    # P35: A-share board lot for share-level execution (orders are
    # lot-floored; full exits are exact).  100 = the A-share board lot.
    lot_size: int = 100
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


def resolve_v2_duckdb_path(config) -> Path:
    """Single resolution rule for the v2 Tushare/PIT price source (p19 §2.1):
    an absolute value passes through unchanged; a relative value must be a
    bare filename (no directory separator) and sits next to ``duckdb_path``
    (the shared data directory).  Every v2 consumer (loader, industry
    membership) resolves through this function — no second resolution rule.

    A01 C4: a separator-bearing relative value (``data/ashare_v2.duckdb``)
    used to be silently double-joined to ``<data_dir>/data/...``; it is now
    refused, so the two legal forms (absolute path, bare filename) are the
    only accepted ones.
    """

    path = Path(config.v2_duckdb_path)
    if path.is_absolute():
        return path
    if path.parent != Path(".") or path.name in ("", "."):
        raise ConfigPathError(
            "v2_duckdb_path must be an absolute path or a bare filename "
            f"relative to the data directory; got {str(config.v2_duckdb_path)!r} "
            f"(it would resolve to "
            f"{Path(config.duckdb_path).parent / path})"
        )
    return Path(config.duckdb_path).parent / path


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
    *,
    path_sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load YAML config, merge the global runtime overrides file, then
    optional .env overrides.

    The runtime overrides file (default: ``runtime_overrides.yaml`` next to
    the config file) is the single source of web-UI config edits such as
    initial capital, position count and the shared fee model. It is merged
    on top of the YAML baseline for every entry point (run_sim, backtest,
    train), so the effective configuration is identical everywhere.

    Paths are resolved relative to the project root.  A01 C2/C7: when
    ``project_root`` is omitted it is inferred from the config file path
    (never from the process cwd), and ``path_sources`` — when provided —
    receives the winning layer of every path key (``default`` /
    ``config:<file>`` / ``runtime_overrides:<file>`` / ``env:<VAR>``).
    """

    if path is None:
        base = Path(project_root) if project_root is not None else Path.cwd()
        path = base / "config" / "ashare_config.yaml"

    config_path = Path(path)
    if project_root is None:
        project_root = infer_project_root(config_path)
    project_root = Path(project_root)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        # A missing baseline must never degrade to a silent defaults run:
        # every entry point either fails fast (CLI) or catches this in its
        # defensive wrapper (web API).
        raise FileNotFoundError(f"config file not found: {config_path}")

    sources = {key: "default" for key in PATH_KEYS}

    baseline = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(baseline, dict):
        raise ConfigPathError(
            f"config file must contain a mapping: {config_path}"
        )
    for key in PATH_KEYS:
        if key in baseline:
            sources[key] = f"config:{config_path.name}"
    raw = baseline

    if overrides_path is None:
        overrides_path = config_path.parent / RUNTIME_OVERRIDES_FILENAME
    overrides_path = Path(overrides_path)
    if overrides_path.exists():
        overrides_raw = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
        if isinstance(overrides_raw, dict):
            for key in PATH_KEYS:
                if key in overrides_raw:
                    sources[key] = f"runtime_overrides:{overrides_path.name}"
            raw = _deep_merge(raw, overrides_raw)

    raw = _resolve_paths(raw, project_root)

    env_path = config_path.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    for key, env_name in _ENV_PATH_KEYS.items():
        value = os.getenv(env_name)
        if value:
            raw[key] = str(_as_path(value, project_root))
            sources[key] = f"env:{env_name}"

    if path_sources is not None:
        path_sources.update(sources)

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


def resolve_runtime_paths(
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    *,
    mode: str = "formal",
    overrides_path: str | Path | None = None,
) -> ResolvedPaths:
    """The single A01 runtime-path resolver: load, resolve, validate.

    ``mode="formal"`` refuses a run whose required input is absent — the
    failure happens before any database is opened and nothing is created
    (A01 C5/C6).  ``mode="audit"`` reports the same facts without raising,
    for diagnostics/effective-config output (A01 C7).

    Inputs are ``duckdb_path`` and ``v2_duckdb_path``
    (:data:`REQUIRED_INPUT_KEYS`); ``data_dir`` / ``parquet_dir`` are
    outputs and may legitimately not exist yet.
    """

    if mode not in ("formal", "audit"):
        raise ValueError(f"unknown mode {mode!r}; expected 'formal' or 'audit'")

    if config_path is None:
        project_root = Path(project_root) if project_root is not None else Path.cwd()
    else:
        config_path = Path(config_path)
        if project_root is None:
            project_root = infer_project_root(config_path)
        project_root = Path(project_root)
        if not config_path.is_absolute():
            config_path = project_root / config_path

    sources: dict[str, str] = {}
    raw = load_config(
        config_path,
        project_root=project_root,
        overrides_path=overrides_path,
        path_sources=sources,
    )
    data_config = make_data_config(raw, project_root)
    v2_path = resolve_v2_duckdb_path(data_config)
    resolved_by_key = {
        "data_dir": Path(data_config.data_dir),
        "duckdb_path": Path(data_config.duckdb_path),
        "v2_duckdb_path": v2_path,
        "parquet_dir": Path(data_config.parquet_dir),
    }

    missing = tuple(
        key for key in REQUIRED_INPUT_KEYS if not resolved_by_key[key].exists()
    )
    resolved = ResolvedPaths(
        data_dir=resolved_by_key["data_dir"],
        duckdb_path=resolved_by_key["duckdb_path"],
        v2_duckdb_path=resolved_by_key["v2_duckdb_path"],
        parquet_dir=resolved_by_key["parquet_dir"],
        sources=dict(sources),
        missing_inputs=missing,
    )
    if mode == "formal" and missing:
        detail = ", ".join(f"{key}={resolved_by_key[key]}" for key in missing)
        raise InputPathMissing(
            f"required input path(s) missing: {detail} "
            f"(config: {config_path}; a formal run refuses to start and "
            "creates neither a database nor a directory)"
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    """CLI: resolve and validate a config's runtime paths (read-only).

        python -m ashare_data.config --config configs/r1_medium_a.yaml --check
        python -m ashare_data.config --config configs/r1_medium_a.yaml --json

    ``--check`` exits 2 when a required input is missing; ``--json`` prints
    the effective config with per-path provenance.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="resolve/validate the runtime paths of a config (read-only)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML config path (default: <project root>/config/ashare_config.yaml)",
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 2 when a required input path is missing",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the effective config as JSON"
    )
    args = parser.parse_args(argv)

    try:
        resolved = resolve_runtime_paths(
            args.config,
            args.project_root,
            mode="formal" if args.check else "audit",
        )
    except (ConfigPathError, FileNotFoundError) as exc:
        print(f"path resolution failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(resolved.effective(), ensure_ascii=False, indent=2))
    else:
        for key, info in resolved.effective().items():
            print(
                f"{key:16s} {info['path']}  [{info['source']}] "
                f"exists={info['exists']}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


def make_model_config(raw: dict[str, Any]) -> ModelConfig:
    defaults = ModelConfig()
    model_raw = raw.get("model", {}) or {}
    data = {
        k: model_raw.get(k, getattr(defaults, k))
        for k in ModelConfig.__dataclass_fields__
    }
    cfg = ModelConfig(**data)
    if cfg.searcher not in ("gp", "tpe", "random", "rl"):
        raise ValueError(
            "model.searcher must be 'gp', 'tpe', 'random' or 'rl', "
            f"got {cfg.searcher!r}"
        )
    if cfg.rl_initialization not in ("imitation", "random"):
        raise ValueError(
            "model.rl_initialization must be 'imitation' or 'random', "
            f"got {cfg.rl_initialization!r}"
        )
    if (
        cfg.imitation_epochs <= 0
        or cfg.imitation_batch_size <= 0
        or cfg.imitation_learning_rate <= 0.0
    ):
        raise ValueError("model imitation training values must be positive")
    return cfg


def make_reward_config(raw: dict[str, Any]) -> RewardConfig:
    defaults = RewardConfig()
    reward_raw = raw.get("reward", {}) or {}
    data = {
        k: reward_raw.get(k, getattr(defaults, k))
        for k in RewardConfig.__dataclass_fields__
    }
    # P6 §3.4: the reward cost weight defaults to the research domain's
    # declared value when not overridden.
    _, domain = _resolve_protocol_domain(raw.get("protocol", {}) or {})
    if domain is not None and "cost_weight" not in reward_raw:
        data["cost_weight"] = domain.cost_weight
    return RewardConfig(**data)


def _make_tier(tier_raw: dict[str, Any] | None, defaults: TierConfig) -> TierConfig:
    tier_raw = tier_raw or {}
    return TierConfig(
        steps=int(tier_raw.get("steps", defaults.steps)),
        batch_size=int(tier_raw.get("batch_size", defaults.batch_size)),
    )


def _resolve_protocol_domain(
    proto_raw: dict[str, Any],
) -> tuple[str, object | None]:
    """Resolve ``protocol.domain``; returns ``(domain_id, domain|None)``.

    ``unified`` (the default) resolves to ``None``: no domain defaults are
    applied and pre-P6 behavior is preserved byte-for-byte.
    """

    domain_id = str(proto_raw.get("domain", "unified"))
    if domain_id == "unified":
        return domain_id, None
    # IP-12 (01-A2): resolve the domain from the shared bottom-tier data
    # module — ``ashare_data`` must not import ``ashare_model``.
    from ashare_domain import resolve_domain

    return domain_id, resolve_domain(domain_id)


def make_protocol_config(raw: dict[str, Any]) -> ProtocolConfig:
    defaults = ProtocolConfig()
    proto_raw = raw.get("protocol", {}) or {}

    folds_raw = proto_raw.get("folds")
    if folds_raw is None:
        folds = defaults.folds
    else:
        folds = [FoldConfig(**dict(f)) for f in folds_raw]
    folds = validate_folds(folds)

    domain_id, domain = _resolve_protocol_domain(proto_raw)
    # p21 §8.1: tier_set grammar is validated at config time (the tier
    # semantics themselves resolve in ashare_model.research_domain).
    tier_set = proto_raw.get("tier_set")
    if tier_set is not None:
        tier_set = str(tier_set)
        if tier_set not in ("A", "AB"):
            raise ValueError(
                f"invalid protocol.tier_set {tier_set!r}: expected "
                "'A', 'AB' or None (docs/"
                "p21_r1_domain_return_research_preregistration.md §8)"
            )
        if domain_id == "unified":
            raise ValueError(
                "protocol.tier_set requires a research domain; the "
                "unified compatible semantic has no tier restriction "
                "(docs/p21_r1_domain_return_research_preregistration.md §8)"
            )
    if domain is not None:
        frequency = str(proto_raw.get("frequency", domain.default_frequency))
        horizon = int(proto_raw.get("horizon", domain.default_horizon))
        baseline_signals = [
            str(name)
            for name in proto_raw.get("baseline_signals", domain.baseline_signals)
        ]
        if not domain.is_legal_execution(frequency, horizon):
            raise ValueError(
                f"protocol (frequency={frequency!r}, horizon={horizon}) is "
                f"not a legal execution point of domain {domain.id!r} "
                "(docs/p6_research_domain_contract.md §1.2)"
            )
        out_of_domain = [
            name
            for name in baseline_signals
            if name not in domain.features
        ]
        if out_of_domain:
            raise ValueError(
                f"baseline signals {out_of_domain} are outside domain "
                f"{domain.id!r} (docs/p6_research_domain_contract.md §3.2)"
            )
    else:
        frequency = str(proto_raw.get("frequency", defaults.frequency))
        horizon = int(proto_raw.get("horizon", defaults.horizon))
        baseline_signals = [
            str(name)
            for name in proto_raw.get("baseline_signals", defaults.baseline_signals)
        ]

    cfg = ProtocolConfig(
        frequency=frequency,
        horizon=horizon,
        domain=domain_id,
        tier_set=tier_set,
        seeds=[int(s) for s in proto_raw.get("seeds", defaults.seeds)],
        folds=folds,
        baseline_signals=baseline_signals,
        screening=_make_tier(proto_raw.get("screening"), defaults.screening),
        confirmation=_make_tier(proto_raw.get("confirmation"), defaults.confirmation),
        random_samples=int(
            proto_raw.get("random_samples", defaults.random_samples)
        ),
        random_seed=int(proto_raw.get("random_seed", defaults.random_seed)),
        random_match_budget=bool(
            proto_raw.get("random_match_budget", defaults.random_match_budget)
        ),
        gp_enabled=bool(proto_raw.get("gp_enabled", defaults.gp_enabled)),
        gp_seed=int(proto_raw.get("gp_seed", defaults.gp_seed)),
        tpe_enabled=bool(proto_raw.get("tpe_enabled", defaults.tpe_enabled)),
        tpe_seed=int(proto_raw.get("tpe_seed", defaults.tpe_seed)),
    )
    if not cfg.seeds:
        raise ValueError("protocol.seeds must not be empty")
    from ashare_portfolio.rebalance import RebalancePolicy

    RebalancePolicy(cfg.frequency, cfg.horizon)
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
    proto_raw = raw.get("protocol", {}) or {}
    # P6 §3: the backtest execution point and the L1 turnover budget
    # default to the research domain's declared values when not overridden,
    # so protocol and backtest stay aligned on the same cadence.
    _, domain = _resolve_protocol_domain(proto_raw)
    if "rebalance_frequency" not in bt_raw:
        data["rebalance_frequency"] = str(
            proto_raw.get(
                "frequency",
                domain.default_frequency
                if domain is not None
                else defaults.rebalance_frequency,
            )
        )
    if "target_horizon" not in bt_raw:
        data["target_horizon"] = int(
            proto_raw.get(
                "horizon",
                domain.default_horizon
                if domain is not None
                else defaults.target_horizon,
            )
        )
    if domain is not None and "turnover_budget" not in bt_raw:
        data["turnover_budget"] = domain.turnover_budget
    cfg = BacktestConfig(**data)
    from ashare_portfolio.rebalance import RebalancePolicy

    RebalancePolicy.from_config(cfg)
    from ashare_portfolio.constructor import validate_portfolio_config

    validate_portfolio_config(cfg)
    return cfg


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
