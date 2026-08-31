"""Immutable RunSpec, spec_id and the resolving factory (P8-03).

The RunSpec is the pre-registration artifact of one research run: frozen
(``frozen=True``), strict (``extra="forbid"``) and fully explicit — every
field is a resolved semantic value, so ``spec_id`` (its strict content
identity, ``content_hash("runspec", payload)`` from the single
``ashare_model.identity`` primitive) changes whenever and only when a
semantic field changes. Field order and machine paths cannot influence
it: non-semantic fields (created_at, hostname, run_id, absolute paths)
are absent from the payload by construction.

``build_runspec`` is the only sanctioned factory. It resolves every
default explicitly (searcher / max_formula_len / seeds / universe policy
from the owning single-authority modules), captures the code version set,
the git commit and the dependency-lock hash, refuses unresolved values,
unknown searcher/domain names, an empty dataset id, and — by default — a
dirty tracked git tree (untracked files are tolerated). It does NOT wire
the RunSpec into training or paper (P8-03 non-goal).

Version note: operator semantics have no standalone version constant in
this repository — the operator registry is versioned through
``FEATURE_REGISTRY_VERSION``/``GRAMMAR_VERSION`` (single registry
authority), so none is invented here.
"""

from __future__ import annotations

import importlib
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ashare_model.identity import content_hash, sha256_hex

RUNSPEC_SCHEMA_VERSION = 1

_SPEC_ID_KIND = "runspec"
_KNOWN_SEARCHERS = ("gp", "tpe", "random", "rl")
_DATE_RE = r"^\d{8}$"
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

# Version owners (AGENTS.md §3.2: search the actual owners, not a list of
# examples). Operator semantics ride on the feature registry + grammar.
_VERSION_IMPORTS: dict[str, tuple[str, str]] = {
    "feature_registry_version": ("ashare_model.feature_registry", "FEATURE_REGISTRY_VERSION"),
    "data_tier_version": ("ashare_model.data_tier", "DATA_TIER_VERSION"),
    "research_domain_version": ("ashare_model.research_domain", "RESEARCH_DOMAIN_VERSION"),
    "grammar_version": ("ashare_model.vocab", "GRAMMAR_VERSION"),
    "model_version": ("ashare_model.alphagpt", "MODEL_VERSION"),
    "protocol_version": ("ashare_model.evaluation", "PROTOCOL_VERSION"),
    "reward_version": ("ashare_model.reward", "REWARD_VERSION"),
    "search_contract_version": ("ashare_model.search_contract", "SEARCH_CONTRACT_VERSION"),
    "semantic_cache_version": ("ashare_model.semantic_cache", "SEMANTIC_CACHE_VERSION"),
    # P9 §9: the factor-computation semantics (sparse-safe event
    # standardization + the P9 feature constructions) carry their own
    # version so a factor-tensor semantic change is never invisible to
    # artifact classification.
    "factor_compute_version": ("ashare_model.factors", "FACTOR_COMPUTE_VERSION"),
    "execution_spec_version": ("ashare_portfolio.execution_spec", "EXECUTION_SPEC_VERSION"),
    "portfolio_constructor_version": (
        "ashare_portfolio.constructor",
        "PORTFOLIO_CONSTRUCTOR_VERSION",
    ),
    "rebalance_policy_version": ("ashare_portfolio.rebalance", "REBALANCE_POLICY_VERSION"),
}


def new_run_id() -> str:
    """Unique per-execution run identity (random UUID4, not content-derived)."""

    return uuid4().hex


LIFECYCLE_CONTRACT_DOC = "docs/p8_research_lifecycle_contract.md"


def lifecycle_contract_hash(repo_root: Path | None = None) -> str:
    """SHA-256 of the governing pre-registration contract document."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    path = root / LIFECYCLE_CONTRACT_DOC
    if not path.is_file():
        raise ValueError(f"missing lifecycle contract document: {path}")
    return sha256_hex(path.read_text(encoding="utf-8"))


def resolve_runtime_thresholds() -> dict[str, float]:
    """The resolved preregistered threshold set (single authorities)."""

    import dataclasses

    from ashare_data.gates import MIN_ELIGIBLE_DEFAULT
    from ashare_model.promotion import PromotionThresholds

    thresholds = {
        f"promotion_{name}": float(value)
        for name, value in dataclasses.asdict(PromotionThresholds()).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    thresholds["min_eligible"] = float(MIN_ELIGIBLE_DEFAULT)
    return thresholds


def resolve_runtime_runspec(
    *,
    dataset_id: str,
    data_cutoff: str,
    data_config,
    backtest_config,
    requested_budget: int,
    n_folds: int,
    research_domain: str = "unified",
    target: str = "forward_return",
    seeds: tuple[int, ...] | None = None,
    searcher: str | None = None,
    max_formula_len: int | None = None,
    repo_root: Path | None = None,
    require_clean_tree: bool = True,
) -> RunSpec:
    """Mechanical runtime pre-registration snapshot (P8-05 wiring).

    Derives the frozen RunSpec from the actual runtime configuration and
    the single-authority constants: dev/holdout windows from the data and
    backtest configs, execution/capital/fee block from the P3 provenance
    authority, thresholds from the promotion/gate authorities, and the
    governing contract hash from the lifecycle contract document. Formal
    entry points call this once per run and open their RunStore run with
    the result.
    """

    from ashare_portfolio.execution_spec import portfolio_config_provenance

    def _d8(value: str) -> str:
        text = str(value).replace("-", "")
        if re.fullmatch(_DATE_RE, text) is None:
            raise ValueError(f"invalid date {value!r} (expected YYYYMMDD-ish)")
        return text

    train_end = _d8(backtest_config.train_end_date)
    return build_runspec(
        dataset_id=dataset_id,
        data_cutoff=_d8(data_cutoff),
        research_domain=research_domain,
        frequency=str(backtest_config.rebalance_frequency),
        target=target,
        horizon=int(backtest_config.target_horizon),
        requested_budget=requested_budget,
        n_folds=int(n_folds),
        dev_window=WindowSpec(start=_d8(data_config.start_date), end=train_end),
        holdout_window=WindowSpec(
            start=train_end, end=_d8(backtest_config.end_date)
        ),
        preregistered_contract_hash=lifecycle_contract_hash(repo_root),
        resolved_thresholds=resolve_runtime_thresholds(),
        portfolio_config=portfolio_config_provenance(backtest_config),
        seeds=seeds,
        searcher=searcher,
        max_formula_len=max_formula_len,
        repo_root=repo_root,
        require_clean_tree=require_clean_tree,
    )


class UniversePolicySpec(BaseModel):
    """Resolved PIT eligibility policy (mirrors ``ashare_data.universe.UniversePolicy``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index_codes: tuple[str, ...] = Field(min_length=1)
    min_listed_sessions: int = Field(ge=0)
    membership_end_inclusive: bool


class WindowSpec(BaseModel):
    """An inclusive YYYYMMDD date window (dev / holdout / paper)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: str = Field(pattern=_DATE_RE)
    end: str = Field(pattern=_DATE_RE)

    @model_validator(mode="after")
    def _ordered(self) -> "WindowSpec":
        if self.start > self.end:
            raise ValueError(f"window start {self.start} is after end {self.end}")
        return self


class FoldSpec(BaseModel):
    """Resolved fold plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_folds: int = Field(ge=1)
    fold_test_days: int | None = Field(default=None, ge=1)


class RunSpec(BaseModel):
    """The frozen, strict pre-registration artifact of one research run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runspec_schema_version: Literal[1] = RUNSPEC_SCHEMA_VERSION

    # data identity
    dataset_id: str = Field(min_length=1)
    data_cutoff: str = Field(pattern=_DATE_RE)
    universe_policy: UniversePolicySpec

    # code version set (captured by the factory, never hand-invented)
    feature_registry_version: int
    data_tier_version: int
    research_domain_version: int
    grammar_version: int
    model_version: int
    protocol_version: str
    reward_version: str
    search_contract_version: int
    semantic_cache_version: int
    execution_spec_version: int
    portfolio_constructor_version: int
    rebalance_policy_version: int
    # P9 §9 (additive, optional so pre-P9 artifacts stay valid): semantics
    # of the factor tensor construction; the factory always populates it.
    factor_compute_version: int | None = None

    # research semantics
    research_domain: str = Field(min_length=1)
    frequency: str = Field(min_length=1)
    target: str = Field(min_length=1)
    horizon: int = Field(ge=1)

    # search plan
    searcher: str
    requested_budget: int = Field(gt=0)
    budget_accounting: Literal["unique_semantic_evaluations"]
    seeds: tuple[int, ...] = Field(min_length=1)
    max_formula_len: int = Field(ge=2)

    # folds and windows
    folds: FoldSpec
    dev_window: WindowSpec
    holdout_window: WindowSpec | None = None
    paper_window: WindowSpec | None = None

    # execution / capital / complete fee block (P3 provenance authority)
    portfolio_config: dict[str, Any]

    # preregistration
    preregistered_contract_hash: str
    resolved_thresholds: dict[str, float]

    # run provenance
    git_commit: str
    dependency_lock_hash: str

    @model_validator(mode="after")
    def _validate_semantics(self) -> "RunSpec":
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique paired seeds")
        if not self.resolved_thresholds:
            raise ValueError("resolved_thresholds must be a non-empty preregistered mapping")
        for name, value in self.resolved_thresholds.items():
            if not name or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"resolved_thresholds[{name!r}] must be a finite number"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"resolved_thresholds[{name!r}] must be a finite number"
                )
        from ashare_portfolio.execution_spec import validate_portfolio_config_provenance

        portfolio = validate_portfolio_config_provenance(self.portfolio_config)
        if (
            portfolio["rebalance_frequency"] != self.frequency
            or portfolio["target_horizon"] != self.horizon
        ):
            raise ValueError(
                "portfolio_config rebalance block "
                f"({portfolio['rebalance_frequency']!r}, horizon "
                f"{portfolio['target_horizon']}) does not match the RunSpec "
                f"frequency/horizon ({self.frequency!r}, {self.horizon})"
            )
        return self

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe deterministic payload (the spec_id hash input)."""

        return self.model_dump(mode="json")

    @property
    def spec_id(self) -> str:
        """Strict content identity of the frozen spec (order/path independent)."""

        return content_hash(_SPEC_ID_KIND, self.to_payload())


def _capture_versions() -> dict[str, Any]:
    """Capture the current code version set from the owning modules."""

    values: dict[str, Any] = {}
    for field_name, (module_name, const_name) in _VERSION_IMPORTS.items():
        values[field_name] = getattr(importlib.import_module(module_name), const_name, None)
    return values


def _git_output(args: list[str], repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo_root), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed under {repo_root}: {proc.stderr.strip()}"
        )
    return proc.stdout


def _dependency_lock_hash(repo_root: Path) -> str:
    lock_paths = sorted(repo_root.glob("requirements*.txt"))
    if not lock_paths:
        raise ValueError(
            f"no dependency lock files (requirements*.txt) found under {repo_root}"
        )
    payload = "".join(
        f"{path.name}:{path.read_text(encoding='utf-8')}" for path in lock_paths
    )
    return sha256_hex(payload)


def build_runspec(
    *,
    dataset_id: str,
    data_cutoff: str,
    frequency: str,
    target: str,
    horizon: int,
    requested_budget: int,
    n_folds: int,
    dev_window: WindowSpec,
    preregistered_contract_hash: str,
    resolved_thresholds: dict[str, float],
    portfolio_config: dict[str, Any],
    research_domain: str = "unified",
    fold_test_days: int | None = None,
    holdout_window: WindowSpec | None = None,
    paper_window: WindowSpec | None = None,
    seeds: tuple[int, ...] | None = None,
    searcher: str | None = None,
    max_formula_len: int | None = None,
    universe_policy: UniversePolicySpec | None = None,
    repo_root: Path | None = None,
    require_clean_tree: bool = True,
) -> RunSpec:
    """Resolve every default explicitly and freeze the pre-registered spec.

    All semantic fields end up with a concrete resolved value — no field
    may enter the spec_id as an unresolved default. Provenance (git
    commit, dependency-lock hash) is captured from ``repo_root`` (default:
    this repository); a dirty *tracked* tree is rejected fail-closed while
    untracked files are tolerated.
    """

    from ashare_data.config import DataConfig, ModelConfig
    from ashare_model.admission import PAIR_SEEDS
    from ashare_model.research_domain import UNIFIED_DOMAIN_ID, resolve_domain
    from ashare_portfolio.rebalance import RebalancePolicy

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string")
    if not isinstance(research_domain, str) or not research_domain.strip():
        raise ValueError("research_domain must be a non-empty string")
    if research_domain != UNIFIED_DOMAIN_ID:
        # ``unified`` is the reserved compatible semantic handled explicitly
        # upstream (research_domain.resolve_domain refuses it by design).
        resolve_domain(research_domain)

    versions = _capture_versions()
    for name, value in versions.items():
        if value is None or isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f"unresolved version for {name}: {value!r}")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"unresolved version for {name}: {value!r}")

    searcher = searcher if searcher is not None else ModelConfig().searcher
    if searcher not in _KNOWN_SEARCHERS:
        raise ValueError(
            f"unknown searcher {searcher!r}; known backends: {list(_KNOWN_SEARCHERS)}"
        )

    if seeds is None:
        seeds = tuple(PAIR_SEEDS)
    seeds = tuple(int(seed) for seed in seeds)

    max_formula_len = (
        max_formula_len if max_formula_len is not None else ModelConfig().max_formula_len
    )
    if isinstance(max_formula_len, bool) or not isinstance(max_formula_len, int):
        raise ValueError("max_formula_len must be an integer >= 2 (EOS included)")
    if max_formula_len < 2:
        raise ValueError("max_formula_len must be an integer >= 2 (EOS included)")

    if isinstance(requested_budget, bool) or not isinstance(requested_budget, int):
        raise ValueError("requested_budget must be a positive integer")
    if requested_budget <= 0:
        raise ValueError("requested_budget must be a positive integer")

    if not resolved_thresholds:
        raise ValueError("resolved_thresholds must be a non-empty preregistered mapping")
    if not _HEX64_RE.match(str(preregistered_contract_hash)):
        raise ValueError(
            "preregistered_contract_hash must be a 64-char lowercase hex sha256 "
            "of the approved pre-registration contract"
        )

    # Reuse the single rebalance authority: illegal frequency/horizon
    # combinations (overlapping labels) are rejected here, not discovered
    # downstream.
    RebalancePolicy(str(frequency), int(horizon))

    if universe_policy is None:
        data_defaults = DataConfig()
        universe_policy = UniversePolicySpec(
            index_codes=tuple(data_defaults.index_codes),
            min_listed_sessions=int(data_defaults.min_listed_sessions),
            membership_end_inclusive=False,
        )

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    git_commit = _git_output(["rev-parse", "HEAD"], root).strip()
    if not _COMMIT_RE.match(git_commit):
        raise ValueError(f"unresolved git commit: {git_commit!r}")
    if require_clean_tree:
        dirty = [
            line
            for line in _git_output(["status", "--porcelain"], root).splitlines()
            if line and not line.startswith("??")
        ]
        if dirty:
            raise ValueError(
                "tracked tree is dirty; pre-registration requires a clean "
                f"tracked tree: {dirty[:5]}"
            )
    dependency_lock_hash = _dependency_lock_hash(root)

    return RunSpec(
        runspec_schema_version=RUNSPEC_SCHEMA_VERSION,
        dataset_id=dataset_id,
        data_cutoff=data_cutoff,
        universe_policy=universe_policy,
        **versions,
        research_domain=research_domain,
        frequency=str(frequency),
        target=str(target),
        horizon=int(horizon),
        searcher=searcher,
        requested_budget=int(requested_budget),
        budget_accounting="unique_semantic_evaluations",
        seeds=seeds,
        max_formula_len=int(max_formula_len),
        folds=FoldSpec(n_folds=int(n_folds), fold_test_days=fold_test_days),
        dev_window=dev_window,
        holdout_window=holdout_window,
        paper_window=paper_window,
        portfolio_config=portfolio_config,
        preregistered_contract_hash=str(preregistered_contract_hash),
        resolved_thresholds=dict(resolved_thresholds),
        git_commit=git_commit,
        dependency_lock_hash=dependency_lock_hash,
    )
