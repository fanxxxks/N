"""Walk-forward evaluation protocol: the measurement system for formula quality.

The protocol exists so that "is this formula good" is answered by a fixed,
versioned measurement instead of by whichever number a run happened to print.
Contract:

* Folds are anchored to absolute dates (never proportions) so they stay
  stable as the database grows; each fold trains up to ``train_end`` and is
  scored out-of-sample on (``train_end``, ``test_end``].
* Multiple seeds train independent formulas per fold; every per-fold,
  per-seed row is kept raw in the result so later analysis can drill down.
* Adjudication metrics are **reward-independent**: they come from the full
  backtest engine (net returns, Sharpe/Sortino, drawdown, turnover) and the
  shared rank-IC code path.  ``val_reward`` is archived for provenance only
  and never used to rank candidates, so results remain comparable across
  ``ashare_model.reward.REWARD_VERSION`` generations.
* Signals are traded in their **learned direction**: a negative-IC signal is
  flipped (direction decided on the training window for baselines, on the
  validation tail for trained formulas) before the long-only top-n engine
  consumes it, so negative-IC factors are never mechanically traded
  backwards.
* A **random-search baseline** (uniform sampling over structurally valid
  formulas, scored with the same reward path) joins every fold: it
  separates "the RL search is ineffective" from "the reward is
  uninformative".
* Candidate multiplicity is corrected with Deflated Sharpe and a
  studentized max-t block bootstrap (single trial matrix for both).

:data:`PROTOCOL_VERSION` is bumped on every semantic change to this module's
measurement behavior and is recorded in protocol artifacts and experiment
archives, exactly like :data:`ashare_model.reward.REWARD_VERSION`.

v5 changes the scored signal for formula candidates: the VM now returns
every formula signal cross-sectionally z-scored per date (terminal
standardization), and the training-side reward that selects candidates
consumes the backtest engine's tradability masks.  Baselines (raw factors,
already winsorized + robustly standardized) and the benchmark are
unchanged.

v6 grows the formula search space: four cross-sectional operators
(CS_RANK / CS_ZSCORE / CS_DEMEAN / CS_NEUTRALIZE) and enumerated windows
(5/10/60 for MA, STD, TS_RANK, CORR, DOWNVOL; DELTA10/20) join the
vocabulary.  Scoring semantics are unchanged, but the candidate pool is
drawn from a strictly larger operator set, so artifacts record the new
version for comparability.

v7 makes candidate scoring/selection common to RL, random search and bare
factors, and resolves every fold through the inclusive-anchor t+2 contract.

v8 applies the PIT universe mask to every measurement: the backtest
engine selects only signal-date AND entry-date eligible stocks, the
default equal-weight benchmark averages only those cells (requiring a
valid target), the benchmark row reuses the engine's benchmark path, and
the protocol row IC is computed over signal-date eligible cells with the
unified rank-IC implementation.  Artifacts record the applied universe
policy fields (no data hash, no lineage).

v9 makes the JUMP operator causal (a trailing 60-session baseline instead
of the full-timeline mean/std look-ahead; see
``ashare_model.reward.REWARD_VERSION`` v8).  The scoring machinery is
unchanged, but every formula containing JUMP now produces a different
signal, so prior protocol results are not comparable for those candidates.

v10 aligns the rolling CAPM factors' market window to each stock's own
valid sessions (see ``ashare_model.reward.REWARD_VERSION`` v9): stocks
with suspension gaps no longer regress on market returns from sessions
they did not trade, so BETA_60/IVOL_60/RSQ_60 and every candidate built
on them changes value.

v11 isolates the training signal from the selection data (see
``ashare_model.reward.REWARD_VERSION`` v10): every candidate-scoring
caller (trainer, random search, baselines) scores its primary window on
the in-sample head that stops where the validation tail begins, and the
rows rename ``full_window_*`` to ``train_*`` accordingly.

v12 adds the tradable-IC diagnostic to every scored row:
``ic_mean_tradable`` / ``icir_tradable`` recompute the rank IC excluding
stocks the engine could not buy on the entry day (suspension / one-word
limit-up opens); the benchmark row carries neutral placeholders.  The
primary ``ic_mean`` / ``icir`` semantics are unchanged.

v13 binds every artifact to the immutable dataset manifest (T1-01):
``dataset_id`` (the content-addressed Merkle root over partition hashes,
see :mod:`ashare_data.manifest`) is recorded in the artifact, and trial
rows loaded from prior artifacts whose dataset_id differs from the current
database are rejected explicitly (:class:`DatasetIdMismatch`) instead of
being silently mixed into the DS/max-t correction pool.  Legacy artifacts
without a dataset_id are accepted as pre-T1-01 (recorded as ``null``).

v14 consumes the T1-02 no-signal weight contract from the shared engine:
under-filled days keep cash (no upward renormalization, so
``single_weight_cap`` is a hard ceiling on every OOS measurement),
dispersion-less cross-sections are never rebalanced, and selection ties
resolve by stable stock identifiers — so every protocol measurement is
invariant under stock-row permutation.

v15 consumes the T1-03 robust measurement contract: candidate scoring
applies the hard signal-quality gates (valid-IC days, effective stock
count vs ``ic_min_stocks``, coverage, activity, sign stability,
validation-window lower quartile), the reward's IC term is the
effective-n shrunk (HAC) ICIR, and complexity is billed from the AST with
a hard ``max_complexity`` ceiling — degenerate and pathological formulas
are rejected before any OOS row is produced.

v16 consumes the T1-04 alignment and selection contract: the reward
basket selects on signal-date AND entry-date PIT eligibility exactly like
the engine; every scored row records the portfolio objectives (active IR,
risk exposure, average turnover, capacity utilization) and the capacity
gate (``capacity_above_maximum``) rejects illiquid positions; best-
formula selection uses constrained/Pareto ranking on those objectives
with IC as the auxiliary tie-break, instead of the fragile scalar.

v17 makes the random-search baseline **budget-matched** (T1-05): by
default it receives the trained candidate's exact evaluation budget
(``tier.steps x tier.batch_size`` unique formula evaluations) per fold,
and artifacts record ``random_budget_matched`` / ``random_budget``, so
RL-vs-baseline comparisons are budget-fair.

v18 (T2-01) makes the **unique semantic formula evaluation** the budget
unit of every searcher: formulas are canonicalized (ADD/MUL child
ordering, double-NEG/identity/invalid-op elimination, structural hashing),
degenerate constant-producing formulas are rejected pre-evaluation, and
numerically equivalent formulas (same rank fingerprint on the fixed
calibration slice) are scored once.  Duplicates never consume budget,
so the trainer, the random baseline and every later searcher are billed
identically.  Artifacts record ``unique_semantic_evals`` and the semantic
cache version.

v19 (T2-03) completes the baseline ladder: strongly-typed GP (DEAP) and
TPE (Optuna) join the random-search baseline with the same matched
budget per fold (``gp_enabled`` / ``tpe_enabled`` config, both on by
default), all three rows run on the shared semantic-budget evaluator,
and every trained row records its actual ``unique_semantic_evals``.
P4's paired admission experiment uses these budget semantics but gives
every arm the same independent pair seed and requested budget; fixed
baseline seeds from the historical T2 artifact are not admissible.

v20 (T4-01) re-establishes the valid-experiment protocol.  The 2021-2026
history has been viewed repeatedly, so it is **development/validation
data** (see :mod:`ashare_model.regime`); the protocol is now a **nested
walk-forward**: the inner loop tunes formula / reward / hyperparameters
inside each fold's training window, the outer loop performs exactly one
algorithm evaluation per (fold, seed), and adjudication consumes the
**stitched** outer OOS returns — one trial = one (candidate, seed)
concatenated series (``stitch_oos_series``), and Sharpe / Deflated
Sharpe / max-t / top-trial all operate on that stitched matrix
(``stitched`` artifact block).  Every trial is automatically recorded in
the append-only experiment ledger (:mod:`ashare_model.ledger`) — failed
and crashed trials included; a run with never-closed trials is tainted,
never silently clean.  Protocol runs refuse folds that touch a strictly
locked holdout slice (:class:`ashare_model.regime.HoldoutViolation`),
and the artifact records the data regime in force.  Prior artifacts
remain readable: their raw rows stitch under the same rule when loaded
with ``--trials``, and the per-fold row-level ``aggregates`` are kept
for drill-down.

v21 (P2) records the free-data credibility tier of every measured
formula: each row, stitched trial and the top trial carry a ``data_tier``
block (``data_tier_version`` / ``max_tier`` / ``tiers_used``) resolved
from the formula's features via :mod:`ashare_model.data_tier`
(``docs/p2_data_tier_contract.md``).  Formula semantics are unchanged;
the artifact schema gains provenance fields, and the promotion gates use
them (Tier A default, Tier B separate comparison, Tier C never).

v22 (P3) makes ``frequency`` / ``horizon`` executable protocol fields.
Every fold and validation subwindow retains its slice of the global
rebalance calendar; research IC/quality consumes the sparse causal target
``open[t+1+horizon] / open[t+1] - 1``, while the portfolio curve separately
consumes adjacent-open daily returns.  Protocol artifacts record execution
spec v2, the portfolio-constructor version and the complete resolved
portfolio configuration; pre-v22 artifacts remain readable history but are
not current promotion evidence.

v23 (P4) makes GP/TPE/Random/RL return one versioned ``SearchResult`` with
truthful requested/consumed budgets, termination/stagnation and
per-evaluation best-so-far curves.  This changes comparison and artifact
semantics, not the candidate reward itself.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    FoldConfig,
    ModelConfig,
    ProtocolConfig,
    RewardConfig,
    TierConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_reward_config,
    validate_baseline_signals,
)
from ashare_data.gates import ProductionGateRunner
from ashare_data.manifest import DatasetIdMismatch, check_dataset_id
from ashare_data.processor import open_to_open_returns
from ashare_logging import export_log_txt, setup_run_logging

from .backtest import AshareBacktestEngine, equal_weight_benchmark_returns
from .baseline_harness import (
    SemanticBudgetEvaluator,
    canonical_form_pool,
)
from .candidates import (
    PARETO_OBJECTIVES,
    CandidateScorer,
    CandidateSelector,
    CandidateSpec,
    score_chunk_size,
)
from .data_loader import AshareDataLoader
from .data_tier import DATA_TIER_VERSION, formula_data_tier_report
from .diagnostics import rank_ic_stats
from .gp_search import run_gp_baseline
from .ir import FormulaSyntaxError, canonical_tokens
from .ledger import ExperimentLedger
from .regime import RegimeRegistry
from .reward import REWARD_VERSION
from .semantic_cache import (
    SEMANTIC_CACHE_VERSION,
    CalibrationSlice,
    make_calibration_execute,
)
from .search_contract import SearchResult
from .time_contract import FoldTimeContract
from .targets import causal_target_returns
from ashare_portfolio.rebalance import RebalancePolicy
from ashare_portfolio.execution_spec import execution_provenance
from .tpe_search import run_tpe_baseline
from .train import (
    AshareTrainer,
    sample_random_formulas,
    validation_start,
    validation_windows,
)
from .vm import StackVM
from .vocab import FEATURE_NAMES, FORMULA_VOCAB

PROTOCOL_VERSION = "23"

# Metrics aggregated across folds/seeds for every candidate.
METRIC_KEYS = (
    "total_return",
    "annual_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "average_turnover",
    "excess_return",
    "benchmark_return",
    "ic_mean",
    "ic_abs_mean",
    "icir",
)

@dataclass(frozen=True)
class Fold:
    """A walk-forward fold resolved against a concrete date axis."""

    contract: FoldTimeContract
    frequency: str = "daily"

    @property
    def policy(self) -> RebalancePolicy:
        return RebalancePolicy(self.frequency, self.contract.horizon)

    @property
    def train_end(self) -> str:
        return self.contract.train_end

    @property
    def test_end(self) -> str:
        return self.contract.test_end

    # Compatibility accessors for consumers of pre-v7 Fold. New code uses
    # the explicit contract fields so anchor, signal and price ends cannot be
    # confused.
    @property
    def train_end_idx(self) -> int:
        return self.contract.train_anchor_end_exclusive

    @property
    def test_end_idx(self) -> int:
        return self.contract.test_price_end


@dataclass(frozen=True)
class FoldData:
    """Price-context slice plus the contract that declares executable columns."""

    factors: np.ndarray
    raw: dict[str, np.ndarray]
    target: np.ndarray
    realized_ret: np.ndarray
    rebalance_mask: np.ndarray
    dates: list[str]
    universe_mask: np.ndarray
    contract: FoldTimeContract

    @property
    def signal_count(self) -> int:
        return self.contract.test_signal_count

    @property
    def local_signal_range(self) -> range:
        return range(self.signal_count)

    def __iter__(self):
        # Preserve the established four-value unpacking API while exposing
        # the contract to new callers as an explicit attribute.
        yield self.factors
        yield self.raw
        yield self.target
        yield self.dates


def resolve_folds(
    fold_cfgs: list[FoldConfig],
    dates: list[str],
    *,
    frequency: str = "daily",
    horizon: int = 1,
) -> list[Fold]:
    """Resolve fold configs to column indices and check data availability.

    Configured anchors are inclusive. Test data retains the exact
    ``1 + horizon`` price-context columns needed to exit its final executable
    signal, while neither train nor test scoring can observe a price beyond
    its anchor.
    """

    policy = RebalancePolicy(frequency, horizon)
    folds: list[Fold] = []
    for cfg in fold_cfgs:
        contract = FoldTimeContract.resolve(
            dates,
            train_end=cfg.train_end,
            test_end=cfg.test_end,
            horizon=policy.horizon,
        )
        if (
            contract.test_price_end == len(dates)
            and dates[-1].replace("-", "") < cfg.test_end.replace("-", "")
        ):
            logger.warning(
                f"fold {cfg.train_end} -> {cfg.test_end}: test_end is past the "
                f"data range; test window truncated at {dates[-1]}"
            )
        folds.append(Fold(contract, frequency=policy.frequency))
    return folds


def epoch_slice(
    loader: AshareDataLoader,
    fold: Fold,
) -> FoldData:
    """Factor stack, raw OHLCV cache, forward targets and dates of the test
    window.  Factor columns carry their own lookback, so slicing the test
    window loses no history (VM execution must still happen on the full
    tensor and be sliced afterwards).

    The sparse forward target is recomputed from the sliced opens using the
    fold's global schedule slice. Passing the pre-resolved mask prevents a
    5/10-day cadence from restarting at the fold boundary.
    """

    contract = fold.contract
    s0, s1 = contract.test_signal_start, contract.test_price_end
    if loader.universe_mask is None:
        raise ValueError(
            "loader carries no universe mask; production evaluation "
            "requires the PIT eligibility mask"
        )
    factors = loader.factor_tensor[:, :, s0:s1].numpy()
    raw = {k: v[:, s0:s1].numpy() for k, v in loader.raw_data_cache.items()}
    universe_mask = loader.universe_mask[:, s0:s1]
    rebalance_mask = fold.policy.rebalance_mask(loader.dates)[s0:s1]
    target = causal_target_returns(
        raw["open"],
        loader.dates[s0:s1],
        fold.policy,
        rebalance_mask=rebalance_mask,
    )
    target = loader.mask_by_universe(target, start=s0)
    realized_ret = open_to_open_returns(raw["open"])
    return FoldData(
        factors=factors,
        raw=raw,
        target=target,
        realized_ret=realized_ret,
        rebalance_mask=rebalance_mask,
        dates=loader.dates[s0:s1],
        universe_mask=universe_mask,
        contract=contract,
    )


def _tradable_ic_mask(
    universe_mask: np.ndarray,
    blocked_buy: np.ndarray,
    signal_count: int,
) -> np.ndarray:
    """Universe mask additionally excluding stocks the engine could not buy.

    A signal at column ``t`` is executed at ``t+1`` (the entry day), so the
    tradable IC reference for column ``t`` is the stock set that is
    signal-date eligible AND buyable at ``t+1``: ``universe_mask[:, t] &
    ~blocked_buy[:, t+1]``.  ``blocked_buy`` is the sliced ``[stock, date]``
    buy-block matrix (suspension / one-word limit-up), covering at least
    ``signal_count + 1`` columns.  Columns beyond the signal range keep the
    raw universe mask (they never enter an IC).  Pure helper so the entry-
    day alignment is unit-testable.
    """

    universe_mask = np.asarray(universe_mask, dtype=bool)
    blocked_buy = np.asarray(blocked_buy, dtype=bool)
    if blocked_buy.shape != universe_mask.shape:
        raise ValueError(
            f"blocked_buy shape {blocked_buy.shape} does not match "
            f"universe_mask shape {universe_mask.shape}"
        )
    if signal_count >= blocked_buy.shape[1]:
        raise ValueError(
            "blocked_buy must cover at least one entry column "
            f"(got {blocked_buy.shape[1]} columns for {signal_count} signals)"
        )
    tradable = universe_mask.copy()
    tradable[:, :signal_count] &= ~blocked_buy[:, 1 : signal_count + 1]
    return tradable


def evaluate_signal(
    signal: np.ndarray,
    loader: AshareDataLoader,
    fold: Fold,
    bt_cfg: BacktestConfig,
) -> dict:
    """Full-engine metrics plus rank IC for one signal on one test window.

    Single code path for trained formulas, single-factor baselines and the
    benchmark row: everything is scored by the same backtest engine with
    real costs, the blocked mask and the equal-weight benchmark.  Both IC
    variants are reported: the universe-masked rank IC and the tradable IC
    (additionally excluding stocks the engine could not buy on the entry
    day).
    """

    config_policy = RebalancePolicy.from_config(bt_cfg)
    if config_policy != fold.policy:
        raise ValueError(
            f"fold policy {fold.policy!r} does not match BacktestConfig "
            f"policy {config_policy!r}"
        )
    fold_data = epoch_slice(loader, fold)
    _, raw, target, dates = fold_data
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape != (len(loader.ts_codes), len(dates)):
        raise ValueError(
            f"signal shape {signal.shape} does not match "
            f"({len(loader.ts_codes)}, {len(dates)})"
        )
    result = AshareBacktestEngine(bt_cfg).run(
        signal,
        raw,
        loader.ts_codes,
        dates,
        signal_range=fold_data.local_signal_range,
        universe_mask=fold_data.universe_mask,
        rebalance_mask=fold_data.rebalance_mask,
    )
    m = result.metrics
    bench_total = (
        float(result.benchmark_equity[-1] - 1.0)
        if result.benchmark_equity
        else 0.0
    )
    excess = (
        (1.0 + m["total_return"]) / (1.0 + bench_total) - 1.0
        if bench_total > -1.0
        else 0.0
    )
    signal_count = fold_data.signal_count
    ic = rank_ic_stats(
        signal[None, :, :signal_count],
        target[:, :signal_count],
        dates[:signal_count],
        names=["formula"],
        eligible=fold_data.universe_mask[:, :signal_count],
    )["formula"]
    # Tradable IC: the same statistic restricted to stocks the engine
    # could actually buy on the entry day (suspension / one-word limit-up
    # opens excluded).  Presented alongside the primary IC so drift from
    # untradable cells is visible instead of silently folded into the
    # signal-quality estimate.
    blocked_buy, _ = loader.tradability_masks()
    tradable = _tradable_ic_mask(
        fold_data.universe_mask,
        blocked_buy[:, fold.contract.test_signal_start : fold.contract.test_price_end],
        signal_count,
    )
    ic_tradable = rank_ic_stats(
        signal[None, :, :signal_count],
        target[:, :signal_count],
        dates[:signal_count],
        names=["formula"],
        eligible=tradable[:, :signal_count],
    )["formula"]
    bench_daily: list[float] = []
    if result.benchmark_equity and len(result.benchmark_equity) >= 2:
        eq = result.benchmark_equity
        bench_daily = [float(eq[i + 1] / eq[i] - 1.0) for i in range(len(eq) - 1)]
    return {
        "n_dates": signal_count,
        "total_return": float(m["total_return"]),
        "annual_return": float(m["annual_return"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "max_drawdown": float(m["max_drawdown"]),
        "calmar": float(m["calmar"]),
        "average_turnover": float(m["average_turnover"]),
        "benchmark_return": bench_total,
        "excess_return": excess,
        "ic_mean": float(ic["ic_mean"]),
        "ic_abs_mean": float(ic["ic_abs_mean"]),
        "icir": float(ic["icir"]),
        "n_ic_dates": int(ic["n_dates"]),
        "ic_mean_tradable": float(ic_tradable["ic_mean"]),
        "icir_tradable": float(ic_tradable["icir"]),
        # Raw per-day series: kept in every row so the DS / max-t corrections
        # and later analysis never have to reconstruct them from aggregates.
        "daily_returns": [float(x) for x in result.daily_returns],
        "benchmark_daily_returns": bench_daily,
    }


def evaluate_formula(
    tokens: list[int],
    loader: AshareDataLoader,
    fold: Fold,
    bt_cfg: BacktestConfig,
    vm: StackVM | None = None,
    direction: int = 1,
) -> dict | None:
    """Execute a formula on the full factor history, slice the test window,
    and score it in its learned trade direction (``direction`` flips the
    signal; the direction is decided on data strictly before the test
    window by the callers).  Returns ``None`` for an invalid formula."""

    if loader.universe_mask is None:
        # load_data always builds the mask; this guard keeps the failure
        # mode explicit for callers that skipped loading.
        raise ValueError(
            "loader carries no universe mask; production evaluation "
            "requires the PIT eligibility mask"
        )
    vm = vm or StackVM(
        FORMULA_VOCAB,
        universe_mask=torch.tensor(loader.universe_mask, dtype=torch.bool),
    )
    vm.industry_codes = getattr(loader, "industry_codes", None)
    # Force the loader's full-window PIT mask even onto an injected VM: the
    # production evaluation path can never execute without the mask.
    vm.universe_mask = torch.tensor(loader.universe_mask, dtype=torch.bool)
    signal = vm.execute(list(tokens), loader.factor_tensor)
    if signal is None:
        return None
    contract = fold.contract
    sliced = signal.detach().cpu().numpy()[
        :, contract.test_signal_start : contract.test_price_end
    ]
    return evaluate_signal(float(direction) * sliced, loader, fold, bt_cfg)


def benchmark_row(
    loader: AshareDataLoader,
    fold: Fold,
) -> dict:
    """Equal-weight benchmark scored on the engine's exact reference path:
    the mean forward target over signal-date AND entry-date eligible cells
    (cost-free buy-and-hold reference, no turnover).  No longer a separate
    per-stock average: the single
    :func:`ashare_model.backtest.equal_weight_benchmark_returns` helper
    computes it, so the row always equals the engine's benchmark curve."""

    fold_data = epoch_slice(loader, fold)
    _, _, target, dates = fold_data
    daily = equal_weight_benchmark_returns(
        fold_data.realized_ret,
        list(fold_data.local_signal_range),
        fold_data.universe_mask,
    )
    equity = [1.0]
    for ret in daily:
        equity.append(equity[-1] * (1.0 + ret))
    m = AshareBacktestEngine._metrics(daily, equity)
    return {
        "candidate": "benchmark:equal_weight",
        "formula_text": "equal_weight",
        "formula": None,
        "fold_train_end": fold.train_end,
        "fold_test_end": fold.test_end,
        "seed": None,
        "val_reward": None,
        "final_avg_reward": None,
        "failed": False,
        "n_dates": fold_data.signal_count,
        "total_return": float(m.get("total_return", 0.0)),
        "annual_return": float(m.get("annual_return", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "sortino": float(m.get("sortino", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "calmar": float(m.get("calmar", 0.0)),
        "average_turnover": None,
        "benchmark_return": float(m.get("total_return", 0.0)),
        "excess_return": 0.0,
        "ic_mean": 0.0,
        "ic_abs_mean": 0.0,
        "icir": 0.0,
        "n_ic_dates": 0,
        "ic_mean_tradable": 0.0,
        "icir_tradable": 0.0,
        "daily_returns": [float(x) for x in daily],
        "benchmark_daily_returns": [float(x) for x in daily],
    }


def baseline_candidates(
    loader: AshareDataLoader,
    proto_cfg: ProtocolConfig,
    fold: Fold,
    bt_cfg: BacktestConfig,
    model_cfg: ModelConfig | None = None,
    reward_cfg: RewardConfig | None = None,
) -> list[dict]:
    """Single-factor baseline rows: the factor row itself as the signal,
    traded in its training-window direction (a negative-IC factor is
    flipped, so the OOS row measures the signal, not a mechanical
    long-the-top backtest of the wrong side)."""

    validate_baseline_signals(proto_cfg.baseline_signals, FEATURE_NAMES)
    model_cfg = model_cfg or ModelConfig()
    reward_cfg = reward_cfg or RewardConfig()
    contract = fold.contract
    factors, _, _, _ = epoch_slice(loader, fold)
    train_price_end = contract.train_label_end
    train_signal_end = contract.train_signal_end
    train_factors = loader.factor_tensor[:, :, :train_price_end].numpy()
    train_open = loader.raw_data_cache["open"][:, :train_price_end].numpy()
    full_rebalance_mask = fold.policy.rebalance_mask(loader.dates)
    train_rebalance_mask = full_rebalance_mask[:train_price_end]
    train_target = causal_target_returns(
        train_open,
        loader.dates[:train_price_end],
        fold.policy,
        rebalance_mask=train_rebalance_mask,
    )
    train_target = loader.mask_by_universe(train_target)
    train_realized_ret = open_to_open_returns(train_open)
    blocked_buy, blocked_sell = loader.tradability_masks()
    val_windows = validation_windows(
        train_signal_end,
        model_cfg,
        rebalance_mask=train_rebalance_mask,
    )
    scorer = CandidateScorer(
        bt_cfg,
        reward_cfg,
    )
    specs: list[CandidateSpec] = []
    train_signals: list[np.ndarray] = []
    indices: list[int] = []
    for name in proto_cfg.baseline_signals:
        idx = FEATURE_NAMES.index(name)
        indices.append(idx)
        specs.append(
            CandidateSpec(
                candidate_id=f"baseline:{name}",
                formula_text=name,
                source="baseline",
                tokens=(idx + 1,),
            )
        )
        train_signals.append(train_factors[idx])
    scores = scorer.score_many(
        specs,
        train_signals,
        train_target,
        val_windows,
        blocked_buy=blocked_buy[:, :train_price_end],
        blocked_sell=blocked_sell[:, :train_price_end],
        train_signal_range=(
            contract.train_signal_start,
            validation_start(train_signal_end, model_cfg),
        ),
        universe_mask=loader.universe_mask[:, :train_price_end],
        tie_break_keys=np.asarray(loader.ts_codes),
        adv=np.asarray(loader.dollar_volume())[:, :train_price_end],
        realized_ret=train_realized_ret,
        rebalance_mask=train_rebalance_mask,
    )
    # The selector is invoked even though the protocol reports every bare
    # factor; this keeps ranking/eligibility behavior on the same code path.
    CandidateSelector().select(scores, pareto_objectives=PARETO_OBJECTIVES)
    rows: list[dict] = []
    for name, idx, score in zip(proto_cfg.baseline_signals, indices, scores):
        direction = score.direction
        metrics = evaluate_signal(
            float(direction) * factors[idx], loader, fold, bt_cfg
        )
        rows.append(
            {
                "candidate": f"baseline:{name}",
                "formula_text": name,
                "formula": None,
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "val_reward": score.val_reward,
                "val_icir": score.val_icir,
                "train_reward": score.train_reward,
                "train_icir": score.train_icir,
                "complexity_penalty": score.complexity_penalty,
                "complexity_cost": score.complexity_cost,
                "active_ir": score.active_ir,
                "risk_exposure": score.risk_exposure,
                "average_turnover": score.average_turnover,
                "capacity_utilization": score.capacity_utilization,
                "eligible": score.eligible,
                "rejection_reasons": list(score.rejection_reasons),
                "final_avg_reward": None,
                "direction": direction,
                "failed": False,
                **metrics,
            }
        )
    return rows


def _build_trainer(
    data_config: DataConfig,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    loader: AshareDataLoader,
    reward_config: RewardConfig | None,
) -> AshareTrainer:
    """Trainer factory seam (tests inject a fake trainer through this)."""

    return AshareTrainer(
        data_config,
        model_config,
        backtest_config,
        loader=loader,
        reward_config=reward_config,
    )


def run_fold(
    loader: AshareDataLoader,
    data_config: DataConfig,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    tier,
    fold: Fold,
    seed: int,
) -> dict:
    """Train one candidate on one fold with one seed, then score it OOS.

    The trainer never saves artifacts (the protocol must not clobber the
    working strategy files); training-side values are archived only.
    """

    trainer = _build_trainer(
        data_config, model_config, backtest_config, loader, reward_config
    )
    if model_config.searcher == "rl":
        tokens = trainer.train(
            steps=tier.steps,
            batch_size=tier.batch_size,
            seed=seed,
            save_artifacts=False,
            train_end_date=fold.train_end,
        )
    else:
        # T2-03: the production searcher (gp / random per model.searcher)
        # replaces RL in the protocol's "trained" row; the row contract
        # (selection, direction, OOS metrics) is unchanged.
        tokens = trainer.train_search(
            searcher=model_config.searcher,
            steps=tier.steps,
            batch_size=tier.batch_size,
            seed=seed,
            save_artifacts=False,
            train_end_date=fold.train_end,
        )
    base = {
        "candidate": "trained",
        "fold_train_end": fold.train_end,
        "fold_test_end": fold.test_end,
        "seed": seed,
    }
    if tokens is None:
        selection = getattr(trainer, "selection_result", None)
        rejected = getattr(selection, "best_rejected", None)
        return {
            **base,
            "failed": True,
            "reason": "no eligible formula found",
            "best_rejected": rejected.to_dict() if rejected else None,
        }
    # T2-03: the trained row records its actual unique-semantic-evaluation
    # budget (v18 ledger), so baseline comparisons can match it exactly.
    base["unique_semantic_evals"] = getattr(
        getattr(trainer, "semantic_cache", None), "budget_used", None
    )
    # The trainer decides the trade direction on its validation tail
    # (strictly before the test window), so a negative-IC formula is
    # evaluated flipped, matching how it would actually be deployed.
    direction = int(getattr(trainer, "best_direction", 1))
    metrics = evaluate_formula(
        tokens, loader, fold, backtest_config, direction=direction
    )
    if metrics is None:
        return {**base, "failed": True, "reason": "formula invalid at eval time"}
    selected = getattr(getattr(trainer, "selection_result", None), "selected", None)
    return {
        **base,
        "failed": False,
        "formula_text": trainer.best_formula,
        "formula": list(tokens),
        "val_reward": float(getattr(trainer, "best_val_reward", trainer.best_reward)),
        "val_icir": float(selected.val_icir) if selected is not None else None,
        "train_reward": (
            float(selected.train_reward) if selected is not None else None
        ),
        "train_icir": (
            float(selected.train_icir) if selected is not None else None
        ),
        "complexity_cost": (
            float(selected.complexity_cost) if selected is not None else None
        ),
        "active_ir": (
            float(selected.active_ir) if selected is not None else None
        ),
        "risk_exposure": (
            float(selected.risk_exposure) if selected is not None else None
        ),
        "average_turnover": (
            float(selected.average_turnover) if selected is not None else None
        ),
        "capacity_utilization": (
            float(selected.capacity_utilization) if selected is not None else None
        ),
        "eligible": bool(selected.eligible) if selected is not None else True,
        "rejection_reasons": (
            list(selected.rejection_reasons) if selected is not None else []
        ),
        "direction": direction,
        "final_avg_reward": (
            float(trainer.history[-1]["avg_reward"]) if trainer.history else None
        ),
        **metrics,
    }


@dataclass
class _SearchWindow:
    """One fold's training-window context shared by every search baseline
    (random / GP / TPE): the VM bindings, masks, target, windows and the
    calibration fingerprint executor."""

    contract: FoldTimeContract
    train_price_end: int
    train_signal_end: int
    vocab: object
    vm: StackVM
    factors: torch.Tensor
    universe_mask: np.ndarray
    target: np.ndarray
    realized_ret: np.ndarray
    rebalance_mask: np.ndarray
    val_windows: list[tuple[int, int]]
    train_signal_range: tuple[int, int]
    blocked_buy: np.ndarray
    blocked_sell: np.ndarray
    execute: Callable[[tuple[int, ...]], np.ndarray | None]
    fingerprint_execute: Callable[[tuple[int, ...]], np.ndarray | None]
    tie_break_keys: np.ndarray
    adv: np.ndarray
    signal_bytes: int


def _build_search_window(
    loader: AshareDataLoader,
    model_config: ModelConfig,
    fold: Fold,
) -> _SearchWindow:
    """Bind the shared training-window context of a fold (the same slices
    the trainer uses), including the calibration fingerprint executor."""

    contract = fold.contract
    train_price_end = contract.train_label_end
    train_signal_end = contract.train_signal_end
    vocab = FORMULA_VOCAB
    # The VM runs on the compute device exactly like the trainer's loop
    # (CUDA when available); only the sliced windows cross back to numpy
    # for the reward path.  Device float32 arithmetic may differ by ~1e-7,
    # the same documented caveat as the trainer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factors = loader.factor_tensor[:, :, :train_price_end].to(device)
    industry_codes = getattr(loader, "industry_codes", None)
    universe_mask = loader.universe_mask
    vm = StackVM(
        vocab,
        industry_codes=(
            industry_codes[:, :train_price_end].to(device)
            if industry_codes is not None
            else None
        ),
        universe_mask=torch.tensor(
            universe_mask[:, :train_price_end], dtype=torch.bool, device=device
        ),
    )
    # The scorer gates every quality statistic to signal-date eligible
    # cells; the mask is sliced to the exact training window like the
    # signals and targets (no off-by-one with the val windows, which are
    # index ranges inside the same slice).
    train_universe_mask = universe_mask[:, :train_price_end]
    train_open = loader.raw_data_cache["open"][:, :train_price_end].numpy()
    full_rebalance_mask = fold.policy.rebalance_mask(loader.dates)
    rebalance_mask = full_rebalance_mask[:train_price_end]
    target = loader.mask_by_universe(
        causal_target_returns(
            train_open,
            loader.dates[:train_price_end],
            fold.policy,
            rebalance_mask=rebalance_mask,
        )
    )
    realized_ret = open_to_open_returns(train_open)
    val_windows = validation_windows(
        train_signal_end,
        model_config,
        rebalance_mask=rebalance_mask,
    )
    # Tradability masks shared by every sampled formula, sliced to the
    # training window like the signals (the same path the trainer uses).
    blocked_buy, blocked_sell = loader.tradability_masks()
    blocked_buy = blocked_buy[:, :train_price_end]
    blocked_sell = blocked_sell[:, :train_price_end]
    train_signal_range = (
        contract.train_signal_start,
        validation_start(train_signal_end, model_config),
    )

    def execute(tokens) -> np.ndarray | None:
        signal = vm.execute(list(tokens), factors)
        if signal is None:
            return None
        return signal.detach().cpu().numpy()

    fingerprint_execute = make_calibration_execute(
        vm,
        factors,
        universe_mask,
        industry_codes,
        CalibrationSlice.of(factors.shape[2]),
    )
    return _SearchWindow(
        contract=contract,
        train_price_end=train_price_end,
        train_signal_end=train_signal_end,
        vocab=vocab,
        vm=vm,
        factors=factors,
        universe_mask=train_universe_mask,
        target=target,
        realized_ret=realized_ret,
        rebalance_mask=rebalance_mask,
        val_windows=val_windows,
        train_signal_range=train_signal_range,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        execute=execute,
        fingerprint_execute=fingerprint_execute,
        tie_break_keys=np.asarray(loader.ts_codes),
        adv=np.asarray(loader.dollar_volume())[:, :train_price_end],
        signal_bytes=factors.shape[1] * factors.shape[2] * 8,
    )


def _search_evaluator(
    window: _SearchWindow,
    loader: AshareDataLoader,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    fold: Fold,
    seed: int,
    budget: int,
    source: str,
    candidate_prefix: str,
    chunk: int | None = None,
) -> SemanticBudgetEvaluator:
    """Shared semantic-budget evaluator for one search run (v18/v19).

    ``chunk`` defaults to the memory-bounded batch size (random baseline);
    sequential searchers (GP/TPE) pass ``chunk=1`` so every proposal is
    scored eagerly.
    """

    return SemanticBudgetEvaluator(
        target=window.target,
        realized_ret=window.realized_ret,
        rebalance_mask=window.rebalance_mask,
        universe_mask=window.universe_mask,
        backtest_config=backtest_config,
        reward_config=reward_config or RewardConfig(),
        val_windows=window.val_windows,
        train_signal_range=window.train_signal_range,
        budget=budget,
        execute=window.execute,
        fingerprint_execute=window.fingerprint_execute,
        dataset_id=loader.dataset_id,
        protocol_version=PROTOCOL_VERSION,
        window_id=(
            f"fold:{fold.train_end}:{fold.test_end}:"
            f"frequency:{fold.policy.frequency}:horizon:{fold.policy.horizon}:"
            f"seed:{seed}"
        ),
        tie_break_keys=window.tie_break_keys,
        adv=window.adv,
        blocked_buy=window.blocked_buy,
        blocked_sell=window.blocked_sell,
        source=source,
        candidate_prefix=candidate_prefix,
        chunk=score_chunk_size(window.signal_bytes) if chunk is None else chunk,
    )


def _search_failed_row(base: dict, reason: str, score=None) -> dict:
    """Failure row shaped exactly like a trained row (DS/max-t compatible)."""

    payload = score.to_dict() if score is not None else {}
    return {
        **base,
        "failed": True,
        "reason": reason,
        "formula_text": payload.get("formula_text"),
        "formula": payload.get("tokens"),
        "val_reward": payload.get("val_reward"),
        "val_icir": payload.get("val_icir"),
        "train_reward": payload.get("train_reward"),
        "train_icir": payload.get("train_icir"),
        "complexity_penalty": payload.get("complexity_penalty"),
        "complexity_cost": payload.get("complexity_cost"),
        "active_ir": payload.get("active_ir"),
        "risk_exposure": payload.get("risk_exposure"),
        "average_turnover": payload.get("average_turnover"),
        "capacity_utilization": payload.get("capacity_utilization"),
        "eligible": False,
        "rejection_reasons": payload.get("rejection_reasons", [reason]),
        "final_avg_reward": None,
        "direction": int(payload.get("direction", 1)),
        "best_rejected": payload or None,
    }


def _search_row(
    base: dict,
    result: SearchResult,
    loader: AshareDataLoader,
    fold: Fold,
    backtest_config: BacktestConfig,
) -> dict:
    """Shape one search result into a protocol row (like a trained row)."""

    base["search_contract_version"] = result.contract_version
    base["requested_budget"] = result.requested_budget
    base["consumed_budget"] = result.consumed_budget
    base["unique_semantic_evals"] = result.consumed_budget
    base["semantic_dedups"] = result.semantic_duplicates
    base["termination_reason"] = result.termination_reason
    base["stagnation_reason"] = result.stagnation_reason
    base["best_so_far"] = list(result.best_so_far)
    selected = result.selected
    if selected is None or selected.tokens is None:
        return _search_failed_row(
            base, "no eligible formula found", result.scores[-1] if result.scores else None
        )
    metrics = evaluate_formula(
        list(selected.tokens),
        loader,
        fold,
        backtest_config,
        direction=selected.direction,
    )
    if metrics is None:
        return _search_failed_row(base, "formula invalid at eval time", selected)
    return {
        **base,
        "failed": False,
        "formula_text": selected.formula_text,
        "formula": list(selected.tokens),
        "val_reward": selected.val_reward,
        "val_icir": selected.val_icir,
        "train_reward": selected.train_reward,
        "train_icir": selected.train_icir,
        "complexity_penalty": selected.complexity_penalty,
        "complexity_cost": selected.complexity_cost,
        "active_ir": selected.active_ir,
        "risk_exposure": selected.risk_exposure,
        "average_turnover": selected.average_turnover,
        "capacity_utilization": selected.capacity_utilization,
        "eligible": selected.eligible,
        "rejection_reasons": list(selected.rejection_reasons),
        "final_avg_reward": None,
        "direction": selected.direction,
        **metrics,
    }


def run_random_search(
    loader: AshareDataLoader,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    fold: Fold,
    n_samples: int,
    seed: int,
    budget: int | None = None,
) -> dict:
    """Uniform random-search baseline over structurally valid formulas.

    Samples formulas with the same legality rules the policy samples
    under, scores each on the training window with the shared reward path
    (validation reward = median over the same sub-windows the trainer
    uses), keeps the best by validation reward and evaluates it
    out-of-sample in its learned direction.  The row is shaped exactly
    like a trained row so aggregates and the DS/max-t corrections treat
    both searches identically.

    With ``budget`` (T1-05) the baseline is **budget-matched**: it scores
    exactly ``budget`` unique semantic formulas (duplicates never count),
    so the comparison against a trained candidate that evaluated
    ``steps x batch_size`` unique formulas is budget-fair.  The budget
    unit is the **unique semantic formula evaluation** (v18, T2-01):
    degenerate constant-producing formulas are rejected pre-evaluation,
    canonical duplicates are merged, and numerically equivalent formulas
    (same calibration fingerprint) are scored once — the exact accounting
    the trainer's semantic cache applies.
    """

    base = {
        "candidate": "random_search",
        "fold_train_end": fold.train_end,
        "fold_test_end": fold.test_end,
        "seed": seed,
        "n_samples": int(n_samples),
        "budget": int(budget) if budget is not None else None,
        "semantic_cache_version": SEMANTIC_CACHE_VERSION,
    }
    contract = fold.contract
    train_signal_end = contract.train_signal_end
    if train_signal_end <= 0 or n_samples <= 0:
        return _search_failed_row(base, "degenerate window or budget")

    target_count = int(budget) if budget is not None else int(n_samples)
    if target_count <= 0:
        return _search_failed_row(base, "degenerate window or budget")
    window = _build_search_window(loader, model_config, fold)
    formulas = canonical_form_pool(
        seed, window.vocab, model_config.max_formula_len, target_count
    )
    evaluator = _search_evaluator(
        window, loader, backtest_config, reward_config, fold, seed,
        target_count, "random_search", "random",
    )
    for key in formulas:
        evaluator.propose(key)
        if evaluator.budget_used >= target_count:
            break
    reason = (
        "budget_exhausted"
        if evaluator.budget_used >= target_count
        else "candidate_pool_exhausted"
    )
    return _search_row(
        base,
        evaluator.finish(
            backend="random",
            seed=seed,
            termination_reason=reason,
        ),
        loader,
        fold,
        backtest_config,
    )


def run_gp_search(
    loader: AshareDataLoader,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    fold: Fold,
    seed: int,
    budget: int,
) -> dict:
    """Strongly-typed GP baseline (DEAP, T2-02) under the matched
    unique-semantic-evaluation budget; row shaped like a trained row."""

    base = {
        "candidate": "gp_search",
        "fold_train_end": fold.train_end,
        "fold_test_end": fold.test_end,
        "seed": seed,
        "budget": int(budget),
        "semantic_cache_version": SEMANTIC_CACHE_VERSION,
    }
    if fold.contract.train_signal_end <= 0 or budget <= 0:
        return _search_failed_row(base, "degenerate window or budget")
    window = _build_search_window(loader, model_config, fold)
    evaluator = _search_evaluator(
        window, loader, backtest_config, reward_config, fold, seed,
        int(budget), "gp_search", "gp",
    )
    result = run_gp_baseline(
        seed=seed,
        evaluator=evaluator,
        max_formula_len=model_config.max_formula_len,
        vocab=window.vocab,
    )
    return _search_row(base, result, loader, fold, backtest_config)


def run_tpe_search(
    loader: AshareDataLoader,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    fold: Fold,
    seed: int,
    budget: int,
) -> dict:
    """TPE baseline (Optuna, T2-02) under the matched unique-semantic-
    evaluation budget; row shaped like a trained row."""

    base = {
        "candidate": "tpe_search",
        "fold_train_end": fold.train_end,
        "fold_test_end": fold.test_end,
        "seed": seed,
        "budget": int(budget),
        "semantic_cache_version": SEMANTIC_CACHE_VERSION,
    }
    if fold.contract.train_signal_end <= 0 or budget <= 0:
        return _search_failed_row(base, "degenerate window or budget")
    window = _build_search_window(loader, model_config, fold)
    evaluator = _search_evaluator(
        window, loader, backtest_config, reward_config, fold, seed,
        int(budget), "tpe_search", "tpe",
    )
    result = run_tpe_baseline(
        seed=seed,
        evaluator=evaluator,
        max_formula_len=model_config.max_formula_len,
        vocab=window.vocab,
    )
    return _search_row(base, result, loader, fold, backtest_config)


def aggregate_results(rows: list[dict]) -> dict:
    """Per-candidate medians/IQR across folds and seeds.

    Medians (not means): reward clipping and heavy tails make the mean a
    poor summary of a candidate's OOS behavior.
    """

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["candidate"], []).append(row)

    out: dict = {}
    for name, group in groups.items():
        entry: dict = {"n_rows": len(group), "metrics": {}}
        for key in METRIC_KEYS:
            values = [
                v
                for r in group
                for v in [r.get(key)]
                if v is not None and math.isfinite(float(v))
            ]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            entry["metrics"][key] = {
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": int(arr.size),
            }
        if group[0].get("formula_text"):
            entry["formula_text"] = group[0]["formula_text"]
        out[name] = entry
    return out


# --- stitched OOS trial matrix (T4-01, v20) ----------------------------------
#
# One trial = one (candidate, seed) stitched outer-OOS series.  The raw
# rows are the per-fold outer evaluations of the nested walk-forward;
# stitching concatenates their daily return series in chronological fold
# order **before** any statistic is computed, so Sharpe / DSR / max-t
# measure the algorithm's full OOS path, not a per-fold average of paths.
# A failed fold is recorded (``failed_folds``) and contributes no
# returns; a (candidate, seed) with no succeeded fold produces no trial.


def _algorithm_of(candidate: str) -> str:
    """Algorithm family of a row's candidate name (``baseline:X`` -> ``baseline``)."""

    return str(candidate).split(":", 1)[0]


def stitch_oos_series(rows: list[dict]) -> list[dict]:
    """Stitch the outer OOS returns of every (candidate, seed) across folds.

    Returns JSON-friendly trial dicts: concatenated daily / benchmark /
    excess series in chronological fold order, per-segment provenance,
    recorded failed folds, and fold-level portfolio statistics aggregated
    over the succeeded segments (mean turnover, max capacity utilization).
    Two succeeded rows for the same (candidate, seed, fold) raise
    ``ValueError``: a fold measured twice is an integrity error, not a
    stitchable observation.
    """

    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for row in rows:
        key = (row.get("candidate"), row.get("seed"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    trials: list[dict] = []
    for key in order:
        candidate, seed = key
        group = groups[key]
        succeeded = [
            r for r in group
            if not r.get("failed") and r.get("daily_returns")
        ]
        if not succeeded:
            continue
        failed = [r for r in group if r.get("failed")]

        def _fold_key(row: dict) -> tuple[str, str]:
            return (
                str(row.get("fold_train_end") or ""),
                str(row.get("fold_test_end") or ""),
            )

        seen: set[tuple[str, str]] = set()
        segments: list[dict] = []
        for row in sorted(succeeded, key=_fold_key):
            fk = _fold_key(row)
            if fk in seen:
                raise ValueError(
                    f"duplicate fold row for candidate={candidate} "
                    f"seed={seed} fold={fk}"
                )
            seen.add(fk)
            segments.append(row)

        daily: list[float] = []
        bench: list[float] = []
        for row in segments:
            d = [float(x) for x in row["daily_returns"]]
            b = row.get("benchmark_daily_returns")
            b = [float(x) for x in b] if b and len(b) == len(d) else [0.0] * len(d)
            daily.extend(d)
            bench.extend(b)
        excess = [a - b for a, b in zip(daily, bench)]
        turnovers = [
            float(r["average_turnover"])
            for r in segments
            if r.get("average_turnover") is not None
        ]
        caps = [
            float(r["capacity_utilization"])
            for r in segments
            if r.get("capacity_utilization") is not None
        ]
        last = segments[-1]
        trials.append(
            {
                "candidate": candidate,
                "seed": seed,
                "algorithm": _algorithm_of(candidate),
                "segments": [
                    {
                        "fold_train_end": r.get("fold_train_end"),
                        "fold_test_end": r.get("fold_test_end"),
                        "n_dates": r.get("n_dates"),
                    }
                    for r in segments
                ],
                "failed_folds": [
                    {
                        "fold_train_end": r.get("fold_train_end"),
                        "fold_test_end": r.get("fold_test_end"),
                    }
                    for r in failed
                ],
                "n_days": len(daily),
                "daily_returns": daily,
                "benchmark_daily_returns": bench,
                "excess_returns": excess,
                "formula_text": last.get("formula_text"),
                "fold_test_end": last.get("fold_test_end"),
                "average_turnover_mean": (
                    float(sum(turnovers) / len(turnovers)) if turnovers else None
                ),
                "capacity_utilization_max": float(max(caps)) if caps else None,
            }
        )
    return trials


def stitched_metrics(trial: dict) -> dict:
    """Full-engine metrics on one stitched trial's concatenated series."""

    daily = [float(x) for x in trial.get("daily_returns") or []]
    if not daily:
        return {"n_days": 0}
    equity = AshareBacktestEngine._equity_curve(daily)
    m = AshareBacktestEngine._metrics(daily, equity)
    bench = [float(x) for x in trial.get("benchmark_daily_returns") or []]
    bench_total = 0.0
    if bench:
        bench_equity = AshareBacktestEngine._equity_curve(bench)
        bench_total = float(max(bench_equity[-1] - 1.0, -1.0))
    total = float(m.get("total_return", 0.0))
    excess = (1.0 + total) / (1.0 + bench_total) - 1.0 if bench_total > -1.0 else 0.0
    return {
        "n_days": len(daily),
        "total_return": total,
        "annual_return": float(m.get("annual_return", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "sortino": float(m.get("sortino", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "calmar": float(m.get("calmar", 0.0)),
        "benchmark_return": bench_total,
        "excess_return": excess,
    }


def top_trial(rows: list[dict]) -> dict | None:
    """Highest-OOS-Sharpe **stitched** trial (v20).

    Deliberately ignores ``val_reward``: adjudication is
    reward-version-independent by construction.  Rows without daily
    return series produce no trial.
    """

    best: dict | None = None
    for trial in stitch_oos_series(rows):
        metrics = stitched_metrics(trial)
        if metrics["n_days"] < 2:
            continue
        candidate = {**trial, **metrics}
        if best is None or candidate["sharpe"] > best["sharpe"]:
            best = candidate
    return best


# --- multiple-testing corrections (Bailey & Lopez de Prado 2014) ------------
#
# Both corrections share the same trial matrix (one trial = one non-failed
# row with its raw OOS excess-return series), so their conclusions are
# directly comparable.  scipy is not a project dependency: the normal CDF
# uses math.erf and the inverse normal CDF uses Acklam's rational
# approximation plus one Halley refinement (|error| ~ 1e-9).

_SQRT2 = math.sqrt(2.0)
_EULER_MASCHERONI = 0.5772156649015329

# Acklam's inverse-normal-CDF coefficients.
_ACKLAM_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)


def _poly(coeffs, x: float) -> float:
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF for ``0 < p < 1`` (Acklam + Halley)."""

    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf undefined for p={p}")
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = _poly(_ACKLAM_C, q) / _poly(_ACKLAM_D + (1.0,), q)
    elif p <= 0.97575:
        q = p - 0.5
        r = q * q
        x = _poly(_ACKLAM_A, r) * q / _poly(_ACKLAM_B + (1.0,), r)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -_poly(_ACKLAM_C, q) / _poly(_ACKLAM_D + (1.0,), q)
    # One Halley refinement step against the true CDF.
    e = 0.5 * math.erfc(-x / _SQRT2) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF via ``erf`` (scalar; the only caller is PSR)."""

    return 0.5 * (1.0 + math.erf(float(x) / _SQRT2))


def psr(sr: float, sr_benchmark: float, t: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio (B&LdP Eq. 14, incl. skew/kurt terms)."""

    num = (sr - sr_benchmark) * math.sqrt(t - 1)
    den = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(norm_cdf(num / den))


def expected_max_sr(n_trials: int, t: int, skew: float, kurt: float) -> float:
    """Expected maximum SR over ``n_trials`` independent null trials
    (B&LdP Eq. 13 with the null benchmark SR of zero)."""

    sr0 = 0.0
    variance = (1.0 - skew * sr0 + (kurt - 1.0) / 4.0 * sr0 * sr0) / (t - 1)
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr0 + math.sqrt(variance) * (
        (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    )


def deflated_sharpe(sr_best, n_trials, t, skew, kurt) -> float:
    """PSR of the best trial against the multiplicity-corrected benchmark."""

    return psr(sr_best, expected_max_sr(n_trials, t, skew, kurt), t, skew, kurt)


def _trial_stats(excess: np.ndarray) -> tuple[float, float, float, int]:
    t = excess.size
    mean = float(excess.mean())
    std = float(excess.std(ddof=1)) if t > 1 else 0.0
    if std < 1e-12:
        return 0.0, 0.0, 3.0, t
    sr = mean / std
    skew = float(((excess - mean) ** 3).mean() / std**3)
    kurt = float(((excess - mean) ** 4).mean() / std**4)
    return sr, skew, kurt, t


def _stitched_pool(
    rows: list[dict], extra_trial_rows: list[dict] | None = None
) -> list[tuple[dict, np.ndarray]]:
    """The v20 trial matrix: (stitched trial, excess array) pairs.

    This run's rows and the extra rows from prior artifacts are stitched
    **separately** — a prior run's ``trained`` trial is a different trial
    from this run's, so they must never share a series.
    """

    pool: list[tuple[dict, np.ndarray]] = []
    for trial in stitch_oos_series(rows) + stitch_oos_series(extra_trial_rows or []):
        excess = np.asarray(trial["excess_returns"], dtype=np.float64)
        if excess.size >= 2:
            pool.append((trial, excess))
    return pool


def _dsr_from_pool(pool: list[tuple[dict, np.ndarray]]) -> dict | None:
    """Deflated Sharpe of the best stitched trial in the pool."""

    stats = []
    for trial, excess in pool:
        sr, skew, kurt, t = _trial_stats(excess)
        stats.append((trial, sr, skew, kurt, t))
    if not stats:
        return None
    best_trial, sr, skew, kurt, t = max(stats, key=lambda x: x[1])
    if len(stats) == 1:
        # No multiplicity: the corrected benchmark collapses to SR = 0.
        dsr = psr(sr, 0.0, t, skew, kurt)
    else:
        dsr = deflated_sharpe(sr, len(stats), t, skew, kurt)
    return {
        "dsr": dsr,
        "n_trials": len(stats),
        "sr_best": sr,
        "t_best": t,
        "skew_best": skew,
        "kurt_best": kurt,
        "best_candidate": best_trial.get("candidate"),
        "best_fold_test_end": best_trial.get("fold_test_end"),
        "best_seed": best_trial.get("seed"),
    }


def dsr_from_rows(
    rows: list[dict], extra_trial_rows: list[dict] | None = None
) -> dict | None:
    """Deflated Sharpe over the **stitched** trial matrix (v20).

    One trial = one (candidate, seed) stitched outer-OOS series;
    ``extra_trial_rows`` are raw rows of prior artifacts, stitched
    separately and joined to the multiplicity pool.
    """

    return _dsr_from_pool(_stitched_pool(rows, extra_trial_rows))


def _max_t_from_pool(
    pool: list[tuple[dict, np.ndarray]],
    n_perms: int = 5000,
    seed: int = 0,
    block_size: int = 10,
) -> dict | None:
    """Studentized max-t block bootstrap (White-style reality check) over
    the stitched trial matrix.

    Each trial's excess series is recentered to the zero-mean null; per
    permutation, circular blocks of ``block_size`` days are resampled per
    trial, the bootstrapped mean is studentized by the observed per-trial
    standard deviation, and the max over trials forms the null distribution
    of the max t-statistic.  (Plain sign-flipping is not used: for a max
    statistic it can never fall below p = 0.5 by construction.)
    """

    series = [excess for _, excess in pool]
    if not series:
        return None

    stds = np.asarray(
        [float(s.std(ddof=1)) for s in series], dtype=np.float64
    )
    centered = [s - float(s.mean()) for s in series]
    tstats = np.asarray(
        [
            math.sqrt(s.size) * (float(s.mean()) / std)
            if std > 1e-12
            else 0.0
            for s, std in zip(series, stds)
        ],
        dtype=np.float64,
    )
    observed = float(tstats.max())

    rng = np.random.default_rng(seed)
    block = min(max(int(block_size), 1), min(s.size for s in series))
    count = 0
    for _ in range(int(n_perms)):
        boot_means = np.empty(len(series), dtype=np.float64)
        for i, s in enumerate(centered):
            n = s.size
            n_blocks = int(math.ceil(n / block))
            starts = rng.integers(0, n, size=n_blocks)
            offsets = np.arange(block)
            idx = (starts[:, None] + offsets[None, :]).ravel() % n
            boot = s[idx[:n]]
            boot_means[i] = float(boot.mean())
        boot_t = np.sqrt(
            np.asarray([s.size for s in series], dtype=np.float64)
        ) * (boot_means / np.maximum(stds, 1e-12))
        if float(boot_t.max()) >= observed:
            count += 1

    p_value = float((count + 1) / (int(n_perms) + 1))
    return {
        "observed_max_t": observed,
        "p_value": p_value,
        "n_perms": int(n_perms),
        "block_size": block,
        "n_trials": len(series),
        "seed": seed,
        "significant_95": bool(p_value <= 0.05),
    }


def max_t_from_rows(
    rows: list[dict],
    n_perms: int = 5000,
    seed: int = 0,
    block_size: int = 10,
    extra_trial_rows: list[dict] | None = None,
) -> dict | None:
    """Studentized max-t over the **stitched** trial matrix (v20)."""

    return _max_t_from_pool(
        _stitched_pool(rows, extra_trial_rows),
        n_perms=n_perms,
        seed=seed,
        block_size=block_size,
    )


def selfcheck_rows(
    loader: AshareDataLoader,
    proto_cfg: ProtocolConfig,
    backtest_config: BacktestConfig,
    seed: int = 1234,
) -> list[dict]:
    """Pure-noise placeholder candidates: the protocol's dry-run acceptance.

    Noise signals are scored through the identical engine path; the DS and
    max-t corrections must then both report insignificance.  A deterministic
    RNG makes the self-check reproducible.
    """

    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for fold_cfg in proto_cfg.folds:
        fold = resolve_folds(
            [fold_cfg],
            loader.dates,
            frequency=proto_cfg.frequency,
            horizon=proto_cfg.horizon,
        )[0]
        _, _, _, dates = epoch_slice(loader, fold)
        signal = rng.normal(0.0, 1.0, size=(len(loader.ts_codes), len(dates)))
        metrics = evaluate_signal(signal, loader, fold, backtest_config)
        rows.append(
            {
                "candidate": "selfcheck:noise",
                "formula_text": "noise",
                "formula": None,
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "val_reward": None,
                "final_avg_reward": None,
                "failed": False,
                **metrics,
            }
        )
    return rows


def _sanitize(value):
    """Replace non-finite floats with ``None`` so results stay valid JSON."""

    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def universe_policy_payload(loader: AshareDataLoader) -> dict | None:
    """The universe policy actually applied to a run, for artifact
    provenance: the configured index codes, the listing-age rule and the
    membership-boundary convention, plus the degraded flag.  No data hash
    and no lineage are recorded."""

    policy = getattr(loader, "universe_policy", None)
    if policy is None:
        return None
    return {
        "index_codes": [str(code) for code in policy.index_codes],
        "min_listed_sessions": int(policy.min_listed_sessions),
        "membership_end_inclusive": bool(policy.membership_end_inclusive),
        "degraded": (
            bool(loader.universe_status.degraded)
            if loader.universe_status is not None
            else None
        ),
    }


def _regime_payload(regime, proto_cfg: ProtocolConfig) -> dict | None:
    """The data regime in force for the artifact (record-only): the dev
    cutoff, the policy, the locked slice (if any) and each fold's window
    classification."""

    if regime is None or regime.regime is None:
        return None
    r = regime.regime
    locked = r.locked_slice
    return {
        "declared_at": r.declared_at,
        "dev_cutoff": r.dev_cutoff,
        "policy": r.policy,
        "locked_slice": (
            {
                "start": locked.start,
                "end": locked.end,
                "dataset_id": locked.dataset_id,
                "locked_at": locked.locked_at,
                "note": locked.note,
            }
            if locked is not None
            else None
        ),
        "folds": [
            {
                "train_end": f.train_end,
                "test_end": f.test_end,
                "kind": regime.classify_window(f.train_end, f.test_end),
            }
            for f in proto_cfg.folds
        ],
    }


def _data_tier_block(formula, formula_text: str | None) -> dict | None:
    """Compact credibility-tier block for one formula (v21, P2-02).

    Resolves the formula's features to their A/B/C tiers via
    :func:`ashare_model.data_tier.formula_data_tier_report` (``formula``
    tokens first; bare baseline rows fall back to ``formula_text``).
    ``None`` when there is no traceable formula (e.g. ``equal_weight``).
    """

    report = formula_data_tier_report(tokens=formula, feature_name=formula_text)
    if report is None:
        return None
    return {
        "data_tier_version": report["data_tier_version"],
        "max_tier": report["max_tier"],
        "tiers_used": report["tiers_used"],
    }


def build_result(
    proto_cfg: ProtocolConfig,
    tier_name: str,
    tier,
    rows: list[dict],
    data_end_date: str | None = None,
    extra_trial_rows: list[dict] | None = None,
    max_t_perms: int = 5000,
    universe_policy: dict | None = None,
    dataset_id: str | None = None,
    random_budget_matched: bool | None = None,
    random_budget: int | None = None,
    ledger: dict | None = None,
    regime=None,
    backtest_config: BacktestConfig | None = None,
) -> dict:
    """Assemble the protocol artifact (schema contract, see module docstring).

    v20: the artifact's adjudication (``top_trial`` / ``dsr`` / ``max_t``)
    consumes the **stitched** trial matrix (one trial = one (candidate,
    seed) series); the raw per-fold rows stay in ``rows`` for drill-down,
    the stitched trials live in the ``stitched`` block, and ``ledger`` /
    ``data_regime`` record the trial-ledger and data-regime provenance of
    the run (``None`` when absent).

    ``extra_trial_rows`` are trial rows from earlier protocol artifacts
    (e.g. screening runs whose OOS trials must count towards the DS/max-t
    multiplicity correction); they are stitched separately — a prior run's
    trial is never merged into this run's series — and join the correction
    pool.  ``universe_policy`` records the PIT universe policy fields that
    produced the rows.  ``dataset_id`` binds the artifact to the immutable
    dataset manifest the rows were measured on (``None`` for legacy
    databases, recorded as ``null``).  ``random_budget_matched`` /
    ``random_budget`` record the T1-05 baseline budget actually used.
    v21 (P2): every row / stitched trial / top trial records its
    free-data credibility tier (``data_tier`` block, resolved from the
    formula's features); ``data_tier_version`` pins the mapping at the
    artifact level.
    """

    if backtest_config is None:
        backtest_config = BacktestConfig(
            rebalance_frequency=proto_cfg.frequency,
            target_horizon=proto_cfg.horizon,
        )
    artifact_provenance = execution_provenance(backtest_config)
    if (
        artifact_provenance["portfolio_config"]["rebalance_frequency"]
        != proto_cfg.frequency
        or artifact_provenance["portfolio_config"]["target_horizon"]
        != proto_cfg.horizon
    ):
        raise ValueError(
            "protocol frequency/horizon must match BacktestConfig execution "
            "provenance"
        )

    # P2-02: annotate rows that were produced before this schema (or by
    # callers outside the protocol) with their credibility tier, derived
    # from the recorded formula tokens / bare feature name.
    for row in rows:
        if row.get("data_tier") is None:
            row["data_tier"] = _data_tier_block(
                row.get("formula"), row.get("formula_text")
            )

    stitched = stitch_oos_series(rows)
    for trial in stitched:
        trial.update(stitched_metrics(trial))
        trial["data_tier"] = _data_tier_block(None, trial.get("formula_text"))
    top = top_trial(rows)
    if top is not None:
        top["data_tier"] = _data_tier_block(None, top.get("formula_text"))
    return _sanitize(
        {
            "protocol_version": PROTOCOL_VERSION,
            "data_tier_version": DATA_TIER_VERSION,
            "reward_version": REWARD_VERSION,
            **artifact_provenance,
            "dataset_id": dataset_id,
            "frequency": proto_cfg.frequency,
            "horizon": proto_cfg.horizon,
            "tier": tier_name,
            "steps": tier.steps,
            "batch_size": tier.batch_size,
            "seeds": list(proto_cfg.seeds),
            "random_samples": proto_cfg.random_samples,
            "random_seed": proto_cfg.random_seed,
            "random_budget_matched": random_budget_matched,
            "random_budget": random_budget,
            "folds": [
                {"train_end": f.train_end, "test_end": f.test_end}
                for f in proto_cfg.folds
            ],
            "baseline_signals": list(proto_cfg.baseline_signals),
            "data_end_date": data_end_date,
            "universe_policy": universe_policy,
            "data_regime": _regime_payload(regime, proto_cfg),
            "ledger": ledger,
            "n_candidates": len(rows),
            "rows": rows,
            "aggregates": aggregate_results(rows),
            "stitched": {
                "n_trials": len(stitched),
                "trials": stitched,
            },
            "top_trial": top,
            "dsr": dsr_from_rows(rows, extra_trial_rows=extra_trial_rows),
            "max_t": max_t_from_rows(
                rows, n_perms=max_t_perms, extra_trial_rows=extra_trial_rows
            ),
            "dsr_extra_trials": len(stitch_oos_series(extra_trial_rows or [])),
        }
    )


def _run_recorded(ledger, fn, *, algorithm: str, candidate: str, seed=None,
                  fold_train_end=None, fold_test_end=None):
    """Run ``fn`` inside one ledger trial (T4-01).

    The trial opens as ``running`` before the work and closes as
    ``succeeded`` or ``failed`` on both paths, so a crashed trial is
    recorded, never silently dropped.  The returned row(s) carry the
    ``trial_id`` back into the artifact.  Without a ledger the call runs
    unrecorded (legacy callers).
    """

    if ledger is None:
        return fn()
    with ledger.trial(
        algorithm=algorithm,
        candidate=candidate,
        seed=seed,
        fold_train_end=fold_train_end,
        fold_test_end=fold_test_end,
    ) as trial_id:
        result = fn()
    rows = result if isinstance(result, list) else [result]
    for row in rows:
        if isinstance(row, dict):
            row["trial_id"] = trial_id
    return result


def run_protocol(
    loader: AshareDataLoader,
    data_config: DataConfig,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    proto_cfg: ProtocolConfig,
    tier_name: str,
    fold_indices: list[int] | None = None,
    seeds: list[int] | None = None,
    extra_trial_rows: list[dict] | None = None,
    max_t_perms: int = 5000,
    ledger=None,
    regime=None,
) -> dict:
    """Run the full protocol: baselines + one trained candidate per
    (fold, seed), all scored by the shared engine path.

    T4-01 wiring: ``regime`` is checked before the first trial (a fold
    touching a strictly locked holdout slice raises
    :class:`~ashare_model.regime.HoldoutViolation`), and ``ledger``
    records every trial automatically.  The benchmark row is a reference
    curve, not a trial, and is never recorded.
    """

    if tier_name not in ("screening", "confirmation"):
        raise ValueError(f"unknown tier: {tier_name}")
    tier = getattr(proto_cfg, tier_name)
    validate_baseline_signals(proto_cfg.baseline_signals, FEATURE_NAMES)
    if fold_indices is not None:
        unknown = [i for i in fold_indices if not 0 <= i < len(proto_cfg.folds)]
        if unknown:
            raise ValueError(
                f"fold indices out of range: {unknown} "
                f"(config has {len(proto_cfg.folds)} folds)"
            )
        fold_cfgs = [proto_cfg.folds[i] for i in fold_indices]
    else:
        fold_cfgs = proto_cfg.folds
    seeds = proto_cfg.seeds if seeds is None else list(seeds)
    if regime is not None:
        regime.assert_folds_clear(fold_cfgs, dataset_id=loader.dataset_id)

    folds = resolve_folds(
        fold_cfgs,
        loader.dates,
        frequency=proto_cfg.frequency,
        horizon=proto_cfg.horizon,
    )
    rows: list[dict] = []
    for fold in folds:
        rows.append(benchmark_row(loader, fold))
        rows.extend(
            _run_recorded(
                ledger,
                lambda: baseline_candidates(
                    loader,
                    proto_cfg,
                    fold,
                    backtest_config,
                    model_config,
                    reward_config,
                ),
                algorithm="baseline",
                candidate="baseline",
                fold_train_end=fold.train_end,
                fold_test_end=fold.test_end,
            )
        )
        if proto_cfg.random_samples > 0:
            # T1-05: the baseline is budget-matched to the trained
            # candidate (steps x batch_size unique evaluations) unless
            # explicitly disabled.
            random_budget = (
                tier.steps * tier.batch_size
                if proto_cfg.random_match_budget
                else None
            )
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} random-search "
                f"baseline budget={random_budget or proto_cfg.random_samples} "
                f"seed={proto_cfg.random_seed}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_random_search(
                        loader,
                        model_config,
                        backtest_config,
                        reward_config,
                        fold,
                        proto_cfg.random_samples,
                        proto_cfg.random_seed,
                        budget=random_budget,
                    ),
                    algorithm="random_search",
                    candidate="random_search",
                    seed=proto_cfg.random_seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
        # T2-03 baseline ladder: GP (DEAP) and TPE (Optuna) join the
        # random baseline with the same matched budget per fold.
        matched_budget = (
            tier.steps * tier.batch_size
            if proto_cfg.random_match_budget
            else proto_cfg.random_samples
        )
        if proto_cfg.gp_enabled and matched_budget > 0:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} gp baseline "
                f"budget={matched_budget} seed={proto_cfg.gp_seed}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_gp_search(
                        loader,
                        model_config,
                        backtest_config,
                        reward_config,
                        fold,
                        seed=proto_cfg.gp_seed,
                        budget=matched_budget,
                    ),
                    algorithm="gp_search",
                    candidate="gp_search",
                    seed=proto_cfg.gp_seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
        if proto_cfg.tpe_enabled and matched_budget > 0:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} tpe baseline "
                f"budget={matched_budget} seed={proto_cfg.tpe_seed}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_tpe_search(
                        loader,
                        model_config,
                        backtest_config,
                        reward_config,
                        fold,
                        seed=proto_cfg.tpe_seed,
                        budget=matched_budget,
                    ),
                    algorithm="tpe_search",
                    candidate="tpe_search",
                    seed=proto_cfg.tpe_seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
        for seed in seeds:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} seed={seed} "
                f"tier={tier_name} steps={tier.steps} batch={tier.batch_size}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_fold(
                        loader,
                        data_config,
                        model_config,
                        backtest_config,
                        reward_config,
                        tier,
                        fold,
                        seed,
                    ),
                    algorithm="trained",
                    candidate="trained",
                    seed=seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
    ledger_payload = None
    if ledger is not None:
        open_trials = ledger.finalize()
        ledger_payload = {
            "path": str(ledger.path),
            "run_id": ledger.run_id,
            "tainted": bool(open_trials),
            "n_trials": len(list(ledger.trials_for())),
        }
    return build_result(
        proto_cfg,
        tier_name,
        tier,
        rows,
        data_end_date=loader.dates[-1],
        extra_trial_rows=extra_trial_rows,
        max_t_perms=max_t_perms,
        universe_policy=universe_policy_payload(loader),
        dataset_id=loader.dataset_id,
        random_budget_matched=(
            proto_cfg.random_match_budget and proto_cfg.random_samples > 0
        ),
        random_budget=(
            tier.steps * tier.batch_size
            if proto_cfg.random_match_budget and proto_cfg.random_samples > 0
            else proto_cfg.random_samples
        ),
        ledger=ledger_payload,
        regime=regime,
        backtest_config=backtest_config,
    )


def load_trial_rows(
    paths: str | None, expected_dataset_id: str | None = None
) -> list[dict]:
    """Trial rows from prior protocol artifacts (comma-separated paths).

    ``expected_dataset_id`` binds the loaded rows to the current database:
    an artifact whose recorded dataset_id differs is rejected with
    :class:`DatasetIdMismatch` (explicit rejection instead of silently
    mixing measurements from different data).  Artifacts without a
    dataset_id are pre-T1-01 legacy and pass as-is.
    """

    rows: list[dict] = []
    if not paths:
        return rows
    for path in paths.split(","):
        path = path.strip()
        if not path:
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        check_dataset_id(payload.get("dataset_id"), expected_dataset_id)
        rows.extend(payload.get("rows", []))
    return rows


def main(argv=None) -> int:
    setup_run_logging(run_name="evaluation")
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation protocol"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--tier", choices=["screening", "confirmation"], default="screening"
    )
    parser.add_argument("--output", default="data/protocol_result.json")
    parser.add_argument(
        "--folds", type=int, default=None, help="run only the first N folds"
    )
    parser.add_argument(
        "--seeds", default=None, help="comma-separated seed override"
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="score pure-noise placeholder candidates and report whether "
        "the DS/max-t corrections stay insignificant (dry-run acceptance)",
    )
    parser.add_argument(
        "--trials",
        default=None,
        help="comma-separated prior protocol_result.json paths whose trial "
        "rows join the DS/max-t multiplicity correction",
    )
    parser.add_argument(
        "--ledger",
        default="data/experiment_ledger.jsonl",
        help="append-only trial ledger path (T4-01; every trial is recorded "
        "automatically, failed trials included)",
    )
    parser.add_argument(
        "--regime",
        default="data/holdout_registry.json",
        help="data-regime registry path (T4-01; folds touching a locked "
        "holdout slice are refused before the first trial)",
    )
    parser.add_argument(
        "--no-random-search",
        action="store_true",
        help="skip the random-search baseline (protocol.random_samples=0)",
    )
    parser.add_argument(
        "--max-t-perms", type=int, default=5000, help="max-t permutation count"
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
    )
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(data_config, min_eligible=args.min_eligible).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        proto_cfg = make_protocol_config(raw)
        if args.no_random_search:
            proto_cfg.random_samples = 0

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        extra_trials = load_trial_rows(args.trials, loader.dataset_id)
        ledger = ExperimentLedger(root / args.ledger)
        regime = RegimeRegistry(root / args.regime)

        if args.selfcheck:
            # Noise measurement is a measurement too: it must not touch a
            # locked holdout slice.
            regime.assert_folds_clear(proto_cfg.folds, dataset_id=loader.dataset_id)
            rows: list[dict] = []
            for fold_cfg in proto_cfg.folds:
                rows.append(
                    _run_recorded(
                        ledger,
                        lambda fold_cfg=fold_cfg: selfcheck_rows(
                            loader,
                            ProtocolConfig(folds=[fold_cfg]),
                            backtest_config,
                        )[0],
                        algorithm="selfcheck",
                        candidate="selfcheck:noise",
                        fold_train_end=fold_cfg.train_end,
                        fold_test_end=fold_cfg.test_end,
                    )
                )
            open_trials = ledger.finalize()
            ledger_payload = {
                "path": str(ledger.path),
                "run_id": ledger.run_id,
                "tainted": bool(open_trials),
                "n_trials": len(list(ledger.trials_for())),
            }
            result = build_result(
                proto_cfg,
                "selfcheck",
                TierConfig(0, 0),
                rows,
                data_end_date=loader.dates[-1],
                extra_trial_rows=extra_trials,
                max_t_perms=args.max_t_perms,
                universe_policy=universe_policy_payload(loader),
                dataset_id=loader.dataset_id,
                ledger=ledger_payload,
                regime=regime,
                backtest_config=backtest_config,
            )
        else:
            fold_indices = list(range(args.folds)) if args.folds else None
            seeds = (
                [int(s) for s in args.seeds.split(",")] if args.seeds else None
            )
            result = run_protocol(
                loader,
                data_config,
                model_config,
                backtest_config,
                reward_config,
                proto_cfg,
                args.tier,
                fold_indices=fold_indices,
                seeds=seeds,
                extra_trial_rows=extra_trials,
                max_t_perms=args.max_t_perms,
                ledger=ledger,
                regime=regime,
            )
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Protocol result written to {out_path}")
        print(json.dumps(result["aggregates"], ensure_ascii=False, indent=2))
        stitched = result.get("stitched") or {}
        print(
            f"stitched trials: {stitched.get('n_trials', 0)} "
            f"(one per (candidate, seed), across all folds)"
        )
        if result["top_trial"]:
            top = result["top_trial"]
            print(
                f"top trial: sharpe={top['sharpe']:.3f} "
                f"candidate={top['candidate']} fold_test_end={top['fold_test_end']} "
                f"seed={top['seed']}"
            )
        if result["dsr"]:
            dsr = result["dsr"]
            print(
                f"dsr: {dsr['dsr']:.3f} (n_trials={dsr['n_trials']}, "
                f"best={dsr['best_candidate']} sr={dsr['sr_best']:.3f})"
            )
        if result["max_t"]:
            mt = result["max_t"]
            print(
                f"max_t: observed={mt['observed_max_t']:.3f} "
                f"p={mt['p_value']:.4f} significant_95={mt['significant_95']}"
            )
        if args.selfcheck and result["dsr"] and result["max_t"]:
            passed = (
                result["dsr"]["dsr"] < 0.95
                and not result["max_t"]["significant_95"]
            )
            print(f"selfcheck passed={passed}")
    finally:
        export_log_txt(run_name="evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
