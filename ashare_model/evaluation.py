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
calibration slice) are evaluated once.  Duplicates never consume budget,
so the trainer, the random baseline and every later searcher are billed
identically.  Artifacts record ``unique_semantic_evals`` and the semantic
cache version.

``frequency`` / ``horizon`` are record-only for now: no rebalance-calendar
mechanism exists yet (weekly / multi-period targets are deferred to a later
phase), but they are written into artifacts so future runs stay comparable.
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
from .candidates import (
    PARETO_OBJECTIVES,
    CandidateScorer,
    CandidateSelector,
    CandidateSpec,
    score_chunk_size,
)
from .data_loader import AshareDataLoader
from .diagnostics import rank_ic_stats
from .complexity import complexity_bill
from .ir import FormulaSyntaxError, canonical_ast, canonical_tokens
from .reward import REWARD_VERSION
from .semantic_cache import (
    SEMANTIC_CACHE_VERSION,
    CalibrationSlice,
    SemanticCache,
    make_calibration_execute,
)
from .time_contract import FoldTimeContract
from .train import (
    AshareTrainer,
    sample_random_formulas,
    validation_start,
    validation_windows,
)
from .vm import StackVM, formula_decode
from .vocab import FEATURE_NAMES, FORMULA_VOCAB

PROTOCOL_VERSION = "18"

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


def resolve_folds(fold_cfgs: list[FoldConfig], dates: list[str]) -> list[Fold]:
    """Resolve fold configs to column indices and check data availability.

    Configured anchors are inclusive. Test data retains the two price-context
    columns needed to exit its final executable signal, while neither train
    nor test scoring can observe a price beyond its anchor.
    """

    folds: list[Fold] = []
    for cfg in fold_cfgs:
        contract = FoldTimeContract.resolve(
            dates, train_end=cfg.train_end, test_end=cfg.test_end
        )
        if (
            contract.test_price_end == len(dates)
            and dates[-1].replace("-", "") < cfg.test_end.replace("-", "")
        ):
            logger.warning(
                f"fold {cfg.train_end} -> {cfg.test_end}: test_end is past the "
                f"data range; test window truncated at {dates[-1]}"
            )
        folds.append(Fold(contract))
    return folds


def epoch_slice(
    loader: AshareDataLoader,
    fold: Fold,
) -> FoldData:
    """Factor stack, raw OHLCV cache, forward targets and dates of the test
    window.  Factor columns carry their own lookback, so slicing the test
    window loses no history (VM execution must still happen on the full
    tensor and be sliced afterwards).

    The forward target is recomputed from the sliced open on the same code
    path the engine uses: the engine zeroes the final two signal columns of
    its input matrix, so scoring on the full-tensor target would attribute
    returns the engine deliberately drops.
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
    target = loader.mask_by_universe(open_to_open_returns(raw["open"]), start=s0)
    return FoldData(
        factors=factors,
        raw=raw,
        target=target,
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
        target, list(fold_data.local_signal_range), fold_data.universe_mask
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
    train_target = open_to_open_returns(train_open)
    train_target = loader.mask_by_universe(train_target)
    blocked_buy, blocked_sell = loader.tradability_masks()
    val_windows = validation_windows(train_signal_end, model_cfg)
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
    tokens = trainer.train(
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
    exactly ``budget`` unique formulas (duplicates never count), so the
    comparison against a trained candidate that evaluated
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

    def failed_row(reason: str, score=None) -> dict:
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

    reward_cfg = reward_config or RewardConfig()
    contract = fold.contract
    train_price_end = contract.train_label_end
    train_signal_end = contract.train_signal_end
    if train_signal_end <= 0 or n_samples <= 0:
        return failed_row("degenerate window or budget")

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
    target = open_to_open_returns(
        loader.raw_data_cache["open"][:, :train_price_end].numpy()
    )
    target = loader.mask_by_universe(target)
    val_windows = validation_windows(train_signal_end, model_config)
    # Tradability masks shared by every sampled formula, sliced to the
    # training window like the signals (the same path the trainer uses).
    blocked_buy, blocked_sell = loader.tradability_masks()
    blocked_buy = blocked_buy[:, :train_price_end]
    blocked_sell = blocked_sell[:, :train_price_end]
    scorer = CandidateScorer(
        backtest_config,
        reward_cfg,
    )
    selector = CandidateSelector()

    # T1-05 matched budget: score exactly ``budget`` unique semantic
    # formulas (duplicates never count against the budget, exactly like
    # the trainer's semantic cache).  Sample with headroom and dedupe
    # canonically: degenerate formulas are dropped and structurally
    # identical sequences collapse.
    target_count = int(budget) if budget is not None else int(n_samples)
    if target_count <= 0:
        return failed_row("degenerate window or budget")
    sampled = sample_random_formulas(
        seed, vocab, model_config.max_formula_len, target_count * 8
    )
    formulas: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for key in sampled:
        try:
            canonical = canonical_tokens(key, vocab)
        except FormulaSyntaxError:
            continue  # structurally invalid: never evaluated
        if canonical is None:
            continue  # degenerate (constant-producing): never evaluated
        ctuple = tuple(canonical)
        if ctuple in seen:
            continue
        seen.add(ctuple)
        formulas.append(ctuple)
        if len(formulas) >= target_count:
            break
    # T2-01 semantic budget: the cache carries the evaluation context and
    # dedups numerically equivalent formulas through the calibration
    # fingerprint; its budget counter is the unique-semantic-evaluation
    # ledger.
    cache = SemanticCache(
        dataset_id=loader.dataset_id,
        reward_version=REWARD_VERSION,
        protocol_version=PROTOCOL_VERSION,
        window_id=f"fold:{fold.train_end}:{fold.test_end}:seed:{seed}",
        cap=target_count,
    )
    fingerprint_execute = make_calibration_execute(
        vm,
        factors,
        universe_mask,
        industry_codes,
        CalibrationSlice.of(factors.shape[2]),
    )
    # Random-search formulas are scored in budget-aware chunks so the
    # float64 signal stack + both-direction copy stay memory-bounded
    # (the shared chunk helper, same budget as the trainer).
    signal_bytes = factors.shape[1] * factors.shape[2] * 8
    chunk = score_chunk_size(signal_bytes)
    scores = []
    n_semantic_dedups = 0
    for start in range(0, len(formulas), chunk):
        specs: list[CandidateSpec] = []
        signals: list[np.ndarray | None] = []
        formula_valid: list[bool] = []
        for key in formulas[start : start + chunk]:
            ckey = cache.key_for(key)
            if ckey is None:
                continue
            canonical = canonical_ast(key, vocab)
            bill = complexity_bill(canonical) if canonical is not None else None
            fingerprint = cache.fingerprint(key, fingerprint_execute, vocab)
            if fingerprint is not None:
                if cache.is_claimed(fingerprint, bill):
                    # Same semantic class already claimed this run: skip
                    # the duplicate evaluation entirely.
                    n_semantic_dedups += 1
                    continue
                score = cache.score_by_fingerprint(fingerprint, bill)
                if score is not None:
                    n_semantic_dedups += 1
                    cache.put(ckey, score, fingerprint, bill)
                    continue
            specs.append(
                CandidateSpec(
                    candidate_id="random:" + ",".join(str(token) for token in key),
                    formula_text=formula_decode(list(key), vocab),
                    source="random_search",
                    tokens=key,
                )
            )
            signal = vm.execute(list(key), factors)
            if signal is None:
                signals.append(None)
                formula_valid.append(False)
                continue
            signals.append(signal.detach().cpu().numpy())
            formula_valid.append(True)
            cache.put(ckey, None, fingerprint, bill)  # claim the semantic class
            if cache.budget_used >= target_count:
                break
        if not specs:
            continue
        scored = scorer.score_many(
            specs,
            signals,
            target,
            val_windows,
            blocked_buy=blocked_buy,
            blocked_sell=blocked_sell,
            formula_valid=formula_valid,
            train_signal_range=(
                contract.train_signal_start,
                validation_start(train_signal_end, model_config),
            ),
            universe_mask=train_universe_mask,
            tie_break_keys=np.asarray(loader.ts_codes),
            adv=np.asarray(loader.dollar_volume())[:, :train_price_end],
        )
        for key, score in zip(specs, scored):
            ckey = cache.key_for(key.tokens)
            if ckey is not None:
                cache.put(ckey, score, None)
        scores.extend(scored)
        if cache.budget_used >= target_count:
            break

    base["unique_semantic_evals"] = cache.budget_used
    base["semantic_dedups"] = n_semantic_dedups

    selection = selector.select(scores, pareto_objectives=PARETO_OBJECTIVES)
    selected = selection.selected
    if selected is None or selected.tokens is None:
        return failed_row("no eligible formula found", selection.best_rejected)

    metrics = evaluate_formula(
        list(selected.tokens),
        loader,
        fold,
        backtest_config,
        direction=selected.direction,
    )
    if metrics is None:
        return failed_row("formula invalid at eval time", selected)
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


def top_trial(rows: list[dict]) -> dict | None:
    """Highest-OOS-Sharpe trial.  Deliberately ignores ``val_reward``:
    adjudication is reward-version-independent by construction."""

    scored = [
        r
        for r in rows
        if not r.get("failed")
        and r.get("sharpe") is not None
        and math.isfinite(float(r["sharpe"]))
    ]
    if not scored:
        return None
    return max(scored, key=lambda r: float(r["sharpe"]))


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


def excess_series(row: dict) -> np.ndarray | None:
    """Raw OOS excess (strategy - benchmark) daily returns of a trial row."""

    if row.get("failed") or not row.get("daily_returns"):
        return None
    strat = np.asarray(row["daily_returns"], dtype=np.float64)
    bench = np.asarray(
        row.get("benchmark_daily_returns") or np.zeros_like(strat),
        dtype=np.float64,
    )
    if bench.shape != strat.shape:
        bench = np.zeros_like(strat)
    return strat - bench


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


def dsr_from_rows(rows: list[dict]) -> dict | None:
    """Deflated Sharpe of the best trial in the row pool."""

    trials: list[dict] = []
    for row in rows:
        excess = excess_series(row)
        if excess is None or excess.size < 2:
            continue
        sr, skew, kurt, t = _trial_stats(excess)
        trials.append(
            {
                "row": row,
                "sr": sr,
                "skew": skew,
                "kurt": kurt,
                "t": t,
            }
        )
    if not trials:
        return None
    best = max(trials, key=lambda x: x["sr"])
    if len(trials) == 1:
        # No multiplicity: the corrected benchmark collapses to SR = 0.
        dsr = psr(best["sr"], 0.0, best["t"], best["skew"], best["kurt"])
    else:
        dsr = deflated_sharpe(
            best["sr"], len(trials), best["t"], best["skew"], best["kurt"]
        )
    return {
        "dsr": dsr,
        "n_trials": len(trials),
        "sr_best": best["sr"],
        "t_best": best["t"],
        "skew_best": best["skew"],
        "kurt_best": best["kurt"],
        "best_candidate": best["row"].get("candidate"),
        "best_fold_test_end": best["row"].get("fold_test_end"),
        "best_seed": best["row"].get("seed"),
    }


def max_t_from_rows(
    rows: list[dict], n_perms: int = 5000, seed: int = 0, block_size: int = 10
) -> dict | None:
    """Studentized max-t block bootstrap (White-style reality check).

    Each trial's excess series is recentered to the zero-mean null; per
    permutation, circular blocks of ``block_size`` days are resampled per
    trial, the bootstrapped mean is studentized by the observed per-trial
    standard deviation, and the max over trials forms the null distribution
    of the max t-statistic.  Shares the exact trial matrix with
    :func:`dsr_from_rows`.  (Plain sign-flipping is not used: for a max
    statistic it can never fall below p = 0.5 by construction.)
    """

    series: list[np.ndarray] = []
    for row in rows:
        excess = excess_series(row)
        if excess is None or excess.size < 2:
            continue
        series.append(excess)
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
        fold = resolve_folds([fold_cfg], loader.dates)[0]
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
) -> dict:
    """Assemble the protocol artifact (schema contract, see module docstring).

    ``extra_trial_rows`` are trial rows from earlier protocol artifacts
    (e.g. screening runs whose OOS trials must count towards the DS/max-t
    multiplicity correction); they join the correction pool but are never
    merged into this run's own rows.  ``universe_policy`` records the PIT
    universe policy fields that produced the rows.  ``dataset_id`` binds
    the artifact to the immutable dataset manifest the rows were measured
    on (``None`` for legacy databases, recorded as ``null``).
    ``random_budget_matched`` / ``random_budget`` record the T1-05
    baseline budget actually used.
    """

    trial_pool = list(rows) + list(extra_trial_rows or [])
    return _sanitize(
        {
            "protocol_version": PROTOCOL_VERSION,
            "reward_version": REWARD_VERSION,
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
            "n_candidates": len(rows),
            "rows": rows,
            "aggregates": aggregate_results(rows),
            "top_trial": top_trial(rows),
            "dsr": dsr_from_rows(trial_pool),
            "max_t": max_t_from_rows(trial_pool, n_perms=max_t_perms),
            "dsr_extra_trials": len(extra_trial_rows or []),
        }
    )


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
) -> dict:
    """Run the full protocol: baselines + one trained candidate per
    (fold, seed), all scored by the shared engine path."""

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

    folds = resolve_folds(fold_cfgs, loader.dates)
    rows: list[dict] = []
    for fold in folds:
        rows.append(benchmark_row(loader, fold))
        rows.extend(
            baseline_candidates(
                loader,
                proto_cfg,
                fold,
                backtest_config,
                model_config,
                reward_config,
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
                run_random_search(
                    loader,
                    model_config,
                    backtest_config,
                    reward_config,
                    fold,
                    proto_cfg.random_samples,
                    proto_cfg.random_seed,
                    budget=random_budget,
                )
            )
        for seed in seeds:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} seed={seed} "
                f"tier={tier_name} steps={tier.steps} batch={tier.batch_size}"
            )
            rows.append(
                run_fold(
                    loader,
                    data_config,
                    model_config,
                    backtest_config,
                    reward_config,
                    tier,
                    fold,
                    seed,
                )
            )
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

        if args.selfcheck:
            rows = selfcheck_rows(loader, proto_cfg, backtest_config)
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
            )
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Protocol result written to {out_path}")
        print(json.dumps(result["aggregates"], ensure_ascii=False, indent=2))
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
