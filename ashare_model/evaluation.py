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

v24 (P6) adds the research-domain dimension
(``docs/p6_research_domain_contract.md``): a run declares one of
``short_price_volume`` / ``medium_cross_section`` / ``slow_fundamental``
(or the reserved compatible semantic ``unified``).  In domain mode the
out-of-domain factor rows are neutral (zeroed), every searcher samples
only the domain's feature tokens, baselines must lie inside the domain,
the (frequency, horizon) execution point must be a legal point of the
domain, and the window id carries the domain.  Artifacts record
``research_domain`` and ``research_domain_version``; rewards from
different domains are never comparable.
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

from .backtest import AshareBacktestEngine
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
from .eval_corrections import (
    deflated_sharpe,
    dsr_from_rows,
    expected_max_sr,
    max_t_from_rows,
    norm_cdf,
    norm_ppf,
    psr,
    selfcheck_rows,
)
from .eval_folds import (
    Fold,
    FoldData,
    epoch_slice,
    resolve_folds,
    search_window_id,
)
from .eval_metrics import (
    METRIC_KEYS,
    _tradable_ic_mask,
    aggregate_results,
    benchmark_row,
    evaluate_formula,
    evaluate_signal,
    stitch_oos_series,
    stitched_metrics,
    top_trial,
)
from .gp_search import run_gp_baseline
from .ir import FormulaSyntaxError, canonical_tokens
from .ledger import ExperimentLedger
from .regime import RegimeRegistry
from .research_domain import (
    RESEARCH_DOMAIN_VERSION,
    UNIFIED_DOMAIN_ID,
    feature_token_ids,
    restrict_tensor,
    resolve_domain,
)
from .reward import REWARD_VERSION
from .semantic_cache import (
    SEMANTIC_CACHE_VERSION,
    CalibrationSlice,
    make_calibration_execute,
)
from .search_contract import SearchResult
from .time_contract import FoldTimeContract
from .targets import causal_target_returns
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

PROTOCOL_VERSION = "24"


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
    domain_id: str = "unified",
    feature_ids: list[int] | None = None,
) -> AshareTrainer:
    """Trainer factory seam (tests inject a fake trainer through this)."""

    return AshareTrainer(
        data_config,
        model_config,
        backtest_config,
        loader=loader,
        reward_config=reward_config,
        domain_id=domain_id,
        feature_ids=feature_ids,
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
    domain_id: str = "unified",
    feature_ids: list[int] | None = None,
) -> dict:
    """Train one candidate on one fold with one seed, then score it OOS.

    The trainer never saves artifacts (the protocol must not clobber the
    working strategy files); training-side values are archived only.
    ``domain_id`` / ``feature_ids`` (P6 §4.2) restrict the search space.
    """

    trainer = _build_trainer(
        data_config,
        model_config,
        backtest_config,
        loader,
        reward_config,
        domain_id=domain_id,
        feature_ids=feature_ids,
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
    domain_id: str | None = None,
) -> SemanticBudgetEvaluator:
    """Shared semantic-budget evaluator for one search run (v18/v19).

    ``chunk`` defaults to the memory-bounded batch size (random baseline);
    sequential searchers (GP/TPE) pass ``chunk=1`` so every proposal is
    scored eagerly.  ``domain_id`` (P6 §4.3) enters the window id so
    domain scores never mix with other domains.
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
        window_id=search_window_id(fold, seed, domain_id=domain_id),
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
    feature_ids: list[int] | None = None,
    domain_id: str | None = None,
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
        seed,
        window.vocab,
        model_config.max_formula_len,
        target_count,
        feature_ids=feature_ids,
    )
    evaluator = _search_evaluator(
        window, loader, backtest_config, reward_config, fold, seed,
        target_count, "random_search", "random",
        domain_id=domain_id,
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
    feature_ids: list[int] | None = None,
    domain_id: str | None = None,
) -> dict:
    """Strongly-typed GP baseline (DEAP, T2-02) under the matched
    unique-semantic-evaluation budget; row shaped like a trained row.
    ``feature_ids`` (P6 §4.2) restricts the terminal set."""

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
        domain_id=domain_id,
    )
    result = run_gp_baseline(
        seed=seed,
        evaluator=evaluator,
        max_formula_len=model_config.max_formula_len,
        vocab=window.vocab,
        feature_ids=feature_ids,
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
    feature_ids: list[int] | None = None,
    domain_id: str | None = None,
) -> dict:
    """TPE baseline (Optuna, T2-02) under the matched unique-semantic-
    evaluation budget; row shaped like a trained row.  ``feature_ids``
    (P6 §4.2) restricts the open feature tokens."""

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
        domain_id=domain_id,
    )
    result = run_tpe_baseline(
        seed=seed,
        evaluator=evaluator,
        max_formula_len=model_config.max_formula_len,
        vocab=window.vocab,
        feature_ids=feature_ids,
    )
    return _search_row(base, result, loader, fold, backtest_config)


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
            # P6 §4.4: the research domain this campaign ran in and the
            # registry generation its defaults resolve from.
            "research_domain": proto_cfg.domain,
            "research_domain_version": RESEARCH_DOMAIN_VERSION,
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

    # P6: research-domain semantics.  In domain mode the execution point
    # must be a legal point of the domain, baselines must stay inside the
    # domain, the out-of-domain factor rows become neutral, and the
    # searchers sample only the domain's feature tokens.  ``unified``
    # (the default) is byte-identical to pre-P6 behavior.
    feature_ids: list[int] | None = None
    domain = None
    if proto_cfg.domain != UNIFIED_DOMAIN_ID:
        domain = resolve_domain(proto_cfg.domain)
        if not domain.is_legal_execution(
            proto_cfg.frequency, proto_cfg.horizon
        ):
            raise ValueError(
                f"protocol (frequency={proto_cfg.frequency!r}, "
                f"horizon={proto_cfg.horizon}) is not a legal execution "
                f"point of domain {domain.id!r} "
                "(docs/p6_research_domain_contract.md §1.2)"
            )
        out_of_domain = [
            name
            for name in proto_cfg.baseline_signals
            if name not in domain.features
        ]
        if out_of_domain:
            raise ValueError(
                f"baseline signals {out_of_domain} are outside domain "
                f"{domain.id!r} (docs/p6_research_domain_contract.md §3.2)"
            )
        feature_ids = feature_token_ids(proto_cfg.domain)
        if loader.factor_tensor is None:
            raise ValueError(
                "the loader must be loaded before a domain protocol run"
            )
        loader.factor_tensor = torch.tensor(
            restrict_tensor(loader.factor_tensor.numpy(), proto_cfg.domain),
            dtype=torch.float32,
        )
        logger.info(
            "protocol.domain domain={} features={} frequency={} horizon={} "
            "turnover_budget={} cost_weight={}",
            domain.id,
            len(domain.features),
            proto_cfg.frequency,
            proto_cfg.horizon,
            backtest_config.turnover_budget,
            reward_config.cost_weight if reward_config is not None else None,
        )
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
                        feature_ids=feature_ids,
                        domain_id=proto_cfg.domain,
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
                        feature_ids=feature_ids,
                        domain_id=proto_cfg.domain,
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
                        feature_ids=feature_ids,
                        domain_id=proto_cfg.domain,
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
                        domain_id=proto_cfg.domain,
                        feature_ids=feature_ids,
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
