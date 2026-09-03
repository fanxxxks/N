"""Search-backend runners for the evaluation protocol.

Extracted from ``evaluation.py`` (P7 Phase A4) by reason-to-change: this
module owns how candidates are *produced and scored inside a fold's
training window* — the single-factor baseline ladder, the trained row
(RL/GP/random via the trainer), the shared ``_SearchWindow`` bindings and
the random/GP/TPE budget-matched search rows.  It changes when a search
backend or its budget plumbing changes — not when fold contracts, metric
definitions, statistical corrections or artifact schemas change.

Monkeypatch compatibility (pre-split surface, verified by
``tests/test_eval_module_split.py`` and the existing suite):

* ``run_fold`` resolves ``_build_trainer`` **through the facade at call
  time** — tests inject fake trainers via
  ``monkeypatch.setattr(evaluation, "_build_trainer", ...)``;
* ``_search_evaluator`` reads ``PROTOCOL_VERSION`` through the facade at
  call time (the version constant's single home is the facade);
* ``CandidateScorer`` is imported directly: tests patch the *class*
  (``monkeypatch.setattr(evaluation.CandidateScorer, "score_many", ...)``),
  which is the same object regardless of the importing module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    ModelConfig,
    ProtocolConfig,
    RewardConfig,
    validate_baseline_signals,
)
from ashare_data.processor import open_to_open_returns

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
from .eval_folds import Fold, epoch_slice, search_window_id
from .eval_metrics import evaluate_formula, evaluate_signal
from .gp_search import run_gp_baseline
from .search_contract import SearchResult
from .semantic_cache import (
    SEMANTIC_CACHE_VERSION,
    CalibrationSlice,
    make_calibration_execute,
)
from .targets import causal_target_returns
from .time_contract import FoldTimeContract
from .tpe_search import run_tpe_baseline
from .train import AshareTrainer, validation_start, validation_windows
from .versions import PROTOCOL_VERSION
from .vm import StackVM
from .vocab import FEATURE_NAMES, FORMULA_VOCAB


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

    # Late binding through the facade: tests inject fake trainers via
    # ``monkeypatch.setattr(evaluation, "_build_trainer", ...)``.
    from ashare_model import evaluation as _facade  # noqa: PLC0415

    trainer = _facade._build_trainer(
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

    # PROTOCOL_VERSION binds its single home ashare_model.versions at
    # module level (IP-07a/IP-15): no late binding, nothing patches it.
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
