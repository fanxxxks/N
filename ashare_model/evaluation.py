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
from ashare_data.processor import open_to_open_returns
from ashare_logging import export_log_txt, setup_run_logging

from .backtest import AshareBacktestEngine
from .data_loader import AshareDataLoader, date_index
from .diagnostics import rank_ic_stats
from .reward import (
    REWARD_VERSION,
    batched_basket_rewards,
    signal_direction,
)
from .train import (
    AshareTrainer,
    sample_random_formulas,
    validation_windows,
)
from .vm import StackVM, formula_decode
from .vocab import FEATURE_NAMES, FORMULA_VOCAB

PROTOCOL_VERSION = "4"

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

# Minimum column counts for a fold to be meaningful.
MIN_TRAIN_COLS = 2
MIN_TEST_COLS = 3


@dataclass(frozen=True)
class Fold:
    """A walk-forward fold resolved against a concrete date axis."""

    train_end: str
    test_end: str
    train_end_idx: int
    test_end_idx: int


def resolve_folds(fold_cfgs: list[FoldConfig], dates: list[str]) -> list[Fold]:
    """Resolve fold configs to column indices and check data availability.

    The train window is ``dates[0 : train_end_idx)`` and the test window is
    ``dates[train_end_idx : test_end_idx)``; each window must hold at least
    its minimum column count, otherwise the fold is reported explicitly
    instead of producing silently degenerate metrics.
    """

    folds: list[Fold] = []
    for cfg in fold_cfgs:
        train_idx = date_index(dates, cfg.train_end)
        test_idx = date_index(dates, cfg.test_end)
        if train_idx >= test_idx:
            raise ValueError(
                f"fold {cfg.train_end} -> {cfg.test_end}: test window is empty "
                f"(data range ends at {dates[-1]})"
            )
        if train_idx < MIN_TRAIN_COLS:
            raise ValueError(
                f"fold {cfg.train_end} -> {cfg.test_end}: train window has "
                f"{train_idx} columns, need at least {MIN_TRAIN_COLS}"
            )
        if test_idx - train_idx < MIN_TEST_COLS:
            raise ValueError(
                f"fold {cfg.train_end} -> {cfg.test_end}: test window has "
                f"{test_idx - train_idx} columns, need at least {MIN_TEST_COLS}"
            )
        if test_idx == len(dates) and dates[-1] < cfg.test_end.replace("-", ""):
            logger.warning(
                f"fold {cfg.train_end} -> {cfg.test_end}: test_end is past the "
                f"data range; test window truncated at {dates[-1]}"
            )
        folds.append(Fold(cfg.train_end, cfg.test_end, train_idx, test_idx))
    return folds


def epoch_slice(
    loader: AshareDataLoader,
    fold: Fold,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, list[str]]:
    """Factor stack, raw OHLCV cache, forward targets and dates of the test
    window.  Factor columns carry their own lookback, so slicing the test
    window loses no history (VM execution must still happen on the full
    tensor and be sliced afterwards).

    The forward target is recomputed from the sliced open on the same code
    path the engine uses: the engine zeroes the final two signal columns of
    its input matrix, so scoring on the full-tensor target would attribute
    returns the engine deliberately drops.
    """

    s0, s1 = fold.train_end_idx, fold.test_end_idx
    factors = loader.factor_tensor[:, :, s0:s1].numpy()
    raw = {k: v[:, s0:s1].numpy() for k, v in loader.raw_data_cache.items()}
    target = open_to_open_returns(raw["open"])
    return factors, raw, target, loader.dates[s0:s1]


def evaluate_signal(
    signal: np.ndarray,
    loader: AshareDataLoader,
    fold: Fold,
    bt_cfg: BacktestConfig,
) -> dict:
    """Full-engine metrics plus rank IC for one signal on one test window.

    Single code path for trained formulas, single-factor baselines and the
    benchmark row: everything is scored by the same backtest engine with
    real costs, the blocked mask and the equal-weight benchmark.
    """

    _, raw, target, dates = epoch_slice(loader, fold)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape != (len(loader.ts_codes), len(dates)):
        raise ValueError(
            f"signal shape {signal.shape} does not match "
            f"({len(loader.ts_codes)}, {len(dates)})"
        )
    result = AshareBacktestEngine(bt_cfg).run(
        signal, raw, loader.ts_codes, dates, stock_names=loader.stock_names
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
    ic = rank_ic_stats(signal[None, :, :], target, dates, names=["formula"])[
        "formula"
    ]
    bench_daily: list[float] = []
    if result.benchmark_equity and len(result.benchmark_equity) >= 2:
        eq = result.benchmark_equity
        bench_daily = [float(eq[i + 1] / eq[i] - 1.0) for i in range(len(eq) - 1)]
    return {
        "n_dates": len(dates),
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

    vm = vm or StackVM(FORMULA_VOCAB)
    signal = vm.execute(list(tokens), loader.factor_tensor)
    if signal is None:
        return None
    sliced = signal.detach().cpu().numpy()[
        :, fold.train_end_idx : fold.test_end_idx
    ]
    return evaluate_signal(float(direction) * sliced, loader, fold, bt_cfg)


def benchmark_row(
    loader: AshareDataLoader,
    fold: Fold,
) -> dict:
    """Equal-weight benchmark scored on the same code path as the engine's
    reference curve (cost-free buy-and-hold reference, no turnover)."""

    _, _, target, dates = epoch_slice(loader, fold)
    daily = [float(np.mean(target[:, t])) for t in range(target.shape[1] - 1)]
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
        "n_dates": len(dates),
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
        "daily_returns": [float(x) for x in daily],
        "benchmark_daily_returns": [float(x) for x in daily],
    }


def baseline_candidates(
    loader: AshareDataLoader,
    proto_cfg: ProtocolConfig,
    fold: Fold,
    bt_cfg: BacktestConfig,
) -> list[dict]:
    """Single-factor baseline rows: the factor row itself as the signal,
    traded in its training-window direction (a negative-IC factor is
    flipped, so the OOS row measures the signal, not a mechanical
    long-the-top backtest of the wrong side)."""

    validate_baseline_signals(proto_cfg.baseline_signals, FEATURE_NAMES)
    factors, _, _, _ = epoch_slice(loader, fold)
    train_factors = loader.factor_tensor[:, :, : fold.train_end_idx].numpy()
    train_target = loader.target_ret[:, : fold.train_end_idx].numpy()
    rows: list[dict] = []
    for name in proto_cfg.baseline_signals:
        idx = FEATURE_NAMES.index(name)
        direction = signal_direction(
            train_factors[idx], train_target, min_stocks=10
        )
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
                "val_reward": None,
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
        return {**base, "failed": True, "reason": "no valid formula found"}
    # The trainer decides the trade direction on its validation tail
    # (strictly before the test window), so a negative-IC formula is
    # evaluated flipped, matching how it would actually be deployed.
    direction = int(getattr(trainer, "best_direction", 1))
    metrics = evaluate_formula(
        tokens, loader, fold, backtest_config, direction=direction
    )
    if metrics is None:
        return {**base, "failed": True, "reason": "formula invalid at eval time"}
    return {
        **base,
        "failed": False,
        "formula_text": trainer.best_formula,
        "formula": list(tokens),
        "val_reward": float(trainer.best_reward),
        "direction": direction,
        "final_avg_reward": (
            float(trainer.history[-1]["avg_reward"]) if trainer.history else None
        ),
        **metrics,
    }


# Random-search formulas are scored in chunks so the float64 signal stack
# stays memory-bounded (16 x [stocks, dates] x 8 bytes).
_RANDOM_SEARCH_CHUNK = 16


def run_random_search(
    loader: AshareDataLoader,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    fold: Fold,
    n_samples: int,
    seed: int,
) -> dict:
    """Uniform random-search baseline over structurally valid formulas.

    Samples ``n_samples`` formulas with the same legality rules the policy
    samples under, scores each on the training window with the shared
    reward path (validation reward = median over the same sub-windows the
    trainer uses), keeps the best by validation reward and evaluates it
    out-of-sample in its learned direction.  The row is shaped exactly like
    a trained row so aggregates and the DS/max-t corrections treat both
    searches identically.
    """

    base = {
        "candidate": "random_search",
        "fold_train_end": fold.train_end,
        "fold_test_end": fold.test_end,
        "seed": seed,
        "n_samples": int(n_samples),
    }
    reward_cfg = reward_config or RewardConfig()
    train_end = fold.train_end_idx
    if train_end <= 2 or n_samples <= 0:
        return {**base, "failed": True, "reason": "degenerate window or budget"}

    vocab = FORMULA_VOCAB
    vm = StackVM(vocab)
    # The VM runs on the compute device exactly like the trainer's loop
    # (CUDA when available); only the sliced windows cross back to numpy
    # for the reward path.  Device float32 arithmetic may differ by ~1e-7,
    # the same documented caveat as the trainer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factors = loader.factor_tensor.to(device)
    target = loader.target_ret[:, :train_end].numpy()
    val_windows = validation_windows(train_end, model_config)

    formulas = sample_random_formulas(
        seed, vocab, model_config.max_formula_len, n_samples
    )
    best_key: tuple[int, ...] | None = None
    best_val: float = -float("inf")
    seen: set[tuple[int, ...]] = set()
    for start in range(0, len(formulas), _RANDOM_SEARCH_CHUNK):
        keys: list[tuple[int, ...]] = []
        signals: list[np.ndarray] = []
        for key in formulas[start : start + _RANDOM_SEARCH_CHUNK]:
            if key in seen:
                continue
            seen.add(key)
            signal = vm.execute(list(key), factors)
            if signal is None:
                continue
            sliced = signal.detach().cpu().numpy()[:, :train_end]
            if np.nanstd(sliced) < 1e-4:
                continue
            keys.append(key)
            signals.append(sliced)
        if not signals:
            continue
        _, val_rewards, _, _ = batched_basket_rewards(
            np.stack(signals), target, backtest_config, reward_cfg, val_windows
        )
        for key, val_reward in zip(keys, val_rewards):
            if float(val_reward) > best_val:
                best_val = float(val_reward)
                best_key = key

    if best_key is None:
        return {**base, "failed": True, "reason": "no valid formula found"}

    signal = vm.execute(list(best_key), factors)
    if signal is None:
        return {**base, "failed": True, "reason": "formula invalid at eval time"}
    full = signal.detach().cpu().numpy()
    direction = signal_direction(
        full[:, :train_end], target, reward_cfg.ic_min_stocks
    )
    metrics = evaluate_signal(
        float(direction) * full[:, train_end : fold.test_end_idx],
        loader,
        fold,
        backtest_config,
    )
    return {
        **base,
        "failed": False,
        "formula_text": formula_decode(list(best_key), vocab),
        "formula": list(best_key),
        "val_reward": best_val,
        "final_avg_reward": None,
        "direction": direction,
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


def build_result(
    proto_cfg: ProtocolConfig,
    tier_name: str,
    tier,
    rows: list[dict],
    data_end_date: str | None = None,
    extra_trial_rows: list[dict] | None = None,
    max_t_perms: int = 5000,
) -> dict:
    """Assemble the protocol artifact (schema contract, see module docstring).

    ``extra_trial_rows`` are trial rows from earlier protocol artifacts
    (e.g. screening runs whose OOS trials must count towards the DS/max-t
    multiplicity correction); they join the correction pool but are never
    merged into this run's own rows.
    """

    trial_pool = list(rows) + list(extra_trial_rows or [])
    return _sanitize(
        {
            "protocol_version": PROTOCOL_VERSION,
            "reward_version": REWARD_VERSION,
            "frequency": proto_cfg.frequency,
            "horizon": proto_cfg.horizon,
            "tier": tier_name,
            "steps": tier.steps,
            "batch_size": tier.batch_size,
            "seeds": list(proto_cfg.seeds),
            "random_samples": proto_cfg.random_samples,
            "random_seed": proto_cfg.random_seed,
            "folds": [
                {"train_end": f.train_end, "test_end": f.test_end}
                for f in proto_cfg.folds
            ],
            "baseline_signals": list(proto_cfg.baseline_signals),
            "data_end_date": data_end_date,
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
        rows.extend(baseline_candidates(loader, proto_cfg, fold, backtest_config))
        if proto_cfg.random_samples > 0:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} random-search "
                f"baseline samples={proto_cfg.random_samples} "
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
    )


def load_trial_rows(paths: str | None) -> list[dict]:
    """Trial rows from prior protocol artifacts (comma-separated paths)."""

    rows: list[dict] = []
    if not paths:
        return rows
    for path in paths.split(","):
        path = path.strip()
        if not path:
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        proto_cfg = make_protocol_config(raw)
        if args.no_random_search:
            proto_cfg.random_samples = 0

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        extra_trials = load_trial_rows(args.trials)

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
