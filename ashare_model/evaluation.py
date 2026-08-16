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
* Candidate multiplicity is corrected with Deflated Sharpe and a max-t
  permutation test (single trial matrix for both).

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
from loguru import logger

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    FoldConfig,
    ModelConfig,
    ProtocolConfig,
    RewardConfig,
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
from .reward import REWARD_VERSION
from .train import AshareTrainer
from .vm import StackVM
from .vocab import FEATURE_NAMES, FORMULA_VOCAB

PROTOCOL_VERSION = "1"

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
    }


def evaluate_formula(
    tokens: list[int],
    loader: AshareDataLoader,
    fold: Fold,
    bt_cfg: BacktestConfig,
    vm: StackVM | None = None,
) -> dict | None:
    """Execute a formula on the full factor history, slice the test window,
    and score it.  Returns ``None`` for an invalid formula."""

    vm = vm or StackVM(FORMULA_VOCAB)
    signal = vm.execute(list(tokens), loader.factor_tensor)
    if signal is None:
        return None
    sliced = signal.detach().cpu().numpy()[
        :, fold.train_end_idx : fold.test_end_idx
    ]
    return evaluate_signal(sliced, loader, fold, bt_cfg)


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
    }


def baseline_candidates(
    loader: AshareDataLoader,
    proto_cfg: ProtocolConfig,
    fold: Fold,
    bt_cfg: BacktestConfig,
) -> list[dict]:
    """Single-factor baseline rows: the factor row itself as the signal."""

    validate_baseline_signals(proto_cfg.baseline_signals, FEATURE_NAMES)
    factors, _, _, _ = epoch_slice(loader, fold)
    rows: list[dict] = []
    for name in proto_cfg.baseline_signals:
        metrics = evaluate_signal(factors[FEATURE_NAMES.index(name)], loader, fold, bt_cfg)
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
    metrics = evaluate_formula(tokens, loader, fold, backtest_config)
    if metrics is None:
        return {**base, "failed": True, "reason": "formula invalid at eval time"}
    return {
        **base,
        "failed": False,
        "formula_text": trainer.best_formula,
        "formula": list(tokens),
        "val_reward": float(trainer.best_reward),
        "final_avg_reward": (
            float(trainer.history[-1]["avg_reward"]) if trainer.history else None
        ),
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
    dsr=None,
    max_t=None,
) -> dict:
    """Assemble the protocol artifact (schema contract, see module docstring)."""

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
            "dsr": dsr,
            "max_t": max_t,
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
        proto_cfg, tier_name, tier, rows, data_end_date=loader.dates[-1]
    )


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
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        proto_cfg = make_protocol_config(raw)

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()

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
    finally:
        export_log_txt(run_name="evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
