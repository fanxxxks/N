"""Signal measurement and OOS-trial stitching for the evaluation protocol.

Extracted from ``evaluation.py`` (P7 Phase A2) by reason-to-change: this
module owns *how one signal/trial is measured* — full-engine metrics, rank
IC (universe and tradable), the equal-weight benchmark row, per-candidate
aggregation and the stitched outer-OOS trial matrix (T4-01).  It changes
when metric/stitching *semantics* change — not when fold contracts,
statistical corrections, search backends or artifact schemas change.

Import direction: depends on :mod:`ashare_model.eval_folds` (leaf-ward);
never imports the ``evaluation`` facade.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ashare_data.config import BacktestConfig
from ashare_portfolio.rebalance import RebalancePolicy

from .backtest import AshareBacktestEngine, equal_weight_benchmark_returns
from .data_loader import AshareDataLoader
from .diagnostics import rank_ic_stats
from .eval_folds import Fold, epoch_slice
from .vm import StackVM
from .vocab import FORMULA_VOCAB

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
        # P10 §5.2 组合活动门 (additive observability): the portfolio
        # activity counters the engine already computes, surfaced so the
        # searcher comparison can disclose suppressed/low-activity books.
        "rebalance_count": int(m.get("rebalance_count", 0)),
        "order_count": int(m.get("order_count", 0)),
        "suppressed_trade_count": int(m.get("suppressed_trade_count", 0)),
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
            # IP-10 01 (itemized in docs/test_runtime_measurement_log.md):
            # on degenerate summaries whose order statistics at the
            # quantile point are both +inf, numpy's quantile lerp computes
            # inf - inf ("invalid value encountered in scalar subtract")
            # and returns NaN -- the NaN summary is the intended value, so
            # the FP-state warning is muted at the call site (np.errstate
            # never changes the computed values).
            with np.errstate(invalid="ignore"):
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
