"""Reproducible real-data acceptance measurement for P3.

The report is deliberately diagnostic, not an alpha claim.  It measures the
shared reward/backtest portfolio trail, causal-label interval overlap, the
100k default order structure, a pre-P3-compatible configuration replay and
one fixed bare factor across the four P3 portfolio quadrants.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
from loguru import logger

from ashare_data.config import (
    BacktestConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
)
from ashare_data.gates import ProductionGateRunner
from ashare_data.io_utils import atomic_write_json
from ashare_data.processor import open_to_open_returns
from ashare_data.universe import UniverseMask
from ashare_logging import export_log_txt, setup_run_logging
from ashare_portfolio.execution_spec import execution_provenance
from ashare_portfolio.rebalance import RebalancePolicy

from .backtest import AshareBacktestEngine
from .bare_factor_backtest import backtest_bare_factors
from .data_loader import AshareDataLoader
from .reward import simulate_basket_daily_returns
from .vocab import FEATURE_NAMES


P3_MEASUREMENT_VERSION = 3
DEFAULT_FACTOR = "REVERSAL_5"
DEFAULT_SEED = 20260829
DEFAULT_STOCKS = 60
DEFAULT_DATES = 126
DEFAULT_RUNTIME_REPEATS = 3

_FEE_FIELDS = (
    "commission_rate",
    "min_commission",
    "stamp_tax_rate",
    "transfer_fee_rate",
    "slippage_rate",
)


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _max_abs_diff(left, right) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise ValueError(f"measurement arrays differ in shape: {lhs.shape} != {rhs.shape}")
    if lhs.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs - rhs)))


def _measurement_view(
    loader: AshareDataLoader,
    *,
    factor_name: str,
    seed: int,
    n_stocks: int,
    n_dates: int,
) -> tuple[AshareDataLoader, dict[str, Any]]:
    if factor_name not in FEATURE_NAMES:
        raise ValueError(f"unknown measurement factor {factor_name!r}")
    if n_stocks < 2:
        raise ValueError("n_stocks must be >= 2")
    if n_dates < 12:
        raise ValueError("n_dates must be >= 12")
    if loader.factor_tensor is None or loader.universe_mask is None:
        raise ValueError("loader must be loaded before P3 measurement")
    if n_dates > len(loader.dates):
        raise ValueError(
            f"requested {n_dates} dates but loader has only {len(loader.dates)}"
        )

    date_start = len(loader.dates) - n_dates
    date_slice = slice(date_start, None)
    factor_index = FEATURE_NAMES.index(factor_name)
    signal = _as_numpy(loader.factor_tensor[factor_index, :, date_slice])
    open_ = _as_numpy(loader.raw_data_cache["open"][:, date_slice])
    mask = np.asarray(loader.universe_mask[:, date_slice], dtype=bool)
    valid = mask & np.isfinite(signal) & np.isfinite(open_) & (open_ > 0.0)
    coverage = np.mean(valid, axis=1)
    initial_valid = valid[:, 0] & valid[:, 1]
    pool = np.flatnonzero(initial_valid & (coverage >= 0.80))
    if pool.size < n_stocks:
        raise ValueError(
            "not enough stocks satisfy initial PIT/price/factor validity and "
            f"80% coverage: need {n_stocks}, have {pool.size}"
        )

    pool = np.asarray(
        sorted(pool.tolist(), key=lambda index: str(loader.ts_codes[index])),
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    selected = rng.choice(pool, size=n_stocks, replace=False)
    selected = np.asarray(
        sorted(selected.tolist(), key=lambda index: str(loader.ts_codes[index])),
        dtype=np.int64,
    )
    selected_list = selected.tolist()

    view = copy.copy(loader)
    view.dates = list(loader.dates[date_slice])
    view.ts_codes = [str(loader.ts_codes[index]) for index in selected_list]
    view.factor_tensor = loader.factor_tensor[:, selected_list, date_slice]
    view.raw_data_cache = {
        key: value[selected_list, date_slice]
        for key, value in loader.raw_data_cache.items()
    }
    view.universe = UniverseMask(
        eligible=np.asarray(loader.universe_mask[selected, date_slice], dtype=bool),
        reasons=np.asarray(
            loader.universe_reason_codes[selected, date_slice], dtype=np.uint16
        ),
    )
    view._tradability_cache = None
    if getattr(loader, "industry_codes", None) is not None:
        industry = loader.industry_codes
        if getattr(industry, "ndim", 0) == 2:
            view.industry_codes = industry[selected_list, date_slice]

    codes_blob = "\n".join(view.ts_codes).encode("utf-8")
    train_end_index = max(5, n_dates // 2) - 1
    return view, {
        "selection_rule": (
            "last n_dates; stocks with signal-day/entry-day PIT+finite factor/"
            "positive open and >=80% valid-window coverage; seeded sample; "
            "final rows sorted by ts_code"
        ),
        "pool_size": int(pool.size),
        "n_stocks": int(n_stocks),
        "n_dates": int(n_dates),
        "full_start_index": int(date_start),
        "start_date": view.dates[0],
        "end_date": view.dates[-1],
        "train_end_date": view.dates[train_end_index],
        "stock_codes_sha256": hashlib.sha256(codes_blob).hexdigest(),
        "stock_codes": list(view.ts_codes),
    }


def _raw_numpy(loader: AshareDataLoader) -> dict[str, np.ndarray]:
    raw = {
        key: _as_numpy(value).astype(np.float64, copy=False)
        for key, value in loader.raw_data_cache.items()
    }
    # Match AshareBacktestEngine's exact missing-bar normalization before the
    # same arrays feed the reward parity path.
    for key in ("open", "high", "low", "pre_close", "volume"):
        raw[key] = np.nan_to_num(
            raw[key], nan=0.0, posinf=0.0, neginf=0.0
        )
    return raw


def _blocked_matrices(
    engine: AshareBacktestEngine,
    raw: dict[str, np.ndarray],
    codes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n_stocks, n_dates = raw["open"].shape
    buy = np.zeros((n_stocks, n_dates), dtype=bool)
    sell = np.zeros((n_stocks, n_dates), dtype=bool)
    for day in range(n_dates):
        buy[:, day] = engine._blocked_mask(  # noqa: SLF001 - parity probe
            day,
            raw["open"],
            raw["high"],
            raw["low"],
            raw["pre_close"],
            raw["volume"],
            codes,
            side="buy",
        )
        sell[:, day] = engine._blocked_mask(  # noqa: SLF001 - parity probe
            day,
            raw["open"],
            raw["high"],
            raw["low"],
            raw["pre_close"],
            raw["volume"],
            codes,
            side="sell",
        )
    return buy, sell


def _run_engine(
    config: BacktestConfig,
    signal: np.ndarray,
    raw: dict[str, np.ndarray],
    codes: list[str],
    dates: list[str],
    universe_mask: np.ndarray,
):
    rebalance_mask = RebalancePolicy.from_config(config).rebalance_mask(dates)
    return AshareBacktestEngine(config).run(
        signal,
        raw,
        codes,
        dates,
        universe_mask=universe_mask,
        rebalance_mask=rebalance_mask,
    )


def _cost_total(result, initial_capital: float) -> tuple[float, float]:
    fractions = list(result.cost_fractions or [])
    if len(fractions) != len(result.daily_returns):
        raise RuntimeError("cost trail is not aligned to the measured returns")
    total = sum(
        float(initial_capital)
        * float(result.equity_curve[index])
        * float(cost_fraction)
        for index, cost_fraction in enumerate(fractions)
    )
    return float(total), float(sum(fractions))


def _summarize_orders(result, config: BacktestConfig) -> dict[str, Any]:
    diagnostics = list(result.construction_diagnostics or [])
    orders = [int(item.get("order_count", 0)) for item in diagnostics]
    initial_index = next((index for index, count in enumerate(orders) if count), None)
    initial_count = orders[initial_index] if initial_index is not None else 0
    subsequent = orders[initial_index + 1 :] if initial_index is not None else orders
    subsequent_due = [
        orders[index]
        for index, item in enumerate(diagnostics)
        if index != initial_index and bool(item.get("rebalance_due"))
    ]

    amounts: list[float] = []
    initial_amounts: list[float] = []
    capital = float(config.initial_capital)
    buys = list(result.buy_weights or [])
    sells = list(result.sell_weights or [])
    for index, (buy, sell) in enumerate(zip(buys, sells)):
        traded = np.asarray(buy, dtype=np.float64) + np.asarray(
            sell, dtype=np.float64
        )
        day_amounts = (traded[traded > 0.0] * capital).tolist()
        amounts.extend(float(value) for value in day_amounts)
        if index == initial_index:
            initial_amounts.extend(float(value) for value in day_amounts)
        capital *= 1.0 + float(result.daily_returns[index])

    minimum = config.min_trade_amount
    small_count = (
        sum(value + 1e-9 < float(minimum) for value in amounts)
        if minimum is not None
        else 0
    )
    total_cost, cost_fraction = _cost_total(result, config.initial_capital)
    return {
        "initial_order_index": initial_index,
        "initial_order_count": int(initial_count),
        "initial_min_order_amount": (
            float(min(initial_amounts)) if initial_amounts else None
        ),
        "minimum_order_amount": float(min(amounts)) if amounts else None,
        "configured_min_trade_amount": (
            float(minimum) if minimum is not None else None
        ),
        "orders_below_configured_minimum": int(small_count),
        "subsequent_average_order_count": (
            float(np.mean(subsequent)) if subsequent else 0.0
        ),
        "subsequent_rebalance_average_order_count": (
            float(np.mean(subsequent_due)) if subsequent_due else 0.0
        ),
        "max_order_count": int(max(orders, default=0)),
        "days_with_30_or_more_orders": int(sum(count >= 30 for count in orders)),
        "order_count": int(result.metrics.get("order_count", 0)),
        "suppressed_trade_count": int(
            result.metrics.get("suppressed_trade_count", 0)
        ),
        "average_turnover": float(result.metrics.get("average_turnover", 0.0)),
        "total_turnover": float(sum(result.turnover)),
        "total_cost": total_cost,
        "total_cost_fraction": cost_fraction,
    }


def _runtime_comparison(
    configs: dict[str, BacktestConfig],
    *,
    signal: np.ndarray,
    raw: dict[str, np.ndarray],
    codes: list[str],
    dates: list[str],
    universe_mask: np.ndarray,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if repeats < 1:
        raise ValueError("runtime_repeats must be >= 1")

    # One untimed warm-up per configuration, then alternate order to reduce
    # systematic first/second-run bias.
    results = {
        name: _run_engine(config, signal, raw, codes, dates, universe_mask)
        for name, config in configs.items()
    }
    timings: dict[str, list[float]] = {name: [] for name in configs}
    names = list(configs)
    for repeat in range(repeats):
        order = names if repeat % 2 == 0 else list(reversed(names))
        for name in order:
            started = time.perf_counter()
            results[name] = _run_engine(
                configs[name], signal, raw, codes, dates, universe_mask
            )
            timings[name].append(time.perf_counter() - started)

    runtime = {
        name: {
            "runtime_repeats": int(repeats),
            "runtime_seconds": [float(value) for value in values],
            "runtime_seconds_median": float(statistics.median(values)),
            "runtime_seconds_min": float(min(values)),
            "runtime_seconds_max": float(max(values)),
        }
        for name, values in timings.items()
    }
    return results, runtime


def _label_measurements(
    full_dates: list[str],
    *,
    sample_start_index: int,
    sample_n_dates: int,
) -> dict[str, Any]:
    sample_stop_index = sample_start_index + sample_n_dates
    if not 0 <= sample_start_index < len(full_dates):
        raise ValueError("sample_start_index is outside the full date axis")
    if sample_n_dates < 1 or sample_stop_index > len(full_dates):
        raise ValueError("sample date window is outside the full date axis")

    policies = (
        RebalancePolicy("daily", 1),
        RebalancePolicy("weekly", 1),
        RebalancePolicy("every_5_days", 5),
        RebalancePolicy("every_10_days", 10),
        # P6: the every_20_days cadence joins the audit (monthly is
        # excluded: its legality is calendar-gated by the policy itself).
        RebalancePolicy("every_20_days", 20),
    )
    rows: list[dict[str, Any]] = []
    overall = 0
    for policy in policies:
        mask = policy.rebalance_mask(full_dates)
        indices = [
            int(index)
            for index in np.flatnonzero(mask)
            if sample_start_index <= index
            and index + 1 + policy.horizon < sample_stop_index
        ]
        intervals = [(index + 1, index + 1 + policy.horizon) for index in indices]
        maximum = 0
        for left, right in zip(intervals, intervals[1:]):
            maximum = max(
                maximum,
                max(0, min(left[1], right[1]) - max(left[0], right[0])),
            )
        overall = max(overall, maximum)
        rows.append(
            {
                "frequency": policy.frequency,
                "horizon": int(policy.horizon),
                "rebalance_indices": indices,
                "sample_rebalance_indices": [
                    index - sample_start_index for index in indices
                ],
                "rebalance_dates": [full_dates[index] for index in indices],
                "valid_label_count": len(indices),
                "max_overlap_sessions": int(maximum),
            }
        )
    return {"max_overlap_sessions": int(overall), "policies": rows}


def _git_metadata(root: Path) -> dict[str, Any]:
    def read(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    return {
        "commit": read("rev-parse", "HEAD"),
        "branch": read("branch", "--show-current"),
        "dirty": bool(read("status", "--porcelain")),
    }


def build_p3_measurement(
    loader: AshareDataLoader,
    bt_cfg: BacktestConfig,
    *,
    factor_name: str = DEFAULT_FACTOR,
    seed: int = DEFAULT_SEED,
    n_stocks: int = DEFAULT_STOCKS,
    n_dates: int = DEFAULT_DATES,
    runtime_repeats: int = DEFAULT_RUNTIME_REPEATS,
) -> dict[str, Any]:
    """Build the JSON-serializable P3 acceptance measurement."""

    view, sample = _measurement_view(
        loader,
        factor_name=factor_name,
        seed=seed,
        n_stocks=n_stocks,
        n_dates=n_dates,
    )
    factor_index = FEATURE_NAMES.index(factor_name)
    signal = _as_numpy(view.factor_tensor[factor_index]).astype(np.float64)
    raw = _raw_numpy(view)
    universe_mask = np.asarray(view.universe_mask, dtype=bool)

    p3_config = replace(
        bt_cfg,
        start_date=sample["start_date"],
        end_date=sample["end_date"],
        train_end_date=sample["train_end_date"],
        rebalance_frequency="daily",
        target_horizon=1,
        portfolio_method="equal_weight",
        initial_capital=100_000.0,
    )
    pre_p3_config = replace(
        p3_config,
        top_n=30,
        buy_rank=None,
        sell_rank=None,
        min_trade_amount=None,
        turnover_budget=None,
        target_weight_change_threshold=0.0,
    )

    rebalance_mask = RebalancePolicy.from_config(p3_config).rebalance_mask(
        view.dates
    )
    engine = AshareBacktestEngine(p3_config)
    backtest = engine.run(
        signal,
        raw,
        view.ts_codes,
        view.dates,
        universe_mask=universe_mask,
        rebalance_mask=rebalance_mask,
    )
    blocked_buy, blocked_sell = _blocked_matrices(
        engine, raw, view.ts_codes
    )
    target_returns = open_to_open_returns(raw["open"])
    reward = simulate_basket_daily_returns(
        signal,
        target_returns,
        p3_config,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        universe_mask=universe_mask,
        tie_break_keys=np.asarray(view.ts_codes),
        adv=raw["volume"] * raw["open"],
        rebalance_mask=rebalance_mask,
    )
    parity = {
        "max_target_weight_diff": _max_abs_diff(
            backtest.target_weights, reward.target_weights
        ),
        "max_buy_weight_diff": _max_abs_diff(
            backtest.buy_weights, reward.buy_weights
        ),
        "max_sell_weight_diff": _max_abs_diff(
            backtest.sell_weights, reward.sell_weights
        ),
        "max_turnover_diff": _max_abs_diff(
            backtest.turnover, reward.turnover
        ),
        "max_order_count_diff": int(
            _max_abs_diff(
                [
                    item["order_count"]
                    for item in backtest.construction_diagnostics
                ],
                [
                    item["order_count"]
                    for item in reward.construction_diagnostics
                ],
            )
        ),
        "max_cost_fraction_diff": _max_abs_diff(
            backtest.cost_fractions, reward.daily_cost_fractions
        ),
        "max_net_return_diff": _max_abs_diff(
            backtest.daily_returns, reward.daily_net_returns
        ),
    }

    comparison_configs = {
        "pre_p3_compatible": pre_p3_config,
        "p3_default": p3_config,
    }
    comparison_results, runtime = _runtime_comparison(
        comparison_configs,
        signal=signal,
        raw=raw,
        codes=view.ts_codes,
        dates=view.dates,
        universe_mask=universe_mask,
        repeats=runtime_repeats,
    )
    summaries = {}
    for name, config in comparison_configs.items():
        summary = _summarize_orders(comparison_results[name], config)
        summary.update(runtime[name])
        summary["portfolio_config"] = execution_provenance(config)[
            "portfolio_config"
        ]
        summaries[name] = summary

    bare = backtest_bare_factors(
        view,
        p3_config,
        names=[factor_name],
        train_end_date=sample["train_end_date"],
    )
    bare_rows = [quadrant["factors"][0] for quadrant in bare["quadrants"]]

    root = Path(__file__).resolve().parents[1]
    pre_summary = summaries["pre_p3_compatible"]
    p3_summary = summaries["p3_default"]
    return {
        "version": P3_MEASUREMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": int(seed),
        "code": _git_metadata(root),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "dataset": {
            "dataset_id": view.dataset_id,
            "full_n_stocks": len(loader.ts_codes),
            "full_n_dates": len(loader.dates),
            "full_start_date": loader.dates[0],
            "full_end_date": loader.dates[-1],
        },
        "sample": {**sample, "factor": factor_name},
        "execution": execution_provenance(p3_config),
        "parity": parity,
        "labels": _label_measurements(
            loader.dates,
            sample_start_index=sample["full_start_index"],
            sample_n_dates=sample["n_dates"],
        ),
        "default_100k": p3_summary,
        "pre_post": {
            "invariants": {
                "same_signal": True,
                "same_dates": True,
                "same_stocks": True,
                "same_pit_universe": True,
                "same_execution_delay": True,
                "same_fees": all(
                    getattr(pre_p3_config, field) == getattr(p3_config, field)
                    for field in _FEE_FIELDS
                ),
            },
            "pre_p3_compatible": pre_summary,
            "p3_default": p3_summary,
            "changes_p3_minus_pre": {
                "runtime_seconds_median": (
                    p3_summary["runtime_seconds_median"]
                    - pre_summary["runtime_seconds_median"]
                ),
                "order_count": p3_summary["order_count"] - pre_summary["order_count"],
                "average_turnover": (
                    p3_summary["average_turnover"]
                    - pre_summary["average_turnover"]
                ),
                "total_cost": p3_summary["total_cost"] - pre_summary["total_cost"],
                "total_cost_fraction": (
                    p3_summary["total_cost_fraction"]
                    - pre_summary["total_cost_fraction"]
                ),
            },
        },
        "bare_factor": {
            "artifact_version": bare["version"],
            "factor": factor_name,
            "direction_reference": bare["direction_reference"],
            "quadrants": bare_rows,
        },
    }


def main(argv: list[str] | None = None) -> int:
    setup_run_logging(run_name="p3_measurement")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data/p3_measurement.json")
    parser.add_argument("--factor", default=DEFAULT_FACTOR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stocks", type=int, default=DEFAULT_STOCKS)
    parser.add_argument("--dates", type=int, default=DEFAULT_DATES)
    parser.add_argument(
        "--runtime-repeats", type=int, default=DEFAULT_RUNTIME_REPEATS
    )
    parser.add_argument("--min-eligible", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw_config = load_config(args.config, project_root=root)
        data_config = make_data_config(raw_config, root)
        ProductionGateRunner(
            data_config, min_eligible=args.min_eligible
        ).require_production()
        model_config = make_model_config(raw_config)
        backtest_config = make_backtest_config(raw_config)
        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        payload = build_p3_measurement(
            loader,
            backtest_config,
            factor_name=args.factor,
            seed=args.seed,
            n_stocks=args.stocks,
            n_dates=args.dates,
            runtime_repeats=args.runtime_repeats,
        )
        output = Path(args.output)
        if not output.is_absolute():
            output = root / output
        atomic_write_json(output, payload)
        logger.success("P3 measurement written to {}", output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        export_log_txt(run_name="p3_measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
