"""Fixed backtest of the bare baseline factors (P1-03).

P1-03 contract: the seven bare factors of ``protocol.baseline_signals``
get a **fixed backtest only** — no searcher, no sampling, no trial-and-
error.  Each factor column enters the shared backtest engine directly
(the same engine, fee schedule and PIT universe mask every formal
backtest uses), and the trade direction is the fixed training-window
rule (sign of the mean forward rank IC on signal-date eligible cells,
``signal_direction`` — the same convention the protocol baseline rows
use), decided once per factor and never searched.

The payload records ``search: "none"`` and the full provenance
(dataset_id, universe policy, top_n, fee params, window), so the report
is auditably a measurement of the bare factors, not a claim about them.

Version 2 adds P3 frequency/horizon, constructor and execution provenance.
Version 3 fixes the four comparison quadrants (daily/weekly x equal-weight/
optimizer), holds each factor's signal direction constant across them and
records flattened return, risk, turnover, order and cost measurements.
Versions 1/2 remain readable history but are not current quadrant evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping

from loguru import logger

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_sim_config,
)
from ashare_data.gates import ProductionGateRunner
from ashare_execution import validate_execution_config
from ashare_logging import export_log_txt, setup_run_logging

from .backtest import AshareBacktestEngine
from .data_loader import AshareDataLoader
from .reward import signal_direction
from .reward import REWARD_VERSION
from .evaluation import PROTOCOL_VERSION
from .targets import causal_target_returns
from .time_contract import TrainingTimeContract
from .vocab import FEATURE_NAMES
from ashare_portfolio.rebalance import RebalancePolicy
from ashare_portfolio.execution_spec import execution_provenance

BARE_FACTOR_BACKTEST_VERSION = 3

_QUADRANTS = (
    ("daily", 1, "equal_weight"),
    ("daily", 1, "optimizer"),
    ("weekly", 1, "equal_weight"),
    ("weekly", 1, "optimizer"),
)


def _backtest_bare_factors_single(
    loader: AshareDataLoader,
    bt_cfg,
    names: list[str],
    train_end_date: str | None = None,
    *,
    directions: Mapping[str, int] | None = None,
) -> dict:
    """Run one fixed configuration through the shared backtest engine.

    When ``directions`` is omitted, the daily reference run learns each
    factor's orientation once from its training window.  The other three
    quadrants pass that mapping back unchanged so the compared signal is
    identical and only the calendar/constructor method varies.
    """

    if not names:
        raise ValueError("names must not be empty")
    for name in names:
        if name not in FEATURE_NAMES:
            raise ValueError(f"unknown factor {name!r} (not in FEATURE_NAMES)")

    effective_train_end = train_end_date or bt_cfg.train_end_date
    rebalance_policy = RebalancePolicy.from_config(bt_cfg)
    full_rebalance_mask = rebalance_policy.rebalance_mask(loader.dates)
    signal_end: int | None = None
    train_target = None
    if directions is None:
        contract = TrainingTimeContract.resolve(
            loader.dates,
            effective_train_end,
            horizon=rebalance_policy.horizon,
        )
        price_end = contract.train_label_end
        signal_end = contract.train_signal_end
        open_np = loader.raw_data_cache["open"][:, :price_end].numpy()
        train_target = causal_target_returns(
            open_np,
            loader.dates[:price_end],
            rebalance_policy,
            rebalance_mask=full_rebalance_mask[:price_end],
        )
        train_target = loader.mask_by_universe(train_target)
    universe_mask = loader.universe_mask
    raw_data = {k: v.numpy() for k, v in loader.raw_data_cache.items()}

    engine = AshareBacktestEngine(bt_cfg)
    rows: list[dict] = []
    for name in names:
        idx = FEATURE_NAMES.index(name)
        factor = loader.factor_tensor[idx]
        if directions is None:
            assert signal_end is not None and train_target is not None
            direction = signal_direction(
                factor[:, :signal_end].numpy(),
                train_target[:, :signal_end],
                universe_mask=universe_mask[:, :signal_end],
            )
        else:
            if name not in directions:
                raise ValueError(f"directions has no entry for factor {name!r}")
            direction = int(directions[name])
            if direction not in (-1, 1):
                raise ValueError(
                    f"direction for factor {name!r} must be -1 or 1"
                )
        signal_np = float(direction) * factor.numpy()
        result = engine.run(
            signal_np,
            raw_data,
            loader.ts_codes,
            loader.dates,
            universe_mask=universe_mask,
            rebalance_mask=full_rebalance_mask,
        )
        cost_fractions = list(result.cost_fractions or [])
        if len(cost_fractions) != len(result.daily_returns):
            raise RuntimeError("backtest cost trail is not aligned to returns")
        total_cost = float(
            sum(
                float(bt_cfg.initial_capital)
                * float(result.equity_curve[index])
                * float(cost_fraction)
                for index, cost_fraction in enumerate(cost_fractions)
            )
        )
        rows.append(
            {
                "name": name,
                "formula_text": name,
                "direction": int(direction),
                "metrics": result.metrics,
                "cost": total_cost,
                "cost_fraction": float(sum(cost_fractions)),
                "total_turnover": float(sum(result.turnover)),
            }
        )
        logger.info(
            "bare factor {} direction={} sharpe={:.3f} "
            "total_return={:.3f} turnover={:.3f}",
            name,
            int(direction),
            result.metrics.get("sharpe", float("nan")),
            result.metrics.get("total_return", float("nan")),
            result.metrics.get("average_turnover", float("nan")),
        )

    universe_policy = getattr(loader, "universe_policy", None)
    return {
        "version": BARE_FACTOR_BACKTEST_VERSION,
        "reward_version": REWARD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        **execution_provenance(bt_cfg),
        "search": "none",  # fixed backtest only: no searcher ever runs
        "dataset_id": loader.dataset_id,
        "universe_policy": (
            {
                "index_codes": [str(code) for code in universe_policy.index_codes],
                "min_listed_sessions": int(universe_policy.min_listed_sessions),
                "membership_end_inclusive": bool(universe_policy.membership_end_inclusive),
                "degraded": (
                    bool(loader.universe_status.degraded)
                    if loader.universe_status is not None
                    else None
                ),
            }
            if universe_policy is not None
            else None
        ),
        "config": {
            "train_end_date": effective_train_end,
            "rebalance_frequency": rebalance_policy.frequency,
            "target_horizon": rebalance_policy.horizon,
            "portfolio_method": bt_cfg.portfolio_method,
            "top_n": int(bt_cfg.top_n),
            "single_weight_cap": float(bt_cfg.single_weight_cap),
            "commission_rate": float(bt_cfg.commission_rate),
            "min_commission": float(bt_cfg.min_commission),
            "stamp_tax_rate": float(bt_cfg.stamp_tax_rate),
            "transfer_fee_rate": float(bt_cfg.transfer_fee_rate),
            "slippage_rate": float(bt_cfg.slippage_rate),
            "benchmark": bt_cfg.benchmark,
        },
        "window": {
            "start_date": loader.dates[0],
            "end_date": loader.dates[-1],
            "n_stocks": int(loader.factor_tensor.shape[1]),
            "n_dates": int(loader.factor_tensor.shape[2]),
        },
        "factors": rows,
    }


def _quadrant_factor_result(
    row: dict,
    *,
    frequency: str,
    horizon: int,
    method: str,
) -> dict:
    """Flatten the required auditable metrics for one factor/quadrant."""

    metrics = row["metrics"]
    required_metrics = (
        "total_return",
        "annual_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "average_turnover",
        "order_count",
    )
    missing = [key for key in required_metrics if key not in metrics]
    if missing:
        raise RuntimeError(
            "bare-factor quadrant backtest has no " + ", ".join(missing)
        )
    return {
        "name": row["name"],
        "formula_text": row["formula_text"],
        "direction": int(row["direction"]),
        "frequency": frequency,
        "horizon": int(horizon),
        "method": method,
        "total_return": float(metrics["total_return"]),
        "annual_return": float(metrics["annual_return"]),
        "sharpe": float(metrics["sharpe"]),
        "sortino": float(metrics["sortino"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "turnover": float(metrics["average_turnover"]),
        "total_turnover": float(row["total_turnover"]),
        "order_count": int(metrics["order_count"]),
        "cost": float(row["cost"]),
        "cost_fraction": float(row["cost_fraction"]),
        "metrics": dict(metrics),
    }


def backtest_bare_factors(
    loader: AshareDataLoader,
    bt_cfg,
    names: list[str],
    train_end_date: str | None = None,
) -> dict:
    """Run the fixed P3 four-quadrant bare-factor measurement.

    The daily/equal-weight quadrant is the reference: it fixes each factor's
    training-window direction once.  Daily/weekly x equal-weight/optimizer
    then run through :class:`AshareBacktestEngine` and its shared
    :class:`~ashare_portfolio.constructor.PortfolioConstructor`.  No search,
    sampling or alternative constructor is involved.
    """

    reference: dict | None = None
    fixed_directions: dict[str, int] | None = None
    quadrants: list[dict] = []
    flat_results: list[dict] = []

    for frequency, horizon, method in _QUADRANTS:
        quadrant_config = replace(
            bt_cfg,
            rebalance_frequency=frequency,
            target_horizon=horizon,
            portfolio_method=method,
        )
        single = _backtest_bare_factors_single(
            loader,
            quadrant_config,
            names,
            train_end_date=train_end_date,
            directions=fixed_directions,
        )
        if fixed_directions is None:
            fixed_directions = {
                row["name"]: int(row["direction"])
                for row in single["factors"]
            }
            reference = single

        factor_results = [
            _quadrant_factor_result(
                row,
                frequency=frequency,
                horizon=horizon,
                method=method,
            )
            for row in single["factors"]
        ]
        flat_results.extend(factor_results)
        quadrants.append(
            {
                "frequency": frequency,
                "horizon": int(horizon),
                "method": method,
                **execution_provenance(quadrant_config),
                "factors": factor_results,
            }
        )

    if reference is None or fixed_directions is None:
        raise RuntimeError("fixed quadrant specification produced no reference run")
    reference["artifact_schema"] = "fixed_four_quadrants"
    reference["direction_reference"] = {
        "frequency": "daily",
        "horizon": 1,
        "method": "equal_weight",
        "directions": fixed_directions,
    }
    reference["quadrants"] = quadrants
    reference["results"] = flat_results
    return reference


def main(argv=None) -> int:
    setup_run_logging(run_name="bare_factor_backtest")
    parser = argparse.ArgumentParser(
        description="Fixed backtest of the bare baseline factors "
        "(P3 version 3 four-quadrant measurement; no search of any kind)"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data/bare_factor_backtest.json")
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
        ProductionGateRunner(
            data_config, min_eligible=args.min_eligible
        ).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        sim_config = make_sim_config(raw, root)
        validate_execution_config(backtest_config, sim_config)
        proto_cfg = make_protocol_config(raw)
        names = list(proto_cfg.baseline_signals)

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        payload = backtest_bare_factors(loader, backtest_config, names=names)

        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Bare-factor backtest written to {out_path}")
        print(json.dumps(payload["results"], ensure_ascii=False, indent=2))
    finally:
        export_log_txt(run_name="bare_factor_backtest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
