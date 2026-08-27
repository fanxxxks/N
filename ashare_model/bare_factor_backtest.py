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

See docs/phase5_measurement_log.md §2 (BARE_FACTOR_BACKTEST_VERSION 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from ashare_data.processor import open_to_open_returns
from ashare_execution import validate_execution_config
from ashare_logging import export_log_txt, setup_run_logging

from .backtest import AshareBacktestEngine
from .data_loader import AshareDataLoader
from .reward import signal_direction
from .time_contract import TrainingTimeContract
from .vocab import FEATURE_NAMES

BARE_FACTOR_BACKTEST_VERSION = 1


def backtest_bare_factors(
    loader: AshareDataLoader,
    bt_cfg,
    names: list[str],
    train_end_date: str | None = None,
) -> dict:
    """Fixed (search-free) backtest of ``names`` factor columns.

    ``names`` must be non-empty feature names from the vocabulary; the
    CLI feeds ``protocol.baseline_signals`` (the seven bare factors).
    Deterministic: no RNG anywhere, identical inputs produce a
    bit-identical payload.
    """

    if not names:
        raise ValueError("names must not be empty")
    for name in names:
        if name not in FEATURE_NAMES:
            raise ValueError(f"unknown factor {name!r} (not in FEATURE_NAMES)")

    effective_train_end = train_end_date or bt_cfg.train_end_date
    contract = TrainingTimeContract.resolve(loader.dates, effective_train_end)
    price_end = contract.train_label_end
    signal_end = contract.train_signal_end
    open_np = loader.raw_data_cache["open"][:, :price_end].numpy()
    train_target = loader.mask_by_universe(open_to_open_returns(open_np))
    universe_mask = loader.universe_mask
    raw_data = {k: v.numpy() for k, v in loader.raw_data_cache.items()}

    engine = AshareBacktestEngine(bt_cfg)
    rows: list[dict] = []
    for name in names:
        idx = FEATURE_NAMES.index(name)
        factor = loader.factor_tensor[idx]
        direction = signal_direction(
            factor[:, :signal_end].numpy(),
            train_target[:, :signal_end],
            universe_mask=universe_mask[:, :signal_end],
        )
        signal_np = float(direction) * factor.numpy()
        result = engine.run(
            signal_np,
            raw_data,
            loader.ts_codes,
            loader.dates,
            universe_mask=universe_mask,
        )
        rows.append(
            {
                "name": name,
                "formula_text": name,
                "direction": int(direction),
                "metrics": result.metrics,
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

    policy = getattr(loader, "universe_policy", None)
    return {
        "version": BARE_FACTOR_BACKTEST_VERSION,
        "search": "none",  # fixed backtest only: no searcher ever runs
        "dataset_id": loader.dataset_id,
        "universe_policy": (
            {
                "index_codes": [str(code) for code in policy.index_codes],
                "min_listed_sessions": int(policy.min_listed_sessions),
                "membership_end_inclusive": bool(policy.membership_end_inclusive),
                "degraded": (
                    bool(loader.universe_status.degraded)
                    if loader.universe_status is not None
                    else None
                ),
            }
            if policy is not None
            else None
        ),
        "config": {
            "train_end_date": effective_train_end,
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


def main(argv=None) -> int:
    setup_run_logging(run_name="bare_factor_backtest")
    parser = argparse.ArgumentParser(
        description="Fixed backtest of the seven bare baseline factors "
        "(P1-03; no search of any kind)"
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
        print(json.dumps(payload["factors"], ensure_ascii=False, indent=2))
    finally:
        export_log_txt(run_name="bare_factor_backtest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
