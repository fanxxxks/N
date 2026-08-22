"""Fast offline tests for the A-share AlphaGPT modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    ModelConfig,
    RewardConfig,
    load_config,
    make_data_config,
)
from ashare_data.db import AshareDB
from ashare_data.processor import filter_universe, normalize_daily_bars
from ashare_model.alphagpt import build_action_mask
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.data_loader import AshareDataLoader
from ashare_model.factors import compute_factor_tensor
from ashare_model.train import AshareTrainer
from ashare_model.vm import StackVM, formula_decode
from ashare_model.vocab import FORMULA_VOCAB
from ashare_trading.matching import SimBroker
from ashare_trading.portfolio import SimulationPortfolio


def _make_bars(tmp_path: Path, n_dates: int = 40) -> tuple[list[str], list[str], pd.DataFrame]:
    dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y%m%d").tolist()
    ts_codes = ["000001.SZ", "600000.SH", "300001.SZ"]
    rows = []
    for code in ts_codes:
        for i, date in enumerate(dates):
            close = 10.0 + i * 0.1 + (0.5 if code == "000001.SZ" else 0.0)
            open_ = close * 0.99
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": open_,
                    "high": close * 1.02,
                    "low": open_ * 0.98,
                    "close": close,
                    "pre_close": close * 0.99,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * close,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
            )
    return dates, ts_codes, pd.DataFrame(rows)


def _make_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_days=1,
        min_amount=1.0,
    )


def _write_db(tmp_path: Path, dates: list[str], ts_codes: list[str], bars: pd.DataFrame):
    cfg = _make_config(tmp_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.parquet_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_daily(bars.to_dict("records"), cfg)
        db.upsert_stocks(
            [
                {"ts_code": c, "name": c, "industry": None, "list_date": "20200101", "is_st": False}
                for c in ts_codes
            ],
            cfg,
        )
        db.upsert_calendar([{"trade_date": d, "is_open": True} for d in dates], cfg)
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": c, "in_date": f"2020010{i + 1}", "out_date": "99991231"}
                for i, c in enumerate(ts_codes)
            ],
            cfg,
        )
    return cfg


def test_config_paths_resolve_to_project_root():
    raw = load_config(None, project_root=Path(__file__).resolve().parents[1])
    cfg = make_data_config(raw, Path(__file__).resolve().parents[1])
    assert "config" not in str(cfg.duckdb_path)
    assert cfg.data_dir.name == "data"


def test_stackvm_valid_and_invalid():
    vm = StackVM(FORMULA_VOCAB)
    factor = torch.randn(FORMULA_VOCAB.feature_count, 2, 4)
    add_id = FORMULA_VOCAB.operator_offset
    out = vm.execute([1, 2, add_id], factor)
    assert out is not None and out.shape == (2, 4)
    assert formula_decode([1, 2, add_id], FORMULA_VOCAB).startswith("(")
    assert vm.execute([1], factor) is not None
    assert vm.execute([1, 2], factor) is None


def test_action_mask_has_legal_initial_tokens():
    open_slots = torch.ones(8, dtype=torch.long)
    stack_sizes = torch.zeros(8, dtype=torch.long)
    mask = build_action_mask(open_slots, stack_sizes, 0, 6, FORMULA_VOCAB)
    assert (mask == 0.0).sum() > 0
    assert not torch.isnan(mask).any()
    # Done samples may only pad.
    open_slots[:] = 0
    mask_done = build_action_mask(open_slots, stack_sizes, 0, 6, FORMULA_VOCAB)
    assert (mask_done == 0.0).sum() == 8


def test_factor_tensor_and_backtest(tmp_path):
    dates, ts_codes, bars = _make_bars(tmp_path, 30)
    cfg = _write_db(tmp_path, dates, ts_codes, bars)
    loader = AshareDataLoader(cfg, ModelConfig())
    loader.load_data()
    assert loader.factor_tensor.shape[0] == FORMULA_VOCAB.feature_count
    assert not torch.isnan(loader.factor_tensor).any()

    factors = loader.factor_tensor[0].numpy()
    engine = AshareBacktestEngine(BacktestConfig(top_n=2, train_end_date="2024-02-01"))
    result = engine.run(
        factors,
        {k: v.numpy() for k, v in loader.raw_data_cache.items()},
        loader.ts_codes,
        loader.dates,
        stock_names={c: c for c in ts_codes},
    )
    assert len(result.daily_returns) == len(loader.dates) - 2
    assert result.equity_curve[-1] > 0
    assert "sortino" in result.metrics


def test_training_smoke(tmp_path):
    dates, ts_codes, bars = _make_bars(tmp_path, 25)
    cfg = _write_db(tmp_path, dates, ts_codes, bars)
    model_cfg = ModelConfig(batch_size=4, train_steps=1, max_formula_len=6)
    loader = AshareDataLoader(cfg, model_cfg)
    loader.load_data()
    # The trainer pins the weight init (init_seed) and train() pins the
    # sampling seed, so this smoke test is deterministic on every device.
    trainer = AshareTrainer(
        cfg,
        model_cfg,
        BacktestConfig(train_end_date="2024-01-20", top_n=2),
        loader,
        # The smoke fixture's returns are noise: the production validation
        # floor is relaxed so the smoke test exercises artifact saving.
        reward_config=RewardConfig(min_val_reward=-1e9, min_val_icir=-1e9),
    )
    tokens = trainer.train(steps=1, batch_size=4)
    assert tokens is not None
    assert trainer.best_formula


def test_sim_broker_t_plus_one_and_costs(tmp_path):
    dates, ts_codes, bars = _make_bars(tmp_path, 6)
    cfg = _write_db(tmp_path, dates, ts_codes, bars)
    sim_cfg = __import__("ashare_data.config", fromlist=["SimConfig"]).SimConfig(
        initial_capital=100000.0,
        state_path=tmp_path / "state.json",
        orders_dir=tmp_path / "orders",
        trades_dir=tmp_path / "trades",
    )
    portfolio = SimulationPortfolio(sim_cfg.initial_capital, sim_cfg.state_path)
    broker = SimBroker(sim_cfg)
    from ashare_data.schemas import SimOrder

    order = SimOrder(
        order_id="o1",
        ts_code=ts_codes[0],
        trade_date=dates[1],
        side="buy",
        quantity=100,
        price=float(bars.iloc[1]["open"]),
    )
    trades = broker.execute_orders(
        [order], bars, dates[1], portfolio, {c: c for c in ts_codes}, BacktestConfig()
    )
    assert trades
    pos = portfolio.positions[ts_codes[0]]
    assert pos.quantity > 0
    assert pos.available_quantity == 0  # T+1 means no same-day sell
