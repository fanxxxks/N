"""Daily A-share paper-trading loop.

Usage:
    python -m ashare_trading.run_sim [--config config/ashare_config.yaml]
                                    [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                                    [--reset | --resume]

Replay safety: when the portfolio state already contains processed history,
a plain start is refused; pass ``--resume`` to continue from the state's
``last_exec_date`` or ``--reset`` to start over from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    ModelConfig,
    SimConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_sim_config,
)
from ashare_data.schemas import SimOrder
from ashare_data.processor import open_to_open_returns
from ashare_data.universe import require_production_universe
from ashare_execution import validate_execution_config

from ashare_model.backtest import AshareBacktestEngine
from ashare_model.data_loader import AshareDataLoader
from ashare_model.reward import signal_direction
from ashare_model.time_contract import TrainingTimeContract
from ashare_model.vm import StackVM, formula_decode
from ashare_model.vocab import FORMULA_VOCAB, resolve_formula_tokens

from .matching import SimBroker
from .portfolio import SimulationPortfolio
from ashare_logging import export_log_txt, setup_run_logging


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_progress_file(
    path: str | Path,
    phase: str,
    current_date: str | None = None,
    equity: float | None = None,
) -> None:
    """Atomically write the sim progress file consumed by the status API.

    Phases: ``loading`` (data/factor warm-up), ``executing`` (day loop),
    ``stopped`` (STOP_SIGNAL), ``finished``, ``error``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "current_date": current_date,
        "equity": equity,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class SimulationRunner:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        backtest_config: BacktestConfig,
        sim_config: SimConfig,
        loader: AshareDataLoader,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.backtest_config = backtest_config
        self.sim_config = sim_config
        self.loader = loader
        validate_execution_config(backtest_config, sim_config)
        self.portfolio = SimulationPortfolio(
            sim_config.initial_capital, sim_config.state_path
        )
        self.broker = SimBroker(sim_config)
        self.vm = StackVM(FORMULA_VOCAB)
        self.formula_tokens: list[int] | None = None
        self.formula_text = ""
        self.direction = 1
        self._has_recorded_direction = False

    def load_formula(self) -> None:
        path = self.data_config.data_dir / "best_ashare_strategy.json"
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.formula_tokens = resolve_formula_tokens(payload, FORMULA_VOCAB)
        self.formula_text = formula_decode(self.formula_tokens, FORMULA_VOCAB)
        # The trainer records the trade direction it learned on its
        # validation tail; legacy artifacts without it fall back to an
        # inference on the training window in run().
        self.direction = int(payload.get("direction", 1))
        self._has_recorded_direction = "direction" in payload
        logger.success(f"Loaded formula: {self.formula_text}")

    def run(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        resume: bool = False,
    ) -> dict[str, object]:
        if not resume and self.portfolio.has_history:
            raise ValueError(
                "Portfolio state already contains processed history "
                f"(last_exec_date={self.portfolio.last_exec_date}). "
                "Pass resume=True to continue or reset the portfolio first."
            )
        self._write_progress("loading")
        if self.loader.factor_tensor is None:
            self.loader.load_data()
        if self.formula_tokens is None:
            self.load_formula()

        # Industry-group context for CS_NEUTRALIZE, aligned with the factor
        # stack (None when the loader carries no industry data).
        self.vm.industry_codes = getattr(self.loader, "industry_codes", None)
        factors = self.vm.execute(self.formula_tokens, self.loader.factor_tensor)
        if factors is None:
            raise ValueError("Formula is invalid")
        signals = factors.detach().cpu().numpy()
        if not self._has_recorded_direction:
            # Legacy artifact: infer the direction from the training window
            # so a negative-IC formula is traded on its learned side.
            contract = TrainingTimeContract.resolve(
                self.loader.dates,
                self.backtest_config.train_end_date,
            )
            signal_end = contract.train_signal_end
            price_end = contract.train_label_end
            target = open_to_open_returns(
                self.loader.raw_data_cache["open"][:, :price_end].numpy()
            )
            target = self.loader.mask_by_universe(target)
            self.direction = signal_direction(
                signals[:, :signal_end],
                target[:, :signal_end],
            )
            logger.info(f"Inferred trade direction from training window: {self.direction}")
        else:
            logger.info(f"Trade direction from artifact: {self.direction}")
        signals = float(self.direction) * signals
        raw = {k: v.numpy() for k, v in self.loader.raw_data_cache.items()}
        dates = self.loader.dates
        ts_codes = self.loader.ts_codes
        stock_names = self._stock_names()

        start_idx = 0
        end_idx = len(dates) - 1
        resume_idx: int | None = None
        if resume:
            last = self.portfolio.last_exec_date
            if not last:
                raise ValueError(
                    "resume=True but the portfolio state has no "
                    "last_exec_date; reset the portfolio before replaying"
                )
            resume_idx = next((i for i, d in enumerate(dates) if d == last), None)
            if resume_idx is None:
                raise ValueError(
                    f"last_exec_date {last} is not in the dataset; "
                    "reset the portfolio or re-sync the data"
                )
            # Signals start at the watermark day so the first execution is
            # the first trading day after it: no overlap, no gap.
            start_idx = resume_idx
        if start_date:
            start_date = start_date.replace("-", "")
            user_idx = next((i for i, d in enumerate(dates) if d >= start_date), 0)
            if resume_idx is not None and user_idx < resume_idx:
                raise ValueError(
                    f"--start {start_date} precedes the last processed date "
                    f"({self.portfolio.last_exec_date}); refusing to replay. "
                    "Use --reset for a full replay."
                )
            start_idx = max(start_idx, user_idx)
        if end_date:
            end_date = end_date.replace("-", "")
            end_idx = next(
                (i for i, d in enumerate(dates) if d > end_date), len(dates) - 1
            )

        self.sim_config.orders_dir.mkdir(parents=True, exist_ok=True)
        self.sim_config.trades_dir.mkdir(parents=True, exist_ok=True)
        # The paper-trading position count is driven by the sim config, with
        # the backtest config as a fallback.
        engine = AshareBacktestEngine(
            replace(
                self.backtest_config,
                top_n=self.sim_config.max_positions or self.backtest_config.top_n,
            )
        )

        self._write_progress("executing", current_date=self.portfolio.last_exec_date)
        last_equity: float | None = None
        stopped = False
        for signal_idx in range(start_idx, end_idx):
            if self._handle_stop_signal():
                stopped = True
                break
            exec_idx = signal_idx + 1
            exec_date = dates[exec_idx]
            signal = np.where(
                (
                    self.loader.universe_mask[:, signal_idx]
                    & self.loader.universe_mask[:, exec_idx]
                ),
                signals[:, signal_idx],
                np.nan,
            )
            selected = engine._select_top_n(
                signal,
                exec_idx,
                raw["open"],
                raw["high"],
                raw["low"],
                raw["pre_close"],
                raw["volume"],
                ts_codes,
                stock_names,
                side="buy",
            )
            target_weights = self._equal_weights(selected, len(ts_codes))
            orders = self._make_orders(
                target_weights,
                selected,
                ts_codes,
                exec_idx,
                exec_date,
                raw,
                stock_names,
            )

            bars = self.loader.bars
            trades = self.broker.execute_orders(
                orders,
                bars,
                exec_date,
                self.portfolio,
                stock_names,
                self.backtest_config,
            )
            # Persist orders after execution so their final status/reason
            # (filled, skipped, limit_up, ...) is part of the paper-trail.
            self._write_json(
                self.sim_config.orders_dir / f"{exec_date}.json",
                [o.__dict__ for o in orders],
            )
            self._write_json(
                self.sim_config.trades_dir / f"{exec_date}.json",
                [t.__dict__ for t in trades],
            )
            close_prices = {
                ts_codes[i]: float(raw["close"][i, exec_idx]) for i in range(len(ts_codes))
            }
            last_equity = self.portfolio.record_equity(exec_date, close_prices)
            # Advance the resume watermark only now: orders/trades files are
            # on disk and the equity snapshot is about to be saved, so a crash
            # before this save replays the whole day cleanly instead of
            # leaving a half-processed day behind.
            self.portfolio.last_exec_date = exec_date
            self.portfolio.save()
            self._write_progress(
                "executing", current_date=exec_date, equity=last_equity
            )

        phase = "stopped" if stopped else "finished"
        self._write_progress(
            phase,
            current_date=self.portfolio.last_exec_date,
            equity=last_equity,
        )
        return {
            "final_cash": self.portfolio.cash,
            "trade_count": self.portfolio.trade_count,
            "equity_history": self.portfolio.equity_history,
        }

    def _write_progress(
        self,
        phase: str,
        current_date: str | None = None,
        equity: float | None = None,
    ) -> None:
        # Progress reporting is auxiliary: a failure here must never abort a
        # simulation that is otherwise running fine.
        try:
            write_progress_file(
                self.sim_config.progress_path, phase, current_date, equity
            )
        except OSError as exc:
            logger.warning(f"Could not write progress file: {exc}")

    def _make_orders(
        self,
        target_weights: np.ndarray,
        selected: list[int],
        ts_codes: list[str],
        exec_idx: int,
        exec_date: str,
        raw: dict[str, np.ndarray],
        stock_names: dict[str, str],
    ) -> list[SimOrder]:
        open_prices = raw["open"][:, exec_idx]
        equity = self.portfolio.cash
        for code, pos in self.portfolio.positions.items():
            if code in ts_codes:
                i = ts_codes.index(code)
                equity += pos.quantity * raw["close"][i, exec_idx - 1]
            else:
                equity += pos.quantity * pos.last_price

        target_shares: dict[int, int] = {}
        for i in selected:
            weight = target_weights[i]
            price = open_prices[i]
            if price > 0:
                # Buys must be whole lots of 100 shares (A-share rule).
                target_shares[i] = (
                    int(equity * weight / price) // 100
                ) * 100

        orders: list[SimOrder] = []
        seen: set[str] = set()
        for i in selected:
            code = ts_codes[i]
            pos = self.portfolio.positions.get(code)
            current = pos.quantity if pos else 0
            target = target_shares.get(i, 0)
            delta = target - current
            if delta == 0:
                continue
            if delta > 0:
                # Whole-lot buys only; skip sub-lot adjustments.
                buy_qty = (delta // 100) * 100
                if buy_qty <= 0:
                    continue
                side = "buy"
                quantity = buy_qty
            else:
                side = "sell"
                quantity = abs(delta)
            orders.append(
                SimOrder(
                    order_id=f"{exec_date}-{code}-{side}-{len(orders)}",
                    ts_code=code,
                    trade_date=exec_date,
                    side=side,
                    quantity=quantity,
                    price=float(open_prices[i]),
                )
            )
            seen.add(code)

        for code, pos in self.portfolio.positions.items():
            if code in seen:
                continue
            if code not in ts_codes:
                continue
            i = ts_codes.index(code)
            if i in selected:
                continue
            orders.append(
                SimOrder(
                    order_id=f"{exec_date}-{code}-sell-{len(orders)}",
                    ts_code=code,
                    trade_date=exec_date,
                    side="sell",
                    quantity=pos.available_quantity,
                    price=float(open_prices[i]),
                )
            )
        return orders

    @staticmethod
    def _equal_weights(selected: list[int], n_stocks: int) -> np.ndarray:
        weights = np.zeros(n_stocks, dtype=np.float64)
        if not selected:
            return weights
        weight = 1.0 / len(selected)
        for i in selected:
            weights[i] = weight
        return weights

    def _stock_names(self) -> dict[str, str]:
        # Prefer real names from the stocks table (needed for ST limit-rate
        # detection); fall back to the code itself when metadata is absent.
        names = dict(self.loader.stock_names)
        for code in self.loader.ts_codes:
            names.setdefault(code, code)
        return names

    def _handle_stop_signal(self) -> bool:
        path = Path(self.sim_config.stop_signal_path)
        if not path.exists():
            return False
        try:
            signal = path.read_text(encoding="utf-8").strip().upper()
        except OSError:
            return True
        if signal not in {"", "STOP", "STOPPED"}:
            return False
        logger.warning(f"STOP signal received from {path}. Simulation will stop.")
        try:
            path.write_text("STOPPED", encoding="utf-8")
        except OSError:
            pass
        return True

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _clear_stop_signal(sim_config: SimConfig) -> None:
    """Remove a leftover STOP/STOPPED file so an explicit restart can run."""

    path = Path(sim_config.stop_signal_path)
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Cleared stop signal: {path}")
    except OSError as exc:
        logger.warning(f"Could not clear stop signal file {path}: {exc}")


def main() -> None:
    setup_run_logging(run_name="sim")
    parser = argparse.ArgumentParser(description="Run A-share paper trading")
    parser.add_argument("--config", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the portfolio state's last processed date",
    )
    args = parser.parse_args()

    try:
        root = _project_root()
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        require_production_universe(data_config)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        sim_config = make_sim_config(raw, root)
        loader = AshareDataLoader(data_config, model_config)
        runner = SimulationRunner(
            data_config, model_config, backtest_config, sim_config, loader
        )
        if args.reset:
            runner.portfolio.reset()
            _clear_stop_signal(sim_config)
        elif args.resume:
            _clear_stop_signal(sim_config)
        elif runner.portfolio.has_history:
            logger.error(
                "Portfolio state already contains processed history "
                "(last_exec_date={}). Pass --resume to continue or --reset "
                "to start over.",
                runner.portfolio.last_exec_date,
            )
            sys.exit(2)
        result = runner.run(args.start, args.end, resume=args.resume)
        logger.success(f"Simulation complete: {result}")
    except Exception:
        sim_cfg = locals().get("sim_config")
        if sim_cfg is not None:
            try:
                write_progress_file(sim_cfg.progress_path, phase="error")
            except OSError:
                pass
        raise
    finally:
        export_log_txt(run_name="sim")


if __name__ == "__main__":
    main()
