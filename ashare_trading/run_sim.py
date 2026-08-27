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
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
from ashare_data.io_utils import atomic_write_json
from ashare_data.schemas import SimOrder
from ashare_data.processor import has_cross_sectional_dispersion, open_to_open_returns
from ashare_data.gates import ProductionGateRunner
from ashare_execution import validate_execution_config

from ashare_model.backtest import AshareBacktestEngine
from ashare_model.data_loader import AshareDataLoader
from ashare_model.reward import signal_direction
from ashare_model.time_contract import TrainingTimeContract
from ashare_model.vm import StackVM, formula_decode
from ashare_model.vocab import FORMULA_VOCAB, resolve_formula_tokens

from .matching import SimBroker
from .orders import build_orders, target_shares_from_weights
from .portfolio import SimulationPortfolio
from .signals import clear_stop_signal, stop_requested
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
    ``stopped`` (STOP_SIGNAL), ``finished``, ``error``.  Shares the
    project-wide atomic writer (ashare_data.io_utils.atomic_write_json).
    """

    payload = {
        "phase": phase,
        "current_date": current_date,
        "equity": equity,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(path, payload)


class SimulationRunner:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        backtest_config: BacktestConfig,
        sim_config: SimConfig,
        loader: AshareDataLoader,
        *,
        today: str | None = None,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.backtest_config = backtest_config
        self.sim_config = sim_config
        self.loader = loader
        # The current ST snapshot (stocks.is_st) is only valid as of today:
        # same-day executions may use it, historical replay must not.
        self.today = (today or datetime.now().strftime("%Y%m%d")).replace("-", "")
        validate_execution_config(backtest_config, sim_config)
        self.portfolio = SimulationPortfolio(
            sim_config.initial_capital, sim_config.state_path
        )
        self.broker = SimBroker()
        # Built lazily in compute_signals() once the loader's PIT mask is
        # guaranteed to exist.
        self.vm: StackVM | None = None
        self.formula_tokens: list[int] | None = None
        self.formula_text = ""
        self.direction = 1
        self._has_recorded_direction = False

    def load_formula(self) -> None:
        path = self.data_config.data_dir / "best_ashare_strategy.json"
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        from ashare_model.artifact_versions import classify_strategy

        verdict = classify_strategy(payload)
        if verdict["legacy"]:
            logger.warning(
                "LEGACY strategy artifact: {} — {}; the paper account is "
                "replaying an old generation, not the current champion",
                path,
                "; ".join(verdict["reasons"]),
            )
        self.formula_tokens = resolve_formula_tokens(payload, FORMULA_VOCAB)
        self.formula_text = formula_decode(self.formula_tokens, FORMULA_VOCAB)
        # The trainer records the trade direction it learned on its
        # validation tail; legacy artifacts without it fall back to an
        # inference on the training window in compute_signals().
        self.direction = int(payload.get("direction", 1))
        self._has_recorded_direction = "direction" in payload
        logger.success(f"Loaded formula: {self.formula_text}")

    def compute_signals(self) -> np.ndarray:
        """Execute the loaded formula over the factor stack.

        Returns the direction-scaled ``[stock, date]`` signal matrix and
        wires the PIT universe mask into the VM's cross-sectional operators
        first.  Single code path for the daily loop and for parity checks
        against the backtest engine on the same signal date.
        """

        if self.loader.factor_tensor is None:
            self.loader.load_data()
        if self.formula_tokens is None:
            self.load_formula()
        if self.loader.universe_mask is None:
            raise ValueError("simulation requires the loader's PIT universe mask")

        # The VM is constructed with the loader's PIT eligibility mask and
        # the industry-group context for CS_NEUTRALIZE, aligned with the
        # factor stack — every formal execution path carries the mask.
        self.vm = StackVM(
            FORMULA_VOCAB,
            industry_codes=getattr(self.loader, "industry_codes", None),
            universe_mask=torch.tensor(
                self.loader.universe_mask, dtype=torch.bool
            ),
        )
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
                universe_mask=self.loader.universe_mask[:, :signal_end],
            )
            logger.info(f"Inferred trade direction from training window: {self.direction}")
        else:
            logger.info(f"Trade direction from artifact: {self.direction}")
        return float(self.direction) * signals

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
        signals = self.compute_signals()
        raw = {k: v.numpy() for k, v in self.loader.raw_data_cache.items()}
        dates = self.loader.dates
        ts_codes = self.loader.ts_codes
        universe_mask = self.loader.universe_mask
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
            # Same-day execution is the only reliable as-of use of the
            # current ST snapshot; historical dates get board rates only.
            same_day = exec_date == self.today
            st_codes = self.loader.current_st_codes if same_day else None
            st_mask = (
                np.asarray([code in st_codes for code in ts_codes], dtype=bool)
                if st_codes
                else None
            )
            # Selection uses universe_mask[:, signal_idx] (and the entry
            # date, exactly like the backtest engine): ineligible stocks
            # can never generate new buy orders.
            eligible = universe_mask[:, signal_idx] & universe_mask[:, exec_idx]
            selected, selectable_values = engine._select_top_n(
                signals[:, signal_idx],
                exec_idx,
                raw["open"],
                raw["high"],
                raw["low"],
                raw["pre_close"],
                raw["volume"],
                ts_codes,
                side="buy",
                eligible=eligible,
                st_mask=st_mask,
            )
            # No-signal semantics (T1-02): a selectable cross-section
            # without dispersion carries no ranking information — the
            # portfolio is held as-is (no signal-driven orders), never
            # churned.  Positions that lost universe eligibility are still
            # liquidated through the ordinary exit path (the mask-exit
            # contract); the remaining book is kept at its current weights.
            if selectable_values and not has_cross_sectional_dispersion(
                selectable_values
            ):
                held_names = []
                hold_weights = np.zeros(len(ts_codes), dtype=np.float64)
                equity = self.portfolio.cash
                for code, pos in self.portfolio.positions.items():
                    if code not in ts_codes:
                        equity += pos.quantity * pos.last_price
                        continue
                    i = ts_codes.index(code)
                    equity += pos.quantity * raw["close"][i, exec_idx - 1]
                    if eligible[i]:
                        held_names.append(i)
                        hold_weights[i] = (
                            pos.quantity * raw["close"][i, exec_idx - 1]
                        )
                if equity > 0:
                    hold_weights /= equity
                orders = self._make_orders(
                    hold_weights,
                    held_names,
                    ts_codes,
                    exec_idx,
                    exec_date,
                    raw,
                )
            else:
                target_weights = self._equal_weights(
                    selected,
                    len(ts_codes),
                    top_n=self.sim_config.max_positions
                    or self.backtest_config.top_n,
                    single_weight_cap=self.backtest_config.single_weight_cap,
                )
                orders = self._make_orders(
                    target_weights,
                    selected,
                    ts_codes,
                    exec_idx,
                    exec_date,
                    raw,
                )

            bars = self.loader.bars
            trades = self.broker.execute_orders(
                orders,
                bars,
                exec_date,
                self.portfolio,
                stock_names,
                self.backtest_config,
                st_codes=st_codes,
            )
            # Persist orders after execution so their final status/reason
            # (filled, skipped, limit_up, ...) is part of the paper-trail.
            atomic_write_json(
                self.sim_config.orders_dir / f"{exec_date}.json",
                [o.__dict__ for o in orders],
            )
            atomic_write_json(
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
    ) -> list[SimOrder]:
        open_prices = raw["open"][:, exec_idx]
        equity = self.portfolio.cash
        for code, pos in self.portfolio.positions.items():
            if code in ts_codes:
                i = ts_codes.index(code)
                equity += pos.quantity * raw["close"][i, exec_idx - 1]
            else:
                equity += pos.quantity * pos.last_price

        # T3-02: one shared weight->order rule (whole-lot buys, sells
        # first, full-exit sells) with the paper-trading lot size.
        target_shares = target_shares_from_weights(
            target_weights, equity, open_prices, lot_size=100
        )
        current_quantities = {
            code: pos.quantity for code, pos in self.portfolio.positions.items()
        }
        return build_orders(
            exec_date,
            ts_codes,
            open_prices,
            target_shares,
            selected,
            current_quantities,
            lot_size=100,
        )

    @staticmethod
    def _equal_weights(
        selected: list[int],
        n_stocks: int,
        *,
        top_n: int,
        single_weight_cap: float,
    ) -> np.ndarray:
        """Target weights of the selected names, engine-aligned (T1-02).

        Each selected name gets ``min(1/top_n, single_weight_cap)`` and the
        weights are never renormalized upward: an under-filled day keeps
        the remainder in cash, and the cap is a hard per-name ceiling —
        exactly the backtest engine's weight construction.
        """

        weights = np.zeros(n_stocks, dtype=np.float64)
        if not selected:
            return weights
        weight = min(1.0 / max(top_n, 1), float(single_weight_cap))
        for i in selected:
            weights[i] = weight
        return weights

    def _stock_names(self) -> dict[str, str]:
        # Display names for orders/portfolio only; fall back to the code
        # itself when metadata is absent.  Limit detection never reads
        # these: it uses board rates, plus the loader's as-of
        # current_st_codes on same-day executions alone.
        names = dict(self.loader.stock_names)
        for code in self.loader.ts_codes:
            names.setdefault(code, code)
        return names

    def _handle_stop_signal(self) -> bool:
        return stop_requested(self.sim_config.stop_signal_path)


def _clear_stop_signal(sim_config: SimConfig) -> None:
    """Remove a leftover STOP/STOPPED file so an explicit restart can run."""
    clear_stop_signal(sim_config.stop_signal_path)


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
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
    )
    args = parser.parse_args()

    try:
        root = _project_root()
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(data_config, min_eligible=args.min_eligible).require_production()
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
