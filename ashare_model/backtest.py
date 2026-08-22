"""Cross-sectional daily backtest and reward scoring."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ashare_data.config import BacktestConfig
from ashare_data.processor import limit_rate, open_to_open_returns, tradability_blocked
from ashare_data.schemas import BacktestResult
from ashare_execution import (
    ExecutionCostModel,
    validate_execution_config,
)
from ashare_logging import export_log_txt, setup_run_logging

from .reward import sortino_ratio
from .time_contract import TrainingTimeContract


class AshareBacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.initial_capital = float(self.config.initial_capital)
        self.cost_model = ExecutionCostModel.from_config(self.config)

    def run(
        self,
        factors: np.ndarray,
        raw_cache: dict[str, np.ndarray],
        ts_codes: list[str],
        dates: list[str],
        benchmark_returns: list[float] | None = None,
        stock_names: dict[str, str] | None = None,
        signal_range: range | tuple[int, int] | None = None,
    ) -> BacktestResult:
        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim != 2:
            raise ValueError("factors must be [stock, date]")
        if factors.shape != (len(ts_codes), len(dates)):
            raise ValueError("factor shape does not match ts_codes/dates")

        open_ = np.nan_to_num(
            np.asarray(raw_cache["open"], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        high = np.nan_to_num(
            np.asarray(raw_cache["high"], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        low = np.nan_to_num(
            np.asarray(raw_cache["low"], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        pre_close = np.nan_to_num(
            np.asarray(
                raw_cache.get("pre_close", np.roll(open_, 1, axis=1)),
                dtype=np.float64,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        volume = np.nan_to_num(
            np.asarray(
                raw_cache.get("volume", np.zeros_like(open_)),
                dtype=np.float64,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        n_stocks, n_dates = factors.shape
        if signal_range is None:
            signal_indices = list(range(max(n_dates - 2, 0)))
        elif isinstance(signal_range, range):
            signal_indices = list(signal_range)
        else:
            signal_indices = list(range(int(signal_range[0]), int(signal_range[1])))
        if any(t < 0 or t + 2 >= n_dates for t in signal_indices):
            raise ValueError("signal_range includes a signal without a complete t+2 exit")
        if signal_indices and signal_indices != list(
            range(signal_indices[0], signal_indices[-1] + 1)
        ):
            raise ValueError("signal_range must be contiguous")

        target_ret = self._open_to_open_returns(open_)
        if benchmark_returns is None:
            # Default benchmark: equal-weight universe return on the same
            # complete t+2 periods as the strategy itself.
            benchmark_returns = [
                float(np.mean(target_ret[:, t])) for t in signal_indices
            ]
        elif len(benchmark_returns) != len(signal_indices):
            raise ValueError("benchmark_returns must align with signal_range")
        prev_weights = np.zeros(n_stocks, dtype=np.float64)
        capital = self.initial_capital
        daily_returns: list[float] = []
        turnover_list: list[float] = []
        positions: list[dict[str, Any]] = []

        for t in signal_indices:
            signal = factors[:, t]
            entry_day = t + 1
            exit_day = t + 2
            selected = self._select_top_n(
                signal,
                entry_day,
                open_,
                high,
                low,
                pre_close,
                volume,
                ts_codes,
                stock_names or {},
                side="buy",
            )
            target_weights = np.zeros(n_stocks, dtype=np.float64)
            weight = 1.0 / max(self.config.top_n, 1)
            for idx in selected:
                target_weights[idx] = min(weight, self.config.single_weight_cap)
            total = target_weights.sum()
            if total > 0:
                target_weights /= total

            # Positions that must be sold need to be tradable on the sell side.
            sell_required = np.where(prev_weights > target_weights)[0]
            sell_blocked = self._blocked_mask(
                entry_day,
                open_,
                high,
                low,
                pre_close,
                volume,
                ts_codes,
                stock_names or {},
                side="sell",
            )
            for idx in sell_required:
                if sell_blocked[idx]:
                    # A limit-down position cannot be sold; keep it and let
                    # the remaining weights contract so cash stays consistent.
                    target_weights[idx] = prev_weights[idx]

            if target_weights.sum() > 0:
                target_weights /= target_weights.sum()

            buy_weights = np.maximum(target_weights - prev_weights, 0.0)
            sell_weights = np.maximum(prev_weights - target_weights, 0.0)
            turnover = float(np.abs(target_weights - prev_weights).sum())

            cost_fraction = float(
                self.cost_model.rebalance_cost_fraction(
                    buy_weights, sell_weights, capital
                )
            )
            gross_ret = float(np.dot(target_weights, target_ret[:, t]))
            net_ret = gross_ret - cost_fraction
            daily_returns.append(net_ret)
            turnover_list.append(turnover)
            capital *= 1.0 + net_ret

            positions.append(
                {
                    "signal_date": dates[t],
                    "entry_date": dates[entry_day],
                    "exit_date": dates[exit_day],
                    "ts_codes": [ts_codes[i] for i in np.where(target_weights > 0)[0]],
                    "weights": [
                        round(float(target_weights[i]), 6)
                        for i in np.where(target_weights > 0)[0]
                    ],
                }
            )
            prev_weights = target_weights

        equity = self._equity_curve(daily_returns)
        benchmark_equity = (
            self._equity_curve(benchmark_returns)
            if benchmark_returns is not None
            else None
        )
        metrics = self._metrics(daily_returns, equity)
        metrics["average_turnover"] = float(np.mean(turnover_list)) if turnover_list else 0.0
        return BacktestResult(
            equity_curve=[float(x) for x in equity],
            # Initial equity is marked at the first entry; subsequent points
            # are marked at each t+2 exit, exactly one date per equity point.
            dates=(
                [dates[signal_indices[0] + 1]]
                + [dates[t + 2] for t in signal_indices]
                if signal_indices
                else dates[:1]
            ),
            daily_returns=[float(x) for x in daily_returns],
            turnover=[float(x) for x in turnover_list],
            benchmark_equity=benchmark_equity,
            metrics=metrics,
            positions=positions,
        )

    @staticmethod
    def _open_to_open_returns(open_: np.ndarray) -> np.ndarray:
        return open_to_open_returns(open_)

    def _select_top_n(
        self,
        signal: np.ndarray,
        exec_day: int,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        pre_close: np.ndarray,
        volume: np.ndarray,
        ts_codes: list[str],
        stock_names: dict[str, str],
        side: str,
    ) -> list[int]:
        blocked = self._blocked_mask(
            exec_day,
            open_,
            high,
            low,
            pre_close,
            volume,
            ts_codes,
            stock_names,
            side=side,
        )
        valid = [(i, float(signal[i])) for i in range(len(signal)) if not blocked[i] and np.isfinite(signal[i])]
        valid.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in valid[: self.config.top_n]]

    @staticmethod
    def _blocked_mask(
        exec_day: int,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        pre_close: np.ndarray,
        volume: np.ndarray,
        ts_codes: list[str],
        stock_names: dict[str, str],
        side: str,
    ) -> np.ndarray:
        n_stocks = open_.shape[0]
        if exec_day >= open_.shape[1]:
            return np.ones(n_stocks, dtype=bool)
        return tradability_blocked(
            open_[:, exec_day],
            high[:, exec_day],
            low[:, exec_day],
            pre_close[:, exec_day],
            volume[:, exec_day],
            ts_codes,
            stock_names,
            side,
        )

    @staticmethod
    def _limit_rate(ts_code: str, name: str) -> float:
        return limit_rate(ts_code, name)

    @staticmethod
    def _equity_curve(daily_returns: list[float]) -> list[float]:
        equity = [1.0]
        for ret in daily_returns:
            equity.append(equity[-1] * (1.0 + ret))
        return equity

    @staticmethod
    def _metrics(daily_returns: list[float], equity: list[float]) -> dict[str, float]:
        if not daily_returns:
            return {}
        returns = np.asarray(daily_returns)
        n = len(returns)
        total_return = float(equity[-1] - 1.0)
        ann_return = float((1.0 + total_return) ** (252 / n) - 1.0)
        vol = float(returns.std(ddof=1) * math.sqrt(252)) if n > 1 else 0.0
        sharpe = (ann_return - 0.02) / (vol + 1e-9)
        sortino = sortino_ratio(returns)
        running_max = np.maximum.accumulate(np.asarray(equity))
        drawdown = 1.0 - np.asarray(equity) / (running_max + 1e-12)
        max_drawdown = float(np.max(drawdown))
        calmar = ann_return / (max_drawdown + 1e-9)
        return {
            "total_return": total_return,
            "annual_return": ann_return,
            "annual_volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
        }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from ashare_data.config import (
        load_config,
        make_backtest_config,
        make_data_config,
        make_model_config,
        make_sim_config,
    )
    from .data_loader import AshareDataLoader
    from .reward import signal_direction
    from .vm import StackVM, formula_decode
    from .vocab import FORMULA_VOCAB, resolve_formula_tokens

    setup_run_logging(run_name="backtest")
    parser = argparse.ArgumentParser(description="Backtest an A-share formula")
    parser.add_argument("--config", default=None)
    parser.add_argument("--formula-file", default=None)
    parser.add_argument("--output", default="data/backtest_result.json")
    args = parser.parse_args()

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        sim_config = make_sim_config(raw, root)
        validate_execution_config(backtest_config, sim_config)
        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()

        formula_file = args.formula_file or data_config.data_dir / "best_ashare_strategy.json"
        formula_file = Path(formula_file)
        if not formula_file.is_absolute():
            formula_file = root / formula_file
        if not formula_file.exists():
            raise SystemExit(f"Formula file not found: {formula_file}")
        payload = json.loads(formula_file.read_text(encoding="utf-8"))
        tokens = resolve_formula_tokens(payload, FORMULA_VOCAB)

        vm = StackVM(
            FORMULA_VOCAB, industry_codes=getattr(loader, "industry_codes", None)
        )
        factors = vm.execute(tokens, loader.factor_tensor)
        if factors is None:
            raise SystemExit("Formula is invalid")

        # Trade direction: prefer the direction recorded by the trainer
        # (decided on its validation tail); legacy artifacts without it
        # infer the direction from the training window's forward rank IC so
        # a negative-IC formula is never mechanically traded backwards.
        direction = int(payload.get("direction", 1))
        if "direction" not in payload:
            contract = TrainingTimeContract.resolve(
                loader.dates, backtest_config.train_end_date
            )
            price_end = contract.train_label_end
            signal_end = contract.train_signal_end
            train_target = open_to_open_returns(
                loader.raw_data_cache["open"][:, :price_end].numpy()
            )
            direction = signal_direction(
                factors[:, :signal_end].detach().cpu().numpy(),
                train_target[:, :signal_end],
            )
        signal_np = float(direction) * factors.detach().cpu().numpy()

        engine = AshareBacktestEngine(backtest_config)
        result = engine.run(
            signal_np,
            {k: v.numpy() for k, v in loader.raw_data_cache.items()},
            loader.ts_codes,
            loader.dates,
            stock_names=loader.stock_names,
        )
        output = {
            "formula": tokens,
            "formula_text": formula_decode(tokens, FORMULA_VOCAB),
            "direction": direction,
            "metrics": result.metrics,
            "dates": result.dates,
            "equity_curve": result.equity_curve,
            "benchmark": backtest_config.benchmark,
            "benchmark_equity": result.benchmark_equity,
            "positions": result.positions,
        }
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        print(f"Result saved to {out_path}")
    finally:
        export_log_txt(run_name="backtest")


if __name__ == "__main__":
    main()
