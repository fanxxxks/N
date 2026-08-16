"""Cross-sectional daily backtest and reward scoring."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ashare_data.config import BacktestConfig
from ashare_data.processor import open_to_open_returns
from ashare_data.schemas import BacktestResult
from ashare_logging import export_log_txt, setup_run_logging

from .reward import sortino_ratio, trading_cost_fraction


class AshareBacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.initial_capital = float(self.config.initial_capital)

    def run(
        self,
        factors: np.ndarray,
        raw_cache: dict[str, np.ndarray],
        ts_codes: list[str],
        dates: list[str],
        benchmark_returns: list[float] | None = None,
        stock_names: dict[str, str] | None = None,
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

        target_ret = self._open_to_open_returns(open_)
        if benchmark_returns is None:
            # Default benchmark: equal-weight universe return on the same
            # open-to-open basis as the strategy itself.
            benchmark_returns = [
                float(np.mean(target_ret[:, t])) for t in range(target_ret.shape[1] - 1)
            ]
        n_stocks, n_dates = factors.shape
        prev_weights = np.zeros(n_stocks, dtype=np.float64)
        daily_returns: list[float] = []
        turnover_list: list[float] = []
        positions: list[dict[str, Any]] = []

        for t in range(n_dates - 1):
            signal = factors[:, t]
            exec_day = t + 1
            selected = self._select_top_n(
                signal,
                exec_day,
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
                exec_day,
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

            cost_fraction = trading_cost_fraction(buy_weights, sell_weights, self.config)
            gross_ret = float(np.dot(target_weights, target_ret[:, t]))
            net_ret = gross_ret - cost_fraction
            daily_returns.append(net_ret)
            turnover_list.append(turnover)

            positions.append(
                {
                    "signal_date": dates[t],
                    "exec_date": dates[exec_day],
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
            # Dates align with the equity curve: dates[0] is the start
            # (equity 1.0) and every later point is the end-of-day value.
            dates=dates,
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
        blocked = np.zeros(n_stocks, dtype=bool)
        if exec_day >= open_.shape[1]:
            blocked[:] = True
            return blocked

        o = open_[:, exec_day]
        h = high[:, exec_day]
        l = low[:, exec_day]
        pc = pre_close[:, exec_day]
        v = volume[:, exec_day]
        suspended = (o <= 0) | (v <= 0) | (pc <= 0)
        blocked |= suspended

        one_word = np.isclose(o, h) & np.isclose(o, l)
        change = np.zeros_like(o)
        valid = pc > 0
        change[valid] = o[valid] / pc[valid] - 1.0
        limit_rate = np.array([AshareBacktestEngine._limit_rate(c, stock_names.get(c, "")) for c in ts_codes])
        limit_up = one_word & (change >= limit_rate - 0.005)
        limit_down = one_word & (change <= -limit_rate + 0.005)
        if side == "buy":
            blocked |= limit_up
        else:
            blocked |= limit_down
        return blocked

    @staticmethod
    def _limit_rate(ts_code: str, name: str) -> float:
        if "ST" in name.upper():
            return 0.05
        prefix = ts_code.split(".")[0][:3]
        if prefix in {"300", "301", "688", "689"}:
            return 0.20
        return 0.10

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
    )
    from .data_loader import AshareDataLoader
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

        vm = StackVM(FORMULA_VOCAB)
        factors = vm.execute(tokens, loader.factor_tensor)
        if factors is None:
            raise SystemExit("Formula is invalid")

        engine = AshareBacktestEngine(backtest_config)
        result = engine.run(
            factors.detach().cpu().numpy(),
            {k: v.numpy() for k, v in loader.raw_data_cache.items()},
            loader.ts_codes,
            loader.dates,
            stock_names=loader.stock_names,
        )
        output = {
            "formula": tokens,
            "formula_text": formula_decode(tokens, FORMULA_VOCAB),
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
