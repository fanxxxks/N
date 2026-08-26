"""Cross-sectional daily backtest and reward scoring."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ashare_data.config import BacktestConfig
from ashare_data.processor import (
    has_cross_sectional_dispersion,
    open_to_open_returns,
    tradability_blocked,
)
from ashare_data.schemas import BacktestResult
from ashare_execution import (
    ExecutionCostModel,
    validate_execution_config,
)
from ashare_logging import export_log_txt, setup_run_logging

from .reward import sortino_ratio
from .time_contract import TrainingTimeContract


def equal_weight_benchmark_returns(
    target_ret: np.ndarray,
    signal_indices: list[int],
    universe_mask: np.ndarray,
) -> list[float]:
    """Equal-weight universe return per executed signal period.

    The benchmark averages the forward target over the cells that are
    eligible on the signal date **and** on the entry date
    (``universe_mask[:, t] & universe_mask[:, t + 1]``) and whose target is
    finite; a period without any such cell earns 0.0 (stable, no NaN
    spread).  The mask is mandatory — there is no unconstrained path.
    Single code path shared by the backtest engine and the evaluation
    protocol's benchmark row, so both report the identical universe return.
    """

    target_ret = np.asarray(target_ret, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if universe_mask.shape != target_ret.shape:
        raise ValueError(
            f"universe_mask shape {universe_mask.shape} does not match "
            f"target shape {target_ret.shape}"
        )
    returns: list[float] = []
    for t in signal_indices:
        eligible = universe_mask[:, t] & universe_mask[:, t + 1]
        values = target_ret[eligible, t]
        values = values[np.isfinite(values)]
        returns.append(float(np.mean(values)) if values.size else 0.0)
    return returns


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
        universe_mask: np.ndarray,
        benchmark_returns: list[float] | None = None,
        signal_range: range | tuple[int, int] | None = None,
    ) -> BacktestResult:
        """Execute the daily top-n strategy over the signal columns.

        ``universe_mask`` is the mandatory ``[stock, date]`` bool PIT
        eligibility mask: a position can only be newly opened in a stock
        eligible on the signal date **and** on the entry date
        (``universe_mask[:, t] & universe_mask[:, t + 1]``), the default
        equal-weight benchmark averages only those cells (and requires a
        valid target), and a day without any eligible stock earns zero for
        both the strategy (flat book) and the benchmark.  Existing
        positions of a stock that exits the universe are sold through the
        ordinary sell path — the mask never erases them silently.  A mask
        whose shape does not match the signal raises ``ValueError``; there
        is no unconstrained execution path.

        Limit detection always uses board rates (main board 10%,
        ChiNext / STAR 20%): the engine replays history, there is no dated
        ST status, and current stock names must never rewrite the past.
        """

        factors = np.asarray(factors, dtype=np.float64)
        if factors.ndim != 2:
            raise ValueError("factors must be [stock, date]")
        if factors.shape != (len(ts_codes), len(dates)):
            raise ValueError("factor shape does not match ts_codes/dates")
        universe_mask = np.asarray(universe_mask, dtype=bool)
        if universe_mask.shape != factors.shape:
            raise ValueError("universe_mask shape does not match factors")

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
            benchmark_returns = equal_weight_benchmark_returns(
                target_ret, signal_indices, universe_mask
            )
        elif len(benchmark_returns) != len(signal_indices):
            raise ValueError("benchmark_returns must align with signal_range")
        prev_weights = np.zeros(n_stocks, dtype=np.float64)
        capital = self.initial_capital
        daily_returns: list[float] = []
        turnover_list: list[float] = []
        positions: list[dict[str, Any]] = []

        for t in signal_indices:
            entry_day = t + 1
            signal = factors[:, t]
            # Signal-date and entry-date eligibility together define the
            # selectable set: a stock must be a member when the signal is
            # observed and still be a member when the buy executes at the
            # t+1 open.  Non-finite signal cells are excluded inside the
            # selection itself.
            eligible = universe_mask[:, t] & universe_mask[:, entry_day]
            exit_day = t + 2
            selected, selectable_values = self._select_top_n(
                signal,
                entry_day,
                open_,
                high,
                low,
                pre_close,
                volume,
                ts_codes,
                side="buy",
                eligible=eligible,
            )
            target_weights = np.zeros(n_stocks, dtype=np.float64)

            # No-signal semantics (T1-02): a selectable cross-section
            # without dispersion (fewer than two distinct values) carries
            # no ranking information — the book is held as-is with zero
            # turnover and zero cost, never arbitrarily rebalanced.  The
            # one exception is the mask-exit contract: positions that lost
            # universe eligibility are still liquidated through the
            # ordinary sell path (force-held when sell-blocked), because a
            # member leaving the pool must never linger silently.
            if selectable_values and not has_cross_sectional_dispersion(
                selectable_values
            ):
                target_weights = prev_weights.copy()
                liquidate = (prev_weights > 0.0) & ~eligible
                if liquidate.any():
                    target_weights[liquidate] = 0.0
                    sell_blocked = self._blocked_mask(
                        entry_day,
                        open_,
                        high,
                        low,
                        pre_close,
                        volume,
                        ts_codes,
                        side="sell",
                    )
                    force_hold = liquidate & sell_blocked
                    target_weights[force_hold] = prev_weights[force_hold]
            else:
                weight = min(
                    1.0 / max(self.config.top_n, 1),
                    self.config.single_weight_cap,
                )
                raw = np.zeros(n_stocks, dtype=np.float64)
                for idx in selected:
                    raw[idx] = weight

                # Positions that must be reduced need to be tradable on the
                # sell side; a sell-blocked reduction is force-held at its
                # previous weight (the sell is deferred, never dropped).
                sell_blocked = self._blocked_mask(
                    entry_day,
                    open_,
                    high,
                    low,
                    pre_close,
                    volume,
                    ts_codes,
                    side="sell",
                )
                held = np.zeros(n_stocks, dtype=np.float64)
                for _ in range(n_stocks + 1):
                    tentative = held + raw
                    new_holds = (
                        sell_blocked
                        & (prev_weights > tentative)
                        & (held == 0.0)
                    )
                    if not new_holds.any():
                        break
                    for idx in np.where(new_holds)[0]:
                        held[idx] = prev_weights[idx]
                        raw[idx] = 0.0
                    raw = self._scale_to_budget(raw, held)
                target_weights = held + self._scale_to_budget(raw, held)
            # Quantize weights to 1e-12 (same as the reward basket): the
            # scale-to-budget arithmetic leaves ~1e-16 summation noise
            # which would fabricate phantom micro-orders that pay the
            # full minimum commission and break permutation invariance.
            target_weights = np.round(target_weights, 12)

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
        side: str,
        eligible: np.ndarray,
        st_mask: np.ndarray | None = None,
    ) -> tuple[list[int], list[float]]:
        """Top-n indices of the current signal column.

        ``eligible`` is the mandatory per-stock bool eligibility vector for
        this signal date (signal-date & entry-date universe membership):
        ineligible stocks are excluded from the selection regardless of
        their signal value, so a position can never be newly opened in a
        stock outside the universe.  ``st_mask`` optionally marks the
        stocks that are ST as of the exact execution date (same-day paper
        trading only); the historical engine leaves it None.

        Ties resolve by ``ts_code`` (a stable identifier, not the row
        index), so the selected set is invariant under stock-row
        permutation.  Returns ``(indices, selectable_values)``: the top-n
        indices and the signal values of every selectable stock (the
        caller's no-signal dispersion check consumes the full selectable
        cross-section, not just the top-n).
        """

        blocked = self._blocked_mask(
            exec_day,
            open_,
            high,
            low,
            pre_close,
            volume,
            ts_codes,
            side=side,
            st_mask=st_mask,
        )
        valid = [
            (i, float(signal[i]))
            for i in range(len(signal))
            if eligible[i] and not blocked[i] and np.isfinite(signal[i])
        ]
        selectable_values = [value for _, value in valid]
        # Primary key: signal value descending; secondary key: ts_code, so
        # exact ties never depend on the physical row order.
        valid.sort(key=lambda item: (-item[1], ts_codes[item[0]]))
        return [i for i, _ in valid[: self.config.top_n]], selectable_values

    @staticmethod
    def _scale_to_budget(raw: np.ndarray, held: np.ndarray) -> np.ndarray:
        """Scale fresh buys down to the cash budget left by force-held
        positions — the no-renormalization contract (T1-02).

        The book is never renormalized upward: per-name weights stay at
        ``min(1/top_n, cap)`` (or below), and when force-held positions
        consume the budget the fresh portion shrinks instead, so the total
        invested weight never exceeds 1 (no leverage, cash remainder).
        """

        budget = max(1.0 - float(held.sum()), 0.0)
        raw_sum = float(raw.sum())
        if raw_sum <= 0.0 or raw_sum <= budget:
            return raw
        return raw * (budget / raw_sum)

    @staticmethod
    def _blocked_mask(
        exec_day: int,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        pre_close: np.ndarray,
        volume: np.ndarray,
        ts_codes: list[str],
        side: str,
        st_mask: np.ndarray | None = None,
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
            side,
            st_mask=st_mask,
        )

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
        # A book that lost more than its capital (costs on a collapsed book)
        # compounds below zero; a busted account cannot lose more than 100%,
        # so clamp the total return before annualization — (1 + total) ** x
        # would otherwise leave the real axis for total < -1 and raise
        # TypeError on the complex result.
        total_return = float(max(equity[-1] - 1.0, -1.0))
        ann_return = float((1.0 + total_return) ** (252 / n) - 1.0)
        vol = float(returns.std(ddof=1) * math.sqrt(252)) if n > 1 else 0.0
        sharpe = (ann_return - 0.02) / (vol + 1e-9)
        sortino = sortino_ratio(returns)
        running_max = np.maximum.accumulate(np.asarray(equity))
        drawdown = np.clip(
            1.0 - np.asarray(equity) / (running_max + 1e-12), 0.0, 1.0
        )
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
    from ashare_data.gates import ProductionGateRunner
    from .data_loader import AshareDataLoader
    from .reward import signal_direction
    from .vm import StackVM, formula_decode
    from .vocab import FORMULA_VOCAB, resolve_formula_tokens

    setup_run_logging(run_name="backtest")
    parser = argparse.ArgumentParser(description="Backtest an A-share formula")
    parser.add_argument("--config", default=None)
    parser.add_argument("--formula-file", default=None)
    parser.add_argument("--output", default="data/backtest_result.json")
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
    )
    args = parser.parse_args()

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(data_config, min_eligible=args.min_eligible).require_production()
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

        import torch

        # load_data always builds the PIT mask; the formal entry never
        # executes without it.
        vm = StackVM(
            FORMULA_VOCAB,
            industry_codes=getattr(loader, "industry_codes", None),
            universe_mask=torch.tensor(
                loader.universe_mask, dtype=torch.bool
            ),
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
            train_target = loader.mask_by_universe(train_target)
            direction = signal_direction(
                factors[:, :signal_end].detach().cpu().numpy(),
                train_target[:, :signal_end],
                universe_mask=loader.universe_mask[:, :signal_end],
            )
        signal_np = float(direction) * factors.detach().cpu().numpy()

        engine = AshareBacktestEngine(backtest_config)
        result = engine.run(
            signal_np,
            {k: v.numpy() for k, v in loader.raw_data_cache.items()},
            loader.ts_codes,
            loader.dates,
            universe_mask=loader.universe_mask,
        )
        # The universe policy actually applied to the result, for provenance.
        # No data hash or lineage is recorded: the policy fields are what
        # makes two results comparable.
        policy = getattr(loader, "universe_policy", None)
        output = {
            "formula": tokens,
            "formula_text": formula_decode(tokens, FORMULA_VOCAB),
            "direction": direction,
            # Data provenance: the immutable dataset manifest the backtest
            # ran on (None for pre-T1-01 databases).
            "dataset_id": loader.dataset_id,
            "metrics": result.metrics,
            "dates": result.dates,
            "equity_curve": result.equity_curve,
            "benchmark": backtest_config.benchmark,
            "benchmark_equity": result.benchmark_equity,
            "positions": result.positions,
            "universe_policy": (
                {
                    "index_codes": list(policy.index_codes),
                    "min_listed_sessions": int(policy.min_listed_sessions),
                    "membership_end_inclusive": bool(
                        policy.membership_end_inclusive
                    ),
                    "degraded": bool(loader.universe_status.degraded)
                    if loader.universe_status is not None
                    else None,
                }
                if policy is not None
                else None
            ),
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
