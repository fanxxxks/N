"""Cross-sectional daily backtest and reward scoring."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from loguru import logger

from ashare_data.config import BacktestConfig
from ashare_data.processor import (
    open_to_open_returns,
    tradability_blocked,
)
from ashare_data.schemas import BacktestResult
from ashare_execution import (
    ExecutionCostModel,
    validate_execution_config,
)
from ashare_logging import (
    canonical_config_sha256,
    emit_run_identity,
    export_log_txt,
    setup_run_logging,
)
from ashare_portfolio.constructor import PortfolioConstructor
from ashare_portfolio.rebalance import RebalancePolicy

from .reward import sortino_ratio
from .targets import causal_target_returns
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
    def __init__(
        self,
        config: BacktestConfig | None = None,
        *,
        portfolio_constructor: PortfolioConstructor | None = None,
    ):
        self.config = config or BacktestConfig()
        self.initial_capital = float(self.config.initial_capital)
        self.cost_model = ExecutionCostModel.from_config(self.config)
        self.portfolio_constructor = (
            portfolio_constructor or PortfolioConstructor(self.config)
        )

    def run(
        self,
        factors: np.ndarray,
        raw_cache: dict[str, np.ndarray],
        ts_codes: list[str],
        dates: list[str],
        universe_mask: np.ndarray,
        benchmark_returns: list[float] | None = None,
        signal_range: range | tuple[int, int] | None = None,
        execution_delay: int = 1,
        rebalance_mask: np.ndarray | None = None,
    ) -> BacktestResult:
        """Execute the daily top-n strategy over the signal columns.

        ``universe_mask`` is the mandatory ``[stock, date]`` bool PIT
        eligibility mask: a position can only be newly opened in a stock
        eligible on the signal date **and** on the entry date
        (``universe_mask[:, t] & universe_mask[:, t + execution_delay]``),
        the default equal-weight benchmark averages only those cells (and
        requires a valid target), and a day without any eligible stock
        earns zero for both the strategy (flat book) and the benchmark.
        Existing positions of a stock that exits the universe are sold
        through the ordinary sell path — the mask never erases them
        silently.  A mask whose shape does not match the signal raises
        ``ValueError``; there is no unconstrained execution path.

        ``execution_delay`` is the number of trading days between the
        signal date and the entry open (1 = next-open execution, the
        historical contract; the golden execution spec stresses 2 =
        one-day-delayed execution with the same parity guarantees).

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
        delay = int(execution_delay)
        if delay < 1:
            raise ValueError("execution_delay must be >= 1")
        if rebalance_mask is None:
            rebalance_mask = RebalancePolicy.from_config(
                self.config
            ).rebalance_mask(dates)
        rebalance_mask = np.asarray(rebalance_mask, dtype=bool)
        if rebalance_mask.shape != (len(dates),):
            raise ValueError(
                f"rebalance_mask shape {rebalance_mask.shape} does not match "
                f"date axis ({len(dates)},)"
            )

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
            signal_indices = list(range(max(n_dates - (delay + 1), 0)))
        elif isinstance(signal_range, range):
            signal_indices = list(signal_range)
        else:
            signal_indices = list(range(int(signal_range[0]), int(signal_range[1])))
        if any(t < 0 or t + delay + 1 >= n_dates for t in signal_indices):
            raise ValueError(
                "signal_range includes a signal without a complete "
                f"t+{delay + 1} exit"
            )
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
        target_weights_list: list[np.ndarray] = []
        buy_weights_list: list[np.ndarray] = []
        sell_weights_list: list[np.ndarray] = []
        cost_fractions: list[float] = []
        construction_diagnostics: list[dict[str, Any]] = []

        for t in signal_indices:
            entry_day = t + delay
            signal = factors[:, t]
            # Signal-date and entry-date eligibility together define the
            # selectable set: a stock must be a member when the signal is
            # observed and still be a member when the buy executes at the
            # t+delay open.  Non-finite signal cells are excluded inside
            # the selection itself.
            eligible = universe_mask[:, t] & universe_mask[:, entry_day]
            exit_day = t + delay + 1
            buy_blocked = self._blocked_mask(
                entry_day,
                open_,
                high,
                low,
                pre_close,
                volume,
                ts_codes,
                side="buy",
            )
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
            output = self.portfolio_constructor.construct(
                signal,
                prev_weights,
                capital=capital,
                eligible=eligible,
                buy_blocked=buy_blocked,
                sell_blocked=sell_blocked,
                stable_keys=np.asarray(ts_codes),
                rebalance_due=bool(rebalance_mask[t]),
                adv=volume[:, entry_day] * open_[:, entry_day],
            )
            target_weights = np.asarray(output.weights)
            buy_weights = np.asarray(output.buy_weights)
            sell_weights = np.asarray(output.sell_weights)
            turnover = output.turnover

            cost_fraction = float(
                self.cost_model.rebalance_cost_fraction(
                    buy_weights, sell_weights, capital
                )
            )
            # target_ret[:, j] spans the open-to-open period (j+1, j+2),
            # which is exactly [entry_day, exit_day] with
            # entry_day = t + execution_delay.
            gross_ret = float(np.dot(target_weights, target_ret[:, entry_day - 1]))
            net_ret = gross_ret - cost_fraction
            daily_returns.append(net_ret)
            turnover_list.append(turnover)
            cost_fractions.append(cost_fraction)
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
            target_weights_list.append(target_weights.copy())
            buy_weights_list.append(buy_weights.copy())
            sell_weights_list.append(sell_weights.copy())
            construction_diagnostics.append(dict(output.diagnostics))
            prev_weights = target_weights

        equity = self._equity_curve(daily_returns)
        benchmark_equity = (
            self._equity_curve(benchmark_returns)
            if benchmark_returns is not None
            else None
        )
        metrics = self._metrics(daily_returns, equity)
        metrics["average_turnover"] = float(np.mean(turnover_list)) if turnover_list else 0.0
        metrics["rebalance_count"] = int(
            sum(bool(item["rebalance_executed"]) for item in construction_diagnostics)
        )
        metrics["rebalance_due_count"] = int(
            sum(bool(item["rebalance_due"]) for item in construction_diagnostics)
        )
        metrics["order_count"] = int(
            sum(int(item["order_count"]) for item in construction_diagnostics)
        )
        metrics["suppressed_trade_count"] = int(
            sum(
                int(item.get("suppressed_trade_count", 0))
                for item in construction_diagnostics
            )
        )
        logger.info(
            "portfolio path method={} frequency={} periods={} due={} "
            "rebalanced={} orders={} suppressed={} avg_turnover={:.6f}",
            self.config.portfolio_method,
            self.config.rebalance_frequency,
            len(signal_indices),
            metrics["rebalance_due_count"],
            metrics["rebalance_count"],
            metrics["order_count"],
            metrics["suppressed_trade_count"],
            metrics["average_turnover"],
        )
        return BacktestResult(
            equity_curve=[float(x) for x in equity],
            # Initial equity is marked at the first entry; subsequent points
            # are marked at each t+delay+1 exit, exactly one date per equity point.
            dates=(
                [dates[signal_indices[0] + delay]]
                + [dates[t + delay + 1] for t in signal_indices]
                if signal_indices
                else dates[:1]
            ),
            daily_returns=[float(x) for x in daily_returns],
            turnover=[float(x) for x in turnover_list],
            benchmark_equity=benchmark_equity,
            metrics=metrics,
            positions=positions,
            target_weights=target_weights_list,
            buy_weights=buy_weights_list,
            sell_weights=sell_weights_list,
            cost_fractions=cost_fractions,
            construction_diagnostics=construction_diagnostics,
        )

    @staticmethod
    def _open_to_open_returns(open_: np.ndarray) -> np.ndarray:
        return open_to_open_returns(open_)

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


def load_formal_strategy(path, store):
    """Load a current strategy and reconstruct its exact source RunSpec.

    Backtest is a lineage follower: it never derives a replacement spec
    from current defaults. Legacy schema/research generations and missing
    or mismatched RunStore evidence are refused before formula execution.
    """

    from .artifact_schemas import ArtifactSchemaError, StrategyArtifact
    from .artifact_versions import classify_strategy
    from .artifact_writer import (
        read_boundary_artifact,
        reconstruct_runspec_from_lineage,
    )

    try:
        loaded = read_boundary_artifact(
            path, model_cls=StrategyArtifact, formal=True
        )
    except ArtifactSchemaError as exc:
        raise SystemExit(f"Backtest refuses non-current strategy: {exc}") from exc
    if loaded is None:
        raise SystemExit(f"Formula file not found: {path}")
    payload, _ = loaded
    verdict = classify_strategy(payload)
    if verdict["legacy"]:
        raise SystemExit(
            "Backtest refuses legacy strategy generation: "
            + "; ".join(verdict["reasons"])
        )
    try:
        spec = reconstruct_runspec_from_lineage(store, payload)
    except ArtifactSchemaError as exc:
        raise SystemExit(
            f"Backtest cannot reconstruct source lineage: {exc}"
        ) from exc
    return payload, spec


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
    from ashare_data.manifest import check_dataset_id
    from .artifact_schemas import BacktestResultArtifact
    from .artifact_writer import write_boundary_artifact
    from .data_loader import AshareDataLoader
    from .run_store import RunStore
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
        from .runspec import new_run_id
        from .versions import PROTOCOL_VERSION
        from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION

        # IP-11 (03-F-07): identity quadruple as the first content line;
        # the full version matrix travels with the boundary artifact.
        emit_run_identity(
            run_id=new_run_id(),
            config_sha256=canonical_config_sha256(raw),
            versions={
                "protocol_version": PROTOCOL_VERSION,
                "execution_spec_version": EXECUTION_SPEC_VERSION,
            },
        )
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
        store = RunStore(data_config.data_dir)
        payload, source_spec = load_formal_strategy(formula_file, store)
        check_dataset_id(payload.get("dataset_id"), loader.dataset_id)
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
            policy = RebalancePolicy.from_config(backtest_config)
            contract = TrainingTimeContract.resolve(
                loader.dates,
                backtest_config.train_end_date,
                horizon=policy.horizon,
            )
            price_end = contract.train_label_end
            signal_end = contract.train_signal_end
            full_rebalance_mask = policy.rebalance_mask(loader.dates)
            train_target = causal_target_returns(
                loader.raw_data_cache["open"][:, :price_end].numpy(),
                loader.dates[:price_end],
                policy,
                rebalance_mask=full_rebalance_mask[:price_end],
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
        with store.open_run(source_spec) as handle:
            write_boundary_artifact(
                handle,
                artifact_type="backtest",
                model_cls=BacktestResultArtifact,
                payload=output,
                candidate_id=payload["candidate_id"],
                convenience_path=out_path,
            )
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        print(f"Result saved to {out_path}")
    finally:
        export_log_txt(run_name="backtest")


if __name__ == "__main__":
    main()
