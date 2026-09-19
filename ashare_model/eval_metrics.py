"""Signal measurement and OOS-trial stitching for the evaluation protocol.

Extracted from ``evaluation.py`` (P7 Phase A2) by reason-to-change: this
module owns *how one signal/trial is measured* — full-engine metrics, rank
IC (universe and tradable), the equal-weight benchmark row, per-candidate
aggregation and the stitched outer-OOS trial matrix (T4-01).  It changes
when metric/stitching *semantics* change — not when fold contracts,
statistical corrections, search backends or artifact schemas change.

Import direction: depends on :mod:`ashare_model.eval_folds` (leaf-ward);
never imports the ``evaluation`` facade.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ashare_data.config import BacktestConfig
from ashare_portfolio.rebalance import RebalancePolicy

from .backtest import AshareBacktestEngine, equal_weight_benchmark_returns
from .data_loader import AshareDataLoader
from .diagnostics import rank_ic_stats
from .eval_folds import Fold, epoch_slice
from .vm import StackVM
from .vocab import FORMULA_VOCAB

# Metrics aggregated across folds/seeds for every candidate.
METRIC_KEYS = (
    "total_return",
    "annual_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "average_turnover",
    "excess_return",
    "benchmark_return",
    "ic_mean",
    "ic_abs_mean",
    "icir",
)


def _tradable_ic_mask(
    universe_mask: np.ndarray,
    blocked_buy: np.ndarray,
    signal_count: int,
) -> np.ndarray:
    """Universe mask additionally excluding stocks the engine could not buy.

    A signal at column ``t`` is executed at ``t+1`` (the entry day), so the
    tradable IC reference for column ``t`` is the stock set that is
    signal-date eligible AND buyable at ``t+1``: ``universe_mask[:, t] &
    ~blocked_buy[:, t+1]``.  ``blocked_buy`` is the sliced ``[stock, date]``
    buy-block matrix (suspension / one-word limit-up), covering at least
    ``signal_count + 1`` columns.  Columns beyond the signal range keep the
    raw universe mask (they never enter an IC).  Pure helper so the entry-
    day alignment is unit-testable.
    """

    universe_mask = np.asarray(universe_mask, dtype=bool)
    blocked_buy = np.asarray(blocked_buy, dtype=bool)
    if blocked_buy.shape != universe_mask.shape:
        raise ValueError(
            f"blocked_buy shape {blocked_buy.shape} does not match "
            f"universe_mask shape {universe_mask.shape}"
        )
    if signal_count >= blocked_buy.shape[1]:
        raise ValueError(
            "blocked_buy must cover at least one entry column "
            f"(got {blocked_buy.shape[1]} columns for {signal_count} signals)"
        )
    tradable = universe_mask.copy()
    tradable[:, :signal_count] &= ~blocked_buy[:, 1 : signal_count + 1]
    return tradable


def lot_gap_fields(result, ts_codes) -> dict:
    """P37 R02 additive observability: the continuous-vs-lot gap.

    Per DUE session, the L1 distance between the constructor's continuous
    target weights and the lot-rounded realized book (the post-execution
    marked weights the engine records in ``positions``), plus the invested
    sum gap and the initial-funding order counts.  This is the measurable
    answer to "how far does the整手 account sit from the ideal weights"
    — cash leftover, unaffordable lots, fees and blocked names all land
    in the same deviation.  Read-only over the engine result; the engine
    and the constructor are untouched.  Fields are ``None`` when the run
    had no due session.
    """

    targets = result.target_weights or []
    positions = result.positions or []
    diag = result.construction_diagnostics or []
    buys = result.buy_weights or []
    sells = result.sell_weights or []
    index = {str(code): i for i, code in enumerate(ts_codes)}
    l1_gaps: list[float] = []
    invested_gaps: list[float] = []
    init_sessions = 0
    init_planned = 0
    init_filled = 0
    for j, session in enumerate(diag):
        if not session.get("rebalance_due"):
            continue
        target = np.asarray(targets[j], dtype=np.float64)
        realized = np.zeros(target.shape[0], dtype=np.float64)
        held = positions[j]
        for code, weight in zip(held["ts_codes"], held["weights"]):
            realized[index[str(code)]] = float(weight)
        l1_gaps.append(float(np.abs(target - realized).sum()))
        invested_gaps.append(
            abs(float(target.sum()) - float(realized.sum()))
        )
        if session.get("initial_funding"):
            init_sessions += 1
            init_planned += int(session.get("order_count", 0))
            init_filled += int(
                np.count_nonzero(np.asarray(buys[j], dtype=np.float64) > 1e-12)
            ) + int(
                np.count_nonzero(
                    np.asarray(sells[j], dtype=np.float64) > 1e-12
                )
            )
    return {
        "lotgap_l1_mean_due": (
            float(np.mean(l1_gaps)) if l1_gaps else None
        ),
        "lotgap_l1_max_due": float(np.max(l1_gaps)) if l1_gaps else None,
        "lotgap_invested_mean_due": (
            float(np.mean(invested_gaps)) if invested_gaps else None
        ),
        "funnel_initial_funding_sessions": init_sessions,
        "funnel_initial_funding_planned_orders": init_planned,
        "funnel_initial_funding_filled_orders": init_filled,
    }


def evaluate_signal(
    signal: np.ndarray,
    loader: AshareDataLoader,
    fold: Fold,
    bt_cfg: BacktestConfig,
) -> dict:
    """Full-engine metrics plus rank IC for one signal on one test window.

    Single code path for trained formulas, single-factor baselines and the
    benchmark row: everything is scored by the same backtest engine with
    real costs, the blocked mask and the equal-weight benchmark.  Both IC
    variants are reported: the universe-masked rank IC and the tradable IC
    (additionally excluding stocks the engine could not buy on the entry
    day).
    """

    config_policy = RebalancePolicy.from_config(bt_cfg)
    if config_policy != fold.policy:
        raise ValueError(
            f"fold policy {fold.policy!r} does not match BacktestConfig "
            f"policy {config_policy!r}"
        )
    fold_data = epoch_slice(loader, fold)
    _, raw, target, dates = fold_data
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape != (len(loader.ts_codes), len(dates)):
        raise ValueError(
            f"signal shape {signal.shape} does not match "
            f"({len(loader.ts_codes)}, {len(dates)})"
        )
    result = AshareBacktestEngine(bt_cfg).run(
        signal,
        raw,
        loader.ts_codes,
        dates,
        signal_range=fold_data.local_signal_range,
        universe_mask=fold_data.universe_mask,
        rebalance_mask=fold_data.rebalance_mask,
    )
    m = result.metrics
    bench_total = (
        float(result.benchmark_equity[-1] - 1.0)
        if result.benchmark_equity
        else 0.0
    )
    excess = (
        (1.0 + m["total_return"]) / (1.0 + bench_total) - 1.0
        if bench_total > -1.0
        else 0.0
    )
    signal_count = fold_data.signal_count
    ic = rank_ic_stats(
        signal[None, :, :signal_count],
        target[:, :signal_count],
        dates[:signal_count],
        names=["formula"],
        eligible=fold_data.universe_mask[:, :signal_count],
    )["formula"]
    # Tradable IC: the same statistic restricted to stocks the engine
    # could actually buy on the entry day (suspension / one-word limit-up
    # opens excluded).  Presented alongside the primary IC so drift from
    # untradable cells is visible instead of silently folded into the
    # signal-quality estimate.
    blocked_buy, _ = loader.tradability_masks()
    tradable = _tradable_ic_mask(
        fold_data.universe_mask,
        blocked_buy[:, fold.contract.test_signal_start : fold.contract.test_price_end],
        signal_count,
    )
    ic_tradable = rank_ic_stats(
        signal[None, :, :signal_count],
        target[:, :signal_count],
        dates[:signal_count],
        names=["formula"],
        eligible=tradable[:, :signal_count],
    )["formula"]
    bench_daily: list[float] = []
    if result.benchmark_equity and len(result.benchmark_equity) >= 2:
        eq = result.benchmark_equity
        bench_daily = [float(eq[i + 1] / eq[i] - 1.0) for i in range(len(eq) - 1)]

    # Test-window active IR (p34 rev 3): the authority formula on this
    # row's own daily series.  Named ``oos_active_ir`` so it can never be
    # confused with the scorer's train-window ``active_ir`` that the
    # baseline rows already carry.
    from .baseline_harness import oos_active_ir

    # P34 rev 3 execution funnel (additive observability): per-row counters
    # separating "no order was needed" from "an order was needed and
    # blocked".  Due sessions partition into no_signal (degenerate signal),
    # executed_clean / partial_suppressed (orders resulted), fully_suppressed
    # (every planned trade dropped by threshold/min-trade) and target_stable
    # (the rank target equals the drifted book — normal low turnover).
    # The constructor's diagnostics carry the order/suppression counts; the
    # signal-degeneracy discriminator recomputes the constructor's own
    # has-signal test from the same mask matrices and the same dispersion
    # authority (``has_cross_sectional_dispersion``), so an empty book (no
    # selectable names) never masquerades as quiet.  Planned counts come
    # from the constructor diagnostics; filled counts from the engine's
    # executed buy/sell weights — their gap is the cash-capped /
    # missing-bar unfilled share.
    from ashare_data.processor import has_cross_sectional_dispersion

    diag = result.construction_diagnostics
    blocked_slice = blocked_buy[
        :,
        fold.contract.test_signal_start : fold.contract.test_price_end,
    ]
    no_signal_sessions = 0
    target_stable_sessions = 0
    fully_supp_sessions = 0
    partial_supp_sessions = 0
    executed_clean_sessions = 0
    # ``local_signal_range`` is ``range(signal_count)``: diagnostics row j
    # is fold-window session j, its entry session j+1.
    for t, session in enumerate(diag):
        if not session.get("rebalance_due"):
            continue
        orders = int(session.get("order_count", 0))
        supp = int(session.get("suppressed_trade_count", 0))
        if orders == 0 and supp == 0:
            eligible_t = (
                fold_data.universe_mask[:, t]
                & fold_data.universe_mask[:, t + 1]
            )
            selectable = (
                np.isfinite(signal[:, t])
                & eligible_t
                & ~blocked_slice[:, t + 1]
            )
            if has_cross_sectional_dispersion(signal[:, t][selectable]):
                target_stable_sessions += 1
            else:
                no_signal_sessions += 1
        elif orders == 0:
            fully_supp_sessions += 1
        elif supp > 0:
            partial_supp_sessions += 1
        else:
            executed_clean_sessions += 1
    filled_buy = filled_sell = 0
    executed_turnover = 0.0
    for buy_w, sell_w in zip(result.buy_weights, result.sell_weights):
        buy_arr = np.asarray(buy_w, dtype=np.float64)
        sell_arr = np.asarray(sell_w, dtype=np.float64)
        filled_buy += int(np.count_nonzero(buy_arr > 1e-12))
        filled_sell += int(np.count_nonzero(sell_arr > 1e-12))
        executed_turnover += float(buy_arr.sum() + sell_arr.sum())
    holdings_counts = [len(p["ts_codes"]) for p in result.positions]
    invested_ratios = [
        float(np.sum(w)) for w in result.target_weights
    ] if result.target_weights else []
    funnel = {
        "funnel_sessions_total": len(diag),
        "funnel_not_due_sessions": sum(
            1 for d in diag if not d.get("rebalance_due")
        ),
        "funnel_due_sessions": sum(
            1 for d in diag if d.get("rebalance_due")
        ),
        "funnel_due_no_signal_sessions": no_signal_sessions,
        "funnel_due_target_stable_sessions": target_stable_sessions,
        "funnel_due_fully_suppressed_sessions": fully_supp_sessions,
        "funnel_due_partial_suppressed_sessions": partial_supp_sessions,
        "funnel_due_executed_clean_sessions": executed_clean_sessions,
        "funnel_planned_order_count": sum(
            int(d.get("order_count", 0)) for d in diag
        ),
        "funnel_filled_order_count": filled_buy + filled_sell,
        "funnel_filled_buy_count": filled_buy,
        "funnel_filled_sell_count": filled_sell,
        "funnel_planned_turnover_sum": sum(
            float(d.get("turnover", 0.0)) for d in diag
        ),
        "funnel_executed_turnover_sum": executed_turnover,
        "funnel_suppressed_threshold_count": sum(
            len(d.get("threshold_dropped", ())) for d in diag
        ),
        "funnel_suppressed_min_trade_count": sum(
            len(d.get("min_trade_dropped", ())) for d in diag
        ),
        "funnel_holdings_min": int(min(holdings_counts)) if holdings_counts else 0,
        "funnel_invested_ratio_min": (
            float(min(invested_ratios)) if invested_ratios else 0.0
        ),
        "funnel_sessions_zero_holdings": sum(
            1 for count in holdings_counts if count == 0
        ),
        # P37 R02 (additive observability): the continuous-vs-lot gap on
        # due sessions (target weights vs the lot-rounded realized book)
        # and the initial-funding order counts — read-only over the same
        # engine result, existing fields bit-for-bit unchanged.
        **lot_gap_fields(result, list(loader.ts_codes)),
    }
    return {
        "n_dates": signal_count,
        "total_return": float(m["total_return"]),
        "annual_return": float(m["annual_return"]),
        "sharpe": float(m["sharpe"]),
        "sortino": float(m["sortino"]),
        "max_drawdown": float(m["max_drawdown"]),
        "calmar": float(m["calmar"]),
        "average_turnover": float(m["average_turnover"]),
        "benchmark_return": bench_total,
        "excess_return": excess,
        "oos_active_ir": oos_active_ir(
            np.asarray(result.daily_returns, dtype=np.float64),
            np.asarray(bench_daily, dtype=np.float64),
        ),
        "ic_mean": float(ic["ic_mean"]),
        "ic_abs_mean": float(ic["ic_abs_mean"]),
        "icir": float(ic["icir"]),
        "n_ic_dates": int(ic["n_dates"]),
        "ic_mean_tradable": float(ic_tradable["ic_mean"]),
        "icir_tradable": float(ic_tradable["icir"]),
        # P10 §5.2 组合活动门 (additive observability): the portfolio
        # activity counters the engine already computes, surfaced so the
        # searcher comparison can disclose suppressed/low-activity books.
        "rebalance_count": int(m.get("rebalance_count", 0)),
        "order_count": int(m.get("order_count", 0)),
        "suppressed_trade_count": int(m.get("suppressed_trade_count", 0)),
        # P34 R01 (additive observability, p34 §4): the remaining engine
        # diagnostics the baseline report needs — due rebalances, realized
        # book size / invested fraction and the total cost fraction.  The
        # engine already computes all of them; existing fields keep their
        # values bit-for-bit.
        "rebalance_due_count": int(m.get("rebalance_due_count", 0)),
        "holdings_mean": (
            float(np.mean([len(p["ts_codes"]) for p in result.positions]))
            if result.positions
            else 0.0
        ),
        "holdings_max": (
            int(max(len(p["ts_codes"]) for p in result.positions))
            if result.positions
            else 0
        ),
        "invested_ratio_mean": (
            float(np.mean([float(np.sum(w)) for w in result.target_weights]))
            if result.target_weights
            else 0.0
        ),
        "cost_fraction_sum": (
            float(np.sum(result.cost_fractions)) if result.cost_fractions else 0.0
        ),
        **funnel,
        # Raw per-day series: kept in every row so the DS / max-t corrections
        # and later analysis never have to reconstruct them from aggregates.
        "daily_returns": [float(x) for x in result.daily_returns],
        "benchmark_daily_returns": bench_daily,
        # p40 R1: the real mark dates of the daily series — the engine's own
        # date axis, one mark date per net return — so downstream stitching
        # and the corrections align trials by real trading dates instead of
        # array position.
        "daily_dates": list(result.dates[1:]),
    }


def evaluate_formula(
    tokens: list[int],
    loader: AshareDataLoader,
    fold: Fold,
    bt_cfg: BacktestConfig,
    vm: StackVM | None = None,
    direction: int = 1,
) -> dict | None:
    """Execute a formula on the full factor history, slice the test window,
    and score it in its learned trade direction (``direction`` flips the
    signal; the direction is decided on data strictly before the test
    window by the callers).  Returns ``None`` for an invalid formula."""

    if loader.universe_mask is None:
        # load_data always builds the mask; this guard keeps the failure
        # mode explicit for callers that skipped loading.
        raise ValueError(
            "loader carries no universe mask; production evaluation "
            "requires the PIT eligibility mask"
        )
    vm = vm or StackVM(
        FORMULA_VOCAB,
        universe_mask=torch.tensor(loader.universe_mask, dtype=torch.bool),
    )
    vm.industry_codes = getattr(loader, "industry_codes", None)
    # Force the loader's full-window PIT mask even onto an injected VM: the
    # production evaluation path can never execute without the mask.
    vm.universe_mask = torch.tensor(loader.universe_mask, dtype=torch.bool)
    signal = vm.execute(list(tokens), loader.factor_tensor)
    if signal is None:
        return None
    contract = fold.contract
    sliced = signal.detach().cpu().numpy()[
        :, contract.test_signal_start : contract.test_price_end
    ]
    return evaluate_signal(float(direction) * sliced, loader, fold, bt_cfg)


def benchmark_row(
    loader: AshareDataLoader,
    fold: Fold,
) -> dict:
    """Equal-weight benchmark scored on the engine's exact reference path:
    the mean forward target over signal-date AND entry-date eligible cells
    (cost-free buy-and-hold reference, no turnover).  No longer a separate
    per-stock average: the single
    :func:`ashare_model.backtest.equal_weight_benchmark_returns` helper
    computes it, so the row always equals the engine's benchmark curve."""

    fold_data = epoch_slice(loader, fold)
    _, _, target, dates = fold_data
    daily = equal_weight_benchmark_returns(
        fold_data.realized_ret,
        list(fold_data.local_signal_range),
        fold_data.universe_mask,
    )
    # p40 R1: the benchmark row carries the same real mark dates as the
    # engine-scored rows (next-open execution, delay=1: the return of
    # signal session t is marked at dates[t+2]); the agreement with
    # evaluate_signal's own date axis is pinned by test
    # test_n5_evaluate_signal_and_benchmark_row_carry_mark_dates.
    mark_dates = [dates[t + 2] for t in fold_data.local_signal_range]
    assert len(mark_dates) == len(daily)
    equity = [1.0]
    for ret in daily:
        equity.append(equity[-1] * (1.0 + ret))
    m = AshareBacktestEngine._metrics(daily, equity)
    return {
        "candidate": "benchmark:equal_weight",
        "formula_text": "equal_weight",
        "formula": None,
        "fold_train_end": fold.train_end,
        "fold_test_end": fold.test_end,
        "seed": None,
        "val_reward": None,
        "final_avg_reward": None,
        "failed": False,
        "n_dates": fold_data.signal_count,
        "total_return": float(m.get("total_return", 0.0)),
        "annual_return": float(m.get("annual_return", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "sortino": float(m.get("sortino", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "calmar": float(m.get("calmar", 0.0)),
        "average_turnover": None,
        "benchmark_return": float(m.get("total_return", 0.0)),
        "excess_return": 0.0,
        "ic_mean": 0.0,
        "ic_abs_mean": 0.0,
        "icir": 0.0,
        "n_ic_dates": 0,
        "ic_mean_tradable": 0.0,
        "icir_tradable": 0.0,
        "oos_active_ir": 0.0,
        "daily_returns": [float(x) for x in daily],
        "benchmark_daily_returns": [float(x) for x in daily],
        "daily_dates": mark_dates,
    }


def aggregate_results(rows: list[dict]) -> dict:
    """Per-candidate medians/IQR across folds and seeds.

    Medians (not means): reward clipping and heavy tails make the mean a
    poor summary of a candidate's OOS behavior.
    """

    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["candidate"], []).append(row)

    out: dict = {}
    for name, group in groups.items():
        entry: dict = {"n_rows": len(group), "metrics": {}}
        for key in METRIC_KEYS:
            values = [
                v
                for r in group
                for v in [r.get(key)]
                if v is not None and math.isfinite(float(v))
            ]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            entry["metrics"][key] = {
                "median": float(np.median(arr)),
                "q25": float(np.quantile(arr, 0.25)),
                "q75": float(np.quantile(arr, 0.75)),
                "iqr": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": int(arr.size),
            }
        if group[0].get("formula_text"):
            entry["formula_text"] = group[0]["formula_text"]
        out[name] = entry
    return out


# --- stitched OOS trial matrix (T4-01, v20) ----------------------------------
#
# One trial = one (candidate, seed) stitched outer-OOS series.  The raw
# rows are the per-fold outer evaluations of the nested walk-forward;
# stitching concatenates their daily return series in chronological fold
# order **before** any statistic is computed, so Sharpe / DSR / max-t
# measure the algorithm's full OOS path, not a per-fold average of paths.
# A failed fold is recorded (``failed_folds``) and contributes no
# returns; a (candidate, seed) with no succeeded fold produces no trial.


def _algorithm_of(candidate: str) -> str:
    """Algorithm family of a row's candidate name (``baseline:X`` -> ``baseline``)."""

    return str(candidate).split(":", 1)[0]


def stitch_oos_series(rows: list[dict]) -> list[dict]:
    """Stitch the outer OOS returns of every (candidate, seed) across folds.

    Returns JSON-friendly trial dicts: concatenated daily / benchmark /
    excess series in chronological fold order with their real mark-date
    axis (``dates``, p40 R2), per-segment provenance, recorded failed
    folds, and fold-level portfolio statistics aggregated over the
    succeeded segments (mean turnover, max capacity utilization).  Two
    succeeded rows for the same (candidate, seed, fold) raise
    ``ValueError``: a fold measured twice is an integrity error, not a
    stitchable observation.  A succeeded row whose ``daily_dates`` are
    missing or mismatched with its series raises ``ValueError`` (p40: a
    row that cannot be placed on a real date axis cannot be stitched),
    as does a duplicated mark date inside one trial.
    """

    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for row in rows:
        key = (row.get("candidate"), row.get("seed"))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    trials: list[dict] = []
    for key in order:
        candidate, seed = key
        group = groups[key]
        succeeded = [
            r for r in group
            if not r.get("failed") and r.get("daily_returns")
        ]
        if not succeeded:
            continue
        failed = [r for r in group if r.get("failed")]

        def _fold_key(row: dict) -> tuple[str, str]:
            return (
                str(row.get("fold_train_end") or ""),
                str(row.get("fold_test_end") or ""),
            )

        seen: set[tuple[str, str]] = set()
        segments: list[dict] = []
        for row in sorted(succeeded, key=_fold_key):
            fk = _fold_key(row)
            if fk in seen:
                raise ValueError(
                    f"duplicate fold row for candidate={candidate} "
                    f"seed={seed} fold={fk}"
                )
            seen.add(fk)
            segments.append(row)

        daily: list[float] = []
        bench: list[float] = []
        mark_dates: list[str] = []
        for row in segments:
            d = [float(x) for x in row["daily_returns"]]
            b = row.get("benchmark_daily_returns")
            b = [float(x) for x in b] if b and len(b) == len(d) else [0.0] * len(d)
            # p40 R2: every stitched segment must carry its real mark dates,
            # 1:1 with its daily series.  A succeeded row without them is a
            # pre-p40 artifact row: it cannot be aligned with anything, so
            # stitching it is an integrity error, never a silent position
            # concatenation (docs/p40_statistical_resampling_draft.md §2).
            dd = row.get("daily_dates")
            if not dd or len(dd) != len(d):
                raise ValueError(
                    f"candidate={candidate} seed={seed} fold="
                    f"{(row.get('fold_train_end'), row.get('fold_test_end'))}: "
                    "succeeded row carries no daily_dates aligned with its "
                    "daily_returns (pre-p40 artifact rows cannot be "
                    "stitched; re-measure the source run under p40+)"
                )
            daily.extend(d)
            bench.extend(b)
            mark_dates.extend(str(x) for x in dd)
        if len(mark_dates) != len(daily):
            raise ValueError(
                f"candidate={candidate} seed={seed}: stitched date axis "
                f"length {len(mark_dates)} != series length {len(daily)}"
            )
        if len(set(mark_dates)) != len(mark_dates):
            raise ValueError(
                f"candidate={candidate} seed={seed}: duplicate date on the "
                "stitched mark-date axis"
            )
        excess = [a - b for a, b in zip(daily, bench)]
        turnovers = [
            float(r["average_turnover"])
            for r in segments
            if r.get("average_turnover") is not None
        ]
        caps = [
            float(r["capacity_utilization"])
            for r in segments
            if r.get("capacity_utilization") is not None
        ]
        last = segments[-1]
        trials.append(
            {
                "candidate": candidate,
                "seed": seed,
                "algorithm": _algorithm_of(candidate),
                "segments": [
                    {
                        "fold_train_end": r.get("fold_train_end"),
                        "fold_test_end": r.get("fold_test_end"),
                        "n_dates": r.get("n_dates"),
                    }
                    for r in segments
                ],
                "failed_folds": [
                    {
                        "fold_train_end": r.get("fold_train_end"),
                        "fold_test_end": r.get("fold_test_end"),
                    }
                    for r in failed
                ],
                "n_days": len(daily),
                "daily_returns": daily,
                "benchmark_daily_returns": bench,
                "excess_returns": excess,
                # p40 R2: the trial's real mark-date axis, in fold order.
                "dates": mark_dates,
                "formula_text": last.get("formula_text"),
                "fold_test_end": last.get("fold_test_end"),
                "average_turnover_mean": (
                    float(sum(turnovers) / len(turnovers)) if turnovers else None
                ),
                "capacity_utilization_max": float(max(caps)) if caps else None,
            }
        )
    return trials


def stitched_metrics(trial: dict) -> dict:
    """Full-engine metrics on one stitched trial's concatenated series."""

    daily = [float(x) for x in trial.get("daily_returns") or []]
    if not daily:
        return {"n_days": 0}
    equity = AshareBacktestEngine._equity_curve(daily)
    m = AshareBacktestEngine._metrics(daily, equity)
    bench = [float(x) for x in trial.get("benchmark_daily_returns") or []]
    bench_total = 0.0
    if bench:
        bench_equity = AshareBacktestEngine._equity_curve(bench)
        bench_total = float(max(bench_equity[-1] - 1.0, -1.0))
    total = float(m.get("total_return", 0.0))
    excess = (1.0 + total) / (1.0 + bench_total) - 1.0 if bench_total > -1.0 else 0.0
    return {
        "n_days": len(daily),
        "total_return": total,
        "annual_return": float(m.get("annual_return", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "sortino": float(m.get("sortino", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "calmar": float(m.get("calmar", 0.0)),
        "benchmark_return": bench_total,
        "excess_return": excess,
    }


def top_trial(rows: list[dict]) -> dict | None:
    """Highest-OOS-Sharpe **stitched** trial (v20).

    Deliberately ignores ``val_reward``: adjudication is
    reward-version-independent by construction.  Rows without daily
    return series produce no trial.
    """

    best: dict | None = None
    for trial in stitch_oos_series(rows):
        metrics = stitched_metrics(trial)
        if metrics["n_days"] < 2:
            continue
        candidate = {**trial, **metrics}
        if best is None or candidate["sharpe"] > best["sharpe"]:
            best = candidate
    return best
