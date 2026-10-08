"""P59 local development measurement: stitch frozen signals and audit wealth.

No selection, promotion, alternate execution or fee implementation lives here.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .r01_baseline import combo_signal
from .time_contract import _date_key

ACCOUNT_SUMMARY_VERSION = 1
MEMBERS = ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")


def create_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def first_rebalance_mask(mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    due = np.flatnonzero(np.asarray(mask, dtype=bool)[:-2])
    if due.size:
        result[due[0]] = True
    return result


def assemble_frozen_signal(factors, universe_mask, dates, payload):
    """Cover every requested date exactly once, with causal frozen B directions."""
    if payload.get("frozen") is not True or payload.get("domain") != "medium":
        raise ValueError("directions must be frozen medium evidence")
    if payload.get("rules", {}).get("B") != "label_ic":
        raise ValueError("B must be the frozen label-IC direction rule")
    if not dates or dates != sorted(set(dates)):
        raise ValueError("date axis must be strictly increasing")
    shape = np.asarray(universe_mask).shape
    if shape != (len(next(iter(factors.values()))), len(dates)):
        raise ValueError("mask/date shape mismatch")
    if set(factors) != set(MEMBERS) or any(np.asarray(v).shape != shape for v in factors.values()):
        raise ValueError("three factor matrices must align with mask")
    entries = [e for e in payload["entries"] if e["rule"] == "B" and e["candidate"] in MEMBERS]
    signal = np.full(shape, np.nan)
    segments = []
    for year in sorted({date[:4] for date in dates}):
        end = year + "1231"
        cutoff = str(int(year) - 1) + "1231"
        chosen = {}
        for name in MEMBERS:
            matches = [e for e in entries if _date_key(e["fold_test_end"]) == end and e["candidate"] == name]
            if len(matches) != 1:
                raise ValueError(f"missing/duplicate frozen direction: {year}/{name}")
            entry = matches[0]
            provenance = entry["provenance"]
            expected_label = {"name": "causal_target_returns", "basis": "adjusted_open_o2o", "horizon": 10}
            if (entry["direction"] not in (-1, 1) or _date_key(provenance["train_cutoff"]) != cutoff
                or provenance["direction"] != entry["direction"] or provenance["candidate"] != name
                or provenance["rule"] != "label_ic" or provenance["label"] != expected_label):
                raise ValueError(f"noncausal/mismatched direction: {year}/{name}")
            chosen[name] = int(entry["direction"])
        columns = np.flatnonzero(np.asarray([d[:4] == year for d in dates]))
        if dates[int(columns[0])] <= cutoff:
            raise ValueError("training cutoff must precede signal")
        signal[:, columns] = combo_signal([(chosen[n], np.asarray(factors[n])[:, columns])
            for n in MEMBERS], np.asarray(universe_mask)[:, columns])
        segments.append(dict(year=year, train_cutoff=cutoff, fold_test_end=end,
            first_signal_date=dates[int(columns[0])], last_signal_date=dates[int(columns[-1])],
            date_count=len(columns), directions=chosen))
    return signal, segments


def annual_returns(dates, equity):
    if len(dates) != len(equity) or not dates or dates != sorted(set(dates)):
        raise ValueError("equity/date alignment must be unique and ordered")
    previous = float(equity[0])
    returns = {}
    for year in sorted({d[:4] for d in dates}):
        end = float(equity[max(i for i, d in enumerate(dates) if d[:4] == year)])
        returns[year] = end / previous - 1
        previous = end
    return returns


def account_periods(result):
    if len(result.positions) != len(result.construction_diagnostics):
        raise ValueError("positions/diagnostics do not align")
    return [dict(**p, **d["account"]) for p, d in zip(result.positions, result.construction_diagnostics)]


def summarize_account(result, initial_capital, raw, codes):
    """Use existing metric authority and recorded fills, never target notionals."""
    if not result.positions:
        raise ValueError("no complete daily account periods")
    annual = annual_returns(result.dates, result.equity_curve)
    periods = account_periods(result)
    fee_names = ("commission", "stamp_tax", "transfer_fee", "slippage", "total")
    fees = {name: float(sum(t[name] for t in result.trades)) for name in fee_names}
    if not np.isclose(sum(fees[n] for n in fee_names[:-1]), fees["total"], rtol=1e-12, atol=1e-8):
        raise ValueError("fee components do not reconcile")
    fee_ledger = sum(p["fees"] for p in periods)
    if not np.isclose(fees["total"], fee_ledger, rtol=1e-12, atol=1e-8):
        raise ValueError("recorded fees do not reconcile to account")
    for i, p in enumerate(periods):
        if (not np.isclose(p["equity_before"] - p["fees"], p["equity_after_execution"], rtol=1e-12, atol=1e-7)
            or not np.isclose(p["equity_exit"], result.equity_curve[i+1] * initial_capital, rtol=1e-12, atol=1e-7)):
            raise ValueError("daily wealth does not reconcile")
    cash = [p["cash"] for p in periods]
    if min(cash) < 0:
        raise ValueError("negative cash")
    exposure = [p["actual_exposure"] for p in periods]
    count = [len(p["ts_codes"]) for p in result.positions]
    rejection = Counter()
    suppression = Counter()
    blocks = Counter()
    for diagnostics in result.construction_diagnostics:
        rejection.update(r["reason"] for r in diagnostics["execution_rejections"])
        for key in ("threshold_dropped", "min_trade_dropped", "budget_skipped_no_viable", "budget_cash_capped"):
            suppression[key] += len(diagnostics.get(key, ()))
        blocks.update(diagnostics["market_block_counts"])
    held = set(result.positions[-1]["ts_codes"])
    terminal = np.asarray(raw["open"])[:, -1]
    missing = [c for i, c in enumerate(codes) if c in held and (not np.isfinite(terminal[i]) or terminal[i] <= 0)]
    return dict(account_summary_version=ACCOUNT_SUMMARY_VERSION, initial_capital=initial_capital,
        first_mark_date=result.dates[0], final_mark_date=result.dates[-1], periods=len(result.daily_returns),
        terminal_net_assets=initial_capital * result.equity_curve[-1],
        net_profit_rmb=initial_capital * (result.equity_curve[-1] - 1),
        net_return=result.metrics["total_return"], net_cagr=result.metrics["annual_return"],
        max_drawdown=result.metrics["max_drawdown"], annual_returns=annual,
        worst_year=min(annual, key=annual.get), worst_year_return=min(annual.values()),
        actual_exposure_mean=float(np.mean(exposure)), actual_exposure_min=min(exposure),
        actual_exposure_final=exposure[-1], actual_holdings_mean=float(np.mean(count)),
        actual_holdings_min=min(count), actual_holdings_final=count[-1],
        fraction_days_below_20=float(np.mean(np.asarray(count) < 20)),
        actual_max_single_weight=max(p["actual_max_single_weight"] for p in periods),
        dispersion_status="实际持仓不足20只的天数需审查" if min(count) < 20 else "实际持仓达到20只",
        actual_turnover_total=float(sum(p["actual_turnover"] for p in periods)),
        actual_turnover_daily_mean=float(np.mean([p["actual_turnover"] for p in periods])),
        plan_turnover_total=float(sum(result.turnover)), plan_order_count=int(result.metrics["order_count"]),
        actual_order_count=len(result.trades), cash_min=min(cash), final_cash=result.final_cash,
        fees=fees, execution_rejections=dict(rejection), constructor_suppression=dict(suppression),
        market_block_counts=dict(blocks), market_block_scope="可选信号股票/当前持仓的约束观察数，非已发订单",
        valuation_complete=not missing, terminal_missing_prices=missing,
        theoretical_equal_weight_return=result.benchmark_equity[-1] - 1)
