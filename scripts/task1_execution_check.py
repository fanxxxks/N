"""Bounded P58 mechanical check; no data reads, signal search or return claims.

Run from the repository root: python -m scripts.task1_execution_check --output <NEW directory>
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import yaml

from ashare_data.config import BacktestConfig, make_backtest_config, make_sim_config
from ashare_execution import validate_execution_config
from ashare_model.backtest import AshareBacktestEngine
from ashare_portfolio.execution_spec import execution_provenance
from ashare_portfolio.rebalance import RebalancePolicy
from ashare_trading.matching import SimBroker
from ashare_trading.portfolio import SimulationPortfolio
from ashare_trading.run_sim import SimulationRunner

ROOT = Path(__file__).resolve().parents[1]


def check(config: BacktestConfig, *, replacement: bool = False) -> dict:
    n = config.buy_rank * (2 if replacement else 1)
    days = 13 if replacement else 3
    codes = [f"{i:06d}.SZ" for i in range(n)]
    dates = pd.bdate_range("2024-01-02", periods=days).strftime("%Y%m%d").tolist()
    opens = np.full((n, days), 10.)
    raw = dict(open=opens, high=opens * 1.01, low=opens * .99,
        pre_close=opens.copy(), volume=np.full_like(opens, 1e6),
        adjusted_open=opens.copy(), up_limit=np.full_like(opens, 11),
        down_limit=np.full_like(opens, 9), suspended=np.zeros_like(opens))
    signal = np.tile(np.arange(n, 0, -1.)[:, None], (1, days))
    if replacement:
        signal[:, 10:] = signal[::-1, 10:]
    result = AshareBacktestEngine(config).run(signal, raw, codes, dates,
        np.ones_like(signal, dtype=bool))
    due = RebalancePolicy.from_config(config).rebalance_mask(dates)
    fees = dict(commission=0., stamp_tax=0., transfer_fee=0., slippage=0.)
    skipped = []
    sessions = []
    with tempfile.TemporaryDirectory(prefix="p58_") as temp:
        portfolio = SimulationPortfolio(config.initial_capital, Path(temp) / "state.json")
        runner = SimulationRunner.__new__(SimulationRunner)
        runner.portfolio, runner.backtest_config = portfolio, config
        for t, target in enumerate(result.target_weights):
            equity = portfolio.cash + portfolio.market_value(dict(zip(codes, opens[:, t + 1])))
            orders = runner._make_orders(target, codes, t + 1, dates[t + 1], raw,
                equity=equity) if due[t] else []
            bars = pd.DataFrame([dict(ts_code=code, trade_date=dates[t + 1],
                **{key: raw[key][i, t + 1] for key in
                ("open", "high", "low", "pre_close", "volume", "up_limit", "down_limit", "suspended")})
                for i, code in enumerate(codes)])
            trades = SimBroker().execute_orders(orders, bars, dates[t + 1], portfolio,
                {}, config, lot_size=config.lot_size)
            for trade in trades:
                for key in fees:
                    fees[key] += float(getattr(trade, key))
            skipped.extend(dict(code=o.ts_code, reason=o.reason) for o in orders if o.status == "skipped")
            sessions.append(dict(date=dates[t + 1],
                buy_yuan=sum(tr.amount for tr in trades if tr.side == "buy"),
                sell_yuan=sum(tr.amount for tr in trades if tr.side == "sell"),
                cash=portfolio.cash))
            assert portfolio.cash >= 0
        quantities = [portfolio.positions[c].quantity if c in portfolio.positions else 0 for c in codes]
        np.testing.assert_allclose(quantities, result.final_shares, rtol=0, atol=1e-7)
        assert abs(portfolio.cash - result.final_cash) < 1e-7
        engine_fees = float(np.dot(result.cost_fractions, result.equity_curve[:-1]) * config.initial_capital)
        assert abs(engine_fees - sum(fees.values())) < 1e-7
    invested = sum(quantities) * 10
    return dict(capital=config.initial_capital, positions=np.count_nonzero(quantities).item(),
        target_positions=config.buy_rank, cap=config.single_weight_cap,
        min_trade_amount=config.min_trade_amount, lot_size=config.lot_size,
        single_target_yuan=config.initial_capital * min(1 / config.buy_rank, config.single_weight_cap),
        one_lot_price_ceiling=config.initial_capital * min(1 / config.buy_rank, config.single_weight_cap) / config.lot_size,
        invested_yuan=invested, actual_invested_ratio=invested / config.initial_capital,
        fees=fees, total_fees=sum(fees.values()), cash=result.final_cash,
        quantities=quantities, parity=True, skipped=skipped, sessions=sessions,
        planned_turnover=result.turnover,
        actual_turnover=[float(np.sum(b) + np.sum(s)) for b, s in zip(result.buy_weights, result.sell_weights)],
        construction_diagnostics=result.construction_diagnostics,
        provenance=execution_provenance(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)  # Never overwrite evidence.
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    profiles = []
    for capital in (10000, 30000, 50000, 100000):
        if capital == 10000:
            config = replace(BacktestConfig(), initial_capital=capital, top_n=25,
                buy_rank=25, sell_rank=35, single_weight_cap=.04, min_trade_amount=1000.,
                rebalance_frequency="every_10_days", target_horizon=10)
        else:
            raw = yaml.safe_load((ROOT / f"config/small_capital_{capital}.yaml").read_text(encoding="utf-8"))
            config = make_backtest_config(raw)
            validate_execution_config(config, make_sim_config(raw, ROOT))
        row = check(config)
        row["status"] = "不适用" if capital == 10000 else "10元合成输入下机械可行；历史待验证"
        profiles.append(row)
    example_cfg = replace(BacktestConfig(), top_n=20, buy_rank=20,
        sell_rank=30, min_trade_amount=1000., single_weight_cap=.05)
    examples = [check(replace(example_cfg, min_trade_amount=minimum)) for minimum in (1000., 5000.)]
    comparisons = [dict(allocation=mode, **check(replace(example_cfg,
        sell_rank=20, turnover_budget=.20, turnover_allocation=mode,
        rebalance_frequency="every_10_days", target_horizon=10), replacement=True))
        for mode in ("priority", "proportional")]
    payload = dict(kind="engineering", git_commit=sha, dirty_tree=dirty,
        price_assumption="所有股票10元，约束完整，无停牌涨跌停，价格不变；无收益推断",
        profiles=profiles, required_example=examples, replacement=comparisons)
    (args.output / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["capital", "positions", "target_positions", "cap", "single_target_yuan", "one_lot_price_ceiling",
        "invested_yuan", "actual_invested_ratio", "total_fees", "cash", "parity", "status"]
    with (args.output / "profiles.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(profiles)
    lines = ["# 任务一机械可行性结果", "", f"被测提交：{sha}；工作树改动：{dirty}。",
        "仅合成10元定价，不代表收益改善或全历史适用。目标25只、上限4%、最低1000元、100股一手。",
        "计划调仓周期与预测周期均保留medium的10交易日；此处不生成训练标签。", "",
        "| 本金 | 成交只数 | 买入金额 | 实际仓位 | 费用 | 剩余现金 | 状态 |",
        "|---:|---:|---:|---:|---:|---:|---|"]
    for row in profiles:
        lines.append(f"| {row['capital']:.0f} | {row['positions']} | {row['invested_yuan']:.2f} | "
            f"{row['actual_invested_ratio']:.2%} | {row['total_fees']:.2f} | {row['cash']:.2f} | {row['status']} |")
    lines += ["", "一手价格理论上限分别为4、12、20、40元；这只是单票约束，仍须同时预留全账户费用。",
        "每次用当时入选股票原始价格复核，不能为了满足上限重排信号或筛低价股票。",
        "费用已从现金扣除；所有案例回测与模拟盘股数、费用、现金一致，现金非负。",
        "", "指定20只示例：最低1000元买入99000元（19×500股+1×400股），费用150.49元、现金849.51元；",
        "最低5000元买入95000元（19×500股），费用143.45元、现金4856.55元。",
        "", "| 换股分配（相同信号/0.20预算） | 当次卖出 | 当次买入 | 当次剩余现金 |",
        "|---|---:|---:|---:|"]
    for row in comparisons:
        session = row["sessions"][10]
        lines.append(f"| {row['allocation']} | {session['sell_yuan']:.2f} | {session['buy_yuan']:.2f} | {session['cash']:.2f} |")
    lines += ["", "priority仍可耗尽卖出预算而不买入；proportional可把腿稀释到最低交易额以下。",
        "本轮未接trade_groups、未改变预算规则；没有据此选收益更优的方案。",
        "旧turnover/order_count仍是计划口径，选型须使用buy_weights/sell_weights和成交记录。"]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
