"""Run the six preregistered P59 development accounts, without input writes.

Usage: python scripts/task2_continuous_account.py --source-root ../AlphaGPT
       --output reports/task2_YYYYMMDD
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from loguru import logger

from ashare_data.config import make_backtest_config, make_data_config, make_model_config
from ashare_data.io_utils import atomic_write_json
from ashare_data.db import AshareDB
from ashare_data.manifest import latest_publication_group
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.continuous_account import (
    MEMBERS, assemble_frozen_signal, create_output_directory, first_rebalance_mask, account_periods,
    summarize_account,
    validate_snapshot_bridge,
)
from ashare_model.data_loader import AshareDataLoader
from ashare_model.ledger import ExperimentLedger
from ashare_model.runspec import _capture_versions, composite_snapshot_id
from ashare_model.vocab import FEATURE_NAMES
from ashare_portfolio.rebalance import RebalancePolicy

FROZEN_DATASET = "d36024cd1b197a8edcf976b2d950b3506bb0fae022f121d6aef06f2a281cde55"
BACKUP_DATASET = "767d955b0fd99507fed95b4275a2bdde5bdab177e1e22c8dc156a45c107e21b2"
FROZEN_FACTORS = {
    "MOMENTUM_20": "0bf61a6d89703f571616f0078cc52528ac788f5345d0b64eb6ae089f052ba72c",
    "RSQ_60": "5cbb94a224cf7805e4387271f691f1db48dfdbb1c81c57c93549430d12816587",
    "REVERSAL_60": "4a63dab2cf3de6a9762712a833070d68debf1e020dfc31e92c051b06a9b62a6a",
}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in row.items()})


def render_report(rows, provenance):
    lines = ["# 任务二：连续账户实际金额（开发验证）", "",
        "2021—2025年，medium B三因子等权，10交易日调仓、10日预测。仅一次投入本金。",
        "1万元不适用：既定25只/单股4%/最低1000元条件无法同时成立。", "",
        "|本金|主动期末净资产|净赚人民币|扣费年化|最大回撤|最差年度|被动净赚|被动收益|相对被动复利超额|",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        if r["mode"] != "active": continue
        lines.append(f'|{r["initial_capital"]:,.0f}|{r["terminal_net_assets"]:,.2f}|{r["net_profit_rmb"]:,.2f}|'
            f'{r["net_cagr"]:.2%}|{r["max_drawdown"]:.2%}|{r["worst_year"]}（{r["worst_year_return"]:.2%}）|'
            f'{r["passive_net_profit_rmb"]:,.2f}|{r["passive_net_return"]:.2%}|{r["compound_excess_vs_passive"]:.2%}|')
    lines += ["", "同本金被动基准：与主动账户同首期实际股票/股数/费用，随后持有。它比较后续调仓的增量，",
        "不代表宽基指数或所有被动选择。全股票池等权是理论参照，不声称可复制。", "",
        "|本金|主动净收益|理论等权收益|相对理论复利超额|主动比被动多赚人民币|平均实际仓位|平均/最少/末期持股|不足20只天数占比|实际双边累计换手|实际成交笔数|",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        if r["mode"] != "active": continue
        lines.append(f'|{r["initial_capital"]:,.0f}|{r["net_return"]:.2%}|{r["theoretical_equal_weight_return"]:.2%}|'
            f'{r["compound_excess_vs_theoretical"]:.2%}|{r["increment_vs_passive_rmb"]:,.2f}|{r["actual_exposure_mean"]:.2%}|'
            f'{r["actual_holdings_mean"]:.1f}/{r["actual_holdings_min"]}/{r["actual_holdings_final"]}|'
            f'{r["fraction_days_below_20"]:.2%}|{r["actual_turnover_total"]:.2f}倍|{r["actual_order_count"]}|')
    lines += ["", "|本金|佣金|印花税|过户费|滑点|合计|计划累计换手|最低实际现金|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["mode"] != "active": continue
        fees = r["fees"]
        lines.append(f'|{r["initial_capital"]:,.0f}|' + "|".join(f'{fees[k]:,.2f}' for k in
            ("commission", "stamp_tax", "transfer_fee", "slippage", "total")) +
            f'|{r["plan_turnover_total"]:.2f}倍|{r["cash_min"]:,.2f}|')
    lines += ["", "## 未成交与年度明细", ""]
    for r in rows:
        if r["mode"] != "active": continue
        lines += [f'### {r["initial_capital"]:,.0f}元', "",
            f'- 年度扣费收益：{json.dumps(r["annual_returns"], ensure_ascii=False)}',
            f'- 执行未成交原因：{json.dumps(r["execution_rejections"], ensure_ascii=False)}',
            f'- 构造阶段抑制：{json.dumps(r["constructor_suppression"], ensure_ascii=False)}',
            f'- 市场约束观察（非订单数）：{json.dumps(r["market_block_counts"], ensure_ascii=False)}',
            f'- 期末缺价持仓：{r["terminal_missing_prices"]}；完整估值：{r["valuation_complete"]}',
            f'- 实际最大单股权重（含价格漂移及扣费）：{r["actual_max_single_weight"]:.2%}', ""]
    lines += ["## 口径与局限", "",
        f'- 实际只读输入：{provenance["snapshot_directory"] or "当前数据库"}；dataset={provenance["dataset_id"]}。原数据库漂移失败已保留，备份按独立身份确认，历史必需输入和三个因子已核对一致。',
        "- 人民币是现有引擎的复权总回报近似估值：红利没有逐笔现金入账；执行时按复权财富重推股数，不能等同券商现金结算。",
        "- 期末仍持有股票，没有假想清仓，也没有扣假想清仓费用。若期末持仓缺价，估值不合格。",
        "- 各年账户收益接续上年末资产，无年度注资；年化沿用252/实际收益期数。现金、持仓跨年持续。",
        "- 实际换手=(买入成交额+卖出成交额)/该日成交前权益，双边口径；计划指标明确标为计划。",
        "- 未成交计数与构造抑制是不同阶段；停牌/涨跌停约束的观察数不能当成已发订单未成交数。",
        "- 真实价格可能使持仓长期低于原20只分散目标，应按实际持仓审查适用性，不通过提高单股上限补救。",
        "- 与原WP18同时存在N/cap/minimum/budget差异，结果变化不能全部归因于执行修复。",
        "- 固定股票费用率覆盖全期，沿用项目正常费率；不是逐年税制复现。",
        "- 2021—2025已反复用于开发，本表不是独立样本外、收益保证或实盘资格。任务三/四未执行。", "",
        f'实际账户计价区间：{rows[0]["first_mark_date"]}—{rows[0]["final_mark_date"]}；实现提交：{provenance["implementation_commit"]}。',
        "完整六账户/年度/费用/原因见results.json和summary.csv；daily_*.csv逐日权益与实际成交，trades_*.csv逐笔成本。",
        "provenance.json记录数据、版本、配置、方向和信号hash，ledger.jsonl保留完整成功/失败链。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-directory", choices=["backup_20260919_p38"],
        help="explicit preregistered recovery; no automatic fallback")
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output.resolve()
    create_output_directory(output)
    logger.add(output / "run.log")
    started = time.perf_counter()
    ledger = ExperimentLedger(output / "ledger.jsonl", run_id=output.name)
    input_trial = ledger.record_trial(algorithm="input_provenance", candidate="frozen_medium_B")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        direction_path = source / "reports/wp18/medium/direction_frozen.json"
        direction_bytes = direction_path.read_bytes()
        payload = json.loads(direction_bytes)
        source_config_path = source / "configs/r1_medium_a.yaml"
        source_config_bytes = source_config_path.read_bytes()
        # Explicit frozen YAML: no UI runtime overrides or environment path substitution.
        source_config = yaml.safe_load(source_config_bytes)
        configs = {}
        hashes = {}
        for capital in (30000, 50000, 100000):
            path = ROOT / f"config/small_capital_{capital}.yaml"
            configs[capital] = make_backtest_config(yaml.safe_load(path.read_bytes()))
            hashes[str(capital)] = digest_bytes(path.read_bytes())
        data_config = make_data_config(source_config, source)
        expected_dataset = FROZEN_DATASET
        snapshot_bridge = None
        if args.snapshot_directory is not None:
            backup_primary = source / "data" / args.snapshot_directory / "ashare.duckdb"
            groups = []
            for path in (data_config.duckdb_path, backup_primary):
                with AshareDB(path, read_only=True) as db:
                    group = latest_publication_group(db)
                if group is None:
                    raise ValueError(f"missing published composite group: {path}")
                groups.append(group)
            identities = [composite_snapshot_id(g.members["primary"], g.members["tushare_v2"])
                for g in groups]
            if identities != [FROZEN_DATASET, BACKUP_DATASET]:
                raise ValueError(f"published recovery identities drifted: {identities}")
            snapshot_bridge = validate_snapshot_bridge(
                {k: v.to_dict() for k, v in groups[0].members.items()},
                {k: v.to_dict() for k, v in groups[1].members.items()})
            snapshot_bridge["publication_ids"] = [g.publication_id for g in groups]
            data_config = replace(data_config, duckdb_path=backup_primary)
            expected_dataset = BACKUP_DATASET
        loader = AshareDataLoader(data_config,
            make_model_config(source_config), require_factor_cache_hit=True)
        loader.load_data()
        if loader.dataset_identity_status != "confirmed" or loader.dataset_id != expected_dataset:
            raise ValueError(f"dataset identity drift: {loader.dataset_id}")
        factors = {}
        factor_hashes = {}
        for name in MEMBERS:
            values = loader.factor_tensor[FEATURE_NAMES.index(name)].numpy()
            factor_hashes[name] = digest_bytes(values.tobytes(order="C"))
            if factor_hashes[name] != FROZEN_FACTORS[name]:
                raise ValueError(f"frozen factor bytes drift: {name}")
            factors[name] = values.copy()
        full_dates = list(loader.dates)
        due = RebalancePolicy.from_config(configs[30000]).rebalance_mask(full_dates)
        columns = np.flatnonzero(np.asarray(["20210101" <= d <= "20251231" for d in full_dates]))
        if not columns.size or np.any(np.diff(columns) != 1):
            raise ValueError("development window must be contiguous")
        sl = slice(int(columns[0]), int(columns[-1]) + 1)
        dates = full_dates[sl]
        if {d[:4] for d in dates} != {str(y) for y in range(2021, 2026)}:
            raise ValueError("five complete development years must be present")
        factors = {n: v[:, sl].copy() for n, v in factors.items()}
        mask = np.asarray(loader.universe_mask)[:, sl].copy()
        raw = {k: v.numpy()[:, sl].copy() for k, v in loader.raw_data_cache.items()}
        codes = list(loader.ts_codes)
        signal, segments = assemble_frozen_signal(factors, mask, dates, payload)
        due = due[sl]
        if not first_rebalance_mask(due).any():
            raise ValueError("no executable first rebalance")
        provenance = dict(implementation_commit=commit, account_summary_version=1,
            development_validation=True, not_independent_oos=True, dataset_id=loader.dataset_id,
            dataset_identity_status=loader.dataset_identity_status, factor_cache_outcome=loader.factor_cache_outcome,
            source_root=str(source), source_config_sha256=digest_bytes(source_config_bytes),
            snapshot_directory=args.snapshot_directory, consumed_primary_path=str(data_config.duckdb_path),
            original_wp18_dataset_id=FROZEN_DATASET, snapshot_bridge=snapshot_bridge,
            direction_file_sha256=digest_bytes(direction_bytes), factor_matrix_sha256=factor_hashes,
            consumed_factor_version=_capture_versions()["factor_compute_version"], source_wp18_factor_version=5,
            factor_version_bridge="only AMPLITUDE differs; all three full matrices byte-identical",
            versions=_capture_versions(), candidate_config_sha256=hashes,
            effective_configs={str(c): asdict(cfg) for c, cfg in configs.items()},
            loaded_window=[full_dates[0], full_dates[-1], len(full_dates)], loaded_stock_count=len(codes),
            account_signal_window=[dates[0], dates[-1], len(dates)], frozen_segments=segments,
            stitched_signal_sha256=digest_bytes(signal.tobytes()),
            rebalance_mask_sha256=digest_bytes(due.tobytes()),
            rebalance_frequency="every_10_days", prediction_horizon=10, single_initial_funding=True,
            price_degraded=loader.price_degraded,
            environment=dict(os=platform.platform(), python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__),
            account_run_budget=6, wallclock_budget_seconds=5400)
        if direction_path.read_bytes() != direction_bytes or source_config_path.read_bytes() != source_config_bytes:
            raise ValueError("source configuration/directions changed during load")
        atomic_write_json(output / "provenance.json", provenance)
        del loader, factors, payload
        gc.collect()
        ledger.complete_trial(input_trial, metrics={"dataset_id": expected_dataset})
    except Exception as exc:
        ledger.fail_trial(input_trial, error=f"{type(exc).__name__}: {exc}")
        raise
    rows = []
    for capital, config in configs.items():
        first_active = None
        for mode in ("active", "passive"):
            trial = ledger.record_trial(algorithm="continuous_account", candidate=f"{capital}/{mode}",
                payload=dict(config_sha256=hashes[str(capital)], signal_sha256=provenance["stitched_signal_sha256"]))
            try:
                if time.perf_counter() - started > 5400:
                    raise TimeoutError("preregistered 90-minute budget exhausted")
                result = AshareBacktestEngine(config).run(signal, raw, codes, dates, mask,
                    rebalance_mask=due if mode == "active" else first_rebalance_mask(due))
                row = summarize_account(result, capital, raw, codes)
                row["mode"] = mode
                first_due = np.flatnonzero(first_rebalance_mask(due))[0]
                first_entry = dates[first_due + 1]
                initial_fills = [t for t in result.trades if t["date"] == first_entry]
                if mode == "active":
                    first_active = initial_fills
                elif initial_fills != first_active or initial_fills != result.trades:
                    raise ValueError("passive initial fill parity failed")
                write_csv(output / f"trades_{capital}_{mode}.csv", result.trades)
                write_csv(output / f"daily_{capital}_{mode}.csv", [dict(mark_date=p["exit_date"],
                    equity_normalized=result.equity_curve[i+1], equity_rmb=p["equity_exit"],
                    net_return=result.daily_returns[i], plan_turnover=result.turnover[i], **p,
                    construction_diagnostics=result.construction_diagnostics[i]) for i, p in enumerate(account_periods(result))])
                ledger.complete_trial(trial, metrics=row)
                rows.append(row)
                atomic_write_json(output / "partial_results.json", rows)
                logger.info("account completed capital={} mode={} profit={} cagr={}", capital, mode,
                    row["net_profit_rmb"], row["net_cagr"])
                del result
            except Exception as exc:
                ledger.fail_trial(trial, error=f"{type(exc).__name__}: {exc}")
                raise
    for row in rows:
        passive = next(r for r in rows if r["initial_capital"] == row["initial_capital"] and r["mode"] == "passive")
        row.update(passive_net_return=passive["net_return"], passive_net_profit_rmb=passive["net_profit_rmb"],
            passive_valuation_complete=passive["valuation_complete"],
            increment_vs_passive_rmb=row["net_profit_rmb"] - passive["net_profit_rmb"],
            compound_excess_vs_passive=(1 + row["net_return"]) / (1 + passive["net_return"]) - 1,
            compound_excess_vs_theoretical=(1 + row["net_return"]) / (1 + row["theoretical_equal_weight_return"]) - 1)
    provenance["wallclock_seconds"] = time.perf_counter() - started
    atomic_write_json(output / "provenance.json", provenance)
    atomic_write_json(output / "results.json", dict(provenance=provenance,
        not_applicable=[dict(initial_capital=10000, reason="25只/4%/最低1000元不能同时成立")], accounts=rows))
    write_csv(output / "summary.csv", rows)
    (output / "RESULTS.md").write_text(render_report(rows, provenance), encoding="utf-8")
    print(f"Completed six continuous accounts: {output}")


if __name__ == "__main__":
    main()
