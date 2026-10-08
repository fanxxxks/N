"""P61: one 100k account, existing budget 0.20 versus explicitly disabled."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from ashare_data.config import BacktestConfig
from ashare_data.io_utils import atomic_write_json
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.continuous_account import account_periods, create_output_directory, summarize_account
from ashare_model.ledger import ExperimentLedger
from ashare_model.runspec import _capture_versions
from scripts.task2_continuous_account import FROZEN_DATASET, write_csv
from scripts.task3_execution_experiment import funded_counts


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_input_hash(path, expected):
    if file_hash(path) != expected:
        raise ValueError('frozen replay input hash mismatch')


def budget_variants(base):
    if base.initial_capital != 100000 or base.turnover_budget != .20:
        raise ValueError('diagnostic requires the original 100k / 0.20 account')
    return base, replace(base, turnover_budget=None)


def verify_baseline(actual, expected):
    for key in ('terminal_net_assets', 'fees'):
        if actual[key] != expected[key]:
            raise ValueError(f'P60 baseline replay differs: {key}')


def sell_only_days(trades):
    by_date = {}
    for trade in trades:
        by_date.setdefault(trade['date'], set()).add(trade['side'])
    return sorted(d for d, sides in by_date.items() if sides == {'sell'})


def assess(before, after, passive):
    restored = (after['actual_exposure_mean'] >= .60
        and after['actual_exposure_mean']-before['actual_exposure_mean'] >= .20)
    increment = after['net_profit_rmb']-before['net_profit_rmb']
    edge = after['net_profit_rmb']-passive['net_profit_rmb']
    if not restored:
        outcome = 'position_not_restored'
    elif edge <= 0:
        outcome = 'restored_without_passive_edge'
    elif increment < 1000 or edge < 1000:
        outcome = 'restored_marginal_increment'
    else:
        outcome = 'inspect_risk_and_costs'
    return dict(outcome=outcome, position_restored=restored,
        increment_vs_budget_rmb=increment, increment_vs_passive_rmb=edge,
        run_cost_stress=outcome == 'inspect_risk_and_costs')


def risk_checks(normal, stressed, passive):
    annual = {y:normal['annual_returns'][y]-passive['annual_returns'][y]
        for y in normal['annual_returns']}
    positives = [v for v in annual.values() if v > 0]
    concentration = max(positives)/sum(positives) if positives else 1.
    checks = dict(at_least_three_positive_increment_years=len(positives)>=3,
        largest_positive_year_share_at_most_70_percent=concentration<=.70,
        drawdown_not_above_passive=normal['max_drawdown']<=passive['max_drawdown'],
        stress_profit_still_above_normal_passive=stressed['net_profit_rmb']>passive['net_profit_rmb'],
        actual_diversification=normal['first_funded_holdings']>=20
            and normal['fraction_funded_days_below_20']<=.05)
    return dict(annual_return_increment_pp=annual, largest_positive_year_share=concentration,
        checks=checks, worth_separate_rule_evaluation=all(checks.values()))


def render(rows, passive, decision, risk):
    a,b = rows[:2]
    lines = ['# 10万元换手预算有限诊断（开发验证）', '',
        '只改变turnover_budget：0.20 vs 显式null。medium B/25只/4%/最低1000元/10日调仓与预测/正常费用保持一致。',
        '连续账户仅一次投入10万元；2021—2025是开发验证。任务三的原失败结论保留。', '',
        '|指标|priority预算0.20|关闭人为预算|', '|---|---:|---:|']
    specs = [('期末净资产（元）','terminal_net_assets',',.2f'),('累计净赚（元）','net_profit_rmb',',.2f'),
        ('扣费年化','net_cagr','.2%'),('最大回撤','max_drawdown','.2%'),
        ('平均实际仓位','actual_exposure_mean','.2%'),('平均实际持股','actual_holdings_mean','.2f'),
        ('最少实际持股（含建仓前）','actual_holdings_min','d'),('末期实际持股','actual_holdings_final','d'),
        ('首期实际持股','first_funded_holdings','d'),('建仓后不足20只日期占比','fraction_funded_days_below_20','.2%'),
        ('总费用（元）','fee_total',',.2f'),('实际双边累计换手（倍）','actual_turnover_total','.2f'),
        ('实际成交笔数','actual_order_count','d'),('卖出而无买入交易日数','sell_only_day_count','d'),
        ('相对原被动净赚差额（元）','increment_vs_passive_rmb','+,.2f'),
        ('相对原被动复利超额','compound_excess_vs_passive','+.2%')]
    for title,key,fmt in specs:
        lines.append(f'|{title}|{a[key]:{fmt}}|{b[key]:{fmt}}|')
    lines += ['', '|年度扣费收益|预算0.20|关闭人为预算|原被动持有|', '|---|---:|---:|---:|']
    for y in a['annual_returns']:
        lines.append(f"|{y}|{a['annual_returns'][y]:.2%}|{b['annual_returns'][y]:.2%}|{passive['annual_returns'][y]:.2%}|")
    lines += ['', '|费用分项（元）|预算0.20|关闭人为预算|', '|---|---:|---:|']
    for name,label in [('commission','佣金'),('stamp_tax','印花税'),('transfer_fee','过户费'),('slippage','滑点')]:
        lines.append(f"|{label}|{a['fees'][name]:,.2f}|{b['fees'][name]:,.2f}|")
    lines += ['', '## 裁决', '',
        f"平均实际仓位变化{b['actual_exposure_mean']-a['actual_exposure_mean']:+.2%}，净赚变化{decision['increment_vs_budget_rmb']:+,.2f}元。"]
    messages = dict(position_not_restored='仓位未明显恢复；结束本次诊断，仅报告现有记录的主要阻断，不自动修复。',
        restored_without_passive_edge='仓位明显恢复，但扣费后仍不如原被动持有，未出现主动管理净增量；停止当前信号的执行修复路线。',
        restored_marginal_increment='仓位恢复，但增量未达到本轮经济价值检查的最低必要门槛；停止，不继续搜索。',
        inspect_risk_and_costs='仓位与净收益出现有意义改善，已按预注册检查年度贡献、回撤及一次固定成本压力。')
    lines += [messages[decision['outcome']], '']
    if risk:
        lines += [f"成本/风险检查：{json.dumps(risk,ensure_ascii=False)}", '',
            f"压力账户净赚{rows[2]['net_profit_rmb']:,.2f}元，年化{rows[2]['net_cagr']:.2%}，回撤{rows[2]['max_drawdown']:.2%}。", '',
            '仅可单独评估简单换仓预算规则，不能直接部署。' if risk['worth_separate_rule_evaluation']
                else '风险/成本或分散条件未通过，停止当前信号的执行修复路线，不实现预算规则。', '']
    else:
        lines += ['未进入条件性成本压力测试；未运行其它预算、频率、因子或规则实验。', '']
    for row in rows[:2]:
        lines += [f"### {row['arm']}的现有阻断记录", '',
            f"构造阶段抑制：{json.dumps(row['constructor_suppression'],ensure_ascii=False)}", '',
            f"成交后拒绝：{json.dumps(row['execution_rejections'],ensure_ascii=False)}", '',
            f"市场约束观察：{json.dumps(row['market_block_counts'],ensure_ascii=False)}（不是已提交订单失败数）。", '']
    lines += ['## 口径', '',
        f"原被动持有净赚{passive['net_profit_rmb']:,.2f}元，年化{passive['net_cagr']:.2%}，回撤{passive['max_drawdown']:.2%}，平均仓位{passive['actual_exposure_mean']:.2%}；为任务三首期实际可买篮子随后持有，不是宽基指数或推荐。",
        'None是已有配置支持的关闭方式；保留现金、费用、整手、最低额、单股上限、排名/阈值及涨跌停停牌约束。没有强制满仓。',
        '比较识别当前人为预算机制的总影响，不能单独证明priority排序相对其它分配规则的优劣。',
        '计划指标未用于选型。抑制与市场观察属于不同阶段，不能相加当成互斥失败订单。',
        '使用任务三已核验快照的完全相同物化输入；不读取实时数据库或新样本。',
        '现有引擎为复权总回报财富近似，红利未逐笔现金入账、执行时按财富重推股数；非券商精确现金账。期末未假想清仓。',
        '预算关闭是机制价值诊断，不能直接作实盘方案；未开展任务四或实现trade_groups/预算规则。',
        '全部账户逐日/成交/配置/费用/年度/原因见results.json、daily_*.csv、trades_*.csv；失败及成功trial见ledger.jsonl。', '']
    return '\n'.join(lines)


def run(output,ledger):
    started = time.perf_counter()
    reference_path = ROOT/'reports/task3_20261007_exact/results.json'
    reference_dir = reference_path.parent
    historical_hashes = {str(p.relative_to(reference_dir)):file_hash(p)
        for p in reference_dir.rglob('*') if p.is_file()}
    input_trial = ledger.record_trial(algorithm='input_provenance',candidate='P60_exact_materialized_input')
    try:
        reference = json.loads(reference_path.read_bytes())
        provenance = reference['paired_provenance']
        if (provenance['dataset_id'] != FROZEN_DATASET or not provenance['dataset_identity_status']=='confirmed'
            or provenance['prediction_horizon']!=10 or provenance['rebalance_frequency']!='every_10_days'
            or provenance['price_degraded'] or _capture_versions()!=provenance['versions']):
            raise ValueError('P60 identity/version/semantics mismatch')
        bundle_path = Path(reference['replay_input'])
        verify_input_hash(bundle_path,reference['replay_input_sha256'])
        with np.load(bundle_path,allow_pickle=False) as bundle:
            signal,mask,due = (bundle[k] for k in ('signal','mask','due'))
            raw = {k[4:]:bundle[k] for k in bundle.files if k.startswith('raw_')}
            codes,dates = bundle['codes'].tolist(),bundle['dates'].tolist()
        for values,key in ((signal,'stitched_signal_sha256'),(due,'rebalance_mask_sha256')):
            if hashlib.sha256(values.tobytes()).hexdigest()!=provenance[key]:
                raise ValueError(f'P60 input mismatch: {key}')
        base = BacktestConfig(**provenance['effective_configs']['100000'])
        configs = budget_variants(base)
        original = next(p for p in reference['pairs'] if p['capital']==100000)
        passive = original['passive']
        if not passive['valuation_complete']:
            raise ValueError('passive reference valuation incomplete')
        run_provenance = dict(run_kind='research_development_diagnostic',
            implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            development_validation=True,not_independent_oos=True,dataset_id=FROZEN_DATASET,
            source_reference=str(reference_path),source_reference_sha256=file_hash(reference_path),
            input_bundle=str(bundle_path),input_bundle_sha256=reference['replay_input_sha256'],
            signal_sha256=provenance['stitched_signal_sha256'],due_sha256=provenance['rebalance_mask_sha256'],
            frozen_segments=provenance['frozen_segments'],factor_sha256=provenance['factor_matrix_sha256'],
            effective_configs={name:asdict(c) for name,c in zip(('budget_020','budget_disabled'),configs)},
            versions=_capture_versions(),prediction_horizon=10,rebalance_frequency='every_10_days',
            account_window=provenance['account_signal_window'],single_initial_funding=True,
            normal_account_budget=2,conditional_pressure_account_budget=1,wallclock_budget_seconds=900,
            source_task3_file_hashes=historical_hashes,
            environment=dict(python=platform.python_version(),os=platform.platform(),numpy=np.__version__,torch=torch.__version__))
        atomic_write_json(output/'provenance.json',run_provenance)
        ledger.complete_trial(input_trial,metrics=dict(dataset_id=FROZEN_DATASET))
    except Exception as exc:
        ledger.fail_trial(input_trial,error=f'{type(exc).__name__}: {exc}')
        raise

    def account(name,config):
        trial = ledger.record_trial(algorithm='continuous_account',candidate=name,payload=asdict(config))
        try:
            if time.perf_counter()-started>900:
                raise TimeoutError('15-minute budget exhausted')
            result = AshareBacktestEngine(config).run(signal,raw,codes,dates,mask,rebalance_mask=due)
            row = summarize_account(result,config.initial_capital,raw,codes)
            row.update(funded_counts(result.positions,result.trades),arm=name,
                turnover_budget=config.turnover_budget,sell_only_day_count=len(sell_only_days(result.trades)),
                sell_only_trade_dates=sell_only_days(result.trades),fee_total=row['fees']['total'],
                increment_vs_passive_rmb=row['net_profit_rmb']-passive['net_profit_rmb'],
                compound_excess_vs_passive=(1+row['net_return'])/(1+passive['net_return'])-1)
            if not row['valuation_complete']:
                raise ValueError('account terminal valuation incomplete')
            if name=='budget_020':
                verify_baseline(row,original['after'])
                import csv
                with (reference_dir/'after/daily_100000_active.csv').open(encoding='utf-8-sig',newline='') as stream:
                    expected=list(csv.DictReader(stream))
                if (len(expected)!=len(result.daily_returns) or any(
                    float(p['equity_normalized'])!=result.equity_curve[i+1]
                    or float(p['net_return'])!=result.daily_returns[i] for i,p in enumerate(expected))):
                    raise ValueError('P60 daily baseline replay differs')
                with (reference_dir/'after/trades_100000_active.csv').open(encoding='utf-8-sig',newline='') as stream:
                    expected=list(csv.DictReader(stream))
                if len(expected)!=len(result.trades) or any(any(
                    (float(t[k]) if isinstance(v,(float,int)) else t[k])!=v for k,v in actual.items())
                    for t,actual in zip(expected,result.trades)):
                    raise ValueError('P60 actual fill baseline replay differs')
            write_csv(output/f'trades_{name}.csv',result.trades)
            write_csv(output/f'daily_{name}.csv',[dict(mark_date=p['exit_date'],
                equity_normalized=result.equity_curve[i+1],net_return=result.daily_returns[i],
                plan_turnover=result.turnover[i],**p,construction_diagnostics=result.construction_diagnostics[i])
                for i,p in enumerate(account_periods(result))])
            ledger.complete_trial(trial,metrics=row)
            return row
        except Exception as exc:
            ledger.fail_trial(trial,error=f'{type(exc).__name__}: {exc}')
            raise

    rows=[account(name,c) for name,c in zip(('budget_020','budget_disabled'),configs)]
    decision=assess(rows[0],rows[1],passive)
    risk=None
    if decision['run_cost_stress']:
        stress=replace(configs[1],commission_rate=configs[1].commission_rate*1.5,
            min_commission=configs[1].min_commission*1.5,slippage_rate=configs[1].slippage_rate*2)
        run_provenance['conditional_stress_config']=asdict(stress)
        rows.append(account('budget_disabled_cost_stress',stress))
        risk=risk_checks(rows[1],rows[2],passive)
    if any(file_hash(reference_dir/p)!=digest for p,digest in historical_hashes.items()):
        raise ValueError('historical task3 evidence changed')
    run_provenance['consumed_account_runs']=len(rows)
    atomic_write_json(output/'provenance.json',run_provenance)
    atomic_write_json(output/'results.json',dict(provenance=run_provenance,accounts=rows,
        passive_reference=passive,decision=decision,risk_checks=risk,account_runs=len(rows),
        source_task3_unchanged=True,wallclock_seconds=time.perf_counter()-started))
    write_csv(output/'summary.csv',rows)
    (output/'RESULTS.md').write_text(render(rows,passive,decision,risk),encoding='utf-8')
    print(json.dumps(dict(outcome=decision,account_runs=len(rows),output=str(output)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    create_output_directory(output)
    ledger=ExperimentLedger(output/'ledger.jsonl',run_id=output.name)
    with ledger.trial(algorithm='bounded_diagnostic',candidate='one_budget_change'):
        run(output,ledger)
    if ledger.finalize():
        raise ValueError('diagnostic ledger contains open trials')
