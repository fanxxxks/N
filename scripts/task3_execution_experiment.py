"""P60: fixed-cadence paired replay, followed by the preregistered stop gate."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BEFORE = 'dbaf7296a41ce9813ae736c0e4879d54d9488b0f'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def historical_worker(args):
    """Import the entire committed historical engine, without copied algorithms."""
    historic = args.historical_root.resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=historic, text=True).strip() != BEFORE:
        raise ValueError('historical checkout must be exactly the preregistered commit')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=historic, text=True).strip():
        raise ValueError('historical checkout must be clean')
    sys.path.insert(0, str(historic))
    from ashare_data.config import BacktestConfig
    from ashare_data.io_utils import atomic_write_json
    from ashare_model.backtest import AshareBacktestEngine
    from ashare_model.ledger import ExperimentLedger
    from ashare_model.runspec import _capture_versions
    # Eager imports above bind execution dependencies to the historical tree.
    dependencies = {name: str(Path(sys.modules[name].__file__).resolve()) for name in
        ('ashare_model.backtest', 'ashare_portfolio.constructor', 'ashare_portfolio.rebalance',
         'ashare_execution', 'ashare_data.processor', 'ashare_data.config')}
    if any(not Path(p).is_relative_to(historic) for p in dependencies.values()):
        raise ValueError('historical execution imported a current dependency')
    args.output.mkdir(parents=True, exist_ok=False)
    ledger = ExperimentLedger(args.output/'ledger.jsonl', run_id=args.output.name)
    provenance = json.loads(args.after_provenance.read_bytes())
    with np.load(args.bundle, allow_pickle=False) as bundle:
        signal, mask, due = (bundle[n] for n in ('signal', 'mask', 'due'))
        raw = {n[4:]: bundle[n] for n in bundle.files if n.startswith('raw_')}
        codes, dates = bundle['codes'].tolist(), bundle['dates'].tolist()
    if hashlib.sha256(signal.tobytes()).hexdigest() != provenance['stitched_signal_sha256']:
        raise ValueError('replay signal drift')
    if hashlib.sha256(due.tobytes()).hexdigest() != provenance['rebalance_mask_sha256']:
        raise ValueError('replay cadence drift')
    atomic_write_json(args.output/'provenance.json', dict(commit=BEFORE, versions=_capture_versions(),
        execution_dependencies=dependencies, dependency_sha256={n:sha(Path(p)) for n,p in dependencies.items()},
        replay_input_sha256=sha(args.bundle), development_validation=True,
        rebalance_frequency='every_10_days', prediction_horizon=10, account_run_budget=3))
    for capital in (30000, 50000, 100000):
        config = BacktestConfig(**provenance['effective_configs'][str(capital)])
        with ledger.trial(algorithm='historical_continuous_account', candidate=str(capital),
                payload=dict(config=asdict(config), dataset_id=provenance['dataset_id'])):
            result = AshareBacktestEngine(config).run(signal, raw, codes, dates, mask, rebalance_mask=due)
            meta = {n:getattr(result,n) for n in ('dates','metrics','positions','construction_diagnostics','final_cash')}
            arrays = {n:np.asarray(getattr(result,n)) for n in
                ('equity_curve','daily_returns','turnover','benchmark_equity','buy_weights','sell_weights',
                 'cost_fractions','final_shares')}
            with (args.output/f'native_{capital}.npz').open('xb') as stream:
                np.savez_compressed(stream, metadata=np.asarray(json.dumps(meta)), **arrays)
    if ledger.finalize():
        raise ValueError('historical ledger has open trials')


def read_native(path):
    from ashare_data.schemas import BacktestResult
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(bundle['metadata'].item())
        for name in bundle.files:
            if name != 'metadata':
                meta[name] = bundle[name].tolist()
    return BacktestResult(**meta)


def funded_counts(positions, trades):
    first = next(t['date'] for t in trades if t['side'] == 'buy')
    start = next(i for i,p in enumerate(positions) if p['entry_date'] == first)
    counts = [len(p['ts_codes']) for p in positions[start:]]
    return dict(first_funded_holdings=counts[0],
        fraction_funded_days_below_20=float(np.mean(np.asarray(counts)<20)))


def render(pairs):
    lines = ['# 任务三：固定10日调仓的执行修复对照', '',
        '2021—2025开发验证；medium B三因子等权，10日预测，连续账户一次本金。',
        '旧/新使用同一信号、费用和25只/4%/最低1000元/priority配置；1万不适用。', '',
        '|本金|修正前净赚|修正后净赚|修复增量人民币|旧/新扣费年化|旧/新最大回撤|被动净赚|新主动比被动多赚|',
        '|---:|---:|---:|---:|---|---|---:|---:|']
    for p in pairs:
        a,b,h = (p[n] for n in ('before','after','passive'))
        lines.append(f"|{a['initial_capital']:,.0f}|{a['net_profit_rmb']:,.2f}|{b['net_profit_rmb']:,.2f}|"
            f"{b['net_profit_rmb']-a['net_profit_rmb']:+,.2f}|{a['net_cagr']:.2%}/{b['net_cagr']:.2%}|"
            f"{a['max_drawdown']:.2%}/{b['max_drawdown']:.2%}|{h['net_profit_rmb']:,.2f}|"
            f"{b['net_profit_rmb']-h['net_profit_rmb']:+,.2f}|")
    lines += ['', '|本金|旧/新平均实际仓位|旧/新平均持股|旧/新总费用|新首期持股|新建仓后不足20只比例|频率准入|',
        '|---:|---|---|---|---:|---:|---|']
    for p in pairs:
        a,b = p['before'],p['after']
        lines.append(f"|{a['initial_capital']:,.0f}|{a['actual_exposure_mean']:.2%}/{b['actual_exposure_mean']:.2%}|"
            f"{a['actual_holdings_mean']:.2f}/{b['actual_holdings_mean']:.2f}|"
            f"{a['fees']['total']:,.2f}/{b['fees']['total']:,.2f}|{b['first_funded_holdings']}|"
            f"{b['fraction_funded_days_below_20']:.2%}|{','.join(p['decision']['reasons']) or '仅允许下一阶段'}|")
    lines += ['', '## 阶段裁决', '']
    if not any(p['decision']['admitted'] for p in pairs):
        lines += ['无配置通过进入条件，停止本轮修复路线。未运行1/5/20日网格及费用压力；',
            '不冻结策略，不启动任务四或未来纸面，不自动转入新因子/ETF。']
    else:
        lines += ['有配置满足下一阶段必要条件，须先完成前瞻性频率契约后再运行一次粗网格；尚无策略晋级。']
    lines += ['', '|交易日调仓|预测周期|状态|', '|---:|---:|---|',
        '|1|10|未运行：阶段进入条件未全部通过，不更改标签绕过检查|',
        '|5|10|未运行：阶段进入条件未全部通过，不更改标签绕过检查|',
        '|10|10|已完成同配置旧/新连续账户对照|',
        '|20|10|未运行：阶段进入条件未全部通过|', '',
        '旧实现成交后拒绝/市场约束观察日志不可用；构造阶段抑制与实际成交仍保留。',
        '旧实际仓位从实际现金及费后财富核对，未使用目标权重；旧最大单股权重记录精度6位（误差≤5e-7）。',
        '共同被动基准使用新实现首期实买股数持有，首期费用也已扣除；不是宽基指数，也不声称与旧首期成交相同。',
        '被动篮子承担不同仓位和回撤，不能将全部差额解释成选股贡献，也不是推荐买入它。',
        '复权总回报近似估值，红利未逐笔现金入账、财富重推股数；人民币不能等同券商精确结算。',
        '期末未假想清仓；各年连续计价，无重复注资。2021—2025非独立样本外。',
        '本轮比较任务一整个执行修复，不声称隔离每个修复点，也不与不同配置WP18直接作归因。',
        'results.json保留所有年度、现金、费用细分、实际/计划换手、门禁原因；before/after保留原生轨迹及逐日/成交明细。', '']
    return '\n'.join(lines)


def run_experiment(args, output, ledger):
    from ashare_data.config import BacktestConfig
    from ashare_data.io_utils import atomic_write_json
    from ashare_model.continuous_account import summarize_account, account_periods
    from ashare_model.execution_experiment import observe_legacy, frequency_admission
    from task2_continuous_account import write_csv
    args.replay_directory.mkdir(parents=True, exist_ok=False)
    bundle_path = args.replay_directory.resolve()/'input.npz'
    started = time.perf_counter()
    with ledger.trial(algorithm='after_stage', candidate='three_active_three_passive', payload=dict(account_runs=6)):
        subprocess.run([sys.executable, str(ROOT/'scripts/task2_continuous_account.py'),
            '--source-root', str(args.source_root.resolve()), '--output', str(output/'after'),
            '--primary-snapshot', str(args.primary_snapshot.resolve()),
            '--export-replay-input', str(bundle_path)], cwd=ROOT, check=True, timeout=5400)
    after_doc = json.loads((output/'after/results.json').read_bytes())
    reference_path = ROOT/'reports/task2_20261007_exact_reference/results_corrected.json'
    reference = json.loads(reference_path.read_bytes())
    for key in ('dataset_id', 'source_config_sha256', 'direction_file_sha256',
                'factor_matrix_sha256', 'candidate_config_sha256', 'effective_configs',
                'stitched_signal_sha256', 'rebalance_mask_sha256', 'prediction_horizon', 'account_signal_window'):
        if after_doc['provenance'][key] != reference['provenance'][key]:
            raise ValueError(f'frozen paired input mismatch: {key}')
    if after_doc['provenance']['price_degraded']:
        raise ValueError('degraded input cannot support this comparison')
    # Paired-current rerun must reproduce every previous terminal result.
    for row, old in zip(after_doc['accounts'], reference['accounts']):
        if row['initial_capital'] != old['initial_capital'] or row['mode'] != old['mode']:
            raise ValueError('reference account order mismatch')
        if row['terminal_net_assets'] != old['terminal_net_assets'] or row['fees'] != old['fees']:
            raise ValueError('current engine replay changed wealth/costs')
    with ledger.trial(algorithm='before_stage', candidate='three_historical_active', payload=dict(account_runs=3)):
        remaining = 5400-(time.perf_counter()-started)
        if remaining <= 0:
            raise TimeoutError('P60 wallclock budget exhausted')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--historical-worker',
            '--historical-root', str(args.historical_root.resolve()), '--bundle', str(bundle_path),
            '--after-provenance', str(output/'after/provenance.json'), '--output', str(output/'before')],
            cwd=args.historical_root, check=True, timeout=remaining)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        raw = {n[4:]:bundle[n] for n in bundle.files if n.startswith('raw_')}
        codes, dates = bundle['codes'].tolist(), bundle['dates'].tolist()
    pairs = []
    for capital in (30000, 50000, 100000):
        config = BacktestConfig(**after_doc['provenance']['effective_configs'][str(capital)])
        native = read_native(output/f'before/native_{capital}.npz')
        if any(p['entry_date'] != dates[i+1] for i,p in enumerate(native.positions)):
            raise ValueError('historical trail is not continuous delay-one daily marks')
        observed = observe_legacy(native, config, raw, codes)
        before = summarize_account(observed, capital, raw, codes)
        before.update(funded_counts(observed.positions, observed.trades), execution_rejections=None,
            market_block_counts=None, observation_source='native actual fills; reconciled reconstruction',
            maximum_single_weight_precision=1e-6)
        write_csv(output/f'before/trades_{capital}.csv', observed.trades)
        write_csv(output/f'before/daily_{capital}.csv', [dict(mark_date=p['exit_date'],
            net_return=observed.daily_returns[i], plan_turnover=observed.turnover[i], **p,
            construction_diagnostics=observed.construction_diagnostics[i])
            for i,p in enumerate(account_periods(observed))])
        after = next(r for r in after_doc['accounts'] if r['initial_capital']==capital and r['mode']=='active')
        passive = next(r for r in after_doc['accounts'] if r['initial_capital']==capital and r['mode']=='passive')
        with (output/f'after/daily_{capital}_active.csv').open(encoding='utf-8-sig', newline='') as stream:
            positions = [dict(entry_date=p['entry_date'], ts_codes=json.loads(p['ts_codes'])) for p in csv.DictReader(stream)]
        with (output/f'after/trades_{capital}_active.csv').open(encoding='utf-8-sig', newline='') as stream:
            trades = list(csv.DictReader(stream))
        after.update(funded_counts(positions, trades))
        pairs.append(dict(capital=capital, before=before, after=after, passive=passive,
            repair_increment_rmb=after['net_profit_rmb']-before['net_profit_rmb'],
            decision=asdict(frequency_admission(after, passive))))
    report = dict(development_validation=True, not_independent_oos=True,
        measured_commit=after_doc['provenance']['implementation_commit'], historical_commit=BEFORE,
        paired_provenance=after_doc['provenance'], historical_provenance=json.loads((output/'before/provenance.json').read_bytes()),
        task2_reference_sha256=sha(reference_path), replay_input=str(bundle_path), replay_input_sha256=sha(bundle_path),
        account_runs=9, execution_implementations=2, frequency_grid_runs=0, cost_stress_runs=0,
        rebalance_days=10, prediction_horizon=10, wallclock_seconds=time.perf_counter()-started,
        not_applicable=[dict(capital=10000, reason='25只/4%/最低1000元不兼容')], pairs=pairs)
    atomic_write_json(output/'results.json', report)
    (output/'RESULTS.md').write_text(render(pairs), encoding='utf-8')
    print(f'P60 completed; admitted={sum(p["decision"]["admitted"] for p in pairs)}; output={output}')


def main(args):
    sys.path.insert(0, str(ROOT))
    from ashare_model.continuous_account import create_output_directory
    from ashare_model.ledger import ExperimentLedger
    output = args.output.resolve()
    create_output_directory(output)
    ledger = ExperimentLedger(output/'ledger.jsonl', run_id=output.name)
    with ledger.trial(algorithm='bounded_experiment', candidate='frozen_medium_B'):
        run_experiment(args, output, ledger)
    if ledger.finalize():
        raise ValueError('stage ledger has open trials')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-worker', action='store_true')
    for name in ('source-root','primary-snapshot','historical-root','replay-directory','bundle','after-provenance'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    required = ('historical_root','bundle','after_provenance') if args.historical_worker else (
        'source_root','primary_snapshot','historical_root','replay_directory')
    if any(getattr(args,n) is None for n in required):
        parser.error(f'required for this mode: {required}')
    historical_worker(args) if args.historical_worker else main(args)
