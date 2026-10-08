"""P61: explicit null budget, actual fills, and bounded diagnostic branches."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from ashare_data.config import make_backtest_config
from ashare_model.backtest import AshareBacktestEngine
from ashare_portfolio.constructor import PortfolioConstructor
from scripts.task61_turnover_budget_diagnostic import (
    budget_variants, sell_only_days, assess, verify_baseline, verify_input_hash, risk_checks,
)
from tests.test_p58_small_capital_execution import cfg, market


def test_explicit_null_disables_budget_without_domain_default_refill():
    protocol = dict(domain='medium_cross_section', frequency='every_10_days', horizon=10)
    omitted = make_backtest_config(dict(protocol=protocol, backtest={}))
    explicit = make_backtest_config(dict(protocol=protocol, backtest=dict(turnover_budget=None)))
    assert omitted.turnover_budget is not None
    assert explicit.turnover_budget is None
    assert explicit.rebalance_frequency == 'every_10_days' and explicit.target_horizon == 10


def test_only_budget_changes_between_normal_variants():
    base = cfg(initial_capital=100000, top_n=25, buy_rank=25, sell_rank=35,
        single_weight_cap=.04, rebalance_frequency='every_10_days', target_horizon=10)
    normal, unbudgeted = budget_variants(base)
    assert normal == base
    changed = {k for k,v in asdict(normal).items() if v != asdict(unbudgeted)[k]}
    assert changed == {'turnover_budget'}
    assert unbudgeted.turnover_budget is None
    assert unbudgeted.turnover_allocation == 'priority'


@pytest.mark.parametrize('change', [dict(initial_capital=50000), dict(turnover_budget=.1)])
def test_variants_reject_other_capital_or_budget(change):
    with pytest.raises(ValueError): budget_variants(cfg(**change))


def test_null_removes_artificial_budget_but_preserves_single_name_and_cash_cap():
    base = cfg(top_n=25, buy_rank=25, sell_rank=25, single_weight_cap=.04)
    signal = np.arange(50, dtype=float)
    prev = np.r_[np.full(25,.04), np.zeros(25)]
    outputs = [PortfolioConstructor(c).construct(signal, prev, capital=100000,
        eligible=np.ones(50,dtype=bool), buy_blocked=np.zeros(50,dtype=bool),
        sell_blocked=np.zeros(50,dtype=bool), stable_keys=np.array([str(i) for i in range(50)]))
        for c in budget_variants(base)]
    assert outputs[0].diagnostics['turnover_budget_mode'] == 'priority'
    assert outputs[0].weights.sum() == pytest.approx(.8)
    assert outputs[1].diagnostics['turnover_budget_mode'] == 'unbudgeted'
    assert outputs[1].weights.sum() == pytest.approx(1.)
    assert outputs[1].turnover == pytest.approx(2.)
    assert max(outputs[1].weights) <= .04


@pytest.mark.parametrize('constraint', ['lot_and_fees','suspension','upper_limit','lower_limit'])
def test_real_engine_keeps_constraints_when_budget_is_none(constraint):
    data = list(market(n=20, days=5))
    base = cfg(turnover_budget=None)
    due = np.array([True,False,True,False,False])
    if constraint == 'suspension': data[1]['suspended'][0,:]=True
    if constraint == 'upper_limit': data[1]['up_limit'][0,:]=10
    if constraint == 'lower_limit':
        data[0][:,2]=0; data[0][1:,2]=np.arange(19,0,-1)
        data[1]['down_limit'][0,3]=10
        base=replace(base,top_n=19,buy_rank=19,sell_rank=19,single_weight_cap=1/19)
    result=AshareBacktestEngine(base).run(*data,rebalance_mask=due)
    assert min(d['account']['cash'] for d in result.construction_diagnostics)>=0
    assert all(t['quantity'] % 100 == 0 for t in result.trades if t['side']=='buy')
    assert all(t['amount']>=1000 for t in result.trades if t['side']=='buy')
    assert sum(t['total'] for t in result.trades)>0
    if constraint in ('suspension','upper_limit'):
        assert not any(t['ts_code']==data[2][0] for t in result.trades)
    if constraint == 'lower_limit':
        assert any(t['ts_code']==data[2][0] and t['side']=='buy' for t in result.trades)
        assert not any(t['ts_code']==data[2][0] and t['side']=='sell' for t in result.trades)


def test_sell_only_days_count_actual_dates_not_legs():
    trades=[dict(date='20210104',side='sell'),dict(date='20210104',side='sell'),
        dict(date='20210105',side='sell'),dict(date='20210105',side='buy'),
        dict(date='20210106',side='buy')]
    assert sell_only_days(trades) == ['20210104']


def row(exposure,profit):
    return dict(actual_exposure_mean=exposure,net_profit_rmb=profit)


@pytest.mark.parametrize('after,expected', [
    (row(.25,50000),'position_not_restored'),
    (row(.8,20000),'restored_without_passive_edge'),
    (row(.8,30500),'restored_marginal_increment'),
    (row(.8,40000),'inspect_risk_and_costs'),
])
def test_single_variable_branch_is_fixed_before_results(after,expected):
    decision=assess(row(.16,10000),after,row(.9,30000))
    assert decision['outcome']==expected
    assert decision['run_cost_stress'] == (expected=='inspect_risk_and_costs')


def test_hash_drift_is_rejected(tmp_path):
    path=tmp_path/'input.npz'; path.write_bytes(b'tampered')
    with pytest.raises(ValueError): verify_input_hash(path,'0'*64)


def test_baseline_mismatch_is_rejected():
    expected=dict(terminal_net_assets=110000.,fees={'total':100.})
    with pytest.raises(ValueError):
        verify_baseline(dict(terminal_net_assets=110001.,fees={'total':100.}),expected)


def test_risk_checks_do_not_treat_one_year_winner_as_robust():
    passive=dict(annual_returns={str(y):0. for y in range(2021,2026)},max_drawdown=.25,net_profit_rmb=100.)
    normal=dict(annual_returns=dict(zip(passive['annual_returns'],[.01,.01,.08,-.01,0.])),
        max_drawdown=.30,first_funded_holdings=25,fraction_funded_days_below_20=0.)
    risk=risk_checks(normal,dict(net_profit_rmb=200.),passive)
    assert risk['largest_positive_year_share']==pytest.approx(.8)
    assert risk['checks']['at_least_three_positive_increment_years']
    assert not risk['checks']['largest_positive_year_share_at_most_70_percent']
    assert not risk['checks']['drawdown_not_above_passive']
    assert not risk['worth_separate_rule_evaluation']
