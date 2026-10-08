"""P60 independent cash fixture and fail-closed conditional experiment."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from ashare_data.schemas import BacktestResult
from ashare_model.execution_experiment import observe_legacy, frequency_admission
from scripts.task2_continuous_account import export_replay_input
from tests.test_p58_small_capital_execution import cfg, market


def legacy_fixture():
    # Nineteen buys of 500 shares, one buy of 400, all at 10 RMB.
    # Fees: 19*7.55 + 7.04 = 150.49; cash = 100000-99000-150.49.
    weights = np.array([.05] * 19 + [.04])
    result = BacktestResult(equity_curve=[1, .9984951], dates=['20210105', '20210106'],
        daily_returns=[-.0015049], turnover=[1.0], benchmark_equity=[1, 1],
        metrics=dict(total_return=-.0015049, annual_return=.9984951**252-1,
            max_drawdown=.0015049, order_count=20),
        positions=[dict(signal_date='20210104', entry_date='20210105', exit_date='20210106',
            ts_codes=[str(i) for i in range(20)], weights=(weights/.9984951).round(6).tolist(),
            shares=[500]*19+[400])], buy_weights=[weights], sell_weights=[np.zeros(20)],
        cost_fractions=[.0015049], construction_diagnostics=[{}], final_cash=849.51)
    raw = dict(open=np.full((20, 3), 10.0))
    return result, raw, [str(i) for i in range(20)]


def test_legacy_observations_reconcile_independent_cash_and_fees():
    result, raw, codes = legacy_fixture()
    original = deepcopy(result)
    observed = observe_legacy(result, cfg(), raw, codes)
    assert observed.trades[-1]['quantity'] == pytest.approx(400)
    assert sum(t['total'] for t in observed.trades) == pytest.approx(150.49)
    p = observed.construction_diagnostics[0]['account']
    assert p['cash'] == pytest.approx(849.51)
    assert p['actual_exposure'] == pytest.approx(99000/99849.51)
    assert p['actual_turnover'] == pytest.approx(.99)
    assert observed.equity_curve == original.equity_curve
    assert observed.positions == original.positions
    np.testing.assert_array_equal(observed.buy_weights, original.buy_weights)
    assert result.trades == [] and result.construction_diagnostics == [{}]


@pytest.mark.parametrize('corruption', ['cash', 'fee', 'missing_trail', 'negative_fill'])
def test_legacy_reporting_rejects_corrupted_facts(corruption):
    result, raw, codes = legacy_fixture()
    if corruption == 'cash': result.final_cash += 1
    if corruption == 'fee': result.cost_fractions[0] += .00001
    if corruption == 'missing_trail': result.buy_weights = None
    if corruption == 'negative_fill': result.buy_weights[0][0] = -.05
    with pytest.raises(ValueError):
        observe_legacy(result, cfg(), raw, codes)


def qualified():
    return dict(valuation_complete=True, cash_min=10., net_profit_rmb=100.,
        first_funded_holdings=25, fraction_funded_days_below_20=0.)


@pytest.mark.parametrize('field,value,reason', [
    ('first_funded_holdings', 6, 'initial_diversification'),
    ('fraction_funded_days_below_20', .06, 'persistent_diversification'),
    ('net_profit_rmb', 0, 'no_positive_net_profit'),
    ('cash_min', -.01, 'negative_cash'),
    ('valuation_complete', False, 'incomplete_valuation'),
])
def test_positive_return_does_not_override_gate(field, value, reason):
    row = qualified(); row[field] = value
    decision = frequency_admission(row, dict(valuation_complete=True, net_profit_rmb=0.))
    assert decision.admitted is False
    assert reason in decision.reasons


def test_positive_absolute_profit_but_passive_loss_stops():
    decision = frequency_admission(qualified(), dict(valuation_complete=True, net_profit_rmb=200.))
    assert decision.admitted is False
    assert decision.reasons == ('no_positive_passive_increment',)


def test_missing_gate_fact_is_rejected_instead_of_default_pass():
    with pytest.raises(KeyError):
        frequency_admission({}, {})


def test_gate_pass_is_only_permission_for_next_stage():
    decision = frequency_admission(qualified(), dict(valuation_complete=True, net_profit_rmb=10.))
    assert decision.admitted and decision.reasons == ()


def test_export_replay_preserves_bytes_and_refuses_overwrite(tmp_path):
    signal, raw, codes, dates, mask = market()
    due = np.array([True, False, False])
    path = tmp_path/'input.npz'
    export_replay_input(path, signal, raw, codes, dates, mask, due)
    with np.load(path, allow_pickle=False) as bundle:
        for key, values in dict(signal=signal, mask=mask, due=due, raw_open=raw['open']).items():
            assert bundle[key].tobytes() == values.tobytes()
        assert bundle['codes'].tolist() == codes and bundle['dates'].tolist() == dates
        assert not any(bundle[k].dtype.hasobject for k in bundle.files)
    with pytest.raises(FileExistsError):
        export_replay_input(path, signal, raw, codes, dates, mask, due)


def test_observation_adapter_is_parity_for_current_engine(tmp_path):
    from ashare_model.backtest import AshareBacktestEngine
    data = market(days=6)
    result = AshareBacktestEngine(cfg()).run(*data)
    legacy = replace(result, trades=[], construction_diagnostics=[
        {k:v for k,v in d.items() if k not in ('account','execution_rejections','market_block_counts')}
        for d in result.construction_diagnostics])
    observed = observe_legacy(legacy, cfg(), data[1], data[2])
    for a, b in zip(result.construction_diagnostics, observed.construction_diagnostics):
        for key in ('cash','fees','equity_before','equity_after_execution','equity_exit','actual_exposure','actual_turnover'):
            assert b['account'][key] == pytest.approx(a['account'][key], abs=1e-7)
    assert [t['quantity'] for t in observed.trades] == pytest.approx([t['quantity'] for t in result.trades])
