from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, SimConfig
from ashare_execution import (
    ExecutionCostModel,
    execution_config_mismatches,
    validate_execution_config,
)
from ashare_model.reward import simulate_basket_daily_returns


CFG = BacktestConfig(
    initial_capital=100000.0,
    top_n=30,
    single_weight_cap=0.05,
    commission_rate=0.00025,
    min_commission=5.0,
    stamp_tax_rate=0.0005,
    transfer_fee_rate=0.00001,
    slippage_rate=0.0005,
)


def test_commission_floor_below_at_and_above_threshold():
    model = ExecutionCostModel.from_config(CFG)
    threshold = CFG.min_commission / CFG.commission_rate
    assert model.buy_cost(threshold - 1).commission == CFG.min_commission
    assert model.buy_cost(threshold).commission == pytest.approx(CFG.min_commission)
    assert model.buy_cost(threshold + 1).commission == pytest.approx(
        (threshold + 1) * CFG.commission_rate
    )


def test_zero_buy_sell_and_fee_components():
    model = ExecutionCostModel.from_config(CFG)
    assert model.buy_cost(0).total == 0.0
    assert model.sell_cost(0).total == 0.0
    buy = model.buy_cost(10000.0)
    sell = model.sell_cost(10000.0)
    assert buy.stamp_tax == 0.0
    assert sell.stamp_tax == pytest.approx(10000 * CFG.stamp_tax_rate)
    assert buy.transfer_fee == pytest.approx(10000 * CFG.transfer_fee_rate)
    assert buy.slippage == pytest.approx(10000 * CFG.slippage_rate)
    assert sell.total - buy.total == pytest.approx(sell.stamp_tax)


def test_scalar_and_batch_rebalance_costs_match():
    model = ExecutionCostModel.from_config(CFG)
    buys = np.array([[0.2, 0.0, 0.1], [0.0, 0.3, 0.0]])
    sells = np.array([[0.0, 0.15, 0.0], [0.2, 0.0, 0.1]])
    capitals = np.array([100000.0, 500000.0])
    batch = model.rebalance_cost(buys, sells, capitals)
    for i in range(2):
        scalar = model.rebalance_cost(buys[i], sells[i], capitals[i])
        for field in ("commission", "stamp_tax", "transfer_fee", "slippage", "total"):
            assert np.asarray(getattr(batch, field))[i] == pytest.approx(
                getattr(scalar, field)
            )


def test_affordable_shares_includes_all_fees_and_whole_lots():
    model = ExecutionCostModel.from_config(CFG)
    shares = model.affordable_shares(100000.0, 10.0, requested=20000)
    assert shares % 100 == 0
    payable = shares * 10.0 + float(model.buy_cost(shares * 10.0).total)
    assert payable <= 100000.0
    next_payable = (shares + 100) * 10.0 + float(
        model.buy_cost((shares + 100) * 10.0).total
    )
    assert next_payable > 100000.0


@pytest.mark.parametrize("capital", [100000.0, 500000.0, 1000000.0])
@pytest.mark.parametrize("top_n", [10, 20, 30, 50])
def test_exact_cost_path_covers_capital_and_topn_grid(capital: float, top_n: int):
    cfg = replace(
        CFG,
        initial_capital=capital,
        top_n=top_n,
        single_weight_cap=1.0,
    )
    n_stocks, n_dates = top_n + 2, 5
    signal = np.tile(np.arange(n_stocks, dtype=float)[:, None], (1, n_dates))
    target = np.zeros_like(signal)
    simulation = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=np.ones_like(signal, dtype=bool)
    )
    assert simulation.daily_cost_fractions.shape == (n_dates - 2,)
    assert simulation.turnover[0] == pytest.approx(1.0)
    assert np.all(simulation.daily_cost_fractions >= 0.0)


def test_execution_config_validation_reports_all_deployment_fields():
    sim = SimConfig(
        initial_capital=CFG.initial_capital,
        max_positions=CFG.top_n,
        single_weight_cap=CFG.single_weight_cap,
    )
    validate_execution_config(CFG, sim)
    drifted = replace(sim, initial_capital=500000.0, max_positions=20)
    mismatches = execution_config_mismatches(CFG, drifted)
    assert set(mismatches) == {"initial_capital", "top_n/max_positions"}
    with pytest.raises(ValueError, match="execution config mismatch"):
        validate_execution_config(CFG, drifted)
