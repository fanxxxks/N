"""P58: independent hand cash ledgers and ordinary-engine/paper parity."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import BacktestConfig
from ashare_data.processor import exact_tradability_matrix
from ashare_model.backtest import AshareBacktestEngine
from ashare_portfolio.constructor import PortfolioConstructor, canonical_weights_to_output_grid
from ashare_trading.matching import SimBroker
from ashare_trading.portfolio import SimulationPortfolio
from ashare_trading.run_sim import SimulationRunner
from tests.conftest import with_price_semantics


def market(n=20, days=3, price=10.0):
    codes = [f"{i:06d}.SZ" for i in range(n)]
    dates = [f"202401{i + 2:02d}" for i in range(days)]
    opens = np.full((n, days), price)
    raw = with_price_semantics(dict(open=opens, high=opens * 1.01,
        low=opens * .99, pre_close=opens.copy(), volume=np.full_like(opens, 1e6)))
    signals = np.tile(np.arange(n, 0, -1.)[:, None], (1, days))
    return signals, raw, codes, dates, np.ones_like(signals, dtype=bool)


def paper(cfg, data, tmp_path, due=None):
    signals, raw, codes, dates, mask = data
    portfolio = SimulationPortfolio(cfg.initial_capital, tmp_path / "state.json")
    # Exercise the runner's real order adapter without strategy/database I/O.
    runner = SimulationRunner.__new__(SimulationRunner)
    runner.portfolio = portfolio
    runner.backtest_config = cfg
    constructor = PortfolioConstructor(cfg)
    blocked_buy, _ = exact_tradability_matrix(raw["open"], raw["up_limit"],
        raw["down_limit"], raw["suspended"], "buy")
    blocked_sell, _ = exact_tradability_matrix(raw["open"], raw["up_limit"],
        raw["down_limit"], raw["suspended"], "sell")
    trades = []
    for t in range(len(dates) - 2):
        day = t + 1
        prev, equity = runner._portfolio_weights(codes, raw["open"][:, day])
        rebalance = True if due is None else bool(due[t])
        output = constructor.construct(signals[:, t], canonical_weights_to_output_grid(prev),
            capital=equity, eligible=mask[:, t] & mask[:, day],
            buy_blocked=blocked_buy[:, day], sell_blocked=blocked_sell[:, day],
            stable_keys=np.asarray(codes), rebalance_due=rebalance,
            adv=raw["volume"][:, day] * raw["open"][:, day])
        orders = runner._make_orders(np.asarray(output.weights), codes, day,
            dates[day], raw, equity=equity) if rebalance else []
        bars = pd.DataFrame([dict(ts_code=code, trade_date=dates[day],
            **{key: raw[key][i, day] for key in
            ("open", "high", "low", "pre_close", "volume", "up_limit", "down_limit", "suspended")})
            for i, code in enumerate(codes)])
        trades.extend(SimBroker().execute_orders(orders, bars, dates[day], portfolio,
            {}, cfg, lot_size=cfg.lot_size))
        assert portfolio.cash >= 0
    return portfolio, trades


def cfg(**kwargs):
    return replace(BacktestConfig(top_n=20, buy_rank=20, sell_rank=30,
        single_weight_cap=.05, min_trade_amount=1000, turnover_budget=.20), **kwargs)


@pytest.mark.parametrize("minimum, shares, fees, cash", [
    (1000, [500] * 19 + [400], 150.49, 849.51),
    (5000, [500] * 19 + [0], 143.45, 4856.55),
])
def test_fees_do_not_remove_a_lot_from_every_stock(tmp_path, minimum, shares, fees, cash):
    config = cfg(min_trade_amount=minimum)
    data = market()
    result = AshareBacktestEngine(config).run(*data)
    portfolio, trades = paper(config, data, tmp_path)
    assert result.final_shares == shares
    assert [portfolio.positions[c].quantity if c in portfolio.positions else 0
        for c in data[2]] == shares
    assert result.final_cash == pytest.approx(cash)
    assert portfolio.cash == pytest.approx(cash)
    assert sum(t.commission + t.stamp_tax + t.transfer_fee + t.slippage
        for t in trades) == pytest.approx(fees)
    assert result.cost_fractions[0] * config.initial_capital == pytest.approx(fees)


@pytest.mark.parametrize("capital, expected", [
    (10000, [0] * 20), (30000, [100] * 20), (50000, [200] * 20),
])
def test_capital_grid_without_relaxing_concentration(tmp_path, capital, expected):
    config = cfg(initial_capital=capital)
    data = market()
    result = AshareBacktestEngine(config).run(*data)
    portfolio, _ = paper(config, data, tmp_path)
    assert result.final_shares == expected
    assert result.final_cash == pytest.approx(portfolio.cash)
    assert result.final_cash >= 0
    assert config.top_n == 20 and config.single_weight_cap == .05


@pytest.mark.parametrize("allocation", ["priority", "proportional"])
def test_ordinary_replacement_has_identical_fills_and_fees(tmp_path, allocation):
    config = cfg(turnover_allocation=allocation, sell_rank=20)
    data = market(40, 5)
    data[0][:, 1:] = data[0][::-1, 1:]
    result = AshareBacktestEngine(config).run(*data)
    portfolio, trades = paper(config, data, tmp_path)
    np.testing.assert_allclose(result.final_shares,
        [portfolio.positions[c].quantity if c in portfolio.positions else 0 for c in data[2]],
        rtol=0, atol=1e-9)
    assert result.final_cash == pytest.approx(portfolio.cash, abs=1e-8)
    assert sum(np.asarray(result.cost_fractions) * np.asarray(result.equity_curve[:-1]) * config.initial_capital) == pytest.approx(
        sum(t.commission + t.stamp_tax + t.transfer_fee + t.slippage for t in trades))
    if allocation == "priority":
        # Existing sells-first budget rule: replacement session has no buys.
        assert not any(t.side == "buy" and t.trade_date == data[3][2] for t in trades)


def test_blocks_and_non_due_days_keep_cash_and_positions(tmp_path):
    config = cfg()
    data = market(days=5)
    data[1]["suspended"][0, :] = 1
    data[1]["up_limit"][1, :] = 10
    data[1]["up_limit"][2, :] = np.nan
    due = np.array([True, False, False, False, False])
    result = AshareBacktestEngine(config).run(*data, rebalance_mask=due)
    portfolio, _ = paper(config, data, tmp_path, due)
    assert result.final_shares[:3] == [0, 0, 0]
    assert result.final_cash == pytest.approx(portfolio.cash)
    assert result.turnover[1:] == [0, 0]


def test_nonstandard_lot_size_is_shared_by_runner_and_engine(tmp_path):
    config = cfg(lot_size=200)
    data = market()
    result = AshareBacktestEngine(config).run(*data)
    portfolio, _ = paper(config, data, tmp_path)
    assert result.final_shares == [400] * 20
    assert [portfolio.positions[c].quantity for c in data[2]] == [400] * 20
    assert result.final_cash == pytest.approx(portfolio.cash)


def test_sell_proceeds_too_small_to_pay_fees_do_not_overdraw(tmp_path):
    config = cfg(initial_capital=105, top_n=1, buy_rank=1, sell_rank=1,
        single_weight_cap=1, min_trade_amount=None, turnover_budget=None,
        commission_rate=0, transfer_fee_rate=0, slippage_rate=0, stamp_tax_rate=0)
    data = market(2, 4, price=1)
    data[0][:, 1:] = data[0][::-1, 1:]
    data[1]["open"][0, 2:] = .01
    data[1]["adjusted_open"] = data[1]["open"].copy()
    result = AshareBacktestEngine(config).run(*data)
    portfolio, trades = paper(config, data, tmp_path)
    assert result.final_cash == portfolio.cash == 0
    np.testing.assert_allclose(result.final_shares, [100, 0], rtol=0, atol=1e-9)
    assert len(trades) == 1 and trades[0].side == "buy"


def test_array_permutation_does_not_change_cash_priority(tmp_path):
    config = cfg()
    data = market()
    reverse = np.arange(19, -1, -1)
    changed = (data[0][reverse], {k: v[reverse] for k, v in data[1].items()},
        [data[2][i] for i in reverse], data[3], data[4][reverse])
    result = AshareBacktestEngine(config).run(*changed)
    portfolio, _ = paper(config, changed, tmp_path)
    assert result.final_shares == [400] + [500] * 19
    assert portfolio.positions[data[2][-1]].quantity == 400
    assert result.final_cash == pytest.approx(portfolio.cash)


@pytest.mark.parametrize("capital, quantities, cash", [
    (10000, [0] * 25, 10000),
    (30000, [100] * 25, 4862.25),
    (50000, [200] * 24 + [100], 850.01),
    (100000, [400] * 24 + [300], 824.51),
])
def test_more_diversified_profile_cash_reserve(tmp_path, capital, quantities, cash):
    config = cfg(initial_capital=capital, top_n=25, buy_rank=25,
        sell_rank=35, single_weight_cap=.04)
    data = market(25)
    result = AshareBacktestEngine(config).run(*data)
    portfolio, _ = paper(config, data, tmp_path)
    assert result.final_shares == quantities
    assert result.final_cash == pytest.approx(cash)
    assert portfolio.cash == pytest.approx(cash)


def test_price_too_high_for_single_name_cap_is_not_bought(tmp_path):
    config = cfg(initial_capital=30000, top_n=25, buy_rank=25,
        sell_rank=35, single_weight_cap=.04)
    data = market(25, price=13)
    result = AshareBacktestEngine(config).run(*data)
    portfolio, trades = paper(config, data, tmp_path)
    assert result.final_shares == [0] * 25
    assert result.final_cash == portfolio.cash == 30000
    assert not trades


def test_old_execution_provenance_cannot_be_reused():
    from ashare_portfolio.execution_spec import execution_provenance, validate_execution_provenance
    config = cfg()
    for version in (4, 5):
        recorded = execution_provenance(config)
        recorded["execution_version"] = version
        with pytest.raises(ValueError, match="does not match"):
            validate_execution_provenance(recorded, config)


@pytest.mark.parametrize("lot", [0, -1, 1.5, True])
def test_invalid_lot_size_is_rejected(lot):
    with pytest.raises(ValueError, match="lot_size"):
        PortfolioConstructor(cfg(lot_size=lot))


def test_price_drift_partial_sells_keep_both_paths_identical(tmp_path):
    config = cfg(initial_capital=1e6, top_n=2, buy_rank=2, sell_rank=2,
        single_weight_cap=.5, min_trade_amount=None, turnover_budget=None,
        commission_rate=.001, stamp_tax_rate=.001, transfer_fee_rate=0, slippage_rate=0)
    data = market(2, 5, price=100)
    data[1]["open"][0, 2:] = 200
    data[1]["adjusted_open"] = data[1]["open"].copy()
    due = np.array([True, True, False, False, False])
    result = AshareBacktestEngine(config).run(*data, rebalance_mask=due)
    portfolio, trades = paper(config, data, tmp_path, due)
    np.testing.assert_allclose(result.final_shares, [3800, 7300], rtol=0, atol=1e-7)
    assert [portfolio.positions[c].quantity for c in data[2]] == [3800, 7300]
    assert result.final_cash == pytest.approx(8290)
    assert portfolio.cash == pytest.approx(8290)
    assert [(tr.side, tr.quantity) for tr in trades] == [
        ("buy", 5000), ("buy", 4900), ("sell", 1200), ("buy", 2400)]


@pytest.mark.parametrize("capital", [30000, 50000, 100000])
def test_delivered_config_uses_same_size_and_normal_fees(capital):
    from pathlib import Path
    import yaml
    from ashare_data.config import make_backtest_config, make_sim_config
    from ashare_execution import validate_execution_config
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / f"config/small_capital_{capital}.yaml").read_text(encoding="utf-8"))
    config = make_backtest_config(raw)
    validate_execution_config(config, make_sim_config(raw, root))
    assert config.initial_capital == capital
    assert config.top_n == config.buy_rank == 25
    assert config.single_weight_cap == .04
    assert config.min_trade_amount == 1000
    assert config.rebalance_frequency == "every_10_days" and config.target_horizon == 10
    assert config.commission_rate == .00025 and config.min_commission == 5
    assert config.stamp_tax_rate == .0005 and config.transfer_fee_rate == .00001
    assert config.slippage_rate == .0005 and config.lot_size == 100
