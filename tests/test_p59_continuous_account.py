"""P59: independent cash/year oracles; no yearly funding or plan-as-fill."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from ashare_model.backtest import AshareBacktestEngine
from ashare_model.continuous_account import (
    annual_returns, assemble_frozen_signal, first_rebalance_mask, account_periods,
    summarize_account, create_output_directory,
)
from tests.test_p58_small_capital_execution import cfg, market, paper


def directions():
    return {"frozen": True, "domain": "medium", "rules": {"B": "label_ic"},
        "entries": [{"candidate": name, "rule": "B", "fold_test_end": f"{year}1231",
            "direction": direction, "provenance": {"candidate": name,
                "direction": direction, "rule": "label_ic", "train_cutoff": f"{year-1}1231",
                "label": {"name": "causal_target_returns", "basis": "adjusted_open_o2o", "horizon": 10}}}
            for year in (2021, 2022) for name, direction in
            (("MOMENTUM_20", -1), ("RSQ_60", 1), ("REVERSAL_60", 1))]}


def test_annual_wealth_chains_previous_year_end():
    # One 100-RMB deposit: 100 -> 120 -> 90 -> 99. Second year +10%, not -1%.
    annual = annual_returns(["20210104", "20210601", "20211231", "20221230"], [1, 1.2, .9, .99])
    assert annual == pytest.approx({"2021": -.1, "2022": .1})
    assert np.prod([1 + x for x in annual.values()]) == pytest.approx(.99)


def test_drawdown_and_worst_year_use_continuous_wealth():
    data = list(market(days=5))
    data[3] = ["20211228", "20211229", "20211230", "20211231", "20221230"]
    data[1]["open"][:] = [10, 10, 12, 9, 9.9]
    data[1]["adjusted_open"][:] = data[1]["open"]
    config = cfg(commission_rate=0, min_commission=0, stamp_tax_rate=0,
        transfer_fee_rate=0, slippage_rate=0)
    result = AshareBacktestEngine(config).run(*data,
        rebalance_mask=np.array([True, False, False, False, False]))
    row = summarize_account(result, 100000, data[1], data[2])
    assert row["terminal_net_assets"] == pytest.approx(99000)
    assert row["net_profit_rmb"] == pytest.approx(-1000)
    assert row["annual_returns"] == pytest.approx({"2021": -.1, "2022": .1})
    assert row["max_drawdown"] == pytest.approx(.25)
    assert row["worst_year"] == "2021"
    assert row["net_cagr"] == pytest.approx(.99 ** (252/3) - 1)


def test_continuous_engine_does_not_inject_funds_at_new_year(tmp_path):
    data = list(market(days=5))
    data[3] = ["20211229", "20211230", "20211231", "20220104", "20220105"]
    config = cfg()
    due = np.array([True, False, False, False, False])
    result = AshareBacktestEngine(config).run(*data, rebalance_mask=due)
    portfolio, fills = paper(config, data, tmp_path, due)
    assert result.final_cash == pytest.approx(849.51)
    assert result.final_cash == pytest.approx(portfolio.cash)
    assert len(result.trades) == len(fills) == 20
    assert {t["date"] for t in result.trades} == {"20211230"}
    assert result.equity_curve[-1] * 100000 == pytest.approx(99849.51)
    report = summarize_account(result, 100000, data[1], data[2])
    assert report["net_profit_rmb"] == pytest.approx(-150.49)
    assert report["fees"]["commission"] == 100
    assert report["fees"]["stamp_tax"] == 0
    assert report["fees"]["transfer_fee"] == pytest.approx(.99)
    assert report["fees"]["slippage"] == pytest.approx(49.5)
    assert report["fees"]["total"] == pytest.approx(150.49)
    assert report["annual_returns"]["2022"] == 0
    assert report["max_drawdown"] == pytest.approx(.0015049)
    assert report["worst_year"] == "2021"
    assert report["actual_turnover_total"] == pytest.approx(.99)
    assert report["cash_min"] >= 0
    for position in account_periods(result):
        assert position["equity_before"] == pytest.approx(position["equity_after_execution"] + position["fees"])


def test_plan_is_not_a_fill_and_lot_failures_are_visible():
    result = AshareBacktestEngine(cfg(initial_capital=10000, min_trade_amount=None)).run(*market())
    report = summarize_account(result, 10000, market()[1], market()[2])
    assert report["plan_turnover_total"] > 0
    assert report["actual_turnover_total"] == 0
    assert report["fees"]["total"] == 0
    assert report["execution_rejections"]["below_board_lot"] == 20


def test_passive_has_same_initial_fills_and_no_later_trades():
    data = market(days=5)
    due = np.ones(5, dtype=bool)
    active = AshareBacktestEngine(cfg()).run(*data, rebalance_mask=due)
    passive = AshareBacktestEngine(cfg()).run(*data, rebalance_mask=first_rebalance_mask(due))
    initial = [t for t in active.trades if t["date"] == data[3][1]]
    assert initial == passive.trades
    assert len(initial) == 20
    assert first_rebalance_mask(np.zeros(5, dtype=bool)).sum() == 0


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "future", "horizon", "unfrozen"])
def test_frozen_segments_fail_closed(mutation):
    payload = directions()
    if mutation == "duplicate": payload["entries"].append(deepcopy(payload["entries"][0]))
    if mutation == "missing": payload["entries"].pop()
    if mutation == "future": payload["entries"][0]["provenance"]["train_cutoff"] = "20210101"
    if mutation == "horizon": payload["entries"][0]["provenance"]["label"]["horizon"] = 1
    if mutation == "unfrozen": payload["frozen"] = False
    factors = {name: np.ones((2, 3)) for name in ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")}
    with pytest.raises(ValueError):
        assemble_frozen_signal(factors, np.ones((2, 3), dtype=bool),
            ["20210104", "20211231", "20220104"], payload)


def test_signal_covers_year_end_without_horizon_gap():
    factors = {name: np.array([[1, 2, 3], [2, 1, 1.]]) for name in
        ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")}
    signal, segments = assemble_frozen_signal(factors, np.ones((2, 3), dtype=bool),
        ["20210104", "20211231", "20220104"], directions())
    assert np.isfinite(signal).all()
    assert sum(s["date_count"] for s in segments) == 3
    assert segments[0]["last_signal_date"] == "20211231"


def test_terminal_missing_price_invalidates_complete_valuation():
    data = market()
    result = AshareBacktestEngine(cfg()).run(*data)
    data[1]["open"][0, -1] = np.nan
    report = summarize_account(result, 100000, data[1], data[2])
    assert report["valuation_complete"] is False
    assert report["terminal_missing_prices"] == [data[2][0]]


def test_evidence_directory_never_overwrites(tmp_path):
    target = tmp_path / "evidence"
    create_output_directory(target)
    (target / "old.json").write_text("old evidence")
    with pytest.raises(FileExistsError): create_output_directory(target)
    assert (target / "old.json").read_text() == "old evidence"


def test_cache_only_mode_refuses_rebuild_before_any_cache_write(populated_db):
    from tests.test_b04_load_cache_performance import _seed_identity_sources
    from ashare_model.data_loader import AshareDataLoader
    config = populated_db
    _seed_identity_sources(config)
    loader = AshareDataLoader(config, require_factor_cache_hit=True)
    with pytest.raises(RuntimeError, match="factor cache required"):
        loader.load_data()
    assert not list((config.data_dir / "factor_tensor_cache").glob("*.npz"))


def snapshot_members():
    def table(name):
        return dict(table=name, schema_digest="same-schema", row_count=2,
            partitions=[dict(partition_key="2021", hash="historical", row_count=1),
                dict(partition_key="2026", hash="recent", row_count=1)])
    return {"primary": {"dataset_id": "primary", "tables": [table(n) for n in
            ("trade_calendar", "constituents", "stocks")]},
        "tushare_v2": {"dataset_id": "v2", "tables": [table(n) for n in
            ("daily_bar_raw", "daily_adj_factor", "daily_price_limit", "daily_suspension")]}}


@pytest.mark.parametrize("change", ["historical_price", "stock", "missing", "schema"])
def test_snapshot_bridge_rejects_historical_mismatch(change):
    from ashare_model.continuous_account import validate_snapshot_bridge
    reference, backup = snapshot_members(), snapshot_members()
    if change == "historical_price": backup["tushare_v2"]["tables"][0]["partitions"][0]["hash"] = "changed"
    if change == "stock": backup["primary"]["tables"][2]["partitions"][1]["hash"] = "changed"
    if change == "missing": backup["tushare_v2"]["tables"].pop()
    if change == "schema": backup["tushare_v2"]["tables"][1]["schema_digest"] = "changed"
    with pytest.raises(ValueError): validate_snapshot_bridge(reference, backup)


def test_snapshot_bridge_allows_only_unconsumed_changes():
    from ashare_model.continuous_account import validate_snapshot_bridge
    reference, backup = snapshot_members(), snapshot_members()
    reference["tushare_v2"]["tables"][2]["partitions"][1]["hash"] = "post-development-window"
    reference["tushare_v2"]["tables"].append(dict(table="source_ingest_runs", metadata="new"))
    bridge = validate_snapshot_bridge(reference, backup)
    assert bridge["historical_inputs_identical"] is True
    assert bridge["through_year"] == "2025"
    assert set(bridge["changed_tables"]["tushare_v2"]) == {"daily_price_limit", "source_ingest_runs"}
