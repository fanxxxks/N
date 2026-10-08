"""p53 / WP16-B: P45 comparability of the parity harness (scoring_parity).

Assertion source: ``reports/wp16/FROZEN_B.md`` (frozen 2026-09-23, before
any WP16 code existed).  Every assertion cites the frozen §/invariant it
pins; hand numbers are exact ``Fraction`` oracles (the §2 header's
fractions-recheck discipline), and the label-basis oracle deliberately
calls the SAME shared ``open_to_open_returns`` the harness uses (B1.1:
no second o2o implementation may appear in tests or harness).

Scenario fixtures realize the frozen §2 expected VALUES on the session
axis the shared o2o defines (col ``t`` spans ``(t+1, t+2)``): the split
is placed so the first session's entry-exit window straddles it, which
makes the negative-control numbers appear verbatim — o2o(adjusted) col0
= 0 vs o2o(raw) col0 = -50%.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_data.processor import (
    exact_tradability_matrix,
    open_to_open_returns,
)
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.reward import simulate_basket_daily_returns
from ashare_model.scoring_parity import (
    compare_candidate_parity,
    parity_bounds,
    proxy_reliability_verdict,
)
from tests.conftest import with_price_semantics

# FROZEN_B §2 header: zero-fee zero-lot constructions are exact to 1e-12.
_REL = 1e-12
# The engine's default execution delay (backtest.py run signature); the
# harness never overrides it, so B3.3's message elements use delay+1 = 2.
_DELAY = 1

# FROZEN_B §4.2 frozen disclosure string (verbatim).
DISCLOSURE = (
    "proxy/engine_direction = 测试窗累计净收益择向（诊断），"
    "非生产 Rule A 验证 reward 择向"
)
# FROZEN_B §4.3 forbidden phrasings (any output surface).
FORBIDDEN = (
    "生产择向路径已复现",
    "production direction path reproduced",
    "择向规则不一致",
    "择向规则失效",
    "direction validated",
    "direction verified",
)


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=105000.0,
        top_n=1,
        single_weight_cap=1.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
        lot_size=1,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _panel(open_raw: np.ndarray, adjusted: np.ndarray) -> dict[str, np.ndarray]:
    """The p19 §2 price-semantics cache for a synthetic two-name panel."""

    return with_price_semantics({
        "open": open_raw,
        "high": open_raw * 1.02,
        "low": open_raw * 0.98,
        "pre_close": open_raw,
        "volume": np.full_like(open_raw, 1_000_000.0),
        "adjusted_open": adjusted,
    })


def _dates(n: int) -> list[str]:
    return [f"202401{i + 1:02d}" for i in range(n)]


# ---------------------------------------------------------------------------
# §2 S1 — split-neutral (B1.1/B1.3/B2.1, direction no-flip)
# ---------------------------------------------------------------------------


def _s1_panel():
    """day2 is the first post-split session (1:2 split between day1/day2).

    adjusted = raw x factor: name0 (20,20,20,21) with raw (20,20,10,10.5)
    and factor (1,1,2,2); name1 is a flat control.  On the shared o2o
    session axis (col t spans (t+1, t+2)):
      o2o(adjusted) name0 = (0, +5%, 0, 0)   <- the correct values
      o2o(raw)      name0 = (-50%, +5%, 0, 0) <- the degenerate basis
    Capital 105000 divides every entry price exactly (20 -> 5250 shares,
    10 -> 10500 shares): zero lot residual, so parity is exact (§2 S1).
    """

    open_raw = np.array(
        [[20.0, 20.0, 10.0, 10.5], [10.0, 10.0, 10.0, 10.0]]
    )
    adjusted = np.array(
        [[20.0, 20.0, 20.0, 21.0], [10.0, 10.0, 10.0, 10.0]]
    )
    raw = _panel(open_raw, adjusted)
    signal = np.array([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    codes = ["000001.SZ", "000002.SZ"]
    return codes, _dates(4), raw, signal


def test_p53_s1_split_neutral_both_paths_exact_and_no_flip():
    # B1.1 oracle: the SAME shared o2o function defines the expected
    # session returns (no second implementation in the test).
    codes, dates, raw, signal = _s1_panel()
    expected = open_to_open_returns(raw["adjusted_open"])
    assert expected[0, 0] == pytest.approx(0.0, abs=0.0)
    assert expected[0, 1] == pytest.approx(F(1, 20), abs=1e-15)
    # Negative control precondition (B1.3): the raw basis really does
    # produce the -50% fake return on the split-straddling session.
    raw_basis = open_to_open_returns(raw["open"])
    assert raw_basis[0, 0] == pytest.approx(-0.5, abs=1e-15)

    cfg = _cfg()  # zero fees, lot 1, capital 105000
    mask = np.ones(4, dtype=bool)
    universe = np.ones((2, 4), dtype=bool)
    lsr = (0, 2)

    sim = simulate_basket_daily_returns(
        signal, open_to_open_returns(raw["adjusted_open"]), cfg,
        universe_mask=universe, tie_break_keys=np.asarray(codes),
        rebalance_mask=mask, signal_range=lsr,
    )
    engine = AshareBacktestEngine(cfg).run(
        signal, dict(raw), list(codes), list(dates),
        universe_mask=universe, rebalance_mask=mask, signal_range=lsr,
    )
    # B3.2: both paths emit exactly s1 - s0 sessions.
    assert sim.daily_net_returns.size == 2
    assert len(engine.daily_returns) == 2

    # §2 S1: both paths' session returns equal o2o(adjusted) exactly
    # (zero fee, zero lot residual; tolerance 0/1e-12)...
    np.testing.assert_allclose(
        sim.daily_net_returns, [0.0, float(F(1, 20))], atol=_REL
    )
    np.testing.assert_allclose(
        engine.daily_returns, [0.0, float(F(1, 20))], atol=_REL
    )
    # ...and the -50% raw-basis value never appears (B1.3 "错误值不出现").
    assert -0.5 not in sim.daily_net_returns
    assert -0.5 not in engine.daily_returns

    # §2 S1: engine shares double at the split with notional unchanged
    # (B2.1: raw open re-derivation) and the equity path is (1, 1, 1.05).
    assert engine.positions[0]["shares"] == [5250.0]
    assert engine.positions[1]["shares"] == [10500.0]
    np.testing.assert_allclose(
        engine.equity_curve, [1.0, 1.0, 1.05], atol=_REL
    )
    # §2 S1: the split session books no phantom fee or turnover (the
    # t=1 entry is the post-split re-derivation with no delta orders).
    assert engine.cost_fractions[1] == 0.0
    assert engine.turnover[1] == 0.0
    assert sim.daily_cost_fractions[1] == 0.0
    assert sim.turnover[1] == 0.0

    # Harness level: same window explicit through both paths (F3), no
    # direction flip for the candidate (§2 S1 "方向无翻转").
    report = compare_candidate_parity(
        [("split", 1, signal)], raw, codes, dates,
        universe_mask=universe, backtest_config=cfg,
        rebalance_mask=mask, local_signal_range=lsr,
    )
    row = report.rows[0]
    assert row.candidate not in report.direction_flips
    assert row.proxy_direction == row.engine_direction == 1
    assert row.daily_net_error_max <= _REL
    assert row.fee_fraction_error_max == 0.0
    assert row.turnover_error_max == 0.0


# ---------------------------------------------------------------------------
# §2 S2 — non-rebalance price move (drift ledger, exact zero cost)
# ---------------------------------------------------------------------------


def _s2_panel():
    """day0 rebalance buys a = (1/2, 1/2) at the day1 open (10, 10); the
    two session returns are r1 = (+0.02, -0.01) then r2 = (+0.04, +0.02)
    on the shared o2o session axis; rebalance_mask is True only on day0.
    """

    open_raw = np.array(
        [[10.0, 10.0, 10.2, 10.608], [10.0, 10.0, 9.9, 10.098]]
    )
    raw = _panel(open_raw, open_raw.copy())
    signal = np.array([[1.0, 1.0, 1.0, 1.0], [0.6, 0.6, 0.6, 0.6]])
    codes = ["000001.SZ", "000002.SZ"]
    return codes, _dates(4), raw, signal


_S2_G1 = F(1, 200)            # 0.5*0.02 + 0.5*(-0.01)
_S2_W1 = (F(102, 201), F(99, 201))   # drifted book after session 0
_S2_G2 = F(101, 3350)         # (102/201)*0.04 + (99/201)*0.02
_S2_CUM = (1 + _S2_G1) * (1 + _S2_G2)   # = 693651/670000


def test_p53_s2_non_rebalance_move_books_on_drift_ledger():
    codes, dates, raw, signal = _s2_panel()
    cfg = _cfg(
        initial_capital=100000.0, top_n=2, single_weight_cap=0.5
    )
    universe = np.ones((2, 4), dtype=bool)
    mask = np.array([True, False, False, False])
    lsr = (0, 2)
    target = open_to_open_returns(raw["adjusted_open"])

    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=universe,
        tie_break_keys=np.asarray(codes), rebalance_mask=mask,
        signal_range=lsr,
    )
    engine = AshareBacktestEngine(cfg).run(
        signal, dict(raw), list(codes), list(dates),
        universe_mask=universe, rebalance_mask=mask, signal_range=lsr,
    )
    # §2 S2 hand path (Fractions): session nets are g1 then g2 (c = 0).
    np.testing.assert_allclose(
        sim.daily_net_returns,
        [float(_S2_G1), float(_S2_G2)], atol=_REL,
    )
    np.testing.assert_allclose(
        engine.daily_returns,
        [float(_S2_G1), float(_S2_G2)], atol=_REL,
    )
    # "两路径逐日 net 之差 = 0（1e-12）" — returns ride the drift ledger.
    np.testing.assert_allclose(
        sim.daily_net_returns, engine.daily_returns, atol=_REL
    )
    np.testing.assert_allclose(
        engine.equity_curve,
        [1.0, float(1 + _S2_G1), float(_S2_CUM)], atol=_REL,
    )
    # §2 S2: the non-rebalance column books EXACT zero cost and turnover
    # (not approximately zero) on both paths; B6.2: with the zero-fee
    # configuration both fee series are identically zero.
    assert sim.daily_cost_fractions[1] == 0.0
    assert sim.turnover[1] == 0.0
    assert engine.cost_fractions[1] == 0.0
    assert engine.turnover[1] == 0.0
    assert np.all(sim.daily_cost_fractions == 0.0)
    assert np.all(np.asarray(engine.cost_fractions) == 0.0)

    # B4.3: engine-reported entry/exit dates live on the actual date axis.
    date_set = set(dates)
    for pos in engine.positions:
        assert pos["signal_date"] in date_set
        assert pos["entry_date"] in date_set
        assert pos["exit_date"] in date_set

    report = compare_candidate_parity(
        [("drift", 1, signal)], raw, codes, dates,
        universe_mask=universe, backtest_config=cfg,
        rebalance_mask=mask, local_signal_range=lsr,
    )
    row = report.rows[0]
    assert row.daily_net_error_max <= _REL
    assert row.fee_fraction_error_max == 0.0
    assert row.turnover_error_max == 0.0
    assert row.blocked_extra_buy_columns == 0


# ---------------------------------------------------------------------------
# §2 S3 — fee financing (affordable equal-weight deployment)
# ---------------------------------------------------------------------------


def _s3_panel():
    open_raw = np.array(
        [[10.0, 10.0, 10.2, 10.2], [10.0, 10.0, 10.1, 10.1]]
    )
    raw = _panel(open_raw, open_raw.copy())
    signal = np.array([[1.0, 1.0, 1.0, 1.0], [0.6, 0.6, 0.6, 0.6]])
    codes = ["000001.SZ", "000002.SZ"]
    return codes, _dates(4), raw, signal


def test_p53_s3_fee_financing_affordable_scaling_and_cash_nonneg():
    codes, dates, raw, signal = _s3_panel()
    cfg = _cfg(
        initial_capital=100000.0, top_n=2, single_weight_cap=0.5,
        commission_rate=0.001,
    )
    universe = np.ones((2, 4), dtype=bool)
    mask = np.array([True, False, False, False])
    lsr = (0, 1)
    target = open_to_open_returns(raw["adjusted_open"])

    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=universe,
        tie_break_keys=np.asarray(codes), rebalance_mask=mask,
        signal_range=lsr,
    )
    engine = AshareBacktestEngine(cfg).run(
        signal, dict(raw), list(codes), list(dates),
        universe_mask=universe, rebalance_mask=mask, signal_range=lsr,
    )
    # §2 S3 engine hand path (one affordability pass converges):
    # 4995 shares per name -> a = (0.4995, 0.4995), c = 999/1e6,
    # R_net = g - c = 6993/500000, cash stays +0.10 >= 0.
    np.testing.assert_allclose(
        engine.buy_weights[0], [0.4995, 0.4995], atol=1e-12
    )
    assert engine.cost_fractions[0] == pytest.approx(
        float(F(999, 1_000_000)), abs=1e-12
    )
    assert engine.daily_returns[0] == pytest.approx(
        float(F(6993, 500_000)), abs=1e-9
    )
    assert engine.final_cash == pytest.approx(0.10, abs=1e-9)
    assert engine.final_cash >= 0.0
    # §2 S3 basket invariants (16-A fixed): the additive identity holds
    # session-wise, the post-trade book satisfies sum(a) + c <= 1 and
    # the carried cash weight is never negative.
    assert sim.daily_net_returns[0] == pytest.approx(
        float(sim.daily_gross_returns[0] - sim.daily_cost_fractions[0]),
        abs=0.0,
    )
    booked = float(sim.target_weights[0].sum())
    assert booked + sim.daily_cost_fractions[0] <= 1.0 + 1e-9
    assert sim.daily_cash_weights[0] >= -1e-12
    # §2 S3 error families: within the lot-granularity bounds (g = 1e-4;
    # the continuous-reserve basket vs the lot-floored engine residual).
    bounds = parity_bounds(cfg, price_ref=10.0, n_periods=1)
    assert abs(sim.daily_cost_fractions[0] - engine.cost_fractions[0]) <= (
        bounds["fee_fraction"]
    )
    assert abs(sim.daily_net_returns[0] - engine.daily_returns[0]) <= (
        bounds["daily_net"]
    )

    report = compare_candidate_parity(
        [("fees", 1, signal)], raw, codes, dates,
        universe_mask=universe, backtest_config=cfg,
        rebalance_mask=mask, local_signal_range=lsr,
    )
    row = report.rows[0]
    assert row.daily_net_error_max <= report.bounds["daily_net"]
    assert row.fee_fraction_error_max <= report.bounds["fee_fraction"]
    assert row.turnover_error_max <= report.bounds["turnover"]
    assert row.blocked_extra_buy_columns == 0


# ---------------------------------------------------------------------------
# §2 S4 — fold tail + 1 day: rejection (B4.1/B4.2/B4.4)
# ---------------------------------------------------------------------------


def test_p53_s4a_engine_side_dates_one_day_longer_is_rejected():
    # (a) dates injection: the date axis carries one more entry than the
    # price/universe columns -> ValueError with BOTH counts (B4.2),
    # before any path runs or error is computed (B4.4).
    codes, dates, raw, signal = _s2_panel()
    with pytest.raises(ValueError, match="date axis"):
        compare_candidate_parity(
            [("s4a", 1, signal)], raw, codes, dates + ["20240199"],
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
        )


def test_p53_s4a_engine_side_cache_one_column_longer_is_rejected():
    # (a) raw_cache injection: the engine-side cache carries one more
    # price column than the window -> ValueError naming the key and both
    # shapes (B4.2 message carries both counts), never a quiet run on
    # the longer tail.
    codes, dates, raw, signal = _s2_panel()
    bad = dict(raw)
    bad["adjusted_open"] = np.concatenate(
        [raw["adjusted_open"], raw["adjusted_open"][:, :1]], axis=1
    )
    with pytest.raises(ValueError, match="adjusted_open"):
        compare_candidate_parity(
            [("s4a2", 1, signal)], bad, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
        )


def test_p53_s4b_sim_side_signal_one_column_longer_is_rejected():
    # (b) sim-side injection: the signal matrix carries one more column
    # than the window -> ValueError (shape check names the candidate).
    codes, dates, raw, signal = _s2_panel()
    long_signal = np.concatenate([signal, signal[:, :1]], axis=1)
    with pytest.raises(ValueError, match="signal shape"):
        compare_candidate_parity(
            [("s4b", 1, long_signal)], raw, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
        )


def _fake_engine_result(n: int):
    return SimpleNamespace(
        daily_returns=[0.01] * n,
        cost_fractions=[0.001] * n,
        turnover=[0.5] * n,
    )


def test_p53_s4c_series_length_mismatch_four_element_valueerror(monkeypatch):
    # B4.1: the harness itself must refuse mismatched output lengths —
    # no min-trim, no partial rows (B4.4).  The stubbed engine returns
    # one session fewer than the basket on the same valid inputs.
    import ashare_model.backtest as backtest_mod

    codes, dates, raw, signal = _s2_panel()

    class _Short:
        def __init__(self, cfg):
            self.config = cfg

        def run(self, *args, **kwargs):
            return _fake_engine_result(1)

    monkeypatch.setattr(backtest_mod, "AshareBacktestEngine", _Short)
    with pytest.raises(ValueError) as excinfo:
        compare_candidate_parity(
            [("s4cand", 1, signal)], raw, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
            rebalance_mask=np.array([True, False, False, False]),
        )
    message = str(excinfo.value)
    # Frozen four elements: "length mismatch" + sim={n} + engine={m} +
    # the candidate/window identifier (FROZEN_B §1 B4.1).
    assert "length mismatch" in message
    assert "sim=2" in message
    assert "engine=1" in message
    assert "s4cand" in message


def test_p53_s4c_fee_and_turnover_families_reject_separately(monkeypatch):
    # B4.1: the fee and turnover families assert their OWN lengths (not
    # the daily family's n).
    import ashare_model.backtest as backtest_mod

    codes, dates, raw, signal = _s2_panel()

    class _ShortFees:
        def __init__(self, cfg):
            self.config = cfg

        def run(self, *args, **kwargs):
            ok = _fake_engine_result(2)
            return SimpleNamespace(
                daily_returns=ok.daily_returns,
                cost_fractions=ok.cost_fractions[:1],
                turnover=ok.turnover,
            )

    monkeypatch.setattr(backtest_mod, "AshareBacktestEngine", _ShortFees)
    with pytest.raises(ValueError, match="cost fractions"):
        compare_candidate_parity(
            [("s4fee", 1, signal)], raw, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
            rebalance_mask=np.array([True, False, False, False]),
        )

    class _ShortTurnover(_ShortFees):
        def run(self, *args, **kwargs):
            ok = _fake_engine_result(2)
            return SimpleNamespace(
                daily_returns=ok.daily_returns,
                cost_fractions=ok.cost_fractions,
                turnover=ok.turnover[:1],
            )

    monkeypatch.setattr(backtest_mod, "AshareBacktestEngine", _ShortTurnover)
    with pytest.raises(ValueError, match="turnover"):
        compare_candidate_parity(
            [("s4turn", 1, signal)], raw, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
            rebalance_mask=np.array([True, False, False, False]),
        )


def test_p53_s4_control_equal_length_input_completes(monkeypatch):
    # §2 S4 reverse assertion: the same stubbed wiring with EQUAL lengths
    # completes normally — the rejection logic does not misfire.
    import ashare_model.backtest as backtest_mod

    codes, dates, raw, signal = _s2_panel()

    class _Same:
        def __init__(self, cfg):
            self.config = cfg

        def run(self, *args, **kwargs):
            return _fake_engine_result(2)

    monkeypatch.setattr(backtest_mod, "AshareBacktestEngine", _Same)
    report = compare_candidate_parity(
        [("s4ok", 1, signal)], raw, codes, dates,
        universe_mask=np.ones((2, 4), dtype=bool),
        backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
        rebalance_mask=np.array([True, False, False, False]),
    )
    assert len(report.rows) == 1
    assert report.rows[0].candidate == "s4ok"


# ---------------------------------------------------------------------------
# §1 F3 — local_signal_range threading, exit validation (B3.x)
# ---------------------------------------------------------------------------


class _SimSpy:
    calls: list[dict] = []

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, signal, target, cfg, *args, **kwargs):
        _SimSpy.calls.append(
            {
                "signal_range": kwargs.get("signal_range"),
                "cfg_id": id(cfg),
                "signal": signal,
                "n_sessions": None,
            }
        )
        return self._inner(signal, target, cfg, *args, **kwargs)


def test_p53_b31_same_local_signal_range_and_cfg_reach_both_paths(monkeypatch):
    # B3.1 + B6.1: one harness call hands the SAME (s0, s1) and the SAME
    # cfg object to the basket and the engine.
    import ashare_model.backtest as backtest_mod
    import ashare_model.reward as reward_mod

    codes, dates, raw, signal = _s2_panel()
    cfg = _cfg(initial_capital=100000.0, top_n=2, single_weight_cap=0.5)

    real_sim = reward_mod.simulate_basket_daily_returns
    spy_calls: list[tuple[str, object]] = []
    sim_spy = _SimSpy(real_sim)
    _SimSpy.calls = []

    real_engine = backtest_mod.AshareBacktestEngine

    class _EngineSpy:
        def __init__(self, engine_cfg):
            spy_calls.append(("engine_cfg", engine_cfg))
            self._inner = real_engine(engine_cfg)

        def run(self, *args, **kwargs):
            spy_calls.append(("engine_range", kwargs.get("signal_range")))
            return self._inner.run(*args, **kwargs)

    monkeypatch.setattr(
        reward_mod, "simulate_basket_daily_returns", sim_spy
    )
    monkeypatch.setattr(backtest_mod, "AshareBacktestEngine", _EngineSpy)

    lsr = (0, 2)
    compare_candidate_parity(
        [("b31", 1, signal)], raw, codes, dates,
        universe_mask=np.ones((2, 4), dtype=bool), backtest_config=cfg,
        rebalance_mask=np.array([True, False, False, False]),
        local_signal_range=lsr,
    )
    sim_ranges = [c["signal_range"] for c in _SimSpy.calls]
    engine_ranges = [v for kind, v in spy_calls if kind == "engine_range"]
    sim_cfg_ids = {c["cfg_id"] for c in _SimSpy.calls}
    engine_cfgs = [v for kind, v in spy_calls if kind == "engine_cfg"]
    assert sim_ranges and all(r == lsr for r in sim_ranges)
    assert engine_ranges and all(r == lsr for r in engine_ranges)
    # B6.1: both paths are built from the one cfg object.
    assert sim_cfg_ids == {id(cfg)}
    assert all(id(ec) == id(cfg) for ec in engine_cfgs)


def test_p53_b33_incomplete_exit_tail_rejected_with_elements():
    # B3.3/B3.5: a window whose last signal lacks a complete t+delay+1
    # exit is refused BEFORE any run, with the offending index, n_dates
    # and the t+delay+1 requirement in the message.
    codes, dates, raw, signal = _s2_panel()
    with pytest.raises(ValueError) as excinfo:
        compare_candidate_parity(
            [("b33", 1, signal)], raw, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
            rebalance_mask=np.array([True, False, False, False]),
            local_signal_range=(0, 3),  # t=2: 2 + delay + 1 = 4 >= n_dates
        )
    message = str(excinfo.value)
    assert "t=2" in message
    assert "n_dates=4" in message
    assert re.search(r"t\+2", message)


def test_p53_b34_engine_contiguity_semantics_pinned():
    # B3.4: the engine's own contiguous-window rule (backtest.py:233-236
    # semantics) is pinned directly — the harness tuple API can only
    # express intervals, so the underlying guard is the authority.  The
    # gapped range [0, 2] on a 6-column window has complete t+2 exits,
    # so ONLY the contiguity guard fires.
    codes, _, raw, signal = _s2_panel()
    open6 = np.hstack([raw["open"], raw["open"][:, -1:], raw["open"][:, -1:]])
    raw6 = _panel(open6, open6.copy())
    signal6 = np.hstack([signal, signal[:, -1:], signal[:, -1:]])
    with pytest.raises(ValueError, match="contiguous"):
        AshareBacktestEngine(
            _cfg(top_n=2, single_weight_cap=0.5)
        ).run(
            signal6, dict(raw6), list(codes), _dates(6),
            universe_mask=np.ones((2, 6), dtype=bool),
            rebalance_mask=np.ones(6, dtype=bool),
            signal_range=range(0, 4, 2),
        )


# ---------------------------------------------------------------------------
# §1 F4/B1.2 — input-axis validation
# ---------------------------------------------------------------------------


def test_p53_b12_missing_adjusted_open_fails_closed_before_comparison():
    # B1.2 harness side: a cache missing adjusted_open is refused before
    # any comparison runs (both paths must fail closed on the label key).
    codes, dates, raw, signal = _s2_panel()
    bad = {k: v for k, v in raw.items() if k != "adjusted_open"}
    with pytest.raises(ValueError, match="adjusted_open"):
        compare_candidate_parity(
            [("b12", 1, signal)], bad, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
        )


def test_p53_b42_duplicate_dates_rejected():
    # B4.2: a date axis with a repeated entry is refused (the session
    # identity would be ambiguous).
    codes, dates, raw, signal = _s2_panel()
    bad_dates = list(dates)
    bad_dates[2] = bad_dates[1]
    with pytest.raises(ValueError, match="duplicate"):
        compare_candidate_parity(
            [("b42", 1, signal)], raw, codes, bad_dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
        )


def test_p53_b52_rebalance_mask_shape_rejected():
    # B5.2: a rebalance mask off the (n_dates,) global axis is refused.
    codes, dates, raw, signal = _s2_panel()
    with pytest.raises(ValueError, match="rebalance_mask shape"):
        compare_candidate_parity(
            [("b52", 1, signal)], raw, codes, dates,
            universe_mask=np.ones((2, 4), dtype=bool),
            backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
            rebalance_mask=np.ones(5, dtype=bool),
        )


# ---------------------------------------------------------------------------
# §1 F5 — blocked-buy asymmetry made explicit (B5.1)
# ---------------------------------------------------------------------------


def test_p53_b5_blocked_extra_buy_columns_quantified():
    codes, dates, raw, signal = _s2_panel()
    cfg = _cfg(initial_capital=100000.0, top_n=2, single_weight_cap=0.5)
    universe = np.ones((2, 4), dtype=bool)
    mask = np.array([True, False, False, False])
    # The engine's own matrix (computed exactly as backtest.py does).
    open_clean = np.nan_to_num(
        np.asarray(raw["open"], dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    engine_buy, _ = exact_tradability_matrix(
        open_clean, raw["up_limit"], raw["down_limit"], raw["suspended"],
        "buy",
    )
    # B5.1: an aligned basket mask -> zero extra columns.
    report = compare_candidate_parity(
        [("b5a", 1, signal)], raw, codes, dates,
        universe_mask=universe, backtest_config=cfg,
        rebalance_mask=mask, blocked_buy=engine_buy.copy(),
        local_signal_range=(0, 2),
    )
    assert report.rows[0].blocked_extra_buy_columns == 0
    # The field quantifies the asymmetry: one basket-only blocked cell on
    # one column -> exactly one counted column.
    extra = engine_buy.copy()
    extra[1, 1] = True
    report2 = compare_candidate_parity(
        [("b5b", 1, signal)], raw, codes, dates,
        universe_mask=universe, backtest_config=cfg,
        rebalance_mask=mask, blocked_buy=extra,
        local_signal_range=(0, 2),
    )
    assert report2.rows[0].blocked_extra_buy_columns == 1
    # No basket mask at all -> nothing to be asymmetric against.
    report3 = compare_candidate_parity(
        [("b5c", 1, signal)], raw, codes, dates,
        universe_mask=universe, backtest_config=cfg,
        rebalance_mask=mask, local_signal_range=(0, 2),
    )
    assert report3.rows[0].blocked_extra_buy_columns == 0


# ---------------------------------------------------------------------------
# §1 F6 — fixed direction semantics (B6.3)
# ---------------------------------------------------------------------------


def test_p53_b63_error_family_uses_the_candidate_given_direction():
    # B6.3: the row's error family runs the candidate's SUPPLIED
    # direction (±1 is a signal multiplier), never the diagnostic's
    # winning orientation.
    codes, dates, raw, signal = _s2_panel()
    cfg = _cfg(initial_capital=100000.0, top_n=2, single_weight_cap=0.5)
    universe = np.ones((2, 4), dtype=bool)
    mask = np.array([True, False, False, False])
    target = open_to_open_returns(raw["adjusted_open"])

    report = compare_candidate_parity(
        [("plus", 1, signal), ("minus", -1, signal)], raw, codes, dates,
        universe_mask=universe, backtest_config=cfg,
        rebalance_mask=mask, local_signal_range=(0, 2),
    )
    rows = {row.candidate: row for row in report.rows}
    assert rows["plus"].direction == 1
    assert rows["minus"].direction == -1
    for name, direction in (("plus", 1), ("minus", -1)):
        sim = simulate_basket_daily_returns(
            float(direction) * signal, target, cfg,
            universe_mask=universe, tie_break_keys=np.asarray(codes),
            rebalance_mask=mask, signal_range=(0, 2),
        )
        total = 1.0
        for value in sim.daily_net_returns:
            total *= 1.0 + float(value)
        assert rows[name].proxy_total_net == pytest.approx(
            total - 1.0, abs=1e-12
        )


# ---------------------------------------------------------------------------
# §4 — naming contract: disclosure present, forbidden phrasings absent
# ---------------------------------------------------------------------------


def test_p53_s42_verdict_carries_disclosure_and_no_forbidden_phrasing():
    codes, dates, raw, signal = _s2_panel()
    report = compare_candidate_parity(
        [("naming", 1, signal)], raw, codes, dates,
        universe_mask=np.ones((2, 4), dtype=bool),
        backtest_config=_cfg(top_n=2, single_weight_cap=0.5),
        rebalance_mask=np.array([True, False, False, False]),
        local_signal_range=(0, 2),
    )
    verdict = proxy_reliability_verdict(report)
    dumped = json.dumps(verdict, ensure_ascii=False)
    # §4.2: any output surfacing proxy/engine directions carries the
    # frozen disclosure string verbatim.
    assert DISCLOSURE in dumped
    # §4.3: none of the forbidden phrasings appears.
    for phrase in FORBIDDEN:
        assert phrase not in dumped


_WP16B_DIR = Path(__file__).resolve().parents[1] / "reports" / "wp16b"
_RERUN_JSONS = sorted(_WP16B_DIR.glob("parity_report_*.json")) if _WP16B_DIR.exists() else []
# Audit fix (2026-09-23, major; FROZEN_B §4.2 "凡出现
# proxy_direction/engine_direction/direction_flips 的输出（JSON、log、
# 报告），必须携带 [披露串]" and §5 red line 4 "对重跑输出文本"): the
# assertion surface is EVERY rerun output text under reports/wp16b/ — the
# JSON reports, comparison.json and the run_console.txt log — not only the
# parity JSONs (the console names direction values and direction_flips in
# its per-candidate rows and pass summaries).
_RERUN_TEXTS = (
    sorted(
        [
            *_WP16B_DIR.glob("parity_report_*.json"),
            *_WP16B_DIR.glob("comparison.json"),
            *_WP16B_DIR.glob("*.txt"),
        ]
    )
    if _WP16B_DIR.exists()
    else []
)


@pytest.mark.skipif(
    not _RERUN_JSONS,
    reason="wp16b panel rerun artifacts not present (run reports/wp16b/run_measurement.py)",
)
def test_p53_s42_rerun_output_text_honors_the_naming_contract():
    # §5 red line 4: the panel rerun outputs carry the disclosure and
    # never a forbidden phrasing — on every output surface, the JSON
    # reports AND the console/log text.
    assert _RERUN_TEXTS, "wp16b rerun outputs missing (run_measurement.py)"
    for path in _RERUN_TEXTS:
        text = path.read_text(encoding="utf-8")
        assert DISCLOSURE in text, path
        for phrase in FORBIDDEN:
            assert phrase not in text, (path, phrase)
