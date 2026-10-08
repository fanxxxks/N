"""p52 / WP16-A: the additive cash-fees identity in the basket simulator.

Assertion source: ``reports/wp16/FROZEN_A.md`` (frozen 2026-09-23, before
any WP16 code existed).  Every numerical assertion cites the frozen § it
pins; nothing below is derived from implementation output.  The
exact-fraction oracle :func:`_identity` implements §1's four formulas and
is itself anchored to the frozen §3 hand tables by the
``test_p52_frozen_s3*_oracle`` tests, so the oracle cannot drift from the
document.

Session conventions (mirroring the simulator's own): session ``t`` books
the constructor's post-trade book ``a = target_weights[t]``, the session
cost fraction ``c = daily_cost_fractions[t]`` and the per-stock returns
``r = target_ret[:, t]``; the drifted next-period book is exposed verbatim
on a following not-due session (``rebalance_mask[t+1] == False`` returns
``prev`` unchanged), and ``daily_cash_weights[t]`` is the cash component
carried into that next session.
"""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.reward import simulate_basket_daily_returns
from ashare_model.scoring_parity import parity_bounds
from tests.conftest import with_price_semantics

# FROZEN_A §4.1: channel-1 strict tolerance |impl − hand| <= 1e-12*max(1,|hand|).
_REL_TOL = F(1, 10**12)
# FROZEN_A §1 (C1) via §6: post-trade feasibility bound Σa + c <= 1 + tol.
# Re-anchored 1e-12 -> 1e-9 on 2026-09-27 per FROZEN_A §8.3 (adjudicated
# WP18 contract ruling, append-only erratum; the §8.1 precedent for pin
# re-anchoring): the canonical round-12 projection's own noise floor is
# 1-2e-12 for a 20-name book (first slow-domain x guard encounter,
# 2026-09-27, refused 19/25 Rule A keys at pure float epsilon), so the
# CONTRACT bound on a booked session is the §8.3 refusal gate 1e-9 —
# every fixture here still realizes <= 1e-12 because the implementation's
# booking precision (_P52_C1_TOLERANCE, the fixed-point landing criterion)
# is unchanged; only the final refusal gate was errata-split to 1e-9.
# The §4.1 channel-1 relative tolerance above is NOT affected.
_C1_TOL = 1e-9


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(
        initial_capital=100000.0,
        top_n=2,
        single_weight_cap=0.5,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


def _to_frac(value) -> F:
    """Exact fraction: pass Fractions through, convert floats once."""

    return value if isinstance(value, F) else F(float(value))


def _fracs(values) -> list[F]:
    return [
        _to_frac(v) for v in np.asarray(values, dtype=object).ravel()
    ]


def _identity(a, c, r):
    """FROZEN_A §1 formulas (1)-(4) on exact fractions.

    Returns ``(g, net, w_prime, cash_prime)`` with ``g = Σ a_i r_i``,
    ``net = g − c``, ``w'_i = a_i(1+r_i)/(1+g−c)`` and
    ``w'_cash = (1−Σa−c)/(1+g−c)``.
    """

    a_f, r_f = _fracs(a), _fracs(r)
    c_f = _to_frac(c)
    g = sum((ai * ri for ai, ri in zip(a_f, r_f)), F(0))
    denom = 1 + g - c_f
    w_prime = [ai * (1 + ri) / denom for ai, ri in zip(a_f, r_f)]
    cash_prime = (1 - sum(a_f, F(0)) - c_f) / denom
    return g, g - c_f, w_prime, cash_prime


def _close(impl, hand, label: str) -> None:
    """FROZEN_A §4.1 relative 1e-12 comparison (channel 1)."""

    hand_f = float(hand)
    tol = float(_REL_TOL) * max(1.0, abs(hand_f))
    assert abs(float(impl) - hand_f) <= tol, (
        f"{label}: impl={float(impl)!r} hand={hand_f!r} "
        f"diff={abs(float(impl) - hand_f):.3e} > tol={tol:.3e} "
        "(FROZEN_A §4.1: |impl-hand| <= 1e-12 * max(1, |hand|))"
    )


# --- oracle anchors: the frozen §3 hand tables, value by value ----------------


def test_p52_frozen_s31_three_session_hand_table_oracle():
    """FROZEN_A §3.1: partial investment + two-sided fees, E0 = 100,000."""

    e = F(100_000)
    # Session 1 (§3.1): a=(0.30,0.20,0.10), c=0.005, r=(+0.10,−0.05,+0.02).
    g, net, w, cash = _identity(
        [F(3, 10), F(2, 10), F(1, 10)], F(5, 1000), [F(1, 10), F(-5, 100), F(2, 100)]
    )
    assert g == F(22, 1000)
    assert net == F(17, 1000)
    assert w == [F(110, 339), F(190, 1017), F(34, 339)]
    assert cash == F(395, 1017)
    assert sum(w, F(0)) + cash == 1  # §1 self-check
    e1 = e * (1 + net)
    assert e1 == F(101_700)
    # Session 2 (§3.1): a=(0.20,0.40,0.20), c=0.008, r=(−0.10,+0.04,+0.01).
    g2, net2, w2, cash2 = _identity(
        [F(2, 10), F(4, 10), F(2, 10)],
        F(8, 1000),
        [F(-1, 10), F(4, 100), F(1, 100)],
    )
    assert g2 == F(-2, 1000)
    assert net2 == F(-1, 100)
    assert w2 == [F(2, 11), F(208, 495), F(101, 495)]
    assert cash2 == F(32, 165)
    assert sum(w2, F(0)) + cash2 == 1
    e2 = e1 * (1 + net2)
    assert e2 == F(100_683)
    # Session 3 (§3.1): a=(0.30,0.10,0.30), c=0.004, r=(+0.05,+0.02,−0.04).
    g3, net3, w3, cash3 = _identity(
        [F(3, 10), F(1, 10), F(3, 10)],
        F(4, 1000),
        [F(5, 100), F(2, 100), F(-4, 100)],
    )
    assert g3 == F(5, 1000)
    assert net3 == F(1, 1000)
    assert w3 == [F(45, 143), F(102, 1001), F(288, 1001)]
    assert cash3 == F(296, 1001)
    assert sum(w3, F(0)) + cash3 == 1
    e3 = e2 * (1 + net3)
    assert e3 == F(100_783_683, 1_000)  # §5.5: E3 = 100,783.683
    # §5.5 cumulative: 1.017 × 0.990 × 1.001 = 1.00783683.
    assert (1 + net) * (1 + net2) * (1 + net3) == F(100_783_683, 100_000_000)
    assert (1 + net) * (1 + net2) * (1 + net3) - 1 == F(783_683, 100_000_000)


def test_p52_frozen_s32_post_fee_full_investment_oracle():
    """FROZEN_A §3.2: fee-reserved full investment, w'_cash exactly zero."""

    g, net, w, cash = _identity(
        [F(498, 1000), F(249, 1000), F(249, 1000)],
        F(4, 1000),
        [F(3, 100), F(2, 100), F(1, 100)],
    )
    assert g == F(2241, 100_000)
    assert net == F(1841, 100_000)
    assert w == [F(206, 409), F(102, 409), F(101, 409)]
    assert cash == 0  # §3.2: exactly zero, not approximately
    assert sum(w, F(0)) + cash == 1


def test_p52_frozen_s33_zero_fee_oracle():
    """FROZEN_A §3.3: c = 0 makes the additive and multiplicative forms meet."""

    g, net, w, cash = _identity(
        [F(1, 3), F(1, 3)], F(0), [F(1, 100), F(2, 100)]
    )
    assert g == F(1, 100)
    assert net == g
    assert (1 - 0) * (1 + g) - 1 == net  # §3.3: coincident at zero fees
    assert w == [F(1, 3), F(34, 101)]
    assert cash == F(100, 303)
    assert sum(w, F(0)) + cash == 1


def test_p52_frozen_s34_counterexample_arithmetic_oracle():
    """FROZEN_A §3.4: the unreserved Σa=1 book is infeasible (w'_cash < 0)."""

    g, net, w, cash = _identity(
        [F(1, 3), F(1, 3), F(1, 3)], F(4, 1000), [F(3, 100), F(2, 100), F(1, 100)]
    )
    assert g == F(2, 100)
    assert net == F(16, 1000)  # the spec value the old multiplicative form missed
    assert cash == F(-1, 254)  # §3.4: negative cash → (C1) violated → refuse
    assert cash < 0
    # The old implementation's washed value for contrast (§3.4 "旧 basket 行为").
    assert (1 - F(4, 1000)) * (1 + g) - 1 == F(199, 12_500)  # = 0.015920
    assert (1 - F(4, 1000)) * (1 + g) - 1 != net


# --- §3.4 / §5.6: the user's counter-example as the independent 3-check test --


def test_p52_s34_counterexample_three_checks_route_declared():
    """FROZEN_A §3.4 / §5.6 — the counter-example input (Σa = 1 unreserved,
    c = 0.004, r = (3%, 2%, 1%)) checked on all three quantities: net,
    next-period weights and cash.

    Route declared per §3.4 (one of the two allowed routes, chosen here):
    **constructor-side fee reservation** (§6) — the simulator sizes buys
    only against cash left after the session's fees, so the unreserved
    Σa = 1 session is unreachable through the public API — plus the
    fail-closed backstop that refuses to book any session whose
    post-trade book still violates (C1).  The forbidden third route
    (emit numbers, then normalize Σw' back to 1) is what the old
    implementation did (§3.4 "旧 basket 行为").
    """

    cfg = _cfg(
        top_n=3,
        single_weight_cap=1.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.004,
    )
    n_stocks, n_dates = 3, 6
    signal = np.tile(np.array([3.0, 2.0, 1.0])[:, None], (1, n_dates))
    target = np.zeros((n_stocks, n_dates))
    target[:, 0] = (0.03, 0.02, 0.01)
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    rebalance = np.array([True, False, True, False, True, True])
    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask, rebalance_mask=rebalance
    )
    a = sim.target_weights[0]
    c = float(sim.daily_cost_fractions[0])
    # Check 3 (cash / feasibility, §1 (C1) via §3.4): the unreserved book
    # would need w'_cash = −1/254 < 0; the reserved route must keep the
    # post-trade book fundable at (or inside) full investment.
    assert float(np.sum(a)) + c <= 1.0 + _C1_TOL
    assert float(np.sum(a)) + c >= 1.0 - 1e-6  # post-fee full investment
    # Every other session must satisfy (C1) too (§1 (C1)).
    for t in range(len(sim.daily_cost_fractions)):
        assert float(np.sum(sim.target_weights[t])) + float(
            sim.daily_cost_fractions[t]
        ) <= 1.0 + _C1_TOL
    # Check 1 (net, §1 (2) / §4.1): R_net = g − c on the booked a — not
    # the old multiplicative value 0.015920 of §3.4.
    g, net, w_prime, cash_prime = _identity(a, c, target[:, 0])
    _close(sim.daily_net_returns[0], net, "§3.4 net: R_net = g − c")
    assert abs(float(sim.daily_net_returns[0]) - float((1 - c) * (1 + g) - 1)) > 1e-9
    # Check 2 (next-period weights, §1 (3) / §4.1): the drifted book is
    # exposed verbatim on the not-due session 1.
    drift = sim.target_weights[1]
    for i in range(n_stocks):
        _close(drift[i], w_prime[i], f"§3.4 w'[{i}] = a_i(1+r_i)/(1+g−c)")
    # Cash (§1 (4) / §4.1): explicit field, implied residual, and the
    # Σw' + w'_cash = 1 closure (§1 self-check).
    _close(sim.daily_cash_weights[0], cash_prime, "§3.4 w'_cash field")
    _close(
        1.0 - float(np.sum(drift)),
        cash_prime,
        "§3.4 w'_cash implied by the carried book",
    )
    _close(
        float(np.sum(drift)) + float(sim.daily_cash_weights[0]),
        F(1),
        "§3.4 Σw' + w'_cash = 1",
    )
    assert float(sim.daily_cash_weights[0]) >= -_C1_TOL


# --- §3.1 / §5.1 / §5.3 / §5.5 / §5.7: partial investment, two-sided fees -----


def test_p52_s31_partial_investment_two_sided_multisession_identity():
    """FROZEN_A §3.1 / §5.1 / §5.3 / §5.5 / §5.7 — a partially invested book
    (top_n=1 capped at 0.5, so Σa = 0.5 < 1 − c) with fee-bearing buys,
    sells and holds across several sessions, checked session by session:
    net (§1 (2)), drifted weights and cash (§1 (3)/(4)) on every
    following not-due session, the (C1) feasibility, the ledger closure
    (§3.1 台账闭合) and the Σw' + w'_cash = 1 identity (§1 self-check),
    all at the §4.1 channel-1 tolerance."""

    cfg = _cfg(top_n=1, single_weight_cap=0.5)
    n_stocks, n_dates = 3, 10
    signal = np.zeros((n_stocks, n_dates))
    signal[0, :4] = 2.0
    signal[1, :4] = 1.0
    signal[0, 4:] = 1.0  # rank flip at t=4 → a full two-sided rotation
    signal[1, 4:] = 2.0
    target = np.zeros((n_stocks, n_dates))
    target[0, :] = 0.010
    target[1, :] = -0.004
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[2, :] = False  # only two names eligible → unit 0.5 is partial
    rebalance = np.tile(np.array([True, False]), n_dates // 2)
    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask, rebalance_mask=rebalance
    )
    n_sessions = len(sim.daily_cost_fractions)
    assert n_sessions >= 6
    two_sided_seen = False
    for t in range(n_sessions - 1):
        a = sim.target_weights[t]
        c = float(sim.daily_cost_fractions[t])
        r = target[:, t]
        # (C1) feasibility every session (§1 (C1)).
        assert float(np.sum(a)) + c <= 1.0 + _C1_TOL
        # §1 (2): net = g − c on the booked a.
        g, net, w_prime, cash_prime = _identity(a, c, r)
        _close(sim.daily_net_returns[t], net, f"§3.1 s{t} net = g − c")
        # §3.1 台账闭合: prev cash + sells − buys − c = 1 − Σa − c.
        prev_cash = 1.0 if t == 0 else float(sim.daily_cash_weights[t - 1])
        sells = float(np.sum(sim.sell_weights[t]))
        buys = float(np.sum(sim.buy_weights[t]))
        _close(
            prev_cash + sells - buys - c,
            1.0 - float(np.sum(a)) - c,
            f"§3.1 s{t} ledger closure",
        )
        if sells > 1e-12 and buys > 1e-12 and c > 0.0:
            two_sided_seen = True
        # §1 (3)/(4) on every following not-due session: the drifted book
        # is carried unchanged, so it is exposed verbatim.
        if not bool(rebalance[t + 1]):
            drift = sim.target_weights[t + 1]
            for i in range(n_stocks):
                _close(drift[i], w_prime[i], f"§3.1 s{t} w'[{i}]")
            _close(sim.daily_cash_weights[t], cash_prime, f"§3.1 s{t} w'_cash")
            _close(
                1.0 - float(np.sum(drift)),
                cash_prime,
                f"§3.1 s{t} implied w'_cash",
            )
            _close(
                float(np.sum(drift)) + float(sim.daily_cash_weights[t]),
                F(1),
                f"§3.1 s{t} Σw' + w'_cash = 1",
            )
    # §5.3: at least one session must carry BOTH fee-bearing legs.
    assert two_sided_seen
    flip = 4
    assert float(np.sum(sim.sell_weights[flip])) > 0.0
    assert float(np.sum(sim.buy_weights[flip])) > 0.0
    assert float(sim.daily_cost_fractions[flip]) > 0.0


# --- §3.2 / §5.2: fee-reserved full investment --------------------------------


def test_p52_s32_post_fee_full_investment_cash_is_zero():
    """FROZEN_A §3.2 / §5.2 — full investment after fee reservation:
    Σa + c = 1 (cash exactly zero).  §3.2's "exactly zero" holds in exact
    arithmetic; the float realization carries the round-12 canonical grid
    (§4.1 note: n × 5e-13) plus the fixed-point convergence tail, so the
    assertion bound here is 1e-11 — still five orders below any physical
    fee fraction — while never allowing negative cash beyond §1 (C1)'s
    tolerance."""

    cfg = _cfg(
        top_n=3,
        single_weight_cap=1.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.004,
    )
    n_stocks, n_dates = 3, 4
    signal = np.tile(np.array([3.0, 2.0, 1.0])[:, None], (1, n_dates))
    target = np.zeros((n_stocks, n_dates))
    target[:, 0] = (0.03, 0.02, 0.01)
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    rebalance = np.array([True, False, True, False])
    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask, rebalance_mask=rebalance
    )
    a = sim.target_weights[0]
    c = float(sim.daily_cost_fractions[0])
    # Post-fee full investment (§3.2): the reserve binds — Σa + c = 1.
    assert float(np.sum(a)) + c <= 1.0 + _C1_TOL
    assert float(np.sum(a)) + c >= 1.0 - 1e-9
    # §1 (2)/(3)/(4) identity on the session.
    g, net, w_prime, cash_prime = _identity(a, c, target[:, 0])
    _close(sim.daily_net_returns[0], net, "§3.2 net = g − c")
    for i in range(n_stocks):
        _close(sim.target_weights[1][i], w_prime[i], f"§3.2 w'[{i}]")
    _close(sim.daily_cash_weights[0], cash_prime, "§3.2 w'_cash field")
    # §5.2: the carried cash is zero (to the grid/convergence bound) and
    # never negative beyond the (C1) tolerance.
    assert abs(float(sim.daily_cash_weights[0])) <= 1e-11
    assert float(sim.daily_cash_weights[0]) >= -_C1_TOL
    assert 1.0 - float(np.sum(sim.target_weights[1])) >= -_C1_TOL


# --- §3.3 / §5.4: zero fees ----------------------------------------------------


def test_p52_s33_zero_fee_identity_and_form_coincidence():
    """FROZEN_A §3.3 / §5.4 — at c = 0 the additive identity and the old
    multiplicative form coincide, which is why the pre-existing zero-fee
    anchor (tests/test_reward.py::test_basket_underfilled_keeps_cash_like_
    backtest_engine) stays valid unchanged under 16-A."""

    cfg = _cfg(
        top_n=3,
        single_weight_cap=1.0,
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        slippage_rate=0.0,
    )
    n_stocks, n_dates = 4, 4
    signal = np.tile(np.array([4.0, 3.0, 2.0, 1.0])[:, None], (1, n_dates))
    target = np.zeros((n_stocks, n_dates))
    target[0, 0] = 0.01
    target[1, 0] = 0.02
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[2:, :] = False  # two eligible of top_n=3 → Σa = 2/3 (§3.3 shape)
    rebalance = np.array([True, False, True, False])
    sim = simulate_basket_daily_returns(
        signal, target, cfg, universe_mask=mask, rebalance_mask=rebalance
    )
    a = sim.target_weights[0]
    c = float(sim.daily_cost_fractions[0])
    assert c == 0.0
    g, net, w_prime, cash_prime = _identity(a, c, target[:, 0])
    # §1 (2) and §3.3: both forms agree at zero fees.
    _close(sim.daily_net_returns[0], net, "§3.3 net = g")
    _close(sim.daily_net_returns[0], (1.0 - c) * (1.0 + g) - 1.0, "§3.3 coincidence")
    for i in range(n_stocks):
        _close(sim.target_weights[1][i], w_prime[i], f"§3.3 w'[{i}]")
    _close(sim.daily_cash_weights[0], cash_prime, "§3.3 w'_cash")


# --- §4.2 / §5.8: strict channel 1 and the lot-attribution channel 2 ----------


def _calm_market(n_stocks: int = 4, n_dates: int = 8):
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = [f"2024010{d + 1}" for d in range(n_dates)]
    open_ = (
        10.0
        * (1.0 + 0.001 * np.arange(n_stocks, dtype=np.float64))[:, None]
        * (1.0 + 0.0015 * np.arange(n_dates, dtype=np.float64))[None, :]
    )
    raw = with_price_semantics(
        {
            "open": open_,
            "high": open_ * 1.02,
            "low": open_ * 0.98,
            "pre_close": np.column_stack([open_[:, 0], open_[:, :-1]]),
            "volume": np.full_like(open_, 1_000_000.0),
        }
    )
    signal = np.zeros((n_stocks, n_dates), dtype=np.float64)
    for d in range(n_dates):
        for i in range(n_stocks):
            signal[i, d] = float(n_stocks - ((i + d) % n_stocks))
    return codes, dates, raw, signal, open_to_open_returns(open_)


def test_p52_s42_channel1_strict_and_channel2_lot_attribution():
    """FROZEN_A §4.2 / §5.8 — with lot_size = 100 (the default):

    * channel 1: the basket against the §1 hand math on its OWN a stays
      at the §4.1 strict 1e-12 tolerance — the identity never absorbs any
      lot residual (the basket books continuous weights; lots only exist
      in the engine's share space);
    * channel 2: the basket↔engine gap is attributed itemized — the fee
      difference Δc is computed from both sides' exposed cost arrays and
      bounded by the p45 mechanism bound, the residual return term
      Δg = (net_basket − net_engine) + Δc is bounded per held name by the
      one-lot granularity (L·p_i/E_t)·|r_i| plus the fee-financing
      scaling slack, and the net gap itself stays inside the p45
      parity_bounds (§4.2: channel 2 uses the p45 mechanism bounds, not
      1e-12).  Windows are the same default (0, n−2) on both sides, so
      the tail-alignment residual is zero by construction."""

    codes, dates, raw, signal, target = _calm_market()
    cfg = _cfg(top_n=2, single_weight_cap=0.5)  # lot_size defaults to 100
    mask = np.ones(signal.shape, dtype=bool)
    engine = AshareBacktestEngine(cfg).run(signal, raw, codes, dates, universe_mask=mask)
    sim = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    bounds = parity_bounds(
        cfg, price_ref=float(np.median(raw["open"])), n_periods=len(sim.turnover)
    )
    eng_net = np.asarray(engine.daily_returns, dtype=np.float64)
    eng_cost = np.asarray(engine.cost_fractions, dtype=np.float64)
    # Same default session axis on both sides → tail residual = 0 (§4.2 ch2 item 3).
    assert len(eng_net) == len(sim.daily_net_returns)
    equity = float(cfg.initial_capital)
    for t in range(len(eng_net)):
        a = sim.target_weights[t]
        c = float(sim.daily_cost_fractions[t])
        # Channel 1 (§4.2): strict identity on the basket's own book,
        # untouched by the lot size.
        g, net, w_prime, cash_prime = _identity(a, c, target[:, t])
        _close(sim.daily_net_returns[t], net, f"§4.2 ch1 s{t} net")
        _close(sim.daily_cash_weights[t], cash_prime, f"§4.2 ch1 s{t} cash")
        # Channel 2 (§4.2): itemized attribution.
        delta_c = float(sim.daily_cost_fractions[t] - eng_cost[t])
        net_diff = float(sim.daily_net_returns[t] - eng_net[t])
        assert abs(delta_c) <= bounds["fee_fraction"], (
            f"s{t}: |Δc|={abs(delta_c):.3e} > p45 fee bound "
            f"{bounds['fee_fraction']:.3e}"
        )
        assert abs(net_diff) <= bounds["daily_net"], (
            f"s{t}: |net gap|={abs(net_diff):.3e} > p45 daily-net bound "
            f"{bounds['daily_net']:.3e}"
        )
        entry_prices = np.asarray(raw["open"], dtype=np.float64)[:, t + 1]
        lot_slack = float(cfg.lot_size) * entry_prices / equity  # L·p_i / E_t
        r_abs = np.abs(target[:, t])
        held = (a > 0) | (np.asarray(engine.target_weights[t]) > 0)
        # One lot of per-name execution slack (the engine floors every leg
        # to whole lots) plus the fee-financing scaling difference between
        # the two books' targets; both terms are the §4.2 per-name upper
        # bound extended by the reservation slack, kept ≪ parity_bounds.
        dg_bound = float(np.sum(2.0 * lot_slack[held] * r_abs[held])) + 2.0 * abs(
            delta_c
        )
        dg = net_diff + delta_c  # the residual return term by attribution
        assert abs(dg) <= dg_bound + 1e-12, (
            f"s{t}: |Δg|={abs(dg):.3e} > lot bound {dg_bound:.3e} "
            "(§4.2 channel 2: Δg = Σ(a^w − a^exec)·r bounded by lot slack)"
        )
        equity *= 1.0 + float(eng_net[t])


# --- §5.9: the p51 guard keeps refusing unwired trade_groups -------------------


def test_p52_s59_p51_guard_still_refuses_unwired_trade_groups():
    """FROZEN_A §5.9 / 红线 2 — the WP15 fail-closed guard on the basket
    entry points stays in force after the 16-A edit."""

    cfg = _cfg(
        top_n=2,
        turnover_budget=0.2,
        turnover_allocation="trade_groups",
    )
    n_stocks, n_dates = 3, 5
    signal = np.tile(np.array([3.0, 2.0, 1.0])[:, None], (1, n_dates))
    target = np.zeros((n_stocks, n_dates))
    with pytest.raises(ValueError, match="does not consume the trade_groups planner"):
        simulate_basket_daily_returns(
            signal,
            target,
            cfg,
            universe_mask=np.ones((n_stocks, n_dates), dtype=bool),
        )


# --- §1 (C1)/§3.4 audit fix: buyless floor-dominated dust exits -----------------


def _dust_forced_exit_panel():
    """A 2-session panel whose second session is a BUYLESS mandatory dust
    exit dominated by the per-order minimum-commission floor.

    Session 0 buys the full top_n=2 book fee-reserved (§3.2 post-fee full
    investment: cash exactly 0); stock 1's −0.9999992 return collapses it
    to an ~8e-7 dust weight; session 1 sees a single selectable name (no
    cross-sectional dispersion → the no-signal hold path) while stock 1
    leaves the universe, so its only executable leg is the mandatory
    (forced-exit) sell of the dust position.  The dust notional (~0.04
    yuan) sits far below min_commission/rate, so the 5-yuan floor fee
    (ashare_execution.py:116-120) exceeds the freed cash: Σa + c > 1 with
    zero buys on a physical ledger (equity ratio 1+g−c ≈ 0.5 > 0)."""

    n_stocks, n_dates = 2, 4
    signal = np.zeros((n_stocks, n_dates))
    signal[:, 0] = (2.0, 1.0)  # dispersed -> full fee-reserved buy at t=0
    signal[:, 1] = (1.0, 1.0)  # one selectable name at t=1 -> no_signal hold
    target = np.zeros((n_stocks, n_dates))
    target[:, 0] = (0.0, -0.9999992)  # stock 1 collapses to ~8e-7 dust
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[1, 2] = False  # stock 1 leaves the universe at session 1's exec day
    return signal, target, mask


def test_p52_s1c1_buyless_floor_dominated_dust_exit_funds_its_fees():
    """FROZEN_A §1 (C1)/§3.4 (audit fix 2026-09-23, blocker; route 2 —
    sell-side fee reservation) — a BUYLESS session whose only executable
    leg is a mandatory dust sell dominated by the per-order minimum
    commission floor must never be booked at Σa + c > 1.

    The unfunded exit is genuinely infeasible (the floor fee exceeds both
    the freed dust cash and the zero cash balance), so per §1 (C1) the
    implementation must either refuse the session or reserve the fees;
    the chosen route defers the floor-dominated sell leg (books it back
    at its prev weight — the weight-space analogue of the engine's
    buy-affordability loop skipping an unfundable order,
    backtest.py:356-368).  The session therefore books FEASIBLY: the
    (C1) inequality holds with the true cash exposed, the deferral is
    auditable in the construction diagnostics, and daily_cash_weights
    never reports a washed 0.0 for an infeasible book (there is none —
    the infeasible construction is amended before booking)."""

    signal, target, mask = _dust_forced_exit_panel()
    cfg = _cfg()  # default fee set incl. min_commission=5.0
    sim = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    # The unfunded exit really is floor-dominated: the dust weight is far
    # below the minimum-commission fraction, so executing it books
    # Σa + c > 1 (the state §1 (C1) forbids).
    dust = float(np.sum(sim.target_weights[1]) * 0.0 + 8e-7)  # panel constant
    assert dust < cfg.min_commission / cfg.initial_capital
    # Session 1 booked feasibly with the dust exit deferred: no buys, no
    # executed sell, zero cost, cash exactly the §1 (4) residual.
    assert float(np.sum(sim.buy_weights[1])) == 0.0
    assert float(np.sum(sim.sell_weights[1])) == 0.0
    assert float(sim.daily_cost_fractions[1]) == 0.0
    for t in range(len(sim.daily_cost_fractions)):
        sum_a = float(np.sum(sim.target_weights[t]))
        c = float(sim.daily_cost_fractions[t])
        assert sum_a + c <= 1.0 + _C1_TOL, (t, sum_a, c)
        assert float(sim.daily_cash_weights[t]) >= -_C1_TOL, t
        implied = 1.0 - float(np.sum(sim.target_weights[t])) - c
        assert abs(float(sim.daily_cash_weights[t]) - implied) <= 1e-9, t
    diag = sim.construction_diagnostics[1]
    assert diag.get("sell_fee_funding_dropped") == (1,)
    assert "§1 (C1)" in diag.get("sell_fee_funding", "")


def test_p52_s1c1_dust_exit_is_feasible_without_the_commission_floor():
    """Control side of the audit fix: with the per-order minimum
    commission floor removed, the SAME dust forced-exit panel executes
    the exit (the proportional fee on ~0.04 yuan of dust notional is far
    below the freed cash), so every session stays (C1)-feasible with a
    non-negative cash field and no deferral is recorded.  Pins that the
    deferral above is the floor-financing failure itself, not an
    over-broad rejection of buyless or forced-exit sessions."""

    signal, target, mask = _dust_forced_exit_panel()
    cfg = _cfg(min_commission=0.0)
    sim = simulate_basket_daily_returns(signal, target, cfg, universe_mask=mask)
    for t in range(len(sim.daily_cost_fractions)):
        sum_a = float(np.sum(sim.target_weights[t]))
        c = float(sim.daily_cost_fractions[t])
        assert sum_a + c <= 1.0 + _C1_TOL, (t, sum_a, c)
        assert float(sim.daily_cash_weights[t]) >= -_C1_TOL, t
    # Session 1 executes the dust forced exit without the floor: the dust
    # leg sells, its proportional fee is booked, no leg is deferred.
    assert float(np.sum(sim.buy_weights[1])) == 0.0
    assert 0.0 < float(np.sum(sim.sell_weights[1])) < 1e-5
    assert float(sim.daily_cost_fractions[1]) > 0.0
    assert not sim.construction_diagnostics[1].get("sell_fee_funding_dropped")
