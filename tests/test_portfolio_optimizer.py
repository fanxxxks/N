"""T3-01 contracts: constrained portfolio optimizer (portfolio layer).

The optimizer consumes the factor layer's expected-alpha output and owns
the portfolio constraints: single-name and industry caps, Beta/size
exposure ranges, ADV participation, turnover budget, min cash, position
count and min trade amount.  The QP core is CVXPY/OSQP — no hand-written
solver.  The two non-convex constraints (``max_positions``,
``min_trade_amount``) are enforced by a documented deterministic
projection after the QP; every convex constraint is exact inside the QP.

Invariants asserted here:

* QP solution is long-only, within budget (``sum(w) <= 1 - min_cash``),
  respects the single-name cap and every supplied convex constraint
  (industry / Beta / size / ADV participation / turnover budget).
* Infeasible problems raise ``PortfolioOptimizationError``; empty
  universes return an all-zero book without calling the solver.
* The projection only shrinks weights (never renormalized upward, the
  repo-wide T1-02 contract), so caps / industry / budget / ADV / turnover
  stay satisfied; Beta/size exposure after projection is reported in
  ``diagnostics`` as the residual.
* ``min_trade_amount`` filters trades (buys and sells) below the amount;
  a held name whose reduction falls below it keeps its previous weight.
* ``max_positions`` is a hard ceiling on ``count(w > 0)``; liquidation
  sells forced by the ceiling are exempt from the min-trade filter.
* Determinism: identical inputs produce bitwise-identical weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_portfolio.optimizer import (
    PORTFOLIO_OPTIMIZER_VERSION,
    PortfolioConstraints,
    PortfolioObjective,
    PortfolioOptimizationError,
    PortfolioOptimizer,
)


def _optimizer(**kwargs) -> PortfolioOptimizer:
    defaults = dict(
        single_weight_cap=0.5,
        industry_cap=None,
        beta_range=None,
        size_range=None,
        adv_participation=None,
        turnover_budget=None,
        min_cash=0.0,
        max_positions=None,
        min_trade_amount=None,
    )
    defaults.update(kwargs)
    return PortfolioOptimizer(
        PortfolioConstraints(**defaults),
        PortfolioObjective(risk_aversion=0.0, turnover_cost=0.0, impact_cost=0.0),
    )


def _alpha(n: int = 4, base: float = 1.0, step: float = -0.1) -> np.ndarray:
    return np.asarray([base + step * i for i in range(n)], dtype=np.float64)


def _diag_cov(n: int, var: float = 0.01) -> np.ndarray:
    return np.eye(n) * var


# --- structure and version ------------------------------------------------


def test_optimizer_version_is_versioned():
    assert PORTFOLIO_OPTIMIZER_VERSION == 1


def test_solution_reports_status_objective_and_diagnostics():
    sol = _optimizer(single_weight_cap=0.5).solve(_alpha(), np.zeros(4))
    assert sol.status == "optimal"
    assert isinstance(sol.objective_value, float)
    assert isinstance(sol.diagnostics, dict)
    assert sol.weights.shape == (4,)


# --- validation (fail fast) ------------------------------------------------


def test_negative_risk_aversion_rejected():
    with pytest.raises(ValueError, match="risk_aversion"):
        PortfolioOptimizer(
            PortfolioConstraints(),
            PortfolioObjective(risk_aversion=-1.0),
        )


def test_negative_turnover_cost_rejected():
    with pytest.raises(ValueError, match="turnover_cost"):
        PortfolioOptimizer(
            PortfolioConstraints(),
            PortfolioObjective(turnover_cost=-0.1),
        )


def test_negative_impact_cost_rejected():
    with pytest.raises(ValueError, match="impact_cost"):
        PortfolioOptimizer(
            PortfolioConstraints(),
            PortfolioObjective(impact_cost=-1.0),
        )


def test_single_weight_cap_must_be_positive_and_at_most_one():
    with pytest.raises(ValueError, match="single_weight_cap"):
        _optimizer(single_weight_cap=0.0)
    with pytest.raises(ValueError, match="single_weight_cap"):
        _optimizer(single_weight_cap=1.5)


def test_industry_cap_requires_industries():
    opt = _optimizer(industry_cap=0.3)
    with pytest.raises(ValueError, match="industries"):
        opt.solve(_alpha(), np.zeros(4))


def test_beta_range_requires_beta_and_ordered_bounds():
    opt = _optimizer(beta_range=(0.8, 1.2))
    with pytest.raises(ValueError, match="beta"):
        opt.solve(_alpha(), np.zeros(4))
    with pytest.raises(ValueError, match="beta"):
        _optimizer(beta_range=(1.2, 0.8))
    with pytest.raises(ValueError, match="beta"):
        _optimizer(beta_range=(0.8, 1.2)).solve(
            _alpha(), np.zeros(4), beta=np.ones(3)
        )


def test_size_range_requires_size_and_ordered_bounds():
    opt = _optimizer(size_range=(-0.5, 0.5))
    with pytest.raises(ValueError, match="size"):
        opt.solve(_alpha(), np.zeros(4))
    with pytest.raises(ValueError, match="size"):
        _optimizer(size_range=(0.5, -0.5))
    with pytest.raises(ValueError, match="size"):
        _optimizer(size_range=(-0.5, 0.5)).solve(
            _alpha(), np.zeros(4), size=np.zeros(3)
        )


def test_adv_participation_requires_adv():
    opt = _optimizer(adv_participation=0.05)
    with pytest.raises(ValueError, match="adv"):
        opt.solve(_alpha(), np.zeros(4))


def test_impact_cost_requires_adv():
    opt = PortfolioOptimizer(
        PortfolioConstraints(),
        PortfolioObjective(impact_cost=1.0),
    )
    with pytest.raises(ValueError, match="adv"):
        opt.solve(_alpha(), np.zeros(4))


def test_min_cash_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="min_cash"):
        _optimizer(min_cash=-0.1)
    with pytest.raises(ValueError, match="min_cash"):
        _optimizer(min_cash=1.0)


def test_max_positions_must_be_positive():
    with pytest.raises(ValueError, match="max_positions"):
        _optimizer(max_positions=0)


def test_min_trade_amount_must_be_positive():
    with pytest.raises(ValueError, match="min_trade_amount"):
        _optimizer(min_trade_amount=0.0)


def test_turnover_budget_must_be_positive():
    with pytest.raises(ValueError, match="turnover_budget"):
        _optimizer(turnover_budget=-1.0)


def test_alpha_prev_cov_shapes_must_agree():
    opt = _optimizer()
    with pytest.raises(ValueError, match="alpha"):
        opt.solve(np.ones(3), np.zeros(4))
    with pytest.raises(ValueError, match="prev_weights"):
        opt.solve(_alpha(4), np.zeros(3))
    risky = PortfolioOptimizer(
        PortfolioConstraints(),
        PortfolioObjective(risk_aversion=1.0),
    )
    with pytest.raises(ValueError, match="cov"):
        risky.solve(_alpha(4), np.zeros(4))
    with pytest.raises(ValueError, match="cov"):
        risky.solve(_alpha(4), np.zeros(4), cov=_diag_cov(3))


# --- basic solve semantics ------------------------------------------------


def test_empty_universe_returns_zero_book_without_solver():
    sol = _optimizer().solve(np.full(4, np.nan), np.zeros(4))
    assert sol.status == "empty"
    assert sol.weights.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert sol.objective_value == 0.0


def test_non_finite_alpha_entries_get_zero_weight():
    alpha = np.array([1.0, np.inf, 0.5, np.nan])
    sol = _optimizer(single_weight_cap=0.5).solve(alpha, np.zeros(4))
    assert sol.status == "optimal"
    # OSQP satisfies the w == 0 binding to solver tolerance, not exactly.
    assert np.abs(sol.weights[1]) < 1e-8 and np.abs(sol.weights[3]) < 1e-8
    assert np.isfinite(sol.weights).all()


def test_long_only_and_budget():
    sol = _optimizer(single_weight_cap=1.0).solve(_alpha(), np.zeros(4))
    assert sol.weights.min() >= -1e-6
    assert sol.weights.sum() <= 1.0 + 1e-6


def test_single_weight_cap_never_broken():
    sol = _optimizer(single_weight_cap=0.3).solve(_alpha(), np.zeros(4))
    assert sol.weights.max() <= 0.3 + 1e-6
    # The cap binds: the two strongest names sit at the cap.
    assert np.isclose(sol.weights[0], 0.3, atol=1e-4)
    assert np.isclose(sol.weights[1], 0.3, atol=1e-4)


def test_weights_monotone_in_alpha():
    sol = _optimizer(single_weight_cap=1.0).solve(_alpha(), np.zeros(4))
    diffs = np.diff(sol.weights)
    assert (diffs <= 1e-6).all()


def test_solve_is_deterministic():
    opt = _optimizer(single_weight_cap=0.3)
    a = opt.solve(_alpha(), np.zeros(4)).weights
    b = opt.solve(_alpha(), np.zeros(4)).weights
    assert np.array_equal(a, b)


def test_risk_aversion_diversifies():
    # With a dominant alpha and no risk penalty the book concentrates in
    # the single best name; a strong risk penalty splits the budget.
    alpha = np.asarray([1.1, 1.0])
    prev = np.zeros(2)
    low = PortfolioOptimizer(
        PortfolioConstraints(single_weight_cap=1.0),
        PortfolioObjective(risk_aversion=0.0),
    ).solve(alpha, prev).weights
    high = PortfolioOptimizer(
        PortfolioConstraints(single_weight_cap=1.0),
        PortfolioObjective(risk_aversion=100.0),
    ).solve(alpha, prev, cov=_diag_cov(2)).weights
    # Higher risk aversion strictly reduces concentration (Herfindahl).
    hhi = lambda w: float(np.sum(w**2))  # noqa: E731
    assert hhi(high) < hhi(low) - 1e-4
    assert low[0] > 0.99  # no-risk book: all in the higher alpha
    assert high[1] > 0.4  # risk penalty: meaningfully split


def test_turnover_cost_penalizes_rebalance():
    alpha = np.asarray([0.01, 0.02, 0.03])
    prev = np.full(3, 1 / 3)
    free = _optimizer(single_weight_cap=0.5).solve(alpha, prev).weights
    costly = PortfolioOptimizer(
        PortfolioConstraints(single_weight_cap=0.5),
        PortfolioObjective(risk_aversion=0.0, turnover_cost=0.1),
    ).solve(alpha, prev).weights
    # With alpha below the cost threshold every move is value-negative:
    # the costly solution holds the book exactly.
    assert np.allclose(costly, prev, atol=1e-4)
    assert np.abs(free - prev).sum() > np.abs(costly - prev).sum() + 1e-3


def test_impact_cost_spreads_trades_by_adv():
    # Analytic optimum: alpha=[0.05,0.05], adv=[1e6,1e6], capital=1e6,
    # impact_cost=1 -> maximize 0.1w - 2w^2 -> w_i = 0.025 each.
    alpha = np.full(2, 0.05)
    adv = np.full(2, 1e6)
    opt = PortfolioOptimizer(
        PortfolioConstraints(single_weight_cap=0.5),
        PortfolioObjective(risk_aversion=0.0, impact_cost=1.0),
    )
    sol = opt.solve(alpha, np.zeros(2), adv=adv, capital=1e6)
    assert np.allclose(sol.weights, 0.025, atol=1e-3)
    # Without impact the same problem concentrates at the caps.
    free = _optimizer(single_weight_cap=0.5).solve(alpha, np.zeros(2)).weights
    assert np.abs(sol.weights - free).sum() > 0.5


def test_weights_invariant_to_capital_without_absolute_terms():
    a = _optimizer(single_weight_cap=0.3).solve(_alpha(), np.zeros(4))
    b = _optimizer(single_weight_cap=0.3).solve(_alpha(), np.zeros(4), capital=1e9)
    assert np.array_equal(a.weights, b.weights)


def test_adv_masked_names_skip_adv_terms():
    # A name with non-finite ADV is excluded from the ADV cap and the
    # impact term, but still tradable inside the other constraints.
    adv = np.array([1e6, np.nan])
    alpha = np.asarray([1.0, 1.0])
    opt = PortfolioOptimizer(
        PortfolioConstraints(single_weight_cap=0.5, adv_participation=0.1),
        PortfolioObjective(risk_aversion=0.0, impact_cost=1.0),
    )
    sol = opt.solve(alpha, np.zeros(2), adv=adv, capital=1e6)
    assert sol.status == "optimal"
    # Name 0 is capped at rate*adv/capital = 0.1; name 1 is unconstrained
    # by ADV and gets the remaining budget up to its cap.
    assert sol.weights[0] <= 0.1 + 1e-4
    assert sol.weights[1] > 0.1


# --- convex constraints ----------------------------------------------------


def test_industry_cap_enforced():
    industries = np.asarray([0, 0, 1, 1])
    sol = _optimizer(single_weight_cap=1.0, industry_cap=0.3).solve(
        _alpha(), np.zeros(4), industries=industries
    )
    g0 = sol.weights[industries == 0].sum()
    g1 = sol.weights[industries == 1].sum()
    assert g0 <= 0.3 + 1e-4
    assert g1 <= 0.3 + 1e-4
    # Both caps bind under equal positive alpha pressure.
    assert np.isclose(g0, 0.3, atol=1e-3)
    assert np.isclose(g1, 0.3, atol=1e-3)


def test_beta_range_enforced():
    beta = np.asarray([1.0, 2.0, 0.5])
    low, high = 0.8, 1.2
    sol = _optimizer(single_weight_cap=1.0, beta_range=(low, high)).solve(
        np.full(3, 1.0), np.zeros(3), beta=beta
    )
    exposure = float(beta @ sol.weights)
    assert low - 1e-4 <= exposure <= high + 1e-4
    assert np.isclose(sol.weights.sum(), 1.0, atol=1e-3)


def test_size_range_enforced():
    size = np.asarray([-1.0, 0.0, 1.0])
    low, high = -0.5, 0.5
    sol = _optimizer(single_weight_cap=1.0, size_range=(low, high)).solve(
        np.full(3, 1.0), np.zeros(3), size=size
    )
    exposure = float(size @ sol.weights)
    assert low - 1e-4 <= exposure <= high + 1e-4


def test_adv_participation_enforced():
    adv = np.full(2, 1e6)
    alpha = np.full(2, 1.0)
    rate = 0.05
    sol = _optimizer(single_weight_cap=1.0, adv_participation=rate).solve(
        alpha, np.zeros(2), adv=adv, capital=1e6
    )
    # |Δw| * capital <= rate * adv  ->  w_i <= 0.05.
    assert sol.weights.max() <= rate + 1e-4
    assert np.isclose(sol.weights[0], rate, atol=1e-3)


def test_turnover_budget_enforced():
    alpha = np.asarray([1.0, 0.0])
    prev = np.full(2, 0.5)
    budget = 0.2
    sol = _optimizer(single_weight_cap=1.0, turnover_budget=budget).solve(
        alpha, prev
    )
    turnover = float(np.abs(sol.weights - prev).sum())
    assert turnover <= budget + 1e-4
    # The budget binds: without it the book would flip entirely.
    assert turnover > 0.05
    assert sol.weights[0] < 0.9


def test_min_cash_enforced():
    sol = _optimizer(single_weight_cap=1.0, min_cash=0.3).solve(
        np.full(4, 1.0), np.zeros(4)
    )
    assert sol.weights.sum() <= 0.7 + 1e-4
    assert np.isclose(sol.weights.sum(), 0.7, atol=1e-3)


def test_infeasible_problem_raises():
    opt = _optimizer(single_weight_cap=0.1, beta_range=(0.5, 0.6))
    with pytest.raises(PortfolioOptimizationError, match="infeasible"):
        opt.solve(np.asarray([1.0]), np.zeros(1), beta=np.asarray([2.0]))


# --- post-pass projection --------------------------------------------------


def test_min_trade_amount_filters_small_trades():
    # QP wants [0.4, 0.4, 0.2] (cap 0.4, budget 1); trades below 0.25 of
    # capital are dropped; the freed budget stays in cash (never
    # renormalized upward).
    sol = _optimizer(
        single_weight_cap=0.4, min_trade_amount=250000.0
    ).solve(_alpha(3, base=1.0, step=-0.1), np.zeros(3), capital=1e6)
    assert sol.weights[2] == 0.0
    assert np.isclose(sol.weights.sum(), 0.8, atol=1e-6)
    assert 2 in sol.diagnostics["min_trade_dropped"]


def test_min_trade_amount_keeps_held_position_on_small_sell():
    # The QP wants to trim name 2 from 0.3 to 0.25 (trade 5e4); below the
    # 1.5e5 minimum the small sell never executes and the position is
    # held at its previous weight.
    alpha = np.asarray([1.0, 0.0, -0.05])
    prev = np.asarray([0.0, 0.0, 0.3])
    sol = _optimizer(
        single_weight_cap=0.5,
        turnover_budget=0.55,
        min_trade_amount=150000.0,
    ).solve(alpha, prev, capital=1e6)
    assert np.isclose(sol.weights[2], 0.3, atol=1e-6)
    assert np.isclose(sol.weights[0], 0.5, atol=1e-4)
    assert 2 in sol.diagnostics["min_trade_dropped"]


def test_max_positions_is_a_hard_ceiling():
    # QP wants [0.4, 0.4, 0.2]; with a ceiling of 2 the smallest weight is
    # dropped (never renormalized upward).
    sol = _optimizer(
        single_weight_cap=0.4, max_positions=2
    ).solve(_alpha(base=1.0, step=-0.1), np.zeros(4))
    assert np.count_nonzero(sol.weights > 0) == 2
    assert sol.weights[2] == 0.0
    assert 2 in sol.diagnostics["position_cap_dropped"]


def test_projection_preserves_caps_industry_and_budget():
    # The QP wants [0.4, 0.1, 0.4, 0.1]; min-trade drops the two 0.1
    # entries and the projection must never break caps, industry caps or
    # the budget on the way down.
    industries = np.asarray([0, 0, 1, 1])
    sol = _optimizer(
        single_weight_cap=0.4,
        industry_cap=0.5,
        max_positions=2,
        min_trade_amount=150000.0,
    ).solve(
        np.asarray([1.0, 0.95, 0.9, 0.85]),
        np.zeros(4),
        industries=industries,
        capital=1e6,
    )
    assert sol.weights.max() <= 0.4 + 1e-6
    assert sol.weights.min() >= -1e-9
    assert sol.weights.sum() <= 1.0 + 1e-9
    assert sol.weights[industries == 0].sum() <= 0.5 + 1e-6
    assert sol.weights[industries == 1].sum() <= 0.5 + 1e-6


def test_projection_does_not_renormalize_upward():
    # After projection the book sum never exceeds the QP book sum.
    qp_only = _optimizer(single_weight_cap=0.4).solve(
        _alpha(3, base=1.0, step=-0.1), np.zeros(3)
    )
    projected = _optimizer(
        single_weight_cap=0.4, max_positions=2
    ).solve(_alpha(base=1.0, step=-0.1), np.zeros(4))
    assert projected.weights.sum() <= qp_only.weights.sum() + 1e-9


def test_min_trade_filters_before_position_cap():
    # Min-trade runs first: name 1's small buy (0.15 -> 1.5e5 < 2e5) is
    # suppressed and the name stays at its previous weight; the position
    # ceiling then drops the smallest remaining weight (name 2, a forced
    # liquidation exempt from the min-trade filter).
    alpha = np.asarray([1.0, 0.9, 0.8])
    prev = np.asarray([0.0, 0.35, 0.15])
    sol = _optimizer(
        single_weight_cap=0.5, max_positions=2, min_trade_amount=200000.0
    ).solve(alpha, prev, capital=1e6)
    assert np.isclose(sol.weights[1], 0.35, atol=1e-6)
    assert np.count_nonzero(sol.weights > 0) == 2
    assert 2 in sol.diagnostics["position_cap_dropped"]


def test_beta_exposure_reported_after_projection():
    beta = np.asarray([1.0, 2.0, 0.5])
    sol = _optimizer(
        single_weight_cap=0.5, beta_range=(0.8, 1.2), max_positions=1
    ).solve(np.full(3, 1.0), np.zeros(3), beta=beta)
    assert "beta_exposure" in sol.diagnostics
    assert "post_beta_exposure" in sol.diagnostics
    # The QP satisfied the range; the projection residual is reported,
    # never silently hidden.
    assert sol.diagnostics["beta_exposure"] >= 0.8 - 1e-4
    assert isinstance(sol.diagnostics["post_beta_exposure"], float)


def test_diagnostics_report_turnover_and_positions():
    prev = np.full(4, 0.25)
    sol = _optimizer(single_weight_cap=0.5).solve(_alpha(), prev)
    assert sol.diagnostics["turnover"] == pytest.approx(
        float(np.abs(sol.weights - prev).sum()), abs=1e-9
    )
    assert sol.diagnostics["positions"] == int(np.count_nonzero(sol.weights > 0))
