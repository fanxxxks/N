"""Portfolio layer: constrained portfolio optimization (T3-01).

The factor layer outputs expected alpha / rank; this layer owns the
portfolio constraints — single-name and industry caps, Beta/size exposure
ranges, ADV participation, turnover budget, min cash, position count and
min trade amount — and produces target weights.

The QP core is CVXPY/OSQP (never a hand-written solver):

    maximize  alpha'w
            - risk_aversion * w'Sigma w
            - turnover_cost  * ||w - w0||_1
            - impact_cost    * sum_i ((w_i - w0_i) * capital / adv_i)^2
    subject to  w >= 0
                sum(w) <= 1 - min_cash
                w_i <= single_weight_cap
                sum_{i in g} w_i <= industry_cap        (per industry)
                beta_low <= beta'w <= beta_high
                size_low <= size'w  <= size_high
                |w_i - w0_i| <= adv_participation * adv_i / capital
                ||w - w0||_1 <= turnover_budget
                w_i = 0 for non-finite alpha

The two non-convex constraints (``max_positions``, ``min_trade_amount``)
cannot live in a convex QP; they are enforced by a documented
deterministic projection after the solve.  The projection only shrinks
weights — never renormalized upward (the repo-wide T1-02 contract) — so
every monotone constraint (caps, industry, budget, ADV, turnover) stays
satisfied; Beta/size exposure after projection is reported in
``diagnostics`` as the residual instead of being silently violated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

PORTFOLIO_OPTIMIZER_VERSION = 1


class PortfolioOptimizationError(ValueError):
    """Raised when the optimizer cannot produce a valid solution."""


@dataclass(frozen=True)
class PortfolioConstraints:
    """Hard constraints of the portfolio construction problem.

    ``None`` disables a constraint.  ``min_trade_amount`` is an absolute
    yuan threshold; ``max_positions`` a hard ceiling on ``count(w > 0)``;
    both are enforced by the post-QP projection (see the module docstring).
    """

    single_weight_cap: float = 0.05
    industry_cap: float | None = None
    beta_range: tuple[float, float] | None = None
    size_range: tuple[float, float] | None = None
    adv_participation: float | None = None
    turnover_budget: float | None = None
    min_cash: float = 0.0
    max_positions: int | None = None
    min_trade_amount: float | None = None


@dataclass(frozen=True)
class PortfolioObjective:
    """Penalty weights of the QP objective (all non-negative).

    ``risk_aversion`` scales ``w'Sigma w`` (requires ``cov``);
    ``turnover_cost`` scales the L1 deviation from the previous weights;
    ``impact_cost`` scales the sum of squared ADV participation rates
    (requires ``adv``).
    """

    risk_aversion: float = 1.0
    turnover_cost: float = 0.0
    impact_cost: float = 0.0


@dataclass(frozen=True)
class PortfolioSolution:
    """Result of one portfolio-construction call.

    ``status`` is ``"optimal"`` (QP solved, projection applied) or
    ``"empty"`` (no finite alpha: all-zero book, solver never called).
    ``diagnostics`` carries the documented keys: ``turnover`` /
    ``post_turnover`` (L1 deviation before/after projection),
    ``qp_positions`` / ``positions`` (counts), ``min_trade_dropped`` /
    ``position_cap_dropped`` (index lists), ``beta_exposure`` /
    ``post_beta_exposure`` and ``size_exposure`` / ``post_size_exposure``
    (QP value and the projection residual), ``industry_exposures`` and
    ``max_adv_participation`` (post-projection values).
    """

    weights: np.ndarray
    status: str
    objective_value: float
    diagnostics: dict[str, Any]
    post_processed: bool


class PortfolioOptimizer:
    """CVXPY/OSQP constrained portfolio optimizer.

    ``solve`` accepts the factor layer's expected-alpha vector plus the
    market data the active constraints need (``cov``, ``industries``,
    ``beta``, ``size``, ``adv``); any data a configured constraint
    requires but the caller did not supply raises ``ValueError`` (fail
    fast, never a silent constraint drop).
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        objective: PortfolioObjective,
    ) -> None:
        self.constraints = constraints
        self.objective = objective
        if not 0.0 < constraints.single_weight_cap <= 1.0:
            raise ValueError(
                f"single_weight_cap must be in (0, 1], got "
                f"{constraints.single_weight_cap}"
            )
        if objective.risk_aversion < 0.0:
            raise ValueError(
                f"risk_aversion must be >= 0, got {objective.risk_aversion}"
            )
        if objective.turnover_cost < 0.0:
            raise ValueError(
                f"turnover_cost must be >= 0, got {objective.turnover_cost}"
            )
        if objective.impact_cost < 0.0:
            raise ValueError(
                f"impact_cost must be >= 0, got {objective.impact_cost}"
            )
        if not 0.0 <= constraints.min_cash < 1.0:
            raise ValueError(
                f"min_cash must be in [0, 1), got {constraints.min_cash}"
            )
        if constraints.max_positions is not None and constraints.max_positions < 1:
            raise ValueError(
                f"max_positions must be >= 1, got {constraints.max_positions}"
            )
        if constraints.min_trade_amount is not None and constraints.min_trade_amount <= 0.0:
            raise ValueError(
                f"min_trade_amount must be > 0, got {constraints.min_trade_amount}"
            )
        if constraints.turnover_budget is not None and constraints.turnover_budget < 0.0:
            raise ValueError(
                f"turnover_budget must be >= 0, got {constraints.turnover_budget}"
            )
        if constraints.beta_range is not None:
            if constraints.beta_range[0] > constraints.beta_range[1]:
                raise ValueError(
                    f"beta_range must be ordered, got {constraints.beta_range}"
                )
        if constraints.size_range is not None:
            if constraints.size_range[0] > constraints.size_range[1]:
                raise ValueError(
                    f"size_range must be ordered, got {constraints.size_range}"
                )

    def solve(
        self,
        alpha: np.ndarray,
        prev_weights: np.ndarray,
        *,
        capital: float = 1e6,
        cov: np.ndarray | None = None,
        industries: np.ndarray | None = None,
        beta: np.ndarray | None = None,
        size: np.ndarray | None = None,
        adv: np.ndarray | None = None,
    ) -> PortfolioSolution:
        """Solve one rebalance: expected alpha -> constrained target weights.

        ``alpha`` is the per-name expected return (any finite vector; a
        rank transformed by the factor layer is fine).  Non-finite alpha
        entries are forced to zero weight.  ``capital`` is the account
        equity used by the absolute terms (``min_trade_amount``, ADV
        participation and impact).  ``adv`` is the per-name daily dollar
        volume; entries that are non-finite or <= 0 are excluded from the
        ADV/impact terms (a name without liquidity data cannot be
        liquidity-constrained), and supplying ``adv`` with every entry
        excluded while an ADV term is configured raises ``ValueError``.
        """

        alpha = np.asarray(alpha, dtype=np.float64)
        prev_weights = np.asarray(prev_weights, dtype=np.float64)
        if alpha.ndim != 1 or prev_weights.ndim != 1:
            raise ValueError("alpha and prev_weights must be 1-D")
        if alpha.shape != prev_weights.shape:
            raise ValueError(
                f"alpha shape {alpha.shape} does not match "
                f"prev_weights shape {prev_weights.shape}"
            )
        if not np.isfinite(prev_weights).all():
            raise ValueError("prev_weights must be finite")
        n = alpha.shape[0]

        finite = np.isfinite(alpha)
        if not finite.any():
            return PortfolioSolution(
                weights=np.zeros(n, dtype=np.float64),
                status="empty",
                objective_value=0.0,
                diagnostics={"turnover": 0.0, "positions": 0, "qp_positions": 0},
                post_processed=False,
            )

        c = self.constraints
        obj = self.objective

        # --- data-dependent validation (fail fast) ----------------------
        if c.industry_cap is not None and industries is None:
            raise ValueError("industry_cap requires industries")
        if c.beta_range is not None and beta is None:
            raise ValueError("beta_range requires beta")
        if c.size_range is not None and size is None:
            raise ValueError("size_range requires size")
        if (c.adv_participation is not None or obj.impact_cost > 0.0) and adv is None:
            raise ValueError("adv_participation/impact_cost require adv")
        if obj.risk_aversion > 0.0 and cov is None:
            raise ValueError("risk_aversion > 0 requires cov")

        for name, data in (
            ("industries", industries),
            ("beta", beta),
            ("size", size),
            ("adv", adv),
        ):
            if data is not None:
                data = np.asarray(data, dtype=np.float64)
                if data.shape != (n,):
                    raise ValueError(f"{name} shape {data.shape} does not match alpha")
        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            if cov.shape != (n, n) or not np.isfinite(cov).all():
                raise ValueError("cov must be a finite [n, n] matrix")

        # ADV-valid names: finite, positive dollar volume.
        adv_valid = np.ones(n, dtype=bool)
        if adv is not None:
            adv_arr = np.asarray(adv, dtype=np.float64)
            adv_valid = np.isfinite(adv_arr) & (adv_arr > 0.0)
            if not adv_valid.any() and (c.adv_participation is not None or obj.impact_cost > 0.0):
                raise ValueError("adv contains no valid (finite, positive) entries")

        w = cp.Variable(n)
        constraints = [w >= 0.0, w <= c.single_weight_cap]
        constraints.append(cp.sum(w) <= 1.0 - c.min_cash)
        if (~finite).any():
            constraints.append(w[~finite] == 0.0)

        if c.industry_cap is not None:
            ind = np.asarray(industries, dtype=np.int64)
            for group in np.unique(ind):
                constraints.append(cp.sum(w[ind == group]) <= c.industry_cap)
        if c.beta_range is not None:
            b = np.asarray(beta, dtype=np.float64)
            constraints.append(b @ w >= c.beta_range[0])
            constraints.append(b @ w <= c.beta_range[1])
        if c.size_range is not None:
            s = np.asarray(size, dtype=np.float64)
            constraints.append(s @ w >= c.size_range[0])
            constraints.append(s @ w <= c.size_range[1])
        if c.adv_participation is not None:
            adv_arr = np.asarray(adv, dtype=np.float64)
            cap_per_name = c.adv_participation * adv_arr / float(capital)
            constraints.append(
                cp.abs(w - prev_weights) <= np.where(adv_valid, cap_per_name, np.inf)
            )
        if c.turnover_budget is not None:
            constraints.append(cp.norm1(w - prev_weights) <= c.turnover_budget)

        # Non-finite alpha entries are forced to zero weight, and their
        # (non-finite) coefficients are zeroed before entering the
        # objective so they can never poison the solve.
        alpha_clean = np.where(finite, alpha, 0.0)
        terms = [alpha_clean @ w]
        if obj.risk_aversion > 0.0:
            terms.append(-obj.risk_aversion * cp.quad_form(w, cov))
        if obj.turnover_cost > 0.0:
            terms.append(-obj.turnover_cost * cp.norm1(w - prev_weights))
        if obj.impact_cost > 0.0:
            adv_arr = np.asarray(adv, dtype=np.float64)
            coef = np.where(adv_valid, float(capital) / adv_arr, 0.0)
            terms.append(
                -obj.impact_cost * cp.sum_squares(cp.multiply(coef, w - prev_weights))
            )

        problem = cp.Problem(cp.Maximize(cp.sum(terms)), constraints)
        problem.solve(solver=cp.OSQP)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise PortfolioOptimizationError(
                f"portfolio optimization failed: solver status {problem.status}"
            )
        weights = np.array(w.value, dtype=np.float64)  # copy: projection mutates
        objective_value = float(problem.value)

        # --- post-pass projection (deterministic, only shrinks) ---------
        delta = weights - prev_weights
        qp_turnover = float(np.abs(delta).sum())
        diagnostics: dict[str, Any] = {
            "turnover": qp_turnover,
            "qp_positions": int(np.count_nonzero(weights > 0.0)),
        }
        post_processed = False
        min_trade_dropped: list[int] = []
        if c.min_trade_amount is not None:
            below = (np.abs(delta) * float(capital) < c.min_trade_amount) & (
                np.abs(delta) > 0.0
            )
            if below.any():
                weights[below] = prev_weights[below]
                min_trade_dropped = [int(i) for i in np.where(below)[0]]
                post_processed = True
        diagnostics["min_trade_dropped"] = min_trade_dropped

        position_cap_dropped: list[int] = []
        if c.max_positions is not None:
            while int(np.count_nonzero(weights > 0.0)) > c.max_positions:
                positive = np.where(weights > 0.0)[0]
                drop = int(positive[np.argmin(weights[positive])])
                weights[drop] = 0.0
                position_cap_dropped.append(drop)
                post_processed = True
        diagnostics["position_cap_dropped"] = position_cap_dropped

        diagnostics.update(
            {
                "beta_exposure": (
                    float(beta @ w.value) if beta is not None else None
                ),
                "size_exposure": (
                    float(size @ w.value) if size is not None else None
                ),
                "post_beta_exposure": (
                    float(beta @ weights) if beta is not None else None
                ),
                "post_size_exposure": (
                    float(size @ weights) if size is not None else None
                ),
            }
        )
        if industries is not None:
            ind = np.asarray(industries, dtype=np.int64)
            diagnostics["industry_exposures"] = {
                int(group): float(weights[ind == group].sum())
                for group in np.unique(ind)
            }
        if adv is not None:
            adv_arr = np.asarray(adv, dtype=np.float64)
            post_delta = weights - prev_weights
            participation = np.where(
                adv_valid & (post_delta != 0.0),
                np.abs(post_delta) * float(capital) / adv_arr,
                0.0,
            )
            diagnostics["max_adv_participation"] = float(participation.max())
        diagnostics["post_turnover"] = float(np.abs(weights - prev_weights).sum())
        diagnostics["positions"] = int(np.count_nonzero(weights > 0.0))

        return PortfolioSolution(
            weights=weights,
            status="optimal",
            objective_value=objective_value,
            diagnostics=diagnostics,
            post_processed=post_processed,
        )
