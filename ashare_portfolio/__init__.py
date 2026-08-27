"""Portfolio layer (T3-01): connects factor outputs to tradable portfolios.

The factor layer (``ashare_model``) produces expected alpha / rank; this
package owns portfolio construction (``optimizer``) and, in T3-02, the
unified execution spec (``golden``) that ties the fast vectorized research
engine and the whole-lot paper matcher to the same golden tests.
"""

from ashare_portfolio.optimizer import (
    PORTFOLIO_OPTIMIZER_VERSION,
    PortfolioConstraints,
    PortfolioObjective,
    PortfolioOptimizationError,
    PortfolioOptimizer,
    PortfolioSolution,
)

__all__ = [
    "PORTFOLIO_OPTIMIZER_VERSION",
    "PortfolioConstraints",
    "PortfolioObjective",
    "PortfolioOptimizationError",
    "PortfolioOptimizer",
    "PortfolioSolution",
]
