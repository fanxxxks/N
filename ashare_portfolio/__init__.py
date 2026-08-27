"""Portfolio layer (T3-01/T3-02): connects factor outputs to tradable
portfolios.

The factor layer (``ashare_model``) produces expected alpha / rank; this
package owns portfolio construction (``optimizer``) and the unified
execution spec (``golden``) that ties the fast vectorized research engine
and the whole-lot paper matcher to the same golden tests.
"""

from ashare_portfolio.golden import (
    EXECUTION_SPEC_VERSION,
    DayParityRecord,
    FillRecord,
    GoldenParity,
    GoldenParityViolation,
    ParityReport,
    apply_blocking_rule,
)
from ashare_portfolio.optimizer import (
    PORTFOLIO_OPTIMIZER_VERSION,
    PortfolioConstraints,
    PortfolioObjective,
    PortfolioOptimizationError,
    PortfolioOptimizer,
    PortfolioSolution,
)

__all__ = [
    "EXECUTION_SPEC_VERSION",
    "DayParityRecord",
    "FillRecord",
    "GoldenParity",
    "GoldenParityViolation",
    "PORTFOLIO_OPTIMIZER_VERSION",
    "ParityReport",
    "PortfolioConstraints",
    "PortfolioObjective",
    "PortfolioOptimizationError",
    "PortfolioOptimizer",
    "PortfolioSolution",
    "apply_blocking_rule",
]
