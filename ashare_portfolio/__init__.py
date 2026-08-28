"""Portfolio construction and execution-parity package.

Package-level exports stay lazy so low-level configuration code can import
``ashare_portfolio.rebalance`` without importing the backtest/golden stack and
creating a dependency cycle. Direct submodule imports remain the preferred
internal style.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "PORTFOLIO_CONSTRUCTOR_VERSION": (
        "ashare_portfolio.constructor",
        "PORTFOLIO_CONSTRUCTOR_VERSION",
    ),
    "PortfolioConstructor": (
        "ashare_portfolio.constructor",
        "PortfolioConstructor",
    ),
    "PortfolioOutput": ("ashare_portfolio.constructor", "PortfolioOutput"),
    "EXECUTION_SPEC_VERSION": ("ashare_portfolio.golden", "EXECUTION_SPEC_VERSION"),
    "DayParityRecord": ("ashare_portfolio.golden", "DayParityRecord"),
    "FillRecord": ("ashare_portfolio.golden", "FillRecord"),
    "GoldenParity": ("ashare_portfolio.golden", "GoldenParity"),
    "GoldenParityViolation": ("ashare_portfolio.golden", "GoldenParityViolation"),
    "ParityReport": ("ashare_portfolio.golden", "ParityReport"),
    "apply_blocking_rule": ("ashare_portfolio.golden", "apply_blocking_rule"),
    "PORTFOLIO_OPTIMIZER_VERSION": (
        "ashare_portfolio.optimizer",
        "PORTFOLIO_OPTIMIZER_VERSION",
    ),
    "PortfolioConstraints": ("ashare_portfolio.optimizer", "PortfolioConstraints"),
    "PortfolioObjective": ("ashare_portfolio.optimizer", "PortfolioObjective"),
    "PortfolioOptimizationError": (
        "ashare_portfolio.optimizer",
        "PortfolioOptimizationError",
    ),
    "PortfolioOptimizer": ("ashare_portfolio.optimizer", "PortfolioOptimizer"),
    "PortfolioSolution": ("ashare_portfolio.optimizer", "PortfolioSolution"),
    "RebalancePolicy": ("ashare_portfolio.rebalance", "RebalancePolicy"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module protocol
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
