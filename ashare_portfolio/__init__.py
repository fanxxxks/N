"""Portfolio construction and execution-parity package.

Package-level exports stay lazy so low-level configuration code can import
``ashare_portfolio.rebalance`` without importing the optimizer stack.
Direct submodule imports remain the preferred internal style.  The golden
execution spec lives in ``ashare_trading.golden`` (arch-review F6): it
depends on the backtest and trading layers, so it belongs to the top
layer and this bottom package no longer re-exports it.
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
    "EXECUTION_SPEC_VERSION": (
        "ashare_portfolio.execution_spec",
        "EXECUTION_SPEC_VERSION",
    ),
    "execution_provenance": (
        "ashare_portfolio.execution_spec",
        "execution_provenance",
    ),
    "portfolio_config_provenance": (
        "ashare_portfolio.execution_spec",
        "portfolio_config_provenance",
    ),
    "validate_portfolio_config_provenance": (
        "ashare_portfolio.execution_spec",
        "validate_portfolio_config_provenance",
    ),
    "validate_execution_provenance": (
        "ashare_portfolio.execution_spec",
        "validate_execution_provenance",
    ),
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
