"""P3 execution-version and portfolio-constructor provenance.

Every artifact that can be used as execution or promotion evidence records
the same resolved configuration.  Keeping this in one lightweight module
prevents strategy, protocol, bare-factor and parity outputs from drifting to
different field sets.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from .constructor import (
    PORTFOLIO_CONSTRUCTOR_VERSION,
    effective_ranks,
    validate_portfolio_config,
)
from .rebalance import RebalancePolicy


# p19 §5: bumped 2 -> 3 — execution semantics switched to raw nominal
# prices (whole lots, cash, commission floor, min trade amount, dollar
# volume) with the exact daily_price_limit/daily_suspension constraint
# tables replacing the approximate board-rate inference.
# p35 §3: bumped 3 -> 4 — the book state evolution switched from frozen
# last-target weights (an implicit daily zero-cost re-normalization) to
# share-conserving drift between executions plus share-level execution
# (lot-floor orders, full exits, cash + fees reconciliation identity).
# P58: ordinary backtest and paper execution share sequential cash sizing.
# Generation 5 is already used by the user's uncommitted lot provenance work.
EXECUTION_SPEC_VERSION = 6

_PORTFOLIO_CONFIG_FIELDS = (
    "rebalance_frequency",
    "target_horizon",
    "portfolio_method",
    "top_n",
    "buy_rank",
    "sell_rank",
    "min_trade_amount",
    "turnover_budget",
    "turnover_allocation",
    "target_weight_change_threshold",
    "lot_size",
    "initial_capital",
    "single_weight_cap",
    "commission_rate",
    "min_commission",
    "stamp_tax_rate",
    "transfer_fee_rate",
    "slippage_rate",
)


def portfolio_config_provenance(config) -> dict[str, Any]:
    """Return the resolved P3 portfolio/execution configuration."""

    validate_portfolio_config(config)
    policy = RebalancePolicy.from_config(config)
    buy_rank, sell_rank = effective_ranks(config)
    return {
        "rebalance_frequency": policy.frequency,
        "target_horizon": policy.horizon,
        "portfolio_method": str(config.portfolio_method),
        "top_n": int(config.top_n),
        "buy_rank": buy_rank,
        "sell_rank": sell_rank,
        "min_trade_amount": (
            None
            if config.min_trade_amount is None
            else float(config.min_trade_amount)
        ),
        "turnover_budget": (
            None
            if config.turnover_budget is None
            else float(config.turnover_budget)
        ),
        "turnover_allocation": str(
            getattr(config, "turnover_allocation", "priority")
        ),
        "target_weight_change_threshold": float(
            config.target_weight_change_threshold
        ),
        "initial_capital": float(config.initial_capital),
        "lot_size": int(config.lot_size),
        "single_weight_cap": float(config.single_weight_cap),
        "commission_rate": float(config.commission_rate),
        "min_commission": float(config.min_commission),
        "stamp_tax_rate": float(config.stamp_tax_rate),
        "transfer_fee_rate": float(config.transfer_fee_rate),
        "slippage_rate": float(config.slippage_rate),
    }


def execution_provenance(config) -> dict[str, Any]:
    """Return the complete versioned constructor provenance block."""

    return {
        "execution_version": EXECUTION_SPEC_VERSION,
        "portfolio_constructor_version": PORTFOLIO_CONSTRUCTOR_VERSION,
        "portfolio_config": portfolio_config_provenance(config),
    }


def validate_portfolio_config_provenance(
    recorded: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require a complete, canonical P3 portfolio configuration record.

    This validates artifact provenance without assuming a particular runtime
    configuration.  It rejects partial dictionaries and values that could not
    have been emitted by :func:`portfolio_config_provenance`; callers that do
    have a runtime config can additionally compare the returned mapping with
    :func:`portfolio_config_provenance`.
    """

    if not isinstance(recorded, Mapping):
        raise ValueError("portfolio_config provenance is missing or not a mapping")
    actual = dict(recorded)
    expected_fields = set(_PORTFOLIO_CONFIG_FIELDS)
    missing = sorted(expected_fields - set(actual))
    extra = sorted(set(actual) - expected_fields)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"unknown fields: {', '.join(extra)}")
        raise ValueError("portfolio_config provenance is incomplete: " + "; ".join(details))

    config = SimpleNamespace(**actual)
    try:
        canonical = portfolio_config_provenance(config)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid portfolio_config provenance: {exc}") from exc
    if actual != canonical:
        differing = sorted(
            key for key in _PORTFOLIO_CONFIG_FIELDS
            if actual.get(key) != canonical.get(key)
        )
        raise ValueError(
            "portfolio_config provenance is not canonical"
            + (f": {', '.join(differing)}" if differing else "")
        )
    return canonical


def validate_execution_provenance(
    recorded: Mapping[str, Any] | None,
    config,
) -> dict[str, Any]:
    """Require external weights to carry the exact current provenance."""

    expected = execution_provenance(config)
    if recorded is None:
        raise ValueError(
            "external target_weights require constructor provenance; "
            "legacy weights cannot claim P3 golden parity"
        )
    actual = dict(recorded)
    validate_portfolio_config_provenance(actual.get("portfolio_config"))
    if actual != expected:
        raise ValueError(
            "external target_weights constructor provenance does not match "
            f"the golden configuration: recorded={actual!r}, expected={expected!r}"
        )
    return expected
