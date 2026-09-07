"""Research-domain data home (shared bottom-tier module, IP-12 / 01-A2).

The pure data part of the research-domain registry — the three domain
definitions, their exhaustive feature partition and their legal execution
points — lives here so that ``ashare_data.config`` can resolve
``protocol.domain`` without importing ``ashare_model`` (AGENTS §9:
``ashare_data`` is the bottom layer and must never depend on the model
package; the two A2 offenders were function-level imports inside
``ashare_data/config.py``).

Contract: ``docs/p6_research_domain_contract.md``.  This module holds the
declarative definitions only; model-side behavior (tensor restriction,
vocabulary token ids) stays in ``ashare_model.research_domain``, which
remains the contract-named module and the single owner of
``RESEARCH_DOMAIN_VERSION`` — this module deliberately carries no version
constant, so no second version owner can drift.

Importing ``ashare_portfolio.rebalance`` is a same-tier dependency
(``ashare_data / ashare_portfolio / shared execution`` form the bottom of
the §9 graph; ``ashare_data/config.py`` already imports the same module
for the shared protocol validation).
"""

from __future__ import annotations

from dataclasses import dataclass

from ashare_portfolio.rebalance import RebalancePolicy

# Reserved compatible semantic: no domain defaults, pre-P6 behavior.
UNIFIED_DOMAIN_ID = "unified"


@dataclass(frozen=True)
class ResearchDomain:
    """Declarative definition of one research domain."""

    id: str
    label: str
    description: str
    features: tuple[str, ...]
    # Research intent: [min, max] target horizon in trading sessions;
    # ``None`` upper bound means unbounded (validated per calendar).
    horizon_range: tuple[int, int | None]
    frequencies: tuple[str, ...]
    default_frequency: str
    default_horizon: int
    baseline_signals: tuple[str, ...]
    # Per-domain reward/turnover parameters (P6 §1): domains never share
    # one turnover constraint, and cross-domain rewards are not comparable.
    turnover_budget: float
    cost_weight: float

    def default_execution(self) -> tuple[str, int]:
        """The domain's default (frequency, horizon) execution point."""
        return (self.default_frequency, self.default_horizon)

    def is_legal_execution(self, frequency: str, horizon: int) -> bool:
        """Whether ``(frequency, horizon)`` is a legal execution point.

        Legal means: the cadence belongs to the domain, the horizon lies
        inside the domain's target range, and ``RebalancePolicy`` accepts
        the pair under the P3 non-overlapping-label rule (the monthly
        calendar-dependent part is validated separately on a concrete
        date axis by ``RebalancePolicy.rebalance_mask``).
        """

        frequency = str(frequency)
        if frequency not in self.frequencies:
            return False
        horizon = int(horizon)
        minimum, maximum = self.horizon_range
        if horizon < minimum:
            return False
        if maximum is not None and horizon > maximum:
            return False
        try:
            RebalancePolicy(frequency, horizon)
        except ValueError:
            return False
        return True


_SHORT_FEATURES = (
    # Price (Tier A daily bars).
    "RET_1",
    "RET_5",
    "RET_10",
    "REVERSAL_5",
    # Volume / turnover.
    "TURNOVER",
    "TURNOVER_CHG",
    "TURNOVER_MA5",
    "TURNOVER_MA20",
    "TURNOVER_STD20",
    "VOLUME_RATIO",
    "VOLUME_IMPACT",
    "AMPLITUDE",
    "CLOSE_POSITION",
    # Limit moves.
    "LIMIT_UP_EVENT",
    "LIMIT_DOWN_EVENT",
    "LIMIT_STREAK",
    "LIMIT_UP_CNT_20",
    "LIMIT_BREAK",
    # Intraday / overnight decomposition.
    "OVERNIGHT_RET",
    "INTRADAY_RET",
    # Liquidity.
    "ILLIQ_20",
    "AMOUNT_SHARE",
    # Microstructure (Tier A price history).
    "SUSPEND_DAYS_60",
    "LIST_AGE",
    # P9 §5.2/§5.3: liquidity shock + volume shrinkage + limit-event
    # conditioning (all Tier A daily bars).
    "LIQ_SHOCK_20",
    "VOLUME_SHRINK_5_20",
    "LIMIT_UP_CNT_5",
    "LIMIT_DOWN_STREAK",
    "LIMIT_BREAK_5",
)

_MEDIUM_FEATURES = (
    # Momentum / reversal / anchoring.
    "MOMENTUM_20",
    "MOMENTUM_60",
    "RET_120",
    "REVERSAL_60",
    "REVERSAL_120",
    "HIGH_52W",
    # Volatility / distribution.
    "VOL_20",
    "VOL_60",
    "SKEW_20",
    "KURT_20",
    "MAX_20",
    # Market-model risk.
    "BETA_60",
    "IVOL_60",
    "RSQ_60",
    # Technical trend.
    "BIAS_20",
    "RSI_14",
    "ATR_14",
    "MACD_DIF",
    "MACD_DEA",
    # Industry-relative cross-section.
    "IND_REL_RET_5",
    "IND_REL_RET_20",
    "IND_REL_VOL_20",
    "IND_REL_TURNOVER",
    # External cross-section flows.
    "INDUSTRY_MOMENTUM",
    "MARGIN_BALANCE_CHG",
    # P9 §5.1/§5.2/§5.4: industry-residualized momentum, price-volume
    # divergence, per-stock crowding.
    "IND_REL_RET_60",
    "IND_REL_RET_120",
    "PV_DIV_20",
    "CROWD_TURNOVER_60",
    "CROWD_AMOUNT_60",
    "MARGIN_CROWD_60",
)

_SLOW_FEATURES = (
    # Valuation.
    "PE_TTM",
    "PB",
    "PS_TTM",
    # Quality.
    "ROE",
    "ROA",
    "GROSS_MARGIN",
    "NET_MARGIN",
    "DEBT_RATIO",
    # Growth.
    "REVENUE_YOY",
    "PROFIT_YOY",
    "DIVIDEND_YIELD",
    # Size (a slow persistent characteristic; daily-bar reconstruction).
    "MARKET_CAP",
    # P13 family ⑤ (docs/p13_fundamental_fields_contract.md §5.5): the
    # cash-flow quality / accruals / asset-growth / earnings-acceleration
    # features are slow-fundamental by construction (report-period data,
    # every_20_days/monthly execution points).
    "CASHFLOW_QUALITY",
    "ACCRUALS",
    "ASSET_GROWTH",
    "EARNINGS_ACCEL",
)

RESEARCH_DOMAINS: dict[str, ResearchDomain] = {
    "short_price_volume": ResearchDomain(
        id="short_price_volume",
        label="短周期价格量",
        description=(
            "Tier A price / volume / limit-move / liquidity / intraday and "
            "microstructure features; 1-5 session targets traded daily or "
            "every 5 sessions."
        ),
        features=_SHORT_FEATURES,
        horizon_range=(1, 5),
        frequencies=("daily", "every_5_days"),
        default_frequency="daily",
        default_horizon=1,
        baseline_signals=(
            "REVERSAL_5",
            "TURNOVER",
            "ILLIQ_20",
            "OVERNIGHT_RET",
            "LIMIT_UP_CNT_20",
        ),
        turnover_budget=0.20,
        cost_weight=1.0,
    ),
    "medium_cross_section": ResearchDomain(
        id="medium_cross_section",
        label="中周期横截面",
        description=(
            "Momentum / reversal / volatility / risk / technical / industry-"
            "relative cross-section features; 5-20 session targets traded "
            "every 5 or 10 sessions (legal points cover 5-10 sessions under "
            "the non-overlap rule)."
        ),
        features=_MEDIUM_FEATURES,
        horizon_range=(5, 20),
        frequencies=("every_5_days", "every_10_days"),
        default_frequency="every_10_days",
        default_horizon=10,
        baseline_signals=(
            "MOMENTUM_20",
            "RSQ_60",
            "REVERSAL_60",
            "IND_REL_RET_20",
        ),
        turnover_budget=0.10,
        cost_weight=1.0,
    ),
    "slow_fundamental": ResearchDomain(
        id="slow_fundamental",
        label="慢周期基本面",
        description=(
            "Valuation / quality / growth / size features; 20+ session "
            "targets traded every 20 sessions by default (monthly stays a "
            "calendar-gated option; event-driven cadence is a documented "
            "non-goal of P6)."
        ),
        features=_SLOW_FEATURES,
        horizon_range=(20, None),
        frequencies=("every_20_days", "monthly"),
        default_frequency="every_20_days",
        default_horizon=20,
        baseline_signals=(
            "ROE",
            "PE_TTM",
            "REVENUE_YOY",
            "DIVIDEND_YIELD",
            "MARKET_CAP",
        ),
        turnover_budget=0.05,
        cost_weight=1.0,
    ),
}


def resolve_domain(domain_id: str) -> ResearchDomain:
    """Resolve one of the three research domains; unknown ids fail fast.

    ``unified`` is not resolvable here: callers handle it explicitly as
    the reserved compatible semantic.
    """

    try:
        return RESEARCH_DOMAINS[str(domain_id)]
    except KeyError:
        allowed = ", ".join(RESEARCH_DOMAINS)
        raise ValueError(
            f"unknown research domain {domain_id!r}; choose from "
            f"{allowed} or {UNIFIED_DOMAIN_ID!r}"
        ) from None


def domain_of_feature(name: str) -> ResearchDomain:
    """The single research domain that owns ``name``.

    The partition is exhaustive over the live vocabulary and disjoint, so
    the owner is unique; deprecated or unknown features are rejected.
    """

    name = str(name)
    for domain in RESEARCH_DOMAINS.values():
        if name in domain.features:
            return domain
    raise ValueError(
        f"feature {name!r} is not owned by any research domain "
        "(unknown or deprecated)"
    )
