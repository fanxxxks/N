"""Authored research metadata for every vocabulary feature (P7 D1).

Contract: ``docs/p7_maintainability_plan.md`` §6.1.  This module is the
single home for the metadata that cannot be *derived* from an existing
authority — the PIT availability rule, the economic hypothesis, the
expected direction, the semantic type and the promotion permission.
Everything derivable stays derived (never copied):

* ``expected_horizon`` — from the P6 research-domain registry
  (:mod:`ashare_model.research_domain`), so the recommendation can never
  drift from the domain feature split;
* ``compute_cost`` / ``depends_on`` — from ``FactorSpec.warmup`` /
  ``FactorSpec.required_columns`` for locally computed factors, with
  authored overrides only for features outside ``FACTOR_REGISTRY``.

The table must cover **every** ``FEATURE_NAMES`` member:
``FeatureRegistry.build`` fails closed on a missing entry.  Fields are
descriptive only — nothing here changes search, scoring or promotion
semantics (the GP legality wiring is a separately pre-registered change,
P7 Phase E).

Expected direction convention: ``+1`` = higher feature value predicts
higher future return under the stated hypothesis, ``-1`` = predicts
lower, ``0`` = no committed expectation (declared honestly rather than
invented).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from ashare_data.capital_flow import EXTERNAL_FACTOR_NAMES
from ashare_data.fundamentals import FUNDAMENTAL_PIT_NAMES

from .factors import FACTOR_REGISTRY, NEUTRAL_FEATURE_NAMES
from .research_domain import RESEARCH_DOMAINS
from .vocab import FEATURE_NAMES


class SemanticType(str, enum.Enum):
    """Semantic class of the signal a feature produces (P7 Phase E will
    constrain operator legality on these; descriptive until then)."""

    PRICE_LIKE = "price_like"  # price-scale or price-position quantities
    RETURN_LIKE = "return_like"  # return-scale quantities (returns, vol, skew)
    VOLUME_LIKE = "volume_like"  # turnover / liquidity / flow quantities
    FUNDAMENTAL_LIKE = "fundamental_like"  # slow report/scale characteristics
    CROSS_SECTIONAL_SIGNAL = "cross_sectional_signal"  # only meaningful vs peers
    BOOLEAN_EVENT_SIGNAL = "boolean_event_signal"  # sparse event-driven signals


class RecommendedHorizon(str, enum.Enum):
    """Recommended prediction horizon band, aligned with the P6 domains."""

    SHORT = "short"  # 1-5 sessions
    MEDIUM = "medium"  # 5-20 sessions
    SLOW = "slow"  # 20+ sessions


class ComputeCost(str, enum.Enum):
    """Coarse compute/data burden, derived from the factor warmup."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Availability-rule templates (PIT declarations per data source class).
_BAR = "daily bars up to session t inclusive; no future session enters the computation; warmup {warmup} sessions"
_INDUSTRY_REL = (
    "daily bars up to session t inclusive (warmup {warmup} sessions); industry "
    "membership from the Shenwan snapshot, PIT-bounded by the universe contract"
)
_FUNDAMENTAL = (
    "point-in-time fundamental table: the latest report period disclosed by "
    "session t (disclosure-lag aware; the current snapshot is never backfilled)"
)
_CAPITAL = "point-in-time margin/capital-flow feed as of session t"
_SNAPSHOT_EXT = (
    "industry membership snapshot + daily bars up to session t inclusive "
    "(PIT-bounded by the universe contract)"
)
_NEUTRAL = "no live data source; the feature stays neutral (0)"


@dataclass(frozen=True)
class AuthoredFeatureMeta:
    """Authored metadata for one feature; ``None`` cost/depends_on means
    "derive from ``FACTOR_REGISTRY``" (see :func:`resolve_cost` /
    :func:`resolve_depends_on`)."""

    availability_rule: str
    hypothesis: str
    expected_direction: int
    semantic_type: SemanticType
    promotion_allowed: bool = True
    compute_cost: ComputeCost | None = None
    depends_on: tuple[str, ...] | None = None


def resolve_cost(name: str, meta: AuthoredFeatureMeta) -> ComputeCost:
    """Compute cost of one feature: derived from the factor warmup when
    declared locally, else the authored value."""

    if meta.compute_cost is not None:
        return meta.compute_cost
    warmup = FACTOR_REGISTRY[name][0].warmup
    if warmup <= 5:
        return ComputeCost.LOW
    if warmup <= 30:
        return ComputeCost.MEDIUM
    return ComputeCost.HIGH


def resolve_depends_on(name: str, meta: AuthoredFeatureMeta) -> tuple[str, ...]:
    """Data dependencies of one feature: derived from the factor's
    declared columns (+ the industry snapshot for industry-relative
    factors), else the authored value."""

    if meta.depends_on is not None:
        return meta.depends_on
    spec = FACTOR_REGISTRY[name][0]
    deps = tuple(spec.required_columns)
    if spec.family == "industry":
        deps = deps + ("industry_membership_snapshot",)
    return deps


def resolve_expected_horizon(name: str) -> RecommendedHorizon | None:
    """Recommended horizon band from the P6 domain the feature belongs to.

    ``None`` only for features outside every research domain (the
    deprecated neutral member); never a second hand-maintained table.
    """

    for domain in RESEARCH_DOMAINS.values():
        if name in domain.features:
            return {
                "short_price_volume": RecommendedHorizon.SHORT,
                "medium_cross_section": RecommendedHorizon.MEDIUM,
                "slow_fundamental": RecommendedHorizon.SLOW,
            }[domain.id]
    return None


def authored_metadata_of(name: str) -> AuthoredFeatureMeta:
    """Authored metadata of one feature; unknown names fail loudly."""

    try:
        return FEATURE_METADATA[name]
    except KeyError:
        raise ValueError(
            f"feature {name!r} has no authored metadata; every vocabulary "
            "feature must be declared in ashare_model.feature_metadata"
        ) from None


# --- the authored table (every FEATURE_NAMES member, no exceptions) ---------

FEATURE_METADATA: dict[str, AuthoredFeatureMeta] = {
    # Short-horizon price reversal family (return-like).
    "RET_1": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Short-term reversal: yesterday's winners underperform tomorrow (liquidity-driven overreaction).",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "RET_5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=6),
        hypothesis="One-week reversal: recent losers rebound as the overreaction unwinds.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        promotion_allowed=False,  # P9 §4.1: mirror of REVERSAL_5
    ),
    "RET_10": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=11),
        hypothesis="Two-week reversal: extended-horizon overreaction correction.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        promotion_allowed=False,  # P9 §4.1: covered by RET_1 + BIAS_20
    ),
    "VOL_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Low-vol anomaly: lower short-term volatility predicts higher risk-adjusted returns.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "VOL_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Low-vol anomaly at the quarterly horizon.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        promotion_allowed=False,  # P9 §4.1: 0.907 correlated with IVOL_60
    ),
    "TURNOVER": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Hot-money crowding: high turnover marks attention-driven overpricing that later underperforms.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    "TURNOVER_CHG": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Turnover spikes mark attention peaks that fade.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    "VOLUME_RATIO": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Abnormal volume signals speculative interest that mean-reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        promotion_allowed=False,  # P9 §4.1: 0.977 correlated with VOLUME_IMPACT
    ),
    "VOLUME_IMPACT": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Log-volume deviation proxies abnormal trading interest; elevated values fade.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    "AMPLITUDE": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Wide intraday ranges reflect disagreement and instability; predictive of underperformance.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "CLOSE_POSITION": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Strong closes reflect intraday accumulation; short-horizon continuation.",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
    ),
    "MOMENTUM_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=21),
        hypothesis="One-month momentum: modest continuation at the medium horizon.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "MOMENTUM_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=61),
        hypothesis="Quarterly momentum: trend persistence at the medium horizon.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        promotion_allowed=False,  # P9 §4.1: mirror of REVERSAL_60
    ),
    "REVERSAL_5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=6),
        hypothesis="Pre-flipped short-term reversal: buying recent losers captures the overreaction correction.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "SKEW_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Lottery preference: positively skewed names are overbid and underperform.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "KURT_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Tail thickness mixes crash-risk premium and lottery demand; no committed expectation.",
        expected_direction=0,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    # Fundamental PIT family (fundamental-like).
    "PE_TTM": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Value: low valuation (high earnings yield) earns a premium.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "PB": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Book-value anchor: low P/B predicts mispricing mean reversion.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "PS_TTM": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Sales-based value: low P/S marks cheaper cross-sections.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "ROE": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: profitable firms compound value and stay underpriced.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "ROA": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: asset efficiency signals durable profitability.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "GROSS_MARGIN": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: gross profitability proxies pricing power / moat.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "NET_MARGIN": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: bottom-line efficiency.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "REVENUE_YOY": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Growth: top-line expansion predicts fundamental continuation.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "PROFIT_YOY": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Growth: earnings acceleration.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "DEBT_RATIO": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Leverage risk: high debt raises distress risk.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "MARKET_CAP": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Size premium: smaller float caps earn higher risk-adjusted returns in A-shares.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
    ),
    "DIVIDEND_YIELD": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Income/value: dividend payers are disciplined and cheaper.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    # Neutral / external flows.
    "NORTHBOUND_CHG": AuthoredFeatureMeta(
        availability_rule=_NEUTRAL,
        hypothesis="Historical northbound flow change; the daily feed was discontinued (2024-08), so the feature stays neutral for vocabulary stability.",
        expected_direction=0,
        semantic_type=SemanticType.VOLUME_LIKE,
        promotion_allowed=False,
        compute_cost=ComputeCost.LOW,
        depends_on=(),
    ),
    "MARGIN_BALANCE_CHG": AuthoredFeatureMeta(
        availability_rule=_CAPITAL,
        hypothesis="Margin-flow momentum: leveraged inflows chase and lift prices over short horizons.",
        expected_direction=1,
        semantic_type=SemanticType.VOLUME_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("capital_flow_pit",),
    ),
    # Limit-move event family (boolean/event).
    "LIMIT_UP_EVENT": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="One-word limit-up marks extreme attention; next-session squeeze continuation.",
        expected_direction=1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
    ),
    "LIMIT_DOWN_EVENT": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="One-word limit-down marks a forced-selling overhang; continued weakness.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
    ),
    "INDUSTRY_MOMENTUM": AuthoredFeatureMeta(
        availability_rule=_SNAPSHOT_EXT,
        hypothesis="Industry momentum spillover: hot industry groups keep rotating slowly.",
        expected_direction=1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        compute_cost=ComputeCost.LOW,
        depends_on=("industry_membership_snapshot", "close"),
    ),
    # Intraday decomposition (return-like).
    "OVERNIGHT_RET": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Overnight gap captures order-imbalance accumulation; short-horizon continuation.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "INTRADAY_RET": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Intraday return reflects day-session speculation; tends to mean-revert.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    # Liquidity (volume-like).
    "ILLIQ_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Illiquidity premium: less liquid names earn a premium.",
        expected_direction=1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    "AMOUNT_SHARE": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Crowding: stocks absorbing disproportionate market turnover underperform.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    # Lottery / anchoring.
    "MAX_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="MAX lottery effect: recent extreme daily gains are overbid and underperform.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "HIGH_52W": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=250),
        hypothesis="52-week-high anchor: proximity to the high predicts continuation.",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
    ),
    # Market-model risk (cross-sectional by construction).
    "BETA_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Low-beta anomaly within the cross-section (betting-against-beta).",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    "IVOL_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Idiosyncratic-volatility puzzle: high IVOL names are overpriced (lottery demand).",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    "RSQ_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Market co-movement: low R-squared names are harder to arbitrage; the premium is contested.",
        expected_direction=0,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    # Technical (defined per FactorSpec.description).
    "BIAS_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="MA-distance extension: overextension mean-reverts at the cross-section.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "RSI_14": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=14),
        hypothesis="Overbought/oversold oscillator: high RSI mean-reverts across stocks.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "ATR_14": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=14),
        hypothesis="True-range volatility proxy: instability trades like the low-vol anomaly.",
        expected_direction=-1,
        semantic_type=SemanticType.PRICE_LIKE,
    ),
    "MACD_DIF": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=26),
        hypothesis="Trend oscillator: positive DIF marks uptrend persistence (momentum).",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
    ),
    "MACD_DEA": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=26),
        hypothesis="Trend signal line: smoothed trend confirmation.",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
        promotion_allowed=False,  # P9 §4.1: 0.926 correlated with MACD_DIF
    ),
    # Microstructure.
    "SUSPEND_DAYS_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Frequent suspensions signal distress or trading games; penalized.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
    ),
    "LIST_AGE": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Seasoning: older listings shed the IPO speculation halo; mature names behave better.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
    ),
    # Medium-term momentum / reversal (return-like).
    "RET_120": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=121),
        hypothesis="Six-month momentum: intermediate-horizon trend persistence.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        promotion_allowed=False,  # P9 §4.1: mirror of REVERSAL_120
    ),
    "REVERSAL_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=61),
        hypothesis="Pre-flipped medium-term reversal: 60-day losers rebound.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    "REVERSAL_120": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=121),
        hypothesis="Pre-flipped long-horizon reversal: 120-day losers rebound.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
    ),
    # Smoothed turnover (volume-like).
    "TURNOVER_MA5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=5),
        hypothesis="Smoothed crowding: elevated turnover predicts underperformance.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        promotion_allowed=False,  # P9 §4.1: TURNOVER + TURNOVER_MA20 cover it
    ),
    "TURNOVER_MA20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Monthly crowding level.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    "TURNOVER_STD20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Turnover volatility: unstable attention marks speculation.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    # Limit-move statistics (event-derived).
    "LIMIT_STREAK": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Consecutive one-word limit-ups: speculative streaks reverse after the run.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        promotion_allowed=False,  # P9 §7.3 conditional deprecation TRIGGERED (0.980)
    ),
    "LIMIT_UP_CNT_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Repeated limit-ups mark a lottery stock; underperforms subsequently.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
    ),
    "LIMIT_BREAK": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Touched limit-up but failed to hold: failed speculation, weak follow-through.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
    ),
    # Industry-relative cross-section.
    "IND_REL_RET_5": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=6),
        hypothesis="Industry-neutral short reversal: stock-specific overreaction vs peers reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    "IND_REL_RET_20": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=21),
        hypothesis="Industry-neutral monthly move: stock-specific deviation vs peers reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    "IND_REL_VOL_20": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=20),
        hypothesis="Stock-specific volatility vs the industry: idiosyncratic risk is overpriced.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    "IND_REL_TURNOVER": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=1),
        hypothesis="Crowding relative to industry peers.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    # --- P9 v4 additions (docs/p9_factor_family_contract.md §5, APPROVED) ---
    # Family ①: industry-residualized medium-horizon momentum/reversal.
    "IND_REL_RET_60": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=61),
        hypothesis="Industry-neutral quarterly reversal: stock-specific deviation vs peers reverts at the medium horizon.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    "IND_REL_RET_120": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=121),
        hypothesis="Industry-neutral semi-annual reversal: stock-specific deviation vs peers reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
    ),
    # Family ②: liquidity shock, volume shrinkage, price-volume divergence.
    "LIQ_SHOCK_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Liquidity shock: a one-day surge in market share marks attention-driven overpricing that reverses.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        promotion_allowed=False,  # P9 §7.1 second-pass: 0.968 vs VOLUME_IMPACT
    ),
    "VOLUME_SHRINK_5_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Volume shrinkage precedes exhaustion of the prevailing move; shrinking volume flips the sign of the drift.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    "PV_DIV_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Price-volume divergence: returns unconfirmed by volume mean-revert.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    # Family ③: limit-event conditioning (sparse-safe standardization).
    # P9 §7 adjudication: family ③ = negative result on the pre-registered
    # incremental gate (no member reached dIC_OOS >= +0.005; the
    # pre-registered continuation direction was falsified).  The features
    # stay computed and samplable but are not promotable.
    "LIMIT_UP_CNT_5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=5),
        hypothesis="Short-window limit-up clustering: speculative continuation within the week.",
        expected_direction=1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        promotion_allowed=False,  # P9 §7: family ③ negative result
    ),
    "LIMIT_DOWN_STREAK": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Consecutive one-word limit-downs: forced-selling overhang continues over the streak.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        promotion_allowed=False,  # P9 §7: family ③ negative result
    ),
    "LIMIT_BREAK_5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=5),
        hypothesis="Repeated failed limit-ups (炸板): speculative churn, weak follow-through.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        promotion_allowed=False,  # P9 §7: family ③ negative result
    ),
    # Family ④: per-stock crowding.
    "CROWD_TURNOVER_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Self-relative turnover crowding: turnover far above the stock's own baseline marks overpriced attention.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        promotion_allowed=False,  # P9 §7.1 second-pass: 0.971 vs CROWD_AMOUNT_60
    ),
    "CROWD_AMOUNT_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Self-relative amount-share crowding: absorbing more market share than usual precedes underperformance.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
    ),
    "MARGIN_CROWD_60": AuthoredFeatureMeta(
        availability_rule=_CAPITAL,
        hypothesis="Margin crowding: a financing balance far above its own baseline marks leveraged crowding that unwinds.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        compute_cost=ComputeCost.LOW,
        depends_on=("capital_flow_pit",),
    ),
}

# Structural invariants checked at import: the table must cover exactly the
# live vocabulary, and the neutral set must not collide with the external
# feed list (a drift here means a feature was added without metadata).
assert set(FEATURE_METADATA) == set(FEATURE_NAMES), (
    "FEATURE_METADATA must cover every FEATURE_NAMES member"
)
assert not (set(NEUTRAL_FEATURE_NAMES) & set(EXTERNAL_FACTOR_NAMES))


# --- P9 §6: pending_data placeholders (family ⑤, NOT in the vocabulary) ----

@dataclass(frozen=True)
class PendingDataFeature:
    """A pre-declared factor family member waiting for its data source.

    Pending features never enter the vocabulary or the search budget
    (``promotion_allowed=False``); they document the gap so a future data
    contract can lift the placeholder without re-deriving the intent.
    """

    name: str
    hypothesis: str
    required_fields: tuple[str, ...]
    promotion_allowed: bool = False


PENDING_DATA_FEATURES: tuple[PendingDataFeature, ...] = (
    PendingDataFeature(
        name="CASHFLOW_QUALITY",
        hypothesis="Cash-flow quality: operating cash flow that confirms reported earnings predicts durable outperformance.",
        required_fields=("cash_flow_statement: net_operate_cash_flow", "income_statement: net_profit"),
    ),
    PendingDataFeature(
        name="ACCRUALS",
        hypothesis="Accruals: earnings backed by accruals rather than cash reverse.",
        required_fields=("cash_flow_statement: net_operate_cash_flow", "income_statement: net_profit", "balance_sheet: total_assets"),
    ),
    PendingDataFeature(
        name="ASSET_GROWTH",
        hypothesis="Asset growth: aggressive balance-sheet expansion underperforms (investment factor).",
        required_fields=("balance_sheet: total_assets",),
    ),
    PendingDataFeature(
        name="EARNINGS_ACCEL",
        hypothesis="Earnings acceleration: the second difference of TTM profit growth predicts continued fundamental momentum.",
        required_fields=("income_statement: profit TTM history (derivable from fundamental_pit.profit_cum once scoped)",),
    ),
)
