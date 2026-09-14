"""Authored research metadata for every vocabulary feature (P7 D1).

Contract: ``docs/p7_maintainability_plan.md`` §6.1.  This module is the
single home for the metadata that cannot be *derived* from an existing
authority — the PIT availability rule, the economic hypothesis, the
expected direction, the semantic type, the structured unit/NaN
declaration (p28 I6) and the promotion permission.
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


class NaNSemantics(str, enum.Enum):
    """Missing-value semantics of a feature's raw factor frame (p28 I6).

    ``MISSING_NAN``: a missing source value stays NaN in the raw frame —
    it never becomes a fake 0 before cross-sectional standardization
    (p24 §2 NaN guards; the engine's post-standardization 0 is the
    uniform neutral convention, not a per-feature degradation).
    ``NEUTRAL_PLACEHOLDER``: the feature has no live source at all and
    stays a declared neutral 0 (the vocabulary-stability placeholders).
    """

    MISSING_NAN = "missing_nan"
    NEUTRAL_PLACEHOLDER = "neutral_placeholder"


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
_FUNDAMENTAL_TTM = (
    "quarterly report, visible after the statutory disclosure season ends "
    "(Q1 04-30 / H1 08-31 / Q3 10-31 / annual next-year 04-30); TTM ratios "
    "require four complete consecutive quarters (docs/p13_fundamental_fields_contract.md §5.3)"
)
_MARGIN = (
    "v2 margin_detail_pit (p17 §5.2 table 18): the detail of session t-1 "
    "becomes visible at session t (p17 §6.1 T+1 availability)"
)
# A04 (p28 I2, F04): the JOINT visibility of the F2 margin-crowding
# factors — two sources on two clocks, and a value exists only when BOTH
# are visible.  Never the quarterly-TTM template; never one half alone.
_MARGIN_X_BASIC = (
    "joint visibility of two sources: the financing detail of session t-1 "
    "becomes visible at session t (v2 margin_detail_pit available_date, "
    "p17 §6.1 T+1) AND the float-cap denominator from the same-session "
    "daily_basic_pit close (trade_date-keyed; the session-t basic values "
    "are known at the session-t close, the D4-era daily-basic convention); "
    "a value is available at session t only when both sources are visible"
)
_SNAPSHOT_EXT = (
    "industry membership snapshot + daily bars up to session t inclusive "
    "(PIT-bounded by the universe contract)"
)
_NEUTRAL = "no live data source; the feature stays neutral (0)"


@dataclass(frozen=True)
class AuthoredFeatureMeta:
    """Authored metadata for one feature; ``None`` cost/depends_on means
    "derive from ``FACTOR_REGISTRY``" (see :func:`resolve_cost` /
    :func:`resolve_depends_on`).

    ``unit`` / ``nan_policy`` (p28 I6): the structured unit declaration
    (a short controlled economic-dimension token; the per-source scale
    chain is audited against p17 §5.2 in the p28 measurement log) and the
    missing-value semantics.  Both are required — a feature added without
    them fails at import, not at audit time.
    """

    availability_rule: str
    hypothesis: str
    expected_direction: int
    semantic_type: SemanticType
    unit: str
    nan_policy: NaNSemantics
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
        deps = deps + ("industry_membership_pit",)
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
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "RET_5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=6),
        hypothesis="One-week reversal: recent losers rebound as the overreaction unwinds.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: mirror of REVERSAL_5
    ),
    "RET_10": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=11),
        hypothesis="Two-week reversal: extended-horizon overreaction correction.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: covered by RET_1 + BIAS_20
    ),
    "VOL_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Low-vol anomaly: lower short-term volatility predicts higher risk-adjusted returns.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "VOL_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Low-vol anomaly at the quarterly horizon.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: 0.907 correlated with IVOL_60
    ),
    "TURNOVER": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Hot-money crowding: high turnover marks attention-driven overpricing that later underperforms.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="pct",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "TURNOVER_CHG": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Turnover spikes mark attention peaks that fade.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "VOLUME_RATIO": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Abnormal volume signals speculative interest that mean-reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: 0.977 correlated with VOLUME_IMPACT
    ),
    "VOLUME_IMPACT": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Log-volume deviation proxies abnormal trading interest; elevated values fade.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="log_volume",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "AMPLITUDE": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Wide intraday ranges reflect disagreement and instability; predictive of underperformance.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "CLOSE_POSITION": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Strong closes reflect intraday accumulation; short-horizon continuation.",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "MOMENTUM_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=21),
        hypothesis="One-month momentum: modest continuation at the medium horizon.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "MOMENTUM_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=61),
        hypothesis="Quarterly momentum: trend persistence at the medium horizon.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: mirror of REVERSAL_60
    ),
    "REVERSAL_5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=6),
        hypothesis="Pre-flipped short-term reversal: buying recent losers captures the overreaction correction.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "SKEW_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Lottery preference: positively skewed names are overbid and underperform.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "KURT_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Tail thickness mixes crash-risk premium and lottery demand; no committed expectation.",
        expected_direction=0,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Fundamental PIT family (fundamental-like).
    "PE_TTM": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Value: low valuation (high earnings yield) earns a premium.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "PB": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Book-value anchor: low P/B predicts mispricing mean reversion.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "PS_TTM": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Sales-based value: low P/S marks cheaper cross-sections.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "ROE": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: profitable firms compound value and stay underpriced.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "ROA": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: asset efficiency signals durable profitability.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "GROSS_MARGIN": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: gross profitability proxies pricing power / moat.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "NET_MARGIN": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Quality: bottom-line efficiency.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "REVENUE_YOY": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Growth: top-line expansion predicts fundamental continuation.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "PROFIT_YOY": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Growth: earnings acceleration.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "DEBT_RATIO": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Leverage risk: high debt raises distress risk.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    "MARKET_CAP": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Size premium: smaller float caps earn higher risk-adjusted returns in A-shares.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="cap_proxy",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "DIVIDEND_YIELD": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL,
        hypothesis="Income/value: dividend payers are disciplined and cheaper.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("fundamental_pit",),
    ),
    # Neutral / external flows.
    "NORTHBOUND_CHG": AuthoredFeatureMeta(
        availability_rule=_NEUTRAL,
        hypothesis="Historical northbound flow change; the daily feed was discontinued (2024-08), so the feature stays neutral for vocabulary stability.",
        expected_direction=0,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="neutral",
        nan_policy=NaNSemantics.NEUTRAL_PLACEHOLDER,
        promotion_allowed=False,
        compute_cost=ComputeCost.LOW,
        depends_on=(),
    ),
    "MARGIN_BALANCE_CHG": AuthoredFeatureMeta(
        availability_rule=_MARGIN,
        # t70 (p20 draft, captain+governance double-signed): the authored
        # +1 "inflows chase and lift prices" was falsified in four audit
        # rounds — the direction re-preregisters to -1 (leveraged-inflow
        # crowding economics, consistent with MARGIN_CROWD_60); the
        # formula, source and PIT visibility are unchanged.
        hypothesis="Margin-flow crowding: leveraged-inflow acceleration marks late-stage chase that subsequently underperforms.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        # A04 (p28 I4): the margin family's depends_on convention is
        # table.field (one batch, one convention; the p17 §5.2 source
        # table and the consumed field are both named).
        depends_on=("margin_detail_pit.rzye",),
    ),
    # Limit-move event family (boolean/event).
    "LIMIT_UP_EVENT": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="One-word limit-up marks extreme attention; next-session squeeze continuation.",
        expected_direction=1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="boolean",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "LIMIT_DOWN_EVENT": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="One-word limit-down marks a forced-selling overhang; continued weakness.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="boolean",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "INDUSTRY_MOMENTUM": AuthoredFeatureMeta(
        availability_rule=_SNAPSHOT_EXT,
        # Dual-finding record (captain final arbitration 2026-09-07):
        # (1) the authored +1 (spillover) is FALSIFIED on the repaired
        # dataset across two post-fix rounds (IC negative at every
        # horizon, corrected-baseline dIC -0.0167, annual direction
        # stability 4/13); (2) the -1 reversal (industry-level reversal)
        # was empirically OBSERVED (the -1-aligned IC positive at every
        # horizon) but its re-hypothesis was REJECTED on the redundancy
        # pre-check (the corrected dIC -0.0167 sits below every IND_REL_*
        # family member's increment +0.0079..+0.0283) — a direction
        # confirmation does not cure the increment deficiency, so the
        # -1 re-hypothesis was NOT adopted.  The direction field keeps the
        # authored original value (the pre-registered-hypothesis history
        # semantics); promotion eligibility stays withdrawn; a future
        # family-structure change may re-open via a NEW pre-registration.
        hypothesis="Industry momentum spillover: hot industry groups keep rotating slowly.",
        expected_direction=1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        promotion_allowed=False,
        depends_on=("industry_membership_pit", "close"),
    ),
    # F1 batch (p23 v4, user-approved via the captain's delegated ruling):
    # the four slow-fundamental quality factors (asset-yield ratios and the
    # leverage trend) — all as-of computed from the v2 version tables via
    # the family-5 single path.  OPERATING_PROFITABILITY's pre-registered
    # proxy: total_profit stands in for operate_profit (absent from the
    # p17 income DDL; an erratum + recompute if the column is added).
    "CFO_TO_ASSETS": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="CFO-to-assets: operating cash flow per unit of average total assets marks cash-creation intensity and financial flexibility.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("n_cashflow_act", "total_assets"),
    ),
    "GROSS_PROFITABILITY": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="Gross profitability (Novy-Marx): gross profit per unit of average total assets marks pricing power and cost-structure quality.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("revenue", "oper_cost", "total_assets"),
    ),
    "OPERATING_PROFITABILITY": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="Operating profitability: total profit per unit of average total assets marks comprehensive earning power.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("total_profit", "total_assets"),
    ),
    "LEVERAGE_CHANGE": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="Leverage trend: a 4-report-period rise in the debt ratio marks distress-driven underperformance.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("total_liab", "total_assets"),
    ),
    # F2 batch (p24, user-approved): the two margin-crowding factors —
    # the raw ratios computed in capital_flow (ashare_data) with the
    # per-date cross-sectional z applied loader-side (the dependency-
    # correct layer for the single xcs-z implementation; p24 §2).  A04
    # (p28 I1-I4, t6 Ruling 2, F04): the metadata now states the real
    # source semantics — the JOINT margin-T+1 × daily_basic-same-session
    # visibility (never the quarterly-TTM template), the
    # CROSS_SECTIONAL_SIGNAL type (the loader-side output IS the per-date
    # cross-sectional z of the raw ratios, so "only meaningful vs peers"
    # applies; this unlocks CS_NEUTRALIZE — the measured search-space
    # delta is in the p28 measurement log), and the table.field depends_on
    # convention (p17 §5.2 units: rzye/rzmre = yuan, circ_mv = wan_yuan;
    # the constant 1e4 scale is absorbed by the scale-invariant xcs-z).
    "MARGIN_BALANCE_TO_CIRC_MV": AuthoredFeatureMeta(
        availability_rule=_MARGIN_X_BASIC,
        hypothesis="Margin balance to float cap: a high leveraged-position intensity relative to the free float marks crowding that unwinds.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("margin_detail_pit.rzye", "daily_basic_pit.circ_mv"),
    ),
    "MARGIN_BUY_PRESSURE_5": AuthoredFeatureMeta(
        availability_rule=_MARGIN_X_BASIC,
        hypothesis="Margin buy pressure: heavy 5-session margin buying per unit of float cap marks leveraged chasing that subsequently retraces (the -1 main hypothesis; the +1 chase-continuation alternative adjudicated by the measurement).",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("margin_detail_pit.rzmre", "daily_basic_pit.circ_mv"),
    ),
    # Intraday decomposition (return-like).
    "OVERNIGHT_RET": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Overnight gap captures order-imbalance accumulation; short-horizon continuation.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "INTRADAY_RET": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Intraday return reflects day-session speculation; tends to mean-revert.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Liquidity (volume-like).
    "ILLIQ_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Illiquidity premium: less liquid names earn a premium.",
        expected_direction=1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="per_amount",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "AMOUNT_SHARE": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Crowding: stocks absorbing disproportionate market turnover underperform.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Lottery / anchoring.
    "MAX_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="MAX lottery effect: recent extreme daily gains are overbid and underperform.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "HIGH_52W": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=250),
        hypothesis="52-week-high anchor: proximity to the high predicts continuation.",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Market-model risk (cross-sectional by construction).
    "BETA_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Low-beta anomaly within the cross-section (betting-against-beta).",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "IVOL_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Idiosyncratic-volatility puzzle: high IVOL names are overpriced (lottery demand).",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "RSQ_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Market co-movement: low R-squared names are harder to arbitrage; the premium is contested.",
        expected_direction=0,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Technical (defined per FactorSpec.description).
    "BIAS_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="MA-distance extension: overextension mean-reverts at the cross-section.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "RSI_14": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=14),
        hypothesis="Overbought/oversold oscillator: high RSI mean-reverts across stocks.",
        expected_direction=-1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="score_0_100",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "ATR_14": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=14),
        hypothesis="True-range volatility proxy: instability trades like the low-vol anomaly.",
        expected_direction=-1,
        semantic_type=SemanticType.PRICE_LIKE,
        unit="price_adj",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "MACD_DIF": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=26),
        hypothesis="Trend oscillator: positive DIF marks uptrend persistence (momentum).",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
        unit="price_adj",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "MACD_DEA": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=26),
        hypothesis="Trend signal line: smoothed trend confirmation.",
        expected_direction=1,
        semantic_type=SemanticType.PRICE_LIKE,
        unit="price_adj",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: 0.926 correlated with MACD_DIF
    ),
    # Microstructure.
    "SUSPEND_DAYS_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Frequent suspensions signal distress or trading games; penalized.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="count",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "LIST_AGE": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=1),
        hypothesis="Seasoning: older listings shed the IPO speculation halo; mature names behave better.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="sessions",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Medium-term momentum / reversal (return-like).
    "RET_120": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=121),
        hypothesis="Six-month momentum: intermediate-horizon trend persistence.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: mirror of REVERSAL_120
    ),
    "REVERSAL_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=61),
        hypothesis="Pre-flipped medium-term reversal: 60-day losers rebound.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "REVERSAL_120": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=121),
        hypothesis="Pre-flipped long-horizon reversal: 120-day losers rebound.",
        expected_direction=1,
        semantic_type=SemanticType.RETURN_LIKE,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Smoothed turnover (volume-like).
    "TURNOVER_MA5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=5),
        hypothesis="Smoothed crowding: elevated turnover predicts underperformance.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="pct",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §4.1: TURNOVER + TURNOVER_MA20 cover it
    ),
    "TURNOVER_MA20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Monthly crowding level.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="pct",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # t22/p9 erratum (t69) Part A: 0.914 vs TURNOVER_STD20 across four rounds
    ),
    "TURNOVER_STD20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Turnover volatility: unstable attention marks speculation.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="pct",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Limit-move statistics (event-derived).
    "LIMIT_STREAK": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Consecutive one-word limit-ups: speculative streaks reverse after the run.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="count",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §7.3 conditional deprecation TRIGGERED (0.980)
    ),
    "LIMIT_UP_CNT_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Repeated limit-ups mark a lottery stock; underperforms subsequently.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="count",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "LIMIT_BREAK": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Touched limit-up but failed to hold: failed speculation, weak follow-through.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="boolean",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Industry-relative cross-section.
    "IND_REL_RET_5": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=6),
        hypothesis="Industry-neutral short reversal: stock-specific overreaction vs peers reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "IND_REL_RET_20": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=21),
        hypothesis="Industry-neutral monthly move: stock-specific deviation vs peers reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "IND_REL_VOL_20": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=20),
        hypothesis="Stock-specific volatility vs the industry: idiosyncratic risk is overpriced.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "IND_REL_TURNOVER": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=1),
        hypothesis="Crowding relative to industry peers.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="pct",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # --- P9 v4 additions (docs/p9_factor_family_contract.md §5, APPROVED) ---
    # Family ①: industry-residualized medium-horizon momentum/reversal.
    "IND_REL_RET_60": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=61),
        hypothesis="Industry-neutral quarterly reversal: stock-specific deviation vs peers reverts at the medium horizon.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "IND_REL_RET_120": AuthoredFeatureMeta(
        availability_rule=_INDUSTRY_REL.format(warmup=121),
        hypothesis="Industry-neutral semi-annual reversal: stock-specific deviation vs peers reverts.",
        expected_direction=-1,
        semantic_type=SemanticType.CROSS_SECTIONAL_SIGNAL,
        unit="return",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    # Family ②: liquidity shock, volume shrinkage, price-volume divergence.
    "LIQ_SHOCK_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Liquidity shock: a one-day surge in market share marks attention-driven overpricing that reverses.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §7.1 second-pass: 0.968 vs VOLUME_IMPACT
    ),
    "VOLUME_SHRINK_5_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Volume shrinkage precedes exhaustion of the prevailing move; shrinking volume flips the sign of the drift.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "PV_DIV_20": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=20),
        hypothesis="Price-volume divergence: returns unconfirmed by volume mean-revert.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
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
        unit="count",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §7: family ③ negative result
    ),
    "LIMIT_DOWN_STREAK": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=2),
        hypothesis="Consecutive one-word limit-downs: forced-selling overhang continues over the streak.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="count",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §7: family ③ negative result
    ),
    "LIMIT_BREAK_5": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=5),
        hypothesis="Repeated failed limit-ups (炸板): speculative churn, weak follow-through.",
        expected_direction=-1,
        semantic_type=SemanticType.BOOLEAN_EVENT_SIGNAL,
        unit="count",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §7: family ③ negative result
    ),
    # Family ④: per-stock crowding.
    "CROWD_TURNOVER_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Self-relative turnover crowding: turnover far above the stock's own baseline marks overpriced attention.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        promotion_allowed=False,  # P9 §7.1 second-pass: 0.971 vs CROWD_AMOUNT_60
    ),
    "CROWD_AMOUNT_60": AuthoredFeatureMeta(
        availability_rule=_BAR.format(warmup=60),
        hypothesis="Self-relative amount-share crowding: absorbing more market share than usual precedes underperformance.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
    ),
    "MARGIN_CROWD_60": AuthoredFeatureMeta(
        availability_rule=_MARGIN,
        hypothesis="Margin crowding: a financing balance far above its own baseline marks leveraged crowding that unwinds.",
        expected_direction=-1,
        semantic_type=SemanticType.VOLUME_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        # A04 (p28 I4): table.field convention, same as the margin family.
        depends_on=("margin_detail_pit.rzye",),
    ),
    # Family ⑤ (P13 §5.3, docs/p13_fundamental_fields_contract.md): the
    # pending_data placeholders become authoritative metadata -- same
    # pre-registered hypotheses, real availability rules, and depends_on
    # covering the fundamental_pit fields each formula consumes.  The
    # family joins/leaves promotion as one atomic unit (§8 adjudication).
    "CASHFLOW_QUALITY": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="Cash-flow quality: operating cash flow that confirms reported earnings predicts durable outperformance.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("net_operate_cash_flow", "profit_cum"),
    ),
    "ACCRUALS": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="Accruals: earnings backed by accruals rather than cash reverse.",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("net_operate_cash_flow", "profit_cum", "total_assets"),
    ),
    "ASSET_GROWTH": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="Asset growth: aggressive balance-sheet expansion underperforms (investment factor).",
        expected_direction=-1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("total_assets",),
    ),
    "EARNINGS_ACCEL": AuthoredFeatureMeta(
        availability_rule=_FUNDAMENTAL_TTM,
        hypothesis="Earnings acceleration: the second difference of TTM profit growth predicts continued fundamental momentum.",
        expected_direction=1,
        semantic_type=SemanticType.FUNDAMENTAL_LIKE,
        unit="ratio",
        nan_policy=NaNSemantics.MISSING_NAN,
        compute_cost=ComputeCost.LOW,
        depends_on=("profit_cum",),
    ),
}

# Structural invariants checked at import: the table must cover exactly the
# live vocabulary, and the neutral set must not collide with the external
# feed list (a drift here means a feature was added without metadata).
assert set(FEATURE_METADATA) == set(FEATURE_NAMES), (
    "FEATURE_METADATA must cover every FEATURE_NAMES member"
)
assert not (set(NEUTRAL_FEATURE_NAMES) & set(EXTERNAL_FACTOR_NAMES))


# --- P9 §6: pending_data placeholders ---------------------------------------

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


# P13 §5.3 (docs/p13_fundamental_fields_contract.md): the four family-⑤
# placeholders (CASHFLOW_QUALITY, ACCRUALS, ASSET_GROWTH, EARNINGS_ACCEL)
# were lifted into FEATURE_METADATA as authoritative members once their
# PIT data fields were registered.  The documentation mechanism itself is
# retained for future data-gap families; it currently carries no members.
PENDING_DATA_FEATURES: tuple[PendingDataFeature, ...] = ()
