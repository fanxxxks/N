"""Free-data credibility tiers (P2): A / B / C and formula traceability.

Every vocabulary feature consumes one or more data sources; the weakest
source decides the feature's credibility tier (contract:
``docs/p2_data_tier_contract.md``):

* **Tier A** — official daily bars (price / volume / turnover) and the
  verified listing date / PIT membership records: usable as of the next
  session after the trading day, no paid data involved.
* **Tier B** — margin-financing balance (exchange daily feed) and
  point-in-time fundamentals aligned to the statutory disclosure-season
  end: usable from the recorded publication date, conservative by
  construction, no restatement tracking.
* **Tier C** — current-snapshot extrapolations (Shenwan industry
  membership), the same-day ST snapshot (no dated ST history exists) and
  placeholder data (``NORTHBOUND_CHG``): research display only, never a
  promotion input.

The mapping from the existing :class:`~ashare_model.feature_registry.PitLevel`
is the single source of truth; :func:`feature_tier` resolves a feature name,
:func:`formula_data_tier_report` traces a whole formula (token list or bare
baseline name) back to the tiers of its features, and :func:`tier_features`
selects the vocabulary of a tier set for diagnostics / ablation reports
(P2-05).  All sources are free AkShare endpoints; this module adds no
dependency and no paid-data path.
"""

from __future__ import annotations

import enum

from .feature_registry import PitLevel, pit_level_of
from .ir import decode, feature_names as ir_feature_names
from .vocab import FEATURE_NAMES

# Bump when the tier schema, the mapping or the time rules change.
DATA_TIER_VERSION = 1

# Human-readable usable-time rules, keyed by tier value.  Recorded verbatim
# in every tier-annotated artifact (P2-02).
TIER_TIME_RULES: dict[str, str] = {
    "A": (
        "值在交易日收盘即确定，自次日会话起可用；LIST_AGE 自首个可用 bar "
        "（上市日近似）起可用；PIT 成员资格自 in_date 起、按数据窗口封顶。"
    ),
    "B": (
        "基本面自法定披露季末日起可见（Q1->04-30、H1->08-31、Q3->10-31、"
        "年报->次年 04-30），前向填充、不追踪重述；两融余额自 feed 交易日"
        "起（保守取次日）可用。"
    ),
    "C": (
        "当前行业快照仅自同步日（当日事实）有效，禁止投射历史；ST 快照"
        "仅用于真实当日撮合；占位数据无可用信号。仅研究展示，永不进入"
        "晋级结果。"
    ),
}


class DataTier(str, enum.Enum):
    """Credibility tier of a feature's weakest data source."""

    A = "A"
    B = "B"
    C = "C"

    @classmethod
    def weakest(cls, tiers: list["DataTier"]) -> "DataTier":
        """The weakest tier in ``tiers`` (C weakest, A strongest)."""
        ordered = (cls.A, cls.B, cls.C)
        return max((t for t in tiers), key=ordered.index)


def tier_of_pit_level(level: PitLevel) -> DataTier:
    """Map a feature's PIT level (weakest data source) to its tier."""
    return {
        PitLevel.PIT_DAILY: DataTier.A,
        PitLevel.PIT_FUNDAMENTAL: DataTier.B,
        PitLevel.PIT_CAPITAL: DataTier.B,
        PitLevel.SNAPSHOT: DataTier.C,
        PitLevel.NEUTRAL: DataTier.C,
    }[level]


def feature_tier(name: str) -> DataTier:
    """Credibility tier of one vocabulary feature."""
    return tier_of_pit_level(pit_level_of(name))


def tier_features(tiers: tuple[DataTier, ...]) -> list[str]:
    """Features whose tier is one of ``tiers``, in vocabulary order."""
    wanted = set(tiers)
    return [name for name in FEATURE_NAMES if feature_tier(name) in wanted]


def formula_feature_names(tokens) -> list[str] | None:
    """Distinct feature names referenced by a formula's token list.

    Returns ``None`` when the token list is not a decodable formula.
    """

    try:
        ir = decode(tokens)
    except Exception:  # noqa: BLE001 - any structural failure = no formula.
        return None
    return sorted(ir_feature_names(ir))


def formula_data_tier_report(
    tokens=None,
    feature_name: str | None = None,
) -> dict | None:
    """Trace a formula back to the tiers of its features.

    ``tokens`` is the canonical postfix token list; ``feature_name`` covers
    bare-factor baseline rows (``formula=None``, ``formula_text=NAME``).
    Returns ``{"data_tier_version", "max_tier", "tiers_used",
    "per_feature"}`` or ``None`` when there is no traceable formula (no
    input, an undecodable token list, or a text that is not a feature).
    """

    if tokens is not None:
        names = formula_feature_names(tokens)
        if names is None:
            return None
    elif feature_name is not None:
        if feature_name not in FEATURE_NAMES:
            return None
        names = [feature_name]
    else:
        return None

    per_feature = {name: feature_tier(name).value for name in names}
    tiers = [DataTier(tier) for tier in per_feature.values()]
    return {
        "data_tier_version": DATA_TIER_VERSION,
        "max_tier": DataTier.weakest(tiers).value,
        "tiers_used": sorted({t.value for t in tiers}),
        "per_feature": per_feature,
    }
