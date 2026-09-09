"""Research domains split by prediction horizon (P6).

Contract: ``docs/p6_research_domain_contract.md``.  The pre-P6 pipeline
searched every vocabulary feature against one ``daily / horizon=1`` target
with one reward and one turnover constraint — a semantic mix of slow
fundamentals, 20-120 day momentum and intraday/overnight features.  A
research domain fixes, per run:

* the **feature set** it owns (the live vocabulary features are
  partitioned exhaustively and disjointly across the three domains;
  deprecated ``NORTHBOUND_CHG`` belongs to none);
* the **target horizon intent** (``horizon_range``) and the **execution
  cadence** (``frequencies``), whose legal combinations are validated
  against the P3 non-overlapping-label rule;
* the **per-domain reward parameters** (``turnover_budget``,
  ``cost_weight``) — domains never share a reward semantic or a turnover
  constraint, and cross-domain rewards are not comparable;
* the default **baseline ladder** (``baseline_signals`` ⊆ features).

``unified`` is the reserved pre-P6 compatible semantic: no defaults are
applied, behavior is byte-identical, and artifacts record
``research_domain: "unified"``.

IP-12 (01-A2): the pure data part (registry table, feature partition,
legal execution points) moved to the shared bottom-tier module
``ashare_domain`` so ``ashare_data.config`` no longer needs a reverse
``ashare_model`` import.  This module remains the contract-named home and
the single owner of ``RESEARCH_DOMAIN_VERSION`` (runspec's version index
and the registry doc generation read it here); everything the data home
defines is re-exported by identity, so the consumer surface is unchanged.
"""

from __future__ import annotations

import numpy as np

from ashare_domain import (
    RESEARCH_DOMAINS,
    ResearchDomain,
    UNIFIED_DOMAIN_ID,
    domain_of_feature,
    resolve_domain,
)
from .vocab import FORMULA_VOCAB

# Bump when the registry schema, a domain's semantics (features, horizons,
# cadences, baselines) or the per-domain reward/turnover defaults change.
# v2 (P9): the eleven orthogonal family features join their domains
# (docs/p9_factor_family_contract.md §5, APPROVED 2026-09-01).
# v3 (P13 §5.5): the four family-⑤ features join slow_fundamental
# (docs/p13_fundamental_fields_contract.md, APPROVED 2026-09-02).
# Home-address migration only (IP-12): the version stayed pinned here
# while the data definitions moved to ``ashare_domain`` — no bump.
RESEARCH_DOMAIN_VERSION = 5

#: p21 §8 (APPROVED 2026-09-06): preregistered tier sets for run-level
#: feature restriction.  Values mirror the P2 §6 collections "A" (仅 A) and
#: "AB" (A+B); tier C is never a run-level restriction target here (the
#: p14 promotion track covers the tier-A promotion path separately).
TIER_SETS: dict[str, tuple[str, ...]] = {"A": ("A",), "AB": ("A", "B")}

__all__ = [
    "RESEARCH_DOMAIN_VERSION",
    "RESEARCH_DOMAINS",
    "ResearchDomain",
    "TIER_SETS",
    "UNIFIED_DOMAIN_ID",
    "domain_of_feature",
    "feature_token_ids",
    "resolve_domain",
    "restrict_tensor",
    "tier_restricted_features",
]


def _resolve_tier_set(tier_set: str | None) -> tuple[str, ...] | None:
    """Validate a preregistered tier-set value against p21 §8."""

    if tier_set is None:
        return None
    key = str(tier_set)
    if key not in TIER_SETS:
        raise ValueError(
            f"invalid tier_set {tier_set!r}: expected one of "
            f"{sorted(TIER_SETS)} or None (docs/"
            "p21_r1_domain_return_research_preregistration.md §8)"
        )
    return TIER_SETS[key]


def tier_restricted_features(
    domain_id: str, tier_set: str | None = None, vocab=None
) -> list[str]:
    """The run's preregistered feature set (p21 §8.2), in vocab order.

    Domain features (the ``ashare_domain`` registry — the single feature
    partition authority), intersected — when a ``tier_set`` is given —
    with the tier set (``data_tier.feature_tier`` — the single tier
    authority) minus the vocabulary's deprecated names (never sampleable,
    grammar v4+ rule).  ``unified`` never combines with a tier set (p21
    §8.1: the compatible semantic stays untouched).  ``tier_set=None``
    resolves to the full domain feature list — today's behavior,
    byte-identical (deprecated members keep their rows and are excluded
    from sampling downstream by the samplers' own deprecated mask).
    """

    vocab = vocab or FORMULA_VOCAB
    allowed_tiers = _resolve_tier_set(tier_set)
    domain_id = str(domain_id)
    if domain_id == UNIFIED_DOMAIN_ID:
        if allowed_tiers is not None:
            raise ValueError(
                "tier_set requires a research domain; the unified "
                "compatible semantic has no tier restriction (p21 §8.1)"
            )
        return list(vocab.feature_names)
    domain = resolve_domain(domain_id)
    owned = set(domain.features)
    if allowed_tiers is None:
        return [name for name in vocab.feature_names if name in owned]
    # Lazy import: data_tier pulls feature_registry → feature_metadata,
    # which imports this module back — a module-level edge would close a
    # cycle (the same lazy pattern feature_registry uses for
    # PENDING_DATA_FEATURES).
    from .data_tier import feature_tier

    deprecated = vocab.deprecated_names
    return [
        name
        for name in vocab.feature_names
        if name in owned
        and name not in deprecated
        and feature_tier(name).value in allowed_tiers
    ]


def restrict_tensor(
    tensor, domain_id: str, vocab=None, *, tier_set: str | None = None
) -> np.ndarray:
    """Domain-restricted factor tensor: out-of-domain rows become neutral.

    Returns a float32 copy with the same ``[feature, stock, date]`` shape
    and the same row order (global token ids stay valid); rows of features
    outside the domain are zeroed (0 is the neutral value after
    cross-sectional standardization).  ``unified`` returns an identical
    copy.  Row count must equal the vocabulary's feature count.

    With a preregistered ``tier_set`` (p21 §8.2) the restriction set is
    ``tier_restricted_features`` instead of the full domain: rows outside
    domain ∩ tier − deprecated are zeroed.  ``tier_set=None`` keeps the
    pre-p21 behavior byte-identical.
    """

    vocab = vocab or FORMULA_VOCAB
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"factor tensor must be 3-D, got {arr.ndim}-D")
    if arr.shape[0] != len(vocab.feature_names):
        raise ValueError(
            f"factor tensor has {arr.shape[0]} rows but the vocabulary "
            f"has {len(vocab.feature_names)} features"
        )
    domain_id = str(domain_id)
    if domain_id == UNIFIED_DOMAIN_ID:
        if tier_set is not None:
            raise ValueError(
                "tier_set requires a research domain; the unified "
                "compatible semantic has no tier restriction (p21 §8.1)"
            )
        return arr.copy()
    allowed = set(
        tier_restricted_features(domain_id, tier_set, vocab=vocab)
    )
    out = arr.copy()
    for index, name in enumerate(vocab.feature_names):
        if name not in allowed:
            out[index] = 0.0
    return out


def feature_token_ids(
    domain_id: str, vocab=None, *, tier_set: str | None = None
) -> list[int] | None:
    """Global vocab token ids of a domain's features, in vocab order.

    ``None`` for ``unified`` (no restriction).  Searchers pass this to
    their sampling mask / primitive set so only in-domain features can
    enter formulas (P6 §4.2).  With a preregistered ``tier_set`` (p21
    §8.2) the ids are the resolved (domain ∩ tier − deprecated) set;
    ``tier_set=None`` keeps today's full-domain behavior (deprecated
    features stay excluded downstream by the samplers' own deprecated
    mask).
    """

    vocab = vocab or FORMULA_VOCAB
    domain_id = str(domain_id)
    if domain_id == UNIFIED_DOMAIN_ID:
        if tier_set is not None:
            raise ValueError(
                "tier_set requires a research domain; the unified "
                "compatible semantic has no tier restriction (p21 §8.1)"
            )
        return None
    names = tier_restricted_features(domain_id, tier_set, vocab=vocab)
    return [
        vocab.feature_offset + vocab.feature_names.index(name)
        for name in names
    ]
