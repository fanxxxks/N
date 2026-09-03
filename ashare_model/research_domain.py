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
RESEARCH_DOMAIN_VERSION = 3

__all__ = [
    "RESEARCH_DOMAIN_VERSION",
    "RESEARCH_DOMAINS",
    "ResearchDomain",
    "UNIFIED_DOMAIN_ID",
    "domain_of_feature",
    "feature_token_ids",
    "resolve_domain",
    "restrict_tensor",
]


def restrict_tensor(tensor, domain_id: str, vocab=None) -> np.ndarray:
    """Domain-restricted factor tensor: out-of-domain rows become neutral.

    Returns a float32 copy with the same ``[feature, stock, date]`` shape
    and the same row order (global token ids stay valid); rows of features
    outside the domain are zeroed (0 is the neutral value after
    cross-sectional standardization).  ``unified`` returns an identical
    copy.  Row count must equal the vocabulary's feature count.
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
        return arr.copy()
    domain = resolve_domain(domain_id)
    owned = set(domain.features)
    out = arr.copy()
    for index, name in enumerate(vocab.feature_names):
        if name not in owned:
            out[index] = 0.0
    return out


def feature_token_ids(domain_id: str, vocab=None) -> list[int] | None:
    """Global vocab token ids of a domain's features, in vocab order.

    ``None`` for ``unified`` (no restriction).  Searchers pass this to
    their sampling mask / primitive set so only in-domain features can
    enter formulas (P6 §4.2).
    """

    vocab = vocab or FORMULA_VOCAB
    domain_id = str(domain_id)
    if domain_id == UNIFIED_DOMAIN_ID:
        return None
    domain = resolve_domain(domain_id)
    return [
        vocab.feature_offset + index
        for index, name in enumerate(vocab.feature_names)
        if name in domain.features
    ]
