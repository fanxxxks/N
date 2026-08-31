"""Authored feature metadata (P7 D1, contract: docs/p7_maintainability_plan.md
§6.1).

Every ``FEATURE_NAMES`` member carries authored research metadata (PIT
availability rule, economic hypothesis, expected direction, semantic
type, promotion permission) while horizon / cost / depends_on are
*derived* from the existing single sources (P6 research domains,
``FACTOR_REGISTRY``) — no second authority.  The registry build fails
closed when any feature lacks authored metadata.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ashare_data.capital_flow import EXTERNAL_FACTOR_NAMES
from ashare_data.fundamentals import FUNDAMENTAL_PIT_NAMES
from ashare_model.factors import FACTOR_REGISTRY
from ashare_model.feature_metadata import (
    FEATURE_METADATA,
    ComputeCost,
    RecommendedHorizon,
    SemanticType,
    authored_metadata_of,
)
from ashare_model.feature_registry import (
    FEATURE_REGISTRY_VERSION,
    FeatureRegistry,
)
from ashare_model.research_domain import RESEARCH_DOMAINS
from ashare_model.semantic_cache import CalibrationSlice
from ashare_model.vocab import FEATURE_NAMES

_DOMAIN_HORIZON = {
    "short_price_volume": RecommendedHorizon.SHORT,
    "medium_cross_section": RecommendedHorizon.MEDIUM,
    "slow_fundamental": RecommendedHorizon.SLOW,
}

# Reverse map from the P6 registry (the single authority for horizons).
_FEATURE_DOMAIN: dict[str, str] = {
    name: domain_id
    for domain_id, domain in RESEARCH_DOMAINS.items()
    for name in domain.features
}


def _synthetic_tensor(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_stocks, n_dates = 30, 40
    tensor = rng.normal(0, 1, (len(FEATURE_NAMES), n_stocks, n_dates))
    return tensor.astype(np.float32)


def _build_registry():
    mask = np.ones((30, 40), dtype=bool)
    return FeatureRegistry.build(
        _synthetic_tensor(),
        mask,
        calibration_slice=CalibrationSlice.of(40),
    )


def test_authored_metadata_covers_every_feature():
    assert set(FEATURE_METADATA) == set(FEATURE_NAMES)


def test_authored_fields_are_informative():
    for name in FEATURE_NAMES:
        meta = authored_metadata_of(name)
        assert meta.availability_rule, name
        assert meta.hypothesis, name
        assert meta.expected_direction in (-1, 0, 1), name
        assert isinstance(meta.semantic_type, SemanticType), name
        assert isinstance(meta.promotion_allowed, bool), name


def test_deprecated_feature_cannot_promote():
    meta = authored_metadata_of("NORTHBOUND_CHG")
    assert meta.promotion_allowed is False


def test_fundamental_and_external_tiers_match_source_lists():
    for name in FUNDAMENTAL_PIT_NAMES:
        assert FEATURE_METADATA[name].semantic_type == SemanticType.FUNDAMENTAL_LIKE
    for name in ("MARGIN_BALANCE_CHG", "INDUSTRY_MOMENTUM"):
        assert name in FEATURE_METADATA
        assert name in EXTERNAL_FACTOR_NAMES


def test_registry_version_is_pinned():
    # v3 (P7 D1): records additionally carry authored research metadata
    # (availability, hypothesis, direction, semantic type, promotion
    # permission) plus horizon/cost/depends_on derived from the P6 domain
    # registry and FACTOR_REGISTRY (contract: plan §6.1).  Descriptive
    # only: no search/scoring/promotion semantics change.
    # v4 (P9, docs/p9_factor_family_contract.md APPROVED 2026-09-01):
    # whitelist §10.1 case 2.
    assert FEATURE_REGISTRY_VERSION == 4


def test_expected_horizon_derives_from_research_domains():
    registry = _build_registry()
    for name, record in registry.records().items():
        domain_id = _FEATURE_DOMAIN.get(name)
        if domain_id is None:
            # Only the deprecated neutral feature sits outside every domain.
            assert record.deprecated, name
            assert record.expected_horizon is None
        else:
            assert record.expected_horizon == _DOMAIN_HORIZON[domain_id], name


def test_compute_cost_and_depends_on_derive_from_factor_registry():
    registry = _build_registry()
    records = registry.records()
    for name, record in records.items():
        assert isinstance(record.compute_cost, ComputeCost), name
        assert isinstance(record.depends_on, tuple), name
        if name in FACTOR_REGISTRY:
            spec = FACTOR_REGISTRY[name][0]
            warmup = spec.warmup
            expected_cost = (
                ComputeCost.LOW
                if warmup <= 5
                else ComputeCost.MEDIUM
                if warmup <= 30
                else ComputeCost.HIGH
            )
            assert record.compute_cost == expected_cost, name
            expected_deps = tuple(spec.required_columns)
            if spec.family == "industry":
                expected_deps = expected_deps + ("industry_membership_snapshot",)
            assert record.depends_on == expected_deps, name
        elif name == "NORTHBOUND_CHG":
            assert record.depends_on == ()
        else:
            assert record.depends_on, name


def test_to_dict_and_summary_carry_new_fields():
    registry = _build_registry()
    payload = registry.to_dict()
    json.dumps(payload)  # JSON-serializable
    by_name = {f["name"]: f for f in payload["features"]}
    row = by_name["ROE"]
    for key in (
        "availability_rule",
        "hypothesis",
        "expected_direction",
        "semantic_type",
        "expected_horizon",
        "promotion_allowed",
        "compute_cost",
        "depends_on",
    ):
        assert key in row, key
    assert row["semantic_type"] == SemanticType.FUNDAMENTAL_LIKE.value
    assert row["expected_horizon"] == RecommendedHorizon.SLOW.value
    assert row["expected_direction"] == 1
    summary = registry.summary()
    assert sum(summary["by_semantic_type"].values()) == len(FEATURE_NAMES)


def test_missing_authored_metadata_fails_closed():
    saved = dict(FEATURE_METADATA)
    try:
        FEATURE_METADATA.pop("ROE")
        # build() re-reads the module-level table; the pop above must make
        # any build fail loudly instead of silently defaulting.
        with pytest.raises(ValueError, match="ROE"):
            FeatureRegistry.build(
                _synthetic_tensor(),
                np.ones((30, 40), dtype=bool),
                calibration_slice=CalibrationSlice.of(40),
            )
    finally:
        FEATURE_METADATA.clear()
        FEATURE_METADATA.update(saved)


def test_semantic_type_counts_match_authoring():
    # The six-type lattice is fully used (contract §6.1 semantic table).
    used = {meta.semantic_type for meta in FEATURE_METADATA.values()}
    assert used == set(SemanticType)


def test_p9_new_features_have_authored_metadata():
    """P9 §5: every v4 feature carries authored research metadata
    (fail-closed; FeatureRegistry.build must never see a gap)."""
    p9_new = (
        "IND_REL_RET_60", "IND_REL_RET_120", "LIQ_SHOCK_20",
        "VOLUME_SHRINK_5_20", "PV_DIV_20", "LIMIT_UP_CNT_5",
        "LIMIT_DOWN_STREAK", "LIMIT_BREAK_5", "CROWD_TURNOVER_60",
        "CROWD_AMOUNT_60", "MARGIN_CROWD_60",
    )
    for name in p9_new:
        meta = FEATURE_METADATA[name]
        assert meta.availability_rule, name
        assert meta.hypothesis, name
        assert meta.expected_direction in (-1, 0, 1), name


def test_p9_pending_data_placeholders_stay_outside_the_vocabulary():
    from ashare_model.feature_metadata import PENDING_DATA_FEATURES
    from ashare_model.vocab import FEATURE_NAMES

    names = {entry.name for entry in PENDING_DATA_FEATURES}
    assert names == {
        "CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL",
    }
    for entry in PENDING_DATA_FEATURES:
        assert not entry.promotion_allowed
        assert entry.name not in FEATURE_METADATA
        assert entry.name not in FEATURE_NAMES
        assert entry.required_fields
