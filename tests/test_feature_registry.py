"""T2-01 contracts: the feature registry.

Every vocabulary feature carries declarative metadata: its PIT data tier,
its empirical coverage on the fixed calibration slice, its correlation
cluster (features whose signals move together on the calibration slice),
and its deprecation status.  The registry is deterministic in its inputs
and versioned, so artifacts can cite exactly which registry generation
produced their numbers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ashare_model.feature_registry import (
    FEATURE_REGISTRY_VERSION,
    FeatureRecord,
    FeatureRegistry,
    PitLevel,
)
from ashare_model.semantic_cache import CalibrationSlice
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB


def _synthetic_tensor(seed: int = 3) -> np.ndarray:
    """``[feature, stock, date]`` tensor with two correlated families."""
    rng = np.random.default_rng(seed)
    n_stocks, n_dates = 30, 40
    tensor = np.zeros((len(FEATURE_NAMES), n_stocks, n_dates), dtype=np.float32)
    momentum = np.cumsum(rng.normal(0, 1, (n_stocks, n_dates)), axis=1)
    noise = rng.normal(0, 1, (n_stocks, n_dates))
    for i, name in enumerate(FEATURE_NAMES):
        if name in ("RET_1", "RET_5", "RET_10", "MOMENTUM_20", "MOMENTUM_60"):
            tensor[i] = momentum + 0.1 * noise
        else:
            tensor[i] = noise + i * 1e-6
    tensor[0, :1, :] = np.nan  # RET_1 has one missing stock row
    return tensor


@pytest.fixture()
def tensor() -> np.ndarray:
    return _synthetic_tensor()


@pytest.fixture()
def mask() -> np.ndarray:
    out = np.ones((30, 40), dtype=bool)
    out[10, :] = False  # one stock never eligible
    return out


def test_registry_covers_every_vocabulary_feature(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    records = registry.records()
    assert set(records) == set(FEATURE_NAMES)
    assert set(records) == set(FORMULA_VOCAB.feature_names)


def test_records_carry_all_metadata_fields(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    for name, record in registry.records().items():
        assert isinstance(record, FeatureRecord)
        assert record.name == name
        assert record.pit_level in PitLevel
        assert 0.0 <= record.coverage <= 1.0
        assert record.correlation_cluster
        assert isinstance(record.deprecated, bool)
        if record.deprecated:
            assert record.deprecation_reason


def test_pit_levels_follow_the_data_tier(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    records = registry.records()
    assert records["RET_1"].pit_level == PitLevel.PIT_DAILY
    assert records["ROE"].pit_level == PitLevel.PIT_FUNDAMENTAL
    assert records["PE_TTM"].pit_level == PitLevel.PIT_FUNDAMENTAL
    assert records["MARGIN_BALANCE_CHG"].pit_level == PitLevel.PIT_CAPITAL
    assert records["IND_REL_RET_5"].pit_level == PitLevel.SNAPSHOT
    assert records["NORTHBOUND_CHG"].pit_level == PitLevel.NEUTRAL


def test_deprecation_status(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    records = registry.records()
    assert records["NORTHBOUND_CHG"].deprecated
    assert "RET_20" not in records  # aliased away, not in the live vocabulary
    assert not records["RET_1"].deprecated


def test_coverage_reflects_missing_data(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    records = registry.records()
    # RET_1 loses one stock row; one stock is never eligible.
    assert records["RET_1"].coverage == pytest.approx(28 / 29, abs=1e-9)
    assert records["RET_5"].coverage == pytest.approx(1.0, abs=1e-9)


def test_correlation_clusters_are_deterministic_and_data_driven(tensor, mask):
    first = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    second = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    assert first.records() == second.records()
    records = first.records()
    # The momentum family is a single cluster; the noise features are not.
    momentum_cluster = records["RET_1"].correlation_cluster
    assert records["RET_5"].correlation_cluster == momentum_cluster
    assert records["RET_10"].correlation_cluster == momentum_cluster
    assert records["MOMENTUM_20"].correlation_cluster == momentum_cluster
    assert records["VOL_20"].correlation_cluster != momentum_cluster


def test_registry_serializes_and_roundtrips(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    payload = registry.to_dict()
    json.dumps(payload)  # must be JSON-serializable
    assert payload["version"] == FEATURE_REGISTRY_VERSION
    assert payload["slice"] == "dates:0:40"
    assert len(payload["features"]) == len(FEATURE_NAMES)


def test_registry_version_is_pinned():
    assert FEATURE_REGISTRY_VERSION == 1


def test_summary_reports_tiers_and_deprecations(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    summary = registry.summary()
    assert summary["n_features"] == len(FEATURE_NAMES)
    assert summary["deprecated"] == ["NORTHBOUND_CHG"]
    assert PitLevel.PIT_DAILY.value in summary["by_pit_level"]
    assert summary["n_clusters"] >= 1


def test_build_requires_aligned_inputs():
    with pytest.raises(ValueError):
        FeatureRegistry.build(
            np.zeros((3, 5, 5), dtype=np.float32),
            np.ones((5, 6), dtype=bool),
            calibration_slice=CalibrationSlice.of(5),
        )
