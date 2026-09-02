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
    # v3 (P7 D1): records additionally carry authored research metadata
    # (availability rule, economic hypothesis, expected direction,
    # semantic type, promotion permission) plus the horizon/cost/
    # depends_on triple derived from the P6 domain registry and
    # FACTOR_REGISTRY.  Descriptive only -- no search, scoring or
    # promotion semantics change (contract: docs/p7_maintainability_plan.md
    # §6.1, which supersedes the pinned v2 assertion on the requirement-
    # change path).  v5 (P9 §7 adjudication, measurement log
    # 2026-09-01).  v6 (P13 §5.3/§6.1): family ⑤ unlocks — whitelist
    # §10.1 case 2, pre-registered in docs/p13_fundamental_fields_contract.md.
    assert FEATURE_REGISTRY_VERSION == 6


def test_registry_records_data_tiers(tensor, mask):
    from ashare_model.data_tier import DATA_TIER_VERSION, DataTier

    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    payload = registry.to_dict()
    assert payload["data_tier_version"] == DATA_TIER_VERSION
    by_name = {f["name"]: f for f in payload["features"]}
    assert by_name["RET_1"]["data_tier"] == DataTier.A.value
    assert by_name["ROE"]["data_tier"] == DataTier.B.value
    assert by_name["MARGIN_BALANCE_CHG"]["data_tier"] == DataTier.B.value
    assert by_name["INDUSTRY_MOMENTUM"]["data_tier"] == DataTier.C.value
    assert by_name["NORTHBOUND_CHG"]["data_tier"] == DataTier.C.value
    summary = registry.summary()
    assert set(summary["by_data_tier"]) == {t.value for t in DataTier}
    assert sum(summary["by_data_tier"].values()) == len(FEATURE_NAMES)


def test_summary_reports_tiers_and_deprecations(tensor, mask):
    registry = FeatureRegistry.build(
        tensor, mask, calibration_slice=CalibrationSlice.of(40)
    )
    summary = registry.summary()
    assert summary["n_features"] == len(FEATURE_NAMES)
    # P9 §4.1 + §7 adjudication (whitelist §10.1 case 2, contract
    # APPROVED): the eight window variants, the triggered conditional
    # LIMIT_STREAK and the two second-pass consolidations join
    # NORTHBOUND_CHG in the deprecated set.
    assert summary["deprecated"] == sorted(
        [
            "NORTHBOUND_CHG",
            "RET_5", "RET_10", "RET_120", "MOMENTUM_60",
            "VOLUME_RATIO", "TURNOVER_MA5", "MACD_DEA", "VOL_60",
            "LIMIT_STREAK", "LIQ_SHOCK_20", "CROWD_TURNOVER_60",
        ]
    )
    assert PitLevel.PIT_DAILY.value in summary["by_pit_level"]
    assert summary["n_clusters"] >= 1


def test_build_requires_aligned_inputs():
    with pytest.raises(ValueError):
        FeatureRegistry.build(
            np.zeros((3, 5, 5), dtype=np.float32),
            np.ones((5, 6), dtype=bool),
            calibration_slice=CalibrationSlice.of(5),
        )


def test_p9_registry_version_deprecations_and_pending_data():
    """P9 §4/§6: the v4 registry records the approved deprecations with
    reasons and exposes the pending_data placeholders without adding them
    to the vocabulary.  v5 = the §7 adjudication (family ③ negative,
    LIMIT_STREAK conditional deprecation triggered, two second-pass
    consolidations).  v6 (P13 §5.3, whitelist §10.1 case 2): family ⑤
    unlocks; the pending set empties."""
    assert FEATURE_REGISTRY_VERSION == 6
    p9_deprecated = (
        "RET_5", "RET_10", "RET_120", "MOMENTUM_60",
        "VOLUME_RATIO", "TURNOVER_MA5", "MACD_DEA", "VOL_60",
    )
    tensor = np.zeros((len(FEATURE_NAMES), 3, 4), dtype=np.float32)
    registry = FeatureRegistry.build(
        tensor,
        np.ones((3, 4), dtype=bool),
        calibration_slice=CalibrationSlice.of(4),
    )
    records = registry.records()
    for name in p9_deprecated:
        record = records[name]
        assert record.deprecated, name
        assert not record.promotion_allowed, name
        assert record.deprecation_reason, name
    # P13 §5.3 (whitelist §10.1 case 2): the family-⑤ placeholders are
    # unlocked into the registry, so the pending_data summary is empty.
    summary = registry.summary()
    assert summary["pending_data"] == []


def test_p13_family5_unlock_appends_vocabulary_without_id_shift():
    """P13 §5.3/§4.4 (RED-5): the four family-⑤ features join the
    vocabulary by *append* — the 73 pre-existing names keep their exact
    order (token ids unchanged, legacy formulas parse identically) and the
    registry records them as PIT-fundamental, promoted, not deprecated."""
    from ashare_model.vocab import FEATURE_NAMES

    # Frozen baseline of the pre-P13 vocabulary (independent authority:
    # the committed pre-implementation registry).
    _PRE_P13_FEATURE_NAMES = (
        'RET_1', 'RET_5', 'RET_10', 'VOL_20', 'VOL_60', 'TURNOVER',
        'TURNOVER_CHG', 'VOLUME_RATIO', 'VOLUME_IMPACT', 'AMPLITUDE',
        'CLOSE_POSITION', 'MOMENTUM_20', 'MOMENTUM_60', 'REVERSAL_5', 'SKEW_20',
        'KURT_20', 'PE_TTM', 'PB', 'PS_TTM', 'ROE', 'ROA', 'GROSS_MARGIN',
        'NET_MARGIN', 'REVENUE_YOY', 'PROFIT_YOY', 'DEBT_RATIO', 'MARKET_CAP',
        'DIVIDEND_YIELD', 'NORTHBOUND_CHG', 'MARGIN_BALANCE_CHG',
        'LIMIT_UP_EVENT', 'LIMIT_DOWN_EVENT', 'INDUSTRY_MOMENTUM',
        'OVERNIGHT_RET', 'INTRADAY_RET', 'ILLIQ_20', 'AMOUNT_SHARE', 'MAX_20',
        'HIGH_52W', 'BETA_60', 'IVOL_60', 'RSQ_60', 'BIAS_20', 'RSI_14',
        'ATR_14', 'MACD_DIF', 'MACD_DEA', 'SUSPEND_DAYS_60', 'LIST_AGE',
        'RET_120', 'REVERSAL_60', 'REVERSAL_120', 'TURNOVER_MA5',
        'TURNOVER_MA20', 'TURNOVER_STD20', 'LIMIT_STREAK', 'LIMIT_UP_CNT_20',
        'LIMIT_BREAK', 'IND_REL_RET_5', 'IND_REL_RET_20', 'IND_REL_VOL_20',
        'IND_REL_TURNOVER', 'IND_REL_RET_60', 'IND_REL_RET_120', 'LIQ_SHOCK_20',
        'VOLUME_SHRINK_5_20', 'PV_DIV_20', 'LIMIT_UP_CNT_5',
        'LIMIT_DOWN_STREAK', 'LIMIT_BREAK_5', 'CROWD_TURNOVER_60',
        'CROWD_AMOUNT_60', 'MARGIN_CROWD_60',
    )
    family5 = [
        "CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL",
    ]
    # Append-only growth: 73 -> 77, existing prefix byte-identical.
    assert len(FEATURE_NAMES) == 77
    assert list(FEATURE_NAMES[:73]) == list(_PRE_P13_FEATURE_NAMES)
    assert list(FEATURE_NAMES[73:]) == family5
    # Sentinel id-parity (independently pinned from the pre-P13 registry):
    assert FEATURE_NAMES[0] == "RET_1"
    assert FEATURE_NAMES[16] == "PE_TTM"
    assert FEATURE_NAMES[19] == "ROE"
    assert FEATURE_NAMES[72] == "MARGIN_CROWD_60"
    assert FEATURE_NAMES.index("MARGIN_CROWD_60") == 72

    tensor = np.zeros((len(FEATURE_NAMES), 3, 4), dtype=np.float32)
    registry = FeatureRegistry.build(
        tensor,
        np.ones((3, 4), dtype=bool),
        calibration_slice=CalibrationSlice.of(4),
    )
    records = registry.records()
    for name in family5:
        record = records[name]
        assert record.pit_level == PitLevel.PIT_FUNDAMENTAL, name
        assert record.family == "fundamental", name
        assert record.promotion_allowed is True, name
        assert not record.deprecated, name
        assert record.availability_rule, name
