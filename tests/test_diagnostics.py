from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import DataConfig, ModelConfig
from ashare_model.data_loader import AshareDataLoader
from ashare_model.diagnostics import (
    correlation_summary,
    factor_coverage,
    factor_report,
    feature_family,
    rank_ic_stats,
)
from ashare_model.factors import FAMILIES, ablate_factors
from ashare_model.reward import rank_ic_series
from ashare_model.vocab import FEATURE_NAMES


def test_feature_family_maps_every_feature():
    for name in FEATURE_NAMES:
        family = feature_family(name)
        assert family in FAMILIES, name


def test_factor_coverage_counts_nonzero_cells():
    tensor = np.zeros((3, 2, 4), dtype=np.float32)
    tensor[0, :, :] = 1.0  # fully informative
    tensor[1, 0, 0] = 1.0  # one cell
    coverage = factor_coverage(tensor, eligible=np.ones((2, 4), dtype=bool))
    assert coverage[FEATURE_NAMES[0]] == pytest.approx(1.0)
    assert coverage[FEATURE_NAMES[1]] == pytest.approx(1.0 / 8.0)
    assert coverage[FEATURE_NAMES[2]] == pytest.approx(0.0)


def test_rank_ic_perfect_alignment_and_inversion():
    n_stocks, n_dates = 30, 10
    rng = np.random.default_rng(7)
    target = rng.normal(size=(n_stocks, n_dates))
    tensor = np.zeros((3, n_stocks, n_dates), dtype=np.float32)
    tensor[0] = target * 10.0  # aligned
    tensor[1] = -target * 10.0  # inverted
    stats = rank_ic_stats(
        tensor, target, [f"d{i}" for i in range(n_dates)],
        eligible=np.ones((n_stocks, n_dates), dtype=bool),
    )
    assert stats[FEATURE_NAMES[0]]["ic_mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats[FEATURE_NAMES[1]]["ic_mean"] == pytest.approx(-1.0, abs=1e-6)
    # The zero feature correlates 0 with anything.
    assert stats[FEATURE_NAMES[2]]["ic_mean"] == pytest.approx(0.0, abs=1e-6)
    assert stats[FEATURE_NAMES[0]]["n_dates"] == n_dates


def test_rank_ic_skips_degenerate_dates():
    n_stocks, n_dates = 30, 5
    target = np.zeros((n_stocks, n_dates))
    target[:, 0] = np.nan  # fewer than 10 valid stocks
    tensor = np.zeros((1, n_stocks, n_dates), dtype=np.float32)
    stats = rank_ic_stats(
        tensor, target, [f"d{i}" for i in range(n_dates)],
        eligible=np.ones((n_stocks, n_dates), dtype=bool),
    )
    # Dates 1..4 are valid but constant targets: the correlation of a
    # zero factor with a constant is NaN -> skipped, so no usable dates.
    assert stats[FEATURE_NAMES[0]]["n_dates"] == 0


def test_correlation_summary_recovers_identical_pairs():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(12, 8)).astype(np.float32)
    tensor = np.stack([base, base, -base], axis=0)  # 3 features, 12 stocks, 8 days
    summary = correlation_summary(tensor, eligible=np.ones(base.shape, dtype=bool))
    top = {(p["a"], p["b"]): p["abs_corr"] for p in summary["top_pairs"]}
    assert top[(FEATURE_NAMES[0], FEATURE_NAMES[1])] == pytest.approx(1.0, abs=1e-4)
    assert top[(FEATURE_NAMES[0], FEATURE_NAMES[2])] == pytest.approx(1.0, abs=1e-4)
    assert summary["within_family"]  # non-empty family aggregates


def test_correlation_summary_survives_constant_cross_sections():
    # A neutral placeholder row (all zeros) and a first-day constant row
    # make np.corrcoef produce NaN cells; the summary must stay finite.
    rng = np.random.default_rng(11)
    base = rng.normal(size=(10, 6)).astype(np.float32)
    tensor = np.stack([base, base, np.zeros_like(base)], axis=0)
    tensor[0, :, 0] = 0.0  # constant cross-section on the first date
    summary = correlation_summary(tensor, eligible=np.ones(base.shape, dtype=bool))
    matrix = summary["matrix"]
    assert all(
        v == v  # no NaN anywhere in the reported matrix
        for row in matrix.values()
        for v in row.values()
    )
    assert summary["top_pairs"][0]["abs_corr"] == pytest.approx(5.0 / 6.0, abs=1e-4)


def test_ablate_factors_zeros_rows_and_keeps_shape():
    tensor = np.ones((len(FEATURE_NAMES), 3, 4), dtype=np.float32)
    out = ablate_factors(tensor, ["RET_1", "PE_TTM"])
    assert out.shape == tensor.shape
    assert np.allclose(out[FEATURE_NAMES.index("RET_1")], 0.0)
    assert np.allclose(out[FEATURE_NAMES.index("PE_TTM")], 0.0)
    assert np.allclose(out[FEATURE_NAMES.index("VOL_20")], 1.0)
    # Unknown names are ignored, never crash.
    assert np.allclose(ablate_factors(tensor, ["NOPE"]), tensor)


def test_factor_report_smoke(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    report = factor_report(loader, "2024-02-01")
    assert report["feature_count"] == len(FEATURE_NAMES)
    assert report["stock_count"] == len(loader.ts_codes)
    # The union size stays, but the daily selectable size is explicit and
    # must not be confused with it.
    assert report["mean_eligible_stocks"] == pytest.approx(3.0)
    assert len(report["per_feature"]) == len(FEATURE_NAMES)
    row = report["per_feature"][FEATURE_NAMES.index("RET_1")]
    assert row["name"] == "RET_1"
    assert row["family"] == "momentum"
    assert "coverage" in row and "ic_mean" in row
    # Round-trips through JSON cleanly.
    json.loads(json.dumps(report))


def test_factor_report_writes_file(populated_db: DataConfig, tmp_path):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    out = tmp_path / "report.json"
    factor_report(loader, "2024-02-01", out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["feature_count"] == len(FEATURE_NAMES)


def test_factor_report_records_data_tiers_and_time_rules(populated_db: DataConfig):
    """P2-02: every reported feature carries its A/B/C tier; the report
    records the tier version and the usable-time rules."""
    from ashare_model.data_tier import DATA_TIER_VERSION, DataTier

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    report = factor_report(loader, "2024-02-01")
    assert report["data_tier_version"] == DATA_TIER_VERSION
    assert set(report["data_tier_rules"]) == {t.value for t in DataTier}
    assert report["tiers"] is None  # full-vocabulary report
    by_name = {row["name"]: row for row in report["per_feature"]}
    assert by_name["RET_1"]["data_tier"] == "A"
    assert by_name["ROE"]["data_tier"] == "B"
    assert by_name["INDUSTRY_MOMENTUM"]["data_tier"] == "C"
    assert sum(report["tier_summary"].values()) == len(FEATURE_NAMES)


def test_factor_report_tier_filter_restricts_the_subset(populated_db: DataConfig):
    """P2-05: a Tier A-only report contains only Tier A features, and the
    correlation matrix is restricted to that subset."""
    from ashare_model.data_tier import DataTier, feature_tier
    from ashare_model.vocab import FEATURE_NAMES

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    report = factor_report(loader, "2024-02-01", tiers=("A",))
    names = [row["name"] for row in report["per_feature"]]
    assert all(feature_tier(name) is DataTier.A for name in names)
    tier_a = {n for n in FEATURE_NAMES if feature_tier(n) is DataTier.A}
    assert set(names) == tier_a
    assert report["tiers"] == ["A"]
    assert set(report["tier_summary"]) == {"A"}
    # The correlation matrix only covers the reported features.
    assert set(report["correlations"]["matrix"]) == tier_a
    for name in names:
        assert set(report["correlations"]["matrix"][name]) == tier_a


# --- PIT universe mask in diagnostics ----------------------------------------


def _masked_tensors():
    n_stocks, n_dates = 6, 20
    join_day = 8
    rng = np.random.default_rng(55)
    target = rng.normal(size=(n_stocks, n_dates))
    base = rng.normal(size=(3, n_stocks, n_dates)).astype(np.float32)
    extreme = base.copy()
    extreme[0, -1, :join_day] = 1e9  # future member's extreme pre-join values
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[-1, :join_day] = False
    return base, extreme, target, mask


def test_coverage_ic_and_correlations_ignore_ineligible_extremes():
    base, extreme, target, mask = _masked_tensors()
    dates = [f"d{i}" for i in range(20)]
    assert factor_coverage(base, eligible=mask) == factor_coverage(
        extreme, eligible=mask
    )
    assert rank_ic_stats(base, target, dates, eligible=mask, min_stocks=3) == (
        rank_ic_stats(extreme, target, dates, eligible=mask, min_stocks=3)
    )
    assert correlation_summary(base, eligible=mask) == correlation_summary(
        extreme, eligible=mask
    )
    # Sanity: with an all-eligible mask the extreme finite values do move the ICs.
    all_eligible = np.ones(mask.shape, dtype=bool)
    assert rank_ic_stats(
        base, target, dates, min_stocks=3, eligible=all_eligible
    ) != rank_ic_stats(
        extreme, target, dates, min_stocks=3, eligible=all_eligible
    )


def test_coverage_denominator_counts_eligible_cells_only():
    mask = np.ones((4, 6), dtype=bool)
    mask[1:] = False  # only stock 0 eligible
    tensor = np.zeros((2, 4, 6), dtype=np.float32)
    tensor[0, 0, :] = 1.0  # feature 0 informative only in the eligible row
    cov = factor_coverage(tensor, eligible=mask)
    assert cov[FEATURE_NAMES[0]] == pytest.approx(1.0)
    # Non-zero cells outside the universe neither count as covered nor
    # inflate the denominator.
    tensor2 = np.ones((2, 4, 6), dtype=np.float32)
    assert factor_coverage(tensor2, eligible=mask)[FEATURE_NAMES[0]] == pytest.approx(1.0)
    tensor3 = np.zeros((2, 4, 6), dtype=np.float32)
    tensor3[0, 1:, :] = 1.0  # informative only in ineligible rows
    assert factor_coverage(tensor3, eligible=mask)[FEATURE_NAMES[0]] == pytest.approx(0.0)


def test_correlation_summary_uses_eligible_cells_per_day():
    # Two features are identical on the eligible rows and wildly different
    # on the ineligible row: the masked summary must still report |corr| 1.
    rng = np.random.default_rng(56)
    n_stocks, n_dates = 5, 6
    base = rng.normal(size=(n_stocks, n_dates)).astype(np.float32)
    tensor = np.stack([base, base])
    tensor[1, 3:, :] = base[3:, :] + 1000.0
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[3:, :] = False
    summary = correlation_summary(tensor, eligible=mask)
    top = {(p["a"], p["b"]): p["abs_corr"] for p in summary["top_pairs"]}
    assert top[(FEATURE_NAMES[0], FEATURE_NAMES[1])] == pytest.approx(1.0, abs=1e-4)
    # With an all-eligible mask the ineligible divergence breaks the correlation.
    open_summary = correlation_summary(
        tensor, eligible=np.ones((n_stocks, n_dates), dtype=bool)
    )
    open_top = {(p["a"], p["b"]): p["abs_corr"] for p in open_summary["top_pairs"]}
    assert open_top[(FEATURE_NAMES[0], FEATURE_NAMES[1])] < 1.0


def test_rank_ic_stats_reuses_unified_rank_ic_series():
    # The diagnostics IC must be the exact unified implementation: same
    # daily series, same summary values (single Spearman code path).
    rng = np.random.default_rng(57)
    n_stocks, n_dates = 25, 12
    target = rng.normal(size=(n_stocks, n_dates))
    tensor = np.stack(
        [target, -target, np.zeros_like(target)]
    ).astype(np.float32)
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[-3:, :6] = False
    stats = rank_ic_stats(
        tensor, target, [f"d{i}" for i in range(n_dates)],
        eligible=mask, min_stocks=5,
    )
    daily = rank_ic_series(tensor, target, 5, universe_mask=mask)
    for i, name in enumerate(FEATURE_NAMES[:3]):
        finite = daily[i][np.isfinite(daily[i])]
        expected_mean = float(finite.mean()) if finite.size else 0.0
        assert stats[name]["ic_mean"] == pytest.approx(expected_mean, abs=1e-12)
        assert stats[name]["n_dates"] == int(finite.size)


def test_diagnostics_reject_mask_shape_mismatch():
    tensor = np.zeros((2, 4, 6), dtype=np.float32)
    target = np.zeros((4, 6))
    bad = np.ones((3, 6), dtype=bool)
    with pytest.raises(ValueError, match="eligible shape"):
        factor_coverage(tensor, eligible=bad)
    with pytest.raises(ValueError, match="eligible shape"):
        rank_ic_stats(tensor, target, [], eligible=bad)
    with pytest.raises(ValueError, match="eligible shape"):
        correlation_summary(tensor, eligible=bad)
