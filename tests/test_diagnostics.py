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
from ashare_model.vocab import FEATURE_NAMES


def test_feature_family_maps_every_feature():
    for name in FEATURE_NAMES:
        family = feature_family(name)
        assert family in FAMILIES, name


def test_factor_coverage_counts_nonzero_cells():
    tensor = np.zeros((3, 2, 4), dtype=np.float32)
    tensor[0, :, :] = 1.0  # fully informative
    tensor[1, 0, 0] = 1.0  # one cell
    coverage = factor_coverage(tensor)
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
    stats = rank_ic_stats(tensor, target, [f"d{i}" for i in range(n_dates)])
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
    stats = rank_ic_stats(tensor, target, [f"d{i}" for i in range(n_dates)])
    # Dates 1..4 are valid but constant targets: the correlation of a
    # zero factor with a constant is NaN -> skipped, so no usable dates.
    assert stats[FEATURE_NAMES[0]]["n_dates"] == 0


def test_correlation_summary_recovers_identical_pairs():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(12, 8)).astype(np.float32)
    tensor = np.stack([base, base, -base], axis=0)  # 3 features, 12 stocks, 8 days
    summary = correlation_summary(tensor)
    top = {(p["a"], p["b"]): p["abs_corr"] for p in summary["top_pairs"]}
    assert top[(FEATURE_NAMES[0], FEATURE_NAMES[1])] == pytest.approx(1.0, abs=1e-4)
    assert top[(FEATURE_NAMES[0], FEATURE_NAMES[2])] == pytest.approx(1.0, abs=1e-4)
    assert summary["within_family"]  # non-empty family aggregates


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
