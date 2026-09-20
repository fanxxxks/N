"""P48 statistical-calibration RED/GREEN tests (WP 12).

Structural invariants and small-sample counterexamples only (contract
§2 B6); the R=1000-scale recalibration is a separate acceptance task.

* the scenario registry, Wilson interval, verdict logic and seed
  discipline exist in one authority (N1);
* Wilson intervals match hand-checked cases (N2);
* the verdict table switches exactly on the frozen thresholds and
  refuses to over-claim (N3);
* the n_perms grid counterexample: with 18 permutations even a planted
  daily-SR-1.0 signal cannot reach p <= 0.05 - the floor function and
  the resampler agree (N4);
* the calibration seed block is disjoint from the p40 tuning blocks and
  the scenario generators have their declared dependence structure (N5).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_model.stat_calibration import (
    CALIBRATION_VERSION,
    SCENARIO_SPECS,
    TUNING_SEED_BLOCKS,
    VALIDATION_SEED_BLOCK,
    calibration_verdict,
    generate_scenario,
    n_perms_floor,
    significance_readiness,
    wilson_ci,
)


# --- N2: Wilson intervals, hand-checked ---------------------------------------


def test_n2_wilson_interval_hand_cases():
    # k=0, n=100: the two-sided Wilson score interval degenerates to a
    # lower bound of exactly 0 (fixture correction: the RED version
    # demanded low > 0, which the score-test inversion never gives);
    # the upper bound must stay below the nominal 5%.
    low, high = wilson_ci(0, 100)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high < 0.05
    # k=50, n=100: contains 0.5; the half-width at p=0.5, n=100 is
    # ~0.096 so the width is ~0.19 (hand-checked).
    low, high = wilson_ci(50, 100)
    assert low < 0.5 < high
    assert high - low == pytest.approx(0.1924, abs=5e-3)
    # Monotone in k at fixed n.
    lo1, hi1 = wilson_ci(3, 100)
    lo2, hi2 = wilson_ci(7, 100)
    assert lo2 > lo1 and hi2 > hi1


# --- N3: the verdict table (frozen thresholds, no widening) --------------------


def test_n3_verdict_switches_on_the_frozen_thresholds():
    # calibrated: CI contains 0.05 with ample reps.
    assert calibration_verdict(
        rejections=50, reps=1000, alpha=0.05
    )["verdict"] == "calibrated"
    # conservative: CI entirely below nominal.
    v = calibration_verdict(rejections=15, reps=1000, alpha=0.05)
    assert v["verdict"] == "conservative"
    # unreliable_at_nominal: CI entirely above nominal -> NO 5% claim.
    v = calibration_verdict(rejections=115, reps=1000, alpha=0.05)
    assert v["verdict"] == "unreliable_at_nominal"
    # insufficient_samples: too few reps (below R_MIN=500).
    v = calibration_verdict(rejections=30, reps=400, alpha=0.05)
    assert v["verdict"] == "insufficient_samples"
    assert "no reliable inference" in v["claim"].lower()


def test_n3_unreliable_never_passes_by_widening():
    # A scenario measuring 12% rejections over 1000 reps has CI_low well
    # above 5%: whatever we wish, the verdict must refuse a 5% claim and
    # report the measured effective size.
    v = calibration_verdict(rejections=120, reps=1000, alpha=0.05)
    assert v["verdict"] == "unreliable_at_nominal"
    assert v["rate"] == pytest.approx(0.12)
    assert v["ci_low"] > 0.05
    assert "0.05" in v["claim"]


def test_n3_significance_readiness_blocks_uncovered_regimes():
    ready = significance_readiness(
        {
            "independent_noise": calibration_verdict(50, 1000),
            "short_window_30d": calibration_verdict(120, 1000),
        }
    )
    assert ready["independent_noise"]["claim_5pct"] is True
    assert ready["short_window_30d"]["claim_5pct"] is False
    # A regime with no calibration record at all cannot support a claim.
    assert ready["unlisted_regime"]["claim_5pct"] is False
    assert "no reliable inference" in ready["unlisted_regime"]["reason"].lower()


# --- N4: the n_perms grid counterexample --------------------------------------


def test_n4_perms_floor_is_19_at_5pct():
    assert n_perms_floor(0.05) == 19
    assert n_perms_floor(0.01) == 99


def test_n4_eighteen_perms_cannot_reject_even_a_huge_signal():
    from ashare_model.eval_corrections import max_t_from_matrix

    rng = np.random.default_rng(7)
    matrix = rng.normal(0.0, 0.01, size=(4, 250))
    matrix[0] += 0.10  # daily SR ~ 10: an unmissable planted edge
    out18 = max_t_from_matrix(matrix, n_perms=18, seed=0, block_size=10)
    assert out18["p_value"] == pytest.approx(1.0 / 19)
    assert out18["significant_95"] is False
    out19 = max_t_from_matrix(matrix, n_perms=19, seed=0, block_size=10)
    assert out19["p_value"] <= 1.0 / 20 + 1e-12
    assert out19["significant_95"] is True


# --- N5: seed discipline and scenario structure -------------------------------


def test_n5_validation_seed_block_disjoint_from_tuning_blocks():
    for block in TUNING_SEED_BLOCKS:
        # Blocks are [start, start + span) ranges; the validation block
        # must not overlap any tuning range.
        assert VALIDATION_SEED_BLOCK[0] >= block[0] + block[1] or (
            VALIDATION_SEED_BLOCK[0] + VALIDATION_SEED_BLOCK[1] <= block[0]
        ), (block, VALIDATION_SEED_BLOCK)


def test_n5_scenario_registry_covers_the_frozen_families():
    names = set(SCENARIO_SPECS)
    assert {
        "independent_noise", "common_factor_095", "ar1_phi04",
        "duplicate_candidates", "short_window_30d",
        "power_sr05", "power_sr025",
    } <= names
    for name, spec in SCENARIO_SPECS.items():
        assert spec["kind"] in ("size", "power")
        assert spec["n_dates"] >= 30
        assert spec["k_trials"] >= 1


def test_n5_generators_have_the_declared_dependence_structure():
    rng = np.random.default_rng(3)
    m = generate_scenario("common_factor_095", rng)
    k, n = m.shape
    assert (k, n) == (SCENARIO_SPECS["common_factor_095"]["k_trials"], 250)
    corr = np.corrcoef(m)[0, 1]
    assert 0.85 < corr <= 1.0  # ~0.95 loading shared factor

    m = generate_scenario("ar1_phi04", rng)
    # lag-1 autocorrelation of each trial ~ phi (finite-sample slack).
    ac = np.mean(
        [np.corrcoef(row[:-1], row[1:])[0, 1] for row in m]
    )
    assert 0.25 < ac < 0.55

    m = generate_scenario("duplicate_candidates", rng)
    assert m.shape[0] == 9
    np.testing.assert_array_equal(m[0], m[1])  # 3x duplicated columns
    np.testing.assert_array_equal(m[3], m[4])

    m = generate_scenario("short_window_30d", rng)
    assert m.shape[1] == 30

    m = generate_scenario("power_sr05", rng)
    # the planted trial has a positive mean edge; the rest are noise.
    assert m[0].mean() > 3 * m[0].std() / math.sqrt(m.shape[1])


def test_n1_version():
    assert CALIBRATION_VERSION == 1
