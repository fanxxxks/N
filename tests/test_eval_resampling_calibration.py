"""P40 resampling rejection-rate calibration (WP 5A, contract §2 R5).

The max-t p-value is the null distribution of the max studentized mean
over trials; under a correctly-built null its rejection rate at
alpha = 0.05 is the nominal size.  These tests measure that rate over
many fixed seeds in the noise regimes the data actually live in — a
common market factor (cross-sectionally correlated trials), AR(1)
autocorrelated trials, and duplicate candidates — plus a short-series
anti-conservative guard and a power guard.  A single-seed "not
significant" check can never certify any of this (the old acceptance).

Bands: binomial(240, 0.05) has a two-sided 99% interval of roughly
[0.014, 0.086]; the band is widened to [0.010, 0.115] to absorb block-
bootstrap finite-sample size distortion (a disclosed tolerance, still
far from both degenerate directions 0.00 and 1.00).  The short-series
regime uses a one-sided anti-conservative bound only: with n = 6 the
block bootstrap is coarse, so the preregistered property is "never the
old always-significant behaviour", not size precision.

The resampler is driven directly on synthetic matrices
(``max_t_from_matrix``); the rows-level end-to-end semantics are pinned
by ``tests/test_p40_statistical_resampling.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

ALPHA_BAND = (0.010, 0.115)
N_PERMS = 200
N_DATES = 250
BLOCK = 10
N_REPS = 240


def _resample(matrix: np.ndarray, seed: int) -> bool:
    from ashare_model.eval_corrections import max_t_from_matrix

    out = max_t_from_matrix(
        matrix, n_perms=N_PERMS, seed=seed, block_size=BLOCK
    )
    return bool(out["significant_95"])


def _rate(rejections: int, reps: int) -> float:
    return rejections / reps


def test_c1_common_factor_correlated_noise_size():
    """K trials sharing a 0.95-loading factor: the joint date-block draw
    must preserve the cross-trial dependence the statistic contains (the
    per-trial independent draw destroys it and collapses the size)."""

    rejections = 0
    for rep in range(N_REPS):
        rng = np.random.default_rng(10_000 + rep)
        factor = rng.normal(0.0, 1.0, size=N_DATES)
        idio = rng.normal(0.0, 1.0, size=(8, N_DATES))
        matrix = 0.95 * factor[None, :] + np.sqrt(1.0 - 0.95**2) * idio
        rejections += _resample(0.01 * matrix, seed=rep)
    rate = _rate(rejections, N_REPS)
    assert ALPHA_BAND[0] <= rate <= ALPHA_BAND[1], rate


def test_c2_ar1_autocorrelated_noise_size():
    """Per-trial AR(1) (phi=0.4) noise: the block draw must keep the size
    controlled under within-series autocorrelation."""

    rejections = 0
    phi = 0.4
    for rep in range(N_REPS):
        rng = np.random.default_rng(20_000 + rep)
        shocks = rng.normal(0.0, 1.0, size=(8, N_DATES))
        matrix = np.empty_like(shocks)
        matrix[:, 0] = shocks[:, 0]
        for t in range(1, N_DATES):
            matrix[:, t] = phi * matrix[:, t - 1] + shocks[:, t]
        rejections += _resample(0.01 * matrix, seed=rep)
    rate = _rate(rejections, N_REPS)
    assert ALPHA_BAND[0] <= rate <= ALPHA_BAND[1], rate


def test_c3_duplicate_candidate_pool_size():
    """3 independent noise trials duplicated 3x each (a re-tested formula
    family): duplicates count as trials, and the joint draw must price
    their induced correlation correctly (perfectly copied columns)."""

    rejections = 0
    for rep in range(N_REPS):
        rng = np.random.default_rng(30_000 + rep)
        base = rng.normal(0.0, 0.01, size=(3, N_DATES))
        matrix = np.repeat(base, 3, axis=0)
        rejections += _resample(matrix, seed=rep)
    rate = _rate(rejections, N_REPS)
    assert ALPHA_BAND[0] <= rate <= ALPHA_BAND[1], rate


def test_c4_short_series_never_automatically_significant():
    """n = 6 < block_size = 10: the old block rule degenerated to a full
    rotation (p = 1/(n_perms+1) on any positive-mean pool, rejection rate
    ~1.0).  The effective-block rule must never behave that way."""

    reps = 120
    rejections = 0
    for rep in range(reps):
        rng = np.random.default_rng(40_000 + rep)
        matrix = rng.normal(0.0, 0.01, size=(4, 6))
        rejections += _resample(matrix, seed=rep)
    rate = _rate(rejections, reps)
    assert rate <= 0.18, rate


def test_c5_strong_signal_is_detected():
    """One trial with a 0.5 daily-SR edge among 7 noise trials: the test
    must still reject (a conservative breakage that never rejects is as
    useless as one that always does)."""

    reps = 120
    rejections = 0
    for rep in range(reps):
        rng = np.random.default_rng(50_000 + rep)
        matrix = rng.normal(0.0, 0.01, size=(8, N_DATES))
        matrix[0] += 0.005
        rejections += _resample(matrix, seed=rep)
    rate = _rate(rejections, reps)
    assert rate >= 0.90, rate
