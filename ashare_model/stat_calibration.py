"""Statistical-method independent calibration (p48, WP 12).

This module owns the *calibration harness* of the significance
machinery — the scenario generators, the Wilson interval, the frozen
verdict rule and the seed discipline — NOT the statistics themselves
(:mod:`ashare_model.eval_corrections` is untouched; p48 declares and
measures, it does not tune).

The frozen decision rule (contract ``docs/p48_stat_calibration_draft.md``
§2 B4) can never be widened: a scenario whose rejection-rate confidence
interval lies entirely above the nominal level is
``unreliable_at_nominal`` and supports NO nominal-5% significance claim;
the only exit is fixing the method under a new contract and
recalibrating on a fresh disjoint seed block.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

CALIBRATION_VERSION = 1

#: Nominal significance level the machinery is calibrated against.
ALPHA = 0.05

#: Wilson z for a two-sided 95% interval.
_Z95 = 1.959963984540054

#: Frozen acceptance thresholds (B4): below this repetition count, or
#: with a CI this wide, the scenario verdict is "insufficient_samples"
#: — no reliable inference, never a pass.
R_MIN = 500
CI_MAX_WIDTH = 0.04

#: The p40 tuning seed blocks (C1-C5 in tests/test_eval_resampling_
#: calibration.py) — seeds that PARTICIPATED in method adjustment and
#: may therefore never be reused for acceptance.
TUNING_SEED_BLOCKS: tuple[tuple[int, int], ...] = (
    (10_000, 240),
    (20_000, 240),
    (30_000, 240),
    (40_000, 120),
    (50_000, 120),
)

#: The p48 validation block: every scenario draws its held-out seeds
#: from its own disjoint slice (start = 900000 + index * 10_000).
VALIDATION_SEED_BLOCK: tuple[int, int] = (900_000, 100_000)

#: The frozen scenario registry (B2).  ``kind`` is "size" (null: the
#: rejection rate should sit at the nominal level) or "power" (a
#: planted edge; the detection rate is reported, never banded).
SCENARIO_SPECS: dict[str, dict[str, Any]] = {
    "independent_noise": {
        "kind": "size", "k_trials": 8, "n_dates": 250, "scale": 0.01,
    },
    "common_factor_095": {
        "kind": "size", "k_trials": 8, "n_dates": 250,
        "loading": 0.95, "scale": 0.01,
    },
    "ar1_phi04": {
        "kind": "size", "k_trials": 8, "n_dates": 250,
        "phi": 0.4, "scale": 0.01,
    },
    "duplicate_candidates": {
        "kind": "size", "k_trials": 9, "n_dates": 250,
        "base": 3, "copies": 3, "scale": 0.01,
    },
    "short_window_30d": {
        "kind": "size", "k_trials": 4, "n_dates": 30, "scale": 0.01,
    },
    "power_sr05": {
        "kind": "power", "k_trials": 8, "n_dates": 250,
        "sr": 0.5, "scale": 0.01,
    },
    "power_sr025": {
        "kind": "power", "k_trials": 8, "n_dates": 250,
        "sr": 0.25, "scale": 0.01,
    },
    # WP17 W5 (FROZEN_17 §5.2): the task-scale power point, REGISTERED
    # but never run in this package (red line 1: no new calibration
    # runs).  Derivation (FROZEN §5.1, frozen arithmetic): the task's
    # edge is <1% annualized relative excess; at the boundary 1%,
    # daily mu = 0.01/252, sigma = scale = 0.01 (the family-wide noise
    # scale), so sr = mu/sigma = 1/252 ~= 0.0039683 and the annualized
    # Sharpe is 1/sqrt(252) ~= 0.063 — 126x/63x BELOW the planted edges
    # of power_sr05/power_sr025.  The name sorts AFTER short_window_30d
    # so it takes seed slice 970000 and the seven p48 scenarios keep
    # their registered slices (measured naming constraint, §5.2).
    "task_scale_excess_1pct": {
        "kind": "power", "k_trials": 8, "n_dates": 250,
        "sr": 1.0 / 252.0, "scale": 0.01,
        "task_scale": True,
        "note": (
            "power_sr05/power_sr025 are NOT extrapolable to the task "
            "scale: daily Sharpe 0.5/0.25 annualize to 7.94/3.97 "
            "(x sqrt(252)), i.e. 126%/63% annualized relative excess at "
            "scale=0.01 - 126x/63x the task-scale edge (<1% annualized "
            "relative excess; task_scale_excess_1pct plants sr=1/252 "
            "~= 0.00397, annualized Sharpe 1/sqrt(252) ~= 0.063). "
            "Detection rates measured at 0.5/0.25 say nothing about "
            "detection at the task scale; no power claim is made at the "
            "task scale until task_scale_excess_1pct is calibrated "
            "under the wp17 W6 protocol."
        ),
    },
}


def wilson_ci(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""

    k = int(k)
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (
        max(center - half, 0.0),
        min(center + half, 1.0),
    )


def n_perms_floor(alpha: float) -> int:
    """Smallest ``n_perms`` whose p-value grid can reach ``alpha``.

    The minimum achievable p is ``1/(n_perms+1)``; a test that cannot
    mathematically reach the rejection level cannot make a reliable
    inference at that level (B1).
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return int(math.ceil(1.0 / alpha)) - 1


def calibration_verdict(
    rejections: int,
    reps: int,
    alpha: float = ALPHA,
    *,
    r_min: int = R_MIN,
    ci_max_width: float = CI_MAX_WIDTH,
) -> dict[str, Any]:
    """The frozen B4 decision for one scenario's measured rate.

    ``insufficient_samples`` (no reliable inference) takes precedence;
    then calibrated / conservative / unreliable_at_nominal by the
    Wilson interval's position relative to ``alpha``.  The rule is
    frozen: an above-nominal interval is NEVER widened into a pass.
    """

    rate = float(rejections) / float(reps)
    low, high = wilson_ci(rejections, reps)
    width = high - low
    # Erratum 1: a decisive interval (entirely one side of alpha) is a
    # valid one-sided 95% ruling regardless of its width; the width cap
    # only guards indecisive intervals straddling the nominal level.
    decisive = low > alpha or high < alpha
    if reps < r_min or (width > ci_max_width and not decisive):
        verdict = "insufficient_samples"
        claim = (
            "No reliable inference: fewer than "
            f"{r_min} repetitions (or CI width {width:.3f} > "
            f"{ci_max_width}) — the measured rate cannot bound the size."
        )
    elif low <= alpha <= high:
        verdict = "calibrated"
        # WP17 W4 (FROZEN_17 §4.2, frozen verbatim): the positive label
        # states NON-REJECTION only.  A CI containing alpha also admits
        # rates above alpha (e.g. [4.6%, 7.5%] contains 5% AND admits
        # >5%), and the CI/test correspondence supports reject /
        # non-reject decisions — it does not automatically verify an
        # upper bound on the false-positive rate.
        claim = (
            f"No significant departure from the nominal {alpha:.0%} "
            f"level detected: rate={rate:.3f}, 95% CI "
            f"[{low:.3f}, {high:.3f}] contains {alpha:.0%} — supports "
            "non-rejection only; not a verification that the "
            "false-positive rate is at most "
            f"{alpha:.0%} (an interval containing {alpha:.0%} also "
            f"admits rates above {alpha:.0%})."
        )
    elif high < alpha:
        verdict = "conservative"
        claim = (
            f"Size below nominal (CI [{low:.3f}, {high:.3f}] entirely "
            f"under {alpha}): {alpha:.0%} claims are valid and "
            "conservative; power is reduced."
        )
    else:
        verdict = "unreliable_at_nominal"
        claim = (
            f"NO {alpha:.0%} significance claim: measured size "
            f"{rate:.3f}, 95% CI [{low:.3f}, {high:.3f}] lies entirely "
            f"above {alpha}. The honest statement is the measured "
            "effective size; fixing the method requires a new contract "
            "and a fresh disjoint seed block."
        )
    return {
        "verdict": verdict,
        "rejections": int(rejections),
        "reps": int(reps),
        "alpha": float(alpha),
        "rate": rate,
        "ci_low": low,
        "ci_high": high,
        "ci_width": width,
        "claim": claim,
    }


def significance_readiness(
    calibrations: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """The gate-facing pre-claim query (B5): which regimes may carry a
    nominal-level significance claim.

    Conservative regimes (measured size entirely below nominal) keep the
    claim; unreliable or insufficient ones do not; regimes with NO
    calibration record default to "no reliable inference".
    """

    out: dict[str, dict[str, Any]] = {}
    for name, verdict in calibrations.items():
        state = verdict.get("verdict")
        claimable = state in ("calibrated", "conservative")
        out[name] = {
            "verdict": state,
            "claim_5pct": claimable,
            "reason": verdict.get("claim", ""),
        }
    out["unlisted_regime"] = {
        "verdict": "uncalibrated",
        "claim_5pct": False,
        "reason": (
            "No reliable inference: the regime has no calibration "
            "record on a held-out seed block (p48 B5)."
        ),
    }
    return out


def generate_scenario(name: str, rng: np.random.Generator) -> np.ndarray:
    """One draw of a frozen scenario as a ``[k, n]`` matrix."""

    spec = SCENARIO_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown scenario {name!r} (p48 B2 registry)")
    n = int(spec["n_dates"])
    k = int(spec["k_trials"])
    scale = float(spec["scale"])
    if name == "independent_noise":
        return scale * rng.normal(0.0, 1.0, size=(k, n))
    if name == "common_factor_095":
        loading = float(spec["loading"])
        factor = rng.normal(0.0, 1.0, size=n)
        idio = rng.normal(0.0, 1.0, size=(k, n))
        return scale * (
            loading * factor[None, :] + math.sqrt(1.0 - loading**2) * idio
        )
    if name == "ar1_phi04":
        phi = float(spec["phi"])
        shocks = rng.normal(0.0, 1.0, size=(k, n))
        rows = np.empty_like(shocks)
        rows[:, 0] = shocks[:, 0]
        for t in range(1, n):
            rows[:, t] = phi * rows[:, t - 1] + shocks[:, t]
        return scale * rows
    if name == "duplicate_candidates":
        base = int(spec["base"])
        copies = int(spec["copies"])
        raw = rng.normal(0.0, 1.0, size=(base, n))
        return scale * np.repeat(raw, copies, axis=0)
    if name == "short_window_30d":
        return scale * rng.normal(0.0, 1.0, size=(k, n))
    if name in ("power_sr05", "power_sr025", "task_scale_excess_1pct"):
        sr = float(spec["sr"])
        m = scale * rng.normal(0.0, 1.0, size=(k, n))
        m[0] = m[0] + sr * scale
        return m
    raise ValueError(f"scenario {name!r} has no generator")


def scenario_seed_start(name: str) -> int:
    """This scenario's held-out seed slice inside the validation block."""

    names = sorted(SCENARIO_SPECS)
    index = names.index(name)
    return VALIDATION_SEED_BLOCK[0] + index * 10_000


def calibrate_scenario(
    name: str,
    reps: int,
    *,
    n_perms: int = 999,
    block_size: int = 10,
    alpha: float = ALPHA,
    episode_token: str | None = None,
) -> dict[str, Any]:
    """Run one scenario's held-out calibration (the heavy task).

    ``reps`` independent draws, each through the REAL
    :func:`eval_corrections.max_t_from_matrix` with seeds from this
    scenario's validation slice.  Returns the verdict plus the
    wall-clock cost and per-call milliseconds (B4 reporting duty).

    WP17 W6 (FROZEN_17 §6.5, guard挂点 1): a calibration run may only
    start inside an OPEN method-revision episode — ``episode_token``
    must name one, otherwise the call refuses (fail-closed; the default
    ``None`` refuses).  There is no bypass surface.
    """

    from .method_revision import require_open_episode

    require_open_episode(episode_token)

    from .eval_corrections import max_t_from_matrix

    seed_start = scenario_seed_start(name)
    rejections = 0
    started = time.perf_counter()
    for rep in range(int(reps)):
        rng = np.random.default_rng(seed_start + rep)
        matrix = generate_scenario(name, rng)
        out = max_t_from_matrix(
            matrix, n_perms=n_perms, seed=seed_start + rep,
            block_size=block_size,
        )
        rejections += int(bool(out["p_value"] <= alpha))
    seconds = time.perf_counter() - started
    verdict = calibration_verdict(rejections, reps, alpha)
    verdict.update(
        {
            "scenario": name,
            "kind": SCENARIO_SPECS[name]["kind"],
            "n_perms": int(n_perms),
            "block_size": int(block_size),
            "seed_start": seed_start,
            "seconds": seconds,
            "ms_per_call": 1000.0 * seconds / max(int(reps), 1),
        }
    )
    return verdict
