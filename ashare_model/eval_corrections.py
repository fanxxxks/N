"""Multiple-testing corrections for the evaluation protocol.

Extracted from ``evaluation.py`` (P7 Phase A3) by reason-to-change: this
module owns the *statistical adjudication* over the stitched trial matrix —
Probabilistic/Deflated Sharpe (Bailey & Lopez de Prado 2014), the
studentized max-t block bootstrap, and the pure-noise self-check.  It
changes when correction *methodology* changes — not when fold contracts,
metric definitions, search backends or artifact schemas change.

p40 (``docs/p40_statistical_resampling_draft.md``): the max-t bootstrap
aligns trials by their **real mark dates** and draws date blocks
**jointly** — one block index per permutation, applied to every trial —
so the null distribution carries the cross-trial dependence the observed
statistic actually contains.  The adjudication window is the intersection
of the current run's trial date indexes; historical trials that do not
cover it stay in the multiplicity count but cannot be aligned into the
joint matrix (recorded ``matrix_excluded``).

Import direction: depends on :mod:`ashare_model.eval_metrics` (the stitched
trial matrix) and :mod:`ashare_model.eval_folds`; never imports the
``evaluation`` facade.
"""

from __future__ import annotations

import math

import numpy as np

from ashare_data.config import BacktestConfig, ProtocolConfig

from .data_loader import AshareDataLoader
from .eval_folds import epoch_slice, resolve_folds
from .eval_metrics import evaluate_signal, stitch_oos_series

# --- multiple-testing corrections (Bailey & Lopez de Prado 2014) ------------
#
# Both corrections share the same trial matrix (one trial = one non-failed
# row with its raw OOS excess-return series), so their conclusions are
# directly comparable.  scipy is not a project dependency: the normal CDF
# uses math.erf and the inverse normal CDF uses Acklam's rational
# approximation plus one Halley refinement (|error| ~ 1e-9).

_SQRT2 = math.sqrt(2.0)
_EULER_MASCHERONI = 0.5772156649015329

# Acklam's inverse-normal-CDF coefficients.
_ACKLAM_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)


def _poly(coeffs, x: float) -> float:
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF for ``0 < p < 1`` (Acklam + Halley)."""

    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf undefined for p={p}")
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = _poly(_ACKLAM_C, q) / _poly(_ACKLAM_D + (1.0,), q)
    elif p <= 0.97575:
        q = p - 0.5
        r = q * q
        x = _poly(_ACKLAM_A, r) * q / _poly(_ACKLAM_B + (1.0,), r)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -_poly(_ACKLAM_C, q) / _poly(_ACKLAM_D + (1.0,), q)
    # One Halley refinement step against the true CDF.
    e = 0.5 * math.erfc(-x / _SQRT2) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF via ``erf`` (scalar; the only caller is PSR)."""

    return 0.5 * (1.0 + math.erf(float(x) / _SQRT2))


def psr(sr: float, sr_benchmark: float, t: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe ratio (B&LdP Eq. 14, incl. skew/kurt terms)."""

    num = (sr - sr_benchmark) * math.sqrt(t - 1)
    den = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    return float(norm_cdf(num / den))


def expected_max_sr(n_trials: int, t: int, skew: float, kurt: float) -> float:
    """Expected maximum SR over ``n_trials`` independent null trials
    (B&LdP Eq. 13 with the null benchmark SR of zero)."""

    sr0 = 0.0
    variance = (1.0 - skew * sr0 + (kurt - 1.0) / 4.0 * sr0 * sr0) / (t - 1)
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr0 + math.sqrt(variance) * (
        (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    )


def deflated_sharpe(sr_best, n_trials, t, skew, kurt) -> float:
    """PSR of the best trial against the multiplicity-corrected benchmark."""

    return psr(sr_best, expected_max_sr(n_trials, t, skew, kurt), t, skew, kurt)


def _trial_stats(excess: np.ndarray) -> tuple[float, float, float, int]:
    t = excess.size
    mean = float(excess.mean())
    std = float(excess.std(ddof=1)) if t > 1 else 0.0
    if std < 1e-12:
        return 0.0, 0.0, 3.0, t
    sr = mean / std
    skew = float(((excess - mean) ** 3).mean() / std**3)
    kurt = float(((excess - mean) ** 4).mean() / std**4)
    return sr, skew, kurt, t


def _stitched_pool(
    rows: list[dict], extra_trial_rows: list[dict] | None = None
) -> list[dict]:
    """The v20 trial matrix: stitched trials, current run first.

    This run's rows and the extra rows from prior artifacts are stitched
    **separately** — a prior run's ``trained`` trial is a different trial
    from this run's, so they must never share a series.  p40: every
    stitched trial carries its real mark-date axis (``dates``).
    """

    pool: list[dict] = []
    for is_current, trial_rows in ((True, rows), (False, extra_trial_rows or [])):
        for trial in stitch_oos_series(trial_rows):
            excess = np.asarray(trial["excess_returns"], dtype=np.float64)
            dates = tuple(str(x) for x in trial.get("dates") or ())
            if excess.size >= 2 and len(dates) == excess.size:
                pool.append(
                    {
                        "trial": trial,
                        "excess": excess,
                        "dates": dates,
                        "current": is_current,
                    }
                )
    return pool


def _dsr_from_pool(pool: list[dict]) -> dict | None:
    """Deflated Sharpe of the best stitched trial in the pool."""

    stats = []
    for entry in pool:
        sr, skew, kurt, t = _trial_stats(entry["excess"])
        stats.append((entry["trial"], sr, skew, kurt, t))
    if not stats:
        return None
    best_trial, sr, skew, kurt, t = max(stats, key=lambda x: x[1])
    if len(stats) == 1:
        # No multiplicity: the corrected benchmark collapses to SR = 0.
        dsr = psr(sr, 0.0, t, skew, kurt)
    else:
        dsr = deflated_sharpe(sr, len(stats), t, skew, kurt)
    return {
        "dsr": dsr,
        "n_trials": len(stats),
        "sr_best": sr,
        "t_best": t,
        "skew_best": skew,
        "kurt_best": kurt,
        "best_candidate": best_trial.get("candidate"),
        "best_fold_test_end": best_trial.get("fold_test_end"),
        "best_seed": best_trial.get("seed"),
    }


def dsr_from_rows(
    rows: list[dict], extra_trial_rows: list[dict] | None = None
) -> dict | None:
    """Deflated Sharpe over the **stitched** trial matrix (v20).

    One trial = one (candidate, seed) stitched outer-OOS series;
    ``extra_trial_rows`` are raw rows of prior artifacts, stitched
    separately and joined to the multiplicity pool.
    """

    return _dsr_from_pool(_stitched_pool(rows, extra_trial_rows))


# p40 R4: the joint window must carry enough dates for the block bootstrap
# to calibrate the max-t tail at all.  Below this floor the resampling null
# is too narrow and any positive-mean pool trends significant (measured:
# n=6 -> rejection rate ~0.4-0.9); the test therefore refuses to run
# instead of returning a garbage p-value.
P40_MIN_WINDOW_DATES = 30


def max_t_from_matrix(
    matrix: np.ndarray,
    *,
    n_perms: int = 5000,
    seed: int = 0,
    block_size: int = 10,
) -> dict:
    """Studentized max-t circular-block bootstrap over one excess matrix.

    ``matrix`` is ``[n_trials, n_dates]`` whose columns share a single
    real date index (the p40 joint matrix).  The window must carry at
    least :data:`P40_MIN_WINDOW_DATES` dates — below that the resampled
    null cannot calibrate the max-t tail, so the call raises
    ``ValueError`` (fail closed: no p-value at all, never the
    always-significant garbage of a degenerate draw).

    The effective block length is ``max(1, min(block_size, n_dates // 6))``
    — a length/ratio rule, not just a cap: a block equal to (or half) the
    window makes every draw a rotation or a two-block splice, whose
    bootstrap means under-disperse and whose p-value collapses towards
    ``1/(n_perms+1)`` on any positive-mean pool (the p40 degeneracy,
    measured rejection rate 0.13-0.24 at n/b = 2 vs 0.054-0.092 at the
    operating point).

    Per permutation **one** block index is drawn on the date axis and
    applied to every trial jointly, so the resampled cross-section keeps
    the dependence structure of the observed statistic.  Each trial's
    bootstrapped mean is studentized by its own observed standard
    deviation, and the max over trials forms the null of the max
    t-statistic.
    """

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 1 or m.shape[1] < 2:
        raise ValueError(
            f"max_t_from_matrix needs a [n_trials, n_dates] matrix with "
            f"n_dates >= 2, got shape {m.shape}"
        )
    k, n = m.shape
    if n < P40_MIN_WINDOW_DATES:
        raise ValueError(
            f"the joint window carries {n} dates, below the p40 minimum "
            f"of {P40_MIN_WINDOW_DATES} (P40_MIN_WINDOW_DATES): the block "
            "bootstrap cannot calibrate the max-t tail on this sample — "
            "fail closed instead of reporting a garbage p-value "
            "(docs/p40_statistical_resampling_draft.md §2 R4)"
        )
    block = max(1, min(int(block_size), n // 6))
    stds = m.std(axis=1, ddof=1)
    tstats = np.where(
        stds > 1e-12,
        np.sqrt(n) * (m.mean(axis=1) / np.maximum(stds, 1e-12)),
        0.0,
    )
    observed = float(tstats.max())
    centered = m - m.mean(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    offsets = np.arange(block)
    count = 0
    for _ in range(int(n_perms)):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]).ravel() % n)[:n]
        boot = centered[:, idx]
        boot_t = np.sqrt(n) * (boot.mean(axis=1) / np.maximum(stds, 1e-12))
        if float(boot_t.max()) >= observed:
            count += 1

    p_value = float((count + 1) / (int(n_perms) + 1))
    return {
        "observed_max_t": observed,
        "p_value": p_value,
        "n_perms": int(n_perms),
        "block_size": block,
        "n_trials": int(k),
        "n_dates": int(n),
        "seed": seed,
        "significant_95": bool(p_value <= 0.05),
    }


def _adjudication_window(current: list[dict]) -> list[str]:
    """W = the intersection of the current run's trial date indexes, in
    the chronological order of the first current trial (its segments are
    stitched in fold order)."""

    if not current:
        raise ValueError(
            "no current-run trials: the adjudication window is undefined"
        )
    shared = set(current[0]["dates"])
    for entry in current[1:]:
        shared &= set(entry["dates"])
    window = [d for d in current[0]["dates"] if d in shared]
    if not window:
        raise ValueError(
            "current-run trials share no common dates: the p40 joint "
            "matrix is undefined"
        )
    return window


def _max_t_from_pool(
    pool: list[dict],
    n_perms: int = 5000,
    seed: int = 0,
    block_size: int = 10,
) -> dict | None:
    """Joint studentized max-t block bootstrap over the dated pool.

    The joint matrix is built on the adjudication window W (§R3): pooled
    trials whose date index covers W are aligned to W by **date** (never
    by position); the others are recorded ``matrix_excluded`` with the
    reason and stay in the multiplicity count.  The resampling itself is
    :func:`max_t_from_matrix` (§R4): one shared date-block index per
    permutation.
    """

    if not pool:
        return None
    current = [entry for entry in pool if entry["current"]]
    window = _adjudication_window(current)
    w_index = {d: i for i, d in enumerate(window)}
    columns: list[np.ndarray] = []
    excluded: list[dict] = []
    for entry in pool:
        dates = entry["dates"]
        cover = set(dates)
        missing = [d for d in window if d not in cover]
        if missing:
            excluded.append(
                {
                    "candidate": entry["trial"].get("candidate"),
                    "seed": entry["trial"].get("seed"),
                    "reason": "window_mismatch",
                    "n_missing_dates": len(missing),
                }
            )
            continue
        pos = {d: i for i, d in enumerate(dates)}
        columns.append(entry["excess"][[pos[d] for d in window]])
    if not columns:
        raise ValueError(
            "no pooled trial covers the adjudication window "
            f"{window[0]}..{window[-1]} ({len(window)} dates): the joint "
            "matrix is undefined"
        )
    matrix = np.vstack(columns)
    stats = max_t_from_matrix(
        matrix, n_perms=n_perms, seed=seed, block_size=block_size
    )
    stats.update(
        {
            "window": {
                "first_date": window[0],
                "last_date": window[-1],
                "n_dates": len(window),
            },
            "matrix_excluded": excluded,
            "n_current": len(current),
            "n_extra": len(pool) - len(current),
            "n_trials": len(pool),
        }
    )
    return stats


def max_t_from_rows(
    rows: list[dict],
    n_perms: int = 5000,
    seed: int = 0,
    block_size: int = 10,
    extra_trial_rows: list[dict] | None = None,
) -> dict | None:
    """Studentized max-t over the **stitched**, date-aligned trial matrix
    (v20 + p40 R3/R4)."""

    return _max_t_from_pool(
        _stitched_pool(rows, extra_trial_rows),
        n_perms=n_perms,
        seed=seed,
        block_size=block_size,
    )


def selfcheck_rows(
    loader: AshareDataLoader,
    proto_cfg: ProtocolConfig,
    backtest_config: BacktestConfig,
    seed: int = 1234,
) -> list[dict]:
    """Pure-noise placeholder candidates: the protocol's dry-run acceptance.

    Noise signals are scored through the identical engine path; the DS and
    max-t corrections must then both report insignificance.  A deterministic
    RNG makes the self-check reproducible.
    """

    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for fold_cfg in proto_cfg.folds:
        fold = resolve_folds(
            [fold_cfg],
            loader.dates,
            frequency=proto_cfg.frequency,
            horizon=proto_cfg.horizon,
        )[0]
        _, _, _, dates = epoch_slice(loader, fold)
        signal = rng.normal(0.0, 1.0, size=(len(loader.ts_codes), len(dates)))
        metrics = evaluate_signal(signal, loader, fold, backtest_config)
        rows.append(
            {
                "candidate": "selfcheck:noise",
                "formula_text": "noise",
                "formula": None,
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "val_reward": None,
                "final_avg_reward": None,
                "failed": False,
                **metrics,
            }
        )
    return rows
