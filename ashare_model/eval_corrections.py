"""Multiple-testing corrections for the evaluation protocol.

Extracted from ``evaluation.py`` (P7 Phase A3) by reason-to-change: this
module owns the *statistical adjudication* over the stitched trial matrix —
Probabilistic/Deflated Sharpe (Bailey & Lopez de Prado 2014), the
studentized max-t block bootstrap, and the pure-noise self-check.  It
changes when correction *methodology* changes — not when fold contracts,
metric definitions, search backends or artifact schemas change.

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
) -> list[tuple[dict, np.ndarray]]:
    """The v20 trial matrix: (stitched trial, excess array) pairs.

    This run's rows and the extra rows from prior artifacts are stitched
    **separately** — a prior run's ``trained`` trial is a different trial
    from this run's, so they must never share a series.
    """

    pool: list[tuple[dict, np.ndarray]] = []
    for trial in stitch_oos_series(rows) + stitch_oos_series(extra_trial_rows or []):
        excess = np.asarray(trial["excess_returns"], dtype=np.float64)
        if excess.size >= 2:
            pool.append((trial, excess))
    return pool


def _dsr_from_pool(pool: list[tuple[dict, np.ndarray]]) -> dict | None:
    """Deflated Sharpe of the best stitched trial in the pool."""

    stats = []
    for trial, excess in pool:
        sr, skew, kurt, t = _trial_stats(excess)
        stats.append((trial, sr, skew, kurt, t))
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


def _max_t_from_pool(
    pool: list[tuple[dict, np.ndarray]],
    n_perms: int = 5000,
    seed: int = 0,
    block_size: int = 10,
) -> dict | None:
    """Studentized max-t block bootstrap (White-style reality check) over
    the stitched trial matrix.

    Each trial's excess series is recentered to the zero-mean null; per
    permutation, circular blocks of ``block_size`` days are resampled per
    trial, the bootstrapped mean is studentized by the observed per-trial
    standard deviation, and the max over trials forms the null distribution
    of the max t-statistic.  (Plain sign-flipping is not used: for a max
    statistic it can never fall below p = 0.5 by construction.)
    """

    series = [excess for _, excess in pool]
    if not series:
        return None

    stds = np.asarray(
        [float(s.std(ddof=1)) for s in series], dtype=np.float64
    )
    centered = [s - float(s.mean()) for s in series]
    tstats = np.asarray(
        [
            math.sqrt(s.size) * (float(s.mean()) / std)
            if std > 1e-12
            else 0.0
            for s, std in zip(series, stds)
        ],
        dtype=np.float64,
    )
    observed = float(tstats.max())

    rng = np.random.default_rng(seed)
    block = min(max(int(block_size), 1), min(s.size for s in series))
    count = 0
    for _ in range(int(n_perms)):
        boot_means = np.empty(len(series), dtype=np.float64)
        for i, s in enumerate(centered):
            n = s.size
            n_blocks = int(math.ceil(n / block))
            starts = rng.integers(0, n, size=n_blocks)
            offsets = np.arange(block)
            idx = (starts[:, None] + offsets[None, :]).ravel() % n
            boot = s[idx[:n]]
            boot_means[i] = float(boot.mean())
        boot_t = np.sqrt(
            np.asarray([s.size for s in series], dtype=np.float64)
        ) * (boot_means / np.maximum(stds, 1e-12))
        if float(boot_t.max()) >= observed:
            count += 1

    p_value = float((count + 1) / (int(n_perms) + 1))
    return {
        "observed_max_t": observed,
        "p_value": p_value,
        "n_perms": int(n_perms),
        "block_size": block,
        "n_trials": len(series),
        "seed": seed,
        "significant_95": bool(p_value <= 0.05),
    }


def max_t_from_rows(
    rows: list[dict],
    n_perms: int = 5000,
    seed: int = 0,
    block_size: int = 10,
    extra_trial_rows: list[dict] | None = None,
) -> dict | None:
    """Studentized max-t over the **stitched** trial matrix (v20)."""

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
