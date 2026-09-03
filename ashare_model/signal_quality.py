"""Robust signal-quality statistics and hard quality gates (T1-03).

The daily rank-IC series is the raw measurement; everything here makes
that measurement trustworthy:

* ``hac_variance`` / ``effective_n`` / ``robust_icir`` — the ICIR is
  shrunk by the effective sample size under autocorrelation
  (Newey-West-style HAC variance with Bartlett weights): a positively
  autocorrelated IC series carries less information than its raw length
  suggests, so its ICIR is discounted toward zero.
* ``block_bootstrap_ic`` — a circular block bootstrap over dates gives the
  bootstrap distribution of the mean IC: the CI and the sign-stability
  probability (P(mean > 0)) with a deterministic seed.
* ``effective_stock_count`` — the harmonic mean of the per-date eligible
  cross-section sizes (the classic effective-n for cross-sections).
* :class:`SignalQuality` — one summary per candidate window: coverage,
  activity, valid-IC days, effective stocks, naive and robust ICIR,
  effective-n, sign stability, and window aggregates (median / q25 of the
  validation-window ICIRs).
* :class:`SignalQualityGates` + :func:`evaluate_quality_gates` — the hard
  gates that reject degenerate signals: all-zero (activity), two-valid-day
  (valid-IC days), extremely sparse (effective stocks), sign instability
  and low validation-window lower quantile.

All statistics are computed on signal-date PIT-eligible cells only, so a
future member can never move them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reward import rank_ic_series

# Andrews (1991) style lag-length rule of thumb, clamped to sane bounds.
def _hac_lags(n: int, max_lags: int | None) -> int:
    if max_lags is not None:
        return max(1, min(int(max_lags), n - 1))
    return max(1, min(int(4.0 * (n / 100.0) ** (2.0 / 9.0)), n - 1))


def hac_variance(ic: np.ndarray, max_lags: int | None = None) -> float:
    """Newey-West (Bartlett) autocorrelation-consistent variance of ``ic``.

    ``v_hac = v * (1 + 2 * sum_k w_k * rho_k)`` with Bartlett weights
    ``w_k = 1 - k / (K + 1)``.  Degenerate series (fewer than two finite
    observations, zero variance) yield ``0.0``.
    """

    ic = np.asarray(ic, dtype=np.float64)
    ic = ic[np.isfinite(ic)]
    if ic.size < 2:
        return 0.0
    mean = float(ic.mean())
    dev = ic - mean
    var = float((dev @ dev) / (ic.size - 1))
    if var <= 0.0:
        return 0.0
    k = _hac_lags(ic.size, max_lags)
    acov = np.correlate(dev, dev, mode="full")[ic.size - 1 :]
    rho = acov[1 : k + 1] / (var * ic.size)  # biased autocovariances
    weights = 1.0 - np.arange(1, k + 1) / (k + 1)
    return float(var * (1.0 + 2.0 * float(weights @ rho)))


def effective_n(ic: np.ndarray, max_lags: int | None = None) -> float:
    """Effective sample size of an IC series under autocorrelation.

    ``n_eff = n * var / var_hac``: independent series keep ~n, positively
    autocorrelated series shrink toward the information they actually
    carry.  Returns ``0.0`` for degenerate series.
    """

    ic = np.asarray(ic, dtype=np.float64)
    finite = ic[np.isfinite(ic)]
    if finite.size < 2:
        return 0.0
    var = float(finite.var(ddof=1))
    if var <= 0.0:
        return 0.0
    return float(finite.size * var / max(hac_variance(finite, max_lags), 1e-12))


def robust_icir(ic: np.ndarray, max_lags: int | None = None) -> float:
    """ICIR with effective-sample-size shrinkage (HAC variance).

    ``icir_hac = naive_icir * sqrt(n_eff / n)`` — the plain mean/std ratio
    shrunk by the effective sample size under autocorrelation.  For
    independent IC series ``n_eff ~ n`` and the ratio is essentially
    unchanged (preserving the calibrated scale); positively autocorrelated
    series are discounted toward zero.  A perfectly constant IC series has
    zero variance and an unbounded ratio (``+/-inf``, clipped by the reward
    band); series with fewer than two finite observations score ``0.0``
    (an under-identified ratio).
    """

    ic = np.asarray(ic, dtype=np.float64)
    finite = ic[np.isfinite(ic)]
    if finite.size < 2:
        return 0.0
    mean = float(finite.mean())
    var = float(finite.var(ddof=1))
    if var <= 0.0:
        # Perfectly stable IC: the ratio is unbounded in the sign of mean.
        return float(np.inf if mean > 0.0 else (-np.inf if mean < 0.0 else 0.0))
    naive = mean / np.sqrt(var)
    return float(naive * np.sqrt(effective_n(finite, max_lags) / finite.size))


def block_bootstrap_ic(
    ic: np.ndarray,
    n_boot: int = 500,
    block_size: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Circular block bootstrap of the mean IC over dates.

    Returns the bootstrap mean, the 90% CI (``ci_low``/``ci_high``) and
    ``sign_stability`` — the bootstrap probability that the mean IC is
    positive, the principled answer to "is this signal's edge stable or a
    fluke of the sample".  Deterministic in ``seed``.
    """

    ic = np.asarray(ic, dtype=np.float64)
    finite = ic[np.isfinite(ic)]
    if finite.size < 2:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "sign_stability": 0.5,
            "n_boot": int(n_boot),
            "block_size": 0,
        }
    n = finite.size
    block = min(max(int(block_size), 1), n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boot_means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n] % n
        boot_means[i] = float(finite[idx].mean())
    ci_low, ci_high = np.quantile(boot_means, [0.05, 0.95])
    return {
        "mean": float(finite.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "sign_stability": float((boot_means > 0.0).mean()),
        "n_boot": int(n_boot),
        "block_size": int(block),
    }


def effective_stock_count(per_date_counts: np.ndarray) -> float:
    """Harmonic-mean effective cross-section size (0 for empty input)."""

    counts = np.asarray(per_date_counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    return float(1.0 / np.mean(1.0 / counts))


@dataclass(frozen=True)
class SignalQuality:
    """One candidate's quality summary over the training window plus the
    validation windows (all statistics on signal-date eligible cells)."""

    ic_mean: float
    ic_abs_mean: float
    icir: float  # naive mean/std ICIR
    icir_hac: float  # effective-n shrunk ICIR
    effective_n: float
    valid_ic_days: int
    coverage: float  # eligible cells with a finite signal
    activity: float  # eligible finite cells that are non-zero
    effective_stocks: float  # harmonic-mean cross-section size
    sign_stability: float  # P(mean IC > 0) from the block bootstrap
    window_icir_median: float | None  # median of validation-window ICIRs
    window_icir_q25: float | None  # lower quartile of validation-window ICIRs
    window_sign_agreement: float | None  # fraction of windows whose IC
    # mean agrees with the training window's sign


@dataclass(frozen=True)
class SignalQualityGates:
    """Hard quality thresholds.  ``None`` disables a gate."""

    min_valid_ic_days: int | None = 8
    min_effective_stocks: int | None = None  # None -> ic_min_stocks
    min_coverage: float | None = 0.2
    min_activity: float | None = 0.05
    min_sign_stability: float | None = 0.5
    min_val_window_q25: float | None = 0.0


def evaluate_quality_gates(
    quality: SignalQuality, gates: SignalQualityGates
) -> tuple[bool, tuple[str, ...]]:
    """Hard-gate check of one summary.  Returns ``(ok, rejection_reasons)``
    with the exact reason strings the candidate scorer records."""

    reasons: list[str] = []
    if gates.min_valid_ic_days is not None and (
        quality.valid_ic_days < int(gates.min_valid_ic_days)
    ):
        reasons.append("valid_ic_days_below_minimum")
    min_stocks = gates.min_effective_stocks
    if min_stocks is not None and quality.effective_stocks < int(min_stocks):
        reasons.append("effective_stocks_below_minimum")
    if gates.min_coverage is not None and quality.coverage < float(
        gates.min_coverage
    ):
        reasons.append("coverage_below_minimum")
    if gates.min_activity is not None and quality.activity < float(
        gates.min_activity
    ):
        reasons.append("signal_activity_below_minimum")
    if gates.min_sign_stability is not None and (
        quality.sign_stability < float(gates.min_sign_stability)
    ):
        reasons.append("sign_instability")
    if gates.min_val_window_q25 is not None and (
        quality.window_icir_q25 is None
        or quality.window_icir_q25 < float(gates.min_val_window_q25)
    ):
        reasons.append("val_window_q25_below_minimum")
    return (not reasons, tuple(reasons))


def _window_ic_summaries(
    signals: np.ndarray,
    target: np.ndarray,
    window: tuple[int, int],
    ic_min_stocks: int,
    universe_mask: np.ndarray,
    max_lags: int | None,
    n_boot: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row IC series, robust ICIR, bootstrap sign stability and
    effective-n for one ``[batch, stock, date]`` window slice."""

    start, end = window
    ics = rank_ic_series(
        signals[:, :, start:end],
        target[:, start:end],
        ic_min_stocks,
        universe_mask=universe_mask[:, start:end],
    )  # [B, dates]
    b = ics.shape[0]
    icir_hac = np.empty(b, dtype=np.float64)
    eff_n = np.empty(b, dtype=np.float64)
    sign = np.empty(b, dtype=np.float64)
    for i in range(b):
        icir_hac[i] = robust_icir(ics[i], max_lags)
        eff_n[i] = effective_n(ics[i], max_lags)
        sign[i] = block_bootstrap_ic(
            ics[i], n_boot=n_boot, block_size=block_size, seed=0
        )["sign_stability"]
    return ics, icir_hac, eff_n, sign


def summarize_signal_quality_batch(
    signals: np.ndarray,
    target: np.ndarray,
    val_windows: list[tuple[int, int]],
    train_signal_range: tuple[int, int],
    *,
    universe_mask: np.ndarray,
    ic_min_stocks: int = 10,
    max_lags: int | None = None,
    n_boot: int = 500,
    block_size: int = 5,
) -> list[SignalQuality]:
    """Per-candidate :class:`SignalQuality` over the training window plus
    every validation window (hard gates and artifacts consume these).

    ``signals`` is ``[B, stocks, dates]`` aligned with ``target`` and
    ``universe_mask``.  The training-window summary covers
    ``train_signal_range``; the validation windows' robust ICIRs are
    aggregated into the median and lower quartile, and each window's IC
    mean sign is compared against the training window's sign.
    """

    signals = np.asarray(signals, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if signals.ndim != 3 or signals.shape[1:] != target.shape:
        raise ValueError("signals must be [batch, stock, date] aligned with target")
    if universe_mask.shape != target.shape:
        raise ValueError("universe_mask must be [stock, date] aligned with target")

    train_start, train_end = train_signal_range
    train_ics, train_icir_hac, train_eff_n, train_sign = _window_ic_summaries(
        signals, target, (train_start, train_end), ic_min_stocks,
        universe_mask, max_lags, n_boot, block_size,
    )
    b = signals.shape[0]

    summaries: list[SignalQuality] = []
    for i in range(b):
        train_ic = train_ics[i]
        finite_train = train_ic[np.isfinite(train_ic)]
        ic_mean = float(finite_train.mean()) if finite_train.size else 0.0
        ic_abs_mean = float(np.abs(finite_train).mean()) if finite_train.size else 0.0
        icir = float(
            finite_train.mean() / finite_train.std(ddof=1)
            if finite_train.size >= 2 and finite_train.std(ddof=1) > 0
            else 0.0
        )
        train_sign_mean = 1.0 if ic_mean >= 0.0 else -1.0

        signal_slice = signals[i]
        eligible = universe_mask
        finite_cells = np.isfinite(signal_slice) & eligible
        covered = int(finite_cells.sum())
        total_eligible = int(eligible.sum())
        non_zero = int((finite_cells & (signal_slice != 0.0)).sum())
        coverage = covered / total_eligible if total_eligible else 0.0
        activity = non_zero / covered if covered else 0.0
        per_date_counts = finite_cells.sum(axis=0)
        eff_stocks = effective_stock_count(per_date_counts)

        window_icirs: list[float] = []
        window_signs: list[float] = []
        for start, end in val_windows:
            window_ics = rank_ic_series(
                signal_slice[None, :, start:end],
                target[:, start:end],
                ic_min_stocks,
                universe_mask=universe_mask[:, start:end],
            )[0]
            finite_window = window_ics[np.isfinite(window_ics)]
            if finite_window.size == 0:
                continue
            window_icirs.append(robust_icir(window_ics, max_lags))
            window_signs.append(
                1.0 if float(finite_window.mean()) >= 0.0 else -1.0
            )
        window_icir_median = (
            float(np.median(window_icirs)) if window_icirs else None
        )
        # IP-10 01 (itemized in docs/test_runtime_measurement_log.md):
        # robust_icir may legitimately return +/-inf for degenerate
        # windows; when the quantile point lands on two inf order
        # statistics, numpy's quantile lerp computes inf - inf ("invalid
        # value encountered in scalar subtract") and returns NaN -- the
        # NaN summary is the intended value, so the FP-state warning is
        # muted at the call site (np.errstate never changes the computed
        # values).
        with np.errstate(invalid="ignore"):
            window_icir_q25 = (
                float(np.quantile(window_icirs, 0.25)) if window_icirs else None
            )
        window_sign_agreement = (
            float(np.mean([1.0 if s == train_sign_mean else 0.0 for s in window_signs]))
            if window_signs
            else None
        )
        summaries.append(
            SignalQuality(
                ic_mean=ic_mean,
                ic_abs_mean=ic_abs_mean,
                icir=icir,
                icir_hac=float(train_icir_hac[i]),
                effective_n=float(train_eff_n[i]),
                valid_ic_days=int(finite_train.size),
                coverage=coverage,
                activity=activity,
                effective_stocks=eff_stocks,
                sign_stability=float(train_sign[i]),
                window_icir_median=window_icir_median,
                window_icir_q25=window_icir_q25,
                window_sign_agreement=window_sign_agreement,
            )
        )
    return summaries
