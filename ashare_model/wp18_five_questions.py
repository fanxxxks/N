"""WP18 (p56) — the report five-question math (FROZEN_18_PLAN §5/§6.2/§7).

Pure functions only — no engine, no loader, no IO.  Each helper pins one
frozen reporting口径:

* Q1 (§7.1) ``quantile_spread`` — per-date cross-sectional quantile
  buckets of the oriented signal vs the fixed causal label, Q-high minus
  Q-low; consumes the SAME rank authority as the IC path
  (``ashare_model.reward._rankdata``) and the same PIT-universe /
  min-stocks gating;
* Q4 (§7.4) ``cash_exposure_decomposition`` — net excess split into
  exposure/cash effect vs selection effect using the ACTUAL post-trade
  holdings (never target weights); arithmetic cumulative contribution,
  geometric cumulative excess and annualized are separate outputs
  (§7 通用口径: three quantities, three columns);
* §5 (iii)/§6.2 — per-leg annualization ``(1+累计)^(1/5)−1`` then the
  dynamic-minus-hold difference, the 50bp gate, the frozen disclaimer
  and the drop-max-contribution-year variant;
* Q5 (§7.5) — fixed-path fee repricing (same executed notionals, new
  rates via ``r01_baseline.cost_variant`` — the single cost authority)
  kept strictly separate from a re-executed stress run; the difference
  is reported as the path effect.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ashare_model.wp18_signal_vs_account import MECHANICAL_SCREEN_ANNOTATION

#: The additive funnel counter name for hold-mode forced-exit sessions
#: (FROZEN_18_PLAN §3.2 观测面: additive key, existing key names untouched).
HOLD_FORCED_EXIT_SESSIONS_KEY = "funnel_hold_forced_exit_sessions"


def hold_forced_exit_session_count(
    construction_diagnostics: Sequence[Mapping[str, Any]],
) -> int:
    """Sessions whose construction ran a hold-mode forced-exit leg —
    either the non-due leg (``not_due_forced_exit``) or the post-funding
    due leg (``hold_forced_exit``).  Read-only over the engine's
    diagnostics; consumed by eval_metrics' additive funnel key."""

    return int(
        sum(
            1
            for session in construction_diagnostics
            if bool(session.get("hold_forced_exit"))
        )
    )


# ---------------------------------------------------------------------------
# Q1 — quantile spread (§7.1)
# ---------------------------------------------------------------------------


def quantile_spread(
    signal: np.ndarray,
    target: np.ndarray,
    *,
    universe_mask: np.ndarray,
    n_quantiles: int = 5,
    min_stocks: int = 10,
) -> dict[str, Any]:
    """Per-date Q-high minus Q-low mean label along the oriented signal.

    For each date: take the eligible & finite cells, rank the signal
    cross-section (``reward._rankdata`` — the same Spearman authority the
    IC path uses), bucket into ``n_quantiles`` equal buckets, average the
    label within each bucket and subtract the LOWEST signal bucket's mean
    from the HIGHEST's.  A date with fewer than ``min_stocks`` usable
    cells yields NaN and is excluded from ``n_dates``/``mean_spread``."""

    from ashare_model.reward import _rankdata

    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    if signal.shape != target.shape or universe_mask.shape != target.shape:
        raise ValueError(
            "signal/target/universe_mask shapes must all match "
            f"(got {signal.shape}, {target.shape}, {universe_mask.shape})"
        )
    n_dates = signal.shape[1]
    spreads = np.full(n_dates, np.nan)
    for day in range(n_dates):
        usable = universe_mask[:, day] & np.isfinite(signal[:, day]) & np.isfinite(
            target[:, day]
        )
        if int(usable.sum()) < min_stocks:
            continue
        values = signal[usable, day]
        labels = target[usable, day]
        ranks = _rankdata(values)
        # bucket 0 = lowest signal, n_quantiles-1 = highest
        buckets = np.minimum(
            ((ranks - 1.0) / len(values) * n_quantiles).astype(int),
            n_quantiles - 1,
        )
        high = labels[buckets == n_quantiles - 1]
        low = labels[buckets == 0]
        if high.size == 0 or low.size == 0:
            continue
        spreads[day] = float(np.mean(high) - np.mean(low))
    finite = spreads[np.isfinite(spreads)]
    return {
        "spread_series": [float(v) if np.isfinite(v) else None
                          for v in spreads],
        "mean_spread": float(np.mean(finite)) if finite.size else 0.0,
        "n_dates": int(finite.size),
        "n_quantiles": int(n_quantiles),
        "min_stocks": int(min_stocks),
    }


# ---------------------------------------------------------------------------
# Q4 — cash-exposure vs selection decomposition (§7.4)
# ---------------------------------------------------------------------------


def cash_exposure_decomposition(
    actual_weights: np.ndarray,
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Decompose per-date net excess into exposure/cash vs selection.

    Inputs: ``actual_weights`` — the ACTUAL post-trade holdings (the
    engine's achieved/drifted book, NOT the constructor target); the
    uninvested residual ``1 − Σw`` is the cash weight earning zero.
    ``asset_returns`` — per-name returns for the same dates;
    ``benchmark_returns`` — the equal-weight full-investment benchmark
    return per date (the row's own benchmark series).

    Per date, with ``E = Σw`` and the within-book mix ``v = w / E``:

    * exposure/cash effect = ``(E − 1) · Σ v·r``  (the under/over-
      investment drag relative to a fully invested book of the same
      mix; for a pure cash book this is exactly ``−benchmark``);
    * selection effect = ``Σ v·r − benchmark`` (the name-choice tilt of
      the held mix at full investment).

    The two effects add to the exact net excess ``Σw·r − benchmark`` by
    construction.  Summary scalars are reported as three separate
    quantities (§7 通用口径): arithmetic cumulative contribution,
    geometric cumulative excess, annualized excess."""

    weights = np.asarray(actual_weights, dtype=np.float64)
    returns = np.asarray(asset_returns, dtype=np.float64)
    benchmark = np.asarray(benchmark_returns, dtype=np.float64)
    if weights.shape != returns.shape:
        raise ValueError(
            f"actual_weights shape {weights.shape} does not match "
            f"asset_returns {returns.shape}"
        )
    if benchmark.ndim != 1 or benchmark.shape[0] != returns.shape[1]:
        raise ValueError(
            f"benchmark_returns must be 1-D aligned with the date axis "
            f"({returns.shape[1]},), got {benchmark.shape}"
        )
    exposure = weights.sum(axis=0)  # E per date
    book_return = (weights * returns).sum(axis=0)  # Σ w·r = r_actual
    # Within-book mix return Σ v·r with v = w / E.  For an empty book
    # (E = 0) the mix is undefined; the convention that keeps the
    # identity exact AND the economics honest is the benchmark mix: a
    # name-less book has no selection tilt, so its entire excess is the
    # exposure/cash effect (a pure cash book reads exactly −benchmark).
    mix_return = np.where(
        exposure > 0.0,
        np.divide(
            book_return,
            np.where(exposure > 0.0, exposure, 1.0),
        ),
        benchmark,
    )
    exposure_effect = (exposure - 1.0) * mix_return
    selection_effect = mix_return - benchmark
    excess = book_return - benchmark
    if not np.allclose(exposure_effect + selection_effect, excess, atol=1e-12):
        raise ValueError("decomposition identity violated")  # defensive
    n_periods = excess.size
    geometric = float(np.prod(1.0 + excess) - 1.0) if n_periods else 0.0
    annualized = (
        float((1.0 + geometric) ** (periods_per_year / max(n_periods, 1)) - 1.0)
        if n_periods and geometric > -1.0
        else -1.0
    )
    return {
        "exposure_effect": [float(v) for v in exposure_effect],
        "selection_effect": [float(v) for v in selection_effect],
        "net_excess": [float(v) for v in excess],
        "arithmetic_cumulative_excess": float(np.sum(excess)) if n_periods else 0.0,
        "geometric_cumulative_excess": geometric,
        "annualized_excess": annualized,
        "arithmetic_cumulative_exposure_effect": (
            float(np.sum(exposure_effect)) if n_periods else 0.0
        ),
        "arithmetic_cumulative_selection_effect": (
            float(np.sum(selection_effect)) if n_periods else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# §5 (iii) / §6.2 — annualization, the mechanical screen, drop-max-year
# ---------------------------------------------------------------------------


def arithmetic_cumulative(series: Sequence[float]) -> float:
    """The arithmetic cumulative contribution (kept separate from the
    geometric reading by construction, §7 通用口径)."""

    values = np.asarray(series, dtype=np.float64)
    return float(values.sum()) if values.size else 0.0


def geometric_cumulative_excess(series: Sequence[float]) -> float:
    """The geometric cumulative excess ``Π(1+e) − 1``."""

    values = np.asarray(series, dtype=np.float64)
    if not values.size:
        return 0.0
    return float(np.prod(1.0 + values) - 1.0)


def annualized_from_cumulative(cumulative: float, *, years: int = 5) -> float:
    """§5 (iii): a leg's annualization ``(1+累计净超额)^(1/年数) − 1``."""

    years = int(years)
    if years <= 0:
        raise ValueError("years must be >= 1")
    if cumulative <= -1.0:
        return -1.0
    return float((1.0 + float(cumulative)) ** (1.0 / years) - 1.0)


def dynamic_minus_hold_annualized(
    dynamic_cumulative: float, hold_cumulative: float, *, years: int = 5
) -> float:
    """§5 (iii): each leg annualized FIRST, then differenced (the frozen
    order — never difference the cumulatives then annualize)."""

    return annualized_from_cumulative(
        dynamic_cumulative, years=years
    ) - annualized_from_cumulative(hold_cumulative, years=years)


def drop_max_contribution_year(
    annual_series: Sequence[float],
) -> tuple[np.ndarray, float]:
    """§6.2: the series with its largest single-year contribution
    removed, plus the dropped value.  Ties drop the first occurrence."""

    values = np.asarray(annual_series, dtype=np.float64)
    if not values.size:
        return values.copy(), 0.0
    index = int(np.argmax(values))
    dropped = float(values[index])
    return np.delete(values, index), dropped


def mechanical_screen(
    annual_net_excess: Sequence[float],
    dynamic_cumulative: float,
    hold_cumulative: float,
    *,
    years: int = 5,
    threshold_bp: float = 50.0,
) -> dict[str, Any]:
    """§5/§6.2, verbatim: (i) ≥3/5 test years with positive net excess,
    (ii) positive median, (iii) dynamic−hold annualized ≥ 50bp — plus the
    drop-max-contribution-year variant of (i)/(ii).  The annotation is
    frozen: this is a RESEARCH-RESOURCE-ALLOCATION gate, not a
    statistical significance proof and not a market law."""

    values = np.asarray(annual_net_excess, dtype=np.float64)
    n_positive = int(np.count_nonzero(values > 0.0))
    median = float(np.median(values)) if values.size else 0.0
    diff = dynamic_minus_hold_annualized(
        dynamic_cumulative, hold_cumulative, years=years
    )
    condition_i = n_positive >= 3
    condition_ii = median > 0.0
    condition_iii = diff >= threshold_bp / 10000.0
    remaining, dropped = drop_max_contribution_year(values)
    ex_positive = int(np.count_nonzero(remaining > 0.0))
    ex_median = float(np.median(remaining)) if remaining.size else 0.0
    return {
        "passed": bool(condition_i and condition_ii and condition_iii),
        "n_positive_years": n_positive,
        "min_positive_years": 3,
        "condition_i_3_of_5_positive": bool(condition_i),
        "median_net_excess": median,
        "condition_ii_median_positive": bool(condition_ii),
        "dynamic_minus_hold_annualized": float(diff),
        "dynamic_minus_hold_threshold": float(threshold_bp) / 10000.0,
        "condition_iii_dynamic_minus_hold_50bp": bool(condition_iii),
        "excluded_year_excess": dropped,
        "ex_minus_max_year": {
            "n_positive_years": ex_positive,
            "median_net_excess": ex_median,
            "n_years": int(remaining.size),
        },
        "annotation": MECHANICAL_SCREEN_ANNOTATION,
    }


# ---------------------------------------------------------------------------
# Q5 — fixed-path fee repricing vs re-executed stress (§7.5)
# ---------------------------------------------------------------------------


def reprice_fixed_path_fees(
    buy_notionals: Sequence[float],
    sell_notionals: Sequence[float],
    backtest_config,
    level: str,
) -> float:
    """Re-price the fees of an ALREADY-EXECUTED 1x path at another cost
    level: same buy/sell notionals, rates from the single cost authority
    ``r01_baseline.cost_variant`` (0x = all parameters zero; 2x = doubled
    rates with the commission floor unchanged).  This answers the
    DIRECT-fee question only — cash/lot/budget path changes are the
    re-execution reading's business (§7.5: the two are reported
    separately and never merged)."""

    from ashare_execution import ExecutionCostModel
    from ashare_model.r01_baseline import cost_variant

    variant = cost_variant(backtest_config, level)
    model = ExecutionCostModel.from_config(variant)
    buys = np.asarray(buy_notionals, dtype=np.float64)
    sells = np.asarray(sell_notionals, dtype=np.float64)
    total = 0.0
    if buys.size:
        total += float(
            np.sum(model.buy_cost(buys).total)
        )
    if sells.size:
        total += float(np.sum(model.sell_cost(sells).total))
    return total


def split_cost_q5_effects(
    *,
    fixed_path_net_excess: float,
    reexecuted_net_excess: float,
) -> dict[str, float]:
    """§7.5: report the fixed-path reading, the re-executed reading and
    their difference (the path effect) as three separate numbers."""

    fixed = float(fixed_path_net_excess)
    reexec = float(reexecuted_net_excess)
    return {
        "direct_fee_reading": fixed,
        "reexecuted_reading": reexec,
        "path_effect": reexec - fixed,
    }
