"""P34 R01 round-1 fixed-baseline diagnostic driver.

Executable projection of ``docs/p34_r01_fixed_baseline_diagnostic_draft.md``
(APPROVED via user direct instruction 2026-09-16 「完成R01」, p30–p33
precedent).  The plan's R01 work package: **baselines complete and
recorded before any search**, on a frozen medium Tier-A configuration.

What this module does (and deliberately does not do):

* walks one run through the real lifecycle chain — genesis (typed
  IdeaRecord) → SPEC_LOCKED (frozen RunSpec over the A02-confirmed
  composite identity) → DATA_QUALIFIED (real G1–G8 gate result + the
  typed loader-frame status, p8 §5.1) → FACTOR_SET_QUALIFIED (real
  §5.2 report with the baseline/ablation evidence produced here);
* produces the frozen baseline arms **through the single semantic
  paths** — ``benchmark_row`` / ``baseline_candidates`` / ``evaluate_signal``
  (the same engine the protocol scores every candidate with), plus the
  frozen rule combos and the cash control built from the same fold
  slices — at the 0x/1x/2x cost ladder;
* registers the report inside the run store (content-addressed
  artifact), writes the measurement-directory copy, and applies the
  frozen adjudication: a positive verdict parks the run at
  ``FACTOR_SET_QUALIFIED`` (round 2 needs its own explicit
  authorization — this module never admits a search plan), a negative
  verdict is a legal ``RETIRED`` result;
* it does NOT run any searcher, does NOT consume the unique-semantic
  budget, does NOT touch production defaults, and does NOT claim
  holdout/promotion semantics for the 2021–2025 development windows.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ashare_data.gates import (
    ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
    ProductionGateRunner,
)
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION

#: Report artifact schema (measurement artifact; not an
#: ``artifact_schemas`` consumer — see p34 §4 version impact table).
#: v2 (WP3, contract rev 3): stitched cross-fold statistics, the annual
#: distribution, the execution funnel, explicit statuses and the
#: descriptive four-way diagnostic.  v1 was never produced.
# p49 erratum 1: bumped 2 -> 3 - the ablation evidence for a
# SINGLE-family factor set falls back to leave-one-signal-out within
# the family (the family ablation is structurally impossible there;
# prospective, preregistered before any economic result existed).
R01_REPORT_SCHEMA_VERSION = 3

#: p34 §3.7 frozen adjudication thresholds (never tuned after seeing data).
R01_ADJUDICATION_RULES: dict[str, float] = {
    "min_positive_folds": 3,
    "min_avg_holdings": 5.0,
    "min_invested_ratio": 0.5,
}

#: p34 §4 Rev 3 descriptive diagnostic thresholds (WP3).  These classify
#: the REPORTED evidence (weak signal / execution blocked / cost
#: swallowed / measurement incomplete); they are not the §3.7
#: adjudication and never decide worth_search.  Carried verbatim inside
#: every report so the classification is self-describing.
R01_DIAGNOSTIC_RULES: dict[str, object] = {
    "fully_suppressed_share_max": 0.10,
    "unfilled_order_rate_max": 0.10,
    "min_invested_ratio": 0.5,
    "priority": [
        "measurement_incomplete",
        "weak_signal",
        "execution_blocked",
        "cost_swallowed",
        "net_space_shown",
    ],
}

#: p34 §3.2 frozen combo construction (members ⊆ the config's preregistered
#: baseline signals; directions come from the same train-window rule the
#: single-factor rows use — frozen at preregistration time, not by OOS).
R01_COMBO_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("combo:all3", ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")),
    ("combo:momentum_reversal", ("MOMENTUM_20", "REVERSAL_60")),
)

COST_LEVELS = ("0x", "1x", "2x")

_ARM_PROBE_KEYS = (
    "excess_return",
    "active_ir",
    "holdings_mean",
    "invested_ratio_mean",
    "cost_fraction_sum",
    "rebalance_due_count",
    "rebalance_count",
)


class R01Refusal(RuntimeError):
    """A fail-closed refusal of the R01 round-1 run (p34 §7)."""


# --- frozen cost ladder -------------------------------------------------------


def cost_variant(backtest_config, level: str):
    """The p34 §3.3 cost ladder: ``0x`` gross (every cost parameter zero),
    ``1x`` the configuration verbatim, ``2x`` doubled cost rates with the
    commission floor unchanged.  Unknown levels are a construction error;
    the caller's config is never mutated."""

    if level == "1x":
        return backtest_config
    if level == "0x":
        return dataclasses.replace(
            backtest_config,
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            slippage_rate=0.0,
        )
    if level == "2x":
        return dataclasses.replace(
            backtest_config,
            commission_rate=2.0 * backtest_config.commission_rate,
            stamp_tax_rate=2.0 * backtest_config.stamp_tax_rate,
            transfer_fee_rate=2.0 * backtest_config.transfer_fee_rate,
            slippage_rate=2.0 * backtest_config.slippage_rate,
        )
    raise ValueError(f"unknown cost level {level!r}: expected 0x/1x/2x")


# --- frozen combo construction ------------------------------------------------


def rank_normalize_signal(
    signal: np.ndarray, universe_mask: np.ndarray
) -> np.ndarray:
    """Per-date cross-sectional rank normalization over the eligible
    finite cells (p34 §3.2).  Ranks come from ``reward._rankdata`` — the
    single rank authority (IP-14); cells outside the universe mask or
    not finite stay NaN, and a fully empty cross-section stays NaN."""

    from .reward import _rankdata

    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(universe_mask, dtype=bool)
    if signal.shape != mask.shape:
        raise ValueError(
            f"signal shape {signal.shape} does not match universe mask "
            f"{mask.shape}"
        )
    out = np.full(signal.shape, np.nan)
    for j in range(signal.shape[1]):
        col = signal[:, j]
        sel = mask[:, j] & np.isfinite(col)
        if sel.any():
            out[sel, j] = _rankdata(col[sel])
    return out


def percentile_normalize_signal(
    signal: np.ndarray, universe_mask: np.ndarray
) -> np.ndarray:
    """Scale-free per-date normalization: ``avg_rank / N_eff`` in (0, 1].

    p46 B5: the raw ranks of :func:`rank_normalize_signal` span
    ``1..N_eff``, so members with different effective cross-sections
    contribute at different scales to :func:`combo_signal`'s average.
    This helper divides by the per-date effective ``N`` so every member
    contributes on one common percentile scale.  It defines a NEW
    candidate family (``combo:all3_pct``); it is deliberately NOT wired
    into ``R01_COMBO_ARMS`` — the raw-rank combination's semantics stay
    bit-identical (no silent replacement).
    """

    ranks = rank_normalize_signal(signal, universe_mask)
    mask = np.asarray(universe_mask, dtype=bool)
    finite = np.isfinite(np.asarray(signal, dtype=np.float64))
    n_eff = (mask & finite).sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = ranks / np.where(n_eff > 0, n_eff, np.nan)
    return out


def combo_signal(
    members: list[tuple[float, np.ndarray]], universe_mask: np.ndarray
) -> np.ndarray:
    """Equal-weight mean of the rank-normalized signed member signals
    (p34 §3.2).  A cell averages the members that have a finite rank
    there; a cell with no available member stays NaN."""

    if not members:
        raise ValueError("combo_signal needs at least one member")
    normalized = [
        rank_normalize_signal(
            float(direction) * np.asarray(signal, dtype=np.float64),
            universe_mask,
        )
        for direction, signal in members
    ]
    stack = np.stack(normalized)  # [members, stocks, dates]
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    totals = np.where(finite, stack, 0.0).sum(axis=0)
    combo = np.full(counts.shape, np.nan)
    has = counts > 0
    combo[has] = totals[has] / counts[has]
    return combo


def cash_signal(n_stocks: int, n_dates: int) -> np.ndarray:
    """The p34 §3.2 cash control: an all-NaN signal the engine cannot
    select from — the book that holds nothing, trades nothing and pays
    nothing, exposing any 'low-cost advantage' that is really just an
    empty book."""

    if n_stocks <= 0 or n_dates <= 0:
        raise ValueError("cash_signal needs a positive shape")
    return np.full((n_stocks, n_dates), np.nan)


# --- row hygiene ----------------------------------------------------------------


def rows_to_jsonable(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` so every report row
    serializes as strict JSON (artifact writes reject NaN/Infinity)."""

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: rows_to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rows_to_jsonable(value) for value in obj]
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return rows_to_jsonable(obj.tolist())
    return obj


# --- baseline arms --------------------------------------------------------------


def run_baseline_arms(
    loader,
    proto_cfg,
    folds: list,
    backtest_config,
    model_config,
    reward_config,
    *,
    on_fold=None,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """All frozen arms × folds × cost levels through the shared engine.

    Returns ``(rows, directions_by_fold)`` where directions map
    ``fold_test_end -> {factor_name: direction}`` (the train-window rule
    outputs, decided once at 1x and reused verbatim).  ``on_fold`` is a
    progress/budget callback receiving ``(index, total, fold)``; an
    exception from it propagates (the budget guard must fail the run, not
    one fold).

    p34 §3.5 failure accounting: every engine invocation is captured per
    (arm, fold, level) — a failure appends a ``failed`` row with its
    reason instead of aborting the round; ``summarize_arms`` consumes
    those rows through the failed_folds/unreliable counters.
    """

    from .eval_folds import epoch_slice
    from .eval_metrics import benchmark_row, evaluate_signal
    from .eval_search import baseline_candidates
    from .vocab import FEATURE_NAMES

    rows: list[dict] = []
    directions_by_fold: dict[str, dict[str, int]] = {}
    total = len(folds)
    for index, fold in enumerate(folds):
        if on_fold is not None:
            on_fold(index, total, fold)

        def _ok(candidate: str, level: str, payload: dict) -> dict:
            payload.update(
                {
                    "candidate": candidate,
                    "formula_text": candidate,
                    "fold_train_end": fold.train_end,
                    "fold_test_end": fold.test_end,
                    "seed": None,
                    "cost_level": level,
                    "failed": False,
                }
            )
            return payload

        def _failed_row(candidate: str, level: str, exc: Exception) -> dict:
            return {
                "candidate": candidate,
                "formula_text": candidate,
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "cost_level": level,
                "failed": True,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }

        # The cost-free equal-weight reference (once — it is the common
        # benchmark every cost level is measured against; tagged "ref"
        # because it does not participate in the cost ladder, p34 §3.3).
        try:
            row = benchmark_row(loader, fold)
            rows.append(_ok("benchmark:equal_weight", "ref", row))
        except Exception as exc:  # noqa: BLE001 — recorded per §3.5
            rows.append(
                _failed_row("benchmark:equal_weight", "ref", exc)
            )

        # 1x single-factor rows + the frozen train-window directions.
        singles_failed: Exception | None = None
        singles: list[dict] = []
        try:
            singles = baseline_candidates(
                loader, proto_cfg, fold, backtest_config, model_config,
                reward_config,
            )
        except Exception as exc:  # noqa: BLE001 — recorded per §3.5
            singles_failed = exc
        directions: dict[str, int] = {}
        for r in singles:
            _ok(str(r["candidate"]), "1x", r)
            rows.append(r)
            directions[str(r["candidate"]).split(":", 1)[1]] = int(
                r["direction"]
            )
        if singles_failed is not None:
            for name in proto_cfg.baseline_signals:
                rows.append(
                    _failed_row(f"baseline:{name}", "1x", singles_failed)
                )
        directions_by_fold[str(fold.test_end)] = dict(directions)

        fold_data = epoch_slice(loader, fold)
        factors = fold_data.factors
        mask = fold_data.universe_mask

        for level in COST_LEVELS:
            if level == "1x":
                continue  # singles already measured on the 1x path above
            variant = cost_variant(backtest_config, level)
            for name in proto_cfg.baseline_signals:
                candidate = f"baseline:{name}"
                try:
                    direction = directions[name]
                    idx = FEATURE_NAMES.index(name)
                    signal = float(direction) * np.asarray(
                        factors[idx], dtype=np.float64
                    )
                    metrics = evaluate_signal(signal, loader, fold, variant)
                    rows.append(_ok(candidate, level, metrics))
                except Exception as exc:  # noqa: BLE001 — per §3.5
                    rows.append(_failed_row(candidate, level, exc))
        # Combos: the frozen construction, evaluated at every cost level.
        for level in COST_LEVELS:
            variant = cost_variant(backtest_config, level)
            for combo_name, member_names in R01_COMBO_ARMS:
                try:
                    members = []
                    for name in member_names:
                        members.append(
                            (directions[name], factors[FEATURE_NAMES.index(name)])
                        )
                    signal = combo_signal(members, mask)
                    metrics = evaluate_signal(signal, loader, fold, variant)
                    rows.append(_ok(combo_name, level, metrics))
                except Exception as exc:  # noqa: BLE001 — per §3.5
                    rows.append(_failed_row(combo_name, level, exc))
        # The cash control at every level (definitionally identical; run
        # mechanically instead of special-casing it).
        cash = cash_signal(len(loader.ts_codes), len(fold_data.dates))
        for level in COST_LEVELS:
            variant = cost_variant(backtest_config, level)
            try:
                metrics = evaluate_signal(cash, loader, fold, variant)
                rows.append(_ok("control:cash", level, metrics))
            except Exception as exc:  # noqa: BLE001 — per §3.5
                rows.append(_failed_row("control:cash", level, exc))
    return rows, directions_by_fold


#: Funnel count fields summed into ``funnel_totals_1x`` (p34 rev 3): the
#: per-row additive counters that aggregate by summation across folds.
_FUNNEL_SUM_KEYS = (
    "funnel_sessions_total",
    "funnel_not_due_sessions",
    "funnel_due_sessions",
    "funnel_due_no_signal_sessions",
    "funnel_due_target_stable_sessions",
    "funnel_due_fully_suppressed_sessions",
    "funnel_due_partial_suppressed_sessions",
    "funnel_due_executed_clean_sessions",
    "funnel_planned_order_count",
    "funnel_filled_order_count",
    "funnel_filled_buy_count",
    "funnel_filled_sell_count",
    "funnel_suppressed_threshold_count",
    "funnel_suppressed_min_trade_count",
    "funnel_sessions_zero_holdings",
)


def _stitched_level_metrics(arm_rows: list[dict], level: str) -> dict:
    """Stitched statistics for one (arm, cost level): the concatenated
    cross-fold daily series through the single stitching authority
    (``eval_metrics.stitch_oos_series``), metrics via ``stitched_metrics``
    and the active IR recomputed on the concatenated series (never the
    mean of per-fold IRs).  Failed folds contribute no returns and are
    recorded."""

    from .baseline_harness import oos_active_ir
    from .eval_metrics import stitch_oos_series, stitched_metrics

    level_rows = [
        row for row in arm_rows if str(row.get("cost_level")) == level
    ]
    failed = [
        {
            "fold_test_end": row.get("fold_test_end"),
            "reason": row.get("failure_reason"),
        }
        for row in level_rows
        if row.get("failed")
    ]
    trials = stitch_oos_series(level_rows)
    if not trials:
        return {
            "n_days": 0,
            "n_segments": 0,
            "failed_folds": failed,
            "note": "no succeeded folds at this level",
        }
    trial = trials[0]
    out = dict(stitched_metrics(trial))
    out["oos_active_ir"] = float(
        oos_active_ir(
            np.asarray(trial["daily_returns"], dtype=np.float64),
            np.asarray(trial["benchmark_daily_returns"], dtype=np.float64),
        )
    )
    out["n_segments"] = len(trial["segments"])
    out["failed_folds"] = failed
    return out


def summarize_arms(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-arm round-1 summary (p34 §3.4/§3.7 + rev 3): every arm, every
    fold — the frozen adjudication consumes the fold-mean statistics
    verbatim; the rev 3 additions (annual distribution, cross-fold
    median/min/max, stitched statistics, statuses, funnel totals) report
    alongside them without ever substituting for them."""

    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(str(row["candidate"]), []).append(row)
    benchmark_rows = arms.get("benchmark:equal_weight", [])
    benchmark_failed_folds = {
        str(row["fold_test_end"])
        for row in benchmark_rows
        if row.get("failed")
    }
    summaries: dict[str, dict[str, Any]] = {}
    for candidate, arm_rows in arms.items():
        by_fold: dict[str, dict[str, dict]] = {}
        for row in arm_rows:
            by_fold.setdefault(str(row["fold_test_end"]), {})[
                str(row["cost_level"])
            ] = row
        # p34 §3.5: a fold fails for an arm when ANY of its levels failed.
        failed_folds = sum(
            1
            for per_level in by_fold.values()
            if any(bool(r.get("failed")) for r in per_level.values())
        )

        def _usable(per_level: dict, level: str, key: str):
            row = per_level.get(level)
            if row is None or row.get("failed"):
                return None
            value = row.get(key)
            return None if value is None else value

        # --- rev 3: per-fold annual decomposition + statuses (§3.4). ---
        annual: dict[str, dict[str, Any]] = {}
        for fold_end, per_level in by_fold.items():
            any_failed = any(
                bool(r.get("failed")) for r in per_level.values()
            )
            levels: dict[str, dict[str, Any]] = {}
            for level, row in per_level.items():
                levels[level] = {
                    "excess_return": None
                    if row.get("failed")
                    else row.get("excess_return"),
                    "oos_active_ir": None
                    if row.get("failed")
                    else row.get("oos_active_ir"),
                    "total_return": None
                    if row.get("failed")
                    else row.get("total_return"),
                }
            if any_failed:
                status, reason = "failed", "; ".join(
                    f"{level}: {r.get('failure_reason')}"
                    for level, r in sorted(per_level.items())
                    if r.get("failed")
                )
            elif fold_end in benchmark_failed_folds:
                status, reason = "benchmark_missing", (
                    "the equal-weight benchmark reference row failed on "
                    "this fold"
                )
            else:
                status, reason = "ok", None
            annual[fold_end] = {
                "status": status,
                "failure_reason": reason,
                "levels": levels,
            }

        excess_1x = [
            value
            for per_level in by_fold.values()
            if (value := _usable(per_level, "1x", "excess_return")) is not None
        ]
        # Rev 3: the per-fold active IR comes from the row's own test-window
        # field (evaluate_signal computes it from the row's daily series —
        # one computation, consumed; the mean over folds stays the frozen
        # §3.7 statistic under its own name).
        active_irs = [
            value
            for per_level in by_fold.values()
            if (value := _usable(per_level, "1x", "oos_active_ir")) is not None
        ]
        fold_excess_values = excess_1x
        fold_ir_values = active_irs

        def _dist(values: list[float]) -> dict[str, Any] | None:
            if not values:
                return None
            arr = np.asarray(values, dtype=np.float64)
            return {
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": int(arr.size),
            }

        # --- rev 3: stitched statistics per level (concatenated series). ---
        levels_present = sorted(
            {str(row.get("cost_level")) for row in arm_rows}
        )
        stitched = {
            level: _stitched_level_metrics(arm_rows, level)
            for level in levels_present
        }
        # --- rev 3: funnel totals over the succeeded 1x folds. ---
        funnel_totals: dict[str, Any] = {}
        ok_1x_rows = [
            per_level["1x"]
            for per_level in by_fold.values()
            if per_level.get("1x") is not None
            and not per_level["1x"].get("failed")
        ]
        for key in _FUNNEL_SUM_KEYS:
            funnel_totals[key] = int(
                sum(int(r.get(key) or 0) for r in ok_1x_rows)
            )
        invested_minima = [
            float(r["funnel_invested_ratio_min"])
            for r in ok_1x_rows
            if r.get("funnel_invested_ratio_min") is not None
        ]
        funnel_totals["funnel_invested_ratio_min"] = (
            float(min(invested_minima)) if invested_minima else None
        )
        holdings_minima = [
            int(r["funnel_holdings_min"])
            for r in ok_1x_rows
            if r.get("funnel_holdings_min") is not None
        ]
        funnel_totals["funnel_holdings_min"] = (
            int(min(holdings_minima)) if holdings_minima else None
        )
        summaries[candidate] = {
            "n_folds": len(by_fold),
            "failed_folds": failed_folds,
            "unreliable": failed_folds >= 2,
            "mean_net_excess_1x": (
                float(np.mean(excess_1x)) if excess_1x else None
            ),
            "active_ir_1x": (
                float(np.mean(active_irs)) if active_irs else None
            ),
            "positive_folds_1x": int(
                sum(1 for value in excess_1x if value > 0)
            ),
            "mean_holdings_1x": (
                float(
                    np.mean(
                        [
                            value
                            for per_level in by_fold.values()
                            if (
                                value := _usable(
                                    per_level, "1x", "holdings_mean"
                                )
                            )
                            is not None
                        ]
                    )
                )
                if any(
                    _usable(per_level, "1x", "holdings_mean") is not None
                    for per_level in by_fold.values()
                )
                else None
            ),
            "invested_ratio_mean_1x": (
                float(
                    np.mean(
                        [
                            value
                            for per_level in by_fold.values()
                            if (
                                value := _usable(
                                    per_level, "1x", "invested_ratio_mean"
                                )
                            )
                            is not None
                        ]
                    )
                )
                if any(
                    _usable(per_level, "1x", "invested_ratio_mean") is not None
                    for per_level in by_fold.values()
                )
                else None
            ),
            "mean_net_excess_2x": (
                float(
                    np.mean(
                        [
                            value
                            for per_level in by_fold.values()
                            if (
                                value := _usable(
                                    per_level, "2x", "excess_return"
                                )
                            )
                            is not None
                        ]
                    )
                )
                if any(
                    _usable(per_level, "2x", "excess_return") is not None
                    for per_level in by_fold.values()
                )
                else None
            ),
            # --- rev 3 additions (reported, never substituted for the
            # frozen fold-mean statistics above). ---
            "annual": annual,
            "fold_stats_1x": {
                "excess_return": _dist(fold_excess_values),
                "oos_active_ir": _dist(fold_ir_values),
            },
            "stitched": stitched,
            "benchmark_missing_folds": sorted(benchmark_failed_folds),
            "funnel_totals_1x": funnel_totals,
        }
    return summaries


def classify_arm_diagnostics(
    arm_summaries: dict[str, dict[str, Any]],
    rules: dict[str, object] | None = None,
) -> dict[str, dict[str, Any]]:
    """The p34 §4 Rev 3 descriptive four-way diagnostic (pure function).

    Separates — per cost-bearing arm — ``measurement_incomplete`` (failed
    folds / a missing benchmark reference), ``weak_signal`` (no GROSS
    space at 0x), ``execution_blocked`` (suppression/unfill/deployment
    funnel evidence) and ``cost_swallowed`` (gross space alive, net space
    dead at 1x).  This DESCRIBES the reported evidence; it is not the
    frozen §3.7 adjudication and never decides worth_search.  The
    benchmark reference and the cash control are references/controls and
    are never classified."""

    merged = dict(R01_DIAGNOSTIC_RULES)
    merged.update(rules or {})
    out: dict[str, dict[str, Any]] = {}
    for candidate, summary in arm_summaries.items():
        if candidate in ("benchmark:equal_weight", "control:cash"):
            continue
        stitched = summary.get("stitched") or {}
        s0 = stitched.get("0x") or {}
        s1 = stitched.get("1x") or {}
        s2 = stitched.get("2x") or {}
        funnel = summary.get("funnel_totals_1x") or {}
        due = int(funnel.get("funnel_due_sessions") or 0)
        fully = int(
            funnel.get("funnel_due_fully_suppressed_sessions") or 0
        )
        planned = int(funnel.get("funnel_planned_order_count") or 0)
        filled = int(funnel.get("funnel_filled_order_count") or 0)
        invested = summary.get("invested_ratio_mean_1x")
        failed = int(summary.get("failed_folds") or 0)
        bench_missing = list(summary.get("benchmark_missing_folds") or [])
        supp_share = (fully / due) if due > 0 else 0.0
        unfill_rate = ((planned - filled) / planned) if planned > 0 else 0.0
        flags: list[str] = []
        if failed >= 2:
            flags.append("unreliable")
        elif failed >= 1:
            flags.append("fold_failure")
        if bench_missing:
            flags.append("benchmark_missing")
        measurement_incomplete = failed >= 1 or bool(bench_missing)
        gross_excess = s0.get("excess_return")
        gross_ir = s0.get("oos_active_ir")
        weak = (gross_excess is not None and gross_excess <= 0.0) or (
            gross_ir is not None and gross_ir <= 0.0
        )
        if weak:
            flags.append("weak_signal")
        blocked_reasons: list[str] = []
        if supp_share > float(merged["fully_suppressed_share_max"]):
            blocked_reasons.append("blocked_by_suppressed_trades")
        if unfill_rate > float(merged["unfilled_order_rate_max"]):
            blocked_reasons.append("blocked_by_unfilled_orders")
        if (
            invested is not None
            and invested < float(merged["min_invested_ratio"])
            and (gross_excess is not None and gross_excess > 0.0)
        ):
            blocked_reasons.append("blocked_uninvested")
        flags.extend(blocked_reasons)
        net_1x = s1.get("excess_return")
        net_2x = s2.get("excess_return")
        cost_swallowed = (
            not weak and net_1x is not None and net_1x <= 0.0
        )
        if cost_swallowed:
            flags.append("cost_swallowed_1x")
        if (
            not weak
            and net_1x is not None
            and net_1x > 0.0
            and net_2x is not None
            and net_2x <= 0.0
        ):
            flags.append("cost_fragile_2x")
        if measurement_incomplete:
            headline = "measurement_incomplete"
        elif weak:
            headline = "weak_signal"
        elif blocked_reasons:
            headline = "execution_blocked"
        elif cost_swallowed:
            headline = "cost_swallowed"
        else:
            headline = "net_space_shown"
        out[candidate] = {
            "headline": headline,
            "flags": flags,
            "evidence": {
                "gross_excess_0x": gross_excess,
                "gross_active_ir_0x": gross_ir,
                "net_excess_1x": net_1x,
                "net_excess_2x": net_2x,
                "fully_suppressed_share": round(supp_share, 6),
                "unfilled_order_rate": round(unfill_rate, 6),
                "invested_ratio_mean_1x": invested,
                "failed_folds": failed,
                "benchmark_missing_folds": bench_missing,
            },
        }
    return out


# --- frozen adjudication --------------------------------------------------------


def decide_adjudication(
    arm_summaries: dict[str, dict[str, Any]],
    rules: dict[str, float] = R01_ADJUDICATION_RULES,
) -> dict[str, Any]:
    """The p34 §3.7 frozen rules (pure function; never sees prices, only
    the summarized rows).  Cost-bearing arms only — the benchmark and
    the cash control are references/controls, not value evidence."""

    flags: dict[str, list[str]] = {}
    qualified: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    for candidate, summary in arm_summaries.items():
        if candidate in ("benchmark:equal_weight", "control:cash"):
            continue
        arm_flags: list[str] = []
        # p34 §3.7 net-space statistic (ex-holdings: the holdings floor
        # lives in the thin_book flag — the field name records that).
        net_space_ex_holdings = (
            summary["mean_net_excess_1x"] is not None
            and summary["active_ir_1x"] is not None
            and summary["mean_net_excess_1x"] > 0
            and summary["active_ir_1x"] > 0
        )
        # p34 §3.5: `unreliable` is the >= 2 failed-folds marker; a single
        # failed fold is an informational flag, not a disqualification.
        failed_folds = int(summary.get("failed_folds") or 0)
        if failed_folds >= 2:
            arm_flags.append("unreliable")
        elif failed_folds:
            arm_flags.append("fold_failure")
        invested = summary["invested_ratio_mean_1x"]
        if invested is None or invested < rules["min_invested_ratio"]:
            arm_flags.append("low_activity")
        holdings = summary["mean_holdings_1x"]
        if holdings is None or holdings < rules["min_avg_holdings"]:
            arm_flags.append("thin_book")
        if net_space_ex_holdings:
            if summary["positive_folds_1x"] < rules["min_positive_folds"]:
                arm_flags.append("single_year_dependence")
            if (summary["mean_net_excess_2x"] or 0.0) <= 0:
                arm_flags.append("cost_fragile")
            if not any(
                flag in arm_flags
                for flag in (
                    "low_activity",
                    "thin_book",
                    "unreliable",
                    "single_year_dependence",
                )
            ):
                qualified.append(candidate)
        flags[candidate] = arm_flags
        details[candidate] = {
            "net_space_ex_holdings": bool(net_space_ex_holdings),
            "flags": list(arm_flags),
        }
    worth_search = bool(qualified)
    bottleneck = (
        "qualified_arms_present"
        if worth_search
        else (
            "no_qualified_arm: no cost-bearing arm shows durable "
            "net-of-cost space under the frozen rules (low_activity / "
            "thin_book / unreliable / single_year_dependence / "
            "non-positive net excess)"
        )
    )
    return {
        "worth_search": worth_search,
        "qualified_arms": qualified,
        "flags": flags,
        "arm_details": details,
        "rules": dict(rules),
        "bottleneck": bottleneck,
    }


# --- lifecycle-driving run ------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_and_missing_sessions(
    loader, data_config, *, cutoff: str
) -> tuple[int, int, list[str]]:
    """Expected open sessions inside the configured window **up to the
    confirmed data cutoff** vs the loaded axis (the §5.1 calendar
    reconciliation, computed from the single calendar authority the
    loader already resolved).  Future pre-listed sessions after the
    cutoff are never 'missing trading days' — the dual-review repair:
    the first real run counted 20260917..20261231 as missing because the
    production calendar pre-lists the whole year."""

    start = str(data_config.start_date).replace("-", "")
    end = min(
        str(data_config.end_date).replace("-", ""), str(cutoff).replace("-", "")
    )
    sessions = [
        str(session)
        for session in loader._contract().sessions
        if start <= str(session) <= end
    ]
    loaded = {str(date).replace("-", "") for date in loader.dates}
    missing = [session for session in sessions if session not in loaded]
    return len(sessions), len(loaded), missing


def _feature_coverage(loader, feature_names: list[str]) -> dict[str, float]:
    """Per-feature finite coverage over the universe-masked cells of the
    loaded factor tensor (the §5.2 check-5 input)."""

    from .vocab import FEATURE_NAMES

    mask = loader.universe_mask
    tensor = loader.factor_tensor.numpy()
    coverage: dict[str, float] = {}
    for name in feature_names:
        idx = FEATURE_NAMES.index(name)
        # The factor tensor is [feature, stock, date]; row idx is the
        # feature's [stock, date] matrix.
        block = tensor[idx]
        finite = np.isfinite(block) & mask
        coverage[name] = (
            float(finite.sum() / mask.sum()) if mask.sum() else 0.0
        )
    return coverage


def run_r01_baseline(
    config_path: str | Path,
    contract_sha256: str,
    out_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    require_clean_tree: bool = True,
    wall_clock_budget_s: float = 6 * 3600,
    min_eligible: int | None = None,
    adjudication_rules: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The p34 round-1 run (see the module docstring).  Returns the run
    summary; a fail-closed refusal raises :class:`R01Refusal` with the
    run left at its honest terminal state.  ``min_eligible`` is the
    ProductionGateRunner deployment tuning knob (the same seam the
    formal entries expose); production callers keep the default.
    ``adjudication_rules`` defaults to the frozen :data:`R01_ADJUDICATION_RULES`
    (the test-only override seam runs the REAL adjudicator with relaxed
    canary-scale thresholds — it never touches production defaults)."""

    from .data_loader import AshareDataLoader
    from .identity import content_hash
    from .lifecycle import (
        TERMINAL_STATUSES,
        DataQualificationError,
        FactorQualificationError,
        LifecycleWriter,
        build_data_qualification_report,
        build_factor_qualification_report,
        replay_run,
    )
    from .research_domain import tier_restricted_features
    from .run_store import RunStore
    from .runspec import resolve_runtime_runspec

    started = time.monotonic()

    def _budget_check(where: str) -> None:
        elapsed = time.monotonic() - started
        if elapsed > wall_clock_budget_s:
            raise R01Refusal(
                f"R01 wall-clock budget exceeded at {where}: "
                f"{elapsed:.0f}s > {wall_clock_budget_s:.0f}s (p34 §3.6)"
            )

    root = Path(repo_root) if repo_root is not None else (
        Path(__file__).resolve().parents[1]
    )
    config_path = Path(config_path)
    out_root = Path(out_root) if out_root is not None else (
        root / "data" / "r1" / "r01_baseline"
    )

    # -- C5 hardening: the supplied contract sha256 must match the
    # committed contract file (the provenance stamp is self-verified, not
    # self-attested).  Refuses before anything expensive runs.
    contract_path = (
        root / "docs" / "p34_r01_fixed_baseline_diagnostic_draft.md"
    )
    actual_contract_sha = _sha256_file(contract_path)
    if str(contract_sha256).lower() != actual_contract_sha:
        raise R01Refusal(
            "contract sha256 mismatch: supplied "
            f"{contract_sha256!r} != committed contract file {actual_contract_sha} "
            f"({contract_path.name}); a formal run never stamps unverified "
            "provenance"
        )

    from ashare_data.config import (
        make_backtest_config,
        make_data_config,
        make_model_config,
        make_protocol_config,
        make_reward_config,
    )

    raw = load_config_file(config_path)
    data_config = make_data_config(raw, root)
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    reward_config = make_reward_config(raw)
    proto_cfg = make_protocol_config(raw)

    # -- A02/A03: identity confirm → frozen spec → open run.
    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    if not loader.dataset_id:
        raise R01Refusal(
            "the composite snapshot identity could not be confirmed; a "
            "formal R01 run refuses to proceed without a dataset identity"
        )
    spec = resolve_runtime_runspec(
        dataset_id=loader.dataset_id,
        data_cutoff=loader.frozen_data_cutoff(),
        purpose="historical_research",
        data_config=data_config,
        backtest_config=backtest_config,
        requested_budget=max(1, proto_cfg.screening.steps * proto_cfg.screening.batch_size),
        n_folds=len(proto_cfg.folds),
        research_domain=proto_cfg.domain,
        seeds=tuple(proto_cfg.seeds),
        searcher=model_config.searcher,
        max_formula_len=int(model_config.max_formula_len),
        repo_root=root,
        require_clean_tree=require_clean_tree,
    )

    # -- formal gates on the frozen snapshot's own timeline (P36): the
    # G1–G7 audit is unchanged; G8 measures the primary daily window
    # against the snapshot cutoff, so the same confirmed snapshot
    # qualifies identically on any run day.  Missing bars at the snapshot
    # window end still refuse (p34 §7 refusal semantics preserved).
    gate_result = ProductionGateRunner(
        data_config, min_eligible=min_eligible
    ).run(
        mode="formal",
        purpose=ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
        snapshot_cutoff=spec.data_cutoff,
    )
    if not gate_result.ok:
        failed = [
            f"{check.name} ({check.detail})"
            for check in gate_result.checks
            if not check.ok
        ]
        raise R01Refusal(
            "production gate check failed (p34 §7): " + "; ".join(failed)
        )
    store = RunStore(data_config.data_dir)
    handle = store.open_run(spec)
    run_id = handle.run_id
    writer = LifecycleWriter(handle)

    def _mark_failed(reason: str) -> None:
        """Honest-terminal guard: mark FAILED unless the run already sits
        in a terminal state (REJECTED/FAILED/RETIRED have no out-edges —
        a double transition is itself a machine error)."""

        if writer.current_state not in TERMINAL_STATUSES:
            writer.fail(reason=reason)

    def _latest_artifact(artifact_type: str):
        index = handle.load_index()
        for entry in reversed(index.get("artifacts", [])):
            if entry.get("artifact_type") == artifact_type:
                return entry
        return None

    try:
        idea = {
            "idea_record_schema_version": 1,
            "hypothesis": (
                "R01 round-1 (p34): fixed simple baselines on the medium "
                "Tier-A configuration reveal whether data-qualified, "
                "cost-bearing simple signals leave net-of-cost active "
                "space worth a small search round."
            ),
            "scope": (
                f"configs/{config_path.name} (sha256 "
                f"{_sha256_file(config_path).upper()}), folds "
                f"{[str(f['test_end']) for f in raw['protocol']['folds']]}, "
                "arms: equal-weight benchmark / preregistered single "
                "factors / frozen-rule rank combos / cash control at "
                "0x/1x/2x costs; preregistration contract sha256 "
                f"{contract_sha256}."
            ),
            "economic_mechanism": (
                "Cross-sectional price momentum, short-horizon risk fit "
                "and reversal captured at the domain's every_10_days / "
                "horizon-10 execution point, net of A-share commission, "
                "stamp tax, transfer fee and slippage at 1x and 2x."
            ),
            "non_goals": (
                "No search arms; no slow_ab/short_a/unified rounds; no "
                "holdout or promotion claims (2021-2025 are development "
                "windows); no production default changes."
            ),
        }
        writer.record_idea(idea)
        writer.lock_spec()

        # -- load, drift guard, typed §5.1 report.
        loader.load_data()
        if loader.dataset_id != spec.dataset_id:
            _mark_failed(
                "the database content drifted between identity freeze "
                "and load (A02 I2/p27 I6); the frozen spec no longer "
                "matches the confirmed composite snapshot"
            )
            raise R01Refusal(
                "dataset identity drifted between spec freeze and load"
            )
        _budget_check("load")
        frame_status = loader.loader_frame_status()
        expected_days, actual_days, missing_days = (
            _expected_and_missing_sessions(
                loader, data_config, cutoff=spec.data_cutoff
            )
        )
        data_report = build_data_qualification_report(
            spec,
            gate_result,
            expected_trading_days=expected_days,
            actual_trading_days=actual_days,
            missing_trading_days=missing_days,
            coverage=float(np.mean(loader.universe_mask)),
            loader_frames=frame_status,
        )
        try:
            writer.qualify_data(data_report)
        except DataQualificationError as exc:
            # The machine discriminator (p8 §4): a COMPLETED gate verdict
            # that fails validation is REJECTED (its registered report is
            # the complete evidence); an evaluation that did not complete
            # is FAILED.  The meanings are not interchangeable.
            entry = _latest_artifact("data_qualification_report")
            completed = (
                bool(
                    handle.read_artifact(entry["artifact_id"]).get("completed")
                )
                if entry is not None
                else False
            )
            if entry is not None and completed:
                writer.reject(
                    evidence={
                        "artifact_id": entry["artifact_id"],
                        "artifact_type": "data_qualification_report",
                    }
                )
            else:
                _mark_failed(
                    f"§5.1 data qualification gate did not complete: {exc}"
                )
            raise R01Refusal(f"data qualification refused: {exc}") from exc

        # -- the frozen baseline arms (p34 §3.2/§3.3).
        folds = resolve_folds_for(proto_cfg, loader)
        rows, _directions = run_baseline_arms(
            loader,
            proto_cfg,
            folds,
            backtest_config,
            model_config,
            reward_config,
            on_fold=lambda i, total, fold: _budget_check(
                f"fold {fold.test_end} ({i + 1}/{total})"
            ),
        )
        _budget_check("baseline arms")

        # -- the report artifact (strict JSON inside the run store).
        safe_rows = rows_to_jsonable(rows)
        arm_summaries = summarize_arms(rows)
        adjudication = decide_adjudication(
            arm_summaries,
            adjudication_rules or R01_ADJUDICATION_RULES,
        )
        arm_diagnostics = classify_arm_diagnostics(arm_summaries)
        benchmark_status = {
            str(row["fold_test_end"]): {
                "status": "failed" if row.get("failed") else "ok",
                "failure_reason": row.get("failure_reason"),
            }
            for row in rows
            if str(row["candidate"]) == "benchmark:equal_weight"
        }
        completeness_reasons = list(frame_status["reasons"])
        if missing_days:
            completeness_reasons.append(
                f"missing trading days: {missing_days}"
            )
        data_completeness = {
            "status": (
                "complete"
                if frame_status["status"] == "clean" and not missing_days
                else "incomplete"
            ),
            "loader_frames": frame_status,
            "expected_trading_days": int(expected_days),
            "actual_trading_days": int(actual_days),
            "missing_trading_days": [str(day) for day in missing_days],
            "reasons": completeness_reasons,
        }
        engine_version_attribution = {
            "execution_spec_version": int(EXECUTION_SPEC_VERSION),
            "portfolio_constructor_version": int(
                PORTFOLIO_CONSTRUCTOR_VERSION
            ),
            "prior_version_numbers_exist": False,
            "prior_version_note": (
                "no pre-p35 (frozen-weight accounting) baseline numbers "
                "exist: the WP0 close-out (p34 measurement log section 6.3) "
                "marked every pre-p35 engine number inadmissible for "
                "parameter selection and no R01 arm ever executed, so no "
                "old-vs-new numeric attribution is computable — recorded, "
                "not fabricated"
            ),
            "semantic_delta_reference": (
                "p35 share-level accounting (EXECUTION_SPEC_VERSION 3->4): "
                "non-due sessions drift weights by the adjusted o2o return; "
                "due sessions execute at share level on the raw entry open "
                "with lot floors and full exits"
            ),
        }
        reporting_notes = {
            "combo_rank_scale": (
                "combos average raw per-date cross-sectional ranks "
                "(reward._rankdata); members with different finite-coverage "
                "counts produce different rank scales — the frozen "
                "construction, disclosed here; percentile standardization "
                "is a new definition reserved for its own package"
            ),
            "stitched_vs_fold_mean": (
                "stitched statistics are computed on the cross-fold "
                "concatenated daily series (never the mean of per-fold "
                "IRs standing in for the stitched IR); the frozen section "
                "3.7 adjudication consumes the fold-mean statistics under "
                "their own names — both are reported, never substituted"
            ),
        }
        report = {
            "r01_report_schema_version": R01_REPORT_SCHEMA_VERSION,
            "provenance": {
                "contract_sha256": str(contract_sha256),
                "contract_path": "docs/p34_r01_fixed_baseline_diagnostic_draft.md",
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path).upper(),
                "dataset_id": spec.dataset_id,
                "data_cutoff": spec.data_cutoff,
                "git_commit": spec.git_commit,
                "spec_id": spec.spec_id,
                "run_id": run_id,
                "research_domain": spec.research_domain,
                "tier_set": proto_cfg.tier_set,
                "seeds_recorded": list(spec.seeds),
                "seeds_note": (
                    "no stochastic arm runs in round 1; seeds are recorded "
                    "for the potential round-2 search and do not affect "
                    "these deterministic baselines"
                ),
                "requested_budget": spec.requested_budget,
                "consumed_semantic_evaluations": 0,
                "cost_ladder_definition": {
                    "0x": "all cost parameters zero (gross)",
                    "1x": "config verbatim (net)",
                    "2x": "cost rates doubled, commission floor unchanged",
                },
            },
            "adjudication": adjudication,
            "data_completeness": data_completeness,
            "benchmark_status": benchmark_status,
            "arm_summaries": arm_summaries,
            "arm_diagnostics": arm_diagnostics,
            "diagnostic_rules": dict(R01_DIAGNOSTIC_RULES),
            "engine_version_attribution": engine_version_attribution,
            "reporting_notes": reporting_notes,
            "rows": safe_rows,
            "completed": True,
        }
        report_record = handle.write_artifact("r01_baseline_report", report)

        # -- §5.2 factor qualification with the real evidence blocks.
        feature_names = tier_restricted_features(
            proto_cfg.domain, proto_cfg.tier_set
        )
        coverage = _feature_coverage(loader, feature_names)
        baseline_evidence = {
            "completed": True,
            "result_hash": content_hash("r01_baseline", report),
            "report_artifact_id": report_record.artifact_id,
            "arm_count": len({r["candidate"] for r in rows}),
            "cost_levels": list(COST_LEVELS),
            "note": (
                "p34 §3.2 arms evaluated through evaluate_signal/"
                "baseline_candidates/benchmark_row (the single engine "
                "path); the frozen-rule combos and the cash control are "
                "part of this evidence"
            ),
        }
        ablation_rows = _leave_family_out_rows(
            loader, proto_cfg, folds, backtest_config, model_config,
            reward_config,
        )
        if not ablation_rows:
            # p34 §3.5 / §5.2 check 6: an ablation with no producible
            # subset (single-family factor set) cannot be completed — a
            # fail-closed refusal, never a hash over an empty list.
            _mark_failed(
                "leave-family-out ablation produced no subsets: the §5.2 "
                "check-6 evidence cannot be produced for a single-family "
                "factor set"
            )
            raise R01Refusal(
                "leave-family-out ablation produced no subsets; the §5.2 "
                "check-6 evidence cannot be produced for a single-family "
                "factor set"
            )
        ablation_evidence = {
            "completed": True,
            "result_hash": content_hash(
                "r01_ablation", rows_to_jsonable(ablation_rows)
            ),
            "leave_family_out": rows_to_jsonable(ablation_rows),
            "note": (
                "leave-one-family-out subsets of the preregistered "
                "baseline signals scored by the same CandidateScorer "
                "(baseline_candidates) on the training windows only"
            ),
        }
        factor_report = build_factor_qualification_report(
            spec,
            feature_names=feature_names,
            universe_codes=list(loader.ts_codes),
            config=data_config,
            coverage_by_feature=coverage,
            baseline_evidence=baseline_evidence,
            ablation_evidence=ablation_evidence,
        )
        try:
            writer.qualify_factor_set(factor_report)
        except FactorQualificationError as exc:
            # Machine discriminator (p8 §4): the builder emits completed
            # reports, so a refusal here is a completed verdict failing
            # validation → REJECTED with the registered evidence.
            entry = _latest_artifact("factor_qualification_report")
            completed = (
                bool(
                    handle.read_artifact(entry["artifact_id"]).get("completed")
                )
                if entry is not None
                else False
            )
            if entry is not None and completed:
                writer.reject(
                    evidence={
                        "artifact_id": entry["artifact_id"],
                        "artifact_type": "factor_qualification_report",
                    }
                )
            else:
                _mark_failed(
                    f"§5.2 factor qualification gate did not complete: {exc}"
                )
            raise R01Refusal(f"factor qualification refused: {exc}") from exc

        # -- frozen adjudication (p34 §3.7): park or retire.
        if adjudication["worth_search"]:
            final_state = writer.current_state  # parked, non-terminal
        else:
            writer.retire(
                reason=(
                    "R01 round-1 adjudication: simple baselines show no "
                    "durable net-of-cost space; report artifact "
                    f"{report_record.artifact_id}; bottleneck: "
                    f"{adjudication['bottleneck']}"
                )
            )
            final_state = writer.current_state
        summary = {
            "spec_id": spec.spec_id,
            "run_id": run_id,
            "state": replay_run(store.root, spec.spec_id, run_id) or final_state,
            "dataset_id": spec.dataset_id,
            "data_cutoff": spec.data_cutoff,
            "adjudication": adjudication,
            "report_artifact_id": report_record.artifact_id,
            "row_count": len(rows),
            "wall_seconds": round(time.monotonic() - started, 1),
        }
    except R01Refusal as exc:
        # p34 §3.6/§7: any post-open refusal (budget overrun, drift,
        # gate refusal re-raised here) leaves the run at its honest
        # terminal — never dangling non-terminal.
        _mark_failed(str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 — the honest-terminal catch-all
        _mark_failed(
            f"R01 round-1 runtime failure: {type(exc).__name__}: {exc}"
        )
        raise R01Refusal(f"runtime failure: {exc}") from exc
    finally:
        handle.close()

    # -- measurement-directory copy (data/r1/r01_baseline/).
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "r01_baseline_report.json"
    report_path.write_text(
        json.dumps(rows_to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_path = out_root / "r01_baseline_run_meta.json"
    meta_path.write_text(
        json.dumps(rows_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["report_sha256"] = _sha256_file(report_path)
    summary["report_path"] = str(report_path)
    summary["meta_path"] = str(meta_path)
    return summary


# --- small helpers kept import-light --------------------------------------------


def load_config_file(config_path: Path) -> dict:
    """YAML parse of the run configuration (the production makers do all
    validation; this helper exists so tests can monkeypatch the file
    boundary)."""

    import yaml

    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def resolve_folds_for(proto_cfg, loader) -> list:
    """Fold resolution through the single authority (eval_folds)."""

    from .eval_folds import resolve_folds

    return resolve_folds(
        proto_cfg.folds,
        loader.dates,
        frequency=proto_cfg.frequency,
        horizon=proto_cfg.horizon,
    )


def ablation_subsets(names: list[str]) -> list[tuple[str, list[str]]]:
    """(left_out_label, subset) pairs for the §5.2 check-6 ablation.

    Multi-family sets leave one FAMILY out (verbatim p34 semantics);
    a single-family set falls back to leave-one-SIGNAL-out within the
    family under the label ``family:signal`` (p49 erratum 1 - the family
    ablation is structurally impossible for a single-family set, and a
    vacuous evidence hash is never an option); a one-signal set has no
    non-empty subset under either rule and the driver keeps its refusal.
    """

    from .feature_registry import family_of

    names = list(names)
    families = sorted({family_of(name) for name in names})
    if len(families) > 1:
        return [
            (family, [n for n in names if family_of(n) != family])
            for family in families
        ]
    if len(names) <= 1:
        return []
    family = families[0]
    return [
        (f"{family}:{name}", [n for n in names if n != name])
        for name in names
    ]


def _leave_family_out_rows(
    loader, proto_cfg, folds, backtest_config, model_config, reward_config
) -> list[dict]:
    """Leave-one-family-out ablation rows (p8 §5.2 check 6): each subset
    of the preregistered baseline signals minus one family, scored by
    the same ``baseline_candidates`` path on the training windows."""

    from ashare_data.config import ProtocolConfig

    from .eval_search import baseline_candidates
    from .feature_registry import family_of

    names = list(proto_cfg.baseline_signals)
    rows: list[dict] = []
    for family, subset in ablation_subsets(names):
        subset_cfg = dataclasses.replace(
            proto_cfg, baseline_signals=list(subset)
        )
        assert isinstance(subset_cfg, ProtocolConfig)
        for fold in folds:
            for row in baseline_candidates(
                loader, subset_cfg, fold, backtest_config, model_config,
                reward_config,
            ):
                rows.append(
                    {
                        "left_out_family": family,
                        "subset": list(subset),
                        "candidate": row["candidate"],
                        "fold_test_end": row["fold_test_end"],
                        "direction": row["direction"],
                        "val_reward": row["val_reward"],
                        "val_icir": row["val_icir"],
                        "train_icir": row["train_icir"],
                    }
                )
    return rows
