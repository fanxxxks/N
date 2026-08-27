"""Factor diagnostics: coverage, rank-IC against the forward target and
cross-sectional correlations.

These are the objective, cheap checks that decide whether a factor family
earns its place in the vocabulary before any training budget is spent on
it.  The report is written as JSON and printed as tables; the family-level
numbers feed the ablation experiments (``scripts/ablate_families.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
)
from ashare_data.fundamentals import FUNDAMENTAL_PIT_NAMES
from ashare_data.capital_flow import EXTERNAL_FACTOR_NAMES
from ashare_data.gates import ProductionGateRunner
from ashare_data.processor import open_to_open_returns
from ashare_logging import export_log_txt, setup_run_logging

from .data_loader import AshareDataLoader
from .factors import FACTOR_REGISTRY, NEUTRAL_FEATURE_NAMES
from .reward import icir_from_series, rank_ic_series
from .time_contract import TrainingTimeContract
from .vocab import FEATURE_NAMES


def _validate_eligible(tensor: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Return ``eligible`` after enforcing the ``[stock, date]`` alignment."""

    eligible = np.asarray(eligible, dtype=bool)
    if eligible.shape != tensor.shape[1:]:
        raise ValueError(
            f"eligible shape {eligible.shape} does not match "
            f"tensor [stock, date] shape {tensor.shape[1:]}"
        )
    return eligible


def feature_family(name: str) -> str:
    """Family label of a vocabulary feature (registered, PIT, external)."""

    entry = FACTOR_REGISTRY.get(name)
    if entry is not None:
        return entry[0].family
    if name in FUNDAMENTAL_PIT_NAMES:
        return "fundamental"
    if name in EXTERNAL_FACTOR_NAMES:
        return "external"
    if name in NEUTRAL_FEATURE_NAMES:
        return "external"  # northbound placeholder groups with capital flow
    return "unknown"


def _names_for(tensor: np.ndarray) -> list[str]:
    """Feature names for the tensor rows (full vocab, or its leading slice)."""

    return list(FEATURE_NAMES[: tensor.shape[0]])


def factor_coverage(
    tensor: np.ndarray,
    *,
    eligible: np.ndarray,
    names: list[str] | None = None,
) -> dict[str, float]:
    """Fraction of non-neutral (non-zero) cells per feature.

    After cross-sectional standardization the neutral value is exactly 0,
    so a non-zero cell means the factor carried information that date.
    ``eligible`` is the mandatory ``[stock, date]`` bool PIT universe mask:
    the denominator counts only eligible cells (cells outside the universe
    neither count as covered nor inflate the denominator), so a future
    member can never change an eligible factor's coverage.  ``names``
    overrides the vocabulary-derived row labels (e.g. a tier subset).
    """

    arr = np.asarray(tensor)
    eligible = _validate_eligible(arr, eligible)
    nonzero = np.count_nonzero(arr[:, eligible], axis=1)
    cells = int(eligible.sum())
    return {
        name: float(nonzero[i]) / max(cells, 1)
        for i, name in enumerate(_names_for(arr) if names is None else names)
    }


def rank_ic_stats(
    tensor: np.ndarray,
    target: np.ndarray,
    dates: list[str],
    names: list[str] | None = None,
    min_stocks: int = 10,
    *,
    eligible: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-feature rank IC of the standardized factor vs the forward target.

    Returns mean IC, mean |IC| and the IC information ratio
    (mean/std over dates) per feature.  Dates without a usable target
    cross-section are skipped.  ``names`` overrides the vocabulary-derived
    row labels (e.g. for a single formula signal the evaluation protocol
    passes ``["formula"]``).  The daily ICs reuse the unified
    :func:`ashare_model.reward.rank_ic_series` implementation (one Spearman
    code path for research, scoring and the protocol); ``eligible`` is the
    mandatory ``[stock, date]`` bool PIT universe mask gating each day's
    correlation to signal-date eligible cells.
    """

    tensor = np.asarray(tensor, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    eligible = _validate_eligible(tensor, eligible)
    names = _names_for(tensor) if names is None else list(names)
    daily_ics = rank_ic_series(tensor, target, min_stocks, universe_mask=eligible)
    stats: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        ics = daily_ics[i][np.isfinite(daily_ics[i])]
        if ics.size == 0:
            stats[name] = {"ic_mean": 0.0, "ic_abs_mean": 0.0, "icir": 0.0, "n_dates": 0}
            continue
        # The unified ICIR implementation: a single IC date is an
        # under-identified ratio (0.0), never a NaN that would later
        # serialize illegally into JSON artifacts.
        icir = float(icir_from_series(ics[None])[0])
        stats[name] = {
            "ic_mean": float(np.mean(ics)),
            "ic_abs_mean": float(np.mean(np.abs(ics))),
            "icir": icir,
            "n_dates": int(len(ics)),
        }
    return stats


def correlation_summary(
    tensor: np.ndarray,
    *,
    eligible: np.ndarray,
    names: list[str] | None = None,
) -> dict[str, object]:
    """Cross-sectional correlation structure of the factor stack.

    Per date, the ``[feature x feature]`` Pearson correlation over stocks
    is computed; the averages (of the absolute values) are returned as a
    matrix, together with the mean within-family |corr| and the top
    correlated pairs.  ``eligible`` is the mandatory ``[stock, date]`` bool
    PIT universe mask: each day's correlation uses only the eligible cells
    of that date, so a future member's extreme pre-join values can never
    move an eligible factor's correlations.  ``names`` overrides the
    vocabulary-derived row labels (e.g. a tier subset).
    """

    f, s, t = tensor.shape
    eligible = _validate_eligible(tensor, eligible)
    names = _names_for(tensor) if names is None else list(names)
    mean_abs = np.zeros((f, f), dtype=np.float64)
    n_dates = 0
    for day in range(t):
        sel = eligible[:, day]
        if int(sel.sum()) < 2:
            continue
        block = tensor[:, sel, day].astype(np.float64)
        if np.count_nonzero(block) == 0:
            continue
        corr = np.corrcoef(block)
        # A constant cross-section (e.g. a neutral placeholder, or the very
        # first return) has undefined correlation: contribute 0, never NaN.
        corr = np.nan_to_num(corr, nan=0.0)
        mean_abs += np.abs(corr)
        n_dates += 1
    if n_dates == 0:
        return {"matrix": [], "within_family": {}, "top_pairs": []}
    mean_abs /= n_dates
    np.fill_diagonal(mean_abs, 0.0)

    families = [feature_family(name) for name in names]
    within: dict[str, float] = {}
    for fam in sorted(set(families)):
        idx = [i for i, f_ in enumerate(families) if f_ == fam]
        if len(idx) < 2:
            continue
        sub = mean_abs[np.ix_(idx, idx)]
        within[fam] = float(sub[np.triu_indices_from(sub, k=1)].mean())

    pairs = []
    for i in range(f):
        for j in range(i + 1, f):
            pairs.append((names[i], names[j], float(mean_abs[i, j])))
    pairs.sort(key=lambda item: item[2], reverse=True)

    return {
        "matrix": {
            names[i]: {
                names[j]: round(float(mean_abs[i, j]), 4) for j in range(f)
            }
            for i in range(f)
        },
        "within_family": within,
        "top_pairs": [
            {"a": a, "b": b, "abs_corr": round(c, 4)} for a, b, c in pairs[:15]
        ],
    }


def factor_report(
    loader: AshareDataLoader,
    train_end_date: str,
    output_path: str | Path | None = None,
    tiers: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Assemble the full diagnostics report over the training window.

    ``tiers`` restricts the report to the features of the given data
    credibility tiers (e.g. ``("A",)`` for the Tier A-only report, P2-05);
    ``None`` reports the full vocabulary.  Every reported feature records
    its ``data_tier``, and the report carries the tier version and the
    usable-time rules (P2-02).
    """

    tensor = loader.factor_tensor
    if tensor is None:
        loader.load_data()
        tensor = loader.factor_tensor
    dates = loader.dates
    contract = TrainingTimeContract.resolve(dates, train_end_date)
    signal_end = contract.train_signal_end
    price_end = contract.train_label_end
    window = tensor[:, :, :signal_end].numpy().copy()
    eligibility = loader.universe_mask[:, :signal_end]
    target = open_to_open_returns(
        loader.raw_data_cache["open"][:, :price_end].numpy()
    )[:, :signal_end]
    target = loader.mask_by_universe(target)[:, :signal_end]

    from .data_tier import (
        DATA_TIER_VERSION,
        TIER_TIME_RULES,
        DataTier,
        feature_tier,
    )

    names: list[str] = list(FEATURE_NAMES)
    if tiers is not None:
        wanted = {DataTier(t) for t in tiers}
        idx = [
            i for i, name in enumerate(FEATURE_NAMES)
            if feature_tier(name) in wanted
        ]
        window = window[idx]
        names = [FEATURE_NAMES[i] for i in idx]

    coverage = factor_coverage(window, eligible=eligibility, names=names)
    ic = rank_ic_stats(
        window, target, dates[:signal_end], names=names, eligible=eligibility
    )
    corr = correlation_summary(window, eligible=eligibility, names=names)
    tier_counts: dict[str, int] = {}
    for name in names:
        tier = feature_tier(name).value
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    report = {
        "feature_count": len(names),
        # The full-period code union, kept for provenance.  It is NOT the
        # daily selectable size: that number is mean_eligible_stocks.
        "stock_count": len(loader.ts_codes),
        "mean_eligible_stocks": float(
            round(eligibility.sum(axis=0).mean(), 2)
        ),
        "date_count": signal_end,
        "dates": [dates[0], dates[signal_end - 1]],
        "data_tier_version": DATA_TIER_VERSION,
        "tiers": sorted(tier_counts) if tiers is not None else None,
        "tier_summary": tier_counts,
        "data_tier_rules": dict(TIER_TIME_RULES),
        "per_feature": [
            {
                "name": name,
                "family": feature_family(name),
                "data_tier": feature_tier(name).value,
                "coverage": round(coverage[name], 4),
                **{k: round(v, 4) for k, v in ic[name].items()},
            }
            for name in names
        ],
        "correlations": corr,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Factor report written to {output_path}")
    return report


def _print_report(report: dict[str, object]) -> None:
    rows = [
        {
            "factor": r["name"],
            "family": r["family"],
            "coverage": r["coverage"],
            "ic_mean": r["ic_mean"],
            "|ic|": r["ic_abs_mean"],
            "icir": r["icir"],
        }
        for r in report["per_feature"]
    ]
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nWithin-family mean |corr|:")
    for fam, value in report["correlations"]["within_family"].items():
        print(f"  {fam:16s} {value:.3f}")
    print("\nTop correlated pairs:")
    for pair in report["correlations"]["top_pairs"][:10]:
        print(f"  {pair['a']:20s} x {pair['b']:20s} {pair['abs_corr']:.3f}")


def main() -> None:
    setup_run_logging(run_name="diagnostics")
    parser = argparse.ArgumentParser(description="Factor quality report")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data/factor_report.json")
    parser.add_argument(
        "--tiers",
        default=None,
        help="Comma-separated data tiers to report (A, B, C); default: all",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
    )
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(data_config, min_eligible=args.min_eligible).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        loader = AshareDataLoader(data_config, model_config)
        tiers = (
            tuple(t.strip().upper() for t in args.tiers.split(",") if t.strip())
            if args.tiers
            else None
        )
        report = factor_report(loader, backtest_config.train_end_date, root / args.output, tiers=tiers)
        _print_report(report)
    finally:
        export_log_txt(run_name="diagnostics")


if __name__ == "__main__":
    main()
