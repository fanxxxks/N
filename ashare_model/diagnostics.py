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
from ashare_logging import export_log_txt, setup_run_logging

from .data_loader import AshareDataLoader
from .factors import FACTOR_REGISTRY, NEUTRAL_FEATURE_NAMES
from .vocab import FEATURE_NAMES


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


def factor_coverage(tensor: np.ndarray) -> dict[str, float]:
    """Fraction of non-neutral (non-zero) cells per feature.

    After cross-sectional standardization the neutral value is exactly 0,
    so a non-zero cell means the factor carried information that date.
    """

    nonzero = np.count_nonzero(tensor, axis=(1, 2))
    cells = tensor.shape[1] * tensor.shape[2]
    return {
        name: float(nonzero[i]) / max(cells, 1)
        for i, name in enumerate(_names_for(tensor))
    }


def rank_ic_stats(
    tensor: np.ndarray,
    target: np.ndarray,
    dates: list[str],
    names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-feature rank IC of the standardized factor vs the forward target.

    Returns mean IC, mean |IC| and the IC information ratio
    (mean/std over dates) per feature.  Dates without a usable target
    cross-section are skipped.  ``names`` overrides the vocabulary-derived
    row labels (e.g. for a single formula signal the evaluation protocol
    passes ``["formula"]``).
    """

    target = np.asarray(target, dtype=np.float64)
    stats: dict[str, dict[str, float]] = {}
    names = _names_for(tensor) if names is None else list(names)
    for i, name in enumerate(names):
        ics = []
        for t in range(tensor.shape[2]):
            signal = tensor[i, :, t]
            label = target[:, t]
            valid = np.isfinite(label)
            if valid.sum() < 10:
                continue
            s = pd.Series(signal[valid])
            l = pd.Series(label[valid])
            ic = float(s.corr(l, method="spearman"))
            if np.isfinite(ic):
                ics.append(ic)
        if not ics:
            stats[name] = {"ic_mean": 0.0, "ic_abs_mean": 0.0, "icir": 0.0, "n_dates": 0}
            continue
        arr = np.asarray(ics)
        stats[name] = {
            "ic_mean": float(np.mean(arr)),
            "ic_abs_mean": float(np.mean(np.abs(arr))),
            "icir": float(np.mean(arr) / (arr.std(ddof=1) + 1e-9)),
            "n_dates": int(len(arr)),
        }
    return stats


def correlation_summary(
    tensor: np.ndarray,
) -> dict[str, object]:
    """Cross-sectional correlation structure of the factor stack.

    Per date, the ``[feature x feature]`` Pearson correlation over stocks
    is computed; the averages (of the absolute values) are returned as a
    matrix, together with the mean within-family |corr| and the top
    correlated pairs.
    """

    f, s, t = tensor.shape
    names = _names_for(tensor)
    mean_abs = np.zeros((f, f), dtype=np.float64)
    n_dates = 0
    for day in range(t):
        block = tensor[:, :, day].astype(np.float64)
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
) -> dict[str, object]:
    """Assemble the full diagnostics report over the training window."""

    tensor = loader.factor_tensor
    if tensor is None:
        loader.load_data()
        tensor = loader.factor_tensor
    dates = loader.dates
    train_end = train_end_date.replace("-", "")
    train_end_idx = len(dates)
    for idx, date in enumerate(dates):
        if date >= train_end:
            train_end_idx = max(idx, 1)
            break
    window = tensor[:, :, :train_end_idx].numpy()
    target = loader.target_ret[:, :train_end_idx].numpy()

    coverage = factor_coverage(window)
    ic = rank_ic_stats(window, target, dates[:train_end_idx])
    corr = correlation_summary(window)
    report = {
        "feature_count": len(FEATURE_NAMES),
        "stock_count": len(loader.ts_codes),
        "date_count": train_end_idx,
        "dates": [dates[0], dates[train_end_idx - 1]],
        "per_feature": [
            {
                "name": name,
                "family": feature_family(name),
                "coverage": round(coverage[name], 4),
                **{k: round(v, 4) for k, v in ic[name].items()},
            }
            for name in FEATURE_NAMES
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
    args = parser.parse_args()
    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        loader = AshareDataLoader(data_config, model_config)
        report = factor_report(loader, backtest_config.train_end_date, root / args.output)
        _print_report(report)
    finally:
        export_log_txt(run_name="diagnostics")


if __name__ == "__main__":
    main()
