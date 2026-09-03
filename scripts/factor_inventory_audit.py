"""Factor inventory audit measurement tool — single parameterized harness.

Consolidates the two formerly duplicated audit scripts (IP-14 concept
convergence; they were mutual ~600-line copies differing only in the
generation profile):

* ``docs/factor_inventory_audit_20260831/audit_run.py`` (t2, 2026-08-31,
  62-feature baseline; frozen in git history at commit 23c82f4);
* ``docs/factor_inventory_audit_v4_20260901/audit_run_v4.py`` (t5,
  2026-09-01, 73-feature post-P9 v4 audit; frozen in git history at
  commit d6a034d).

The per-directory ``metrics.json`` / ``audit_report*.md`` measurement
evidence stays in place; ``COMPATIBILITY.md`` in each directory points
here.  Per the p13 contract freeze
(``docs/p13_fundamental_fields_contract.md`` §3.3) the historical scripts
are never edited — adjudication re-tests run THIS script (a new script)
against a profile, never the frozen evidence.

RESEARCH DIAGNOSTIC -- not promotion evidence.  Measures the factor
vocabulary on the real local DuckDB history through the project's single
semantic paths only:

* data + universe mask : ashare_model.data_loader.AshareDataLoader (strict PIT
  contract, no dev fallback);
* factor values        : the loader's official compute_factor_tensor output
  (winsorized per-date z-scores, 0 = neutral) -- the exact representation the
  search/training stack consumes;
* raw missingness      : the same FACTOR_REGISTRY functions evaluated on the
  same FactorContext (isfinite before the neutral 0-fill);
* PIT frames           : ashare_data.fundamentals.build_pit_frames /
  ashare_data.capital_flow.build_capital_frames (the loader's own builders);
* rank IC / ICIR       : ashare_model.reward.rank_ic_series / icir_from_series
  (the single rank-correlation semantic path; IP-14 also routes the
  factor-vs-size exposure Spearman through it);
* cost estimate        : ashare_model.cost_matrix.round_trip_cost under the
  shared BacktestConfig fee schedule.

Targets are h-day forward open-to-open returns (all sessions in (t, t+h]
must have finite opens; suspension gaps yield NaN).  Horizons:
1, 2, 3, 5, 10, 15, 20 sessions.  Incremental-OOS split: IS = dates < 20220101,
OOS = dates >= 20220101 (both inside the dev/validation regime; the locked
holdout is NOT touched).  No randomness; the run is deterministic.

Correlation clusters reuse the registry's measure (per-date z-scored Pearson
over eligible cells, union-find at |rho| >= 0.9); the cluster union-find is
evaluated on the pooled correlation matrix computed here for memory reasons
and cross-checked against feature_registry._correlation_clusters on a
date-subsample.

Writes one JSON artifact (metrics + provenance) into the profile's evidence
directory (override with ``--out-dir``).  The narrative report is authored
separately from this JSON.  Read-only with respect to the database.

Usage:
    python scripts/factor_inventory_audit.py --generation v1
    python scripts/factor_inventory_audit.py --generation v4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ashare_data.capital_flow import (  # noqa: E402
    build_capital_frames,
    build_industry_member_frame,
)
from ashare_data.config import (  # noqa: E402
    load_config,
    make_backtest_config,
    make_data_config,
)
from ashare_data.db import AshareDB  # noqa: E402
from ashare_data.fundamentals import build_pit_frames  # noqa: E402
from ashare_data.manifest import build_dataset_manifest  # noqa: E402
from ashare_data.processor import pivot_wide  # noqa: E402
from ashare_model.cost_matrix import round_trip_cost  # noqa: E402
from ashare_model.data_loader import AshareDataLoader  # noqa: E402
from ashare_model.feature_registry import (  # noqa: E402
    CORR_THRESHOLD,
    _correlation_clusters,
)
from ashare_model.factors import (  # noqa: E402
    AshareFactorEngine,
    FACTOR_REGISTRY,
    NEUTRAL_FEATURE_NAMES,
)
from ashare_model.reward import (  # noqa: E402
    icir_from_series,
    rank_ic_series,
)
from ashare_model.vocab import FEATURE_NAMES  # noqa: E402

HORIZONS = (1, 2, 3, 5, 10, 15, 20)
MIN_STOCKS = 10
IS_CUTOFF = "20220101"
BASELINE_SIGNALS = ("REVERSAL_5", "RSQ_60", "ILLIQ_20", "OVERNIGHT_RET",
                    "MOMENTUM_20", "ROE", "TURNOVER")
CORR_REPORT_THRESHOLD = 0.7

# Generation profiles: the complete v1 -> v4 delta of the retired scripts
# (feature-count expectation + provenance strings + v4-only vocabulary
# assertions and the factor-compute version entry).  Everything else is
# shared code, so each profile reproduces its original JSON schema exactly.
_PROFILES: dict[str, dict] = {
    "v1": {
        "out_dirname": "factor_inventory_audit_20260831",
        "expected_features": 62,
        "audit_generation": None,
        "baseline_predecessor": None,
        "manifest_note": (
            "manifest rebuild pending (t1 blocker); id computed read-only"
        ),
        "check_v4_tail": False,
        "include_factor_compute": False,
    },
    "v4": {
        "out_dirname": "factor_inventory_audit_v4_20260901",
        "expected_features": 73,
        "audit_generation": "v4 (P9 post-implementation final audit, task t5)",
        "baseline_predecessor": (
            "docs/factor_inventory_audit_20260831/metrics.json "
            "(t2, 62-feature baseline)"
        ),
        "manifest_note": (
            "manifest persisted 2026-08-31T15:23:51Z (t10); "
            "id computed read-only matches"
        ),
        "check_v4_tail": True,
        "include_factor_compute": True,
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Factor inventory audit measurement harness (research "
            "diagnostic; --generation selects the historical profile)."
        )
    )
    parser.add_argument(
        "--generation",
        choices=sorted(_PROFILES),
        required=True,
        help=(
            "audit generation profile: v1 = t2 62-feature baseline "
            "(2026-08-31), v4 = t5 73-feature post-P9 audit (2026-09-01)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "output directory for metrics.json (default: the profile's "
            "evidence directory under docs/)"
        ),
    )
    return parser.parse_args(argv)


def jdefault(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"unserializable {type(obj)}")


def fmean(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def series_stats(ic: np.ndarray) -> dict:
    ic = np.asarray(ic, dtype=float)
    ic = ic[np.isfinite(ic)]
    if ic.size == 0:
        return {"n": 0, "ic_mean": None, "icir": None, "hit": None}
    icir = float(icir_from_series(ic)[0]) if ic.size >= 2 else None
    return {
        "n": int(ic.size),
        "ic_mean": float(ic.mean()),
        "icir": icir,
        "hit": float((ic > 0).mean()),
    }


def year_slice_stats(ic: np.ndarray, years: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for year in sorted(set(years)):
        idx = np.array([i for i, d in enumerate(years) if d == year])
        st = series_stats(ic[idx])
        if st["n"]:
            out[year] = st
    return out


def forward_open_returns(open_np: np.ndarray, h: int) -> np.ndarray:
    """target[s,t] = open[t+1+h]/open[t+1] - 1 (h-session holding entered at the
    next open), NaN unless open[t+1..t+1+h] are all finite (no suspension gap)."""
    s_n, t_n = open_np.shape
    out = np.full((s_n, t_n), np.nan)
    max_t = t_n - h - 1  # last signal column with t+h+1 <= t_n-1
    if max_t < 1:
        return out
    valid = np.ones((s_n, max_t), dtype=bool)
    for j in range(1, h + 2):  # opens at t+1 .. t+1+h must exist
        valid &= np.isfinite(open_np[:, j: max_t + j])
    with np.errstate(all="ignore"):
        rets = open_np[:, 1 + h: 1 + h + max_t] / open_np[:, 1: 1 + max_t] - 1.0
    rets = np.where(np.isfinite(rets), rets, np.nan)
    out[:, :max_t] = np.where(valid, rets, np.nan)
    return out


def chunked_ic(tensor: np.ndarray, target: np.ndarray, mask: np.ndarray,
               chunk: int = 8) -> np.ndarray:
    n_f, _, t_n = tensor.shape
    ics = np.full((n_f, t_n), np.nan)
    for start in range(0, n_f, chunk):
        block = np.asarray(tensor[start:start + chunk], dtype=np.float64)
        ics[start:start + chunk] = rank_ic_series(
            block, target, min_stocks=MIN_STOCKS, universe_mask=mask
        )
    return ics


def quintile_turnover(z_row: np.ndarray, mask: np.ndarray, h: int,
                      top: bool) -> dict | None:
    """Mean one-way turnover of the direction-implied quintile basket,
    rebalanced every h sessions.  Baskets covering >50% of the eligible
    cross-section (tie-degenerate quintiles, e.g. sparse event features)
    are counted into ``degenerate_share``."""
    s_n, t_n = z_row.shape
    taus: list[float] = []
    degenerate = 0
    total = 0
    prev: set[int] | None = None
    for t in range(0, t_n - h, h):
        col = z_row[:, t]
        elig = mask[:, t] & np.isfinite(col)
        n_elig = int(elig.sum())
        if n_elig < 50:
            prev = None
            continue
        q = np.quantile(col[elig], 0.8 if top else 0.2)
        basket = set(np.where(elig & ((col >= q) if top else (col <= q)))[0].tolist())
        total += 1
        if len(basket) > 0.5 * n_elig:
            degenerate += 1
        if prev:
            taus.append(1.0 - len(basket & prev) / len(prev))
        prev = basket
    if not taus:
        return None
    return {
        "tau": float(np.mean(taus)),
        "degenerate_share": float(degenerate) / float(max(total, 1)),
    }


def union_find_clusters(corr: np.ndarray, names: list[str],
                        threshold: float) -> dict[str, str]:
    """Connected components at |corr| >= threshold (the registry's semantics)."""
    n = corr.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    members: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        members.setdefault(find(i), []).append(name)
    groups = sorted((sorted(v) for v in members.values()), key=lambda v: v[0])
    return {name: f"c{k}" for k, g in enumerate(groups) for name in g}


def main(argv: list[str] | None = None) -> int:
    t_start = time.time()
    provenance: dict = {}

    args = _parse_args(argv)
    profile = _PROFILES[args.generation]

    raw = load_config(None, project_root=ROOT)
    config = make_data_config(raw, ROOT)
    bt_cfg = make_backtest_config(raw)
    loader = AshareDataLoader(config)
    loader.load_data()
    ts_codes = loader.ts_codes
    dates = list(loader.dates)
    years = [d[:4] for d in dates]
    mask = np.asarray(loader.universe_mask, dtype=bool)
    tensor = loader.factor_tensor.numpy().astype(np.float32)  # [n_f, S, T]
    open_np = loader.raw_data_cache["open"].numpy().astype(np.float64)

    n_f, s_n, t_n = tensor.shape
    assert n_f == len(FEATURE_NAMES) == profile["expected_features"], (
        f"expected {profile['expected_features']} features, got {n_f}"
    )
    if profile["check_v4_tail"]:
        # P9 v4 vocabulary: 73 features (62 pre-v4 + 11 P9 additions, tail),
        # deprecated members still computed and therefore still measurable.
        from ashare_model.vocab import DEPRECATED_FEATURE_NAMES, _FEATURE_NAMES_V4

        assert list(FEATURE_NAMES[-len(_FEATURE_NAMES_V4):]) == list(
            _FEATURE_NAMES_V4
        )
        assert set(DEPRECATED_FEATURE_NAMES) <= set(FEATURE_NAMES)
    assert mask.shape == (s_n, t_n)
    is_oos = np.array([d >= IS_CUTOFF for d in dates])
    is_mask_flat = np.broadcast_to(is_oos, (s_n, t_n)).reshape(-1)

    with AshareDB(config.duckdb_path, read_only=True) as db:
        dataset_id_current = build_dataset_manifest(
            db, config, use_cache=True
        ).dataset_id

    from ashare_model.alphagpt import MODEL_VERSION
    from ashare_model.data_tier import DATA_TIER_VERSION
    from ashare_model.feature_registry import FEATURE_REGISTRY_VERSION
    if profile["include_factor_compute"]:
        from ashare_model.factors import FACTOR_COMPUTE_VERSION
    from ashare_model.research_domain import RESEARCH_DOMAIN_VERSION
    from ashare_model.reward import REWARD_VERSION
    from ashare_model.search_contract import SEARCH_CONTRACT_VERSION
    from ashare_model.versions import PROTOCOL_VERSION
    from ashare_model.vocab import GRAMMAR_VERSION
    from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
    from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION
    from ashare_portfolio.rebalance import REBALANCE_POLICY_VERSION
    provenance["versions"] = {
        "protocol": PROTOCOL_VERSION, "reward": REWARD_VERSION,
        "model": MODEL_VERSION, "grammar": GRAMMAR_VERSION,
        **(
            {"factor_compute": FACTOR_COMPUTE_VERSION}
            if profile["include_factor_compute"] else {}
        ),
        "feature_registry": FEATURE_REGISTRY_VERSION,
        "data_tier": DATA_TIER_VERSION,
        "research_domain": RESEARCH_DOMAIN_VERSION,
        "search_contract": SEARCH_CONTRACT_VERSION,
        "execution_spec": EXECUTION_SPEC_VERSION,
        "portfolio_constructor": PORTFOLIO_CONSTRUCTOR_VERSION,
        "rebalance_policy": REBALANCE_POLICY_VERSION,
    }
    generation_fields: dict = {}
    if profile["audit_generation"] is not None:
        generation_fields["audit_generation"] = profile["audit_generation"]
    if profile["baseline_predecessor"] is not None:
        generation_fields["baseline_predecessor"] = profile[
            "baseline_predecessor"
        ]
    provenance.update({
        "run_type": "research_diagnostic (not promotion evidence)",
        **generation_fields,
        "dataset_window": [dates[0], dates[-1]],
        "n_stocks": int(s_n), "n_dates": int(t_n),
        "dataset_id_computed_unpersisted": dataset_id_current,
        "manifest_note": profile["manifest_note"],
        "is_oos_split": {"is": f"< {IS_CUTOFF}", "oos": f">= {IS_CUTOFF}"},
        "min_stocks": MIN_STOCKS,
        "baseline_signals": list(BASELINE_SIGNALS),
        "horizons": list(HORIZONS),
        "note_industry_snapshot": "industry membership is a current snapshot (non-PIT)",
    })

    # ---- targets ----------------------------------------------------------
    targets = {h: forward_open_returns(open_np, h) for h in HORIZONS}
    # Sanity: h=1 must be the adjacent open-to-open return, not a degenerate
    # same-column ratio (off-by-one guard).
    _t1 = targets[1]
    _m1 = np.isfinite(_t1) & mask
    assert _m1.any(), "h=1 target has no valid cells"
    assert float(np.abs(_t1[_m1]).mean()) > 1e-6, "h=1 target is degenerate"
    target_finite_fraction = {
        str(h): float((np.isfinite(t) & mask).sum()) / float(mask.sum())
        for h, t in targets.items()
    }

    # ---- IC / decay / annual stability ------------------------------------
    ic_by_h: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        ic_by_h[h] = chunked_ic(tensor, targets[h], mask)
        pooled = np.concatenate([ic_by_h[h][i] for i in range(n_f)])
        print(f"ic h={h}: pooled_mean_ic={np.nanmean(pooled):.5f}", flush=True)

    ic_noover_stats: dict[int, dict] = {}
    for h in HORIZONS:
        idx = list(range(0, max(1, t_n - h - 1), h)) if h > 1 else list(range(t_n))
        ic_noover_stats[h] = idx

    annual = {
        h: {name: year_slice_stats(ic_by_h[h][i], years)
            for i, name in enumerate(FEATURE_NAMES)}
        for h in (1, 10, 20)
    }

    # ---- raw coverage + exposure ------------------------------------------
    engine = AshareFactorEngine()
    close_df = pivot_wide(loader.bars, ts_codes, dates, "close")
    pit_frames = build_pit_frames(config, ts_codes, dates, close_df)
    capital_frames = build_capital_frames(config, ts_codes, dates)
    industry_frame = build_industry_member_frame(config, ts_codes, dates)
    elig_frame = pd.DataFrame(mask, index=ts_codes, columns=dates)
    ctx = engine._build_context(loader.bars, ts_codes, dates, close_df,
                                elig_frame, industry_frame)

    with np.errstate(all="ignore"):
        cap_raw = (ctx.amount / ctx.turnover).to_numpy(dtype=float)
        log_cap = np.where((cap_raw > 0) & np.isfinite(cap_raw),
                           np.log(cap_raw), np.nan)
    if ctx.industry is not None and not ctx.industry.empty:
        ind_codes = ctx.industry.iloc[:, 0].to_numpy(dtype=object)
    else:
        ind_codes = np.full(s_n, None, dtype=object)
    ind_arr = np.array([str(c) if isinstance(c, str) and c else ""
                        for c in ind_codes], dtype=object)
    ind_groups = {g: i for i, g in enumerate(sorted({c for c in ind_arr if c}))}
    ind_idx = np.array([ind_groups.get(c, -1) for c in ind_arr], dtype=int)
    n_groups = max(len(ind_groups), 1)

    def exposure_of(vals: np.ndarray, fin: np.ndarray) -> dict:
        # Size exposure rides the single rank-correlation semantic path
        # (reward.rank_ic_series; IP-14): its per-day eligible set --
        # universe_mask & isfinite(target) & isfinite(signal), min_stocks
        # on the intersection -- is exactly the set the retired local
        # cross_sectional_spearman wrapper evaluated, so the rho values are
        # bit-identical (parity property test: tests/test_reward.py).
        size_rhos = rank_ic_series(
            vals[None], log_cap, min_stocks=MIN_STOCKS, universe_mask=mask
        )[0]
        sr = size_rhos[np.isfinite(size_rhos)]
        ind_r2: list[float] = []
        for t in range(t_n):
            col_ok = fin[:, t]
            if int(col_ok.sum()) < MIN_STOCKS:
                continue
            ii = ind_idx[col_ok]
            good = ii >= 0
            if int(good.sum()) < MIN_STOCKS:
                continue
            gid = ii[good]
            if np.unique(gid).size < 2:
                continue
            y = vals[col_ok, t][good]
            counts = np.bincount(gid, minlength=n_groups)
            sums = np.bincount(gid, weights=y, minlength=n_groups)
            nz = counts > 0
            means = sums[nz] / counts[nz]
            grand = float(y.mean())
            n_g = counts[nz].astype(float)
            between = float(np.sum(n_g * (means - grand) ** 2)) / float(y.size)
            total = float(np.sum((y - grand) ** 2)) / float(y.size)
            if total > 1e-12:
                ind_r2.append(between / total)
        r2 = np.asarray(ind_r2, dtype=float)
        return {
            "size_spearman_mean": float(sr.mean()) if sr.size else None,
            "size_spearman_abs_mean": float(np.abs(sr).mean()) if sr.size else None,
            "industry_r2_mean": float(r2.mean()) if r2.size else None,
            "n_dates_size": int(sr.size),
        }

    injected_raw: dict[str, pd.DataFrame] = dict(pit_frames)
    injected_raw.update(capital_frames)
    coverage: dict[str, dict] = {}
    exposure: dict[str, dict] = {}
    year_cols = {y: np.array([i for i, d in enumerate(years) if d == y])
                 for y in sorted(set(years))}
    mask_total = int(mask.sum())
    for name in FEATURE_NAMES:
        entry = FACTOR_REGISTRY.get(name)
        if entry is not None:
            frame = entry[1](ctx)
        elif name in injected_raw:
            frame = injected_raw[name]
        else:
            frame = pd.DataFrame(np.nan, index=ts_codes, columns=dates)
        frame = frame.reindex(index=ts_codes, columns=dates)
        vals = frame.to_numpy(dtype=float)
        fin = np.isfinite(vals) & mask
        cov_year = {}
        for year, cols in year_cols.items():
            e = int(mask[:, cols].sum())
            cov_year[year] = float(fin[:, cols].sum()) / max(e, 1)
        coverage[name] = {"overall": float(fin.sum()) / max(mask_total, 1),
                          "by_year": cov_year}
        if name in NEUTRAL_FEATURE_NAMES:
            exposure[name] = {"size_spearman_mean": None,
                              "size_spearman_abs_mean": None,
                              "industry_r2_mean": None, "n_dates_size": 0}
        else:
            exposure[name] = exposure_of(vals, fin)
        print(f"raw pass done: {name}", flush=True)
        del frame, vals, fin

    del ctx
    # ---- correlation clusters ---------------------------------------------
    z = tensor.reshape(n_f, -1).astype(np.float64)
    cell_ok = mask.reshape(-1)
    corr = np.full((n_f, n_f), np.nan)
    for i in range(n_f):
        zi = np.where(cell_ok, z[i], np.nan)
        zi0 = zi - np.nanmean(zi)
        for j in range(i, n_f):
            zj = np.where(cell_ok, z[j], np.nan)
            zj0 = zj - np.nanmean(zj)
            ok = np.isfinite(zi0) & np.isfinite(zj0)
            if int(ok.sum()) < 1000:
                continue
            a, b = zi0[ok], zj0[ok]
            den = math.sqrt(float(a @ a) * float(b @ b))
            corr[i, j] = corr[j, i] = float(a @ b) / den if den > 1e-12 else np.nan
    clusters = union_find_clusters(corr, list(FEATURE_NAMES), CORR_THRESHOLD)
    # Cross-check the union-find semantics against the registry implementation
    # on a date-subsample (memory-bounded; same measure, independent call).
    step = max(1, t_n // 400)
    registry_clusters = _correlation_clusters(
        tensor[:, :, ::step], mask[:, ::step], CORR_THRESHOLD
    )
    mine_groups: dict[str, set] = {}
    for name, cid in clusters.items():
        mine_groups.setdefault(cid, set()).add(name)
    reg_groups: dict[str, set] = {}
    for name, cid in registry_clusters.items():
        reg_groups.setdefault(cid, set()).add(name)
    clusters_check = {
        "note": "registry impl on date-subsample vs full-window union-find",
        "n_clusters_full": len(mine_groups),
        "n_clusters_registry_subsample": len(reg_groups),
    }
    high_pairs = [
        {"a": FEATURE_NAMES[i], "b": FEATURE_NAMES[j], "rho": float(corr[i, j])}
        for i in range(n_f) for j in range(i + 1, n_f)
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= CORR_REPORT_THRESHOLD
    ]
    high_pairs.sort(key=lambda p: -abs(p["rho"]))
    del z

    # ---- turnover / cost pressure -----------------------------------------
    is_idx = np.array([k for k, d in enumerate(dates) if d < IS_CUTOFF])
    rt_cost = round_trip_cost(100000.0, 20, bt_cfg)
    per_yuan_rt = rt_cost / (100000.0 / 20.0)
    turnover: dict[str, dict] = {}
    for i, name in enumerate(FEATURE_NAMES):
        if name in NEUTRAL_FEATURE_NAMES:
            turnover[name] = {
                "basket_side": None,
                "note": "neutral feature (no data source); turnover not defined",
            }
            continue
        # Direction from the IS-window h=10 IC; fall back to the full-window
        # IC when the IS slice carries no finite IC for this feature.
        v = fmean(ic_by_h[10][i][is_idx])
        if v is None:
            v = fmean(ic_by_h[10][i])
        if v is None:
            turnover[name] = {
                "basket_side": None,
                "note": "no finite h=10 IC anywhere; turnover not defined",
            }
            continue
        top = v >= 0
        entry: dict = {"basket_side": "top" if top else "bottom"}
        for h in (1, 5, 10):
            res = quintile_turnover(tensor[i].astype(np.float64), mask, h, top=top)
            entry[f"tau_h{h}"] = None if res is None else res["tau"]
            entry[f"quintile_degenerate_share_h{h}"] = (
                None if res is None else res["degenerate_share"]
            )
            if res is not None:
                entry[f"cost_drag_pct_h{h}"] = (
                    2.0 * res["tau"] * (252.0 / h) * per_yuan_rt * 100.0
                )
        turnover[name] = entry
        print(f"turnover done: {name}", flush=True)

    # ---- incremental OOS vs linear baseline -------------------------------
    h = 10
    tgt = targets[h]
    tgt_flat = tgt.reshape(-1)
    ok_flat = mask.reshape(-1) & np.isfinite(tgt_flat)
    B = tensor[[FEATURE_NAMES.index(s) for s in BASELINE_SIGNALS]].astype(
        np.float64
    ).mean(axis=0)  # [S, T] as-consumed mean of z-scored baselines
    ic_B = rank_ic_series(B[None], tgt, min_stocks=MIN_STOCKS,
                          universe_mask=mask)[0]
    oos_ic_B = series_stats(ic_B[is_oos])
    is_ic_B = series_stats(ic_B[~is_oos])
    B_flat = B.reshape(-1)
    results_incr: dict[str, dict] = {}
    for i, name in enumerate(FEATURE_NAMES):
        z_flat = tensor[i].reshape(-1).astype(np.float64)
        rows = ok_flat
        X = np.column_stack([np.ones(int(rows.sum())), B_flat[rows], z_flat[rows]])
        yv = tgt_flat[rows]
        is_sel = is_mask_flat[rows]
        coef, *_ = np.linalg.lstsq(X[is_sel], yv[is_sel], rcond=None)
        w_b, w_f = float(coef[1]), float(coef[2])
        comp = w_b * B + w_f * tensor[i].astype(np.float64)
        ic_C = rank_ic_series(comp[None], tgt, min_stocks=MIN_STOCKS,
                              universe_mask=mask)[0]
        resid = np.full((s_n, t_n), np.nan)
        zr = tensor[i].astype(np.float64)
        for t in range(t_n):
            ok = mask[:, t] & np.isfinite(zr[:, t]) & np.isfinite(B[:, t])
            n_ok = int(ok.sum())
            if n_ok < MIN_STOCKS:
                continue
            A = np.column_stack([np.ones(n_ok), B[ok, t]])
            sol, *_ = np.linalg.lstsq(A, zr[ok, t], rcond=None)
            resid[ok, t] = zr[ok, t] - A @ sol
        ic_R = rank_ic_series(resid[None], tgt, min_stocks=MIN_STOCKS,
                              universe_mask=mask)[0]
        comp_oos = series_stats(ic_C[is_oos])
        delta = None
        if comp_oos["ic_mean"] is not None and oos_ic_B["ic_mean"] is not None:
            delta = comp_oos["ic_mean"] - oos_ic_B["ic_mean"]
        results_incr[name] = {
            "is_coef_baseline": w_b,
            "is_coef_factor": w_f,
            "baseline_is": is_ic_B,
            "baseline_oos": oos_ic_B,
            "composite_oos": comp_oos,
            "delta_ic_oos": delta,
            "residual_is": series_stats(ic_R[~is_oos]),
            "residual_oos": series_stats(ic_R[is_oos]),
        }
        print(f"incremental done: {name}", flush=True)

    # ---- assemble per-factor block ----------------------------------------
    per_factor: dict[str, dict] = {}
    for i, name in enumerate(FEATURE_NAMES):
        ic_entry = {}
        for h in HORIZONS:
            idx = ic_noover_stats[h]
            ic_entry[str(h)] = {
                "full": series_stats(ic_by_h[h][i]),
                "nonoverlap": series_stats(ic_by_h[h][i][idx]),
            }
        per_factor[name] = {
            "ic_by_horizon": ic_entry,
            "annual_ic_h1": annual[1][name],
            "annual_ic_h10": annual[10][name],
            "annual_ic_h20": annual[20][name],
            "coverage": coverage[name],
            "exposure": exposure[name],
            "cluster": clusters[name],
            "turnover": turnover[name],
            "incremental_oos_h10": results_incr[name],
        }

    metrics = {
        "provenance": provenance,
        "shapes": {"n_features": n_f, "n_stocks": int(s_n), "n_dates": int(t_n)},
        "target_finite_fraction": target_finite_fraction,
        "ic_full_window_pooled": {
            str(h): series_stats(
                np.concatenate([ic_by_h[h][i] for i in range(n_f)])
            )
            for h in HORIZONS
        },
        "per_factor": per_factor,
        "correlation_matrix": corr.tolist(),
        "feature_order": list(FEATURE_NAMES),
        "clusters_0p9": clusters,
        "clusters_check": clusters_check,
        "high_corr_pairs_0p7": high_pairs,
        "cost_model": {
            "capital": 100000.0, "positions": 20,
            "round_trip_cost_yuan": rt_cost,
            "per_yuan_round_trip": per_yuan_rt,
            "source": "cost_matrix.round_trip_cost under BacktestConfig",
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    out_dir = (
        Path(args.out_dir) if args.out_dir
        else ROOT / "docs" / profile["out_dirname"]
    )
    out = out_dir / "metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=jdefault),
                   encoding="utf-8")
    print(f"WROTE {out} in {metrics['runtime_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
