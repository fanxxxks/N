"""WP18 (p56) ANALYSIS-side replay: Q4 cash decomposition + Q5 fixed-path
fee repricing, computed from the ACTUAL post-trade holdings.

Analysis-only (2026-09-27, report writer): this is NOT a stage driver and
adds NO formal evaluation rows.  It deterministically replays the SAME 1x
account rows (same frozen directions, same reference config, same folds,
same engine call as ``scripts/wp18_m3_account.py`` / ``wp18_s3_account.py``)
solely to observe engine internals the account rows do not carry:

* ``positions[].shares`` + entry opens -> exact achieved book weights
  (the ACTUAL post-trade holdings; ``invested_ratio_mean`` in the rows is
  TARGET-weight based and is explicitly NOT used, FROZEN_18_PLAN §7);
* per-session buy/sell notionals -> ``reprice_fixed_path_fees`` (Q5 direct
  fee reading on the FIXED 1x path, §7.5);
* the canonical decomposition helpers of ``ashare_model/wp18_five_questions``
  (``cash_exposure_decomposition``, §7.4).

Fail-closed discipline inherited from the stage drivers: factor cache must
HIT (a cold rebuild is refused), no ledger writes, no data/r1 writes; the
only output is ``reports/wp18/_analysis/replay_<domain>.json``.

Every replayed row is verified bit-for-bit against the stored account row's
``daily_returns`` before any decomposition is accepted.

Usage (cwd = AlphaGPT): python scripts/wp18_analysis_replay.py --domain medium|slow
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_data.config import (  # noqa: E402
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
)
from ashare_data.processor import open_to_open_returns  # noqa: E402
from ashare_model.backtest import AshareBacktestEngine  # noqa: E402
from ashare_model.data_loader import AshareDataLoader  # noqa: E402
from ashare_model.eval_folds import epoch_slice, resolve_folds  # noqa: E402
from ashare_model.r01_baseline import combo_signal  # noqa: E402
from ashare_model.r02_matrix import R02_REFERENCE_CELL, cell_variant  # noqa: E402
from ashare_model.vocab import FEATURE_NAMES  # noqa: E402
from ashare_model.wp18_five_questions import (  # noqa: E402
    annualized_from_cumulative,
    arithmetic_cumulative,
    cash_exposure_decomposition,
    geometric_cumulative_excess,
    reprice_fixed_path_fees,
)
from ashare_model.wp18_signal_vs_account import (  # noqa: E402
    WP18_MODES,
    WP18_RULES,
    load_frozen_directions,
    wp18_candidates,
    wp18_domain_spec,
)

from ashare_portfolio.constructor import HoldModeConstructor  # noqa: E402

DEV_NOTE = "开发验证（development validation, 2021-2025 测试窗）"
DELAY = 1  # engine default execution_delay (next-open execution)
OUT_DIR = _ROOT / "reports/wp18/_analysis"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _replay_row(engine_cfg, oriented, loader, fold, fold_data, mode):
    """Run the engine exactly as evaluate_signal does, return the result."""

    engine = (
        AshareBacktestEngine(engine_cfg)
        if mode == "dynamic"
        else AshareBacktestEngine(
            engine_cfg, portfolio_constructor=HoldModeConstructor(engine_cfg)
        )
    )
    _, raw, _target, dates = fold_data
    return engine.run(
        oriented,
        raw,
        loader.ts_codes,
        dates,
        signal_range=fold_data.local_signal_range,
        universe_mask=fold_data.universe_mask,
        rebalance_mask=fold_data.rebalance_mask,
    )


def _decompose(result, raw, account_row, engine_cfg, base_config, code_to_idx):
    """Q4 (actual-holdings decomposition) + Q5 (fixed-path fee repricing).

    Returns a dict of fold-level series/aggregate inputs, or raises on any
    identity violation.  All weights here are the ACHIEVED post-trade book
    (shares x entry open / post-fee equity), never the constructor target.
    """

    n_sess = len(result.daily_returns)
    adjusted = np.nan_to_num(
        np.asarray(raw["adjusted_open"], dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    open_ = np.nan_to_num(
        np.asarray(raw["open"], dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    target_ret = open_to_open_returns(adjusted)
    n_stocks = open_.shape[0]

    net = np.asarray(result.daily_returns, dtype=np.float64)
    cf = np.asarray(
        [0.0 if c is None else float(c) for c in (result.cost_fractions or [])],
        dtype=np.float64,
    )
    initial_capital = float(engine_cfg.initial_capital)
    # capital_before[k] = initial * prod_{s<k}(1+net[s])  (exact recursion)
    capital_before = initial_capital * np.cumprod(np.concatenate(([1.0], 1.0 + net)))[:n_sess]
    capital_after = capital_before * (1.0 - cf)
    gross_implied = (1.0 + net) / (1.0 - cf) - 1.0

    # --- actual achieved weights per session ------------------------------
    # Replicates the engine's own bookkeeping exactly (backtest.py):
    # a session's gross-return weights are the DRIFTED book, and only a
    # due / not_due_forced_exit session re-derives shares from the drifted
    # weights at the entry open (positions[].shares are STALE on non-due
    # sessions — never use them there).  Drift per session k>0:
    #   w_k_pre = w_{k-1} * clip(1+R[:,k-1],0) / max(1+g_{k-1},0)
    #   (rescaled if the total exceeds 1, non-finite -> 0), then for
    #   execution sessions w_k = shares*p_entry/capital_after with
    #   missing-bar names left at their drifted notional
    #   (= w_k_pre * capital_before).
    R = target_ret[:, :n_sess]  # engine books gross on target_ret[:, t]
    R_prev_col: np.ndarray | None = None
    diag = result.construction_diagnostics or []
    w = np.zeros((n_stocks, n_sess), dtype=np.float64)
    w_prev = np.zeros(n_stocks, dtype=np.float64)
    g_prev = 1.0
    for k, pos in enumerate(result.positions):
        # 1) drift the previous post-trade book to this session's entry
        if k >= 1:
            growth = np.clip(1.0 + R_prev_col, 0.0, None)
            book_growth = max(1.0 + g_prev, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                drifted = w_prev * growth / (
                    book_growth if book_growth > 0.0 else 1.0
                )
            total = float(drifted.sum())
            if total > 1.0:
                drifted = drifted / total
            w_cur = np.where(np.isfinite(drifted), drifted, 0.0)
        else:
            w_cur = np.zeros(n_stocks, dtype=np.float64)
        # 2) execution sessions: exact achieved book from the share record
        #    (the engine executes the share re-derivation on due sessions
        #    and on hold-mode non-due forced-exit legs; the latter is the
        #    only non-due execution path — backtest.py's gate reads
        #    output.reason == "not_due_forced_exit", whose diagnostic
        #    footprint is hold_forced_exit=True with rebalance_due=False)
        session_diag = diag[k] if k < len(diag) else {}
        executed = bool(session_diag.get("rebalance_due")) or bool(
            session_diag.get("hold_forced_exit")
        )
        if executed:
            codes = pos["ts_codes"]
            idxs = [code_to_idx[c] for c in codes]
            entry = k + DELAY
            p = open_[idxs, entry]
            shares = np.asarray(pos["shares"], dtype=np.float64)
            marked = shares * p
            missing = ~(p > 0.0)
            if missing.any():
                # engine marks a missing-bar name at its drifted notional
                marked[missing] = w_cur[idxs][missing] * capital_before[k]
            w_cur = np.zeros(n_stocks, dtype=np.float64)
            w_cur[idxs] = marked / capital_after[k]
        w[:, k] = w_cur
        w_prev = w_cur
        g_prev = float(gross_implied[k])
        R_prev_col = R[:, k]

    book_gross = (w * R).sum(axis=0)
    gross_err = float(np.max(np.abs(book_gross - gross_implied))) if n_sess else 0.0

    bench = np.asarray(account_row["benchmark_daily_returns"], dtype=np.float64)
    if bench.shape[0] != n_sess:
        raise ValueError("benchmark series length mismatch")

    dec = cash_exposure_decomposition(w, R, bench, periods_per_year=float(n_sess))
    fee_drag = net - book_gross  # exact residual: net excess = exp + sel + fee
    net_excess = net - bench
    if not np.allclose(
        np.asarray(dec["exposure_effect"]) + np.asarray(dec["selection_effect"])
        + fee_drag, net_excess, atol=1e-10,
    ):
        raise ValueError("session identity net_excess = exposure+selection+fee failed")

    # --- Q5 fixed-path fee repricing (same notionals, new rates) ---------
    from ashare_execution import ExecutionCostModel
    from ashare_model.r01_baseline import cost_variant

    model_2x = ExecutionCostModel.from_config(cost_variant(base_config, "2x"))
    fees_1x_sessions = cf * capital_before

    def _fixed_path(level: str) -> dict:
        if level == "0x":
            f_sessions = np.zeros(n_sess)
        else:
            f_sessions = np.zeros(n_sess)
            for k in range(n_sess):
                bw = np.asarray(result.buy_weights[k], dtype=np.float64)
                sw = np.asarray(result.sell_weights[k], dtype=np.float64)
                bn = bw[bw > 1e-12] * capital_before[k]
                sn = sw[sw > 1e-12] * capital_before[k]
                total = 0.0
                if bn.size:
                    total += float(np.sum(model_2x.buy_cost(bn).total))
                if sn.size:
                    total += float(np.sum(model_2x.sell_cost(sn).total))
                f_sessions[k] = total
        equity = initial_capital
        for k in range(n_sess):
            equity = (equity - f_sessions[k]) * (1.0 + gross_implied[k])
        total_ret = equity / initial_capital - 1.0
        bench_tot = float(account_row["benchmark_return"])
        excess = (
            (1.0 + total_ret) / (1.0 + bench_tot) - 1.0
            if bench_tot > -1.0 else 0.0
        )
        return {
            "level": level,
            "fees_total": float(f_sessions.sum()),
            "fees_1x_total_actual": float(fees_1x_sessions.sum()),
            "total_return": float(total_ret),
            "excess_return": float(excess),
        }

    # 1x sanity: fixed-path recursion with the ACTUAL fees must reproduce
    # the row's own total to float precision.
    equity = initial_capital
    for k in range(n_sess):
        equity = (equity - fees_1x_sessions[k]) * (1.0 + gross_implied[k])
    fixed_1x_total = equity / initial_capital - 1.0

    return {
        "gross_identity_max_abs_err": gross_err,
        "n_sessions": n_sess,
        "series": {
            "invested_ratio_actual": [float(x) for x in w.sum(axis=0)],
            "net_excess": [float(x) for x in net_excess],
            "exposure_effect": dec["exposure_effect"],
            "selection_effect": dec["selection_effect"],
            "fee_drag": [float(x) for x in fee_drag],
        },
        "fold_aggregates": {
            "arithmetic_cumulative_net_excess": float(np.sum(net_excess)),
            "geometric_cumulative_net_excess": geometric_cumulative_excess(net_excess),
            "arithmetic_cumulative_exposure_effect": float(
                np.sum(dec["exposure_effect"])
            ),
            "arithmetic_cumulative_selection_effect": float(
                np.sum(dec["selection_effect"])
            ),
            "arithmetic_cumulative_fee_drag": float(np.sum(fee_drag)),
            "invested_ratio_actual_mean": float(np.mean(w.sum(axis=0))),
            "invested_ratio_target_mean_row": float(
                account_row.get("invested_ratio_mean", float("nan"))
            ),
        },
        "q5_fixed_path": {
            "sanity_fixed_1x_total_return": float(fixed_1x_total),
            "row_total_return": float(account_row["total_return"]),
            "levels": [_fixed_path("0x"), _fixed_path("2x")],
        },
        "fees_1x_total": float(fees_1x_sessions.sum()),
    }


loader_ts_codes_global: list[str] = []


def run_domain(domain: str) -> dict:
    started = time.perf_counter()
    spec = wp18_domain_spec(domain)
    candidates = wp18_candidates(domain)

    account = json.loads(
        (_ROOT / f"reports/wp18/{domain}/account_report.json").read_text(
            encoding="utf-8"
        )
    )
    rows_by_key = {
        (r["candidate"], r["fold_test_end"]): r for r in account["rows"]
    }

    raw = load_config(str(_ROOT / spec["config_path"]), project_root=_ROOT)
    data_config = make_data_config(raw, _ROOT)
    if domain == "slow":
        # §9(b): the slow replay must run on the WP18 recovery data_dir
        # (its factor_tensor_cache holds the current tensor identity).
        import dataclasses

        data_config = dataclasses.replace(
            data_config, data_dir=_ROOT / "data/r1/wp18_slow"
        )
    proto = make_protocol_config(raw)
    model_cfg = make_model_config(raw)
    bt_cfg = make_backtest_config(raw)
    if domain == "medium":
        base_config = cell_variant(bt_cfg, R02_REFERENCE_CELL)
    else:
        base_config = bt_cfg

    loader = AshareDataLoader(data_config, model_cfg)
    loader.load_data()
    cache_outcome = str(getattr(loader, "factor_cache_outcome", "unknown"))
    print(f"[{domain}] factor cache outcome={cache_outcome}", flush=True)
    if cache_outcome != "hit":
        raise SystemExit(
            f"[{domain}] factor cache outcome={cache_outcome!r} — analysis "
            "replay refuses a cold rebuild (FROZEN_18_PLAN §9 discipline)"
        )
    global loader_ts_codes_global
    loader_ts_codes_global = list(loader.ts_codes)
    code_to_idx = {c: i for i, c in enumerate(loader_ts_codes_global)}

    folds = resolve_folds(
        proto.folds, loader.dates,
        frequency=proto.frequency, horizon=proto.horizon,
    )
    book = load_frozen_directions(
        _ROOT / f"reports/wp18/{domain}/direction_frozen.json",
        expected_keys=None,
    )

    stage = {"medium": "M3", "slow": "S3"}[domain]
    out_rows: list[dict] = []
    mismatches: list[str] = []
    for fold in folds:
        fold_data = epoch_slice(loader, fold)
        factors = fold_data.factors
        mask = fold_data.universe_mask
        for cand_id, kind, members in candidates:
            for letter, rule_id in WP18_RULES:
                member_dirs = {
                    m: book.direction(m, letter, fold.test_end) for m in members
                }
                if kind == "single":
                    oriented = float(member_dirs[cand_id]) * np.asarray(
                        factors[FEATURE_NAMES.index(cand_id)], dtype=np.float64
                    )
                else:
                    oriented = combo_signal(
                        [
                            (
                                float(member_dirs[m]),
                                np.asarray(
                                    factors[FEATURE_NAMES.index(m)],
                                    dtype=np.float64,
                                ),
                            )
                            for m in members
                        ],
                        mask,
                    )
                for mode in WP18_MODES:
                    encoded = f"{stage}:{letter}:{mode}:{cand_id}"
                    key = (encoded, fold.test_end)
                    account_row = rows_by_key.get(key)
                    if account_row is None or account_row.get("failed"):
                        raise SystemExit(f"[{domain}] missing account row {key}")
                    result = _replay_row(
                        base_config, oriented, loader, fold, fold_data, mode
                    )
                    replayed = [float(x) for x in result.daily_returns]
                    stored = [float(x) for x in account_row["daily_returns"]]
                    if replayed != stored:
                        mismatches.append(key[0] + "/" + key[1])
                        raise SystemExit(
                            f"[{domain}] replay mismatch for {key}: daily "
                            "returns differ from the stored account row — "
                            "the replay is not faithful, refusing"
                        )
                    dec = _decompose(
                        result, fold_data.raw, account_row, base_config,
                        base_config, code_to_idx,
                    )
                    out_rows.append(
                        {
                            "candidate": encoded,
                            "rule_letter": letter,
                            "rule_id": rule_id,
                            "mode": mode,
                            "cand_id": cand_id,
                            "fold_test_end": fold.test_end,
                            "cost_level": "1x",
                            "development_validation": DEV_NOTE,
                            **dec,
                        }
                    )
        print(
            f"[{domain}] replay fold {fold.test_end}: {len(out_rows)} rows",
            flush=True,
        )

    # cross-fold aggregation per (candidate): concatenated-session 3 columns
    agg: dict[str, dict] = {}
    for r in out_rows:
        agg.setdefault(r["candidate"], {"folds": []})
    for r in out_rows:
        agg[r["candidate"]]["folds"].append(
            {
                "fold_test_end": r["fold_test_end"],
                **r["fold_aggregates"],
                "q5_fixed_path": r["q5_fixed_path"],
                "fees_1x_total": r["fees_1x_total"],
            }
        )
    for cand, blob in agg.items():
        blob["folds"].sort(key=lambda x: x["fold_test_end"])
        net_series: list[float] = []
        exp_series: list[float] = []
        sel_series: list[float] = []
        fee_series: list[float] = []
        inv_series: list[float] = []
        for r in out_rows:
            if r["candidate"] != cand:
                continue
            net_series.extend(r["series"]["net_excess"])
            exp_series.extend(r["series"]["exposure_effect"])
            sel_series.extend(r["series"]["selection_effect"])
            fee_series.extend(r["series"]["fee_drag"])
            inv_series.extend(r["series"]["invested_ratio_actual"])
        geom = geometric_cumulative_excess(net_series)
        blob["three_columns"] = {
            "arithmetic_cumulative_net_excess": arithmetic_cumulative(net_series),
            "geometric_cumulative_net_excess": geom,
            "annualized_net_excess_5y": annualized_from_cumulative(geom, years=5),
            "arithmetic_cumulative_exposure_effect": arithmetic_cumulative(exp_series),
            "arithmetic_cumulative_selection_effect": arithmetic_cumulative(sel_series),
            "arithmetic_cumulative_fee_drag": arithmetic_cumulative(fee_series),
            "invested_ratio_actual_mean_all_sessions": float(np.mean(inv_series)),
        }
        # Q5 fixed-path per-level aggregation across folds
        for li, level in enumerate(("0x", "2x")):
            lv = [
                f["q5_fixed_path"]["levels"][li] for f in blob["folds"]
            ]
            geom_lv = geometric_cumulative_excess(
                [f["excess_return"] for f in lv]
            )
            blob[f"fixed_path_{level}"] = {
                "per_fold_excess": [f["excess_return"] for f in lv],
                "fees_total_sum": float(sum(f["fees_total"] for f in lv)),
                "geometric_cumulative_excess": geom_lv,
                "annualized_5y": annualized_from_cumulative(geom_lv, years=5),
            }
        geom1x = geometric_cumulative_excess(
            [f["q5_fixed_path"]["sanity_fixed_1x_total_return"] for f in blob["folds"]]
        )
        blob["fixed_path_1x_sanity"] = {
            "per_fold_total_return": [
                f["q5_fixed_path"]["sanity_fixed_1x_total_return"]
                for f in blob["folds"]
            ],
            "row_per_fold_total_return": [
                f["q5_fixed_path"]["row_total_return"] for f in blob["folds"]
            ],
            "max_abs_total_return_gap": max(
                abs(
                    f["q5_fixed_path"]["sanity_fixed_1x_total_return"]
                    - f["q5_fixed_path"]["row_total_return"]
                )
                for f in blob["folds"]
            ),
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "wp18_analysis_replay_schema": 1,
        "domain": domain,
        "generated": "2026-09-27",
        "development_validation": DEV_NOTE,
        "purpose": (
            "Q4 cash-exposure vs selection decomposition on ACTUAL post-trade "
            "holdings + Q5 fixed-path fee repricing; deterministic replay of "
            "the 1x account rows, zero new formal evaluation rows, zero "
            "search; daily_returns verified bit-for-bit against the stored "
            "account rows"
        ),
        "factor_cache_outcome": cache_outcome,
        "replay_mismatches": mismatches,
        "row_count": len(out_rows),
        "max_gross_identity_abs_err": max(
            r["gross_identity_max_abs_err"] for r in out_rows
        ),
        "per_row": out_rows,
        "per_candidate": agg,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }
    out_path = OUT_DIR / f"replay_{domain}.json"
    out_path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[{domain}] wrote {out_path} rows={len(out_rows)} "
        f"elapsed={payload['elapsed_sec']}s",
        flush=True,
    )
    return {"domain": domain, "rows": len(out_rows), "path": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("medium", "slow"))
    args = parser.parse_args()
    print(json.dumps(run_domain(args.domain), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
