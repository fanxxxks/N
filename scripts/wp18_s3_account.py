"""WP18 (p56) slow account-evaluation driver: stage S3 (130 rows, 1x).

Contract: ``reports/wp18/FROZEN_18_PLAN.md`` (pre-registered 2026-09-27,
read-only for the implementer).  This driver executes the SLOW domain
account stage (§4.4) and NOTHING else — the domain gate (§10.2) has been
passed by the medium domain (M1+M2+M3 complete, 455.6 s << 3600 s cap,
reconcile clean), so slow stage 3 may run:

* 6 candidates (ROE / PE_TTM / REVENUE_YOY / DIVIDEND_YIELD / MARKET_CAP /
  combo:slow_all5 — the ONE pre-registered new id, equal-weight rank mean
  of the five existing slow members, §1.2/§1.3)
  x 2 rules (A=reference_return, B=label_ic) x 2 modes
  (dynamic / hold) x 5 folds  = 120 rows at the SLOW reference =
  the ``configs/r1_slow_ab.yaml`` backtest block VERBATIM (no cell axis,
  §2.5), cost level 1x (the config verbatim);
* + 2 control arms x 5 folds = 10 rows
  (``benchmark:equal_weight`` at "ref", ``control:cash`` at "1x");
* total = 130 formal evaluation rows (§4.6).  Cumulative formal rows
  after S3: 90 (M3) + 130 (S3) = 220 <= 300 two-domain cap.

Red lines honored (same as the M3 driver, slow deltas only):
* directions are consumed from ``reports/wp18/slow/
  direction_frozen.json`` ONLY (fail-closed against the S1 key set;
  no learner import, no re-learn — §2.4; the 19 §8.3-relearned entries
  are part of that frozen book and are consumed like any other);
* zero algorithm search: no GP/searcher, no weight fitting, no new
  factors beyond the pre-registered combo id, no grid scan (§3.6);
* hold mode = the additive ``HoldModeConstructor`` on the SAME engine
  path / session grid / signal input / cost ladder (§3.2); the dynamic
  rows are the base constructor verbatim;
* run gate: WP14 precheck + config/contract sha anchor against the
  materialized plan BEFORE anything runs (§10.2);
* factor cache: run under the §9(b) recovery data_dir
  ``data/r1/wp18_slow`` whose ``factor_tensor_cache`` already holds the
  current tensor identity (ed8dc37c...); any non-hit outcome is a
  refusal (E3 discipline);
* outputs only into the WP18 new directories (§9); the stage-1+2 rows
  already in the domain report are preserved verbatim and the S3 rows
  are appended.

Usage (cwd = AlphaGPT): python scripts/wp18_s3_account.py
"""

from __future__ import annotations

import argparse
import dataclasses
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

from ashare_model.data_loader import AshareDataLoader  # noqa: E402
from ashare_model.eval_folds import epoch_slice, resolve_folds  # noqa: E402
from ashare_model.eval_metrics import benchmark_row, evaluate_signal  # noqa: E402
from ashare_model.experiment_integrity.plan import (  # noqa: E402
    materialize_plan,
)
from ashare_model.experiment_integrity.precheck import run_precheck  # noqa: E402
from ashare_model.ledger import ExperimentLedger  # noqa: E402
from ashare_model.r01_baseline import (  # noqa: E402
    _sha256_file,
    cash_signal,
    combo_signal,
)
from ashare_model.vocab import FEATURE_NAMES  # noqa: E402
from ashare_model.wp18_signal_vs_account import (  # noqa: E402
    WP18_MODES,
    WP18_RULES,
    load_frozen_directions,
    wp18_candidates,
    wp18_domain_spec,
    wp18_row_key_candidate,
    wp18_stage_row_keys,
)

from ashare_portfolio.constructor import HoldModeConstructor  # noqa: E402

WP18_REPORT_SCHEMA_VERSION = 1
DEV_VALIDATION_NOTE = "开发验证（development validation, 2021-2025 测试窗）"
RUN_STAMP = "wp18-s3-20260927"
ACCOUNT_STAGE = "S3"
#: §9(b) recovery data_dir: its factor_tensor_cache holds the CURRENT
#: tensor identity (populated by the registered read-only copy; see
#: errata E3 + the budget:factor_cache_miss trials in the slow ledger).
S3_DATA_DIR = _ROOT / "data/r1/wp18_slow"


def _jsonable(value):
    """Strict-JSON conversion (non-finite floats -> None, ndarray -> list)."""

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _precheck_gate(domain: str, plan_payload: dict) -> None:
    """§10.2 run gate: precheck exit 0 + sha anchor against the plan."""

    experiment_id = f"wp18_{domain}"
    plan = materialize_plan(experiment_id)
    report = run_precheck(plan)
    if not report.ok:
        issues = "; ".join(f"{i.code}:{i.detail[:120]}" for i in report.issues)
        raise SystemExit(
            f"[{domain}] WP14 precheck REFUSED the {experiment_id} plan "
            f"(FROZEN_18_PLAN §10.2): {issues}"
        )
    spec = wp18_domain_spec(domain)
    config_sha = _sha256_file(_ROOT / spec["config_path"])
    contract_sha = _sha256_file(_ROOT / "reports/wp18/FROZEN_18_PLAN.md")
    if str(config_sha).lower() != str(
        plan_payload["sources"]["config"]["sha256"]
    ).lower():
        raise SystemExit(f"[{domain}] config sha drifted from the plan")
    if str(contract_sha).lower() != str(
        plan_payload["sources"]["contract"]["sha256"]
    ).lower():
        raise SystemExit(f"[{domain}] frozen-plan sha drifted from the plan")
    print(f"[{domain}] precheck gate passed (run_precheck ok, sha anchored)",
          flush=True)


def run_domain(domain: str) -> dict:
    if domain != "slow":
        raise SystemExit(
            "this driver executes the SLOW account stage only (medium ran "
            "first and passed the §10.2 domain gate; see "
            "reports/wp18/medium/run_meta.json)"
        )
    started = time.perf_counter()
    spec = wp18_domain_spec(domain)
    candidates = wp18_candidates(domain)
    integ_dir = _ROOT / f"reports/wp18/{domain}/integrity"
    plan_payload = json.loads(
        (integ_dir / "plan.json").read_text(encoding="utf-8")
    )
    _precheck_gate(domain, plan_payload)

    # --- config, loader (§9(b) recovery data_dir => factor cache hit) -----
    raw = load_config(str(_ROOT / spec["config_path"]), project_root=_ROOT)
    data_config = make_data_config(raw, _ROOT)
    data_config = dataclasses.replace(data_config, data_dir=S3_DATA_DIR)
    proto = make_protocol_config(raw)
    model_cfg = make_model_config(raw)
    bt_cfg = make_backtest_config(raw)
    # §2.1/§2.5: the slow reference = the r1_slow_ab backtest block VERBATIM
    # (no cell axis — the r02 cell grid is a medium-only authority).
    reference_config = bt_cfg
    reference_note = "config_backtest_block_verbatim"

    loader = AshareDataLoader(data_config, model_cfg)
    loader.load_data()
    cache_outcome = str(getattr(loader, "factor_cache_outcome", "unknown"))
    cache_reason = str(getattr(loader, "factor_cache_reason", "unknown"))
    print(f"[{domain}] factor cache outcome={cache_outcome} reason={cache_reason}",
          flush=True)
    if cache_outcome != "hit":
        raise SystemExit(
            f"[{domain}] factor cache outcome={cache_outcome!r} — a cold "
            "rebuild is a budget event; refusing per FROZEN_18_PLAN §9 "
            "(E3 discipline: no write into any existing cache dir)"
        )

    folds = resolve_folds(
        proto.folds, loader.dates,
        frequency=proto.frequency, horizon=proto.horizon,
    )

    # --- frozen direction book: consumption ONLY, fail-closed (§2.4) -----
    stage1_keys = wp18_stage_row_keys(domain, spec["direction_stage"])
    frozen_path = _ROOT / f"reports/wp18/{domain}/direction_frozen.json"
    book = load_frozen_directions(frozen_path, expected_keys=stage1_keys)
    print(
        f"[{domain}] frozen direction book loaded: {book.size} entries "
        f"(no re-learn; consumption only)",
        flush=True,
    )

    # --- trial ledger (append-only, hash-chained; §8) ---------------------
    ledger = ExperimentLedger(
        _ROOT / f"data/r1/wp18_{domain}/runs/wp18_s3/{RUN_STAMP}/trials.jsonl",
        run_id=RUN_STAMP,
    )

    # --- the 130 account rows (§4.4) ---------------------------------------
    stage_keys = wp18_stage_row_keys(domain, ACCOUNT_STAGE)
    rows: list[dict] = []
    failures: dict[tuple[str, str], str] = {}

    def _base_row(encoded: str, level: str, fold) -> dict:
        return {
            "candidate": encoded,
            "fold_train_end": fold.train_end,
            "fold_test_end": fold.test_end,
            "seed": None,
            "cost_level": level,
            "failed": False,
        }

    def _run_row(encoded: str, level: str, fold, *, kind, letter, rule_id,
                 mode, member_dirs, fn) -> None:
        trial_id = ledger.record_trial(
            algorithm="stage:account_row",
            candidate=encoded,
            fold_train_end=fold.train_end,
            fold_test_end=fold.test_end,
            payload={
                "domain": domain,
                "stage": ACCOUNT_STAGE,
                "kind": kind,
                "rule_letter": letter,
                "rule_id": rule_id,
                "mode": mode,
                "cost_level": level,
                "reference": reference_note,
                "direction_consumption": (
                    "frozen book (stage-1 artifact, no re-learn)"
                    if member_dirs is not None else
                    "none (control arm: rule-free, mode-free)"
                ),
            },
        )
        try:
            metrics = fn()
        except Exception as exc:  # noqa: BLE001 — r0x §8 convention
            ledger.fail_trial(
                trial_id, error=f"{type(exc).__name__}: {exc}"
            )
            reason = f"{type(exc).__name__}: {exc}"
            failures[(encoded, fold.test_end)] = reason
            row = _base_row(encoded, level, fold)
            row.update(
                {
                    "failed": True,
                    "failure_reason": reason,
                    "stage": ACCOUNT_STAGE,
                    "kind": kind,
                    "rule_letter": letter,
                    "rule_id": rule_id,
                    "mode": mode,
                    "reference": reference_note,
                    "development_validation": DEV_VALIDATION_NOTE,
                }
            )
            rows.append(row)
            print(
                f"[{domain}] S3 FAILED {encoded}/{fold.test_end}: "
                f"{reason[:160]}",
                flush=True,
            )
            return
        ledger.complete_trial(
            trial_id,
            metrics={
                k: metrics.get(k)
                for k in (
                    "total_return", "excess_return", "oos_active_ir",
                    "average_turnover", "cost_fraction_sum",
                    "funnel_hold_forced_exit_sessions",
                )
            },
        )
        row = _base_row(encoded, level, fold)
        row.update(metrics)
        row.update(
            {
                "stage": ACCOUNT_STAGE,
                "kind": kind,
                "rule_letter": letter,
                "rule_id": rule_id,
                "mode": mode,
                "reference": reference_note,
                "development_validation": DEV_VALIDATION_NOTE,
            }
        )
        if member_dirs is not None:
            row["directions_consumed"] = {
                k: int(v) for k, v in member_dirs.items()
            }
            row["direction_source"] = "frozen_book(no_relearn)"
        rows.append(row)

    for fold in folds:
        fold_data = epoch_slice(loader, fold)
        factors = fold_data.factors
        mask = fold_data.universe_mask

        for cand_id, kind, members in candidates:
            for letter, rule_id in WP18_RULES:
                # fail-closed consumption: a missing member direction is a
                # refusal, never a fallback (§2.4)
                member_dirs = {
                    m: book.direction(m, letter, fold.test_end)
                    for m in members
                }
                if kind == "single":
                    oriented = float(member_dirs[cand_id]) * np.asarray(
                        factors[FEATURE_NAMES.index(cand_id)],
                        dtype=np.float64,
                    )
                else:
                    # §1.3: combo:slow_all5 = equal-weight rank mean of the
                    # five members, each oriented by its OWN same-rule
                    # frozen direction — r02_matrix combo_signal pattern.
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
                    encoded = (
                        f"{ACCOUNT_STAGE}:{letter}:{mode}:{cand_id}"
                    )
                    if mode == "dynamic":
                        def _fn(oriented=oriented):
                            return evaluate_signal(
                                oriented, loader, fold, reference_config
                            )
                    else:
                        def _fn(oriented=oriented):
                            return evaluate_signal(
                                oriented, loader, fold, reference_config,
                                portfolio_constructor=HoldModeConstructor(
                                    reference_config
                                ),
                            )
                    _run_row(
                        encoded, "1x", fold,
                        kind=kind, letter=letter, rule_id=rule_id,
                        mode=mode,
                        member_dirs=member_dirs, fn=_fn,
                    )
        # §3.5 controls: benchmark at "ref" (cost-free equal-weight
        # reference, cell-independent) and the all-NaN cash book at the
        # reference config, 1x — the r02_matrix conventions.
        _run_row(
            "benchmark:equal_weight", "ref", fold,
            kind="benchmark", letter=None, rule_id=None, mode=None,
            member_dirs=None,
            fn=lambda: benchmark_row(loader, fold),
        )
        cash = cash_signal(
            len(loader.ts_codes), len(fold_data.dates)
        )
        _run_row(
            "control:cash", "1x", fold,
            kind="control", letter=None, rule_id=None, mode=None,
            member_dirs=None,
            fn=lambda: evaluate_signal(cash, loader, fold, reference_config),
        )
        print(
            f"[{domain}] S3 fold {fold.test_end}: {len(rows)} account rows "
            "so far",
            flush=True,
        )

    # --- frozen §4.4 key-set check (130 rows, exact keys) ------------------
    if len(rows) != len(stage_keys):
        raise SystemExit(
            f"[{domain}] S3 produced {len(rows)} rows, expected "
            f"{len(stage_keys)} (FROZEN_18_PLAN §4.4)"
        )
    expected_key_set = {
        (wp18_row_key_candidate(k), k.cost_level, k.fold_test_end)
        for k in stage_keys
    }
    actual_key_set = {
        (r["candidate"], r["cost_level"], r["fold_test_end"]) for r in rows
    }
    if expected_key_set != actual_key_set:
        raise SystemExit(
            f"[{domain}] S3 row keys diverge from the frozen §4.4 key set: "
            f"missing={sorted(expected_key_set - actual_key_set)[:3]} "
            f"unexpected={sorted(actual_key_set - expected_key_set)[:3]}"
        )

    with ledger.trial(
        algorithm=f"stage:{ACCOUNT_STAGE}",
        candidate=f"{domain}:{ACCOUNT_STAGE}",
        payload={
            "domain": domain,
            "rows": len(rows),
            "cost_level": "1x (config verbatim; cost_variant identity)",
            "reference": reference_note,
            "modes": list(WP18_MODES),
            "mode_semantics": (
                "dynamic = base constructor verbatim; hold = the additive "
                "HoldModeConstructor on the SAME engine path, session grid, "
                "signal input and cost ladder (FROZEN_18_PLAN §3.2)"
            ),
            "direction_consumption": (
                "frozen book only (loaded fail-closed against the stage-1 "
                "expected keys); combo:slow_all5 consumes its five members' "
                "same-rule frozen directions (equal-weight rank mean, the "
                "pre-registered §1.3 hypothesis); no re-learn, no searcher"
            ),
            "controls": (
                "benchmark:equal_weight at ref (cost-free, cell-independent) "
                "+ control:cash at the reference config, 1x (r02 conventions)"
            ),
            "formal_row_accounting": (
                "130 formal evaluation rows (120 account + 10 control); "
                "cumulative formal rows 90 (M3) + 130 (S3) = 220 <= 300 "
                "two-domain cap (C1/C2 conditional stress still unrun)"
            ),
        },
    ):
        pass

    # --- stage products (§9 layout) ----------------------------------------
    elapsed = time.perf_counter() - started
    report_dir = _ROOT / f"reports/wp18/{domain}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "account_report.json").write_text(
        json.dumps(
            {
                "wp18_stage_schema": "account_report",
                "wp18_stage_schema_version": 1,
                "domain": domain,
                "stage": ACCOUNT_STAGE,
                "generated": RUN_STAMP,
                "development_validation": DEV_VALIDATION_NOTE,
                "reference": reference_note,
                "row_count": len(rows),
                "failed_rows": sum(1 for r in rows if r.get("failed")),
                "rows": _jsonable(rows),
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # --- the integrity-consumable domain report: APPEND the S3 rows to the
    # stage-1+2 rows (preserved verbatim) and refresh the progress fields.
    data_root = _ROOT / f"data/r1/wp18_{domain}"
    report_path = data_root / f"wp18_{domain}_report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    prior_rows = list(report_payload["rows"])
    if report_payload.get("stages_complete") != [
        spec["direction_stage"], spec["summary_stage"]
    ]:
        raise SystemExit(
            f"[{domain}] domain report is not in the expected stage-1+2 "
            f"state: stages_complete={report_payload.get('stages_complete')!r}"
        )
    if len(prior_rows) != 110:
        raise SystemExit(
            f"[{domain}] domain report carries {len(prior_rows)} stage-1+2 "
            "rows, expected 110 (50 S1 + 60 S2) — refusing to append into "
            "an unexpected state"
        )
    all_rows = prior_rows + _jsonable(rows)
    arms = sorted({r["candidate"] for r in all_rows})
    failed_folds_by_arm: dict[str, set[str]] = {}
    for row in all_rows:
        if row.get("failed"):
            failed_folds_by_arm.setdefault(
                row["candidate"], set()
            ).add(row["fold_test_end"])
    report_payload.update(        {
            "generated": RUN_STAMP,
            "stages_complete": [
                spec["direction_stage"],
                spec["summary_stage"],
                ACCOUNT_STAGE,
            ],
            "stages_pending": [spec["stress_stage"]],
            "row_count": len(all_rows),
            "arm_summaries": {
                arm: {
                    "failed_folds": len(failed_folds_by_arm.get(arm, ())),
                    "n_rows": 0,
                }
                for arm in arms
            },
            "rows": all_rows,
        }
    )
    for row in all_rows:
        report_payload["arm_summaries"][row["candidate"]]["n_rows"] += 1
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_meta_path = data_root / f"wp18_{domain}_run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    run_meta.update(
        {
            "run_id": RUN_STAMP,
            "stages": [
                spec["direction_stage"], spec["summary_stage"], ACCOUNT_STAGE
            ],
            "row_count": len(all_rows),
            "account_rows": len(rows),
            "failed_rows": sum(1 for r in all_rows if r.get("failed")),
            "s3_failed_keys": sorted(
                f"{c}/{f}" for (c, f) in failures
            ),
            "factor_cache_outcome": cache_outcome,
            "elapsed_sec": round(elapsed, 1),
            "s3_elapsed_sec": round(elapsed, 1),
            "algorithm_search_budget": 0,
            "development_validation": DEV_VALIDATION_NOTE,
            "formal_row_accounting": (
                "cumulative formal evaluation rows after S3 = 220 <= 300 "
                "(90 M3 + 130 S3; C1/C2 conditional stress unrun)"
            ),
        }
    )
    run_meta_path.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "run_meta.json").write_text(
        run_meta_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    return {
        "domain": domain,
        "stage": ACCOUNT_STAGE,
        "rows": len(rows),
        "failed_rows": sum(1 for r in rows if r.get("failed")),
        "elapsed_sec": round(elapsed, 1),
        "factor_cache_outcome": cache_outcome,
        "frozen_book": str(frozen_path),
        "account_report": str(report_dir / "account_report.json"),
        "domain_report": str(report_path),
        "run_meta": str(run_meta_path),
        "trials_ledger": str(ledger.path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("slow",))
    args = parser.parse_args()
    result = run_domain(args.domain)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
