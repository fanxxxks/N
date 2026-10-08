"""WP18 (p56) conditional cost-stress driver: stages C1/C2 (RE-EXECUTION).

Contract: ``reports/wp18/FROZEN_18_PLAN.md`` §5 + §4.5 (pre-registered
2026-09-27, read-only for the implementer).  This driver executes the
CONDITIONAL cost-stress stage of ONE domain and NOTHING else:

* §5 1x screen first: the judgment unit is (domain, main combo, rule).
  The main combo's 1x account rows (stage M3/S3 product, reference grid,
  concatenated test years 2021-2025) must satisfy ALL THREE frozen
  mechanical conditions on the MAIN-STRATEGY (dynamic) leg for (i)/(ii)
  and the paired dynamic/hold legs for (iii):
    (i)   >= 3/5 test years with positive net excess,
    (ii)  positive annual median net excess,
    (iii) dynamic-minus-hold annualized net excess >= 50bp
          (each leg (1+cumulative)^(1/5)-1 FIRST, then differenced).
  The canonical math is ``wp18_five_questions.mechanical_screen`` (p56
  tested); the annotation is frozen: 研发资源分配门槛，非统计显著性证明
  也非市场规律.
* Passing rules only: 0x and 2x RE-EXECUTED rows for the main combo —
  1 main combo x passing rules x 2 modes (dynamic/hold) x 2 levels
  (0x/2x) x 5 folds, <= 40 rows per domain (§4.5).  This is the
  RE-EXECUTION stress reading of §7.5 — the fixed-path fee repricing is
  a DIFFERENT question answered elsewhere; the two never merge.
* Non-passing rules: NO stress rows; the skip decision, its evidence
  and the failing conditions are recorded in the trial ledger and in
  ``cost_stress.json`` (§8: every screening decision registered,
  including the rejected ones).

Red lines honored:
* directions consumed from the frozen stage-1 book ONLY (no re-learn,
  §2.4 — the stress rows reuse the SAME frozen directions the 1x
  account rows consumed, across modes and across cost levels);
* zero algorithm search: no GP/searcher, no weight fitting, no new
  factors, reference grid only (§3.6);
* cost levels via the single authority ``r01_baseline.cost_variant``
  (0x = all five parameters zero, 2x = doubled rates with the
  commission floor unchanged; §3.3);
* run gate: WP14 precheck + config/contract sha anchor BEFORE anything
  runs (§10.2); outputs only into the §9 WP18 directories;
* formal-row accounting: stress rows count against the two-domain 300
  cap (§4.6) and the driver refuses to exceed it.

Usage (cwd = AlphaGPT): python scripts/wp18_cost_stress.py --domain medium|slow
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
from ashare_model.eval_metrics import evaluate_signal  # noqa: E402
from ashare_model.experiment_integrity.plan import (  # noqa: E402
    materialize_plan,
)
from ashare_model.experiment_integrity.precheck import run_precheck  # noqa: E402
from ashare_model.ledger import ExperimentLedger  # noqa: E402
from ashare_model.r01_baseline import (  # noqa: E402
    _sha256_file,
    combo_signal,
    cost_variant,
)
from ashare_model.r02_matrix import R02_REFERENCE_CELL, cell_variant  # noqa: E402
from ashare_model.vocab import FEATURE_NAMES  # noqa: E402
from ashare_model.wp18_five_questions import (  # noqa: E402
    geometric_cumulative_excess,
    mechanical_screen,
)
from ashare_model.wp18_signal_vs_account import (  # noqa: E402
    MECHANICAL_SCREEN_ANNOTATION,
    STRESS_LEVELS,
    WP18_MODES,
    WP18_RULES,
    load_frozen_directions,
    wp18_candidates,
    wp18_domain_spec,
    wp18_row_key_candidate,
    wp18_stage_row_keys,
)

from ashare_portfolio.constructor import HoldModeConstructor  # noqa: E402

WP18_STRESS_SCHEMA_VERSION = 1
DEV_VALIDATION_NOTE = "开发验证（development validation, 2021-2025 测试窗）"

#: §9(b) recovery data_dir for the slow domain (its factor_tensor_cache
#: holds the CURRENT tensor identity; E3 discipline: never write into any
#: pre-existing cache dir).
SLOW_DATA_DIR = _ROOT / "data/r1/wp18_slow"

#: (i)/(ii) leg reading, pinned ONCE here and carried in every screening
#: record: §5 names no mode for conditions (i)/(ii) and the judgment unit
#: is (domain, main combo, rule) — mode-free — so the annual net-excess
#: series is the MAIN STRATEGY leg (dynamic); §6.1 #5 states the chain
#: as properties of the executable strategy ("拼接净超额与 active IR 为
#: 正、…、动态优于持有").  The hold leg's (i)/(ii) numbers are recorded
#: alongside as audit fields, never merged into the verdict.
CONDITIONS_I_II_LEG = "dynamic"

_DOMAIN_DELTA: dict[str, dict] = {
    "medium": {
        "stress_stage": "C1",
        "run_stamp": "wp18-c1-20260927",
        "ledger_spec": "wp18_c1",
        "account_stage": "M3",
        "prior_row_count": 160,
        "use_reference_cell": True,
        "override_data_dir": None,
    },
    "slow": {
        "stress_stage": "C2",
        "run_stamp": "wp18-c2-20260927",
        "ledger_spec": "wp18_c2",
        "account_stage": "S3",
        "prior_row_count": 240,
        "use_reference_cell": False,
        "override_data_dir": SLOW_DATA_DIR,
    },
}


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


def _account_rows_for_screen(domain: str, delta: dict) -> dict:
    """The 1x account rows of the domain's main combo (§5 input face).

    Fail-closed (stop-condition #1: measurement integrity before any
    economic adjudication): the stage-M3/S3 product must exist, carry
    exactly the 2 rules x 2 modes x 5 folds main-combo rows, with NO
    failed row among them."""

    spec = wp18_domain_spec(domain)
    combo_id = spec["combo"][0]
    account_stage = delta["account_stage"]
    path = _ROOT / f"reports/wp18/{domain}/account_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != account_stage:
        raise SystemExit(
            f"[{domain}] account_report.json stage is "
            f"{payload.get('stage')!r}, expected {account_stage!r}"
        )
    if payload.get("failed_rows"):
        raise SystemExit(
            f"[{domain}] account stage carries {payload['failed_rows']} "
            "failed rows — the §5 screen may not run on an incomplete "
            "measurement (stop-condition #1)"
        )
    wanted = {
        f"{account_stage}:{letter}:{mode}:{combo_id}"
        for letter, _ in WP18_RULES
        for mode in WP18_MODES
    }
    rows = [
        r for r in payload["rows"] if r.get("candidate") in wanted
    ]
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row["candidate"]), str(row["fold_test_end"]))
        if key in by_key:
            raise SystemExit(
                f"[{domain}] duplicate main-combo account row {key}"
            )
        by_key[key] = row
    expected_folds = sorted(
        k.fold_test_end
        for k in wp18_stage_row_keys(domain, account_stage)
        if k.candidate == combo_id and k.rule == "A" and k.mode == "dynamic"
    )
    missing = [
        (c, f)
        for c in sorted(wanted)
        for f in expected_folds
        if (c, f) not in by_key
    ]
    if missing:
        raise SystemExit(
            f"[{domain}] main-combo 1x account rows incomplete: "
            f"{len(missing)} missing starting with {missing[:3]} — the "
            "screen may not run on a partial measurement"
        )
    for row in by_key.values():
        if row.get("failed"):
            raise SystemExit(
                f"[{domain}] main-combo account row "
                f"{row['candidate']}/{row['fold_test_end']} is failed"
            )
    return {
        "combo_id": combo_id,
        "account_stage": account_stage,
        "rows": by_key,
        "folds": expected_folds,
        "account_report": str(path),
    }


def _leg_series(account: dict, letter: str, mode: str) -> list[float]:
    stage = account["account_stage"]
    combo = account["combo_id"]
    encoded = f"{stage}:{letter}:{mode}:{combo}"
    return [
        float(account["rows"][(encoded, fold)]["excess_return"])
        for fold in account["folds"]
    ]


def run_domain(domain: str) -> dict:
    if domain not in _DOMAIN_DELTA:
        raise SystemExit(f"unknown domain {domain!r}")
    delta = _DOMAIN_DELTA[domain]
    stress_stage = delta["stress_stage"]
    spec = wp18_domain_spec(domain)
    started = time.perf_counter()

    integ_dir = _ROOT / f"reports/wp18/{domain}/integrity"
    plan_payload = json.loads(
        (integ_dir / "plan.json").read_text(encoding="utf-8")
    )
    _precheck_gate(domain, plan_payload)

    # --- §5 input: the 1x account rows of the main combo -------------------
    account = _account_rows_for_screen(domain, delta)
    combo_id = account["combo_id"]

    # --- trial ledger (append-only; §8 — every decision, incl. rejects) ----
    ledger = ExperimentLedger(
        _ROOT
        / f"data/r1/wp18_{domain}/runs/{delta['ledger_spec']}/"
        / delta["run_stamp"]
        / "trials.jsonl",
        run_id=delta["run_stamp"],
    )

    # --- §5 adjudication per (domain, main combo, rule) --------------------
    adjudication: dict[str, dict] = {}
    passing_letters: list[str] = []
    for letter, rule_id in WP18_RULES:
        dyn = _leg_series(account, letter, "dynamic")
        hold = _leg_series(account, letter, "hold")
        dyn_cum = geometric_cumulative_excess(dyn)
        hold_cum = geometric_cumulative_excess(hold)
        screen = mechanical_screen(
            dyn, dyn_cum, hold_cum, years=5, threshold_bp=50.0
        )
        failed_conditions = [
            name
            for name, ok in (
                ("i_3_of_5_positive", screen["condition_i_3_of_5_positive"]),
                ("ii_median_positive", screen["condition_ii_median_positive"]),
                (
                    "iii_dynamic_minus_hold_50bp",
                    screen["condition_iii_dynamic_minus_hold_50bp"],
                ),
            )
            if not ok
        ]
        passed = bool(screen["passed"])
        record = {
            "judgment_unit": {
                "domain": domain,
                "main_combo": combo_id,
                "rule_letter": letter,
                "rule_id": rule_id,
            },
            "basis": (
                f"1x account rows, reference grid, concatenated test "
                f"years 2021-2025 ({account['account_report']})"
            ),
            "conditions_i_ii_leg": CONDITIONS_I_II_LEG,
            "conditions_i_ii_leg_reading": (
                "§5 names no mode for (i)/(ii) and the judgment unit is "
                "mode-free; the annual net-excess series is the main "
                "strategy (dynamic) leg; the hold leg's numbers are "
                "recorded as audit fields, never merged into the verdict"
            ),
            "annual_net_excess_dynamic": dyn,
            "annual_net_excess_hold_audit": hold,
            "dynamic_cumulative_net_excess": dyn_cum,
            "hold_cumulative_net_excess": hold_cum,
            "screen": screen,
            "annotation": MECHANICAL_SCREEN_ANNOTATION,
            "passed": passed,
            "failed_conditions": failed_conditions,
            "stress_rows_authorized": 20 if passed else 0,
            "skip_reason": (
                None
                if passed
                else (
                    "FROZEN_18_PLAN §5: conditions "
                    f"{'+'.join(failed_conditions)} not met — no 0x/2x "
                    "stress rows for this rule (研发资源分配门槛)"
                )
            ),
        }
        trial_id = ledger.record_trial(
            algorithm="screen:1x_initial",
            candidate=f"{domain}:{combo_id}:{letter}",
            payload=record,
        )
        ledger.complete_trial(trial_id, metrics={
            "passed": passed,
            "n_positive_years": screen["n_positive_years"],
            "median_net_excess": screen["median_net_excess"],
            "dynamic_minus_hold_annualized": screen[
                "dynamic_minus_hold_annualized"
            ],
            "stress_rows_authorized": 20 if passed else 0,
        })
        adjudication[letter] = record
        if passed:
            passing_letters.append(letter)
        print(
            f"[{domain}] §5 screen {combo_id}/{letter}: "
            f"{'PASS' if passed else 'SKIP'} "
            f"(pos={screen['n_positive_years']}/5, "
            f"median={screen['median_net_excess']:.5f}, "
            f"dyn-hold ann={screen['dynamic_minus_hold_annualized']:.5f}, "
            f"failed={failed_conditions or 'none'})",
            flush=True,
        )

    rows: list[dict] = []
    failures: dict[tuple[str, str], str] = {}
    cache_outcome = "not_applicable(no stress row executed)"

    if passing_letters:
        # --- config, loader, folds (only when rows will actually run) ------
        raw = load_config(str(_ROOT / spec["config_path"]), project_root=_ROOT)
        data_config = make_data_config(raw, _ROOT)
        if delta["override_data_dir"] is not None:
            data_config = dataclasses.replace(
                data_config, data_dir=delta["override_data_dir"]
            )
        proto = make_protocol_config(raw)
        model_cfg = make_model_config(raw)
        bt_cfg = make_backtest_config(raw)
        if delta["use_reference_cell"]:
            # §2.5/§3.6: medium reference = the r02 reference cell.
            base_config = cell_variant(bt_cfg, R02_REFERENCE_CELL)
            reference_note = f"reference_cell={R02_REFERENCE_CELL}"
        else:
            # §2.5: slow reference = the backtest block verbatim.
            base_config = bt_cfg
            reference_note = "config_backtest_block_verbatim"

        loader = AshareDataLoader(data_config, model_cfg)
        loader.load_data()
        cache_outcome = str(getattr(loader, "factor_cache_outcome", "unknown"))
        cache_reason = str(getattr(loader, "factor_cache_reason", "unknown"))
        print(
            f"[{domain}] factor cache outcome={cache_outcome} "
            f"reason={cache_reason}",
            flush=True,
        )
        if cache_outcome != "hit":
            raise SystemExit(
                f"[{domain}] factor cache outcome={cache_outcome!r} — a "
                "cold rebuild is a budget event; refusing per "
                "FROZEN_18_PLAN §9"
            )

        folds = resolve_folds(
            proto.folds, loader.dates,
            frequency=proto.frequency, horizon=proto.horizon,
        )

        # --- frozen direction book: consumption ONLY (§2.4) ----------------
        stage1_keys = wp18_stage_row_keys(domain, spec["direction_stage"])
        frozen_path = _ROOT / f"reports/wp18/{domain}/direction_frozen.json"
        book = load_frozen_directions(frozen_path, expected_keys=stage1_keys)
        print(
            f"[{domain}] frozen direction book loaded: {book.size} entries "
            "(no re-learn; consumption only)",
            flush=True,
        )

        # §4.5 guard: 1 main combo x rules x 2 modes x 2 levels x 5 folds
        max_rows = 2 * len(WP18_MODES) * len(STRESS_LEVELS) * 5
        planned_rows = 20 * len(passing_letters)
        if planned_rows > max_rows:
            raise SystemExit(
                f"[{domain}] stress row set {planned_rows} exceeds the "
                f"§4.5 per-domain cap {max_rows}"
            )

        candidates = dict(
            (cand, (kind, members)) for cand, kind, members in
            wp18_candidates(domain)
        )
        _kind, members = candidates[combo_id]

        def _base_row(encoded: str, level: str, fold) -> dict:
            return {
                "candidate": encoded,
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "cost_level": level,
                "failed": False,
            }

        def _run_row(encoded, level, fold, *, letter, rule_id, mode,
                     member_dirs, fn) -> None:
            trial_id = ledger.record_trial(
                algorithm="stage:stress_row",
                candidate=encoded,
                fold_train_end=fold.train_end,
                fold_test_end=fold.test_end,
                payload={
                    "domain": domain,
                    "stage": stress_stage,
                    "kind": "combo",
                    "rule_letter": letter,
                    "rule_id": rule_id,
                    "mode": mode,
                    "cost_level": level,
                    "reference": reference_note,
                    "reading": (
                        "re-execution cost stress (full path re-run at the "
                        "level; NOT the fixed-path fee repricing — §7.5 "
                        "keeps the two questions separate)"
                    ),
                    "direction_consumption": (
                        "frozen book (stage-1 artifact, no re-learn; the "
                        "SAME directions the 1x account rows consumed)"
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
                        "stage": stress_stage,
                        "rule_letter": letter,
                        "rule_id": rule_id,
                        "mode": mode,
                        "reference": reference_note,
                        "development_validation": DEV_VALIDATION_NOTE,
                    }
                )
                rows.append(row)
                print(
                    f"[{domain}] {stress_stage} FAILED "
                    f"{encoded}/{fold.test_end}: {reason[:160]}",
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
                    "stage": stress_stage,
                    "rule_letter": letter,
                    "rule_id": rule_id,
                    "mode": mode,
                    "reference": reference_note,
                    "development_validation": DEV_VALIDATION_NOTE,
                    "directions_consumed": {
                        k: int(v) for k, v in member_dirs.items()
                    },
                    "direction_source": "frozen_book(no_relearn)",
                }
            )
            rows.append(row)

        for fold in folds:
            fold_data = epoch_slice(loader, fold)
            factors = fold_data.factors
            mask = fold_data.universe_mask
            for letter in passing_letters:
                rule_id = dict(WP18_RULES)[letter]
                member_dirs = {
                    m: book.direction(m, letter, fold.test_end)
                    for m in members
                }
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
                    encoded = f"{stress_stage}:{letter}:{mode}:{combo_id}"
                    for level in STRESS_LEVELS:
                        level_cfg = cost_variant(base_config, level)
                        if mode == "dynamic":
                            def _fn(oriented=oriented, cfg=level_cfg):
                                return evaluate_signal(
                                    oriented, loader, fold, cfg
                                )
                        else:
                            def _fn(oriented=oriented, cfg=level_cfg):
                                return evaluate_signal(
                                    oriented, loader, fold, cfg,
                                    portfolio_constructor=HoldModeConstructor(
                                        cfg
                                    ),
                                )
                        _run_row(
                            encoded, level, fold,
                            letter=letter, rule_id=rule_id, mode=mode,
                            member_dirs=member_dirs, fn=_fn,
                        )
            print(
                f"[{domain}] {stress_stage} fold {fold.test_end}: "
                f"{len(rows)} stress rows so far",
                flush=True,
            )

        # --- §4.5 executed key-set check -----------------------------------
        expected_keys = {
            (wp18_row_key_candidate(k), k.cost_level, k.fold_test_end)
            for k in wp18_stage_row_keys(domain, stress_stage)
            if k.rule in passing_letters
        }
        actual_keys = {
            (r["candidate"], r["cost_level"], r["fold_test_end"])
            for r in rows
        }
        if len(rows) != planned_rows or expected_keys != actual_keys:
            raise SystemExit(
                f"[{domain}] {stress_stage} produced {len(rows)} rows "
                f"(planned {planned_rows}); key-set divergence: "
                f"missing={sorted(expected_keys - actual_keys)[:3]} "
                f"unexpected={sorted(actual_keys - expected_keys)[:3]}"
            )

    # --- §4.6 formal-row budget guard --------------------------------------
    formal_rows = {
        ("medium", "M3"): 90, ("medium", "C1"): len(rows) if domain == "medium" else 0,
        ("slow", "S3"): 130, ("slow", "C2"): len(rows) if domain == "slow" else 0,
    }
    # the already-executed sibling domain's stress rows, read from disk
    for other, other_stage in (
        ("medium", "C1"), ("slow", "C2")
    ):
        if other == domain:
            continue
        other_report = (
            _ROOT / f"reports/wp18/{other}/cost_stress.json"
        )
        if other_report.exists():
            payload = json.loads(other_report.read_text(encoding="utf-8"))
            formal_rows[(other, other_stage)] = int(
                payload.get("row_count", 0)
            )
    total_formal = sum(formal_rows.values())
    if total_formal > 300:
        raise SystemExit(
            f"formal evaluation rows {total_formal} exceed the two-domain "
            "300 cap (FROZEN_18_PLAN §4.6)"
        )

    with ledger.trial(
        algorithm=f"stage:{stress_stage}",
        candidate=f"{domain}:{stress_stage}",
        payload={
            "domain": domain,
            "main_combo": combo_id,
            "screening": {
                letter: {
                    "passed": rec["passed"],
                    "failed_conditions": rec["failed_conditions"],
                    "skip_reason": rec["skip_reason"],
                    "n_positive_years": rec["screen"]["n_positive_years"],
                    "median_net_excess": rec["screen"]["median_net_excess"],
                    "dynamic_minus_hold_annualized": rec["screen"][
                        "dynamic_minus_hold_annualized"
                    ],
                }
                for letter, rec in adjudication.items()
            },
            "rows_executed": len(rows),
            "rows_skipped": 40 - len(rows),
            "levels": list(STRESS_LEVELS),
            "reference": (
                f"reference_cell={R02_REFERENCE_CELL}"
                if delta["use_reference_cell"]
                else "config_backtest_block_verbatim"
            ),
            "reading": (
                "re-execution cost stress (0x/2x full path re-run); the "
                "fixed-path fee repricing of §7.5 is a separate output "
                "and is never merged with these rows"
            ),
            "direction_consumption": (
                "frozen book only — the SAME frozen directions the 1x "
                "account rows consumed (§2.4)"
            ),
            "formal_row_accounting": {
                **{f"{d}/{s}": n for (d, s), n in formal_rows.items()},
                "total": total_formal,
                "cap": 300,
            },
            "annotation": MECHANICAL_SCREEN_ANNOTATION,
        },
    ):
        pass

    # --- stage products (§9 layout) -----------------------------------------
    elapsed = time.perf_counter() - started
    report_dir = _ROOT / f"reports/wp18/{domain}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "cost_stress.json").write_text(
        json.dumps(
            {
                "wp18_stage_schema": "cost_stress",
                "wp18_stage_schema_version": WP18_STRESS_SCHEMA_VERSION,
                "domain": domain,
                "stage": stress_stage,
                "generated": delta["run_stamp"],
                "development_validation": DEV_VALIDATION_NOTE,
                "reference": (
                    f"reference_cell={R02_REFERENCE_CELL}"
                    if delta["use_reference_cell"]
                    else "config_backtest_block_verbatim"
                ),
                "main_combo": combo_id,
                "screen_adjudication": adjudication,
                "conditions_i_ii_leg": CONDITIONS_I_II_LEG,
                "annotation": MECHANICAL_SCREEN_ANNOTATION,
                "row_count": len(rows),
                "failed_rows": sum(1 for r in rows if r.get("failed")),
                "rows": _jsonable(rows),
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # --- the integrity-consumable domain report: APPEND the stress rows ---
    data_root = _ROOT / f"data/r1/wp18_{domain}"
    report_path = data_root / f"wp18_{domain}_report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    prior_rows = list(report_payload["rows"])
    expected_complete = [
        spec["direction_stage"], spec["summary_stage"],
        spec["account_stage"],
    ]
    if report_payload.get("stages_complete") != expected_complete:
        raise SystemExit(
            f"[{domain}] domain report is not in the expected "
            f"post-account state: stages_complete="
            f"{report_payload.get('stages_complete')!r}"
        )
    if len(prior_rows) != delta["prior_row_count"]:
        raise SystemExit(
            f"[{domain}] domain report carries {len(prior_rows)} rows, "
            f"expected {delta['prior_row_count']} — refusing to append "
            "into an unexpected state"
        )
    all_rows = prior_rows + _jsonable(rows)
    arms = sorted({r["candidate"] for r in all_rows})
    failed_folds_by_arm: dict[str, set[str]] = {}
    for row in all_rows:
        if row.get("failed"):
            failed_folds_by_arm.setdefault(
                row["candidate"], set()
            ).add(row["fold_test_end"])
    config_sha = _sha256_file(_ROOT / spec["config_path"])
    contract_sha = _sha256_file(_ROOT / "reports/wp18/FROZEN_18_PLAN.md")
    report_payload.update(
        {
            "generated": delta["run_stamp"],
            "stages_complete": expected_complete + [stress_stage],
            "stages_pending": [],
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
            "run_id": delta["run_stamp"],
            "stages": expected_complete + [stress_stage],
            "row_count": len(all_rows),
            "stress_rows": len(rows),
            "stress_failed_keys": sorted(
                f"{c}/{f}" for (c, f) in failures
            ),
            "stress_screen": {
                letter: {
                    "passed": rec["passed"],
                    "skip_reason": rec["skip_reason"],
                }
                for letter, rec in adjudication.items()
            },
            "factor_cache_outcome": cache_outcome,
            "elapsed_sec": round(elapsed, 1),
            f"{stress_stage.lower()}_elapsed_sec": round(elapsed, 1),
            "formal_row_total": total_formal,
            "algorithm_search_budget": 0,
            "development_validation": DEV_VALIDATION_NOTE,
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
        "stage": stress_stage,
        "rows": len(rows),
        "failed_rows": sum(1 for r in rows if r.get("failed")),
        "screen": {
            letter: {
                "passed": rec["passed"],
                "failed_conditions": rec["failed_conditions"],
                "skip_reason": rec["skip_reason"],
            }
            for letter, rec in adjudication.items()
        },
        "elapsed_sec": round(elapsed, 1),
        "factor_cache_outcome": cache_outcome,
        "formal_row_total": total_formal,
        "cost_stress": str(report_dir / "cost_stress.json"),
        "domain_report": str(report_path),
        "run_meta": str(run_meta_path),
        "trials_ledger": str(ledger.path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", required=True, choices=tuple(_DOMAIN_DELTA)
    )
    args = parser.parse_args()
    result = run_domain(args.domain)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
