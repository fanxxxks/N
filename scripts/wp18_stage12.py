"""WP18 (p56) stage-1+2 driver: direction evidence + pure-signal summary.

Contract: ``reports/wp18/FROZEN_18_PLAN.md`` (pre-registered 2026-09-27,
read-only for the implementer).  This driver executes, for one domain
(``--domain medium|slow``):

* Stage 1 (M1/S1, §4.1) — direction evidence rows, 8 singles x 2 rules
  x 5 folds across the two domains (medium 30 + slow 50):
  - Rule A (``reference_return``): ONE ``CandidateScorer.score_many``
    batch per (fold, single) through the module authority
    ``direction.reference_return_direction`` — the decision and BOTH
    fixed orientations' raw evidence come out of that single interleaved
    ±signal pass (WP16-C mechanism; the wp16c measurement precedent,
    "Rule A, one batch each (16-C)");
  - Rule B (``label_ic``): the pure-signal control on the
    pre-validation train slice ``(train_signal_start, val_start)``
    (p46/wp16c pinned form), evidence = ``label_ic_direction(...).to_json()``
    written into the row AND the trial ledger (the §2.2 gap closure);
  - the directions are learned ONCE per fold per rule and frozen to
    ``reports/wp18/<domain>/direction_frozen.json`` (§2.4); every
    learning event (both rules, agreed or not) is registered in the
    append-only trial ledger (§8).
* Stage 2 (M2/S2, §4.2) — pure-signal summary rows, 10 candidates x 2
  rules x 5 folds (medium 40 + slow 60): on each fold's TEST window the
  frozen-direction oriented signal is measured against the fixed causal
  label — dual IC (universe-masked + tradable), valid dates, quantile
  spread (Q high-low, the §7.1 additive helper) and effective coverage.
  NO engine run, NO budget constraint (the two-track discipline: signal
  diagnosis must not be inferred from budget-limited execution).

Red lines honored: zero algorithm search (no GP/searcher, no weight
fitting, no new factors — ``combo:slow_all5`` is the pre-registered
equal-weight rank mean of the five existing slow members); directions
are consumed from the frozen book only (stage 2 loads it fail-closed
with the stage-1 expected keys); all outputs go to NEW directories
(``reports/wp18/<domain>/`` and ``data/r1/wp18_<domain>/``); existing
``data/r1/**`` artifacts are read-only (the loader reuses the existing
config data_dir so the factor tensor cache HITS — verified and refused
otherwise, §9).

Usage (cwd = AlphaGPT):
    python scripts/wp18_stage12.py --domain medium
    python scripts/wp18_stage12.py --domain slow
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_data.config import (  # noqa: E402
    RewardConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
)
from ashare_data.processor import open_to_open_returns  # noqa: E402

from ashare_model.data_loader import AshareDataLoader  # noqa: E402
from ashare_model.diagnostics import factor_coverage, rank_ic_stats  # noqa: E402
from ashare_model.direction import (  # noqa: E402
    RULE_LABEL_IC,
    RULE_REFERENCE_RETURN,
    label_definition,
    label_ic_direction,
    reference_return_direction,
)
from ashare_model.eval_folds import epoch_slice, resolve_folds  # noqa: E402
from ashare_model.eval_metrics import _tradable_ic_mask  # noqa: E402
from ashare_model.eval_search import validation_start, validation_windows  # noqa: E402
from ashare_model.ledger import ExperimentLedger  # noqa: E402
from ashare_model.r01_baseline import _sha256_file, combo_signal  # noqa: E402
from ashare_model.r02_matrix import R02_REFERENCE_CELL, cell_variant  # noqa: E402
from ashare_model.targets import causal_target_returns  # noqa: E402
from ashare_model.vocab import FEATURE_NAMES  # noqa: E402
from ashare_model.wp18_five_questions import quantile_spread  # noqa: E402
from ashare_model.wp18_signal_vs_account import (  # noqa: E402
    WP18_RULES,
    load_frozen_directions,
    wp18_candidates,
    wp18_domain_spec,
    wp18_row_key_candidate,
    wp18_stage_row_keys,
    write_frozen_directions,
)

WP18_REPORT_SCHEMA_VERSION = 1
DEV_VALIDATION_NOTE = "开发验证（development validation, 2021-2025 测试窗）"
N_QUANTILES = 5
MIN_STOCKS = 10
RUN_STAMP = "wp18-stage12-20260927"


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


def run_domain(domain: str, *, data_dir: Path | None = None,
               cache_source: Path | None = None,
               relearn_failed: bool = False) -> dict:
    started = time.perf_counter()
    spec = wp18_domain_spec(domain)
    direction_stage = spec["direction_stage"]      # M1 / S1
    summary_stage = spec["summary_stage"]          # M2 / S2
    singles = [name for name, kind, _ in wp18_candidates(domain) if kind == "single"]
    candidates = wp18_candidates(domain)

    # --- config, loader (existing data_dir => factor cache hit), folds ---
    raw = load_config(str(_ROOT / spec["config_path"]), project_root=_ROOT)
    data_config = make_data_config(raw, _ROOT)
    if data_dir is not None:
        # §9 option (b): run under a NEW data_dir (writes land only in new
        # directories) after read-only-copying an existing cache with the
        # current tensor identity into it.
        data_config = dataclasses.replace(data_config, data_dir=data_dir)
    if cache_source is not None:
        src = Path(cache_source)
        if not src.is_dir():
            raise SystemExit(f"cache source {src} is not a directory")
        dst = Path(data_config.data_dir) / "factor_tensor_cache"
        dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for entry in sorted(src.iterdir()):
            if entry.is_file():
                shutil.copy2(entry, dst / entry.name)
                copied += 1
        print(
            f"[{domain}] §9(b) cache copy: {copied} entries {src} -> {dst} "
            "(source read-only, destination is the NEW run data_dir)",
            flush=True,
        )
    proto = make_protocol_config(raw)
    model_cfg = make_model_config(raw)
    reward_cfg = RewardConfig()
    bt_cfg = make_backtest_config(raw)
    if domain == "medium":
        # §2.5: the frozen medium reference = R02_REFERENCE_CELL n20_m5000
        # (== the config backtest block verbatim); reached through the
        # cell_variant authority, never a hand-rolled replace().
        reference_config = cell_variant(bt_cfg, R02_REFERENCE_CELL)
        reference_note = f"reference_cell={R02_REFERENCE_CELL}"
    else:
        # §2.1/§2.5: slow reference = the r1_slow_ab backtest block verbatim.
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
            "rebuild (~45 min) is a budget event; refusing per FROZEN_18_PLAN "
            "§9 (the run must reuse the existing cache read-only)"
        )

    folds = resolve_folds(
        proto.folds, loader.dates,
        frequency=proto.frequency, horizon=proto.horizon,
    )

    # --- trial ledger (append-only, hash-chained; §8) --------------------
    # ALWAYS under the NEW wp18 run directory (§9 layout) — a data_dir
    # override only relocates the loader's cache, never the ledger.
    ledger = ExperimentLedger(
        _ROOT / f"data/r1/wp18_{domain}/runs/wp18_stage12/{RUN_STAMP}/trials.jsonl",
        run_id=RUN_STAMP,
    )
    if cache_source is not None:
        # the §9 miss event: the slow_ab cache lacks the current tensor
        # identity (only a stale entry); the run refused before any cold
        # rebuild and recovered via the pre-registered §9(b) copy path.
        event_id = ledger.record_trial(
            algorithm="budget:factor_cache_miss",
            candidate=f"wp18_{domain}",
            payload={
                "event": (
                    "factor cache outcome=miss (reason=miss:no_entry) on the "
                    "configured data_dir; the driver REFUSED before any cold "
                    "rebuild (no ~45 min rebuild cost incurred)"
                ),
                "recovery": (
                    "FROZEN_18_PLAN §9 option (b): read-only copy of an "
                    "existing cache holding the CURRENT tensor identity "
                    "(identity = v2 duckdb + dataset_id + dates + ts_codes, "
                    "domain-independent — the same key the medium domain "
                    "hit) into a NEW run data_dir; outcome re-verified=hit"
                ),
                "cache_source": str(cache_source),
                "new_data_dir": str(data_config.data_dir),
            },
        )
        ledger.complete_trial(event_id, metrics={"cold_rebuild_seconds": 0})

    # =====================================================================
    # Stage 1 (M1/S1): direction evidence — learn once per fold per rule,
    # freeze to direction_frozen.json.  If a complete frozen book already
    # exists for this domain it is CONSUMED (fail-closed against the
    # stage-1 expected keys) and the learning leg is skipped — the
    # direction is learned exactly once per (fold, rule) and never
    # re-learned, on resume exactly as downstream (§2.4).
    # =====================================================================
    stage1_keys = wp18_stage_row_keys(domain, direction_stage)
    frozen_path = _ROOT / f"reports/wp18/{domain}/direction_frozen.json"
    decisions: list[dict] = []
    direction_rows: list[dict] = []
    stage1_failures: dict[tuple[str, str, str], str] = {}
    # --relearn-failed (FROZEN_A §8.3 adjudicated fix): the ORIGINAL frozen
    # entries are preserved verbatim (attempt 1 stands); ONLY the keys the
    # frozen book lacks (their attempt-1 learning failed) are re-learned
    # post-erratum, each merged with an explicit relearn provenance
    # (attempt number + erratum basis).  Never re-learns a held key.
    preserved_entries: list[dict] = []
    relearn_targets: set[tuple[str, str, str]] = set()
    if relearn_failed and frozen_path.is_file():
        original = json.loads(frozen_path.read_text(encoding="utf-8"))
        preserved_entries = [dict(e) for e in original["entries"]]
        held = {
            (e["candidate"], e["rule"], e["fold_test_end"])
            for e in preserved_entries
        }
        relearn_targets = {
            (key.candidate, key.rule, key.fold_test_end)
            for key in stage1_keys
        } - held
        print(
            f"[{domain}] §8.3 relearn: {len(preserved_entries)} frozen "
            f"entries preserved verbatim; {len(relearn_targets)} failed "
            "key(s) to re-learn post-erratum",
            flush=True,
        )
    blocked_buy_full, blocked_sell_full = loader.tradability_masks()
    resumed_book = None
    if frozen_path.is_file():
        try:
            resumed_book = load_frozen_directions(
                frozen_path, expected_keys=stage1_keys
            )
        except Exception as exc:
            # a partial book (stage-1 keys whose learning FAILED) is
            # still loadable — its schema/frozen/duplicate guarantees all
            # hold; only the completeness refusal fires.  Anything else
            # is a real refusal and propagates.
            if "incomplete" not in str(exc).lower():
                raise
            resumed_book = load_frozen_directions(
                frozen_path, expected_keys=None
            )
            if resumed_book.size >= len(stage1_keys):
                raise
            print(
                f"[{domain}] consuming PARTIAL frozen direction book "
                f"({resumed_book.size}/{len(stage1_keys)} entries; the "
                "missing keys' stage-1 learning failed — rows recorded "
                "failed, never re-learned)",
                flush=True,
            )
        else:
            print(
                f"[{domain}] consuming existing frozen direction book "
                f"({resumed_book.size} entries) — stage-1 learning "
                "SKIPPED, no re-learn (FROZEN_18_PLAN §2.4)",
                flush=True,
            )

    def _consume_held(name: str, letter: str, rule_id: str, fold) -> None:
        provenance = resumed_book.provenance(name, letter, fold.test_end)
        decisions.append(
            {
                "candidate": name,
                "rule": letter,
                "fold_test_end": fold.test_end,
                "direction": resumed_book.direction(
                    name, letter, fold.test_end
                ),
                "provenance": provenance,
            }
        )
        direction_rows.append(
            {
                "candidate": f"{direction_stage}:{letter}:None:{name}",
                "cost_level": "dir",
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "failed": False,
                "direction": int(
                    resumed_book.direction(name, letter, fold.test_end)
                ),
                "rule_letter": letter,
                "rule_id": rule_id,
                "decision": _jsonable(provenance),
                "development_validation": DEV_VALIDATION_NOTE,
            }
        )

    def _record_failed(name: str, letter: str, rule_id: str, fold,
                       reason: str) -> None:
        stage1_failures.setdefault((name, letter, fold.test_end), reason)
        direction_rows.append(
            {
                "candidate": f"{direction_stage}:{letter}:None:{name}",
                "cost_level": "dir",
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "failed": True,
                "failure_reason": reason,
                "rule_letter": letter,
                "rule_id": rule_id,
                "development_validation": DEV_VALIDATION_NOTE,
            }
        )

    for fold in folds:
        train_ctx: dict | None = None  # lazy: built on first learning need

        def _train_ctx():
            nonlocal train_ctx
            if train_ctx is not None:
                return train_ctx
            contract = fold.contract
            train_price_end = contract.train_label_end
            train_signal_end = contract.train_signal_end
            train_factors = loader.factor_tensor[
                :, :, :train_price_end
            ].numpy()
            train_open = loader.raw_data_cache["adjusted_open"][
                :, :train_price_end
            ].numpy()
            full_rebalance = fold.policy.rebalance_mask(loader.dates)
            train_rebalance = full_rebalance[:train_price_end]
            train_target = causal_target_returns(
                train_open, loader.dates[:train_price_end], fold.policy,
                rebalance_mask=train_rebalance,
            )
            train_target = loader.mask_by_universe(train_target)
            train_ctx = {
                "train_price_end": train_price_end,
                "train_factors": train_factors,
                "train_target": train_target,
                "train_mask": loader.universe_mask[:, :train_price_end],
                "train_realized": open_to_open_returns(train_open),
                "val_windows": validation_windows(
                    train_signal_end, model_cfg,
                    rebalance_mask=train_rebalance,
                ),
                "train_range": (
                    int(contract.train_signal_start),
                    int(validation_start(train_signal_end, model_cfg)),
                ),
                "label_def": label_definition(fold.policy),
                "train_dates": [
                    str(d) for d in loader.dates[:train_price_end]
                ],
                "train_rebalance": train_rebalance,
            }
            return train_ctx

        for name in singles:
            for letter, rule_id in WP18_RULES:
                key = (name, letter, fold.test_end)
                held = resumed_book is not None and resumed_book.has(
                    name, letter, fold.test_end
                )
                if held:
                    _consume_held(name, letter, rule_id, fold)
                    continue
                if resumed_book is not None and key not in relearn_targets:
                    # not held and not re-learning: the original stage-1
                    # learning FAILED (fail_trial in the ledger) — the row
                    # is re-materialized as the same failed row.
                    _record_failed(
                        name, letter, rule_id, fold,
                        "direction unavailable: not in the frozen book — "
                        "the stage-1 Rule A scoring was refused "
                        "(fee-financing guard; see trials.jsonl fail_trial "
                        "of the originating run)",
                    )
                    continue
                # learning leg (fresh run, or the §8.3 relearn of a key
                # whose attempt-1 learning failed)
                ctx = _train_ctx()
                is_relearn = key in relearn_targets
                sig = ctx["train_factors"][FEATURE_NAMES.index(name)]
                # every direction learning event is registered (agreed or
                # not, both rules) with compact evidence metrics; the full
                # both-sides provenance lives in direction_frozen.json and
                # the report rows.
                trial_id = ledger.record_trial(
                    algorithm=f"direction_rule:{rule_id}",
                    candidate=name,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                    payload={
                        "domain": domain,
                        "stage": direction_stage,
                        "rule_letter": letter,
                        "rule_id": rule_id,
                        "reference": reference_note,
                        "scoring_pass": (
                            "one interleaved score_many batch per (fold, "
                            "single): decision + both orientations' raw "
                            "evidence (WP16-C)"
                            if rule_id == RULE_REFERENCE_RETURN
                            else "no scoring pass — pure-signal rank IC sign"
                        ),
                        **(
                            {
                                "relearn_attempt": 2,
                                "relearn_basis": (
                                    "FROZEN_A §8.3 errata (adjudicated "
                                    "2026-09-27): attempt-1 learning was "
                                    "refused by the (C1) guard at float "
                                    "epsilon; only the failed keys are "
                                    "re-learned, held entries untouched"
                                ),
                            }
                            if is_relearn
                            else {}
                        ),
                    },
                )
                try:
                    if rule_id == RULE_REFERENCE_RETURN:
                        decision = reference_return_direction(
                            name, sig, ctx["train_target"],
                            universe_mask=ctx["train_mask"],
                            val_windows=ctx["val_windows"],
                            train_signal_range=ctx["train_range"],
                            train_cutoff=fold.train_end,
                            label_def=ctx["label_def"],
                            backtest_config=reference_config,
                            reward_config=reward_cfg,
                            blocked_buy=blocked_buy_full[
                                :, :ctx["train_price_end"]
                            ],
                            blocked_sell=blocked_sell_full[
                                :, :ctx["train_price_end"]
                            ],
                            realized_ret=ctx["train_realized"],
                            rebalance_mask=ctx["train_rebalance"],
                            tie_break_keys=np.asarray(loader.ts_codes),
                            dates=ctx["train_dates"],
                        )
                        evidence_summary = {
                            k: decision.evidence.get(k)
                            for k in (
                                "val_reward_plus", "val_reward_minus",
                                "val_icir_plus", "val_icir_minus",
                                "train_reward_plus", "train_reward_minus",
                                "tie_break_reason", "reward_clip",
                            )
                        }
                    else:
                        decision = label_ic_direction(
                            name, sig, ctx["train_target"],
                            universe_mask=ctx["train_mask"],
                            train_signal_range=ctx["train_range"],
                            train_cutoff=fold.train_end,
                            label_def=ctx["label_def"],
                        )
                        evidence_summary = {
                            "ic_mean": decision.evidence["ic_mean"],
                            "n_dates": decision.evidence["n_dates"],
                        }
                except Exception as exc:  # noqa: BLE001 — r0x §8 convention
                    # a failed direction learning is DATA (rows-layer
                    # authority): the trial closes as failed, the row is
                    # recorded failed with its reason, and the measurement
                    # continues — the frozen book simply has no entry for
                    # this key (stage 2 records the dependent row failed).
                    ledger.fail_trial(
                        trial_id, error=f"{type(exc).__name__}: {exc}"
                    )
                    reason = f"{type(exc).__name__}: {exc}"
                    _record_failed(name, letter, rule_id, fold, reason)
                    print(
                        f"[{domain}] stage1 FAILED {name}/{letter}/"
                        f"{fold.test_end}: {reason[:160]}",
                        flush=True,
                    )
                    continue
                ledger.complete_trial(
                    trial_id,
                    metrics=_jsonable(
                        {
                            "direction": int(decision.direction),
                            "evidence": evidence_summary,
                        }
                    ),
                )
                provenance_json = decision.to_json()
                if is_relearn:
                    provenance_json["wp18_relearn"] = {
                        "attempt": 2,
                        "basis": (
                            "FROZEN_A §8.3 errata (adjudicated 2026-09-27): "
                            "attempt-1 refused by the (C1) guard at float "
                            "epsilon"
                        ),
                    }
                decisions.append(
                    {
                        "candidate": name,
                        "rule": letter,
                        "fold_test_end": fold.test_end,
                        "direction": int(decision.direction),
                        "provenance": provenance_json,
                    }
                )
                direction_rows.append(
                    {
                        "candidate": (
                            f"{direction_stage}:{letter}:None:{name}"
                        ),
                        "cost_level": "dir",
                        "fold_train_end": fold.train_end,
                        "fold_test_end": fold.test_end,
                        "seed": None,
                        "failed": False,
                        "direction": int(decision.direction),
                        "rule_letter": letter,
                        "rule_id": rule_id,
                        "decision": _jsonable(provenance_json),
                        "development_validation": DEV_VALIDATION_NOTE,
                    }
                )
        print(f"[{domain}] stage1 fold {fold.test_end}: "
              f"{len(direction_rows)} direction rows so far", flush=True)

    if len(decisions) + len(stage1_failures) != len(stage1_keys):
        raise SystemExit(
            f"[{domain}] stage1 produced {len(decisions)} decisions + "
            f"{len(stage1_failures)} failures, expected {len(stage1_keys)} "
            "rows total (FROZEN_18_PLAN §4.1)"
        )
    if stage1_failures:
        print(
            f"[{domain}] stage1 completed with {len(stage1_failures)} "
            "failed learning key(s) — recorded as failed rows; the frozen "
            "book is partial and stage 2 marks the dependent rows failed",
            flush=True,
        )
    if relearn_failed and preserved_entries:
        # §8.3 merge: the held entries stay VERBATIM (attempt 1); only the
        # re-learned keys are appended, each carrying its wp18_relearn
        # provenance.  The `decisions` list also holds consumed entries for
        # the row layer — they are excluded here so the book never carries
        # a duplicate (the writer refuses re-learned keys by construction).
        merged = list(preserved_entries) + [
            d for d in decisions
            if (d["candidate"], d["rule"], d["fold_test_end"])
            in relearn_targets
        ]
        frozen_path = write_frozen_directions(
            frozen_path,
            domain=domain,
            decisions=merged,
            reference=reference_note,
        )
        print(
            f"[{domain}] direction book re-frozen post-errata: "
            f"{len(preserved_entries)} preserved + "
            f"{len(merged) - len(preserved_entries)} re-learned = "
            f"{len(merged)} entries at {frozen_path}",
            flush=True,
        )
    elif resumed_book is None:
        frozen_path = write_frozen_directions(
            frozen_path,
            domain=domain,
            decisions=decisions,
            reference=reference_note,
        )
        print(f"[{domain}] direction book frozen: {frozen_path}", flush=True)

    # =====================================================================
    # Stage 2 (M2/S2): pure-signal summary on the TEST window, oriented by
    # the frozen book ONLY.  With zero stage-1 failures the load is
    # fail-closed against the full stage-1 key set; with failures the
    # book is partial and every dependent stage-2 row is recorded FAILED
    # (direction unavailable) — never oriented by any fallback.
    # =====================================================================
    book = load_frozen_directions(
        frozen_path,
        expected_keys=None if stage1_failures else stage1_keys,
    )
    summary_rows: list[dict] = []
    member_direction_log: list[dict] = []

    for fold in folds:
        fold_data = epoch_slice(loader, fold)
        factors = fold_data.factors
        mask = fold_data.universe_mask
        target = fold_data.target
        dates = fold_data.dates
        signal_count = fold_data.signal_count
        blocked_buy = blocked_buy_full[
            :, fold.contract.test_signal_start : fold.contract.test_price_end
        ]
        tradable = _tradable_ic_mask(mask, blocked_buy, signal_count)

        for cand_id, kind, members in candidates:
            for letter, _rule_id in WP18_RULES:
                if not all(
                    book.has(m, letter, fold.test_end) for m in members
                ):
                    missing = [
                        m for m in members
                        if not book.has(m, letter, fold.test_end)
                    ]
                    summary_rows.append(
                        {
                            "candidate": (
                                f"{summary_stage}:{letter}:None:{cand_id}"
                            ),
                            "cost_level": "sig",
                            "fold_train_end": fold.train_end,
                            "fold_test_end": fold.test_end,
                            "seed": None,
                            "failed": True,
                            "failure_reason": (
                                "direction unavailable: the stage-1 frozen "
                                "book has no entry for member(s) "
                                f"{missing} under rule {letter} "
                                f"({fold.test_end}) — see the stage-1 "
                                "failed row (Rule A scoring refusal)"
                            ),
                            "kind": kind,
                            "rule_letter": letter,
                            "development_validation": DEV_VALIDATION_NOTE,
                        }
                    )
                    continue
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
                sig_slice = oriented[:, :signal_count]
                tgt_slice = target[:, :signal_count]
                msk_slice = mask[:, :signal_count]
                ic = rank_ic_stats(
                    sig_slice[None], tgt_slice, dates[:signal_count],
                    names=["signal"], eligible=msk_slice,
                )["signal"]
                ic_trad = rank_ic_stats(
                    sig_slice[None], tgt_slice, dates[:signal_count],
                    names=["signal"], eligible=tradable[:, :signal_count],
                )["signal"]
                spread = quantile_spread(
                    sig_slice, tgt_slice, universe_mask=msk_slice,
                    n_quantiles=N_QUANTILES,
                )
                coverage = factor_coverage(
                    sig_slice[None], eligible=msk_slice, names=[cand_id],
                )[cand_id]
                n_eff = (msk_slice & np.isfinite(sig_slice)).sum(axis=0)
                summary_rows.append(
                    {
                        "candidate": f"{summary_stage}:{letter}:None:{cand_id}",
                        "cost_level": "sig",
                        "fold_train_end": fold.train_end,
                        "fold_test_end": fold.test_end,
                        "seed": None,
                        "failed": False,
                        "kind": kind,
                        "rule_letter": letter,
                        "directions_consumed": member_dirs,
                        "direction_source": "frozen_book(no_relearn)",
                        "n_dates": int(signal_count),
                        "ic_mean": float(ic["ic_mean"]),
                        "icir": float(ic["icir"]),
                        "n_ic_dates": int(ic["n_dates"]),
                        "ic_mean_tradable": float(ic_trad["ic_mean"]),
                        "icir_tradable": float(ic_trad["icir"]),
                        "quantile_mean_spread": float(spread["mean_spread"]),
                        "quantile_n_dates": int(spread["n_dates"]),
                        "quantile_n": int(spread["n_quantiles"]),
                        "coverage_nonneutral": float(coverage),
                        "n_eff_mean": float(np.mean(n_eff)),
                        "n_eff_min": int(np.min(n_eff)),
                        "n_eff_max": int(np.max(n_eff)),
                        "development_validation": DEV_VALIDATION_NOTE,
                    }
                )
                member_direction_log.append(
                    {
                        "candidate": cand_id,
                        "rule": letter,
                        "fold_test_end": fold.test_end,
                        "directions": member_dirs,
                    }
                )
        print(f"[{domain}] stage2 fold {fold.test_end}: "
              f"{len(summary_rows)} summary rows so far", flush=True)

    stage2_keys = wp18_stage_row_keys(domain, summary_stage)
    if len(summary_rows) != len(stage2_keys):
        raise SystemExit(
            f"[{domain}] stage2 produced {len(summary_rows)} rows, expected "
            f"{len(stage2_keys)} (FROZEN_18_PLAN §4.2)"
        )
    expected_key_set = {
        (wp18_row_key_candidate(k), k.cost_level, k.fold_test_end)
        for k in stage2_keys
    }
    actual_key_set = {
        (r["candidate"], r["cost_level"], r["fold_test_end"])
        for r in summary_rows
    }
    if expected_key_set != actual_key_set:
        raise SystemExit(
            f"[{domain}] stage2 row keys diverge from the frozen §4.2 key set: "
            f"missing={sorted(expected_key_set - actual_key_set)[:3]} "
            f"unexpected={sorted(actual_key_set - expected_key_set)[:3]}"
        )
    stage1_encoded = {
        (wp18_row_key_candidate(k), k.cost_level, k.fold_test_end)
        for k in stage1_keys
    }
    actual_dir_keys = {
        (r["candidate"], r["cost_level"], r["fold_test_end"])
        for r in direction_rows
    }
    if stage1_encoded != actual_dir_keys:
        raise SystemExit(
            f"[{domain}] stage1 row keys diverge from the frozen §4.1 key set: "
            f"missing={sorted(stage1_encoded - actual_dir_keys)[:3]} "
            f"unexpected={sorted(actual_dir_keys - stage1_encoded)[:3]}"
        )

    with ledger.trial(
        algorithm="stage:signal_summary",
        candidate=f"{domain}:{summary_stage}",
        payload={
            "domain": domain,
            "rows": len(summary_rows),
            "window": "fold test window (pure-signal, no engine, no budget)",
            "direction_consumption": (
                "frozen book only (loaded fail-closed against the stage-1 "
                "expected keys); no re-learn, no searcher"
            ),
            "combo_construction": (
                "combo_signal equal-weight rank mean of the members with "
                "their same-rule frozen directions (r02_matrix pattern; "
                "NO R01_COMBO_ARMS for slow)"
            ),
        },
    ):
        pass

    # --- stage products (§9 layout) --------------------------------------
    elapsed = time.perf_counter() - started
    report_dir = _ROOT / f"reports/wp18/{domain}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "signal_summary.json").write_text(
        json.dumps(
            {
                "wp18_stage_schema": "signal_summary",
                "wp18_stage_schema_version": 1,
                "domain": domain,
                "stage": summary_stage,
                "generated": RUN_STAMP,
                "development_validation": DEV_VALIDATION_NOTE,
                "reference": reference_note,
                "row_count": len(summary_rows),
                "rows": _jsonable(summary_rows),
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    # --- the integrity-consumable domain report (plan-declared product) --
    rows = direction_rows + summary_rows
    arms = sorted({r["candidate"] for r in rows})
    # arm_summaries.failed_folds rebuilt with the r01 semantics frozen at
    # r01_baseline.py:492-497 (a fold fails for an arm when ANY of its
    # rows failed) — reconcile compares this against its own rebuild.
    failed_folds_by_arm: dict[str, set[str]] = {}
    for row in rows:
        if row.get("failed"):
            failed_folds_by_arm.setdefault(
                row["candidate"], set()
            ).add(row["fold_test_end"])
    plan_payload = json.loads(
        (_ROOT / f"reports/wp18/{domain}/integrity/plan.json").read_text(
            encoding="utf-8"
        )
    )
    config_sha = _sha256_file(_ROOT / spec["config_path"])
    contract_sha = _sha256_file(_ROOT / "reports/wp18/FROZEN_18_PLAN.md")
    plan_config_sha = plan_payload["sources"]["config"]["sha256"]
    plan_contract_sha = plan_payload["sources"]["contract"]["sha256"]
    if str(config_sha).lower() != str(plan_config_sha).lower():
        raise SystemExit(f"[{domain}] config sha drifted from the plan")
    if str(contract_sha).lower() != str(plan_contract_sha).lower():
        raise SystemExit(f"[{domain}] frozen-plan sha drifted from the plan")

    data_root = _ROOT / f"data/r1/wp18_{domain}"
    data_root.mkdir(parents=True, exist_ok=True)
    report_payload = {
        f"wp18_{domain}_report_schema_version": WP18_REPORT_SCHEMA_VERSION,
        "experiment_id": f"wp18_{domain}",
        "domain": domain,
        "generated": RUN_STAMP,
        # the artifact is a fully generated report OF THE STAGES RUN SO FAR;
        # the plan's later stages (M3/C1 or S3/C2) are disclosed below and
        # drive the status label to measurement_incomplete (honest).
        "completed": True,
        "stages_complete": [direction_stage, summary_stage],
        "stages_pending": [spec["account_stage"], spec["stress_stage"]],
        "development_validation": DEV_VALIDATION_NOTE,
        "reference": reference_note,
        "provenance": {
            "config_path": spec["config_path"],
            "config_sha256": config_sha,
            "contract_path": "reports/wp18/FROZEN_18_PLAN.md",
            "contract_sha256": contract_sha,
            "factor_cache_outcome": cache_outcome,
            "factor_cache_reason": cache_reason,
        },
        "row_count": len(rows),
        "arm_summaries": {
            arm: {
                "failed_folds": len(failed_folds_by_arm.get(arm, ())),
                "n_rows": 0,
            }
            for arm in arms
        },
        "rows": _jsonable(rows),
    }
    for row in rows:
        report_payload["arm_summaries"][row["candidate"]]["n_rows"] += 1
    report_path = data_root / f"wp18_{domain}_report.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_meta_path = data_root / f"wp18_{domain}_run_meta.json"
    run_meta_path.write_text(
        json.dumps(
            {
                "run_id": RUN_STAMP,
                "stages": [direction_stage, summary_stage],
                "row_count": len(rows),
                "direction_rows": len(direction_rows),
                "summary_rows": len(summary_rows),
                "failed_rows": sum(1 for r in rows if r.get("failed")),
                "stage1_failed_keys": sorted(
                    f"{c}/{r}/{f}" for (c, r, f) in stage1_failures
                ),
                "factor_cache_outcome": cache_outcome,
                "elapsed_sec": round(elapsed, 1),
                "algorithm_search_budget": 0,
                "development_validation": DEV_VALIDATION_NOTE,
            },
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (report_dir / "run_meta.json").write_text(
        (data_root / f"wp18_{domain}_run_meta.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    return {
        "domain": domain,
        "direction_rows": len(direction_rows),
        "summary_rows": len(summary_rows),
        "elapsed_sec": round(elapsed, 1),
        "factor_cache_outcome": cache_outcome,
        "frozen_book": str(frozen_path),
        "report": str(report_path),
        "run_meta": str(run_meta_path),
        "trials_ledger": str(ledger.path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=("medium", "slow"))
    parser.add_argument(
        "--data-dir",
        default=None,
        help="override data_dir (§9(b): run under a NEW data_dir)",
    )
    parser.add_argument(
        "--cache-source",
        default=None,
        help="existing factor_tensor_cache dir to copy read-only into the "
        "run data_dir before load (§9(b) miss recovery)",
    )
    parser.add_argument(
        "--relearn-failed",
        action="store_true",
        help="FROZEN_A §8.3 adjudicated fix: re-learn ONLY the stage-1 "
        "keys the frozen book lacks (their attempt-1 learning failed); "
        "held entries are preserved verbatim and never re-learned",
    )
    args = parser.parse_args()
    result = run_domain(
        args.domain,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        cache_source=Path(args.cache_source) if args.cache_source else None,
        relearn_failed=args.relearn_failed,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
