"""WP18 pre-registration trial ledger writer (FROZEN_18_PLAN §8).

Run once at pre-registration time (2026-09-27).  Registers EVERY
direction/rule/grid/combo/mode selection of this package — not only the
surviving candidates — as hash-chained JSONL entries via the ledger
authority (ExperimentLedger; the RunHandle-bound record_trial face,
§8 入口).  Location per §9: <data_dir>/runs/<spec_id>/<run_id>/
trials.jsonl with data_dir = data/r1/wp18_medium (medium runs first;
the entries themselves name which domains they concern).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_model.ledger import ExperimentLedger
from ashare_model.wp18_signal_vs_account import (
    MEDIUM_REFERENCE_CELL,
    SLOW_COMBO_ID,
    SLOW_SINGLES,
    WP18_ADJUDICATION_RULES,
    WP18_MEDIUM_COMBO_ID,
    WP18_MEDIUM_SINGLES,
    WP18_RULES,
    WP18_STAGES,
)

RUN_ID = "wp18-prereg-20260927"
RUN_DIR = Path(
    "data/r1/wp18_medium/runs/wp18_preregistration/wp18-prereg-20260927"
)

ENTRIES = [
    {
        "algorithm": "grid:reference_cell",
        "candidate": f"reference_cell={MEDIUM_REFERENCE_CELL}",
        "payload": {
            "decision": (
                "medium reference cell = R02_REFERENCE_CELL "
                f"'{MEDIUM_REFERENCE_CELL}' (r02_matrix.py:96); slow "
                "reference = the r1_slow_ab.yaml backtest block verbatim "
                "(no cell axis)"
            ),
            "rejected_alternatives": [
                {
                    "option": "n10_m1000",
                    "reason": (
                        "task-brief wording; it is only the FIRST grid cell "
                        "(r02_matrix.py:85-92), not the code authority — "
                        "rejected as the reference"
                    ),
                },
                {
                    "option": "n10_m2500",
                    "reason": (
                        "the cell the p46/wp16c measurement scripts used "
                        "(D:/minequant/reports/p46/run_measurement.py:61-64); "
                        "a third 'reference' — rejected for WP18, the code "
                        "authority wins"
                    ),
                },
            ],
            "evidence": (
                "n20_m5000 coupling (top_n=20/buy_rank=20/sell_rank=30/"
                "cap=0.05/min_trade=5000) matches the configs/"
                "r1_medium_a.yaml:59-65 backtest block verbatim; "
                "FROZEN_18_PLAN §2.5"
            ),
            "concerns_domains": ["medium", "slow"],
        },
    },
    {
        "algorithm": "grid:no_scan",
        "candidate": "reference_cell_only",
        "payload": {
            "decision": (
                "account evaluation and cost stress run on the reference "
                "cell ONLY; grid scan vetoed"
            ),
            "reason": (
                "algorithm search budget is zero (FROZEN_18_PLAN §3.6 and "
                "the hard red line: no grid sweep)"
            ),
            "concerns_domains": ["medium", "slow"],
        },
    },
    {
        "algorithm": "direction_rule:label_ic",
        "candidate": "label_ic",
        "payload": {
            "decision": (
                "formal name MINTED for Rule B: "
                "训练窗验证段之前切片对固定因果标签的截面秩 IC 符号择向"
                "（Rule B） (direction.py LABEL_IC_RULE_NAME)"
            ),
            "existing_form_pinned": (
                "window = (train_signal_start, val_start) — the "
                "pre-validation train slice "
                "(D:/minequant/reports/p46/run_measurement.py:103; "
                "eval_search.py:147-150); evidence fields {ic_mean, "
                "n_dates} (direction.py:155); no config parameters reach "
                "the rule"
            ),
            "wp18_gap_closed": (
                "no r0x report row carried Rule B provenance before; WP18 "
                "must write label_ic_direction(...).to_json() into every "
                "direction row and this ledger"
            ),
            "concerns_domains": ["medium", "slow"],
        },
    },
    {
        "algorithm": "direction_rule:b_fulltrain",
        "candidate": "b_fulltrain",
        "payload": {
            "decision": (
                "NOT retained in WP18: the full-train-window (incl. "
                "validation tail) variant of B does not exist in any code "
                "or doc (repo-wide grep zero hits) — keeping it would mint "
                "a brand-new policy identity and the formal-row budget "
                "sits exactly at the 300 cap"
            ),
            "future_rule": (
                "if ever needed it must be preregistered as its own policy "
                "identity 'B-fulltrain', never mixed with the "
                "validation-window B"
            ),
            "concerns_domains": ["medium", "slow"],
        },
    },
    {
        "algorithm": "direction_rule:a_b_main_comparison",
        "candidate": "rules=A/B",
        "payload": {
            "decision": (
                "A/B compared with both rules' evidence dates taken INSIDE "
                "the same fold's training window (A: the validation tail "
                "sub-windows; B: the pre-validation slice); neither "
                "crosses train_end, neither touches the test window"
            ),
            "rules_frozen": dict(WP18_RULES),
            "concerns_domains": ["medium", "slow"],
        },
    },
    {
        "algorithm": "combo:slow_all5",
        "candidate": SLOW_COMBO_ID,
        "payload": {
            "decision": (
                f"preregistered NEW combo id '{SLOW_COMBO_ID}' = the five "
                "existing slow singles, equal-weight rank mean "
                "(r01_baseline.combo_signal construction pattern; NOT via "
                "R01_COMBO_ARMS whose members are medium factors — the "
                "historical slow KeyError defect)"
            ),
            "members": list(SLOW_SINGLES),
            "hypothesis": (
                "分散化假设（FROZEN_18_PLAN §1.3 逐字）: 五个慢速基本面定义"
                "各自度量经济假象的不同侧面，等权秩空间平均可降低单一定义的"
                "度量噪声与定义特定偏差；等权是唯一聚合方式，无任何权重拟合，"
                "不引入新信息源。预期效果为降噪与波动下降，不预期产生超出成员"
                "的新信息。"
            ),
            "no_weight_fitting": True,
            "no_new_factors": True,
            "concerns_domains": ["slow"],
        },
    },
    {
        "algorithm": "combo:medium_all3",
        "candidate": WP18_MEDIUM_COMBO_ID,
        "payload": {
            "decision": (
                f"existing combo id '{WP18_MEDIUM_COMBO_ID}' reused "
                f"verbatim (members {list(WP18_MEDIUM_SINGLES)}); no "
                "change, registered for completeness"
            ),
            "concerns_domains": ["medium"],
        },
    },
    {
        "algorithm": "mode:hold",
        "candidate": "hold",
        "payload": {
            "decision": (
                "hold mode = first-funding session builds normally; every "
                "later due session closes active re-selection (target=prev "
                "with forced-risk-exit names zeroed, no buys — even after "
                "full liquidation); non-due sessions run the exit-only leg"
            ),
            "implementation_choices": [
                "1. constructor.construct: forced_exit computed BEFORE the "
                "not_due early return; opt-in kwarg not_due_forced_exit "
                "(default False, legacy path byte-identical)",
                "2. backtest engine execution gate admits reason "
                "'not_due_forced_exit' so the sell leg + fees book "
                "(FROZEN_18_PLAN §3.2 item 2 verbatim)",
                "3. HoldModeConstructor wrapper owns the run-level "
                "first-funding state (the re-entry-after-full-liquidation "
                "case never re-buys)",
                "additive observability: diagnostics key hold_forced_exit "
                "+ funnel key funnel_hold_forced_exit_sessions; existing "
                "key names untouched",
            ],
            "red_evidence": (
                "reports/p56_red_evidence.txt (the not_due early-exit "
                "defect reproduced on the unmodified code, then GREEN)"
            ),
            "concerns_domains": ["medium", "slow"],
        },
    },
    {
        "algorithm": "budget:formal_rows",
        "candidate": "formal_row_accounting",
        "payload": {
            "decision": (
                "formal evaluation rows = account (1x) + stress (0x/2x); "
                "cap 300 across both domains at full extension "
                "(90+40+130+40=300); measurement rows (80 direction + 100 "
                "summary) materialized and reconciled but outside the "
                "budget"
            ),
            "stage_counts": {
                f"{d}/{s}": n for (d, s), n in sorted(WP18_STAGES.items())
            },
            "adjudication_rules": dict(WP18_ADJUDICATION_RULES),
            "algorithm_search_budget": 0,
            "concerns_domains": ["medium", "slow"],
        },
    },
]


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger(RUN_DIR / "trials.jsonl", run_id=RUN_ID)
    for entry in ENTRIES:
        tid = ledger.record_trial(
            algorithm=entry["algorithm"],
            candidate=entry["candidate"],
            payload=entry["payload"],
        )
        ledger.complete_trial(tid, metrics={"registered": True})
    reloaded = ExperimentLedger(RUN_DIR / "trials.jsonl", run_id=RUN_ID)
    print(
        json_summary(
            RUN_DIR / "trials.jsonl", len(ENTRIES), len(reloaded._entries)
        )
    )


def json_summary(path: Path, written: int, reloaded: int) -> str:
    import json

    return json.dumps(
        {
            "trial_ledger": str(path),
            "entries_written": written,
            "chain_reload_verified": reloaded == written * 2,
            "entries_on_reload": reloaded,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
