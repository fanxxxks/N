"""P50 WP14 errata-stage RED/GREEN tests: the r01_slow_baseline ERRATUM
(R6) is pinned to the REAL artifact, not to narrative.

Contract: ``docs/p50_experiment_integrity_draft.md`` §7 (R6; the contract
file lands with the WP14 contract-writing stage per DESIGN §1 — every
clause asserted here is pinned to a DESIGN §7 field and re-measured live
from the frozen product, never copied from the task narrative).

Invariants (errata stage — the DESIGN §9 INV numbering belongs to the
full-suite file and PL1-PL6 to the plan stage; this module uses ER1-ER7
to avoid collision):

* ER1 — the two PRE-EXISTING artifact files are byte-frozen: report
  sha256 95fac73…f550b / 2049495 bytes, run_meta sha256 3d39a934…462a /
  2209 bytes ("绝不修改 data/ 下任何既有文件" made executable);
* ER2 — the errata write surface is CLOSED: the slow directory contains
  exactly the two originals plus the two NEW errata files, nothing else;
* ER3 — errata.json target block matches the live file digests and the
  report's own completed stamp;
* ER4 — every errata fact is re-measured live at BOTH layers (rows
  authoritative, arm_summaries cross-check): 30 failed rows =
  combo:all3 15 + combo:momentum_reversal 15 on the full
  {0x,1x,2x} x 5-fold grid, the single variant "KeyError:
  'MOMENTUM_20'", the five slow singles with zero failed rows, and the
  root cause measured against research_domain (combo members outside
  the slow_fundamental AB domain, singles inside);
* ER5 — classification: measurement_program_error / overall
  measurement_incomplete / adjudication quarantined — worth_search=false
  is preserved verbatim but unreadable as economic evidence;
* ER6 — the forward rules pin the four red lines (no re-run, no
  redefinition of combo:all3, NEW candidate ids + NEW experiment plan
  for any future slow combo, load-time precheck gate) and stay
  consistent with the still-frozen R01_COMBO_ARMS;
* ER7 — ERRATUM.md is the human view of the SAME facts: digests, the
  measured numbers, the five citable singles, the classification and
  all four forward rules, every fact sentence carrying a 实测自产物
  layer annotation (DESIGN §7.2).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import pytest

from ashare_model.r01_baseline import R01_COMBO_ARMS
from ashare_model.research_domain import tier_restricted_features

#: The frozen r1 artifact root, resolved from the package (never from
#: the process cwd) — read-only access, the WP14 red line.
_ROOT = Path(__file__).resolve().parents[1]
_SLOW_DIR = _ROOT / "data" / "r1" / "r01_slow_baseline"
_REPORT = _SLOW_DIR / "r01_baseline_report.json"
_RUN_META = _SLOW_DIR / "r01_baseline_run_meta.json"
_ERRATUM_MD = _SLOW_DIR / "ERRATUM.md"
_ERRATA_JSON = _SLOW_DIR / "errata.json"
pytestmark = pytest.mark.skipif(not _REPORT.is_file(), reason="optional frozen historical run archive absent")

#: F1 (DESIGN §0): the audited report bytes.  The run_meta digest was
#: measured in the same read-only session (2209 bytes) and is pinned so
#: BOTH pre-existing files are guarded, not just the report.
_SLOW_REPORT_SHA256 = (
    "95fac73249c0ee93f86e0f385ce25ddf5ae7e7a92bc830db0706710bc05f550b"
)
_SLOW_REPORT_BYTES = 2049495
_RUN_META_SHA256 = (
    "3d39a934e2d9d8b1d75b3fb4aebf5a9603a0afa64350651a47151aef8626462a"
)
_RUN_META_BYTES = 2209

_CITABLE_SINGLES = (
    "baseline:ROE",
    "baseline:PE_TTM",
    "baseline:REVENUE_YOY",
    "baseline:DIVIDEND_YIELD",
    "baseline:MARKET_CAP",
)
_COMBO_ARMS = ("combo:all3", "combo:momentum_reversal")
_REASON_VARIANT = "KeyError: 'MOMENTUM_20'"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_report() -> dict:
    return json.loads(_REPORT.read_text(encoding="utf-8"))


def _load_errata() -> dict:
    return json.loads(_ERRATA_JSON.read_text(encoding="utf-8"))


def _facts_by_name(payload: dict) -> dict[str, dict]:
    return {fact["fact"]: fact for fact in payload["facts"]}


# --- ER1: the pre-existing artifact bytes are frozen ---------------------


def test_p50_er1_slow_artifact_digests_frozen():
    # RED if anyone ever modifies the two pre-existing files; GREEN only
    # while they stay byte-identical to the audited artifacts.
    assert _REPORT.is_file() and _RUN_META.is_file()
    assert _sha256(_REPORT) == _SLOW_REPORT_SHA256
    assert len(_REPORT.read_bytes()) == _SLOW_REPORT_BYTES
    assert _sha256(_RUN_META) == _RUN_META_SHA256
    assert len(_RUN_META.read_bytes()) == _RUN_META_BYTES


# --- ER2: the errata write surface is closed ------------------------------


def test_p50_er2_directory_surface_is_two_originals_plus_two_errata():
    # The ONLY permitted data/ write in all of WP14 is these two NEW
    # files; any third addition (or a removal) must fail here.
    surface = {p.name for p in _SLOW_DIR.iterdir()}
    assert surface == {
        "r01_baseline_report.json",
        "r01_baseline_run_meta.json",
        "ERRATUM.md",
        "errata.json",
    }
    assert _ERRATUM_MD.is_file() and _ERRATA_JSON.is_file()


# --- ER3: errata.json target block matches the live product ---------------


def test_p50_er3_errata_target_matches_live_digests_and_stamp():
    payload = _load_errata()
    target = payload["target"]
    assert target["experiment_id"] == "r01_slow_baseline"
    assert target["report_path"] == "data/r1/r01_slow_baseline/r01_baseline_report.json"
    assert target["report_sha256"] == _sha256(_REPORT) == _SLOW_REPORT_SHA256
    assert target["report_bytes"] == _SLOW_REPORT_BYTES
    report = _load_report()
    assert target["completed_as_stamped"] is True
    assert target["completed_as_stamped"] == report["completed"]
    assert target["run_meta_sha256"] == _sha256(_RUN_META)
    meta = json.loads(_RUN_META.read_text(encoding="utf-8"))
    assert target["run_meta_row_count"] == meta["row_count"] == len(report["rows"])
    assert payload["errata_schema_version"] == 1


# --- ER4: every errata fact is re-measured live, both layers --------------


def test_p50_er4_errata_facts_match_live_two_layer_measurement():
    payload = _load_errata()
    facts = _facts_by_name(payload)
    report = _load_report()
    rows = report["rows"]
    failed = [row for row in rows if row.get("failed")]

    # Narrative "两个组合合计约 30 个失败行" — measured, not copied.
    failed_fact = facts["failed_row_count"]
    assert failed_fact["value"] == len(failed) == 30
    assert failed_fact["measured_from"] == "rows layer"
    assert failed_fact["breakdown"] == dict(
        Counter(row["candidate"] for row in failed)
    ) == {"combo:all3": 15, "combo:momentum_reversal": 15}

    # Full grid, none escaped: each combo's failed set is exactly
    # {0x,1x,2x} x all five fold_test_end values.
    fold_ends = {row["fold_test_end"] for row in rows}
    assert len(fold_ends) == 5
    for candidate in _COMBO_ARMS:
        failed_cells = {
            (row["cost_level"], row["fold_test_end"])
            for row in failed
            if row["candidate"] == candidate
        }
        assert failed_cells == {
            (cost, fold) for cost in ("0x", "1x", "2x") for fold in fold_ends
        }, f"{candidate}: the failed grid must be the full 3x5 lattice"

    # The single error-string variant, at BOTH layers.
    reason_fact = facts["failure_reason_variants"]
    assert reason_fact["value"] == sorted(
        {row["failure_reason"] for row in failed}
    ) == [_REASON_VARIANT]
    summaries = report["arm_summaries"]
    for candidate in _COMBO_ARMS:
        assert summaries[candidate]["failed_folds"] == 5
        annual = summaries[candidate]["annual"]
        assert len(annual) == 5
        assert all(block["status"] == "failed" for block in annual.values())
        joined = {block["failure_reason"] for block in annual.values()}
        assert len(joined) == 1
        joined_reason = joined.pop()
        for cost in ("0x", "1x", "2x"):
            assert f"{cost}: KeyError: 'MOMENTUM_20'" in joined_reason

    # The five slow singles: zero failed rows, complete summaries —
    # citable per R6 while the combos are program errors.
    citable_fact = facts["citable_single_factor_arms"]
    assert tuple(citable_fact["value"]) == _CITABLE_SINGLES
    for name in _CITABLE_SINGLES:
        arm_rows = [row for row in rows if row["candidate"] == name]
        assert len(arm_rows) == 15
        assert not any(row.get("failed") for row in arm_rows)
        annual = summaries[name]["annual"]
        assert all(block.get("status") == "ok" for block in annual.values())
        assert all(not block.get("failure_reason") for block in annual.values())
    # Everyone else in the summary layer is failure-free too.
    others = {
        arm: block["failed_folds"]
        for arm, block in summaries.items()
        if arm not in _COMBO_ARMS
    }
    assert others and set(others.values()) == {0}

    # Root cause, measured against the domain authority right here.
    domain = set(tier_restricted_features("slow_fundamental", "AB"))
    members = {m for _, arms in R01_COMBO_ARMS for m in arms}
    assert members == {"MOMENTUM_20", "RSQ_60", "REVERSAL_60"}
    assert not (members & domain), "combo members must be outside the slow domain"
    assert all(name.split(":", 1)[1] in domain for name in _CITABLE_SINGLES)
    root_fact = facts["root_cause"]
    assert root_fact["measured_from"] == (
        "code + research_domain.tier_restricted_features"
    )
    for member in sorted(members):
        assert member in root_fact["value"]
    assert "slow_fundamental" in root_fact["value"]

    # Every fact carries its measured_from provenance tag (DESIGN §7.1).
    for fact in payload["facts"]:
        assert fact["measured_from"], f"{fact['fact']} lacks measured_from"


# --- ER5: classification quarantines the negative reading -----------------


def test_p50_er5_classification_is_program_error_not_economic():
    payload = _load_errata()
    classification = payload["classification"]
    assert classification["failed_rows_class"] == "measurement_program_error"
    assert classification["overall_label"] == "measurement_incomplete"
    assert classification["adjudication_quarantined"] is True
    # The frozen adjudication is preserved verbatim and matches the live
    # report — quarantine means "unreadable as economic evidence", never
    # "edited away".
    quarantined = classification["quarantined_as_recorded"]
    live = _load_report()["adjudication"]
    assert quarantined["worth_search"] is live["worth_search"] is False
    assert quarantined["qualified_arms"] == live["qualified_arms"] == []
    assert quarantined["bottleneck"] == live["bottleneck"]
    assert quarantined["bottleneck"].startswith("no_qualified_arm")
    assert classification["authority"] == "WP14 reconcile+status (p50)"


# --- ER6: forward rules pin the four red lines ----------------------------


def test_p50_er6_forward_rules_pin_red_lines_and_frozen_definitions():
    forward = _load_errata()["forward_rule"]
    assert set(forward) == {
        "no_rerun",
        "no_redefinition",
        "new_candidate_ids",
        "precheck_gate",
    }
    # (a) 旧产物保留原样 — no re-run, byte-identical.
    assert "byte-identical" in forward["no_rerun"]
    assert "no re-run" in forward["no_rerun"]
    # (b) 不得修改 combo:all3 旧定义 — the old ids stay frozen.
    assert "combo:all3" in forward["no_redefinition"]
    assert "frozen verbatim" in forward["no_redefinition"]
    assert "R01_COMBO_ARMS unchanged" in forward["no_redefinition"]
    # The clause must still describe the live code constant.
    assert R01_COMBO_ARMS == (
        ("combo:all3", ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")),
        ("combo:momentum_reversal", ("MOMENTUM_20", "REVERSAL_60")),
    )
    # (c) 未来测 slow 组合 = 新候选 ID + 新实验计划。
    rule = forward["new_candidate_ids"]
    assert "NEW candidate ids" in rule
    assert "NEW experiment plan" in rule
    assert "tier_restricted_features('slow_fundamental','AB')" in rule
    assert "never reuse or extend the old ids" in rule
    # (d) 装载前 precheck 门禁。
    assert "member_outside_domain" in forward["precheck_gate"]
    assert "before any future data load" in forward["precheck_gate"]


# --- ER7: ERRATUM.md is the consistent human view -------------------------


def test_p50_er7_erratum_md_is_consistent_human_view():
    text = _ERRATUM_MD.read_text(encoding="utf-8")
    # Target artifact: digest and size, verbatim.
    assert _SLOW_REPORT_SHA256 in text
    assert str(_SLOW_REPORT_BYTES) in text
    assert "completed=true" in text
    # Fact paragraphs: the measured numbers, both layers annotated.
    assert "30 个失败行" in text
    for name in _COMBO_ARMS:
        assert name in text
    assert _REASON_VARIANT in text
    for name in _CITABLE_SINGLES:
        assert name in text
    # DESIGN §7.2: every fact sentence ends with a layer annotation.
    assert text.count("（实测自产物：") >= 10
    for layer in (
        "rows layer",
        "arm_summaries layer",
        "arm_diagnostics layer",
        "file digest",
    ):
        assert f"（实测自产物：{layer}）" in text
    # Classification: program error, not an economic negative.
    assert "measurement_program_error" in text
    assert "measurement_incomplete" in text
    assert "不是经济负结果" in text
    # Forward rules: all four clauses present in the human view too.
    assert "byte-identical" in text
    assert "R01_COMBO_ARMS unchanged" in text
    assert "combo:slow_<members>_v1" in text
    assert "member_outside_domain" in text
    # Signature line: WP14 + the p50 contract reference.
    assert "WP14" in text
    assert "p50" in text
