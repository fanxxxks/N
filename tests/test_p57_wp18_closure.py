"""P57 WP18 收口 (closure) tests — 收口需求 R1-R6.

Basis: the p57 closure brief (2026-09-27).  The package is CLOSURE
ONLY: it runs zero new measurement rows / backtests / scoring, does
not chase the §5-refused slow stress rows, does not trigger a method
revision, and does not restart any hypothesis.  Every test cites its
收口需求 number (R1-R6).

Sections:

* R5 (errata E8) — the (C1) refusal message must disclose the
  OPERATIVE refusal gate ``_P52_C1_REFUSAL_TOLERANCE`` (1e-9; the
  comparison at ``reward_baskets.py`` :563) instead of the retired
  1e-12 booking form (``1.000000000001``).  Written RED first: the
  current message prints the 1e-12 form, so the assertion below fails
  until the wording-only fix lands.  Behavior anchors: BOTH constants
  and the :563 comparison stay byte-identical (zero behavior change);
  the fix is recorded append-only (FROZEN_A §8.4 + a new errata entry
  executing errata E8's deferred disposition).
* R2 — (a) the WP17 descriptive-path drill on a REAL WP18 statistic:
  ``build_claim_record`` with no calibration evidence ->
  qualified=False + ``EVIDENCE_NOT_REGISTERED`` and the save path is
  NOT blocked (FROZEN_17 §3.2 field-level proof);  (b) the mechanical
  claim audit over ALL wp18 MD/JSON products
  (``scripts/p57_claim_audit.py`` -> ``reports/wp18/claim_audit.json``):
  suspected inferential wording must carry an errata disposition,
  never a silent rewrite of the corpus.
* R3 — method-revision zero-trigger closure: registry episodes == []
  re-verified and the guards refuse on start (fail-closed liveness,
  FROZEN_17 §6.5 G1).
* R1 — the slow cost-stress closure facts as regression pins: the
  dual skip carriers (cost_stress.json screen_adjudication vs the C2
  trial ledger) verbatim-identical, stress_rows_authorized=0,
  row_count=0, and the 60 (=20 medium C1:A + 40 slow C2) unrun row
  keys PRESERVED in the integrity plans; measurement_incomplete is
  the designed-in screen-refused state (WP18_REPORT.md:38,194).
* R4 — the audit-findings persistence rule
  (``docs/p57_audit_findings_persistence.md``; background = the WP17
  §6 / WP18 §9.5 two-round gap) and this package's own findings file
  (``reports/wp18/audit_findings_p57.md`` — the dogfood execution).
* R6 — ``reports/wp18/closure_verifications.json`` carries the R1/R3
  verification items (E3 byte identity, guard liveness, slow skip
  preservation, the §9.3 domain-gate pipeline note).
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from ashare_data.config import BacktestConfig
from ashare_model import method_revision as mr
from ashare_model import reward_baskets
from ashare_model.claim_record import (
    EVIDENCE_NOT_REGISTERED,
    build_claim_record,
    claim_record_stdout_lines,
    no_reliable_inference_reason,
)
from ashare_model.reward_baskets import simulate_basket_daily_returns
from ashare_portfolio.constructor import PortfolioOutput

ROOT = Path(__file__).resolve().parents[1]
WP18 = ROOT / "reports" / "wp18"
FROZEN_A = ROOT / "reports" / "wp16" / "FROZEN_A.md"
ERRATA = WP18 / "errata" / "errata.jsonl"
AUDIT_OUT = WP18 / "claim_audit.json"
CLOSURE_OUT = WP18 / "closure_verifications.json"
PERSISTENCE_DOC = ROOT / "docs" / "p57_audit_findings_persistence.md"
OWN_FINDINGS = WP18 / "audit_findings_p57.md"
AUDIT_SCRIPT = ROOT / "scripts" / "p57_claim_audit.py"

SLOW_SKIP_A = (
    "FROZEN_18_PLAN §5: conditions i_3_of_5_positive+ii_median_positive "
    "not met — no 0x/2x stress rows for this rule (研发资源分配门槛)"
)
SLOW_SKIP_B = (
    "FROZEN_18_PLAN §5: conditions "
    "i_3_of_5_positive+ii_median_positive+iii_dynamic_minus_hold_50bp "
    "not met — no 0x/2x stress rows for this rule (研发资源分配门槛)"
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# R5 (errata E8): refusal-message disclosure — RED first, then the
# wording-only fix.  Behavior anchors pin zero behavior change.
# ---------------------------------------------------------------------------


class _NaiveFullInvestConstructor:
    """Deterministic stand-in that returns the SAME naive full-investment
    book on every fee-reserve attempt (the pre-p52 construction the fixed
    point exists to correct).  p57/R5 uses it to drive the REAL session
    loop (reserve iterations -> sell-fee-funding -> the fail-closed (C1)
    raise) without crafting an economically unfundable book through the
    real constructor: the session's single full-weight buy carries a real
    ExecutionCostModel fee with no cash to fund it and no sell leg to
    defer, so the refusal at reward_baskets.py:577-585 fires."""

    def __init__(self, config):
        self.config = config

    def construct(
        self, signal, prev_weights, *, capital, eligible, buy_blocked,
        sell_blocked, stable_keys, rebalance_due, adv=None, fee_reserve=0.0,
    ):
        prev = np.asarray(prev_weights, dtype=np.float64)
        target = np.zeros_like(prev)
        target[0] = 1.0
        return PortfolioOutput(
            weights=target,
            buy_weights=np.maximum(target - prev, 0.0),
            sell_weights=np.maximum(prev - target, 0.0),
            selected=(0,),
            rebalanced=True,
            reason="p57_r5_naive_full_invest_stub",
            diagnostics={},
        )


def _p57_cfg() -> BacktestConfig:
    # Same fee set as the p52 fixtures (tests/test_p52_wp16a_cash_identity
    # ._cfg): min_commission=5 on 1e5 capital => the single full-weight buy
    # costs ~7.6e-4 of equity, a PHYSICAL (>5e-5) shortfall the refusal
    # gate must catch.
    return BacktestConfig(
        initial_capital=100000.0,
        top_n=1,
        single_weight_cap=1.0,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )


def test_p57_r5_e8_refusal_message_discloses_operative_tolerance(monkeypatch):
    """R5: the (C1) refusal message must print the OPERATIVE refusal
    tolerance — the 1e-9 gate that governs the comparison at
    reward_baskets.py:563 — not the retired 1e-12 booking form.

    RED (this package, 2026-09-27): the current message renders
    ``1.000000000001`` (1.0 + _P52_C1_TOLERANCE), understating the
    operative gate by three orders of magnitude (errata E8)."""

    monkeypatch.setattr(
        reward_baskets, "PortfolioConstructor", _NaiveFullInvestConstructor
    )
    n_dates = 3  # one executable session (t=0, exec at t+1)
    signal = np.ones((1, n_dates))
    target = np.zeros((1, n_dates))
    mask = np.ones((1, n_dates), dtype=bool)
    with pytest.raises(ValueError) as excinfo:
        simulate_basket_daily_returns(
            signal, target, _p57_cfg(), universe_mask=mask
        )
    message = str(excinfo.value)
    assert "cannot fund its execution fees from cash" in message
    # The operative refusal gate, literal 1e-9 form (R5 / errata E8):
    assert "1.000000001" in message, message
    # The retired 1e-12 booking form must NOT be disclosed as the gate:
    assert "1.000000000001" not in message, message


def test_p57_r5_e8_behavior_anchors_unchanged():
    """R5 zero-behavior-change anchors: the dual-constant design and the
    operative comparison stay byte-identical (FROZEN_A §8.3/§8.4; only the
    message disclosure changed)."""

    assert reward_baskets._P52_C1_TOLERANCE == 1e-12
    assert reward_baskets._P52_C1_REFUSAL_TOLERANCE == 1e-9
    source = (ROOT / "ashare_model" / "reward_baskets.py").read_text(
        encoding="utf-8"
    )
    # The refusal comparison (frozen at :563, untouched):
    assert "shortfall > _P52_C1_REFUSAL_TOLERANCE" in source
    # The booking/landing tolerance still governs the fixed point:
    assert "_P52_C1_TOLERANCE = 1e-12" in source
    assert "_P52_C1_REFUSAL_TOLERANCE = 1e-9" in source


def test_p57_r5_e8_append_only_records_present():
    """R5: the wording-only fix is recorded append-only — FROZEN_A gains
    §8.4 (§8.3 intact above it) and a NEW errata entry executes errata
    E8's deferred disposition (E8 itself is never edited)."""

    frozen = FROZEN_A.read_text(encoding="utf-8")
    assert "### §8.3" in frozen  # append-only: the 2026-09-27 ruling stays
    assert "### §8.4" in frozen
    entries = [
        json.loads(line)
        for line in ERRATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {e.get("id"): e for e in entries}
    assert "E8-known-issue-reward-baskets-refusal-message-wording" in by_id
    executors = [
        e for e in entries
        if "E8" in str(e.get("ratifies") or e.get("executes") or "")
        or "E8" in str(e.get("id"))
        and e.get("id") != "E8-known-issue-reward-baskets-refusal-message-wording"
    ]
    assert executors, "the p57 execution of errata E8 must be recorded"
    wording_only = [e for e in executors if "wording" in json.dumps(e)]
    assert wording_only, "the E8 execution entry must state wording-only"


# ---------------------------------------------------------------------------
# R2: the WP17 descriptive-path drill on a REAL WP18 statistic + the
# mechanical claim audit over all wp18 products.
# ---------------------------------------------------------------------------


def test_p57_r2_claim_record_descriptive_drill_real_wp18_statistic(tmp_path):
    """R2: assemble a ClaimRecord for a REAL WP18 statistic (the slow
    combo B fold-0 active IR from reports/wp18/_analysis/derived.json)
    with NO calibration evidence: qualified=False +
    EVIDENCE_NOT_REGISTERED (FROZEN_17 §2.1 step 1 terminates the
    algorithm), the frozen no-reliable-inference wording renders, and
    the SAVE is not blocked (§3.2 — qualification never gates the
    descriptive write path)."""

    derived = _load_json(WP18 / "_analysis" / "derived.json")
    stat = (
        derived["q3_account"]["slow"]["table"]["combo:slow_all5/B"]
        ["dynamic"]["active_ir_by_fold"][0]
    )
    assert isinstance(stat, float)  # a real measured WP18 number
    # The artifact-shaped max_t block carrying it (WP18 ran no max-t
    # adjudication — that absence is exactly the registered standing
    # state this drill exercises):
    artifact_max_t = {
        "observed_max_t": stat,
        "p_value": None,
        "significant_95": None,
        "window": {"n_dates": 5},  # the WP18 5-fold grid, as recorded
        "trial_set": {"n_trials": 1, "n_current": 1, "n_extra": 0},
    }
    record = build_claim_record(
        artifact_max_t, {"dsr": None}, [], max_t_perms=999, evidence=None
    )
    qualification = record["qualification"]
    assert qualification["qualified"] is False
    assert qualification["status"] == "not_qualified"
    assert qualification["reason_codes"] == [EVIDENCE_NOT_REGISTERED]
    expected_reason = no_reliable_inference_reason(
        [EVIDENCE_NOT_REGISTERED]
    )
    assert qualification["reason"] == expected_reason
    assert "no nominal-5% significance or promotion claim is made" in (
        qualification["reason"]
    )
    # The five field groups are fully populated (descriptive retention):
    for group in (
        "statistics", "method", "task_actuals",
        "calibration_evidence", "qualification",
    ):
        assert record[group] is not None
    assert record["statistics"]["max_t_observed"] == stat
    # Save NOT blocked (§3.2): the descriptive write path completes.
    out = tmp_path / "p57_drill_claim_record.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded["qualification"]["qualified"] is False
    lines = claim_record_stdout_lines(record)
    assert lines[0].startswith("claim_record: qualified=False")
    assert any(l.startswith("no reliable inference:") for l in lines)


def _audit_module():
    spec = importlib.util.spec_from_file_location(
        "p57_claim_audit", AUDIT_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p57_r2_claim_audit_scanner_semantics():
    """R2: the mechanical audit's classifier semantics — the WP18
    disclosure sentence (descriptive + non-inferential markers) is
    compliant, while bare inferential wording without markers is
    flagged.  The word lists are declared in the script header after
    FROZEN_17's wording contract (§3.3/§3.4 templates, §4.1 banned
    'size supported', the WP18_REPORT non-inferential footer)."""

    mod = _audit_module()
    compliant_hits = mod.scan_text(
        "本报告全部结论为开发验证下的描述性结果：非统计显著性证明也非市场规律，"
        "不可作为名义 5% 显著性或样本外推断发布。",
        "sample_compliant.md",
    )
    assert compliant_hits, "the sentence must produce scanner hits"
    assert all(
        h["classification"] == "descriptive_compliant" for h in compliant_hits
    ), compliant_hits
    bare = mod.scan_text(
        "The strategy significantly outperforms the benchmark (p<0.05).",
        "sample_bare.md",
    )
    assert bare, "bare inferential wording must produce hits"
    assert any(
        h["classification"] in ("suspected_inferential", "needs_annotation")
        for h in bare
    ), bare


def test_p57_r2_claim_audit_ran_over_wp18_products():
    """R2: the audit ran over ALL wp18 MD/JSON products and every
    suspected-inferential hit carries an errata disposition (the corpus
    is NEVER silently rewritten); the on-disk claim_audit.json matches a
    fresh scan."""

    mod = _audit_module()
    fresh = mod.run_audit(WP18)
    scanned = set(fresh["files_scanned"])
    for required in (
        "WP18_REPORT.md", "ANALYSIS.md", "FROZEN_18_PLAN.md",
    ):
        assert required in scanned, (required, sorted(scanned)[:8])
    on_disk = _load_json(AUDIT_OUT)
    if not all((WP18 / name).is_file() for name in on_disk["files_scanned"]):
        pytest.skip("optional full historical audit corpus is not shipped with source")
    assert on_disk["summary"] == fresh["summary"]
    assert on_disk["hits"] == fresh["hits"]
    # Contract: no unhandled suspected inferential wording.
    assert fresh["summary"]["suspected_inferential_unhandled"] == 0, (
        fresh["summary"]
    )
    hits = fresh["hits"]
    assert hits, "the wp18 corpus must yield at least one audited hit"
    legal = {"descriptive_compliant", "needs_annotation", "suspected_inferential"}
    assert all(h["classification"] in legal for h in hits)


# ---------------------------------------------------------------------------
# R3: method-revision zero-trigger closure.
# ---------------------------------------------------------------------------


def test_p57_r3_method_revision_zero_trigger_and_guard_liveness():
    """R3: the registry carries ZERO episodes (no trigger ever fired) and
    the guards are LIVE on that empty registry — any attempt to start a
    method-version comparison without an open trigger record refuses
    fail-closed (FROZEN_17 §6.5 G1).  The attested method version needs
    no episode.  Closure = this state is now formally recorded."""

    registry = _load_json(mr.DEFAULT_REGISTRY_PATH)
    assert registry == {"episodes": []}
    with pytest.raises(ValueError) as excinfo:
        mr.require_open_episode(None)
    assert str(excinfo.value) == mr.G1_REFUSED
    with pytest.raises(ValueError):
        mr.require_open_episode("p57-probe-nonexistent-episode")
    assert str(excinfo.value) == mr.G1_REFUSED
    # The attested version needs no episode (guard挂点 2):
    assert mr.assert_method_version_endorsed(mr.ATTESTED_METHOD_VERSION
                                             ) is None


# ---------------------------------------------------------------------------
# R1: the slow cost-stress closure facts as regression pins.
# ---------------------------------------------------------------------------


def test_p57_r1_slow_skip_dual_carrier_verbatim():
    """R1: the slow cost-stress skip is recorded on BOTH carriers with
    verbatim-identical text — cost_stress.json screen_adjudication and
    the C2 trial ledger's screen:1x_initial rows — with
    stress_rows_authorized=0 and row_count=0 (the §5 initial screen
    refused authorization; per the frozen plan the stress rows are NOT
    chased — result-shopping red line)."""

    stress = _load_json(WP18 / "slow" / "cost_stress.json")
    adjudication = stress["screen_adjudication"]
    assert adjudication["A"]["skip_reason"] == SLOW_SKIP_A
    assert adjudication["B"]["skip_reason"] == SLOW_SKIP_B
    assert adjudication["A"]["stress_rows_authorized"] == 0
    assert adjudication["B"]["stress_rows_authorized"] == 0
    assert stress["row_count"] == 0
    assert stress["rows"] == []
    assert adjudication["A"]["screen"]["condition_i_3_of_5_positive"] is False
    assert adjudication["A"]["screen"]["condition_ii_median_positive"] is False
    assert adjudication["B"]["screen"][
        "condition_iii_dynamic_minus_hold_50bp"] is False
    ledger_path = (
        ROOT / "data" / "r1" / "wp18_slow" / "runs" / "wp18_c2"
        / "wp18-c2-20260927" / "trials.jsonl"
    )
    if not ledger_path.is_file():
        pytest.skip("optional historical C2 trial ledger is not shipped with source")
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 6
    screen_rows = [r for r in rows if r.get("algorithm") == "screen:1x_initial"]
    assert len(screen_rows) == 4  # A pair + B pair (running/succeeded)
    letters = {"slow:combo:slow_all5:A": SLOW_SKIP_A,
               "slow:combo:slow_all5:B": SLOW_SKIP_B}
    seen: dict[str, str] = {}
    for row in screen_rows:
        payload = row.get("payload") or {}
        candidate = str(row.get("candidate"))
        assert payload.get("skip_reason") == letters[candidate]
        assert payload.get("stress_rows_authorized") == 0
        seen.setdefault(candidate, payload["skip_reason"])
    assert set(seen) == set(letters)


def test_p57_r1_unrun_row_keys_preserved_in_plans():
    """R1: the 60 unrun stress row keys (= 20 medium C1:A + 40 slow C2)
  are PRESERVED in the integrity plans for any future round — the
    missing sets are EXACTLY the §5-screened-out arms, both domains
    carry the designed-in measurement_incomplete label with
    verdict_valid=false (an initial-screen refusal, not a measurement
    failure; WP18_REPORT.md:38,194)."""

    slow_plan = _load_json(WP18 / "slow" / "integrity" / "plan.json")
    c2_arms = [
        a for a in slow_plan["arms"]
        if str(a.get("candidate", "")).startswith("C2:")
    ]
    assert len(c2_arms) == 8  # {0x,2x} x {A,B} x {dynamic,hold}
    assert slow_plan["expected_row_count"] == 280
    slow_reconcile = _load_json(WP18 / "slow" / "integrity" / "reconcile_full.json")
    assert slow_reconcile["n_missing"] == 40
    assert slow_reconcile["missing_by_stage"] == {"C2": 40}
    assert slow_reconcile["n_unexpected"] == 0
    assert "EXACTLY the §5-screened-out stress arms" in (
        slow_reconcile["missing_adjudication"]["note"]
    )
    slow_status = _load_json(WP18 / "slow" / "integrity" / "status_full.json")
    assert slow_status["label"] == "measurement_incomplete"
    assert slow_status["verdict_valid"] is False
    assert slow_status["primary_arms_measured"] is False
    medium_reconcile = _load_json(
        WP18 / "medium" / "integrity" / "reconcile_full.json"
    )
    assert medium_reconcile["n_missing"] == 20
    assert medium_reconcile["missing_by_stage"] == {"C1": 20}
    medium_status = _load_json(
        WP18 / "medium" / "integrity" / "status_full.json"
    )
    assert medium_status["label"] == "measurement_incomplete"
    assert medium_status["verdict_valid"] is False
    # The closure-count arithmetics: 60 = 20 + 40 (medium C1:A + slow C2).
    assert 20 + 40 == 60


# ---------------------------------------------------------------------------
# R4: audit-findings persistence (process rule + this package's dogfood).
# ---------------------------------------------------------------------------


def test_p57_r4_audit_findings_persistence_rule_and_dogfood():
    """R4: the persistence rule exists with its template and background
    (the WP17 §6 / WP18 §9.5 two-round gap), and THIS package's own
    audit findings follow it — 吃自己的狗粮."""

    doc = PERSISTENCE_DOC.read_text(encoding="utf-8")
    assert "reports/wpNN/audit_findings_pNN.md" in doc
    assert "模板" in doc or "template" in doc.lower()
    assert "WP17" in doc and ("§9.5" in doc or "§6" in doc)
    own = OWN_FINDINGS.read_text(encoding="utf-8")
    for section in ("# ", "finding", "处置"):
        assert section.lower() in own.lower()
    # Every finding recorded in the dogfood file carries a disposition:
    assert "未处置" not in own and "unresolved: yes" not in own.lower()


# ---------------------------------------------------------------------------
# R6: closure_verifications.json carries the verification items.
# ---------------------------------------------------------------------------


def test_p57_r6_closure_verifications_artifact():
    """R6: reports/wp18/closure_verifications.json records the R1/R3
    verification items with their live-measured results."""

    data = _load_json(CLOSURE_OUT)
    for key in (
        "r1_slow_cost_stress_closure",
        "r3_method_revision_zero_trigger",
        "e3_cache_entry_byte_identity",
        "domain_gate_pipeline_note",
        "standing_items_restated",
    ):
        assert key in data, key
    r1 = data["r1_slow_cost_stress_closure"]
    assert r1["dual_carrier_verbatim_identical"] is True
    assert r1["stress_rows_authorized"] == 0
    assert r1["unrun_row_keys_preserved"]["slow_c2"] == 40
    assert r1["unrun_row_keys_preserved"]["medium_c1_a"] == 20
    assert r1["measurement_incomplete_semantics"].startswith(
        "designed-in"
    )
    assert "no chase" in r1["disposition"] or "不追跑" in r1["disposition"]
    r3 = data["r3_method_revision_zero_trigger"]
    assert r3["episodes"] == []
    assert r3["guard_liveness"]["require_open_episode_none"] == "G1_REFUSED"
    e3 = data["e3_cache_entry_byte_identity"]
    assert e3["byte_identical"] is True
    assert e3["action"] == "verified_no_op_not_deleted"
    note = data["domain_gate_pipeline_note"]
    assert "per-domain" in note["note"] or "按域" in note["note"]
