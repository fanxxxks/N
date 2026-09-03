"""P8-01 lifecycle contract structure guard (docs/p8_research_lifecycle_contract.md).

The lifecycle contract is the pre-registered arbitration source for the P8
research-lifecycle program: statuses, legal transitions, evidence matrix,
identity hierarchy and the legacy/paper-account policy. The document is
hand-maintained, so unlike the generated registry docs this guard asserts
structure, not bytes: the canonical status identifiers, the legal transition
edges, the identity names, the required adjudication sections and the
pending-activation block in AGENTS.md must stay present. Wording may evolve,
but only through a contract revision that updates this guard in the same
commit.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "p8_research_lifecycle_contract.md"
AGENTS_PATH = ROOT / "AGENTS.md"

# Canonical lifecycle status identifiers (contract §3). P8-06+ implement
# these verbatim; the contract must enumerate exactly this set.
STATUSES = (
    "IDEA",
    "SPEC_LOCKED",
    "DATA_QUALIFIED",
    "FACTOR_SET_QUALIFIED",
    "SEARCH_PLAN_ADMITTED",
    "OOS_QUALIFIED",
    "PAPER_OBSERVING",
    "PROMOTED",
    "REJECTED",
    "FAILED",
    "RETIRED",
)

# Canonical legal transition edges (contract §4, machine-readable block).
# Anything not enumerated is illegal and must fail closed.
TRANSITIONS = (
    "IDEA -> SPEC_LOCKED",
    "IDEA -> FAILED | RETIRED",
    "SPEC_LOCKED -> DATA_QUALIFIED",
    "SPEC_LOCKED -> REJECTED | FAILED | RETIRED",
    "DATA_QUALIFIED -> FACTOR_SET_QUALIFIED",
    "DATA_QUALIFIED -> REJECTED | FAILED | RETIRED",
    "FACTOR_SET_QUALIFIED -> SEARCH_PLAN_ADMITTED",
    "FACTOR_SET_QUALIFIED -> REJECTED | FAILED | RETIRED",
    "SEARCH_PLAN_ADMITTED -> OOS_QUALIFIED",
    "SEARCH_PLAN_ADMITTED -> REJECTED | FAILED | RETIRED",
    "OOS_QUALIFIED -> PAPER_OBSERVING",
    "OOS_QUALIFIED -> REJECTED | FAILED | RETIRED",
    "PAPER_OBSERVING -> PROMOTED",
    "PAPER_OBSERVING -> REJECTED | FAILED | RETIRED",
    "PROMOTED -> RETIRED",
)

# Identity names the contract must define (contract §2).
IDENTITIES = ("spec_id", "run_id", "candidate_id", "artifact_id", "account_id")

# Section headers fixing the adjudication layout (contract TOC).
SECTIONS = (
    "## 2. 身份层次",
    "## 3. 生命周期状态与机器证据矩阵",
    "## 4. 合法转换与终态判别",
    "## 5. 资格门禁的精确机器标准",
    "## 6. 受保护 OOS 与 PR 开发反馈隔离",
    "## 7. Paper 观察数据单调扩展",
    "## 8. 旧产物与 paper account 策略",
    "## 9. 激活边界与 formal run 行为",
)


def _contract_text() -> str:
    assert DOC_PATH.exists(), (
        "docs/p8_research_lifecycle_contract.md is missing: the lifecycle "
        "contract must exist before any P8 implementation stage lands"
    )
    return DOC_PATH.read_text(encoding="utf-8")


def test_contract_doc_exists() -> None:
    _contract_text()


def test_canonical_status_set_is_enumerated() -> None:
    text = _contract_text()
    missing = [status for status in STATUSES if status not in text]
    assert not missing, f"contract is missing canonical statuses: {missing}"


def test_legal_transition_edges_are_enumerated() -> None:
    normalized = " ".join(_contract_text().split())
    missing = [edge for edge in TRANSITIONS if edge not in normalized]
    assert not missing, f"contract is missing legal transition edges: {missing}"


def test_identity_definitions_are_present() -> None:
    text = _contract_text()
    missing = [name for name in IDENTITIES if name not in text]
    assert not missing, f"contract is missing identity definitions: {missing}"


def test_append_only_replay_is_the_only_status_source() -> None:
    text = _contract_text()
    assert "append-only" in text, "status must be derived from append-only events"
    assert "重放" in text, "status must be computed by event replay, not stored"
    assert "LIFECYCLE_CONTRACT_VERSION = 1" in text, (
        "the lifecycle contract version constant must be declared"
    )


def test_required_adjudication_sections_are_present() -> None:
    text = _contract_text()
    missing = [header for header in SECTIONS if header not in text]
    assert not missing, f"contract is missing adjudication sections: {missing}"


def test_agents_md_carries_pending_activation_block() -> None:
    text = AGENTS_PATH.read_text(encoding="utf-8")
    assert "p8_research_lifecycle_contract.md" in text, (
        "AGENTS.md must reference the lifecycle contract document"
    )
    assert "待 lifecycle v1 激活" in text, (
        "lifecycle hard constraints must be marked pending lifecycle v1 "
        "activation until P8-15 flips them to mandatory"
    )


# -- IP-02 contract revision anchors (2026-09-03) ------------------------------
#
# The IP-02 contract revision (approval basis: the user-approved
# AlphaGPT-improvement-plan-20260903.md IP-02 [02-INT-01][02-INT-02]
# [02-INT-07][02-INT-08][05-⑤], staging-table change per
# campaign_closure_decisions_20260902.md ⑤) fixed three stale spots in the
# contract. These guards pin the revised invariants so they cannot drift
# back. They strengthen the guard set; nothing is weakened or removed
# (§10.1 whitelist category 2: requirement/contract changed first, tests
# synced with evidence in the same commit).
#
# * Promotion gates are G1-G7 (``ashare_model/promotion.py``,
#   ``PROMOTION_RULE_VERSION = "2"``; G7 feature registry status added by
#   P12, authority ``docs/p12_promotion_enforcement_contract.md``). The two
#   G numbering spaces (data qualification vs promotion) both span G1-G7
#   but remain independent spaces.
# * ``PROMOTED -> RETIRED`` is activated with the P8-06 row
#   (``ashare_model/lifecycle.py`` ``ACTIVATED_EDGES`` activates every legal
#   "-> RETIRED" edge; PROMOTED is unreachable at P8-06, so this is not
#   fail-open) and must not be listed a second time in the P8-09 row.
# * ``REBALANCE_POLICY_VERSION`` is consumed by ``ashare_model/runspec.py``
#   (RunSpec version collection) and must not be enumerated as consumerless;
#   ``TARGET_CONTRACT_VERSION`` (``ashare_model/targets.py``) still has no
#   consumer and keeps the independent-retirement-task annotation.


def test_promotion_gate_numbering_is_g1_g7() -> None:
    text = _contract_text()
    stale = [mark for mark in ("晋级门禁 G1–G6", "晋级门禁 G1-G6") if mark in text]
    assert not stale, (
        "stale promotion-gate numbering in the contract: promotion.py gates "
        'are G1-G7 (PROMOTION_RULE_VERSION = "2", G7 added by P12)'
    )
    assert "晋级门禁 G1–G7" in text, (
        "the promotion-gate space must be named G1-G7"
    )
    assert "数据资格 G1–G7" in text, (
        "the data-qualification space must stay explicitly distinguished "
        "from the promotion-gate space (two independent G numbering spaces)"
    )
    assert 'PROMOTION_RULE_VERSION = "2"' in text, (
        "the promotion gate-set semantic version must be cited"
    )
    assert "docs/p12_promotion_enforcement_contract.md" in text, (
        "the P12 promotion-enforcement contract must be cited as the G7 "
        "authority"
    )


def test_staged_activation_promoted_retired_is_unambiguous() -> None:
    lines = _contract_text().splitlines()
    p806 = [line for line in lines if line.startswith("| P8-06 ")]
    p809 = [line for line in lines if line.startswith("| P8-09 ")]
    assert len(p806) == 1 and len(p809) == 1, (
        "the staged activation table must have exactly one P8-06 row and "
        "one P8-09 row"
    )
    assert "PROMOTED -> RETIRED" in p806[0], (
        "PROMOTED -> RETIRED must be explicitly covered by the P8-06 row "
        "(ACTIVATED_EDGES activates every legal -> RETIRED edge; "
        "campaign_closure_decisions_20260902.md ⑤)"
    )
    assert "PROMOTED -> RETIRED" not in p809[0], (
        "the P8-09 row must not re-list PROMOTED -> RETIRED: the edge has "
        "exactly one activation stage (P8-06)"
    )


def test_retirement_ledger_matches_runspec_consumers() -> None:
    text = _contract_text()
    s13 = text.split("## 13. Retirement", 1)[1]
    consumerless_claims = s13.split("勘误", 1)[0]
    assert "REBALANCE_POLICY_VERSION" not in consumerless_claims, (
        "REBALANCE_POLICY_VERSION is consumed by ashare_model/runspec.py "
        "(RunSpec version collection) and must not be listed consumerless"
    )
    assert "TARGET_CONTRACT_VERSION" in consumerless_claims, (
        "TARGET_CONTRACT_VERSION (ashare_model/targets.py) still has no "
        "consumer and keeps the independent retirement-task annotation"
    )
    assert "runspec.py" in s13, (
        "the erratum must cite the consumer evidence (ashare_model/"
        "runspec.py)"
    )
