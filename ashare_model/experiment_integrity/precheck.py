"""WP14 (p50) R2: the pre-load precheck of a materialized run plan.

Contract: ``docs/p50_experiment_integrity_draft.md`` §3.2 (the contract
file lands with the WP14 contract-writing stage per DESIGN §1; every
check below is pinned to a DESIGN §3.2 bullet and, where it re-creates
a historical failure, to the measured fact that failure explains).

Three check classes — one per R2 clause — run on the plan plus the two
code authorities only (the feature vocabulary and the research-domain
registry, both imported lazily inside functions), so the whole precheck
runs BEFORE any large data is loaded:

* P-DOM — 研究域成员资格: every signal-arm member must belong to the
  run's research domain (``member_outside_domain``).  On the real slow
  plan this is the load-time equivalent of the 30 measured KeyError:
  'MOMENTUM_20' failed rows (DESIGN §0 F4): MOMENTUM_20 / RSQ_60 /
  REVERSAL_60 are medium_cross_section features, not slow_fundamental
  AB ones, so the ``directions`` dict built from the config's baseline
  signals (r01_baseline.py:343-349) can never contain them and the
  combo path dies at ``directions[name]`` (r01_baseline.py:380-391).
* P-DEP — 依赖与方向定义: members exist in the vocabulary
  (``member_not_in_vocab``); members have direction evidence in this
  run, i.e. are declared as a single arm — singles are the only arms
  whose directions are learned, combos reuse member directions
  (``member_missing_direction_evidence``); the direction rule is a
  known authority rule id at the runtime authority version
  (``unknown_direction_rule``; a signal arm carrying no rule at all is
  ``missing_direction_rule`` — R2's "是否具备方向定义" read literally);
  and the fold structure is legal — non-empty, ISO dates,
  train_end < test_end, strictly ascending by test_end
  (``fold_window_invalid``) and pairwise disjoint test windows
  (``fold_overlap``).
* P-REF — 组合引用完整性: combo members exist in the vocabulary
  (``combo_dangling_member``); combo members are declared as singles in
  the plan (``combo_member_not_in_plan_singles``); combo members are
  not repeated (``combo_duplicate_member``); every arm carries a
  non-empty cost ladder and the benchmark ladder is exactly ("ref",)
  (``cost_ladder_invalid``); the arm list is non-empty
  (``empty_arm_list``); and the arm SHAPE is vocabulary-clean — every
  arm's kind is inside ``types.ARM_KINDS`` (``unknown_arm_kind``) and
  no benchmark/control arm carries members
  (``comparison_arm_with_members``).  The two shape codes close the
  hand-edited-plan bypass measured at review time: a typo kind
  ('comboo') or a ghost member hidden on a comparison arm used to
  make the issue list SHRINK (15 -> 6, zero issues naming the arm)
  because every member-level lens keys on the exact kind string.

Issue order is deterministic (documented per class below) so two runs
on the same plan return byte-identical reports.

What this module deliberately does not do: no IO of any kind (no file
reads, no loader/engine imports — the two authorities are pure code
registries); it does not raise on constructible problems and never
short-circuits — one call runs every class and returns the complete
issue list (rejecting the plan is the caller's decision on
``PrecheckReport.ok``, not an exception's); it does not re-run, patch
or re-adjudicate any historical artifact; and it does not invent
domain or vocabulary membership — an unresolvable research domain is
reported as ``domain_unresolvable`` and skips only the P-DOM member
loop (P-DEP/P-REF still run), never silently treated as an empty
domain that would flag every member.
"""

from __future__ import annotations

from typing import Sequence

from .plan import REF_COST_LEVELS
from .types import ARM_KINDS, PrecheckIssue, PrecheckReport, RunPlan

#: p50 §3.2 — the three R2 check classes, in execution order.  Every
#: class runs on every call; ``PrecheckReport.checks_run`` records
#: exactly these names.
CHECK_CLASSES: tuple[str, ...] = ("P-DOM", "P-DEP", "P-REF")

#: p50 §3.2 P-DOM codes (研究域成员资格).
CODE_MEMBER_OUTSIDE_DOMAIN = "member_outside_domain"
CODE_DOMAIN_UNRESOLVABLE = "domain_unresolvable"

#: p50 §3.2 P-DEP codes (依赖与方向定义).
CODE_MEMBER_NOT_IN_VOCAB = "member_not_in_vocab"
CODE_MEMBER_MISSING_DIRECTION_EVIDENCE = "member_missing_direction_evidence"
CODE_UNKNOWN_DIRECTION_RULE = "unknown_direction_rule"
#: R2 literal: a signal arm without any direction rule lacks a
#: direction definition outright (benchmark/control arms legitimately
#: carry None — r01_baseline.py:819-821 skips them by kind).
CODE_MISSING_DIRECTION_RULE = "missing_direction_rule"
CODE_FOLD_WINDOW_INVALID = "fold_window_invalid"
CODE_FOLD_OVERLAP = "fold_overlap"

#: p50 §3.2 P-REF codes (组合引用完整性).
CODE_COMBO_DANGLING_MEMBER = "combo_dangling_member"
CODE_COMBO_MEMBER_NOT_IN_PLAN_SINGLES = "combo_member_not_in_plan_singles"
CODE_COMBO_DUPLICATE_MEMBER = "combo_duplicate_member"
CODE_COST_LADDER_INVALID = "cost_ladder_invalid"
CODE_EMPTY_ARM_LIST = "empty_arm_list"
#: p50 §3.2 P-REF arm-shape codes (fail-closed on the kind vocabulary):
#: an arm kind outside ``types.ARM_KINDS`` silently excuses the arm
#: from EVERY member-level lens (they all test the exact kind string),
#: so it is flagged locatably instead — the precheck's second line of
#: defence behind plan.py's ``_arm_from_json`` refusal, covering plans
#: constructed directly in Python with no JSON round-trip.
CODE_UNKNOWN_ARM_KIND = "unknown_arm_kind"
#: A benchmark/control arm carrying members hides signal members in
#: the one arm class no member-level lens ever reads (comparison arms
#: are outside SIGNAL_ARM_KINDS by design) — flagged locatably with
#: the hidden members named in the detail.
CODE_COMPARISON_ARM_WITH_MEMBERS = "comparison_arm_with_members"

#: The arm kinds whose members are direction-bearing signal arms
#: (types.ARM_KINDS minus the comparison kinds; R1/§3.2).
SIGNAL_ARM_KINDS: tuple[str, ...] = ("single", "combo")

#: The comparison kinds, DERIVED from the single ARM_KINDS authority
#: (never a second hand-maintained literal that could drift).
COMPARISON_ARM_KINDS: tuple[str, ...] = tuple(
    kind for kind in ARM_KINDS if kind not in SIGNAL_ARM_KINDS
)

#: The comparison-arm kind whose ladder is frozen at the zero-cost
#: reference level ("ref", p34 §3.3 — plan.REF_COST_LEVELS, the single
#: authority, is compared directly so no second literal can drift).
BENCHMARK_KIND = "benchmark"


def _locator(plan: RunPlan, arm: str | None, member: str | None) -> str:
    """The (experiment, arm, member) position string of types §2."""

    return (
        f"(experiment={plan.experiment_id}, "
        f"arm={arm if arm is not None else 'none'}, "
        f"member={member if member is not None else 'none'})"
    )


def _issue(
    plan: RunPlan,
    code: str,
    detail: str,
    *,
    arm: str | None = None,
    member: str | None = None,
) -> PrecheckIssue:
    return PrecheckIssue(
        code=code,
        experiment_id=plan.experiment_id,
        arm=arm,
        member=member,
        locator=_locator(plan, arm, member),
        detail=detail,
    )


def _default_vocab() -> tuple[str, ...]:
    """The full-table feature-name authority (F16: vocab.FEATURE_NAMES)."""

    from ashare_model.vocab import FEATURE_NAMES

    return tuple(FEATURE_NAMES)


def _resolve_domain_features(
    plan: RunPlan,
) -> tuple[list[str] | None, PrecheckIssue | None]:
    """The run's preregistered feature set (F16:
    research_domain.tier_restricted_features).  An unresolvable domain
    is a locatable FINDING, not a crash — the issue travels with the
    report and only the P-DOM member loop is skipped (an empty set
    would smear ``member_outside_domain`` over every legal member)."""

    from ashare_model.research_domain import tier_restricted_features

    try:
        features = tier_restricted_features(plan.research_domain, plan.tier_set)
    except (KeyError, ValueError) as exc:
        return None, _issue(
            plan,
            CODE_DOMAIN_UNRESOLVABLE,
            "the plan's research domain cannot be resolved by the "
            "registry authority, so member eligibility is not decidable "
            "— fix the plan before any data load\n"
            f"registry refusal: {exc}",
        )
    return list(features), None


def _known_direction_rules() -> tuple[frozenset[str], int]:
    """The direction-rule authority (direction.py:38-41): the legal rule
    ids and the runtime DIRECTION_RULE_VERSION a future load would run."""

    from ashare_model import direction

    return (
        frozenset({direction.RULE_REFERENCE_RETURN, direction.RULE_LABEL_IC}),
        int(direction.DIRECTION_RULE_VERSION),
    )


def _is_iso_date(value: object) -> bool:
    """YYYY-MM-DD shape guard so downstream string comparisons (train
    before test, ascending order) never compare garbage silently."""

    return (
        isinstance(value, str)
        and len(value) == 10
        and value[4] == "-"
        and value[7] == "-"
        and value[0:4].isdigit()
        and value[5:7].isdigit()
        and value[8:10].isdigit()
    )


def _signal_arms(plan: RunPlan):
    return [arm for arm in plan.arms if arm.kind in SIGNAL_ARM_KINDS]


def _check_arm_shape(plan: RunPlan) -> list[PrecheckIssue]:
    """P-REF arm-shape guard (§3.2): every arm's kind is inside the
    ARM_KINDS vocabulary and no comparison arm (benchmark/control)
    carries members.  Issue order: plan arm order.  This is the
    precheck-side companion of plan.py's ``_arm_from_json`` refusal —
    it covers plans constructed directly in Python, and it guarantees
    the arm can never VANISH from the member-level lenses silently:
    the bypass is converted into a locatable issue naming the arm."""

    issues: list[PrecheckIssue] = []
    for arm in plan.arms:
        if arm.kind not in ARM_KINDS:
            issues.append(
                _issue(
                    plan,
                    CODE_UNKNOWN_ARM_KIND,
                    "arm kind is outside the ARM_KINDS vocabulary — "
                    "every member-level check keys on the exact kind "
                    "string, so an unrecognized kind silently bypasses "
                    "all three check classes (fail-visible here, "
                    "refused outright on the plan_from_json path)\n"
                    f"kind={arm.kind!r}, expected one of {ARM_KINDS}",
                    arm=arm.candidate,
                )
            )
            continue
        if arm.kind in COMPARISON_ARM_KINDS and arm.members:
            issues.append(
                _issue(
                    plan,
                    CODE_COMPARISON_ARM_WITH_MEMBERS,
                    "comparison arm carries a non-empty members tuple — "
                    "comparison arms sit outside every member-level "
                    "lens by design, so members hidden here would be "
                    "invisible to the domain/vocabulary/direction "
                    "checks; a comparison arm is memberless by "
                    "construction\n"
                    f"kind={arm.kind!r}, members={list(arm.members)}",
                    arm=arm.candidate,
                )
            )
    return issues


def _single_member_union(plan: RunPlan) -> frozenset[str]:
    """The direction-evidence set: singles are the only arms whose
    directions are learned in a run (r01_baseline.py:343-349 builds
    ``directions`` from the singles path; r03 learns R03_MEMBERS and
    lists those same members as its reference-cell singles), so a
    member absent from every single arm has no direction evidence and
    the combo path would die at ``directions[name]``
    (r01_baseline.py:380-391)."""

    return frozenset(
        member
        for arm in plan.arms
        if arm.kind == "single"
        for member in arm.members
    )


def _check_domain_membership(
    plan: RunPlan,
    domain_features: Sequence[str] | None,
) -> list[PrecheckIssue]:
    """P-DOM (DESIGN §3.2 bullet 1): every signal-arm member belongs to
    the run's research domain.  Issue order: plan arm order, then
    declared member order."""

    issues: list[PrecheckIssue] = []
    if domain_features is not None:
        allowed = set(domain_features)
    else:
        features, domain_issue = _resolve_domain_features(plan)
        if domain_issue is not None:
            return [domain_issue]
        allowed = set(features or ())
    for arm in _signal_arms(plan):
        for member in arm.members:
            if member not in allowed:
                issues.append(
                    _issue(
                        plan,
                        CODE_MEMBER_OUTSIDE_DOMAIN,
                        "member is not in the run's research domain — the "
                        "load-time equivalent of the r01 combo KeyError "
                        "(r01_baseline.py:380-391): the directions dict "
                        "only ever holds in-domain baseline signals\n"
                        f"member={member!r}, research_domain="
                        f"{plan.research_domain!r}, tier_set="
                        f"{plan.tier_set!r}, n_domain_features="
                        f"{len(allowed)}",
                        arm=arm.candidate,
                        member=member,
                    )
                )
    return issues


def _check_folds(plan: RunPlan) -> list[PrecheckIssue]:
    """P-DEP fold structure (§3.2 bullet 2, tail): non-empty, ISO dates,
    train_end < test_end, strictly ascending by test_end, and pairwise
    disjoint test windows (train_end, test_end].  Malformed entries are
    reported and skipped — the pairwise pass never compares garbage."""

    issues: list[PrecheckIssue] = []
    folds = plan.folds
    if not folds:
        return [
            _issue(
                plan,
                CODE_FOLD_WINDOW_INVALID,
                "the plan declares no folds — nothing can be measured "
                "and no fold window exists to validate",
            )
        ]
    valid: list[tuple[int, str, str]] = []  # (index, train_end, test_end)
    for index, fold in enumerate(folds):
        if not (_is_iso_date(fold.train_end) and _is_iso_date(fold.test_end)):
            issues.append(
                _issue(
                    plan,
                    CODE_FOLD_WINDOW_INVALID,
                    "fold dates must be YYYY-MM-DD strings\n"
                    f"locator=folds[{index}], train_end={fold.train_end!r}, "
                    f"test_end={fold.test_end!r}",
                )
            )
            continue
        if not fold.train_end < fold.test_end:
            issues.append(
                _issue(
                    plan,
                    CODE_FOLD_WINDOW_INVALID,
                    "fold train_end must strictly precede test_end\n"
                    f"locator=folds[{index}], train_end={fold.train_end!r}, "
                    f"test_end={fold.test_end!r}",
                )
            )
            continue
        valid.append((index, fold.train_end, fold.test_end))
    for (i, _, test_i), (j, _, test_j) in zip(valid, valid[1:]):
        if not test_i < test_j:
            issues.append(
                _issue(
                    plan,
                    CODE_FOLD_WINDOW_INVALID,
                    "folds must be strictly ascending by test_end\n"
                    f"locator=folds[{i}]->folds[{j}], test_end[{i}]="
                    f"{test_i!r}, test_end[{j}]={test_j!r}",
                )
            )
    for pos, (i, train_i, test_i) in enumerate(valid):
        for j, train_j, test_j in valid[pos + 1 :]:
            # Windows (train, test] intersect iff each opens before the
            # other closes.  Adjacent folds (train_j == test_i) are the
            # legal real-world shape and never flagged.
            if train_i < test_j and train_j < test_i:
                issues.append(
                    _issue(
                        plan,
                        CODE_FOLD_OVERLAP,
                        "test windows overlap — two folds measure the same "
                        "dates, so the same economic object would be "
                        "counted twice\n"
                        f"locator=folds[{i}]x folds[{j}], window[{i}]=("
                        f"{train_i!r}, {test_i!r}], window[{j}]=({train_j!r}, "
                        f"{test_j!r}]",
                    )
                )
    return issues


def _check_member_vocab(
    plan: RunPlan, vocab_set: frozenset[str]
) -> list[PrecheckIssue]:
    """P-DEP (§3.2 bullet 2): every signal-arm member exists in the
    vocabulary — a name the registry never heard of has neither data
    nor a direction.  Issue order: arm order, member order."""

    return [
        _issue(
            plan,
            CODE_MEMBER_NOT_IN_VOCAB,
            "member is not in the feature vocabulary — no data column "
            "and no direction can exist for it (r01_baseline validates "
            "singles against this same authority)\n"
            f"member={member!r}",
            arm=arm.candidate,
            member=member,
        )
        for arm in _signal_arms(plan)
        for member in arm.members
        if member not in vocab_set
    ]


def _check_direction_evidence(
    plan: RunPlan, evidence: frozenset[str]
) -> list[PrecheckIssue]:
    """P-DEP (§3.2 bullet 2): every signal-arm member has direction
    evidence in this run.  Singles carry their own evidence by
    construction (their members build the evidence set), so this fires
    exactly for combo members the run never learned a direction for —
    the plan-side image of ``directions[name]`` raising KeyError
    (r01_baseline.py:380-391).  Issue order: arm order, member order."""

    return [
        _issue(
            plan,
            CODE_MEMBER_MISSING_DIRECTION_EVIDENCE,
            "member has no direction evidence in this run: it is not "
            "declared as any single arm, so the learned directions dict "
            "cannot contain it — the load-time equivalent of KeyError: "
            f"{member!r} on the combo path (r01_baseline.py:380-391)\n"
            f"declared single members: {sorted(evidence)}",
            arm=arm.candidate,
            member=member,
        )
        for arm in _signal_arms(plan)
        for member in arm.members
        if member not in evidence
    ]


def _check_direction_rules(plan: RunPlan) -> list[PrecheckIssue]:
    """P-DEP (§3.2 bullet 2): every signal arm states a direction rule,
    the rule id is one of the authority's, and its version matches the
    runtime authority version a future load would execute.  Issue
    order: arm order."""

    known_rules, runtime_version = _known_direction_rules()
    issues: list[PrecheckIssue] = []
    for arm in _signal_arms(plan):
        rule = arm.direction_rule
        if rule is None:
            issues.append(
                _issue(
                    plan,
                    CODE_MISSING_DIRECTION_RULE,
                    "signal arm carries no direction rule — R1 requires "
                    "the rule to be explicit and R2 requires it to exist "
                    "before any data load",
                    arm=arm.candidate,
                )
            )
            continue
        if rule.rule not in known_rules:
            issues.append(
                _issue(
                    plan,
                    CODE_UNKNOWN_DIRECTION_RULE,
                    "direction rule id is not in the authority enum "
                    "(direction.py:40-41)\n"
                    f"rule={rule.rule!r}, known_rules="
                    f"{sorted(known_rules)}",
                    arm=arm.candidate,
                )
            )
            continue
        if rule.rule_version != runtime_version:
            issues.append(
                _issue(
                    plan,
                    CODE_UNKNOWN_DIRECTION_RULE,
                    "direction rule version does not match the runtime "
                    "authority — the plan would load data under a rule "
                    "version the current code does not implement "
                    "(direction.py:38)\n"
                    f"rule={rule.rule!r}, plan rule_version="
                    f"{rule.rule_version!r}, runtime "
                    f"DIRECTION_RULE_VERSION={runtime_version!r}",
                    arm=arm.candidate,
                )
            )
    return issues


def _check_combo_references(
    plan: RunPlan,
    vocab_set: frozenset[str],
    evidence: frozenset[str],
) -> list[PrecheckIssue]:
    """P-REF (§3.2 bullet 3): combo reference integrity plus the cost
    ladders.  Issue order: empty-arm-list first, then per combo arm (in
    plan order) dangling/not-in-singles per member (declared order),
    then duplicate members, then cost ladders per arm in plan order."""

    issues: list[PrecheckIssue] = []
    if not plan.arms:
        issues.append(
            _issue(
                plan,
                CODE_EMPTY_ARM_LIST,
                "the plan declares no arms — there is nothing to "
                "precheck and nothing a run could measure",
            )
        )
        return issues
    for arm in plan.arms:
        if arm.kind != "combo":
            continue
        for member in arm.members:
            if member not in vocab_set:
                issues.append(
                    _issue(
                        plan,
                        CODE_COMBO_DANGLING_MEMBER,
                        "combo references a member that is not in the "
                        "vocabulary — a dangling reference no data load "
                        "can satisfy\n"
                        f"member={member!r}",
                        arm=arm.candidate,
                        member=member,
                    )
                )
            if member not in evidence:
                issues.append(
                    _issue(
                        plan,
                        CODE_COMBO_MEMBER_NOT_IN_PLAN_SINGLES,
                        "combo references a member no single arm declares "
                        "— the plan itself is internally dangling, "
                        "independent of any registry\n"
                        f"member={member!r}, declared single members: "
                        f"{sorted(evidence)}",
                        arm=arm.candidate,
                        member=member,
                    )
                )
        seen: set[str] = set()
        for member in arm.members:
            if member in seen:
                issues.append(
                    _issue(
                        plan,
                        CODE_COMBO_DUPLICATE_MEMBER,
                        "combo repeats a member — the member would be "
                        "double-counted in the combination weight vector\n"
                        f"member={member!r}, members={list(arm.members)}",
                        arm=arm.candidate,
                        member=member,
                    )
                )
            seen.add(member)
    for arm in plan.arms:
        if not arm.cost_levels:
            issues.append(
                _issue(
                    plan,
                    CODE_COST_LADDER_INVALID,
                    "arm carries an empty cost ladder — the measurement "
                    "grid for this arm is empty by construction",
                    arm=arm.candidate,
                )
            )
        elif arm.kind == BENCHMARK_KIND and tuple(arm.cost_levels) != REF_COST_LEVELS:
            issues.append(
                _issue(
                    plan,
                    CODE_COST_LADDER_INVALID,
                    "benchmark arm ladder must be exactly the frozen "
                    "zero-cost reference level\n"
                    f"cost_levels={tuple(arm.cost_levels)!r}, expected="
                    f"{REF_COST_LEVELS!r}",
                    arm=arm.candidate,
                )
            )
    return issues


def run_precheck(
    plan: RunPlan,
    *,
    vocab: Sequence[str] | None = None,
    domain_features: Sequence[str] | None = None,
) -> PrecheckReport:
    """Run the three R2 check classes on one materialized plan.

    Pure function: zero IO, zero loader/engine imports; the vocabulary
    and the research-domain registry are the only authorities and are
    read lazily (``ashare_model.vocab.FEATURE_NAMES`` and
    ``research_domain.tier_restricted_features``) unless the caller
    injects ``vocab`` / ``domain_features`` explicitly (tests and
    hypothetical-registry what-ifs never monkeypatch).  Never raises
    on constructible problems and never short-circuits: all three
    classes run and the complete, deterministically ordered issue list
    comes back — rejection is ``report.ok is False``, decided by the
    caller (the CLI exits 1 on it; a future data load refuses on it).
    """

    vocab_set = frozenset(vocab if vocab is not None else _default_vocab())
    evidence = _single_member_union(plan)

    issues: list[PrecheckIssue] = []
    issues.extend(_check_arm_shape(plan))
    issues.extend(_check_domain_membership(plan, domain_features))
    issues.extend(_check_folds(plan))
    issues.extend(_check_member_vocab(plan, vocab_set))
    issues.extend(_check_direction_evidence(plan, evidence))
    issues.extend(_check_direction_rules(plan))
    issues.extend(_check_combo_references(plan, vocab_set, evidence))
    return PrecheckReport(issues=tuple(issues), checks_run=CHECK_CLASSES)
