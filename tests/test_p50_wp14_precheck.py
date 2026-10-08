"""P50 WP14 precheck-stage RED/GREEN tests: the R2 load-time checks
(DESIGN §3.2; contract file ``docs/p50_experiment_integrity_draft.md``
lands with the WP14 contract-writing stage per DESIGN §1, so every
clause asserted here is pinned to a DESIGN §3.2 bullet and a measured
fact F4/F16 or the S1 counterexample of DESIGN §6).

Invariants (precheck stage — the DESIGN §9 INV numbering belongs to the
full-suite file; this module uses PC1-PC12 to avoid collision with the
plan stage's PL1-PL6):

* PC1 — S1 wrong-domain member (DESIGN §6 row 1, precheck half): the
  REAL slow plan is rejected BEFORE any data load with locatable
  ``member_outside_domain`` errors on exactly the KeyError root-cause
  members, while the five in-domain single-factor arms stay clean
  (reject the wrong verdict, preserve the usable part);
* PC2 — the three medium-family real plans pass clean: no false
  positives (the P45 lesson — never compare across mismatched
  calibers — starts with not flagging a consistent plan);
* PC3/PC4/PC5 — the three R2 clauses fire surgically: a member can
  violate exactly one of {domain, vocabulary, direction-evidence}
  without tripping the others;
* PC6 — direction definitions: missing rule, unknown rule id, rule
  version mismatch against the runtime authority;
* PC7 — fold structure: empty, malformed date, train>=test,
  non-ascending, and overlap-only (each isolated);
* PC8 — combo reference integrity: dangling member, duplicate member,
  empty ladder, wrong benchmark ladder, empty arm list;
* PC9 — no short-circuit and determinism: all three classes report in
  one pass on a kitchen-sink plan; two runs return identical issues;
* PC10 — an unresolvable research domain is a locatable finding, not a
  crash and not a silently-empty domain;
* PC11 — the authorities are injectable arguments (no monkeypatching);
* PC12 — the precheck module's import surface is load-free (module
  level touches only .plan/.types; every authority import is lazy);
* PC13 — the ARM_KINDS vocabulary is enforced fail-closed: a hand-
  edited typo kind ('comboo') is REFUSED by plan_from_json with a
  locator naming experiment / arm.candidate / kind / ARM_KINDS, and a
  directly-constructed plan with the same typo gets a locatable
  unknown_arm_kind issue — the arm can no longer silently vanish from
  the three member-level check classes (reviewer-reproduced bypass:
  issue count 15 -> 6 with zero issues naming the arm);
* PC14 — a comparison arm (benchmark/control) carrying a non-empty
  members tuple is a locatable comparison_arm_with_members issue
  naming the hidden members (reviewer probe 2: a ghost member on
  benchmark:equal_weight used to produce zero warnings).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ashare_model.experiment_integrity.plan import (
    materialize_plan,
    plan_from_json,
    plan_to_json,
)
from ashare_model.experiment_integrity.precheck import (
    CHECK_CLASSES,
    run_precheck,
)
from ashare_model.experiment_integrity.types import (
    DirectionRule,
    PlanValidationError,
    PlannedArm,
    PlannedFold,
    RunPlan,
)

#: The five folds every real r1 config declares (configs/r1_slow_ab.yaml
#: protocol.folds; asserted by the plan stage's PL2).
_REAL_FOLDS = tuple(
    PlannedFold(train_end=train, test_end=test)
    for train, test in [
        ("2020-12-31", "2021-12-31"),
        ("2021-12-31", "2022-12-31"),
        ("2022-12-31", "2023-12-31"),
        ("2023-12-31", "2024-12-31"),
        ("2024-12-31", "2025-12-31"),
    ]
)

#: The five in-domain slow singles (config baseline_signals).
_SLOW_SINGLES = ("ROE", "PE_TTM", "REVENUE_YOY", "DIVIDEND_YIELD", "MARKET_CAP")
#: The out-of-domain combo members that produced the 30 measured
#: KeyError: 'MOMENTUM_20' failed rows (DESIGN §0 F4).
_COMBO_MEMBERS = ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")


def _rule(rule: str = "reference_return", version: int = 1) -> DirectionRule:
    return DirectionRule(
        rule=rule, rule_version=version, scope="per_fold", learned_at="none"
    )


def _single(member: str, **overrides) -> PlannedArm:
    defaults = dict(
        candidate=f"baseline:{member}",
        kind="single",
        members=(member,),
        direction_rule=_rule(),
        cost_levels=("0x", "1x"),
        is_primary=True,
        members_source="config.protocol.baseline_signals",
    )
    defaults.update(overrides)
    return PlannedArm(**defaults)


def _combo(name: str, members: tuple[str, ...], **overrides) -> PlannedArm:
    defaults = dict(
        candidate=name,
        kind="combo",
        members=members,
        direction_rule=_rule(),
        cost_levels=("0x", "1x"),
        is_primary=True,
        members_source="code.R01_COMBO_ARMS",
    )
    defaults.update(overrides)
    return PlannedArm(**defaults)


def _benchmark(cost_levels: tuple[str, ...] = ("ref",)) -> PlannedArm:
    return PlannedArm(
        candidate="benchmark:equal_weight",
        kind="benchmark",
        members=(),
        direction_rule=None,
        cost_levels=cost_levels,
        is_primary=False,
        members_source="code.r0x(benchmark:equal_weight)",
    )


def _plan(
    *arms: PlannedArm,
    folds: tuple[PlannedFold, ...] | None = None,
    domain: str = "slow_fundamental",
    tier: str | None = "AB",
) -> RunPlan:
    return RunPlan(
        experiment_id="r01_slow_baseline",
        research_domain=domain,
        tier_set=tier,
        folds=folds if folds is not None else _REAL_FOLDS,
        arms=tuple(arms),
        key_axes=("candidate", "cost_level", "fold_test_end"),
        adjudication_rules={},
        verdict_map={},
        sources={},
    )


def _codes(report) -> list[str]:
    return [issue.code for issue in report.issues]


# --- PC1: S1 — wrong-domain member rejected locatably, singles kept ----


def test_p50_pc1_s1_wrong_domain_member_rejects_and_singles_stay_clean():
    # DESIGN §6 S1, precheck half.  The REAL materialized slow plan
    # carries combo members that live in medium_cross_section, not
    # slow_fundamental AB — the exact root cause of the 30 KeyError:
    # 'MOMENTUM_20' failed rows (F4).  RED: the precheck must refuse
    # the plan before any data load; GREEN: the five legal singles and
    # both comparison arms produce no issue of any code.
    plan = materialize_plan("r01_slow_baseline")
    report = run_precheck(plan)

    assert report.ok is False
    assert report.checks_run == CHECK_CLASSES == ("P-DOM", "P-DEP", "P-REF")

    domain_issues = [
        i for i in report.issues if i.code == "member_outside_domain"
    ]
    # 3 members in combo:all3 + 2 in combo:momentum_reversal, plan order.
    assert [(i.arm, i.member) for i in domain_issues] == [
        ("combo:all3", "MOMENTUM_20"),
        ("combo:all3", "RSQ_60"),
        ("combo:all3", "REVERSAL_60"),
        ("combo:momentum_reversal", "MOMENTUM_20"),
        ("combo:momentum_reversal", "REVERSAL_60"),
    ]
    # Locatable failure: the locator names experiment, arm and member.
    assert report.issues[0].locator == (
        "(experiment=r01_slow_baseline, arm=combo:all3, member=MOMENTUM_20)"
    )
    assert report.issues[0].experiment_id == "r01_slow_baseline"
    # The detail points at the runtime failure it prevents.
    assert "KeyError" in report.issues[0].detail

    # The same defect under the two other lenses that see it (§3.2):
    # no direction evidence (P-DEP) and no plan-declared single (P-REF).
    assert _codes(report).count("member_missing_direction_evidence") == 5
    assert _codes(report).count("combo_member_not_in_plan_singles") == 5
    assert len(report.issues) == 15  # 5 x three codes, nothing else

    # GREEN: the in-domain single arms stay clean — the precheck
    # rejects the plan without smearing its legal arms.
    single_arm_names = {f"baseline:{name}" for name in _SLOW_SINGLES}
    assert not [i for i in report.issues if i.arm in single_arm_names]
    assert not [
        i for i in report.issues if i.arm in ("benchmark:equal_weight", "control:cash")
    ]


# --- PC2: the medium family passes clean (no false positives) ----------


@pytest.mark.parametrize(
    "experiment_id", ["r02_matrix", "r02_periods", "r03_combos"]
)
def test_p50_pc2_medium_family_real_plans_pass_precheck(experiment_id):
    # The same members that doom the slow plan ARE medium_cross_section
    # tier-A features (DESIGN §0 P §1), and those plans declare them as
    # singles — a consistent plan must sail through with zero issues.
    plan = materialize_plan(experiment_id)
    report = run_precheck(plan)
    assert report.issues == ()
    assert report.ok is True
    assert report.checks_run == ("P-DOM", "P-DEP", "P-REF")


# --- PC3/PC4/PC5: the three R2 clauses fire surgically ------------------


def test_p50_pc3_domain_violation_alone_is_locatable():
    # In the vocabulary, declared as a single, but outside the injected
    # domain: the domain clause fires exactly once with a locator
    # naming the exact (arm, member); the vocabulary and
    # dangling-reference lenses stay silent (the member exists in the
    # registry and its combo reference resolves).
    plan = _plan(
        _single("ROE"),
        _combo("combo:test", ("MOMENTUM_20",)),
        _benchmark(),
    )
    report = run_precheck(plan, domain_features=("ROE",))
    assert _codes(report).count("member_outside_domain") == 1
    issue = report.issues[0]
    assert (issue.arm, issue.member) == ("combo:test", "MOMENTUM_20")
    assert issue.locator == (
        "(experiment=r01_slow_baseline, arm=combo:test, member=MOMENTUM_20)"
    )
    # Vocabulary and registry-reference lenses stay silent: the member
    # exists and is referenced consistently.
    assert "member_not_in_vocab" not in _codes(report)
    assert "combo_dangling_member" not in _codes(report)


def test_p50_pc4_vocabulary_violation_alone_fires():
    # Owned by the (injected) domain registry, but absent from the real
    # vocabulary authority: ONLY the dependency clause fires.
    plan = _plan(_single("GHOST_NEVER_REGISTERED"), _benchmark())
    report = run_precheck(plan, domain_features=("GHOST_NEVER_REGISTERED",))
    assert _codes(report) == ["member_not_in_vocab"]
    assert report.issues[0].member == "GHOST_NEVER_REGISTERED"


def test_p50_pc5_missing_direction_evidence_alone_fires():
    # In-domain, in-vocabulary, but no single arm declares it: the run
    # can never learn a direction for it — the plan-side image of
    # KeyError: 'MOMENTUM_20'.  Both evidence lenses fire, nothing else.
    plan = _plan(
        _single("ROE"),
        _combo("combo:mixed", ("ROE", "PE_TTM")),
        _benchmark(),
    )
    report = run_precheck(plan)  # default authorities (slow AB)
    assert sorted(_codes(report)) == [
        "combo_member_not_in_plan_singles",
        "member_missing_direction_evidence",
    ]
    evidence = [
        i for i in report.issues
        if i.code == "member_missing_direction_evidence"
    ]
    assert [(i.arm, i.member) for i in evidence] == [
        ("combo:mixed", "PE_TTM")
    ]
    assert "KeyError" in evidence[0].detail
    # ROE itself never appears: it has direction evidence.
    assert not [i for i in report.issues if i.member == "ROE"]


# --- PC6: direction definitions exist and are known ---------------------


def test_p50_pc6_direction_rule_defects_are_separately_locatable():
    from ashare_model import direction

    plan = _plan(
        _single("ROE", direction_rule=None),
        _single("PE_TTM", direction_rule=_rule(rule="crystal_ball")),
        _single("REVENUE_YOY", direction_rule=_rule(version=99)),
        _benchmark(),
    )
    report = run_precheck(
        plan, domain_features=_SLOW_SINGLES, vocab=_SLOW_SINGLES
    )
    assert _codes(report) == [
        "missing_direction_rule",
        "unknown_direction_rule",
        "unknown_direction_rule",
    ]
    missing, bad_id, bad_version = report.issues
    assert missing.arm == "baseline:ROE"
    assert bad_id.arm == "baseline:PE_TTM"
    assert "crystal_ball" in bad_id.detail
    assert bad_version.arm == "baseline:REVENUE_YOY"
    # The version mismatch names both sides (runtime authority = 1).
    assert "plan rule_version=99" in bad_version.detail
    assert (
        f"runtime DIRECTION_RULE_VERSION={direction.DIRECTION_RULE_VERSION}"
        in bad_version.detail
    )
    # A comparison arm with no rule is legal — only signal arms checked.
    assert not [i for i in report.issues if i.arm == "benchmark:equal_weight"]


# --- PC7: fold structure ------------------------------------------------


def test_p50_pc7_empty_folds_refuse():
    report = run_precheck(_plan(_single("ROE"), folds=()))
    assert _codes(report) == ["fold_window_invalid"]
    issue = report.issues[0]
    assert issue.arm is None and issue.member is None


def test_p50_pc7_malformed_date_refuses_naming_the_fold():
    folds = (
        PlannedFold(train_end="2020/12/31", test_end="2021-12-31"),
    ) + _REAL_FOLDS[1:]
    report = run_precheck(_plan(_single("ROE"), folds=folds))
    assert _codes(report) == ["fold_window_invalid"]
    assert "folds[0]" in report.issues[0].detail


def test_p50_pc7_train_not_before_test_refuses():
    folds = (
        PlannedFold(train_end="2022-12-31", test_end="2021-12-31"),
    ) + _REAL_FOLDS[1:]
    report = run_precheck(_plan(_single("ROE"), folds=folds))
    assert _codes(report) == ["fold_window_invalid"]
    assert "train_end must strictly precede test_end" in report.issues[0].detail


def test_p50_pc7_non_ascending_test_ends_refuse():
    folds = (
        PlannedFold("2020-12-31", "2021-12-31"),
        PlannedFold("2023-12-31", "2024-12-31"),
        PlannedFold("2021-12-31", "2022-12-31"),
    )
    report = run_precheck(_plan(_single("ROE"), folds=folds))
    assert _codes(report) == ["fold_window_invalid"]
    assert "strictly ascending" in report.issues[0].detail


def test_p50_pc7_overlapping_windows_alone_fire_fold_overlap():
    # Windows (2020, 2022-06] and (2021, 2023] intersect while every
    # per-fold rule and the ordering pass — ONLY fold_overlap fires.
    folds = (
        PlannedFold("2020-12-31", "2022-06-30"),
        PlannedFold("2021-12-31", "2023-12-31"),
    )
    report = run_precheck(_plan(_single("ROE"), folds=folds))
    assert _codes(report) == ["fold_overlap"]
    assert "folds[0]x folds[1]" in report.issues[0].detail


def test_p50_pc7_adjacent_real_folds_never_flagged():
    # The declared real shape (each train_end == previous test_end) is
    # adjacent, not overlapping — the default folds produce no fold issue.
    report = run_precheck(_plan(_single("ROE"), _benchmark()))
    assert not [c for c in _codes(report) if c.startswith("fold_")]


# --- PC8: combo reference integrity --------------------------------------


def test_p50_pc8_dangling_combo_member_fires_both_reference_lenses():
    plan = _plan(_single("ROE"), _combo("combo:bad", ("GHOST_X",)), _benchmark())
    report = run_precheck(plan, domain_features=("ROE", "GHOST_X"))
    codes = _codes(report)
    assert "member_not_in_vocab" in codes
    assert "combo_dangling_member" in codes
    dangling = [i for i in report.issues if i.code == "combo_dangling_member"]
    assert [(i.arm, i.member) for i in dangling] == [("combo:bad", "GHOST_X")]


def test_p50_pc8_duplicate_combo_member_fires_alone():
    plan = _plan(
        _single("MOMENTUM_20"),
        _combo("combo:dup", ("MOMENTUM_20", "MOMENTUM_20")),
        _benchmark(),
    )
    report = run_precheck(
        plan, domain_features=("MOMENTUM_20",), vocab=("MOMENTUM_20", "ROE")
    )
    assert _codes(report) == ["combo_duplicate_member"]
    issue = report.issues[0]
    assert (issue.arm, issue.member) == ("combo:dup", "MOMENTUM_20")


def test_p50_pc8_cost_ladder_defects_fire_per_arm():
    plan = _plan(
        _single("ROE"),
        _benchmark(cost_levels=("0x",)),  # benchmark must be ("ref",)
    )
    report = run_precheck(plan, domain_features=("ROE",))
    assert _codes(report) == ["cost_ladder_invalid"]
    issue = report.issues[0]
    assert issue.arm == "benchmark:equal_weight"
    assert "('ref',)" in issue.detail

    empty_ladder = _plan(_single("ROE", cost_levels=()))
    report2 = run_precheck(empty_ladder, domain_features=("ROE",))
    codes2 = _codes(report2)
    assert codes2.count("cost_ladder_invalid") == 1
    assert report2.issues[0].arm == "baseline:ROE"


def test_p50_pc8_empty_arm_list_refuses():
    report = run_precheck(_plan())
    assert _codes(report) == ["empty_arm_list"]
    issue = report.issues[0]
    assert issue.arm is None and issue.member is None
    assert issue.locator == (
        "(experiment=r01_slow_baseline, arm=none, member=none)"
    )


# --- PC9: no short-circuit, deterministic ordering -----------------------


def test_p50_pc9_all_three_classes_report_in_one_pass_and_repeat_identically():
    # Kitchen sink: an out-of-domain duplicate combo member (P-DOM +
    # P-DEP evidence + P-REF duplicate/not-in-singles), an overlapping
    # fold pair (P-DEP folds) and a wrong benchmark ladder (P-REF).
    plan = _plan(
        _single("ROE"),
        _combo("combo:bad", ("MOMENTUM_20", "MOMENTUM_20")),
        _benchmark(cost_levels=("0x",)),
        folds=(
            PlannedFold("2020-12-31", "2022-06-30"),
            PlannedFold("2021-12-31", "2023-12-31"),
        ),
    )
    report = run_precheck(plan, domain_features=("ROE",))
    codes = _codes(report)
    # P-DOM reported…
    assert codes.count("member_outside_domain") == 2
    # …despite P-DEP findings…
    assert "fold_overlap" in codes
    assert codes.count("member_missing_direction_evidence") == 2
    # …despite P-REF findings.
    assert codes.count("combo_duplicate_member") == 1
    assert "cost_ladder_invalid" in codes
    assert report.checks_run == ("P-DOM", "P-DEP", "P-REF")
    # Pure function: identical input, byte-identical output.
    assert run_precheck(plan, domain_features=("ROE",)).issues == report.issues


# --- PC10: unresolvable domain is a finding, not a crash -----------------


def test_p50_pc10_unresolvable_domain_is_reported_not_raised():
    from ashare_model.experiment_integrity.plan import (
        plan_from_json,
        plan_to_json,
    )

    payload = plan_to_json(materialize_plan("r01_slow_baseline"))
    payload["research_domain"] = "no_such_domain"
    plan = plan_from_json(payload)

    report = run_precheck(plan)  # must not raise
    assert report.ok is False
    assert report.issues[0].code == "domain_unresolvable"
    # The P-DOM member loop is skipped (no empty-domain smear)…
    assert not [
        i for i in report.issues if i.code == "member_outside_domain"
    ]
    # …but P-DEP/P-REF still ran to completion on the same call.
    assert _codes(report).count("member_missing_direction_evidence") == 5
    assert report.checks_run == ("P-DOM", "P-DEP", "P-REF")


# --- PC11: authorities are injectable arguments --------------------------


def test_p50_pc11_injected_authorities_replace_the_defaults():
    # A hypothetical registry that owns BOTH the slow singles and the
    # combo members silences the domain clause entirely — while the
    # evidence clauses still see the real defect (the slow plan
    # declares no singles for the combo members): each clause reads
    # only its own question, and the injected argument is the authority.
    plan = materialize_plan("r01_slow_baseline")
    hypothetical = list(_SLOW_SINGLES) + list(_COMBO_MEMBERS)
    report = run_precheck(plan, domain_features=hypothetical)
    assert not [
        i for i in report.issues if i.code == "member_outside_domain"
    ]
    assert _codes(report).count("member_missing_direction_evidence") == 5

    # A restricted injected vocabulary flags only the members it lacks.
    tight = _plan(_single("ROE"), _single("PE_TTM"), _benchmark())
    report2 = run_precheck(
        tight, domain_features=("ROE", "PE_TTM"), vocab=("ROE",)
    )
    assert _codes(report2) == ["member_not_in_vocab"]
    assert report2.issues[0].member == "PE_TTM"


# --- PC12: the precheck import surface is load-free ----------------------


def test_p50_pc12_precheck_module_imports_are_lazy_and_engine_free():
    source = (
        Path(__file__).resolve().parents[1]
        / "ashare_model"
        / "experiment_integrity"
        / "precheck.py"
    ).read_text(encoding="utf-8")
    for banned in (
        "data_loader",
        "gates",
        "evaluate_signal",
        "run_r0",
        "ashare_data",
        "open(",
        "Path(",
    ):
        assert banned not in source, f"precheck must stay load-free: {banned}"
    module_level_imports = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    ]
    # Only value types + the frozen ladder constant load at import
    # time; the three authorities (vocab / research_domain / direction)
    # are function-local so R2 runs before any data load.  ARM_KINDS
    # joins the frozen type imports as the kind-vocabulary authority
    # PC13 enforces (a frozen tuple in types.py — zero IO, still
    # load-free).
    assert module_level_imports == [
        "from __future__ import annotations",
        "from typing import Sequence",
        "from .plan import REF_COST_LEVELS",
        "from .types import ARM_KINDS, PrecheckIssue, PrecheckReport, RunPlan",
    ]


# --- PC13: arm-kind vocabulary enforced fail-closed -----------------------


def test_p50_pc13_typo_arm_kind_is_refused_and_can_no_longer_vanish():
    # The reviewer-reproduced bypass, verbatim: with the REAL slow
    # plan's arms[candidate='combo:all3'].kind hand-edited to 'comboo',
    # plan_from_json used to accept the payload silently and the
    # precheck issue count DROPPED from 15 to 6 with ZERO issues
    # naming the arm — every member-level check class keys on the
    # exact kind string (_signal_arms / 'single' / 'combo'), so an
    # out-of-vocabulary kind excused the arm from all three classes
    # and the plan looked CLEANER for being more broken.
    payload = plan_to_json(materialize_plan("r01_slow_baseline"))
    for arm in payload["arms"]:
        if arm["candidate"] == "combo:all3":
            arm["kind"] = "comboo"
    with pytest.raises(PlanValidationError) as excinfo:
        plan_from_json(payload)
    message = str(excinfo.value)
    # Locatable: experiment, arm.candidate, the observed kind, and the
    # expected ARM_KINDS vocabulary are all named.
    assert "comboo" in message
    assert "combo:all3" in message
    assert "r01_slow_baseline" in message
    assert all(kind in message for kind in ("single", "combo", "benchmark", "control"))

    # The non-JSON path — a plan constructed directly in Python with
    # the same typo — is REPORTED with a locatable unknown_arm_kind
    # issue; the surgical companion: nothing else fires on this plan.
    plan = _plan(
        _single("ROE"),
        _combo("combo:all3", _COMBO_MEMBERS, kind="comboo"),
        _benchmark(),
    )
    report = run_precheck(plan, domain_features=("ROE",))
    assert _codes(report) == ["unknown_arm_kind"]
    issue = report.issues[0]
    assert issue.arm == "combo:all3"
    assert issue.member is None
    assert issue.experiment_id == "r01_slow_baseline"
    assert issue.locator == (
        "(experiment=r01_slow_baseline, arm=combo:all3, member=none)"
    )
    assert "comboo" in issue.detail

    # GREEN: vocabulary-clean plans never produce the code — the real
    # slow plan's 15 issues are exactly PC1's three codes, no fourth.
    real = run_precheck(materialize_plan("r01_slow_baseline"))
    assert not [i for i in real.issues if i.code == "unknown_arm_kind"]
    assert len(real.issues) == 15


# --- PC14: a comparison arm cannot hide members ---------------------------


def test_p50_pc14_comparison_arm_hiding_members_is_flagged():
    # Reviewer probe 2, verbatim: benchmark:equal_weight carrying
    # members=['GHOST_MEMBER_X'] used to produce ZERO warnings —
    # comparison arms sit outside every member-level lens by design,
    # so a hand-edited control arm was the one place a member could
    # hide where no member check ever reads.  Both comparison kinds
    # are guarded, locatably, with the hidden members named.
    ghost_benchmark = PlannedArm(
        candidate="benchmark:equal_weight",
        kind="benchmark",
        members=("GHOST_MEMBER_X",),
        direction_rule=None,
        cost_levels=("ref",),
        is_primary=False,
        members_source="code.r0x(benchmark:equal_weight)",
    )
    plan = _plan(_single("ROE"), _combo("combo:test", ("ROE",)), ghost_benchmark)
    report = run_precheck(plan, domain_features=("ROE", "GHOST_MEMBER_X"))
    assert _codes(report) == ["comparison_arm_with_members"]
    issue = report.issues[0]
    assert issue.arm == "benchmark:equal_weight"
    assert issue.locator == (
        "(experiment=r01_slow_baseline, arm=benchmark:equal_weight, member=none)"
    )
    assert "GHOST_MEMBER_X" in issue.detail

    ghost_control = PlannedArm(
        candidate="control:cash",
        kind="control",
        members=("GHOST_MEMBER_Y",),
        direction_rule=None,
        cost_levels=("0x",),
        is_primary=False,
        members_source="code.r01_baseline(control:cash)",
    )
    plan2 = _plan(_single("ROE"), ghost_control, _benchmark())
    report2 = run_precheck(plan2, domain_features=("ROE", "GHOST_MEMBER_Y"))
    assert _codes(report2) == ["comparison_arm_with_members"]
    assert report2.issues[0].arm == "control:cash"
    assert "GHOST_MEMBER_Y" in report2.issues[0].detail

    # GREEN: the memberless comparison arms of a clean plan — and of
    # every real plan (PC1's 15 / PC2's 0 already prove it) — never
    # fire the code.
    clean = _plan(_single("ROE"), _combo("combo:test", ("ROE",)), _benchmark())
    assert run_precheck(clean, domain_features=("ROE",)).issues == ()
