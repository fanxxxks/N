"""WP14 (p50) R1: run-plan parsing and materialization.

Contract: ``docs/p50_experiment_integrity_draft.md`` §3.1/§8.  A run
plan is rebuilt OFFLINE from the four source layers (P §0: L1 contract
/ L2 config / L3 code constants / L4 run records) — config folds and
``baseline_signals`` from L2, combo members / cost ladders / panel axes
from the frozen r0x constants (L3), with sha256 snapshots of the
contract and config files taken at materialization time (L1/L2) and
the artifact paths recorded for the later L4 cross-check.  The report
itself is NEVER read to build the expected row set — that would be
circular (§8).

What this module deliberately does not do: it does not re-run,
patch or re-adjudicate any historical artifact; it does not assume a
uniform cost ladder (every arm carries its own ``cost_levels`` —
benchmark ``("ref",)``, r02/r03 cash ``("1x",)``, r01 signal arms
``("0x","1x","2x")``, S §2.2); it does not fold the r02_periods
``period`` axis away (arms are expanded per period and every RowKey
carries it, F9).  r0x modules are imported lazily inside functions so
``ashare_model.experiment_integrity`` stays import-light for R2
consumers that must run before any data load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .types import (
    ARM_KINDS,
    EXPERIMENT_IDS,
    DirectionRule,
    PlanValidationError,
    PlannedArm,
    PlannedFold,
    RowKey,
    RunPlan,
)

#: p50 §8: the exhaustive verdict-word -> economic-label map, frozen
#: VERBATIM from each experiment's code vocabulary (F23).  No wildcard
#: or "everything else" entry exists or may be added: an outcome word
#: absent from the map is a fail-closed IntegrityRefusal in status.py
#: (§5.4 #4), never a silent default.
VERDICT_MAPS: dict[str, dict[str, str]] = {
    # r01_baseline.py:864 — worth_search = bool(qualified)（bool 结局，
    # 规范化为字面键；adjudication_path="adjudication.worth_search"）
    "r01_slow_baseline": {
        "worth_search:true": "economic_positive",
        "worth_search:false": "economic_negative",
        # false 的 bottleneck="no_qualified_arm: no cost-bearing arm shows
        # durable net-of-cost space under the frozen rules" — 程序无错且臂齐时
        # 是冻结规则下的干净负结果（M3）
    },
    # r02_matrix.py:520-575 五词逐字（grid_adjudication.verdict）
    "r02_matrix": {
        "stable_region_found": "economic_positive",
        "isolated_single_winner": "inconclusive_mixed",   # UNSTABLE
        "multiple_disjoint_viable": "inconclusive_mixed",  # UNSTABLE
        "gross_info_execution_blocked": "inconclusive_mixed",  # 执行诊断续走
        "panel_no_reliable_info": "economic_negative",
        # note: "no viable cell and no gross information anywhere …
        #  the bottleneck is the signal, not the execution structure" —
        # 信号负，非执行负（r02_matrix.py:560-566）
    },
    # r02_matrix.py:630-660 四词逐字（period_adjudication.verdict）
    "r02_periods": {
        "stable_period_region": "economic_positive",
        "isolated_period_winner": "inconclusive_mixed",   # UNSTABLE
        "gross_info_execution_blocked_period_panel": "inconclusive_mixed",
        "period_panel_no_reliable_info": "economic_negative",
        # note: "no viable period and no gross information … the bottleneck
        #  is the signal, not the cadence" — 信号负（r02_matrix.py:655-660）
    },
    # r03_combos.py:258-287 四词逐字（adjudication.verdict）
    "r03_combos": {
        "fitted_increment_region": "economic_positive",
        "equal_weight_stands": "economic_positive",   # 复制成立=正
        "combo_no_net_space": "economic_negative",    # p37 净空间未复制
        "panel_no_reliable_info": "inconclusive_mixed",
        # note: "mixed panel: viable in n/3 cells — neither a replication,
        #  an increment, nor a clean negative"（r03_combos.py:281-287）
    },
    # WP18 (p56, FROZEN_18_PLAN §6.1): the stop/continue conditions are
    # the experiment's adjudication vocabulary, enumerated verbatim as
    # outcome words the WP18 driver writes to adjudication.verdict.  No
    # wildcard entry: a word outside this map is a fail-closed
    # IntegrityRefusal in status.py, never a silent default.
    "wp18_medium": {
        # §6.1 #5: 可执行、拼接净超额与 active IR 为正、年度表现较一致、
        # 动态优于持有、成本压力下仍有空间（才考虑有限 GP/Random 对照）
        "executable_edge_with_cost_space": "economic_positive",
        # §6.1 #2: 纯信号有信息但账户执行后消失（只处理已定位的执行问题）
        "signal_only_execution_loss": "inconclusive_mixed",
        # §6.1 #3: 持有价值、动态没有增量（停止宣称动态选股有效）
        "hold_value_no_dynamic_increment": "economic_negative",
        # §6.1 #4: A/B 对结果影响很大但都无稳健增量（保留负结果）
        "direction_rules_unstable_no_increment": "economic_negative",
        # §6.1 #6: 两域在有效、可比的测量下均无增量（暂停这批假设）
        "no_increment_pause_hypotheses": "economic_negative",
        # §6.1 #1: 测量完整性问题（measurement_incomplete shadows every
        # economic label in status.py, so this never surfaces as one）
        "measurement_blocked": "inconclusive_mixed",
    },
    # WP18 slow carries the identical frozen condition table (§6.1 is
    # package-level, not per-domain).
    "wp18_slow": {
        "executable_edge_with_cost_space": "economic_positive",
        "signal_only_execution_loss": "inconclusive_mixed",
        "hold_value_no_dynamic_increment": "economic_negative",
        "direction_rules_unstable_no_increment": "economic_negative",
        "no_increment_pause_hypotheses": "economic_negative",
        "measurement_blocked": "inconclusive_mixed",
    },
}

#: The comparison arms (R1: the plan must list them explicitly; they
#: are measured and reported but never adjudicated as value evidence).
BENCHMARK_CANDIDATE = "benchmark:equal_weight"
CASH_CANDIDATE = "control:cash"

#: p34 §3.3: the zero-cost benchmark level is outside every ladder.
REF_COST_LEVELS = ("ref",)

#: p50 §8 per-experiment source table (L1 contract / L2 config / L4
#: product paths + the adjudication outcome-word location F22 and the
#: summary-layer declarations consumed by reconcile, F20).
_EXPERIMENT_SOURCES: dict[str, dict[str, Any]] = {
    "r01_slow_baseline": {
        "contract": "docs/p34_r01_fixed_baseline_diagnostic_draft.md",
        "config": "configs/r1_slow_ab.yaml",
        "code_constants": [
            {
                "module": "ashare_model.r01_baseline",
                "names": [
                    "R01_COMBO_ARMS",
                    "COST_LEVELS",
                    "R01_ADJUDICATION_RULES",
                ],
            },
        ],
        "product": {
            "report_path": "data/r1/r01_slow_baseline/r01_baseline_report.json",
            "run_meta_path": (
                "data/r1/r01_slow_baseline/r01_baseline_run_meta.json"
            ),
        },
        "adjudication_path": "adjudication.worth_search",
        "summary_layer_path": "arm_summaries",
        "summary_layer_arms": "all_arms",
        "summary_layer_period_prefilter": False,
        "key_axes": ("candidate", "cost_level", "fold_test_end"),
    },
    "r02_matrix": {
        "contract": "docs/p37_r02_portfolio_cost_matrix_draft.md",
        "config": "configs/r1_medium_a.yaml",
        "code_constants": [
            {
                "module": "ashare_model.r02_matrix",
                "names": [
                    "R02_GRID",
                    "R02_REFERENCE_CELL",
                    "R02_COST_LEVELS",
                    "R02_ADJUDICATION_RULES",
                ],
            },
            {
                "module": "ashare_model.r01_baseline",
                "names": ["R01_COMBO_ARMS"],
            },
        ],
        "product": {
            "report_path": "data/r1/r02_matrix/r02_matrix_report.json",
            "run_meta_path": "data/r1/r02_matrix/r02_matrix_run_meta.json",
        },
        "adjudication_path": "grid_adjudication.verdict",
        "summary_layer_path": "arm_summaries",
        "summary_layer_arms": "all_arms",
        "summary_layer_period_prefilter": False,
        "key_axes": ("cell", "candidate", "cost_level", "fold_test_end"),
    },
    "r02_periods": {
        "contract": "docs/p39_r02_periods_draft.md",
        "config": "configs/r1_medium_a.yaml",
        "code_constants": [
            {
                "module": "ashare_model.r02_matrix",
                "names": [
                    "R02_PERIODS",
                    "R02_REFERENCE_CELL",
                    "R02_COST_LEVELS",
                    "R02_ADJUDICATION_RULES",
                ],
            },
            {
                "module": "ashare_model.r01_baseline",
                "names": ["R01_COMBO_ARMS"],
            },
        ],
        "product": {
            "report_path": "data/r1/r02_periods/r02_periods_report.json",
            "run_meta_path": "data/r1/r02_periods/r02_periods_run_meta.json",
        },
        "adjudication_path": "period_adjudication.verdict",
        # F20: the summary layer lives per period block and its arm domain
        # is ONLY the 5 reference-cell signal arms (no cash, no benchmark);
        # rows must be period-prefiltered before the rebuild comparison.
        "summary_layer_path": "period_blocks.<freq>.arms",
        "summary_layer_arms": "reference_cell_signal_arms",
        "summary_layer_period_prefilter": True,
        "key_axes": (
            "period",
            "cell",
            "candidate",
            "cost_level",
            "fold_test_end",
        ),
    },
    "r03_combos": {
        "contract": "docs/p42_r03_combo_weights_draft.md",
        "config": "configs/r1_medium_a.yaml",
        "code_constants": [
            {
                "module": "ashare_model.r03_combos",
                "names": [
                    "R03_CELLS",
                    "R03_REFERENCE_CELL",
                    "R03_LAMBDAS",
                    "R03_ADJUDICATED_ARMS",
                    "R03_MEMBERS",
                    "R03_COST_LEVELS",
                    "R03_ADJUDICATION_RULES",
                ],
            },
        ],
        "product": {
            "report_path": "data/r1/r03_combos/r03_combo_report.json",
            "run_meta_path": "data/r1/r03_combos/r03_combo_run_meta.json",
        },
        "adjudication_path": "adjudication.verdict",
        "summary_layer_path": "arm_summaries",
        "summary_layer_arms": "all_arms",
        "summary_layer_period_prefilter": False,
        "key_axes": ("cell", "candidate", "cost_level", "fold_test_end"),
    },
    # WP18 (p56, FROZEN_18_PLAN §10.2 additive registration).  The
    # contract is the frozen plan itself; the config is the domain's
    # existing L2 file (fold/baseline_signals authority); the code
    # constants live in the WP18 module; the products land in the NEW
    # §9 directories only.  Cell/period axes are unused (stage/rule/mode
    # travel inside the candidate encoding, key_axes documents that).
    "wp18_medium": {
        "contract": "reports/wp18/FROZEN_18_PLAN.md",
        "config": "configs/r1_medium_a.yaml",
        "code_constants": [
            {
                "module": "ashare_model.wp18_signal_vs_account",
                "names": [
                    "WP18_MEDIUM_SINGLES",
                    "WP18_MEDIUM_COMBO_ID",
                    "WP18_RULES",
                    "WP18_MODES",
                    "WP18_STAGES",
                    "WP18_ADJUDICATION_RULES",
                ],
            },
            {
                "module": "ashare_model.r02_matrix",
                "names": ["R02_REFERENCE_CELL"],
            },
        ],
        "product": {
            "report_path": "data/r1/wp18_medium/wp18_medium_report.json",
            "run_meta_path": "data/r1/wp18_medium/wp18_medium_run_meta.json",
        },
        "adjudication_path": "adjudication.verdict",
        "summary_layer_path": "arm_summaries",
        "summary_layer_arms": "all_arms",
        "summary_layer_period_prefilter": False,
        "key_axes": ("candidate", "cost_level", "fold_test_end"),
    },
    "wp18_slow": {
        "contract": "reports/wp18/FROZEN_18_PLAN.md",
        "config": "configs/r1_slow_ab.yaml",
        "code_constants": [
            {
                "module": "ashare_model.wp18_signal_vs_account",
                "names": [
                    "SLOW_SINGLES",
                    "SLOW_COMBO_ID",
                    "WP18_RULES",
                    "WP18_MODES",
                    "WP18_STAGES",
                    "WP18_ADJUDICATION_RULES",
                ],
            },
        ],
        "product": {
            "report_path": "data/r1/wp18_slow/wp18_slow_report.json",
            "run_meta_path": "data/r1/wp18_slow/wp18_slow_run_meta.json",
        },
        "adjudication_path": "adjudication.verdict",
        "summary_layer_path": "arm_summaries",
        "summary_layer_arms": "all_arms",
        "summary_layer_period_prefilter": False,
        "key_axes": ("candidate", "cost_level", "fold_test_end"),
    },
}


def _default_repo_root() -> Path:
    """The AlphaGPT repository root (…/AlphaGPT), three parents up."""

    return Path(__file__).resolve().parents[2]


def _rule_version() -> int:
    """direction.DIRECTION_RULE_VERSION at the code the plan describes
    (§8: anchored to the RUNTIME code's value — 1 for all four audited
    artifacts — not to any "materialization day" notion)."""

    from ashare_model import direction

    return int(direction.DIRECTION_RULE_VERSION)


def _direction_rule(scope: str, learned_at: str) -> DirectionRule:
    return DirectionRule(
        rule="reference_return",
        rule_version=_rule_version(),
        scope=scope,
        learned_at=learned_at,
    )


def _benchmark_arm(*, period: str | None = None) -> PlannedArm:
    """R1: the zero-cost equal-weight comparison arm, listed explicitly
    and never mixed among signal arms (cost ladder is ("ref",) only,
    cell axis None — benchmark rows carry no cell field, F8/F19)."""

    return PlannedArm(
        candidate=BENCHMARK_CANDIDATE,
        kind="benchmark",
        members=(),
        direction_rule=None,
        cost_levels=REF_COST_LEVELS,
        is_primary=False,
        members_source="code.r0x(benchmark:equal_weight)",
        cell=None,
        period=period,
        note="comparison arm: measured and reported, never adjudicated",
    )


def _r01_arms(baseline_signals: tuple[str, ...]) -> tuple[PlannedArm, ...]:
    """§8 r01 row: 5 singles + 2 combos on the full 0x/1x/2x ladder,
    cash on the same ladder, benchmark on ref — all cell-free."""

    from ashare_model.r01_baseline import COST_LEVELS, R01_COMBO_ARMS

    rule = _direction_rule("per_fold", "none(reference=own train window)")
    arms: list[PlannedArm] = [
        PlannedArm(
            candidate=f"baseline:{name}",
            kind="single",
            members=(name,),
            direction_rule=rule,
            cost_levels=tuple(COST_LEVELS),
            is_primary=True,
            members_source="config.protocol.baseline_signals",
        )
        for name in baseline_signals
    ]
    arms.extend(
        PlannedArm(
            candidate=combo_name,
            kind="combo",
            members=tuple(members),
            direction_rule=rule,
            cost_levels=tuple(COST_LEVELS),
            is_primary=True,
            members_source="code.R01_COMBO_ARMS",
        )
        for combo_name, members in R01_COMBO_ARMS
    )
    arms.append(
        PlannedArm(
            candidate=CASH_CANDIDATE,
            kind="control",
            members=(),
            direction_rule=None,
            cost_levels=tuple(COST_LEVELS),
            is_primary=False,
            members_source="code.r01_baseline(control:cash)",
        )
    )
    arms.append(_benchmark_arm())
    return tuple(arms)


def _r02_matrix_arms(baseline_signals: tuple[str, ...]) -> tuple[PlannedArm, ...]:
    """§8 r02_matrix row: (6 cells x 5 signal arms) x {0x,1x} + cash on
    the reference cell at 1x only + benchmark on ref — 32 expanded arms
    (== its arm_summaries key count, F8)."""

    from ashare_model.r01_baseline import R01_COMBO_ARMS
    from ashare_model.r02_matrix import (
        R02_COST_LEVELS,
        R02_GRID,
        R02_REFERENCE_CELL,
    )

    rule = _direction_rule("per_fold", f"reference_cell={R02_REFERENCE_CELL}")
    arms: list[PlannedArm] = []
    for cell in R02_GRID:
        for name in baseline_signals:
            arms.append(
                PlannedArm(
                    candidate=f"{cell.name}::baseline:{name}",
                    kind="single",
                    members=(name,),
                    direction_rule=rule,
                    cost_levels=tuple(R02_COST_LEVELS),
                    is_primary=True,
                    members_source="config.protocol.baseline_signals",
                    cell=cell.name,
                )
            )
        for combo_name, members in R01_COMBO_ARMS:
            arms.append(
                PlannedArm(
                    candidate=f"{cell.name}::{combo_name}",
                    kind="combo",
                    members=tuple(members),
                    direction_rule=rule,
                    cost_levels=tuple(R02_COST_LEVELS),
                    is_primary=True,
                    members_source="code.R01_COMBO_ARMS",
                    cell=cell.name,
                )
            )
    # F19: cash runs ONLY on the reference cell, and its candidate string
    # carries NO cell:: prefix while the row's cell axis is the reference
    # cell — the asymmetry the RowKey must reproduce exactly.
    arms.append(
        PlannedArm(
            candidate=CASH_CANDIDATE,
            kind="control",
            members=(),
            direction_rule=None,
            cost_levels=("1x",),
            is_primary=False,
            members_source="code.r02_matrix(control:cash)",
            cell=R02_REFERENCE_CELL,
            note="reference cell only; candidate string has no cell:: prefix",
        )
    )
    arms.append(_benchmark_arm())
    return tuple(arms)


def _r02_periods_arms(baseline_signals: tuple[str, ...]) -> tuple[PlannedArm, ...]:
    """§8 r02_periods row: the r02_matrix reference-cell arm set re-run
    per period — signal arms, cash AND benchmark all expand over the
    period axis (F19: cash 10 rows, benchmark 10 rows); directions are
    the only per-(period, fold) relearn in the family (p39 §3)."""

    from ashare_model.r01_baseline import R01_COMBO_ARMS
    from ashare_model.r02_matrix import (
        R02_COST_LEVELS,
        R02_PERIODS,
        R02_REFERENCE_CELL,
    )

    arms: list[PlannedArm] = []
    for period_name, _days in R02_PERIODS:
        rule = _direction_rule(
            "per_(period,fold)", f"reference_cell={R02_REFERENCE_CELL}"
        )
        for name in baseline_signals:
            arms.append(
                PlannedArm(
                    candidate=f"{R02_REFERENCE_CELL}::baseline:{name}",
                    kind="single",
                    members=(name,),
                    direction_rule=rule,
                    cost_levels=tuple(R02_COST_LEVELS),
                    is_primary=True,
                    members_source="config.protocol.baseline_signals",
                    cell=R02_REFERENCE_CELL,
                    period=period_name,
                )
            )
        for combo_name, members in R01_COMBO_ARMS:
            arms.append(
                PlannedArm(
                    candidate=f"{R02_REFERENCE_CELL}::{combo_name}",
                    kind="combo",
                    members=tuple(members),
                    direction_rule=rule,
                    cost_levels=tuple(R02_COST_LEVELS),
                    is_primary=True,
                    members_source="code.R01_COMBO_ARMS",
                    cell=R02_REFERENCE_CELL,
                    period=period_name,
                )
            )
        arms.append(
            PlannedArm(
                candidate=CASH_CANDIDATE,
                kind="control",
                members=(),
                direction_rule=None,
                cost_levels=("1x",),
                is_primary=False,
                members_source="code.r02_matrix(control:cash)",
                cell=R02_REFERENCE_CELL,
                period=period_name,
            )
        )
        arms.append(_benchmark_arm(period=period_name))
    return tuple(arms)


def _r03_arms(baseline_signals: tuple[str, ...]) -> tuple[PlannedArm, ...]:
    """§8 r03 row: 3 cells x 4 combo arms on {0x,1x} (is_primary frozen
    by R03_ADJUDICATED_ARMS — the descriptive lambdas are never primary)
    + the 3 reference-cell singles at 1x only (direction-path byproduct)
    + benchmark on ref — 16 expanded arms (== arm_summaries keys, F10)."""

    from ashare_model.r03_combos import (
        R03_ADJUDICATED_ARMS,
        R03_CELLS,
        R03_COST_LEVELS,
        R03_EQ_ARM,
        R03_LAMBDAS,
        R03_MEMBERS,
        R03_REFERENCE_CELL,
    )

    rule = _direction_rule("per_fold", f"reference_cell={R03_REFERENCE_CELL}")
    adjudicated = tuple(R03_ADJUDICATED_ARMS)
    arms: list[PlannedArm] = []
    for cell in R03_CELLS:
        arms.append(
            PlannedArm(
                candidate=f"{cell}::{R03_EQ_ARM}",
                kind="combo",
                members=tuple(R03_MEMBERS),
                direction_rule=rule,
                cost_levels=tuple(R03_COST_LEVELS),
                is_primary=True,
                members_source="code.R03_MEMBERS",
                cell=cell,
            )
        )
        for arm_name in R03_LAMBDAS:
            primary = arm_name in adjudicated
            arms.append(
                PlannedArm(
                    candidate=f"{cell}::{arm_name}",
                    kind="combo",
                    members=tuple(R03_MEMBERS),
                    direction_rule=rule,
                    cost_levels=tuple(R03_COST_LEVELS),
                    is_primary=primary,
                    members_source="code.R03_MEMBERS",
                    cell=cell,
                    note=(
                        ""
                        if primary
                        else "descriptive lambda arm: disclosed, never adjudicated"
                    ),
                )
            )
    for name in R03_MEMBERS:
        arms.append(
            PlannedArm(
                candidate=f"{R03_REFERENCE_CELL}::baseline:{name}",
                kind="single",
                members=(name,),
                direction_rule=rule,
                cost_levels=("1x",),
                is_primary=False,
                members_source="code.R03_MEMBERS",
                cell=R03_REFERENCE_CELL,
                note="reference-cell direction-path byproduct (1x only)",
            )
        )
    arms.append(_benchmark_arm())
    return tuple(arms)


def _wp18_arms(
    experiment_id: str, baseline_signals: tuple[str, ...]
) -> tuple[PlannedArm, ...]:
    """§8 WP18 row: the frozen plan's per-domain arm family (p56,
    FROZEN_18_PLAN §1/§4 via ``wp18_planned_arms``).  The config's L2
    ``baseline_signals`` is the drift guard: it MUST equal THIS
    experiment's frozen WP18 singles or the plan refuses (a hand-edited
    or cross-domain config must never silently re-define the
    preregistered candidate pool)."""

    from ashare_model.wp18_signal_vs_account import (
        SLOW_SINGLES,
        WP18_MEDIUM_SINGLES,
        wp18_planned_arms,
    )

    frozen: dict[str, tuple[str, ...]] = {
        "wp18_medium": tuple(WP18_MEDIUM_SINGLES),
        "wp18_slow": tuple(SLOW_SINGLES),
    }
    expected = frozen[experiment_id]
    if tuple(baseline_signals) != expected:
        raise PlanValidationError(
            "config protocol.baseline_signals does not match the frozen "
            "WP18 single-factor pool of this experiment — the "
            "preregistered candidate registry is frozen "
            "(FROZEN_18_PLAN §1)\n"
            f"(experiment={experiment_id}, "
            f"observed={list(baseline_signals)!r}, "
            f"frozen={list(expected)!r})"
        )
    return wp18_planned_arms(experiment_id.removeprefix("wp18_"))


def _wp18_medium_arms(
    baseline_signals: tuple[str, ...],
) -> tuple[PlannedArm, ...]:
    return _wp18_arms("wp18_medium", baseline_signals)


def _wp18_slow_arms(
    baseline_signals: tuple[str, ...],
) -> tuple[PlannedArm, ...]:
    return _wp18_arms("wp18_slow", baseline_signals)


_ARM_EXTRACTORS = {
    "r01_slow_baseline": _r01_arms,
    "r02_matrix": _r02_matrix_arms,
    "r02_periods": _r02_periods_arms,
    "r03_combos": _r03_arms,
    "wp18_medium": _wp18_medium_arms,
    "wp18_slow": _wp18_slow_arms,
}

_ADJUDICATION_RULE_LOADERS = {
    "r01_slow_baseline": "ashare_model.r01_baseline:R01_ADJUDICATION_RULES",
    "r02_matrix": "ashare_model.r02_matrix:R02_ADJUDICATION_RULES",
    "r02_periods": "ashare_model.r02_matrix:R02_ADJUDICATION_RULES",
    "r03_combos": "ashare_model.r03_combos:R03_ADJUDICATION_RULES",
    "wp18_medium": (
        "ashare_model.wp18_signal_vs_account:WP18_ADJUDICATION_RULES"
    ),
    "wp18_slow": (
        "ashare_model.wp18_signal_vs_account:WP18_ADJUDICATION_RULES"
    ),
}


def _load_adjudication_rules(experiment_id: str) -> dict[str, float]:
    """The frozen rules object of the experiment's own adjudicator
    (r03's is the p37 authority consumed as the same object — one
    implementation, never two)."""

    import importlib

    module_name, _, attr = _ADJUDICATION_RULE_LOADERS[experiment_id].partition(":")
    try:
        value = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise PlanValidationError(
            f"missing code constant for adjudication rules\n"
            f"(experiment={experiment_id}, "
            f"locator={_ADJUDICATION_RULE_LOADERS[experiment_id]}): {exc}"
        ) from exc
    return {str(k): float(v) for k, v in value.items()}


def _parse_protocol(
    experiment_id: str, config: Mapping[str, Any]
) -> tuple[str, str | None, tuple[PlannedFold, ...], tuple[str, ...]]:
    """L2 extraction with locatable failures: domain / tier_set / folds /
    baseline_signals.  Anything absent or malformed is a PlanValidationError
    naming (experiment, locator) — before any data is loaded (R2 gate)."""

    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise PlanValidationError(
            "config is missing the protocol block\n"
            f"(experiment={experiment_id}, locator=protocol)"
        )
    domain = protocol.get("domain")
    if not isinstance(domain, str) or not domain:
        raise PlanValidationError(
            "config is missing protocol.domain\n"
            f"(experiment={experiment_id}, locator=protocol.domain)"
        )
    tier_set = protocol.get("tier_set")
    if tier_set is not None and not isinstance(tier_set, str):
        raise PlanValidationError(
            "protocol.tier_set must be a string or absent\n"
            f"(experiment={experiment_id}, locator=protocol.tier_set, "
            f"value={tier_set!r})"
        )
    raw_folds = protocol.get("folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise PlanValidationError(
            "config is missing protocol.folds (the only fold-date source, "
            "L2 authority)\n"
            f"(experiment={experiment_id}, locator=protocol.folds)"
        )
    folds: list[PlannedFold] = []
    for index, fold in enumerate(raw_folds):
        if not isinstance(fold, Mapping):
            raise PlanValidationError(
                "fold entry is not a mapping\n"
                f"(experiment={experiment_id}, locator=protocol.folds[{index}])"
            )
        train_end = fold.get("train_end")
        test_end = fold.get("test_end")
        if not isinstance(train_end, str) or not isinstance(test_end, str):
            raise PlanValidationError(
                "fold entry lacks string train_end/test_end dates\n"
                f"(experiment={experiment_id}, "
                f"locator=protocol.folds[{index}], "
                f"train_end={train_end!r}, test_end={test_end!r})"
            )
        folds.append(PlannedFold(train_end=train_end, test_end=test_end))
    signals = protocol.get("baseline_signals")
    if not isinstance(signals, list) or not signals or not all(
        isinstance(name, str) and name for name in signals
    ):
        raise PlanValidationError(
            "config is missing protocol.baseline_signals (the single-factor "
            "member source)\n"
            f"(experiment={experiment_id}, locator=protocol.baseline_signals)"
        )
    return domain, tier_set, tuple(folds), tuple(signals)


def plan_sources(
    experiment_id: str,
    config_path: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """The L1-L4 source snapshot of one experiment (§8): contract and
    config paths with sha256 digests computed from the CURRENT files
    (the recon anchor; a later reconcile compares them against the
    report's recorded provenance), the code-constant module/names the
    plan materializes from, the product paths, the adjudication
    outcome-word location (F22) and the summary-layer declarations
    (F20).  Pure snapshot — no judgment, no report read."""

    if experiment_id not in EXPERIMENT_IDS:
        raise PlanValidationError(
            "unknown experiment_id\n"
            f"(experiment={experiment_id!r}, locator=experiment_id; "
            f"expected one of {EXPERIMENT_IDS})"
        )
    spec = _EXPERIMENT_SOURCES[experiment_id]
    root = Path(repo_root) if repo_root is not None else _default_repo_root()

    from ashare_model.r01_baseline import _sha256_file

    resolved_config = (
        Path(config_path)
        if config_path is not None
        else root / spec["config"]
    )

    def _digest(path: Path) -> str | None:
        return _sha256_file(path) if path.is_file() else None

    return {
        "contract": {
            "path": spec["contract"],
            "sha256": _digest(root / spec["contract"]),
        },
        "config": {
            "path": spec["config"],
            "sha256": _digest(resolved_config),
        },
        "code_constants": [dict(entry) for entry in spec["code_constants"]],
        "product": dict(spec["product"]),
        "adjudication_path": spec["adjudication_path"],
        "summary_layer_path": spec["summary_layer_path"],
        "summary_layer_arms": spec["summary_layer_arms"],
        "summary_layer_period_prefilter": spec["summary_layer_period_prefilter"],
    }


def materialize_plan(
    experiment_id: str,
    config_path: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
) -> RunPlan:
    """Materialize the R1 run plan: arms (cell/period-EXPANDED, each
    carrying its own members, direction rule and cost ladder), folds,
    key axes, frozen adjudication rules, the exhaustive verdict map and
    the L1-L4 source snapshot.  The expected row set follows as
    cross(arms, cost_levels, folds) in :func:`expected_keys` — never a
    uniform-ladder assumption.  Raises PlanValidationError (locating
    the experiment and the offending field) for an unknown experiment,
    a config missing folds/baseline_signals, or a missing code
    constant."""

    if experiment_id not in EXPERIMENT_IDS:
        raise PlanValidationError(
            "unknown experiment_id\n"
            f"(experiment={experiment_id!r}, locator=experiment_id; "
            f"expected one of {EXPERIMENT_IDS})"
        )
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    spec = _EXPERIMENT_SOURCES[experiment_id]
    resolved_config = (
        Path(config_path)
        if config_path is not None
        else root / spec["config"]
    )
    if not resolved_config.is_file():
        raise PlanValidationError(
            "config file not found\n"
            f"(experiment={experiment_id}, "
            f"locator=config:{resolved_config})"
        )

    from ashare_model.r01_baseline import load_config_file

    config = load_config_file(Path(resolved_config))
    domain, tier_set, folds, baseline_signals = _parse_protocol(
        experiment_id, config
    )
    arms = _ARM_EXTRACTORS[experiment_id](baseline_signals)
    if not arms:
        raise PlanValidationError(
            "materialized arm list is empty\n"
            f"(experiment={experiment_id}, locator=arms)"
        )
    return RunPlan(
        experiment_id=experiment_id,
        research_domain=domain,
        tier_set=tier_set,
        folds=folds,
        arms=arms,
        key_axes=tuple(spec["key_axes"]),
        adjudication_rules=_load_adjudication_rules(experiment_id),
        verdict_map=dict(VERDICT_MAPS[experiment_id]),
        sources=plan_sources(experiment_id, resolved_config, repo_root=root),
    )


def expected_keys(plan: RunPlan) -> list[RowKey]:
    """The expected row-key set: cross(arms, cost_levels, folds), each
    key carrying its arm's own cell/period axis values (F19: the cash
    key has cell=reference-cell with an unprefixed candidate string;
    the benchmark key has cell=None but a period under r02_periods).
    Deterministic order: arm order, then cost level, then fold."""

    return [
        RowKey(
            period=arm.period,
            cell=arm.cell,
            candidate=arm.candidate,
            cost_level=cost_level,
            fold_test_end=fold.test_end,
        )
        for arm in plan.arms
        for cost_level in arm.cost_levels
        for fold in plan.folds
    ]


def _arm_to_json(arm: PlannedArm) -> dict[str, Any]:
    return {
        "candidate": arm.candidate,
        "kind": arm.kind,
        "members": list(arm.members),
        "direction_rule": (
            None
            if arm.direction_rule is None
            else {
                "rule": arm.direction_rule.rule,
                "rule_version": arm.direction_rule.rule_version,
                "scope": arm.direction_rule.scope,
                "learned_at": arm.direction_rule.learned_at,
            }
        ),
        "cost_levels": list(arm.cost_levels),
        "is_primary": arm.is_primary,
        "members_source": arm.members_source,
        "cell": arm.cell,
        "period": arm.period,
        "note": arm.note,
    }


def _arm_from_json(
    payload: Mapping[str, Any], *, experiment_id: str = ""
) -> PlannedArm:
    # Fail closed on the kind vocabulary (§3.1): every member-level
    # precheck class keys on the exact kind string (precheck's
    # SIGNAL_ARM_KINDS / 'single' / 'combo' membership tests), so an
    # out-of-vocabulary kind — a hand-edited plan file with 'comboo' —
    # would silently excuse the arm from all three classes AND shrink
    # the issue list (measured: 15 -> 6 with zero issues naming the
    # arm).  Refused here, locatably, before any data load.
    kind = str(payload.get("kind", ""))
    if kind not in ARM_KINDS:
        raise PlanValidationError(
            "arm kind is outside the ARM_KINDS vocabulary — the "
            "member-level precheck classes key on the exact kind "
            "string, so an unrecognized kind silently bypasses them\n"
            f"(experiment={experiment_id!r}, "
            f"arm.candidate={payload.get('candidate')!r}, "
            f"locator=arms[].kind, kind={kind!r}, "
            f"expected one of {ARM_KINDS})"
        )
    rule_payload = payload.get("direction_rule")
    direction_rule = (
        None
        if rule_payload is None
        else DirectionRule(
            rule=str(rule_payload["rule"]),
            rule_version=int(rule_payload["rule_version"]),
            scope=str(rule_payload["scope"]),
            learned_at=str(rule_payload["learned_at"]),
        )
    )
    return PlannedArm(
        candidate=str(payload["candidate"]),
        kind=kind,
        members=tuple(str(m) for m in payload.get("members", ())),
        direction_rule=direction_rule,
        cost_levels=tuple(str(c) for c in payload.get("cost_levels", ())),
        is_primary=bool(payload.get("is_primary", False)),
        members_source=str(payload.get("members_source", "")),
        cell=payload.get("cell"),
        period=payload.get("period"),
        note=str(payload.get("note", "")),
    )


def plan_to_json(plan: RunPlan) -> dict[str, Any]:
    """The CLI round-trip form (C10: the writer adds
    ``ensure_ascii=False, indent=2``; the dict itself is jsonable)."""

    return {
        "plan_schema_version": plan.plan_schema_version,
        "experiment_id": plan.experiment_id,
        "research_domain": plan.research_domain,
        "tier_set": plan.tier_set,
        "folds": [
            {"train_end": fold.train_end, "test_end": fold.test_end}
            for fold in plan.folds
        ],
        "arms": [_arm_to_json(arm) for arm in plan.arms],
        "key_axes": list(plan.key_axes),
        "adjudication_rules": dict(plan.adjudication_rules),
        "verdict_map": dict(plan.verdict_map),
        "sources": dict(plan.sources),
        "expected_row_count": plan.expected_row_count,
    }


def plan_from_json(payload: Mapping[str, Any]) -> RunPlan:
    """Inverse of :func:`plan_to_json`.  Raises PlanValidationError with
    a locator for a wrong schema version, an unknown experiment id, an
    arm kind outside ARM_KINDS, or missing required sections — never a
    silent default."""

    try:
        schema_version = int(payload["plan_schema_version"])
        experiment_id = str(payload["experiment_id"])
        research_domain = str(payload["research_domain"])
        tier_set = payload.get("tier_set")
        folds = tuple(
            PlannedFold(
                train_end=str(fold["train_end"]),
                test_end=str(fold["test_end"]),
            )
            for fold in payload["folds"]
        )
        arms = tuple(
            _arm_from_json(arm, experiment_id=experiment_id)
            for arm in payload["arms"]
        )
        key_axes = tuple(str(axis) for axis in payload["key_axes"])
        adjudication_rules = {
            str(k): float(v) for k, v in payload["adjudication_rules"].items()
        }
        verdict_map = {
            str(k): str(v) for k, v in payload["verdict_map"].items()
        }
        sources = dict(payload["sources"])
    except PlanValidationError:
        # Locatable arm-level refusals (e.g. an out-of-ARM_KINDS kind)
        # travel unchanged — never re-wrapped into a generic
        # "malformed payload" that would lose the (experiment, arm)
        # locator.
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanValidationError(
            f"plan payload is malformed: {exc}\n(locator=plan_json)"
        ) from exc
    if schema_version != 1:
        raise PlanValidationError(
            "unsupported plan_schema_version\n"
            f"(experiment={experiment_id}, locator=plan_schema_version, "
            f"value={schema_version}, expected=1)"
        )
    if experiment_id not in EXPERIMENT_IDS:
        raise PlanValidationError(
            "unknown experiment_id\n"
            f"(experiment={experiment_id!r}, locator=experiment_id; "
            f"expected one of {EXPERIMENT_IDS})"
        )
    return RunPlan(
        experiment_id=experiment_id,
        research_domain=research_domain,
        tier_set=tier_set,
        folds=folds,
        arms=arms,
        key_axes=key_axes,
        adjudication_rules=adjudication_rules,
        verdict_map=verdict_map,
        sources=sources,
        plan_schema_version=schema_version,
    )
