"""WP18 (p56) — frozen row sets, direction book and registration surface.

Contract: ``reports/wp18/FROZEN_18_PLAN.md`` (pre-registered 2026-09-27;
read-only for the implementer).  This module is the additive WP18
implementation face named by the frozen plan:

* §1 — the frozen candidate registry (8 singles = medium 3 + slow 5,
  10 candidates; ``combo:slow_all5`` is the ONE pre-registered new id,
  equal-weight rank mean of the five existing slow members — no new
  factors, no fitted weights);
* §4 — per-stage expected row-key sets (行集物化): every stage's
  ``(domain, stage, candidate, rule, mode, cost_level, fold)`` key set
  with the frozen counts (M1 30 / M2 40 / M3 90 / C1 40, S1 50 / S2 60
  / S3 130 / C2 40), plus the §4.6 formal-row budget accounting
  (account 1x + stress 0x/2x, two-domain cap 300);
* §2.4 — the frozen direction book: stage-1 ``direction_frozen.json``
  writer/loader.  Account (M3/S3) and stress (C1/C2) stages consume the
  book ONLY — the book carries no learning surface at all, a re-learn
  entry (two directions for one candidate/rule/fold) is refused, and an
  unfrozen artifact is refused;
* §10.2 — the experiment_integrity additive registration data: planned
  arms for the ``wp18_medium`` / ``wp18_slow`` experiment ids and the
  frozen §5/§6.2 adjudication-rule numbers.

What this module deliberately does not do: it runs NOTHING (no engine,
no loader, no searcher — the algorithm search budget is zero); it never
imports the direction LEARNERS (``direction.reference_return_direction``
/ ``label_ic_direction``) — stage-1 code produces ``direction_frozen.json``
once, everything downstream reads the frozen artifact; and it does not
wire ``trade_groups`` (out of WP18 scope).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ashare_model import direction as _direction

WP18_SCHEMA_VERSION = 1
DIRECTION_FROZEN_SCHEMA_VERSION = 1

#: §2 — the two pre-fixed rules, letter -> direction.py authority id.
WP18_RULES: tuple[tuple[str, str], ...] = (
    ("A", _direction.RULE_REFERENCE_RETURN),
    ("B", _direction.RULE_LABEL_IC),
)
RULE_LETTERS: tuple[str, ...] = tuple(letter for letter, _ in WP18_RULES)
RULE_IDS: tuple[str, ...] = tuple(rule_id for _, rule_id in WP18_RULES)

#: §3.2 — the two account-execution modes.
WP18_MODES: tuple[str, str] = ("dynamic", "hold")

#: §1.1 — medium singles (config r1_medium_a.yaml protocol.baseline_signals).
WP18_MEDIUM_SINGLES: tuple[str, ...] = (
    "MOMENTUM_20",
    "RSQ_60",
    "REVERSAL_60",
)

#: §1.1 — the existing medium combo id (r01_baseline.R01_COMBO_ARMS
#: "combo:all3"; members frozen).
WP18_MEDIUM_COMBO_ID = "combo:all3"

#: §1.2 — slow singles (config r1_slow_ab.yaml protocol.baseline_signals).
SLOW_SINGLES: tuple[str, ...] = (
    "ROE",
    "PE_TTM",
    "REVENUE_YOY",
    "DIVIDEND_YIELD",
    "MARKET_CAP",
)

#: §1.2/§1.3 — the ONE pre-registered new id: equal-weight rank mean of
#: the five existing slow members (diversification hypothesis, frozen
#: verbatim in FROZEN_18_PLAN §1.3; no weight fitting, no new factors).
SLOW_COMBO_ID = "combo:slow_all5"

#: §4.6 — the two-domain formal evaluation row cap (account 1x + stress
#: 0x/2x; measurement rows never enter this budget).
FORMAL_EVALUATION_ROW_CAP = 300

#: §3.3 — the formal cost levels.  "dir"/"sig" are measurement-row pseudo
#: levels (direction evidence / signal summary): they never enter cost
#: accounting and never appear on a formal evaluation row.
DIRECTION_ROW_LEVEL = "dir"
SIGNAL_ROW_LEVEL = "sig"
ACCOUNT_LEVEL = "1x"
STRESS_LEVELS: tuple[str, str] = ("0x", "2x")
BENCHMARK_LEVEL = "ref"

#: The stages whose rows are formal evaluation rows (§4.6); everything
#: else is a diagnostic measurement row.
FORMAL_STAGES: frozenset[str] = frozenset({"M3", "S3", "C1", "C2"})
MEASUREMENT_STAGES: frozenset[str] = frozenset({"M1", "M2", "S1", "S2"})

#: §4 — the per-(domain, stage) frozen row counts (the REQ budget table).
WP18_STAGES: dict[tuple[str, str], int] = {
    ("medium", "M1"): 30,
    ("medium", "M2"): 40,
    ("medium", "M3"): 90,
    ("medium", "C1"): 40,
    ("slow", "S1"): 50,
    ("slow", "S2"): 60,
    ("slow", "S3"): 130,
    ("slow", "C2"): 40,
}

#: §2.5 — the frozen reference cell (medium); the slow reference is the
#: r1_slow_ab backtest block verbatim (no cell axis).
MEDIUM_REFERENCE_CELL = "n20_m5000"

_DOMAIN_CONFIG: dict[str, dict[str, Any]] = {
    "medium": {
        "singles": WP18_MEDIUM_SINGLES,
        "combo": (WP18_MEDIUM_COMBO_ID, WP18_MEDIUM_SINGLES),
        "direction_stage": "M1",
        "summary_stage": "M2",
        "account_stage": "M3",
        "stress_stage": "C1",
        "research_domain": "medium_cross_section",
        "tier_set": "A",
        "config_path": "configs/r1_medium_a.yaml",
        "reference": f"reference_cell={MEDIUM_REFERENCE_CELL}",
        "report_path": "data/r1/wp18_medium/wp18_medium_report.json",
        "run_meta_path": "data/r1/wp18_medium/wp18_medium_run_meta.json",
    },
    "slow": {
        "singles": SLOW_SINGLES,
        "combo": (SLOW_COMBO_ID, SLOW_SINGLES),
        "direction_stage": "S1",
        "summary_stage": "S2",
        "account_stage": "S3",
        "stress_stage": "C2",
        "research_domain": "slow_fundamental",
        "tier_set": "AB",
        "config_path": "configs/r1_slow_ab.yaml",
        "reference": "config_backtest_block_verbatim",
        "report_path": "data/r1/wp18_slow/wp18_slow_report.json",
        "run_meta_path": "data/r1/wp18_slow/wp18_slow_run_meta.json",
    },
}

#: §5/§6.2 — the frozen mechanical-screen numbers (resource-allocation
#: gate; NOT a statistical significance proof, NOT a market law).  This
#: object is the experiment_integrity adjudication-rules constant for the
#: wp18 experiment ids (plan._ADJUDICATION_RULE_LOADERS).
WP18_ADJUDICATION_RULES: dict[str, float] = {
    "min_positive_years": 3.0,
    "median_must_be_positive": 1.0,
    "dynamic_minus_hold_annualized_bp": 50.0,
}

#: §6.2 annotation, frozen verbatim.
MECHANICAL_SCREEN_ANNOTATION = (
    "研发资源分配门槛，非统计显著性证明也非市场规律"
)


class WP18PlanError(ValueError):
    """A constructible, locatable WP18 plan/registry violation."""


class FrozenDirectionRefused(RuntimeError):
    """Fail-closed refusal of frozen-direction consumption (§2.4)."""


@dataclass(frozen=True)
class WP18RowKey:
    """§4 row-key uniqueness: (domain, stage, candidate, rule, mode,
    cost_level, fold).  Controls carry rule=None/mode=None (规则无关、
    模式无关, §3.5)."""

    domain: str
    stage: str
    candidate: str
    rule: str | None
    mode: str | None
    cost_level: str
    fold_test_end: str


def _folds(domain: str) -> tuple[str, ...]:
    """The five fold test-ends (2021..2025) from the domain's L2 config —
    the only fold-date source (both configs share the same five folds,
    verified at registration time by the experiment_integrity L2 parse).
    Kept in-module so row-set materialization needs no config read."""

    return (
        "2021-12-31",
        "2022-12-31",
        "2023-12-31",
        "2024-12-31",
        "2025-12-31",
    )


def wp18_candidates(domain: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    """§1 candidate registry: (candidate_id, kind, members) in frozen
    order — singles first, the domain combo last."""

    if domain not in _DOMAIN_CONFIG:
        raise WP18PlanError(
            f"unknown WP18 domain {domain!r}: expected one of "
            f"{sorted(_DOMAIN_CONFIG)} (FROZEN_18_PLAN §0)"
        )
    spec = _DOMAIN_CONFIG[domain]
    out = [(name, "single", (name,)) for name in spec["singles"]]
    combo_id, members = spec["combo"]
    out.append((combo_id, "combo", tuple(members)))
    return tuple(out)


def _validate_stage(domain: str, stage: str) -> None:
    if domain not in _DOMAIN_CONFIG:
        raise WP18PlanError(f"unknown WP18 domain {domain!r}")
    if (domain, stage) not in WP18_STAGES:
        raise WP18PlanError(
            f"unknown WP18 stage {stage!r} for domain {domain!r}: "
            f"expected one of {sorted(s for _, s in WP18_STAGES)}"
        )


def wp18_stage_row_keys(
    domain: str, stage: str, *, folds: Sequence[str] | None = None
) -> tuple[WP18RowKey, ...]:
    """The expected row-key set of one stage (§4), deterministic order.

    * M1/S1 (direction evidence): singles x rules x folds, level "dir";
      combos produce no direction rows (they consume member directions).
    * M2/S2 (signal summary): candidates x rules x folds, level "sig".
    * M3/S3 (account, 1x): candidates x rules x modes x folds + the two
      control arms x folds (benchmark at "ref", cash at "1x").
    * C1/C2 (stress, 0x/2x): the domain's main combo x rules x modes x
      the two stress levels x folds (§4.5 upper bound; the executed
      subset follows the §5 screening granularity).
    """

    _validate_stage(domain, stage)
    spec = _DOMAIN_CONFIG[domain]
    fold_ends = tuple(folds) if folds is not None else _folds(domain)
    candidates = wp18_candidates(domain)
    stage_kind = {
        "direction": stage in ("M1", "S1"),
        "summary": stage in ("M2", "S2"),
        "account": stage in ("M3", "S3"),
        "stress": stage in ("C1", "C2"),
    }
    keys: list[WP18RowKey] = []
    if stage_kind["direction"]:
        for name, _kind, _members in candidates:
            if _kind != "single":
                continue
            for letter, _rule_id in WP18_RULES:
                for fold in fold_ends:
                    keys.append(
                        WP18RowKey(domain, stage, name, letter, None,
                                   DIRECTION_ROW_LEVEL, fold)
                    )
    elif stage_kind["summary"]:
        for cand, _kind, _members in candidates:
            for letter, _rule_id in WP18_RULES:
                for fold in fold_ends:
                    keys.append(
                        WP18RowKey(domain, stage, cand, letter, None,
                                   SIGNAL_ROW_LEVEL, fold)
                    )
    elif stage_kind["account"]:
        for cand, kind, _members in candidates:
            for letter, _rule_id in WP18_RULES:
                for mode in WP18_MODES:
                    for fold in fold_ends:
                        keys.append(
                            WP18RowKey(domain, stage, cand, letter, mode,
                                       ACCOUNT_LEVEL, fold)
                        )
        for control, level in (
            ("benchmark:equal_weight", BENCHMARK_LEVEL),
            ("control:cash", ACCOUNT_LEVEL),
        ):
            for fold in fold_ends:
                keys.append(
                    WP18RowKey(domain, stage, control, None, None, level, fold)
                )
    elif stage_kind["stress"]:
        combo_id, _members = spec["combo"]
        for letter, _rule_id in WP18_RULES:
            for mode in WP18_MODES:
                for level in STRESS_LEVELS:
                    for fold in fold_ends:
                        keys.append(
                            WP18RowKey(domain, stage, combo_id, letter, mode,
                                       level, fold)
                        )
    expected = WP18_STAGES[(domain, stage)]
    if len(keys) != expected:  # defensive: the table is frozen
        raise WP18PlanError(
            f"materialized {domain}/{stage} row set has {len(keys)} rows, "
            f"the frozen budget table declares {expected} (FROZEN_18_PLAN §4)"
        )
    return tuple(keys)


def wp18_formal_row_budget(domains: Iterable[str]) -> int:
    """§4.6 formal-row accounting: account rows (1x) + stress rows
    (0x/2x) summed over the given domains; measurement rows contribute
    zero.  Two-domain full extension is exactly 300."""

    total = 0
    for domain in domains:
        for stage in ("M3", "C1", "S3", "C2"):
            if (domain, stage) in WP18_STAGES:
                total += WP18_STAGES[(domain, stage)]
    return total


# ---------------------------------------------------------------------------
# experiment_integrity bridge (§10.2 additive registration)
# ---------------------------------------------------------------------------


def wp18_row_key_candidate(row: WP18RowKey) -> str:
    """The RowKey.candidate encoding: ``{stage}:{rule}:{mode}:{candidate}``
    for signal rows; the two control arms keep their BARE names
    (``benchmark:equal_weight`` / ``control:cash``) — the r02_matrix
    convention (control rows carry no prefix; the stage is fixed: the
    controls exist only in the account stage M3/S3, §3.5)."""

    if row.rule is None and row.mode is None:
        return row.candidate
    return f"{row.stage}:{row.rule}:{row.mode}:{row.candidate}"


def parse_wp18_candidate(encoded: str) -> tuple[str, str | None, str | None, str]:
    """Inverse of :func:`wp18_row_key_candidate` (the two bare control
    names map to stage=None, rule=None, mode=None)."""

    if encoded in ("benchmark:equal_weight", "control:cash"):
        return "", None, None, encoded
    parts = encoded.split(":", 3)
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    raise WP18PlanError(f"malformed WP18 candidate encoding {encoded!r}")


def wp18_planned_arms(domain: str) -> tuple:
    """The cell/period-free PlannedArm tuple of one WP18 domain — every
    stage's arms (measurement + formal + stress upper bound), each with
    the direction rule of its letter, its cost ladder and its members.
    Consumed by experiment_integrity plan._ARM_EXTRACTORS."""

    from ashare_model.experiment_integrity.types import DirectionRule, PlannedArm

    _validate_domain(domain)
    spec = _DOMAIN_CONFIG[domain]
    rule_version = int(_direction.DIRECTION_RULE_VERSION)

    def _rule(letter: str):
        rule_id = dict(WP18_RULES)[letter]
        return DirectionRule(
            rule=rule_id,
            rule_version=rule_version,
            scope="per_fold",
            learned_at=spec["reference"],
        )

    candidates = wp18_candidates(domain)
    arms: list[PlannedArm] = []
    seen_arms: set[tuple[str, ...]] = set()
    stages = (
        spec["direction_stage"],
        spec["summary_stage"],
        spec["account_stage"],
        spec["stress_stage"],
    )
    for stage in stages:
        for row in wp18_stage_row_keys(domain, stage):
            encoded = wp18_row_key_candidate(row)
            # one arm per distinct (stage, rule, mode, candidate, level):
            # expected_keys crosses each arm with the fold axis exactly
            # once, so fold-iteration must not mint duplicate arms.
            arm_id = (stage, row.rule or "", row.mode or "",
                      row.candidate, row.cost_level)
            if arm_id in seen_arms:
                continue
            seen_arms.add(arm_id)
            if row.rule is None:  # control rows
                is_benchmark = row.candidate.startswith("benchmark:")
                arms.append(
                    PlannedArm(
                        candidate=encoded,
                        kind="benchmark" if is_benchmark else "control",
                        members=(),
                        direction_rule=None,
                        cost_levels=(row.cost_level,),
                        is_primary=False,
                        members_source=(
                            "wp18_frozen_plan§3.5(comparison arm: measured "
                            "and reported, never adjudicated)"
                        ),
                        cell=None,
                        period=None,
                        note=f"stage={stage}",
                    )
                )
                continue
            cand_id, kind, members = next(
                (c, k, m) for c, k, m in candidates if c == row.candidate
            )
            del cand_id
            is_formal = stage in FORMAL_STAGES
            arms.append(
                PlannedArm(
                    candidate=encoded,
                    kind=kind,
                    members=members,
                    direction_rule=_rule(row.rule),
                    cost_levels=(row.cost_level,),
                    is_primary=True,
                    members_source="wp18_frozen_plan§1(FROZEN_18_PLAN candidates)",
                    cell=None,
                    period=None,
                    note=(
                        f"stage={stage}; mode={row.mode}; "
                        + (
                            "formal evaluation row"
                            if is_formal
                            else "diagnostic measurement row (outside the "
                            "300 formal-row budget)"
                        )
                    ),
                )
            )
    return tuple(arms)


def _validate_domain(domain: str) -> None:
    if domain not in _DOMAIN_CONFIG:
        raise WP18PlanError(
            f"unknown WP18 domain {domain!r}: expected one of "
            f"{sorted(_DOMAIN_CONFIG)}"
        )


def wp18_domain_spec(domain: str) -> dict[str, Any]:
    """The registration spec of one domain (read-only copy)."""

    _validate_domain(domain)
    return dict(_DOMAIN_CONFIG[domain])


# ---------------------------------------------------------------------------
# frozen direction book (§2.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BookEntry:
    candidate: str
    rule: str
    fold_test_end: str
    direction: int
    provenance: dict[str, Any]


class FrozenDirectionBook:
    """The stage-1 direction artifact consumed by M3/S3 and C1/C2.

    Constructed ONLY from a ``direction_frozen.json`` written by stage 1
    (M1/S1); this class has no learning surface — there is no code path
    from here to ``direction.reference_return_direction`` /
    ``label_ic_direction`` / ``decide_direction``.  Consumption of an
    absent (candidate, rule, fold) is a fail-closed refusal."""

    __slots__ = ("domain", "rules", "_entries")

    def __init__(self, domain: str, rules: Mapping[str, str],
                 entries: Sequence[_BookEntry]):
        self.domain = str(domain)
        self.rules = dict(rules)
        self._entries = {
            (e.candidate, e.rule, e.fold_test_end): e for e in entries
        }

    @property
    def size(self) -> int:
        return len(self._entries)

    def has(self, candidate: str, rule_letter: str, fold_test_end: str) -> bool:
        return (candidate, rule_letter, fold_test_end) in self._entries

    def direction(self, candidate: str, rule_letter: str,
                  fold_test_end: str) -> int:
        entry = self._entries.get((candidate, rule_letter, fold_test_end))
        if entry is None:
            raise FrozenDirectionRefused(
                "frozen direction book has no entry for this key — the "
                "account/stress stages may only consume stage-1 frozen "
                "directions, and re-learning them is forbidden "
                "(FROZEN_18_PLAN §2.4)\n"
                f"(domain={self.domain}, candidate={candidate!r}, "
                f"rule={rule_letter!r}, fold_test_end={fold_test_end!r})"
            )
        return int(entry.direction)

    def provenance(self, candidate: str, rule_letter: str,
                   fold_test_end: str) -> dict[str, Any]:
        entry = self._entries.get((candidate, rule_letter, fold_test_end))
        if entry is None:
            raise FrozenDirectionRefused(
                "frozen direction book has no entry for this key\n"
                f"(domain={self.domain}, candidate={candidate!r}, "
                f"rule={rule_letter!r}, fold_test_end={fold_test_end!r})"
            )
        return dict(entry.provenance)


def _validate_book_payload(payload: Mapping[str, Any]) -> None:
    if int(payload.get("wp18_direction_schema_version", -1)) != (
        DIRECTION_FROZEN_SCHEMA_VERSION
    ):
        raise FrozenDirectionRefused(
            "unsupported direction_frozen schema version\n"
            f"(observed={payload.get('wp18_direction_schema_version')!r}, "
            f"expected={DIRECTION_FROZEN_SCHEMA_VERSION})"
        )
    if payload.get("frozen") is not True:
        raise FrozenDirectionRefused(
            "direction artifact is not marked frozen=True — only a frozen "
            "stage-1 artifact may be consumed; an in-progress or unfrozen "
            "artifact is refused (FROZEN_18_PLAN §2.4 '阶段产出后冻结')"
        )


def _book_entries(payload: Mapping[str, Any]) -> list[_BookEntry]:
    rules = payload.get("rules")
    if not isinstance(rules, Mapping) or dict(rules) != dict(WP18_RULES):
        raise FrozenDirectionRefused(
            "direction artifact rule map does not match the frozen A/B "
            f"rules {dict(WP18_RULES)!r}"
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise FrozenDirectionRefused("direction artifact carries no entries")
    entries: list[_BookEntry] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, Mapping):
            raise FrozenDirectionRefused(
                f"entry {index} is not a mapping\n(locator=entries[{index}])"
            )
        try:
            candidate = str(item["candidate"])
            rule = str(item["rule"])
            fold = str(item["fold_test_end"])
            direction = int(item["direction"])
            provenance = dict(item.get("provenance", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise FrozenDirectionRefused(
                f"entry {index} is malformed: {exc}\n(locator=entries[{index}])"
            ) from exc
        if rule not in RULE_LETTERS:
            raise FrozenDirectionRefused(
                f"entry {index} rule {rule!r} is outside the frozen letters "
                f"{RULE_LETTERS}\n(locator=entries[{index}])"
            )
        if direction not in (1, -1):
            raise FrozenDirectionRefused(
                f"entry {index} direction {direction!r} is not +/-1 "
                f"(a neutral/undecided direction must never be consumed)\n"
                f"(locator=entries[{index}])"
            )
        entries.append(_BookEntry(candidate, rule, fold, direction, provenance))
    return entries


def _refuse_duplicates(entries: Sequence[_BookEntry]) -> None:
    seen: dict[tuple[str, str, str], int] = {}
    for entry in entries:
        key = (entry.candidate, entry.rule, entry.fold_test_end)
        if key in seen:
            raise FrozenDirectionRefused(
                "direction artifact carries two entries for one "
                "(candidate, rule, fold) — a re-learn after the freeze is "
                "refused, not averaged or last-wins "
                "(FROZEN_18_PLAN §2.4 '不得重学')\n"
                f"(candidate={entry.candidate!r}, rule={entry.rule!r}, "
                f"fold_test_end={entry.fold_test_end!r})"
            )
        seen[key] = 1


def write_frozen_directions(
    path: str | Path,
    *,
    domain: str,
    decisions: Sequence[Mapping[str, Any]],
    reference: str | None = None,
    frozen: bool = True,
) -> Path:
    """Write a stage-1 ``direction_frozen.json`` (M1/S1 output face).

    ``decisions`` entries carry candidate / rule (letter) /
    fold_test_end / direction / provenance (the full
    ``DirectionDecision.to_json()`` of the rule — Rule A's oriented
    evidence or Rule B's ``label_ic_direction(...).to_json()``).
    Duplicate keys are refused HERE as well: the writer cannot emit a
    re-learned book."""

    _validate_domain(domain)
    spec = _DOMAIN_CONFIG[domain]
    if frozen:
        # validate before writing: duplicates are a re-learn
        probe = {
            "wp18_direction_schema_version": DIRECTION_FROZEN_SCHEMA_VERSION,
            "domain": domain,
            "rules": dict(WP18_RULES),
            "frozen": True,
            "entries": list(decisions),
        }
        _validate_book_payload(probe)
        _refuse_duplicates(_book_entries(probe))
    payload = {
        "wp18_direction_schema_version": DIRECTION_FROZEN_SCHEMA_VERSION,
        "domain": domain,
        "rules": dict(WP18_RULES),
        "reference": reference if reference is not None else spec["reference"],
        "frozen": bool(frozen),
        "entries": [dict(entry) for entry in decisions],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_frozen_directions(
    path: str | Path,
    *,
    expected_keys: Sequence[WP18RowKey] | None = None,
) -> FrozenDirectionBook:
    """Load and validate a frozen direction book (the only consumption
    face of M3/S3/C1/C2).  Fail-closed on: wrong schema version, an
    unfrozen artifact, duplicate/re-learned keys, non-±1 directions and
    — when ``expected_keys`` (the stage-1 row set) is given — any
    missing (single, rule, fold) entry."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenDirectionRefused(
            f"direction artifact {path} cannot be read: {exc}"
        ) from exc
    _validate_book_payload(payload)
    entries = _book_entries(payload)
    _refuse_duplicates(entries)
    book = FrozenDirectionBook(
        domain=str(payload.get("domain", "")),
        rules=payload.get("rules", {}),
        entries=entries,
    )
    if expected_keys is not None:
        missing = sorted(
            {
                (key.candidate, key.rule, key.fold_test_end)
                for key in expected_keys
                if not book.has(key.candidate, key.rule, key.fold_test_end)
            }
        )
        if missing:
            shown = ", ".join(
                f"({c}/{r}/{f})" for c, r, f in missing[:5]
            )
            raise FrozenDirectionRefused(
                f"direction book is incomplete: {len(missing)} stage-1 "
                f"key(s) missing starting with {shown} — the account stage "
                "must not run against a partial freeze (FROZEN_18_PLAN §2.4)"
            )
    return book


def frozen_oriented_signal(
    raw_signal: np.ndarray,
    *,
    book: FrozenDirectionBook,
    candidate: str,
    rule_letter: str,
    fold_test_end: str,
) -> np.ndarray:
    """The account/stress-stage signal face (§2.4): direction x raw
    signal with the direction read from the frozen book ONLY.  There is
    deliberately no fallback and no learner import on this path."""

    direction = book.direction(candidate, rule_letter, fold_test_end)
    signal = np.asarray(raw_signal, dtype=np.float64)
    if not np.isfinite(direction):
        raise FrozenDirectionRefused("non-finite frozen direction")
    if not math.isclose(abs(float(direction)), 1.0):
        raise FrozenDirectionRefused(
            f"frozen direction must be +/-1, got {direction!r}"
        )
    return float(direction) * signal
