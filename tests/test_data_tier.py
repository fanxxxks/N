"""P2 contract tests: free-data credibility tiers (A / B / C).

Contract source: ``docs/p2_data_tier_contract.md``.  Every feature of the
vocabulary maps to exactly one credibility tier (the weakest data source
it consumes), every tier carries usable-time rules, and any formula can be
traced back to the tiers of the features it uses.  Tier A must be
independent of fundamentals, current-industry snapshots and placeholder
data; Tier C is research-display only and can never enter promotion.
"""

from __future__ import annotations

import pytest

from ashare_data.capital_flow import EXTERNAL_FACTOR_NAMES
from ashare_data.fundamentals import FUNDAMENTAL_PIT_NAMES
from ashare_model.data_tier import (
    DATA_TIER_VERSION,
    TIER_TIME_RULES,
    DataTier,
    feature_tier,
    formula_data_tier_report,
    formula_feature_names,
    tier_features,
    tier_of_pit_level,
)
from ashare_model.factors import BAR_COLUMNS, FACTOR_REGISTRY, NEUTRAL_FEATURE_NAMES
from ashare_model.feature_registry import PitLevel
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB


def _tok(name: str) -> int:
    return FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name)


def _tier_set(*tiers: DataTier) -> set[str]:
    return {
        name for name in FEATURE_NAMES if feature_tier(name) in set(tiers)
    }


def test_data_tier_version_is_pinned():
    # p17 §9 (D6): bumped 1 -> 2 — the fundamentals B-rule switched to
    # revision-tracking semantics and the industry-membership B-rule
    # records the D5 gate state.
    assert DATA_TIER_VERSION == 2


def test_every_feature_has_a_data_tier():
    for name in FEATURE_NAMES:
        assert feature_tier(name) in DataTier, name


def test_tier_mapping_follows_the_contract():
    # Tier A: price / volume / turnover / listing age (local daily bars).
    for name in ("RET_1", "TURNOVER", "VOLUME_RATIO", "LIST_AGE", "MARKET_CAP"):
        assert feature_tier(name) is DataTier.A, name
    # Tier B: margin balance and conservative-disclosure fundamentals.
    for name in ("PE_TTM", "ROE", "DIVIDEND_YIELD", "MARGIN_BALANCE_CHG"):
        assert feature_tier(name) is DataTier.B, name
    # Tier C: current industry snapshot / placeholder data.
    for name in ("INDUSTRY_MOMENTUM", "IND_REL_RET_5", "IND_REL_TURNOVER",
                 "NORTHBOUND_CHG"):
        assert feature_tier(name) is DataTier.C, name


def test_pit_level_mapping_is_exhaustive_and_contractual():
    assert tier_of_pit_level(PitLevel.PIT_DAILY) is DataTier.A
    assert tier_of_pit_level(PitLevel.PIT_FUNDAMENTAL) is DataTier.B
    assert tier_of_pit_level(PitLevel.PIT_CAPITAL) is DataTier.B
    assert tier_of_pit_level(PitLevel.SNAPSHOT) is DataTier.C
    assert tier_of_pit_level(PitLevel.NEUTRAL) is DataTier.C
    # Every PitLevel is covered (no tierless feature can hide).
    assert {tier_of_pit_level(level) for level in PitLevel} == set(DataTier)


def test_tier_a_is_independent_of_fundamentals_industry_and_placeholders():
    """Acceptance: Tier A experiments must not depend on fundamentals,
    current industry or historical ST approximation."""
    tier_a = _tier_set(DataTier.A)
    assert not (tier_a & set(FUNDAMENTAL_PIT_NAMES))
    assert "MARGIN_BALANCE_CHG" not in tier_a
    assert "INDUSTRY_MOMENTUM" not in tier_a
    assert not (tier_a & {n for n in FEATURE_NAMES if n.startswith("IND_REL_")})
    assert not (tier_a & NEUTRAL_FEATURE_NAMES)
    # Every Tier A feature is a local daily-bar computation: registered with
    # required columns drawn only from the daily bars.
    for name in tier_a:
        spec = FACTOR_REGISTRY[name][0]
        assert set(spec.required_columns) <= set(BAR_COLUMNS), name


def test_tier_sets_partition_the_vocabulary():
    tier_b = _tier_set(DataTier.B)
    tier_c = _tier_set(DataTier.C)
    # P9 §5.4 (whitelist §10.1 case 2, contract APPROVED): MARGIN_CROWD_60
    # rides the same PIT margin feed as MARGIN_BALANCE_CHG -> Tier B.
    # p24 (whitelist §10.1 case 2, user-approved F2 batch): the two
    # margin-crowding factors ride the same feed -> Tier B.
    assert tier_b == set(FUNDAMENTAL_PIT_NAMES) | {
        "MARGIN_BALANCE_CHG",
        "MARGIN_CROWD_60",
        "MARGIN_BALANCE_TO_CIRC_MV",
        "MARGIN_BUY_PRESSURE_5",
    }
    assert tier_c == {n for n in FEATURE_NAMES if n.startswith("IND_REL_")} | {
        "INDUSTRY_MOMENTUM",
        "NORTHBOUND_CHG",
    }
    assert len(tier_a := _tier_set(DataTier.A)) > 0
    assert (tier_a | tier_b | tier_c) == set(FEATURE_NAMES)
    assert not (tier_a & tier_b) and not (tier_b & tier_c)


def test_tier_strength_ordering_for_max_tier():
    # max_tier is the weakest tier used (C weakest, A strongest).
    assert DataTier.weakest([DataTier.A, DataTier.B]) is DataTier.B
    assert DataTier.weakest([DataTier.B, DataTier.C]) is DataTier.C
    assert DataTier.weakest([DataTier.A, DataTier.C]) is DataTier.C
    assert DataTier.weakest([DataTier.A]) is DataTier.A


def test_formula_feature_names_via_ir():
    tokens = [_tok("RET_1"), _tok("ROE"), FORMULA_VOCAB.operator_offset]  # ADD
    assert formula_feature_names(tokens) == ["RET_1", "ROE"]


def test_formula_data_tier_report_traces_every_feature():
    tokens = [_tok("RET_1"), _tok("ROE"), FORMULA_VOCAB.operator_offset]
    report = formula_data_tier_report(tokens=tokens)
    assert report is not None
    assert report["max_tier"] == "B"
    assert set(report["tiers_used"]) == {"A", "B"}
    assert report["per_feature"] == {"RET_1": "A", "ROE": "B"}
    assert report["data_tier_version"] == DATA_TIER_VERSION


def test_formula_data_tier_report_weakest_tier_wins():
    tokens = [_tok("RET_1"), _tok("INDUSTRY_MOMENTUM"),
              FORMULA_VOCAB.operator_offset]
    report = formula_data_tier_report(tokens=tokens)
    assert report["max_tier"] == "C"
    assert set(report["tiers_used"]) == {"A", "C"}


def test_formula_data_tier_report_bare_feature_name():
    # Baseline rows carry formula_text but no tokens (formula=None).
    report = formula_data_tier_report(feature_name="ROE")
    assert report is not None
    assert report["max_tier"] == "B"
    assert report["per_feature"] == {"ROE": "B"}
    # A non-feature text (e.g. "equal_weight") has nothing to trace.
    assert formula_data_tier_report(feature_name="equal_weight") is None
    assert formula_data_tier_report() is None


def test_formula_data_tier_report_invalid_tokens_is_none():
    assert formula_data_tier_report(tokens=[999999]) is None


def test_formula_grammar5_tokens_decode_via_declared_generation():
    """t56 F1 (Plan A): a grammar-5-era bare token list (the P10 seed-7
    formula, tokens read from the git-tracked P10 summary) decodes against
    the FROZEN grammar-5 layout when the artifact declares
    grammar_version=5; without the declaration the shifted live layout
    fails closed (docs/p13_fundamental_fields_contract.md §6.1/§6.2
    erratum)."""
    import json as _json

    from pathlib import Path

    summary_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "p10_searcher_comparison_20260901"
        / "summary.json"
    )
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    row = next(
        r
        for r in summary["rows"]
        if r["seed"] == 7 and r["searcher"] == "random"
    )
    tokens = list(row["selected"]["tokens"])

    expected = [
        "ATR_14",
        "CROWD_AMOUNT_60",
        "LIMIT_UP_CNT_5",
        "MARGIN_CROWD_60",
        "TURNOVER",
    ]
    assert formula_feature_names(tokens, grammar_version=5) == expected
    report = formula_data_tier_report(tokens=tokens, grammar_version=5)
    assert report is not None
    assert report["per_feature"]["MARGIN_CROWD_60"] == "B"
    assert report["per_feature"]["LIMIT_UP_CNT_5"] == "A"
    # Without the declared generation the shifted live layout fails closed.
    assert formula_feature_names(tokens) is None
    assert formula_data_tier_report(tokens=tokens) is None


def test_formula_grammar2_3_4_tier_decode_uses_frozen_layouts():
    """P1-A: the data_tier decode consumes the SAME single grammar ->
    vocabulary authority as vocab.resolve_formula_tokens (t57 F2裁定:
    grammar 2/3 -> the 62-feature grammar-3 layout, 4/5 -> the 73-feature
    grammar-5 layout).  The old data_tier branch fell back to the live
    vocabulary for 2/3/4, misdecoding or rejecting legal legacy formulas:
    under the grammar-3 layout token 63 is ADD and EOS is 102, while under
    the live layout 63 is the feature IND_REL_RET_60 and 102 is TS_RANK5."""
    # grammar 2/3: RET_1 RET_5 ADD EOS (operator offset 63, EOS 102).
    g3_tokens = [1, 2, 63, 102]
    for generation in (2, 3):
        names = formula_feature_names(g3_tokens, grammar_version=generation)
        assert names == ["RET_1", "RET_5"], generation
        report = formula_data_tier_report(tokens=g3_tokens, grammar_version=generation)
        assert report is not None and report["max_tier"] == "A", generation
    # grammar 4 (shares grammar 5's feature block): IND_REL_RET_60 RET_1
    # ADD EOS (operator offset 74, EOS 113).  The Tier-C IND_REL_ feature
    # is only visible through the frozen layout; a live-layout decode
    # reads token 74 as the feature CASHFLOW_QUALITY (Tier B) instead.
    g4_tokens = [63, 1, 74, 113]
    assert formula_feature_names(g4_tokens, grammar_version=4) == [
        "IND_REL_RET_60",
        "RET_1",
    ]
    report = formula_data_tier_report(tokens=g4_tokens, grammar_version=4)
    assert report == {
        "data_tier_version": DATA_TIER_VERSION,
        "max_tier": "C",
        "tiers_used": ["A", "C"],
        "per_feature": {"IND_REL_RET_60": "C", "RET_1": "A"},
    }
    # 对照 vocab 裁定: the same payload resolved through
    # resolve_formula_tokens yields the same feature names.
    from ashare_model.vocab import resolve_formula_tokens

    resolved = resolve_formula_tokens(
        {"formula": g4_tokens, "grammar_version": 4}
    )
    assert formula_feature_names(resolved) == ["IND_REL_RET_60", "RET_1"]


def test_formula_grammar6_tier_decode_uses_frozen_layout():
    """P1-A: a legal grammar-6 bare payload ([1, 1, 78, 117] = RET_1 RET_1
    ADD EOS under the frozen 77-feature layout) is tier-traceable; the old
    live-layout decode read 78 as CFO_TO_ASSETS and 117 as CORR60, leaving
    the stack unterminated and the artifact untraceable (None)."""
    tokens = [1, 1, 78, 117]
    assert formula_feature_names(tokens, grammar_version=6) == ["RET_1"]
    report = formula_data_tier_report(tokens=tokens, grammar_version=6)
    assert report is not None
    assert report["per_feature"] == {"RET_1": "A"}
    # grammar 7 shares grammar 6's layout (sampling-set change only).
    assert formula_feature_names(tokens, grammar_version=7) == ["RET_1"]


def test_formula_tier_decode_fails_closed_on_unknown_declared_generation():
    """P1-A: a declared-but-unknown generation must never silently decode
    against the live table (the t28 mechanism); it fails closed to "no
    formula" just like resolve_formula_tokens rejects it."""
    for generation in (1, 10, "bogus"):
        assert formula_feature_names([1, 2], grammar_version=generation) is None
        assert (
            formula_data_tier_report(tokens=[1, 2], grammar_version=generation) is None
        )


def test_tier_features_filters_by_tier_set():
    assert set(tier_features((DataTier.A,))) == _tier_set(DataTier.A)
    assert set(tier_features((DataTier.A, DataTier.B))) == (
        _tier_set(DataTier.A) | _tier_set(DataTier.B)
    )
    assert set(tier_features((DataTier.A, DataTier.B, DataTier.C))) == (
        set(FEATURE_NAMES)
    )


def test_every_tier_has_usable_time_rules():
    assert set(TIER_TIME_RULES) == {t.value for t in DataTier}
    for tier, rule in TIER_TIME_RULES.items():
        assert isinstance(rule, str) and rule.strip(), tier
    # The rules must encode the availability semantics of the contract:
    # A is known at close and usable from the next session; B fundamentals
    # are gated by the statutory disclosure-season end; C is snapshot-only
    # and never usable for promotion decisions.
    assert "次日" in TIER_TIME_RULES["A"]
    assert "修订" in TIER_TIME_RULES["B"] and "次一开市会话" in TIER_TIME_RULES["B"]
    assert "快照" in TIER_TIME_RULES["C"]
