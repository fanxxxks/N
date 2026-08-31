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
    assert DATA_TIER_VERSION == 1


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
    assert tier_b == set(FUNDAMENTAL_PIT_NAMES) | {
        "MARGIN_BALANCE_CHG",
        "MARGIN_CROWD_60",
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
    assert "04-30" in TIER_TIME_RULES["B"] and "08-31" in TIER_TIME_RULES["B"]
    assert "快照" in TIER_TIME_RULES["C"]
