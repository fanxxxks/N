"""P49 slow-route RED/GREEN tests (WP 13, erratum 1).

The slow round's fail-closed refusal exposed a structural precondition:
the p34 §5.2 check-6 leave-family-out ablation cannot produce subsets for
a SINGLE-family factor set (all five slow signals map to 'fundamental').
Erratum 1 (prospective, preregistered BEFORE any economic result existed
- the run died at factor qualification with zero OOS rows): a
single-family set falls back to leave-one-SIGNAL-out within the family,
same machinery, same hash discipline; a set of ONE signal stays vacuous
and the driver keeps its refusal.
"""

from __future__ import annotations

import pytest

from ashare_model.r01_baseline import (
    R01_REPORT_SCHEMA_VERSION,
    ablation_subsets,
)


def test_multi_family_set_leaves_one_family_out():
    subsets = ablation_subsets(["MOMENTUM_20", "RSQ_60", "ROE"])
    # families: momentum (MOMENTUM_20), risk (RSQ_60), fundamental (ROE)
    labels = sorted(label for label, _ in subsets)
    assert labels == ["fundamental", "momentum", "risk"]
    by_label = dict(subsets)
    assert by_label["momentum"] == ["RSQ_60", "ROE"]
    assert by_label["risk"] == ["MOMENTUM_20", "ROE"]
    assert by_label["fundamental"] == ["MOMENTUM_20", "RSQ_60"]


def test_single_family_set_falls_back_to_leave_one_signal_out():
    names = ["ROE", "PE_TTM", "REVENUE_YOY", "DIVIDEND_YIELD", "MARKET_CAP"]
    subsets = ablation_subsets(names)
    labels = sorted(label for label, _ in subsets)
    assert labels == sorted(
        f"fundamental:{name}" for name in names
    )
    by_label = dict(subsets)
    assert by_label["fundamental:ROE"] == [
        "PE_TTM", "REVENUE_YOY", "DIVIDEND_YIELD", "MARKET_CAP",
    ]
    # every subset keeps four of the five signals.
    for _, subset in subsets:
        assert len(subset) == 4


def test_single_signal_set_stays_vacuous():
    # One signal in one family has no non-empty subset under either rule:
    # the driver's fail-closed refusal stands (never a hash over nothing).
    assert ablation_subsets(["ROE"]) == []


def test_mixed_family_with_industry_uses_family_rule():
    subsets = ablation_subsets(["ROE", "PE_TTM", "IND_REL_RET_20"])
    labels = sorted(label for label, _ in subsets)
    # Two families -> the family rule applies (no per-signal fallback).
    assert labels == ["fundamental", "industry"]
    by_label = dict(subsets)
    assert by_label["fundamental"] == ["IND_REL_RET_20"]
    assert by_label["industry"] == ["ROE", "PE_TTM"]


def test_schema_version_bumped_for_erratum_1():
    # The ablation-evidence semantics change for single-family sets rides
    # a schema bump (p34 -> 3): consumers can discriminate old/new reports.
    assert R01_REPORT_SCHEMA_VERSION == 3
