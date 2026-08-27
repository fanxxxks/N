"""P2-05 contract tests: tier-grouped diagnostics and ablation reports.

Contract (``docs/p2_data_tier_contract.md`` §6): the report covers three
tier sets — A, A+B, all — each with a diagnostics block and an ablation
row; every reported formula carries its data-tier trace; the payload is
versioned (``TIER_REPORT_VERSION``).
"""

from __future__ import annotations

from ashare_data.config import DataConfig, ModelConfig
from ashare_model.data_tier import DataTier, feature_tier, tier_features
from ashare_model.data_loader import AshareDataLoader
from ashare_model.tier_reports import (
    TIER_REPORT_VERSION,
    TIER_SETS,
    assemble_tier_report,
    build_tier_ablation_plan,
    run_tier_diagnostics,
    tier_set_features,
)
from ashare_model.vocab import FEATURE_NAMES


def test_tier_report_version_is_pinned():
    assert TIER_REPORT_VERSION == 1


def test_tier_sets_cover_a_ab_and_all():
    assert TIER_SETS == (("A",), ("A", "B"), ("A", "B", "C"))


def test_tier_set_features_match_the_tier_filter():
    assert set(tier_set_features(("A",))) == set(tier_features((DataTier.A,)))
    assert set(tier_set_features(("A", "B"))) == set(
        tier_features((DataTier.A, DataTier.B))
    )
    assert set(tier_set_features(("A", "B", "C"))) == set(FEATURE_NAMES)


def test_ablation_plan_covers_the_three_tier_sets():
    plan = build_tier_ablation_plan()
    assert [entry["label"] for entry in plan] == ["all", "AB", "A"]
    by_label = {entry["label"]: entry for entry in plan}
    # "all" excludes nothing; AB excludes exactly Tier C; A excludes B+C.
    assert by_label["all"]["excluded_features"] == []
    tier_c = set(tier_features((DataTier.C,)))
    assert set(by_label["AB"]["excluded_features"]) == tier_c
    assert set(by_label["A"]["excluded_features"]) == set(
        tier_features((DataTier.B, DataTier.C))
    )
    # Included tiers are recorded and never contain Tier C for promotion-
    # relevant sets (C only appears in the research "all" set).
    assert by_label["all"]["included_tiers"] == ["A", "B", "C"]
    assert by_label["AB"]["included_tiers"] == ["A", "B"]
    assert by_label["A"]["included_tiers"] == ["A"]
    # Excluding the complement restores the full vocabulary.
    for entry in plan:
        excluded = set(entry["excluded_features"])
        included = {n for n in FEATURE_NAMES if n not in excluded}
        assert included == set(tier_set_features(tuple(entry["included_tiers"])))


def test_assemble_tier_report_payload_is_versioned_and_complete():
    diagnostics = {
        "A": {"feature_count": 3, "per_feature": []},
        "AB": {"feature_count": 5, "per_feature": []},
        "all": {"feature_count": 8, "per_feature": []},
    }
    ablation = {
        "all": {
            "best_reward": 0.5,
            "best_formula": "RET_1",
            "formula_data_tier": {
                "data_tier_version": 1,
                "max_tier": "A",
                "tiers_used": ["A"],
                "per_feature": {"RET_1": "A"},
            },
        },
        "AB": {
            "best_reward": 0.4,
            "best_formula": "ROE",
            "formula_data_tier": {
                "data_tier_version": 1,
                "max_tier": "B",
                "tiers_used": ["B"],
                "per_feature": {"ROE": "B"},
            },
        },
        "A": {
            "best_reward": 0.3,
            "best_formula": "TURNOVER",
            "formula_data_tier": {
                "data_tier_version": 1,
                "max_tier": "A",
                "tiers_used": ["A"],
                "per_feature": {"TURNOVER": "A"},
            },
        },
    }
    payload = assemble_tier_report(
        diagnostics=diagnostics,
        ablation=ablation,
        dataset_id="ds-test",
        steps=8,
        batch_size=4,
        seed=1,
    )
    assert payload["tier_report_version"] == TIER_REPORT_VERSION
    assert payload["dataset_id"] == "ds-test"
    assert set(payload["diagnostics"]) == {"A", "AB", "all"}
    assert set(payload["ablation"]["runs"]) == {"all", "AB", "A"}
    assert payload["ablation"]["budget"] == {
        "steps": 8, "batch_size": 4, "seed": 1
    }
    # Every ablation run carries the formula's data-tier trace.
    for run in payload["ablation"]["runs"].values():
        assert "formula_data_tier" in run
        assert run["formula_data_tier"]["data_tier_version"] >= 1


def test_run_tier_diagnostics_per_set(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    for tier_set in TIER_SETS:
        report = run_tier_diagnostics(loader, "2024-02-01", tier_set)
        names = {row["name"] for row in report["per_feature"]}
        assert names == set(tier_set_features(tier_set)), tier_set
        assert set(report["tiers"]) == set(tier_set)
        for row in report["per_feature"]:
            assert row["data_tier"] in tier_set
