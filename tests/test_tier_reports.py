"""P2-05 contract tests: tier-grouped diagnostics and ablation reports.

Contract (``docs/p2_data_tier_contract.md`` §6): the report covers three
tier sets — A, A+B, all — each with a diagnostics block and an ablation
row; every reported formula carries its data-tier trace; the payload is
versioned (``TIER_REPORT_VERSION``).
"""

from __future__ import annotations

from ashare_data.config import DataConfig, ModelConfig
from ashare_model.candidates import CandidateScore, SelectionResult
from ashare_model.data_tier import DataTier, feature_tier, tier_features
from ashare_model.data_loader import AshareDataLoader
from ashare_model.tier_reports import (
    TIER_REPORT_VERSION,
    TIER_SETS,
    assemble_tier_report,
    best_confined_candidate,
    build_tier_ablation_plan,
    run_tier_diagnostics,
    tier_set_features,
)
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB


def _tok(name: str) -> int:
    return FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name)


def _candidate(tokens, val_reward: float, *, eligible: bool) -> CandidateScore:
    return CandidateScore(
        tokens=tuple(tokens) if tokens is not None else None,
        candidate_id=str(val_reward),
        formula_text="f",
        source="test",
        direction=1,
        val_reward=val_reward,
        val_icir=0.1,
        train_reward=val_reward,
        train_icir=0.1,
        complexity_penalty=0.0,
        eligible=eligible,
        rejection_reasons=() if eligible else ("gate",),
    )


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
            "confined": True,
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
            "confined": True,
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
            "confined": True,
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
        assert "confined" in run


def test_best_confined_candidate_filters_by_tier():
    """Contract §6: the fallback candidate is the best eligible one whose
    tokens reference only the tier set's features."""
    ret_1 = [_tok("RET_1")]
    roe = [_tok("ROE")]
    both = [_tok("RET_1"), _tok("ROE"), FORMULA_VOCAB.operator_offset]
    selection = SelectionResult(
        selected=None,
        best_rejected=None,
        candidates=(
            _candidate(ret_1, 0.5, eligible=True),
            _candidate(roe, 0.9, eligible=True),          # out of tier set
            _candidate(ret_1, 1.0, eligible=False),       # ineligible
            _candidate(both, 0.7, eligible=True),         # mixed tiers
            _candidate([999999], 0.8, eligible=True),     # undecodable
        ),
    )
    tier_a = {n for n in FEATURE_NAMES if feature_tier(n) is DataTier.A}
    best = best_confined_candidate(selection, tier_a)
    assert best is not None
    assert best.val_reward == 0.5  # the only eligible Tier-A-only candidate
    assert tuple(best.tokens) == tuple(ret_1)

    # No confined candidate -> None (caller falls back with confined=false).
    assert best_confined_candidate(selection, {"RET_5"}) is None
    assert best_confined_candidate(selection, set()) is None


def test_confined_selection_prefers_the_pipeline_selection():
    """Contract §6: the reported formula is the pipeline's selected
    candidate when it stays inside the tier set; the best confined
    candidate only replaces an out-of-set selection."""
    from ashare_model.tier_reports import _candidate_within_tier

    tier_a = {n for n in FEATURE_NAMES if feature_tier(n) is DataTier.A}
    in_set = _candidate([_tok("RET_1")], 0.5, eligible=True)
    out_set = _candidate([_tok("ROE")], 0.9, eligible=True)
    assert _candidate_within_tier(in_set, tier_a) is True
    assert _candidate_within_tier(out_set, tier_a) is False
    assert _candidate_within_tier(None, tier_a) is False
    assert _candidate_within_tier(_candidate(None, 0.0, eligible=True), tier_a) is False


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

def test_candidate_within_tier_grammar5_dispatch():
    """t56 F1 (Plan A): the tier-confinement decode threads the declared
    grammar generation -- grammar-5-era tokens (the P10 seed-7 formula)
    decode via the frozen grammar-5 layout instead of the shifted live
    feature block; without the declaration the decode fails closed."""
    import json as _json

    from pathlib import Path

    from ashare_model.tier_reports import _candidate_within_tier

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

    import types

    score = types.SimpleNamespace(tokens=tokens, eligible=True)

    allowed = {
        "TURNOVER",
        "ATR_14",
        "LIMIT_UP_CNT_5",
        "CROWD_AMOUNT_60",
        "MARGIN_CROWD_60",
    }
    assert _candidate_within_tier(score, allowed, grammar_version=5) is True
    # Without the declared generation the shifted live layout misdecodes
    # the operator/EOS ids -> fail closed.
    assert _candidate_within_tier(score, allowed) is False
