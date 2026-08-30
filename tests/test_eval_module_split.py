"""Import-surface contract for the P7 evaluation.py split (Phase A).

``ashare_model/evaluation.py`` is decomposed by reason-to-change into
``eval_folds`` / ``eval_metrics`` / ``eval_corrections`` / ``eval_search`` /
``eval_artifacts`` while the original module stays a compatibility facade.
The parity contract: every name that consumers import today — production
code, scripts and tests, including the monkeypatch surface
(``CandidateScorer``, ``REWARD_VERSION``, ``_build_trainer``) — must remain
an attribute of ``ashare_model.evaluation``.

The frozen surface below is captured from the pre-split module (A0) and is
the arbitration source for the split: a name disappearing from the facade
is a regression in the split, never a signal to edit this list.  Each
extraction PR additionally asserts facade/extracted-module object identity
for the names it moves, so re-exports can never drift into a second copy.
"""

from __future__ import annotations

import importlib

# Names imported by production modules, scripts and tests before the split
# (see tests/test_evaluation.py, test_stitched_oos.py, test_universe.py,
# test_research_domain.py, test_promotion.py, ashare_model/promotion.py,
# scripts/baseline_harness.py, scripts/admission_experiment.py), plus the
# monkeypatch surface used via attribute access.
FACADE_SURFACE = (
    # Version constants
    "PROTOCOL_VERSION",
    "METRIC_KEYS",
    # Folds / windows
    "Fold",
    "FoldData",
    "search_window_id",
    "resolve_folds",
    "epoch_slice",
    # Signal metrics / stitching
    "evaluate_signal",
    "evaluate_formula",
    "benchmark_row",
    "aggregate_results",
    "stitch_oos_series",
    "stitched_metrics",
    "top_trial",
    # Search backends
    "baseline_candidates",
    "run_fold",
    "run_random_search",
    "run_gp_search",
    "run_tpe_search",
    # Statistical corrections
    "norm_ppf",
    "norm_cdf",
    "psr",
    "expected_max_sr",
    "deflated_sharpe",
    "dsr_from_rows",
    "max_t_from_rows",
    "selfcheck_rows",
    # Artifacts / orchestration
    "build_result",
    "run_protocol",
    "load_trial_rows",
    "main",
    # Test-used private helpers
    "_tradable_ic_mask",
    "_build_trainer",
    # Monkeypatch surface (imported names living on the facade)
    "CandidateScorer",
    "REWARD_VERSION",
)


def test_facade_surface_complete():
    evaluation = importlib.import_module("ashare_model.evaluation")
    missing = [name for name in FACADE_SURFACE if not hasattr(evaluation, name)]
    assert not missing, f"facade lost names: {missing}"


def test_eval_folds_reexport_identity():
    """A1: fold/window names moved to eval_folds; the facade re-exports the
    same objects (no second copy, no drift)."""
    evaluation = importlib.import_module("ashare_model.evaluation")
    eval_folds = importlib.import_module("ashare_model.eval_folds")
    for name in (
        "Fold",
        "FoldData",
        "search_window_id",
        "resolve_folds",
        "epoch_slice",
    ):
        assert getattr(evaluation, name) is getattr(eval_folds, name), name


def test_eval_metrics_reexport_identity():
    """A2: metric/stitching names moved to eval_metrics; the facade
    re-exports the same objects (no second copy, no drift)."""
    evaluation = importlib.import_module("ashare_model.evaluation")
    eval_metrics = importlib.import_module("ashare_model.eval_metrics")
    for name in (
        "METRIC_KEYS",
        "_tradable_ic_mask",
        "evaluate_signal",
        "evaluate_formula",
        "benchmark_row",
        "aggregate_results",
        "stitch_oos_series",
        "stitched_metrics",
        "top_trial",
    ):
        assert getattr(evaluation, name) is getattr(eval_metrics, name), name
