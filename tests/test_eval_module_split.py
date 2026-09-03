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

Retirement conditions (IP-15, 01-A4):

* **Single homes.**  Every frozen name has exactly one owner module,
  registered in ``FACADE_NAME_OWNERS`` and machine-checked by
  ``test_facade_surface_single_home`` (identity, no second copy).
* **New code imports owners, not the facade.**  The convention for all
  new/consolidated code: import from the ``eval_*`` submodule (or the
  version-constant leaf ``ashare_model.versions`` / ``ashare_model.reward``),
  never from ``ashare_model.evaluation``.  Enforced by
  ``test_facade_import_allowlist_only_shrinks``: the production facade
  importer set may only shrink — new imports fail the guard.
* **Monkeypatch surface.**  ``REWARD_VERSION`` pins moved to the reward
  owner (``monkeypatch.setattr(reward, "REWARD_VERSION", ...)``, IP-15);
  ``PROTOCOL_VERSION`` has no patch surface (direct leaf binding).
  ``_build_trainer`` injection (11 sites across test_evaluation /
  test_research_domain / test_stitched_oos) remains facade-bound — the
  registered retirement blocker; migrating it means re-pointing those
  patches to ``eval_search._build_trainer`` (the owner).
* **Retirement target.**  The facade's re-export body may be retired when
  the production importer set is empty; the registered blockers are the
  allowlist below plus the ``_build_trainer`` seam and the runspec owner
  pointer (t21).  Consumers in the allowlist migrate by importing their
  names from the owner modules recorded in ``FACADE_NAME_OWNERS``.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

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


# IP-15: the migration registry — every frozen facade name mapped to the
# single owner module consumers must import from (new code imports owners,
# never the facade).  ``load_trial_rows`` and ``main`` are facade-owned
# until their own extraction.
FACADE_NAME_OWNERS = {
    "PROTOCOL_VERSION": "ashare_model.versions",
    "METRIC_KEYS": "ashare_model.eval_metrics",
    "Fold": "ashare_model.eval_folds",
    "FoldData": "ashare_model.eval_folds",
    "search_window_id": "ashare_model.eval_folds",
    "resolve_folds": "ashare_model.eval_folds",
    "epoch_slice": "ashare_model.eval_folds",
    "evaluate_signal": "ashare_model.eval_metrics",
    "evaluate_formula": "ashare_model.eval_metrics",
    "benchmark_row": "ashare_model.eval_metrics",
    "aggregate_results": "ashare_model.eval_metrics",
    "stitch_oos_series": "ashare_model.eval_metrics",
    "stitched_metrics": "ashare_model.eval_metrics",
    "top_trial": "ashare_model.eval_metrics",
    "norm_ppf": "ashare_model.eval_corrections",
    "norm_cdf": "ashare_model.eval_corrections",
    "psr": "ashare_model.eval_corrections",
    "expected_max_sr": "ashare_model.eval_corrections",
    "deflated_sharpe": "ashare_model.eval_corrections",
    "dsr_from_rows": "ashare_model.eval_corrections",
    "max_t_from_rows": "ashare_model.eval_corrections",
    "selfcheck_rows": "ashare_model.eval_corrections",
    "baseline_candidates": "ashare_model.eval_search",
    "run_fold": "ashare_model.eval_search",
    "run_random_search": "ashare_model.eval_search",
    "run_gp_search": "ashare_model.eval_search",
    "run_tpe_search": "ashare_model.eval_search",
    "_build_trainer": "ashare_model.eval_search",
    "build_result": "ashare_model.eval_artifacts",
    "run_protocol": "ashare_model.evaluation",
    "_tradable_ic_mask": "ashare_model.eval_metrics",
    "load_trial_rows": "ashare_model.evaluation",
    "main": "ashare_model.evaluation",
    "CandidateScorer": "ashare_model.candidates",
    "REWARD_VERSION": "ashare_model.reward",
}


def test_facade_name_owner_registry_covers_surface():
    """IP-15: the owner registry must cover exactly the frozen surface —
    a name added to FACADE_SURFACE without an owner is a registration
    gap, not a convention exception."""

    assert set(FACADE_NAME_OWNERS) == set(FACADE_SURFACE)


def test_facade_surface_single_home():
    """IP-15: every frozen name is the SAME object as its registered
    owner module's attribute (identity, no second copy, no drift) — the
    machine anchor for the consumer migration list: a consumer migrating
    off the facade imports from FACADE_NAME_OWNERS[name] and observes
    identical behavior."""

    evaluation = importlib.import_module("ashare_model.evaluation")
    owner_modules = {}
    for name, owner in FACADE_NAME_OWNERS.items():
        if owner not in owner_modules:
            owner_modules[owner] = importlib.import_module(owner)
        assert getattr(evaluation, name) is getattr(owner_modules[owner], name), (
            f"facade name {name!r} drifted from its owner {owner}"
        )


# IP-15: production modules still importing the evaluation facade.  The
# set may only shrink: each entry is a registered consumer with a migration
# target (FACADE_NAME_OWNERS); new facade imports are forbidden by
# test_facade_import_allowlist_only_shrinks.
FACADE_IMPORT_ALLOWLIST = (
    "ashare_model/artifact_versions.py",  # PROTOCOL_VERSION -> versions
    "ashare_model/bare_factor_backtest.py",  # PROTOCOL_VERSION -> versions
    "ashare_model/eval_search.py",  # _build_trainer seam (registered blocker)
    "ashare_model/promotion.py",  # PROTOCOL_VERSION -> versions
    "ashare_model/research_doctor.py",  # PROTOCOL_VERSION -> versions
    "scripts/admission_experiment.py",  # PROTOCOL_VERSION/evaluate_formula/resolve_folds
    "scripts/baseline_harness.py",  # PROTOCOL_VERSION/evaluate_signal/resolve_folds
)

_FACADE_IMPORT_PATTERNS = (
    "from .evaluation import",
    "from ashare_model.evaluation import",
    "from ashare_model import evaluation",
)


def _production_facade_importers() -> set[str]:
    root = Path(importlib.import_module("ashare_model").__file__).parent.parent
    importers: set[str] = set()
    scan_roots = [root / "ashare_model", root / "ashare_trading", root / "scripts", root / "webapi"]
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        for py in scan_root.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            rel = py.relative_to(root).as_posix()
            if any(pattern in text for pattern in _FACADE_IMPORT_PATTERNS):
                importers.add(rel)
    return importers


def test_facade_import_allowlist_only_shrinks():
    """IP-15: no new production module may import the evaluation facade.
    The allowlist above is the registered consumer migration list; when it
    drains to empty (and the ``_build_trainer`` seam migrates to
    ``eval_search``), the facade's re-export body can be retired."""

    importers = _production_facade_importers()
    unexpected = sorted(importers - set(FACADE_IMPORT_ALLOWLIST))
    assert not unexpected, (
        "new evaluation-facade imports are forbidden (IP-15: import from "
        f"the owner modules instead): {unexpected}"
    )
    fully_migrated = sorted(set(FACADE_IMPORT_ALLOWLIST) - importers)
    if fully_migrated:
        print(
            "facade consumers fully migrated (allowlist may shrink): "
            f"{fully_migrated}"
        )


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


def test_eval_corrections_reexport_identity():
    """A3: statistical-correction names moved to eval_corrections; the
    facade re-exports the same objects (no second copy, no drift)."""
    evaluation = importlib.import_module("ashare_model.evaluation")
    eval_corrections = importlib.import_module("ashare_model.eval_corrections")
    for name in (
        "norm_ppf",
        "norm_cdf",
        "psr",
        "expected_max_sr",
        "deflated_sharpe",
        "dsr_from_rows",
        "max_t_from_rows",
        "selfcheck_rows",
    ):
        assert getattr(evaluation, name) is getattr(eval_corrections, name), name


def test_eval_search_reexport_identity():
    """A4: search-backend runners moved to eval_search; the facade
    re-exports the same objects (no second copy, no drift)."""
    evaluation = importlib.import_module("ashare_model.evaluation")
    eval_search = importlib.import_module("ashare_model.eval_search")
    for name in (
        "_build_trainer",
        "baseline_candidates",
        "run_fold",
        "run_random_search",
        "run_gp_search",
        "run_tpe_search",
    ):
        assert getattr(evaluation, name) is getattr(eval_search, name), name


def test_eval_artifacts_reexport_identity():
    """A5: artifact-assembly names moved to eval_artifacts; the facade
    re-exports the same objects (no second copy, no drift)."""
    evaluation = importlib.import_module("ashare_model.evaluation")
    eval_artifacts = importlib.import_module("ashare_model.eval_artifacts")
    for name in (
        "_run_recorded",
        "build_result",
        "universe_policy_payload",
    ):
        assert getattr(evaluation, name) is getattr(eval_artifacts, name), name


def test_protocol_version_single_home_is_versions_module():
    """IP-07a: ``PROTOCOL_VERSION``'s single home is the leaf module
    ``ashare_model.versions``; the evaluation facade re-exports the same
    object (no second copy, no drift) so the frozen facade surface and
    every ``from .evaluation import PROTOCOL_VERSION`` consumer
    (artifact_versions, bare_factor_backtest, promotion, research_doctor)
    keep working unchanged."""
    evaluation = importlib.import_module("ashare_model.evaluation")
    versions = importlib.import_module("ashare_model.versions")
    assert versions.PROTOCOL_VERSION == "25"
    assert evaluation.PROTOCOL_VERSION is versions.PROTOCOL_VERSION


def test_train_binds_protocol_version_from_versions_at_import_time():
    """IP-07a (updated by IP-07b): the train module chain binds
    ``PROTOCOL_VERSION`` at import time from the leaf versions module —
    no chain module binds it through the evaluation facade, and no chain
    module carries a lazy ``from .evaluation import`` so the
    train⇄evaluation import edge stays broken.  After IP-07b the
    constant's consumers live in ``train_loop`` / ``train_artifacts`` /
    ``train_search_run`` (the facade itself no longer needs it)."""
    import re
    from pathlib import Path

    versions = importlib.import_module("ashare_model.versions")
    chain = (
        "ashare_model.train",
        "ashare_model.train_loop",
        "ashare_model.train_artifacts",
        "ashare_model.train_search_run",
    )
    for name in chain:
        module = importlib.import_module(name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert not re.search(r"from \.evaluation import", source), (
            f"{name} reintroduced a lazy evaluation import"
        )
        if name != "ashare_model.train":
            assert module.PROTOCOL_VERSION is versions.PROTOCOL_VERSION, name


def test_versions_module_imports_without_train_or_evaluation():
    """IP-07a: ``ashare_model.versions`` is the cycle-break point for the
    train⇄evaluation edge: importing it standalone in a fresh interpreter
    must not import ``ashare_model.train`` or ``ashare_model.evaluation``."""
    code = (
        "import sys\n"
        "import ashare_model.versions\n"
        "for name in (\"ashare_model.train\", \"ashare_model.evaluation\"):\n"
        "    assert name not in sys.modules, name\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
