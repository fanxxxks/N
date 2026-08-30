"""Import-surface contract for the P7 train.py split (Phase B).

``ashare_model/train.py`` keeps the ``AshareTrainer`` RL core and stays a
compatibility facade while window/sampling helpers move to
``train_windows``.  The parity contract: every name consumers import —
including the monkeypatch surface (``batched_basket_rewards`` patched on
the ``train`` module namespace) — must remain an attribute of
``ashare_model.train``, and moved names must be the *same objects* as the
extracted module's (no second copy, no drift).
"""

from __future__ import annotations

import importlib

# Names imported by production modules, scripts and tests before the split
# (see ashare_model/eval_search.py, baseline_harness.py, searcher_bench.py,
# tier_reports.py, scripts/, tests/test_train.py, test_grammar.py,
# test_universe.py, test_research_domain.py), plus the monkeypatch surface
# used via attribute access on the train module.
FACADE_SURFACE = (
    # Trainer core (stays in the facade)
    "AshareTrainer",
    "main",
    # Window / sampling helpers (moved to train_windows in B1)
    "_TrainWindow",
    "_project_root",
    "validation_start",
    "validation_windows",
    "sample_random_formulas",
    "resolve_device",
    # Monkeypatch surface (imported names living on the facade)
    "batched_basket_rewards",
)


def test_facade_surface_complete():
    train = importlib.import_module("ashare_model.train")
    missing = [name for name in FACADE_SURFACE if not hasattr(train, name)]
    assert not missing, f"train facade lost names: {missing}"


def test_train_windows_reexport_identity():
    """B1: window/sampling names moved to train_windows; the facade
    re-exports the same objects (no second copy, no drift)."""
    train = importlib.import_module("ashare_model.train")
    train_windows = importlib.import_module("ashare_model.train_windows")
    for name in (
        "_TrainWindow",
        "_project_root",
        "validation_start",
        "validation_windows",
        "sample_random_formulas",
        "resolve_device",
    ):
        assert getattr(train, name) is getattr(train_windows, name), name
