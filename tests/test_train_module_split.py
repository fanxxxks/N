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

import ast
import importlib
from pathlib import Path

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


_FACADE_LINE_BUDGET = 300
_METHOD_LINE_BUDGET = 100


def _train_chain_modules() -> list[Path]:
    train = importlib.import_module("ashare_model.train")
    package_dir = Path(train.__file__).parent
    return [Path(train.__file__)] + sorted(package_dir.glob("train_*.py"))


def test_train_artifacts_reexport_identity():
    """B2 (IP-07b): artifact persistence moved by reason-to-change to
    ``train_artifacts``; the facade re-exports the same function object
    (no second copy, no drift) and ``_write_artifact`` stays on the
    facade class as the stable patch seam."""

    train = importlib.import_module("ashare_model.train")
    train_artifacts = importlib.import_module("ashare_model.train_artifacts")
    assert (
        train.write_trainer_artifact is train_artifacts.write_trainer_artifact
    )
    assert hasattr(train.AshareTrainer, "_write_artifact")


def test_train_facade_line_budget():
    """IP-07b acceptance: ``train.py`` becomes a compatibility facade under
    300 lines, and no function or method anywhere in the train module
    chain (``train.py`` plus ``train_*.py``) spans more than 100 lines —
    the P7 evaluation-split maintainability contract applied to the RL
    trainer."""

    for module_path in _train_chain_modules():
        if str(module_path).endswith("train.py"):
            line_count = len(module_path.read_text(encoding="utf-8").splitlines())
            assert line_count < _FACADE_LINE_BUDGET, (
                f"train.py facade has {line_count} lines "
                f"(budget {_FACADE_LINE_BUDGET})"
            )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                assert length <= _METHOD_LINE_BUDGET, (
                    f"{module_path.name}:{node.name} spans {length} lines "
                    f"(budget {_METHOD_LINE_BUDGET})"
                )
