"""Import-surface contract for the P7 train.py split (Phase B).

``ashare_model/train.py`` keeps the ``AshareTrainer`` RL core and stays a
compatibility facade while window/sampling helpers move to
``train_windows``.  The parity contract: every name consumers import —
including the monkeypatch surface (``batched_basket_rewards`` patched on
the ``train`` module namespace) — must remain an attribute of
``ashare_model.train``, and moved names must be the *same objects* as the
extracted module's (no second copy, no drift).

Convention (IP-15, mirroring the evaluation facade's registered rule):
new code imports from the ``train_*`` submodule that owns the name —
``train_loop`` / ``train_search_run`` / ``train_artifacts`` /
``train_windows`` — never from the ``train`` facade, so the facade's
re-export body can retire when the registered consumer list drains.  The
registered monkeypatch surface (batched_basket_rewards, score_chunk_size,
logger on the module namespace; ``AshareTrainer.train`` / ``.train_search``
class attributes; ``trainer.vm.execute`` instances) is the t16 registry;
it moves only together with its consumer tests.
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


def test_train_chain_has_no_unregistered_lazy_imports():
    """IP-07a2 (t21): the train chain is clean of accidental lazy imports.
    The ONLY registered ``# noqa: PLC0415`` in the chain is the
    ``_facade_logger`` seam in ``train_loop.py`` — a call-time facade
    import required by the ``train_module.logger`` monkeypatch surface
    (IP-15 registry).  Any other lazy import (e.g. the historical
    baseline_harness cycle break, removed in t21 by re-pointing
    baseline_harness to ``train_windows``) fails this guard."""

    found: list[str] = []
    for module_path in _train_chain_modules():
        rel = module_path.name
        for lineno, line in enumerate(
            module_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "noqa: PLC0415" in line:
                found.append(f"{rel}:{lineno}")
                if rel != "train_loop.py":
                    raise AssertionError(
                        f"unregistered lazy import at {rel}:{lineno}: "
                        f"{line.strip()}"
                    )
    assert len(found) == 1 and found[0].startswith("train_loop.py:"), (
        "train-chain lazy imports must be only the registered "
        f"_facade_logger seam in train_loop.py; found: {found}"
    )


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
