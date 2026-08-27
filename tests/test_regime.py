"""T4-01 contract tests: the data-regime registry.

Phase-4 rule: history 2021-2026 has been repeatedly viewed, so it is
development/validation data — never a "final holdout".  The next true
final evaluation may only consume **future** data (after the declared
``dev_cutoff``) or a **strictly locked** slice that no experiment may
touch.  These tests pin that contract before the implementation exists:

* window classification (dev / locked_holdout / future);
* any protocol fold that touches a locked slice is rejected up front
  (train and test alike — the fold's train window starts at the
  beginning of history, so a fold reaching the lock consumes it);
* a locked slice declared on one dataset id is meaningless on another;
* a final evaluation must use future or locked data only;
* the registry persists atomically and a missing registry means "no
  regime declared" (``None``, never a fabricated default).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_data.config import FoldConfig
from ashare_model.regime import (
    DataRegime,
    HoldoutViolation,
    LockedSlice,
    RegimeRegistry,
    declare_dev_regime,
    lock_slice,
)

DEV_CUTOFF = "2026-12-31"


def _regime() -> DataRegime:
    return declare_dev_regime(DEV_CUTOFF)


def test_classify_dev_future_and_locked(tmp_path):
    regime = lock_slice(
        _regime(),
        start="2027-06-01",
        end="2027-12-31",
        dataset_id="ds-1",
        note="final holdout",
    )
    registry = RegimeRegistry(tmp_path / "registry.json", regime)
    assert registry.classify_window("2024-01-01", "2025-12-31") == "dev"
    assert registry.classify_window("2027-01-01", "2027-03-31") == "future"
    assert registry.classify_window("2027-06-01", "2027-12-31") == "locked_holdout"
    assert registry.classify_window("2027-06-15", "2027-07-01") == "locked_holdout"
    # Boundary: the locked start day itself is locked (closed interval).
    assert registry.classify_window("2026-12-31", "2027-06-01") == "locked_holdout"


def test_window_ending_at_dev_cutoff_is_dev(tmp_path):
    registry = RegimeRegistry(tmp_path / "registry.json", _regime())
    assert registry.classify_window("2026-01-01", "2026-12-31") == "dev"


def test_no_regime_declared_classifies_as_dev(tmp_path):
    # No regime file: legacy behavior — everything is dev/validation.
    registry = RegimeRegistry(tmp_path / "missing.json")
    assert registry.regime is None
    assert registry.classify_window("2027-01-01", "2027-12-31") == "dev"


def test_folds_before_locked_slice_pass(tmp_path):
    locked = lock_slice(
        _regime(), start="2027-01-01", end="2027-12-31", dataset_id="ds-1"
    )
    registry = RegimeRegistry(tmp_path / "registry.json", locked)
    folds = [
        FoldConfig("2024-12-31", "2025-12-31"),
        FoldConfig("2025-12-31", "2026-12-31"),
    ]
    registry.assert_folds_clear(folds, dataset_id="ds-1")  # no raise


def test_fold_test_window_touching_lock_rejected(tmp_path):
    locked = lock_slice(
        _regime(), start="2027-01-01", end="2027-12-31", dataset_id="ds-1"
    )
    registry = RegimeRegistry(tmp_path / "registry.json", locked)
    with pytest.raises(HoldoutViolation):
        registry.assert_folds_clear(
            [FoldConfig("2026-12-31", "2027-06-30")], dataset_id="ds-1"
        )


def test_fold_training_on_locked_data_rejected(tmp_path):
    # Train windows start at the beginning of history: a fold whose
    # training end reaches the lock consumes locked data even if its test
    # window stays clear.
    locked = lock_slice(
        _regime(), start="2027-01-01", end="2027-12-31", dataset_id="ds-1"
    )
    registry = RegimeRegistry(tmp_path / "registry.json", locked)
    with pytest.raises(HoldoutViolation):
        registry.assert_folds_clear(
            [FoldConfig("2027-03-31", "2027-12-31")], dataset_id="ds-1"
        )
    with pytest.raises(HoldoutViolation):
        registry.assert_folds_clear(
            [FoldConfig("2027-06-30", "2028-06-30")], dataset_id="ds-1"
        )


def test_locked_slice_binds_to_dataset_id(tmp_path):
    locked = lock_slice(
        _regime(), start="2027-01-01", end="2027-12-31", dataset_id="ds-1"
    )
    registry = RegimeRegistry(tmp_path / "registry.json", locked)
    # The lock was declared on ds-1; the current database is ds-2, so the
    # lock is unverifiable and no run may proceed on this data.
    with pytest.raises(HoldoutViolation):
        registry.assert_folds_clear(
            [FoldConfig("2025-12-31", "2026-12-31")], dataset_id="ds-2"
        )
    # A lock without a dataset id cannot be checked against the database.
    anonymous = RegimeRegistry(
        tmp_path / "registry2.json",
        lock_slice(_regime(), start="2027-01-01", end="2027-12-31"),
    )
    anonymous.assert_folds_clear(
        [FoldConfig("2025-12-31", "2026-12-31")], dataset_id="anything"
    )  # no raise


def test_final_evaluation_requires_future_or_locked(tmp_path):
    registry = RegimeRegistry(tmp_path / "registry.json", _regime())
    with pytest.raises(HoldoutViolation):
        registry.assert_final_evaluation(
            [FoldConfig("2025-12-31", "2026-12-31")]
        )
    # Future data: allowed.
    registry.assert_final_evaluation([FoldConfig("2026-12-31", "2027-12-31")])
    # Locked slice: allowed as the final evaluation window.
    locked = RegimeRegistry(
        tmp_path / "registry2.json",
        lock_slice(_regime(), start="2027-01-01", end="2027-12-31", dataset_id="ds"),
    )
    locked.assert_final_evaluation([FoldConfig("2026-12-31", "2027-06-30")])


def test_registry_save_load_roundtrip(tmp_path):
    regime = lock_slice(
        _regime(),
        start="2027-01-01",
        end="2027-12-31",
        dataset_id="ds-1",
        note="strictly locked final window",
    )
    registry = RegimeRegistry(tmp_path / "registry.json", regime)
    registry.save()

    loaded = RegimeRegistry(tmp_path / "registry.json")
    assert loaded.regime is not None
    assert loaded.regime.dev_cutoff == DEV_CUTOFF
    assert loaded.regime.policy == "locked_slice"
    assert loaded.regime.locked_slice == LockedSlice(
        start="2027-01-01",
        end="2027-12-31",
        dataset_id="ds-1",
        locked_at=regime.locked_slice.locked_at,
        note="strictly locked final window",
    )
    assert loaded.classify_window("2027-06-01", "2027-06-30") == "locked_holdout"


def test_save_writes_single_json_file(tmp_path):
    registry = RegimeRegistry(tmp_path / "registry.json", _regime())
    registry.save()
    payload = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert payload["dev_cutoff"] == DEV_CUTOFF
    assert payload["policy"] == "future_only"
    assert payload["locked_slice"] is None


def test_load_missing_registry_returns_none(tmp_path):
    assert RegimeRegistry(tmp_path / "nope.json").regime is None


def test_lock_rejects_invalid_window():
    with pytest.raises(ValueError):
        lock_slice(_regime(), start="2027-12-31", end="2027-01-01")
    with pytest.raises(ValueError):
        lock_slice(_regime(), start="2026-12-31", end="2027-12-31")  # overlaps dev


def test_cli_declares_and_locks(tmp_path):
    import subprocess
    import sys

    registry_path = tmp_path / "holdout_registry.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ashare_model.regime",
            "--registry",
            str(registry_path),
            "--dev-cutoff",
            "2026-12-31",
            "--lock",
            "2027-01-01",
            "2027-12-31",
            "--note",
            "final",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    loaded = RegimeRegistry(registry_path)
    assert loaded.regime is not None
    assert loaded.regime.policy == "locked_slice"
    assert loaded.regime.locked_slice is not None
    assert loaded.regime.locked_slice.note == "final"


def test_cli_refuses_to_overwrite_lock_without_force(tmp_path):
    import subprocess
    import sys

    registry_path = tmp_path / "holdout_registry.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ashare_model.regime",
            "--registry",
            str(registry_path),
            "--dev-cutoff",
            "2026-12-31",
            "--lock",
            "2027-01-01",
            "2027-12-31",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "ashare_model.regime",
            "--registry",
            str(registry_path),
            "--dev-cutoff",
            "2026-12-31",
            "--lock",
            "2028-01-01",
            "2028-12-31",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert second.returncode != 0
