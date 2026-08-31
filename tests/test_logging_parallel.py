"""Parallel-session logging contracts (pytest-xdist readiness, PR3/B2+C2).

conftest's session-scoped ``_run_logging`` fixture executes once **per
xdist worker**. With the historical naming scheme (1-second-resolution
timestamp, no worker component) every worker starting in the same second
opened the *same* ``.log`` file, and every worker exported its buffer to
the *same* ``logs/pytest.txt`` — races documented by the 2026-08-31
test-runtime audit (ashare_logging in-process deque, 1s timestamps,
non-atomic ``write_text``).

The contracts under test:

1. the worker id from ``PYTEST_XDIST_WORKER`` must disambiguate log file
   names (per-worker ``.log`` files);
2. a ``worker_log_suffix`` helper exposes that component (single
   implementation, consumed by the conftest export path);
3. ``export_log_txt`` must be crash-safe: the target file is replaced
   atomically, so a failure during export leaves the previous content
   intact and no temporary file behind.

Serial (no xdist) behavior must stay byte-compatible: no suffix, same
default paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_worker_log_suffix_reflects_xdist_worker_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ashare_logging import worker_log_suffix

    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    assert worker_log_suffix() == ""

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    assert worker_log_suffix() == "_gw3"


def test_setup_run_logging_names_log_file_per_xdist_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ashare_logging import setup_run_logging

    worker_dir = tmp_path / "worker"
    serial_dir = tmp_path / "serial"

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    setup_run_logging(log_dir=worker_dir, run_name="pytest", reset=True)
    assert list(worker_dir.glob("pytest_gw3_*.log")), (
        "worker run must open a worker-suffixed .log file"
    )
    assert not list(worker_dir.glob("pytest_2*.log")), (
        "worker run must not open the controller (timestamp-only) .log file"
    )

    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    setup_run_logging(log_dir=serial_dir, run_name="pytest", reset=True)
    assert list(serial_dir.glob("pytest_2*.log")), (
        "serial run must keep the historical naming scheme"
    )
    assert not list(serial_dir.glob("pytest_gw*.log")), (
        "serial run must not create worker-suffixed files"
    )


def test_export_log_txt_crash_leaves_target_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash during export must never corrupt the target file.

    The injection simulates a partial write followed by a crash on every
    ``Path.write_text`` call. The contract: the *target* keeps its
    previous (good) content, and no temporary file survives the failure.
    """

    from ashare_logging import export_log_txt, setup_run_logging

    setup_run_logging(log_dir=tmp_path, run_name="run", reset=True)
    target = tmp_path / "out.txt"
    export_log_txt(path=target)
    good = target.read_text(encoding="utf-8")

    real_write_text = Path.write_text

    def failing_write_text(self: Path, data: str, *args: object, **kwargs: object) -> int:
        real_write_text(self, "partial-garbage", encoding="utf-8")
        raise OSError("simulated crash mid-export")

    monkeypatch.setattr(Path, "write_text", failing_write_text)
    with pytest.raises(OSError):
        export_log_txt(path=target)
    monkeypatch.undo()

    assert target.read_text(encoding="utf-8") == good, (
        "crashed export must not corrupt the target (atomic replace required)"
    )
    assert not list(tmp_path.glob("*.tmp")), "failed export must clean up temp files"
