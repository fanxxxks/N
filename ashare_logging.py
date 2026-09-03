"""Run logging helpers for AlphaGPT.

The project already uses :mod:`loguru` for console logging. This module adds
conveniences needed for reproducible runs and test sessions:

* an in-memory sink that keeps the latest formatted records;
* :func:`export_log_txt` which writes those records to a plain ``.txt`` file;
* the critical-path run-identity header (IP-11 / AGENTS §4.5):
  :func:`emit_run_identity` prints the ``run_id`` / git commit / config
  sha256 / version-set quadruple as one fixed-format line through the same
  pipeline — never a second telemetry path.

Call :func:`setup_run_logging` at the start of a real run or a pytest session
and :func:`export_log_txt` at the end.  The default log directory is
``<project>/logs``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Mapping
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# Bounded memory buffer: an unbounded sink would grow without limit across
# long training/sync runs.
_LOG_LINES: deque[str] = deque(maxlen=10_000)

_TXT_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{line} | "
    "{message}"
)


def _memory_sink(message: Any) -> None:
    """Append a formatted loguru message to the in-memory buffer."""

    _LOG_LINES.append(str(message).rstrip())


def worker_log_suffix() -> str:
    """File-name component isolating per-xdist-worker log files.

    pytest-xdist sets ``PYTEST_XDIST_WORKER`` (e.g. ``gw0``) in every
    worker process. Without this component, workers starting in the same
    second would share one ``.log`` file and race on one export target
    (test-runtime audit 2026-08-31). Serial runs get an empty suffix, so
    the historical naming scheme is preserved byte-for-byte.
    """

    worker = os.environ.get("PYTEST_XDIST_WORKER")
    return f"_{worker}" if worker else ""


def setup_run_logging(
    log_dir: str | Path | None = None,
    run_name: str = "run",
    level: str = "DEBUG",
    reset: bool = True,
) -> Any:
    """Configure console, memory, and file logging.

    Parameters
    ----------
    log_dir:
        Directory for the persistent ``.log`` file.  Defaults to
        ``Path(__file__).parent / "logs"``.
    run_name:
        Prefix for the generated log file.
    level:
        Minimum level captured to the file and memory sinks.
    reset:
        If ``True`` (the default), remove existing loguru handlers and clear
        the in-memory buffer first.
    """

    if reset:
        logger.remove()
        _LOG_LINES.clear()

    root = Path(__file__).resolve().parent
    log_dir = Path(log_dir) if log_dir is not None else root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}{worker_log_suffix()}_{timestamp}.log"

    logger.add(sys.stderr, level=level, colorize=True)
    logger.add(_memory_sink, level=level, format=_TXT_FORMAT)
    logger.add(
        log_path,
        level=level,
        format=_TXT_FORMAT,
        encoding="utf-8",
        rotation="10 MB",
        retention=14,
    )
    logger.debug(f"Run logging configured: {log_path}")
    return logger


def get_log_text() -> str:
    """Return the current in-memory log buffer as plain text."""

    return "\n".join(_LOG_LINES)


def new_log_run_id() -> str:
    """Log correlation id for critical-path entries below ``ashare_model``.

    Same shape as ``ashare_model.runspec.new_run_id()`` (``uuid4().hex``)
    but defined here because ``ashare_data`` must not import
    ``ashare_model`` (AGENTS §9 dependency direction; IP-12 removes the
    existing reverse edge).  This is a log-correlation identity only:
    formal research run identities stay owned by
    ``ashare_model.runspec`` (P8) and no artifact consumes this value.
    """

    return uuid.uuid4().hex


def git_commit(project_root: str | Path | None = None) -> str | None:
    """Current ``HEAD`` commit of the repository at ``project_root``.

    The capture point for run-identity headers (IP-11).  Returns ``None``
    when git is unavailable or the directory is not a repository — the
    header renders that as ``unknown`` instead of guessing.  Artifact
    manifests keep their own provenance capture; consolidating those into
    this helper is a separate mechanical task, deliberately not done here.
    """

    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parent
    )
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    return commit or None


def canonical_config_sha256(config: Any) -> str:
    """SHA-256 over the canonical JSON of an effective config.

    Dataclasses are expanded via ``dataclasses.asdict``; values JSON cannot
    represent (e.g. ``Path``) render through ``str``.  ``sort_keys`` plus
    compact separators make the hash order-independent and stable, so
    equivalent representations of the same effective config hash equal.
    """

    payload = (
        dataclasses.asdict(config)
        if dataclasses.is_dataclass(config) and not isinstance(config, type)
        else config
    )
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def emit_run_identity(
    *,
    run_id: str,
    config_sha256: str | None = None,
    versions: Mapping[str, str] | None = None,
    commit: str | None = None,
    project_root: str | Path | None = None,
) -> str:
    """Emit the critical-path identity header (AGENTS §4.5, IP-11/F-07).

    One fixed-format ``INFO`` line through the standard loguru pipeline::

        run identity: run_id=<id> git_commit=<sha|unknown> \
config_sha256=<hex|none> versions=<sorted-compact-json>

    ``commit`` defaults to :func:`git_commit` captured at emission time.
    Returns the emitted message so callers and contract tests can assert
    on it.
    """

    if commit is None:
        commit = git_commit(project_root)
    rendered_versions = json.dumps(
        dict(versions or {}), sort_keys=True, separators=(",", ":"), default=str
    )
    line = (
        f"run identity: run_id={run_id} "
        f"git_commit={commit or 'unknown'} "
        f"config_sha256={config_sha256 or 'none'} "
        f"versions={rendered_versions}"
    )
    logger.info(line)
    return line


def export_log_txt(
    path: str | Path | None = None,
    log_dir: str | Path | None = None,
    run_name: str = "run",
) -> Path:
    """Write the captured logs to a ``.txt`` file.

    If ``path`` is omitted a timestamped file is created under ``log_dir``
    (or the project ``logs`` directory by default).
    """

    if path is None:
        root = Path(__file__).resolve().parent
        log_dir = Path(log_dir) if log_dir is not None else root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"{run_name}_{timestamp}.txt"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = get_log_text()
    if content:
        content += "\n"
    # Atomic export (test-runtime audit 2026-08-31): concurrent or failed
    # exports must never corrupt the target — write a temp file, replace.
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.success(f"Logs exported to {path}")
    return path
