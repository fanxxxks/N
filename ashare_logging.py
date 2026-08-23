"""Run logging helpers for AlphaGPT.

The project already uses :mod:`loguru` for console logging. This module adds
two conveniences needed for reproducible runs and test sessions:

* an in-memory sink that keeps the latest formatted records;
* :func:`export_log_txt` which writes those records to a plain ``.txt`` file.

Call :func:`setup_run_logging` at the start of a real run or a pytest session
and :func:`export_log_txt` at the end.  The default log directory is
``<project>/logs``.
"""

from __future__ import annotations

import sys
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
    log_path = log_dir / f"{run_name}_{timestamp}.log"

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
    path.write_text(content, encoding="utf-8")
    logger.success(f"Logs exported to {path}")
    return path
