"""STOP-signal file helpers — one implementation for manager, runner, web UI.

The stop path crosses three modules (job manager writes it, the daily
runner polls and acknowledges it, the dashboard and the reset flow clear
it); all of them previously re-implemented the same file protocol.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

_STOP_CONTENTS = {"", "STOP", "STOPPED"}


def request_stop(path: str | Path) -> None:
    """Write the STOP request; OSError degrades to a warning."""

    path = Path(path)
    try:
        path.write_text("STOP", encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not write stop signal {path}: {exc}")


def stop_requested(path: str | Path) -> bool:
    """Runner-side poll: True when the file carries a stop request.

    Acknowledges by rewriting the file to ``STOPPED`` so a crash between
    the poll and the shutdown still leaves a terminal marker; an unreadable
    file counts as a request (fail safe).
    """

    path = Path(path)
    if not path.exists():
        return False
    try:
        content = path.read_text(encoding="utf-8").strip().upper()
    except OSError:
        return True
    if content not in _STOP_CONTENTS:
        return False
    logger.warning(f"STOP signal received from {path}. Simulation will stop.")
    try:
        path.write_text("STOPPED", encoding="utf-8")
    except OSError:
        pass
    return True


def clear_stop_signal(path: str | Path) -> None:
    """Remove a leftover STOP/STOPPED file so an explicit restart can run."""

    path = Path(path)
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"Cleared stop signal: {path}")
    except OSError as exc:
        logger.warning(f"Could not clear stop signal file {path}: {exc}")
