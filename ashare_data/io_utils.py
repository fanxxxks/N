"""Defensive JSON file I/O shared across the project.

Single implementation for two recurring concerns (previously duplicated
in portfolio/manager/run_sim/webapi/dashboard):

* :func:`read_json_safe` — a missing or corrupt file degrades to ``None``
  with a logged warning instead of crashing a read-only consumer (the
  dashboard crashed on bad artifacts while the web API defended the same
  reads — two standards for one operation);
* :func:`atomic_write_json` — write to a sibling temp file and
  ``os.replace`` so a crash mid-save can never leave a truncated state
  file behind.

This module is also the single durability authority for evidence files
(:func:`durable_write_text` / :func:`durable_append_line`): every write
that claims to be evidence flushes and ``os.fsync``-es its bytes to the
disk before the operation is considered complete, because
``os.replace``/append alone are atomic (or detectable) but **not
durable** — after a power loss a renamed file can surface with its new
name while its bytes never reached the platter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


def read_json_safe(path: str | Path) -> dict | list | None:
    """Load a JSON file, returning ``None`` when missing or corrupt."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        logger.warning(f"Could not read JSON {path}: {exc}")
        return None
    return payload if isinstance(payload, (dict, list)) else None


def durable_write_text(
    path: str | Path, text: str, *, tmp_name: str | None = None
) -> None:
    """Write ``text`` to ``path`` durably and atomically.

    The payload lands in a sibling temp file that is flushed and
    ``os.fsync``-ed **before** ``os.replace`` publishes it: the rename is
    atomic on NTFS but not durable, so without the fsync a power loss
    after the replace can expose a file with the new name whose bytes
    never reached the disk (or roll the rename back). ``tmp_name``
    overrides the temp file name (kept in the same directory) for callers
    with their own collision-safe naming scheme; the default
    ``<name>.tmp-<pid>-<uuid8>`` is multi-process safe.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = (
        path.with_name(tmp_name)
        if tmp_name is not None
        else path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid4().hex[:8]}")
    )
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def durable_append_line(path: str | Path, line: str) -> None:
    """Append one newline-terminated ``line`` and fsync it before returning.

    Append-only evidence (hash-chained ledgers, lifecycle and lineage
    event chains) must survive a power loss at line granularity: the
    write is flushed and ``os.fsync``-ed while the handle is open, so an
    append that returned is on disk. A crash *mid-line* still leaves a
    detectable partial line — the fail-closed reload paths treat it as
    corruption — but a completed append can no longer silently vanish.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically write ``payload`` as pretty JSON (durably: the temp file
    is fsync-ed before ``os.replace`` — see :func:`durable_write_text`)."""

    path = Path(path)
    durable_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
        tmp_name=path.with_suffix(".tmp.json").name,
    )
