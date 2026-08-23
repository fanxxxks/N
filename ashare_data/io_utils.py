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
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Atomically write ``payload`` as pretty JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
