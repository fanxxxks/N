"""Read-only lock <-> environment drift audit (IP-04, review finding 03-F-01).

``requirements.lock`` is the frozen closure captured via ``importlib.metadata``
from the interpreter that last ran ``scripts/freeze_lock.py`` — currently a
clean CI-faithful venv, NOT a live snapshot of any given development machine
(see ``docs/test_runtime_measurement_log.md`` for the regeneration
provenance).  Nothing in CI or the test suite compares that closure against
the installed environment of a working machine, so version drift accumulates
silently.  This tool makes the drift visible: it prints a per-package diff
report between ``requirements.lock`` and the environment of the running
interpreter and writes nothing.

This is a reporting tool, not a repair path.  Never align an environment with
``pip install -r requirements.lock`` wholesale: the torch pin in the lock is
the ``+cpu`` wheel (P0-06 CPU/CUDA split) and a wholesale installation would
override a locally installed CUDA torch (observed with ``2.11.0+cu128``).
Environment alignment happens per package through individually authorized
commands.

Usage:
    python scripts/lock_audit.py    # print the lock<->env diff report

Exit codes:
    0  no actionable drift (torch local-tag differences are by design)
    1  drift reported (version mismatches, lock-only, env-only, or unparsed
       lock lines)
    2  ``requirements.lock`` missing under the repo root
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = "requirements.lock"

_PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==([^=]+)$")

# P0-06: the only package whose local wheel tag is machine-specific by
# design (base pins carry +cpu, GPU machines install +cuNN).
_TORCH = "torch"

_DIRECTION_LOCK_NEWER = "lock_newer"
_DIRECTION_ENV_NEWER = "env_newer"
_DIRECTION_DIFF = "diff"


def canon_name(name: str) -> str:
    """PEP 503 canonical project name (lock and dist-info spellings agree)."""

    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock_lines(lines: list[str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Parse ``name==version`` pins; fail closed on unparsable lines.

    Returns ``({canonical_name: (display_name, version)}, [unparsed lines])``;
    comments and pip directives (``--``) are skipped, anything else that does
    not match an exact pin is reported back instead of silently dropped.
    """

    pins: dict[str, tuple[str, str]] = {}
    unparsed: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = _PIN_RE.match(line)
        if match is None:
            unparsed.append(line)
            continue
        name, version = match.group(1), match.group(3)
        pins[canon_name(name)] = (name, version)
    return pins, unparsed


def installed_map() -> dict[str, tuple[str, str]]:
    """Installed distributions of the running interpreter (read-only)."""

    installed: dict[str, tuple[str, str]] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            installed[canon_name(name)] = (name, dist.version)
    return installed


def _direction(lock_version: str, env_version: str) -> str:
    """Best-effort PEP 440 ordering; ``diff`` when versions are uncomparable."""

    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return _DIRECTION_DIFF
    try:
        lock_parsed = Version(lock_version)
        env_parsed = Version(env_version)
    except InvalidVersion:
        return _DIRECTION_DIFF
    if lock_parsed > env_parsed:
        return _DIRECTION_LOCK_NEWER
    if env_parsed > lock_parsed:
        return _DIRECTION_ENV_NEWER
    return _DIRECTION_DIFF


@dataclass(frozen=True)
class Mismatch:
    """One package present on both sides with differing pinned versions."""

    package: str
    lock_version: str
    env_version: str
    direction: str


@dataclass(frozen=True)
class LockAuditReport:
    """Structured lock<->environment diff (``lock_audit`` contract)."""

    mismatches: tuple[Mismatch, ...]
    lock_only: tuple[str, ...]
    env_only: tuple[str, ...]
    torch_local_tag_diffs: tuple[tuple[str, str, str], ...]
    unparsed_lock_lines: tuple[str, ...]
    lock_packages: int
    env_packages: int

    @property
    def has_drift(self) -> bool:
        return bool(
            self.mismatches
            or self.lock_only
            or self.env_only
            or self.unparsed_lock_lines
        )


def diff_lock_env(
    lock: dict[str, tuple[str, str]],
    env: dict[str, tuple[str, str]],
    unparsed: tuple[str, ...] = (),
) -> LockAuditReport:
    """Diff a lock pin map against an installed-environment pin map."""

    mismatches: list[Mismatch] = []
    torch_local_tag_diffs: list[tuple[str, str, str]] = []
    for key in sorted(set(lock) & set(env)):
        lock_name, lock_version = lock[key]
        _env_name, env_version = env[key]
        if key == _TORCH:
            lock_base = lock_version.split("+", 1)[0]
            env_base = env_version.split("+", 1)[0]
            if lock_base == env_base:
                if lock_version != env_version:
                    torch_local_tag_diffs.append((lock_name, lock_version, env_version))
                continue
        if lock_version == env_version:
            continue
        mismatches.append(
            Mismatch(
                package=lock_name,
                lock_version=lock_version,
                env_version=env_version,
                direction=_direction(lock_version, env_version),
            )
        )
    return LockAuditReport(
        mismatches=tuple(mismatches),
        lock_only=tuple(lock[key][0] for key in sorted(set(lock) - set(env))),
        env_only=tuple(env[key][0] for key in sorted(set(env) - set(lock))),
        torch_local_tag_diffs=tuple(torch_local_tag_diffs),
        unparsed_lock_lines=tuple(unparsed),
        lock_packages=len(lock),
        env_packages=len(env),
    )


def render(report: LockAuditReport) -> str:
    """Human-readable report text (stdout payload of ``main``)."""

    lines = [
        "lock audit: requirements.lock vs installed environment "
        "(read-only; alignment is a separately authorized operation)",
        f"lock packages: {report.lock_packages}  installed: {report.env_packages}",
    ]
    sections: list[tuple[str, list[str]]] = [
        (
            f"VERSION MISMATCH ({len(report.mismatches)})",
            [
                f"  {m.package}: lock={m.lock_version} env={m.env_version} "
                f"[{m.direction}]"
                for m in report.mismatches
            ],
        ),
        ("LOCK-ONLY (in lock, not installed)", [f"  {n}" for n in report.lock_only]),
        ("ENV-ONLY (installed, not in lock)", [f"  {n}" for n in report.env_only]),
        (
            "TORCH LOCAL WHEEL TAG DIFFERS (P0-06 by design, not drift)",
            [
                f"  {name}: lock={lock_version} env={env_version}"
                for name, lock_version, env_version in report.torch_local_tag_diffs
            ],
        ),
        (
            f"UNPARSED LOCK LINES ({len(report.unparsed_lock_lines)})",
            [f"  {line}" for line in report.unparsed_lock_lines],
        ),
    ]
    for title, rows in sections:
        if title.startswith("TORCH") and not rows:
            continue
        lines.append(f"{title}:")
        lines.extend(rows if rows else ["  (none)"])
    if report.has_drift:
        lines.append(
            "result: DRIFT - align per package via authorized commands; never "
            "`pip install -r requirements.lock` wholesale (the +cpu torch pin "
            "would override a CUDA torch)."
        )
    else:
        lines.append("result: IN SYNC (no actionable drift)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the audit against ``ROOT/requirements.lock``; print, never write."""

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    parser.parse_args(argv)

    lock_path = ROOT / LOCK_FILE
    if not lock_path.exists():
        print(f"lock audit: missing {LOCK_FILE} under {ROOT}", file=sys.stderr)
        return 2
    pins, unparsed = parse_lock_lines(lock_path.read_text(encoding="utf-8").splitlines())
    report = diff_lock_env(pins, installed_map(), unparsed=tuple(unparsed))
    print(render(report))
    return 1 if report.has_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
