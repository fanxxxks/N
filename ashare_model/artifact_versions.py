"""Legacy-artifact classification and stamping (P0-04).

Pre-Phase-0 artifacts — the reward-v10 strategy
(``data/best_ashare_strategy.json``) and the protocol-v12 result
(``data/protocol_result.json``) — are older than the current code
generation and must never be mistaken for the current champion.  This
module is the single source of the classification rules:

* :func:`classify_strategy` / :func:`classify_protocol` — pure rules over
  a payload: an artifact is legacy when any recorded version/provenance
  field differs from or predates the current code generation;
* :func:`stamp_legacy_artifacts` — the migration: rewrites legacy local
  artifacts in place, adding ``legacy: true`` plus a human-readable
  ``legacy_reason`` list and a stamp timestamp (idempotent; current
  artifacts are left untouched);
* consumers read the stamp: the research doctor reports it, the web API
  refuses to present an unmarked old artifact as current, and the
  backtest/simulation entry points log a loud warning.

Migration/refusal policy: old artifacts are never deleted or converted —
they stay readable and archived, but every consumer that would treat them
as champion evidence either refuses (promotion, protocol checks) or flags
them legacy (stamp + API + doctor).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ashare_data.io_utils import atomic_write_json, read_json_safe

from .alphagpt import MODEL_VERSION
from .evaluation import PROTOCOL_VERSION
from .reward import REWARD_VERSION

# Stamp field names — the contract every consumer reads.
LEGACY_FIELD = "legacy"
LEGACY_REASON_FIELD = "legacy_reason"
LEGACY_STAMPED_AT_FIELD = "legacy_stamped_at"

_ARTIFACTS = (
    ("strategy", "best_ashare_strategy.json"),
    ("protocol", "protocol_result.json"),
)


def _version_mismatch(field: str, recorded: Any, current: str) -> str | None:
    if recorded is None:
        return None
    if str(recorded) != current:
        return f"{field} {recorded} != current {current}"
    return None


def classify_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy verdict for a strategy artifact payload (pure)."""

    reasons: list[str] = []
    if "searcher" not in payload:
        reasons.append("no searcher field (pre-T2-03)")
    mismatch = _version_mismatch("reward_version", payload.get("reward_version"), REWARD_VERSION)
    if mismatch:
        reasons.append(mismatch)
    if "protocol_version" not in payload:
        reasons.append("no protocol_version (pre-T2-01)")
    if "model_version" not in payload:
        reasons.append("no model_version (pre-T2-03)")
    if "dataset_id" not in payload:
        reasons.append("no dataset_id (pre-T1-01)")
    return {"legacy": bool(reasons), "reasons": reasons}


def classify_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy verdict for a protocol-result artifact payload (pure)."""

    reasons: list[str] = []
    mismatch = _version_mismatch("protocol_version", payload.get("protocol_version"), PROTOCOL_VERSION)
    if mismatch:
        reasons.append(mismatch)
    mismatch = _version_mismatch("reward_version", payload.get("reward_version"), REWARD_VERSION)
    if mismatch:
        reasons.append(mismatch)
    if "dataset_id" not in payload:
        reasons.append("no dataset_id (pre-T1-01)")
    if "stitched" not in payload:
        reasons.append("no stitched OOS block (pre-T4-01)")
    if "ledger" not in payload:
        reasons.append("no ledger (pre-T4-01)")
    return {"legacy": bool(reasons), "reasons": reasons}


def classify_artifact(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the per-kind classifier (unknown kinds are never legacy)."""

    if name == "strategy":
        return classify_strategy(payload)
    if name == "protocol":
        return classify_protocol(payload)
    return {"legacy": False, "reasons": []}


def stamp_legacy_artifacts(data_dir: Path) -> list[dict[str, Any]]:
    """Mark legacy strategy/protocol artifacts in ``data_dir``.

    For every existing artifact that classifies as legacy and is not
    already stamped, adds ``legacy`` / ``legacy_reason`` /
    ``legacy_stamped_at`` in place (atomic write).  Idempotent: already
    stamped or current artifacts are left untouched.  Returns one record
    per artifact with the outcome.
    """

    data_dir = Path(data_dir)
    outcomes: list[dict[str, Any]] = []
    for name, filename in _ARTIFACTS:
        path = data_dir / filename
        outcome: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "stamped": False,
            "legacy": False,
            "reasons": [],
        }
        payload = read_json_safe(path)
        if not isinstance(payload, dict):
            outcomes.append(outcome)
            continue
        if payload.get(LEGACY_FIELD) is True:
            outcome["legacy"] = True
            outcome["reasons"] = list(payload.get(LEGACY_REASON_FIELD) or [])
            outcomes.append(outcome)
            continue
        verdict = classify_artifact(name, payload)
        if not verdict["legacy"]:
            outcomes.append(outcome)
            continue
        payload[LEGACY_FIELD] = True
        payload[LEGACY_REASON_FIELD] = verdict["reasons"]
        payload[LEGACY_STAMPED_AT_FIELD] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        atomic_write_json(path, payload)
        outcome.update(stamped=True, legacy=True, reasons=verdict["reasons"])
        outcomes.append(outcome)
    return outcomes
