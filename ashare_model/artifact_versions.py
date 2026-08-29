"""Legacy-artifact classification and stamping (P0-04).

Pre-P3 artifacts — including the reward-v10 strategy
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
  backtest/simulation entry points log a loud warning. Fixed bare-factor
  artifacts are classified and stamped by the same rules.

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
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import (
    EXECUTION_SPEC_VERSION,
    validate_portfolio_config_provenance,
)

# Stamp field names — the contract every consumer reads.
LEGACY_FIELD = "legacy"
LEGACY_REASON_FIELD = "legacy_reason"
LEGACY_STAMPED_AT_FIELD = "legacy_stamped_at"

_ARTIFACTS = (
    ("strategy", "best_ashare_strategy.json"),
    ("protocol", "protocol_result.json"),
    ("bare_factor", "bare_factor_backtest.json"),
)


def _version_mismatch(field: str, recorded: Any, current: Any) -> str | None:
    if recorded is None:
        return None
    if str(recorded) != str(current):
        return f"{field} {recorded} != current {current}"
    return None


def _required_version_reason(
    payload: dict[str, Any],
    field: str,
    current: Any,
    missing_reason: str,
) -> str | None:
    if payload.get(field) is None:
        return missing_reason
    return _version_mismatch(field, payload.get(field), current)


def _required_execution_reasons(payload: dict[str, Any]) -> list[str]:
    """P3 execution/provenance fields shared by current artifacts."""

    reasons: list[str] = []
    for field, current, missing_reason in (
        (
            "execution_version",
            EXECUTION_SPEC_VERSION,
            "no execution_version (pre-P3)",
        ),
        (
            "portfolio_constructor_version",
            PORTFOLIO_CONSTRUCTOR_VERSION,
            "no portfolio_constructor_version (pre-P3)",
        ),
    ):
        reason = _required_version_reason(
            payload, field, current, missing_reason
        )
        if reason:
            reasons.append(reason)
    if payload.get("portfolio_config") is None:
        reasons.append("no portfolio_config (pre-P3)")
    else:
        try:
            validate_portfolio_config_provenance(payload.get("portfolio_config"))
        except ValueError as exc:
            reasons.append(str(exc))
    return reasons


def classify_strategy(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy verdict for a strategy artifact payload (pure)."""

    reasons: list[str] = []
    if "searcher" not in payload:
        reasons.append("no searcher field (pre-T2-03)")
    mismatch = _required_version_reason(
        payload,
        "reward_version",
        REWARD_VERSION,
        "no reward_version (pre-P3)",
    )
    if mismatch:
        reasons.append(mismatch)
    mismatch = _required_version_reason(
        payload,
        "protocol_version",
        PROTOCOL_VERSION,
        "no protocol_version (pre-T2-01)",
    )
    if mismatch:
        reasons.append(mismatch)
    if "model_version" not in payload:
        reasons.append("no model_version (pre-T2-03)")
    if "dataset_id" not in payload:
        reasons.append("no dataset_id (pre-T1-01)")
    reasons.extend(_required_execution_reasons(payload))
    return {"legacy": bool(reasons), "reasons": reasons}


def classify_protocol(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy verdict for a protocol-result artifact payload (pure)."""

    reasons: list[str] = []
    mismatch = _required_version_reason(
        payload,
        "protocol_version",
        PROTOCOL_VERSION,
        "no protocol_version (pre-P3)",
    )
    if mismatch:
        reasons.append(mismatch)
    mismatch = _required_version_reason(
        payload,
        "reward_version",
        REWARD_VERSION,
        "no reward_version (pre-P3)",
    )
    if mismatch:
        reasons.append(mismatch)
    if "dataset_id" not in payload:
        reasons.append("no dataset_id (pre-T1-01)")
    if "stitched" not in payload:
        reasons.append("no stitched OOS block (pre-T4-01)")
    if "ledger" not in payload:
        reasons.append("no ledger (pre-T4-01)")
    reasons.extend(_required_execution_reasons(payload))
    return {"legacy": bool(reasons), "reasons": reasons}


def classify_bare_factor(payload: dict[str, Any]) -> dict[str, Any]:
    """Legacy verdict for the fixed bare-factor measurement artifact."""

    from .bare_factor_backtest import BARE_FACTOR_BACKTEST_VERSION

    reasons: list[str] = []
    for field, current in (
        ("version", BARE_FACTOR_BACKTEST_VERSION),
        ("protocol_version", PROTOCOL_VERSION),
        ("reward_version", REWARD_VERSION),
    ):
        if payload.get(field) is None:
            reasons.append(f"no {field} (pre-P3)")
            continue
        mismatch = _version_mismatch(field, payload.get(field), current)
        if mismatch:
            reasons.append(mismatch)
    if "dataset_id" not in payload:
        reasons.append("no dataset_id (pre-T1-01)")
    reasons.extend(_required_execution_reasons(payload))
    return {"legacy": bool(reasons), "reasons": reasons}


def classify_artifact(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to the per-kind classifier (unknown kinds are never legacy)."""

    if name == "strategy":
        return classify_strategy(payload)
    if name == "protocol":
        return classify_protocol(payload)
    if name == "bare_factor":
        return classify_bare_factor(payload)
    return {"legacy": False, "reasons": []}


def stamp_legacy_artifacts(data_dir: Path) -> list[dict[str, Any]]:
    """Mark legacy strategy/protocol/bare-factor artifacts in ``data_dir``.

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
