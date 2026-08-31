"""RunStore-bound write/read path for the formal boundary artifacts (P8-05).

This is the **only** sanctioned path for writing and formally consuming
strategy / protocol / backtest / paper-state artifacts after the schema
v2 bump (``docs/p8_research_lifecycle_contract.md`` §2/§8):

* writes go through a :class:`~ashare_model.run_store.RunHandle` — the
  artifact is stamped with the run's lifecycle identity
  (``spec_id``/``run_id``/``candidate_id``, plus ``account_id`` for paper
  state), validated fail-closed against its Pydantic spine, persisted
  content-addressed in the RunStore (atomic file write, then atomic index
  registration), and only then mirrored to the convenience file (atomic);
  a failure anywhere before the mirror leaves every existing file intact;
* formal reads require the **current** schema (v2) — legacy v0/v1
  payloads are audit-only (:func:`classify_schema_version` verdict
  ``legacy``) and must never enter promotion/resume/backtest consumption;
* cross-artifact lineage consistency is enforced with
  :func:`ashare_model.artifact_schemas.assert_consistent_lineage`.

The convenience file keeps a full atomic mirror of the current payload so
display consumers (dashboard/webapi) keep working; it is a cache, never
the formal evidence — formal consumers resolve identity from the
RunStore-registered artifact record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_data.io_utils import atomic_write_json

from .artifact_schemas import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactSchemaError,
    _LineageBase,
    assert_consistent_lineage,
    classify_schema_version,
)
from .identity import content_hash
from .run_store import ArtifactRecord, RunHandle

__all__ = [
    "artifact_id_of",
    "read_boundary_artifact",
    "write_boundary_artifact",
]


def artifact_id_of(payload: dict[str, Any]) -> str:
    """The content id of a current boundary artifact payload.

    Single implementation: ``content_hash("artifact", payload)`` over the
    payload as persisted in the RunStore (which is exactly what
    ``RunHandle.write_artifact`` registered). Recomputable from the
    payload alone, verifiable against the run index.
    """

    return content_hash("artifact", payload)


def write_boundary_artifact(
    handle: RunHandle,
    *,
    artifact_type: str,
    model_cls: type[_LineageBase],
    payload: dict[str, Any],
    candidate_id: str,
    account_id: str | None = None,
    convenience_path: str | Path | None = None,
) -> ArtifactRecord:
    """Validate, persist through the RunStore, then mirror to the
    convenience file. Fail-closed: nothing is written unless the payload
    satisfies the current schema and the run's lineage."""

    if not isinstance(handle, RunHandle):
        raise ArtifactSchemaError(
            "formal artifact writes require an open RunHandle (RunStore)"
        )
    expected = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "spec_id": handle.spec.spec_id,
        "run_id": handle.run_id,
        "candidate_id": candidate_id,
    }
    if account_id is not None:
        expected["account_id"] = account_id
    conflicts = {
        key: {"payload": payload[key], "expected": value}
        for key, value in expected.items()
        if key in payload and payload[key] != value
    }
    if conflicts:
        raise ArtifactSchemaError(
            f"payload identity conflicts with the open run "
            f"({handle.spec.spec_id}/{handle.run_id}); refusing to stamp "
            f"over contradictory identity: {conflicts}"
        )
    final = {**payload, **expected}
    # Fail-closed validation (P7-C §2.1): raises ArtifactSchemaError and
    # never touches disk on failure.
    model_cls.validate_payload(final)
    record = handle.write_artifact(artifact_type, final, candidate_id=candidate_id)
    if convenience_path is not None:
        atomic_write_json(Path(convenience_path), final)
    return record


def read_boundary_artifact(
    path: str | Path,
    *,
    model_cls: type[_LineageBase],
    formal: bool,
) -> tuple[dict[str, Any], str] | None:
    """Read one boundary artifact through the classify matrix.

    Returns ``(payload, verdict)`` or ``None`` when the file is absent.
    ``formal=True`` hard-rejects legacy (v0/v1) payloads — audit-only
    artifacts must never enter promotion/resume/backtest consumption;
    ``formal=False`` returns them unchanged for audit/display paths.
    Unknown/future versions are always rejected.
    """

    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    verdict = classify_schema_version(payload)
    if verdict == "unknown":
        raise ArtifactSchemaError(
            f"{artifact_path} has an unknown/future artifact_schema_version; "
            "rejected — upgrade the code instead of downgrading the artifact"
        )
    if verdict == "current":
        model_cls.validate_payload(payload)
        return payload, verdict
    if formal:
        raise ArtifactSchemaError(
            f"{artifact_path} is legacy (v0/v1, classified {verdict!r}); "
            "formal consumption requires current schema "
            f"v{ARTIFACT_SCHEMA_VERSION} with a lifecycle identity block — "
            "legacy artifacts are audit-only (contract §8)"
        )
    return payload, verdict
