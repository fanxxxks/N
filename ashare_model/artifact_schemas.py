"""Typed schemas for the formal boundary artifacts (P7-C).

Contract: ``docs/p7_artifact_schema_contract.md``.  The strategy
(``best_ashare_strategy.json``), protocol (``protocol_result.json``),
backtest (``backtest_result.json``) and paper-state
(``sim_portfolio_state.json``) payloads carry
``artifact_schema_version`` and are validated with Pydantic:

* **write side (fail-closed)**: writers stamp the version and validate
  with :meth:`_SchemaBase.model_validate` before persisting — an invalid
  payload raises ``ValidationError`` and never reaches disk;
* **read side (classify matrix)**: :func:`classify_schema_version` is the
  single classification entry point (readers must not implement a second
  version comparison): ``current`` payloads validate via
  ``validate_payload`` (``ArtifactSchemaError`` on failure), ``legacy``
  payloads flow through the pre-contract read path byte-identically, and
  ``unknown``/future versions are hard-rejected.

Top-level models are ``extra="forbid"`` by contract: new top-level
provenance requires an explicit schema change plus a bump of
:data:`ARTIFACT_SCHEMA_VERSION` — fields never diffuse silently.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

# Bump when any top-level field is added, removed or retyped; the contract
# (§4) rejects unknown/future versions on read.
ARTIFACT_SCHEMA_VERSION = 1


class ArtifactSchemaError(ValueError):
    """A boundary artifact failed the schema-version matrix or validation."""


def classify_schema_version(
    payload: Mapping[str, Any],
) -> Literal["current", "legacy", "unknown"]:
    """Classify one artifact payload by its ``artifact_schema_version``.

    The single version-classification entry point (contract §4): a missing
    key is a pre-contract ``legacy`` (v0) artifact; exactly
    :data:`ARTIFACT_SCHEMA_VERSION` is ``current``; anything else —
    non-integer, boolean, or a version this code does not know — is
    ``unknown`` and must be hard-rejected by the reader.
    """

    version = payload.get("artifact_schema_version")
    if version is None:
        return "legacy"
    if isinstance(version, bool) or not isinstance(version, int):
        return "unknown"
    if version == ARTIFACT_SCHEMA_VERSION:
        return "current"
    return "unknown"


def require_current_schema(payload: Mapping[str, Any], *, artifact: str) -> None:
    """Hard-reject payloads that are not ``current`` schema.

    Raises :class:`ArtifactSchemaError` for ``unknown``/``legacy``
    payloads; readers with a legacy path must branch on
    :func:`classify_schema_version` *before* calling this.
    """

    verdict = classify_schema_version(payload)
    if verdict != "current":
        raise ArtifactSchemaError(
            f"{artifact} artifact is not current schema "
            f"v{ARTIFACT_SCHEMA_VERSION} (classified {verdict!r}); "
            "unknown/future versions are rejected — upgrade the code "
            "instead of downgrading the artifact"
        )


def apply_schema_matrix(
    payload: Mapping[str, Any], *, artifact: str, model: type
) -> Literal["current", "legacy"]:
    """The read-side version matrix (contract §4), single implementation.

    ``unknown``/future versions raise :class:`ArtifactSchemaError`;
    ``current`` payloads are validated against ``model`` (a
    :class:`_SchemaBase` subclass); ``legacy`` payloads pass through to
    the caller's pre-contract read path unchanged.  Returns the verdict
    so the caller can branch legacy/current handling.
    """

    verdict = classify_schema_version(payload)
    if verdict == "unknown":
        raise ArtifactSchemaError(
            f"{artifact} artifact has an unknown/future "
            "artifact_schema_version; rejected — upgrade the code instead "
            "of downgrading the artifact"
        )
    if verdict == "current":
        model.validate_payload(payload)
    return verdict


class _SchemaBase(BaseModel):
    """Shared config: top-level ``extra="forbid"`` (contract §3)."""

    model_config = ConfigDict(extra="forbid")

    artifact_schema_version: int

    @field_validator("artifact_schema_version")
    @classmethod
    def _version_is_current(cls, value: int) -> int:
        if value != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact_schema_version {value} != current "
                f"{ARTIFACT_SCHEMA_VERSION}"
            )
        return value

    @classmethod
    def validate_payload(cls, payload: Mapping[str, Any]):
        """Read-side validation: classify must be ``current`` and the
        payload must satisfy the model; both failures raise
        :class:`ArtifactSchemaError`."""

        require_current_schema(payload, artifact=cls.__name__)
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ArtifactSchemaError(
                f"{cls.__name__} payload failed schema validation: {exc}"
            ) from exc

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump()


class StrategyArtifact(_SchemaBase):
    """``best_ashare_strategy.json`` spine (writer: trainer._write_artifact).

    Field list enumerated from the writer code (contract §3); keys that
    may be null are still *required keys* (e.g. ``dataset_id`` is ``null``
    for pre-T1-01 databases but must be present).
    """

    formula: list[int]
    candidate_id: str
    formula_text: str
    source: str
    direction: int
    data_tier: dict | None
    val_reward: float | None
    val_icir: float | None
    train_reward: float | None
    train_icir: float | None
    complexity_penalty: float
    complexity_cost: float
    active_ir: float | None
    risk_exposure: float | None
    average_turnover: float | None
    capacity_utilization: float | None
    eligible: bool
    rejection_reasons: list[str]
    history: list[dict]
    searcher: str
    init_seed: int
    device: str
    model_version: int
    feature_names: list[str]
    operator_names: list[str]
    feature_version: str
    grammar_version: int
    reward_version: str
    protocol_version: str
    research_domain: str
    research_domain_version: int
    execution_version: int
    portfolio_constructor_version: int
    portfolio_config: dict
    semantic_cache_version: int
    unique_semantic_evals: int | None
    semantic_cache_stats: dict | None
    search_contract_version: int | None
    search_result: dict | None
    dataset_id: str | None
    # RL-only fields (absent for non-RL searchers).
    rl_initialization: str | None = None
    imitation: dict | None = None
    experimental: bool | None = None
    # Legacy stamp, present only on files stamped post-hoc by
    # artifact_versions (never written by the trainer).
    legacy: bool | None = None
    legacy_reason: list[str] | None = None
    legacy_stamped_at: str | None = None


class ProtocolResultArtifact(_SchemaBase):
    """``protocol_result.json`` spine (writer: eval_artifacts.build_result)."""

    protocol_version: str
    data_tier_version: int
    reward_version: str
    research_domain: str
    research_domain_version: int
    execution_version: int
    portfolio_constructor_version: int
    portfolio_config: dict
    dataset_id: str | None
    frequency: str
    horizon: int
    tier: str
    steps: int
    batch_size: int
    seeds: list[int]
    random_samples: int
    random_seed: int
    random_budget_matched: bool | None
    random_budget: int | None
    folds: list[dict]
    baseline_signals: list[str]
    data_end_date: str | None
    universe_policy: dict | None
    data_regime: dict | None
    ledger: dict | None
    n_candidates: int
    rows: list[dict]
    aggregates: dict
    stitched: dict
    top_trial: dict | None
    dsr: dict | None
    max_t: dict | None
    dsr_extra_trials: int
