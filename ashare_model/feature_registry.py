"""Feature registry: per-feature metadata for the search pipeline (T2-01).

Every vocabulary feature carries declarative metadata: its **PIT tier**
(the weakest data source it consumes: local daily bars, PIT fundamentals,
capital-flow feeds, current-snapshot extrapolations, or neutral), its
empirical **coverage** on the fixed calibration slice (fraction of
eligible cells that are finite), its **correlation cluster** (features
whose signals move together on the calibration slice form one cluster —
the natural family for redundancy-aware search), and its **deprecation
status**.

The registry is a deterministic function of its inputs and is versioned
(:data:`FEATURE_REGISTRY_VERSION`), so artifacts can cite exactly which
registry generation produced their numbers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from ashare_data.capital_flow import EXTERNAL_FACTOR_NAMES
from ashare_data.fundamentals import FUNDAMENTAL_PIT_NAMES

from .feature_metadata import (
    ComputeCost,
    RecommendedHorizon,
    SemanticType,
    authored_metadata_of,
    resolve_cost,
    resolve_depends_on,
    resolve_expected_horizon,
)
from .factors import FACTOR_REGISTRY, NEUTRAL_FEATURE_NAMES
from .semantic_cache import CalibrationSlice
from .vocab import FEATURE_NAMES

# Bump when the metadata schema, the tier mapping or the clustering
# semantics change.
#
# v3 (P7 D1, docs/p7_maintainability_plan.md §6.1): records additionally
# carry authored research metadata (availability rule, economic
# hypothesis, expected direction, semantic type, promotion permission)
# plus the horizon/cost/depends_on triple derived from the P6 research
# domains and FACTOR_REGISTRY.  Descriptive only — no search, scoring or
# promotion semantics change.
# v4 (P9, docs/p9_factor_family_contract.md APPROVED): the four orthogonal
# families join the vocabulary, eight window variants are deprecated, and
# the pending_data placeholders (family ⑤) are surfaced in the summary.
# v5 (P9 §7 adjudication, measurement log 2026-09-01): family ③ recorded
# as a negative result (its features keep promotion_allowed=False while
# staying samplable), the conditional LIMIT_STREAK deprecation triggered,
# and two second-pass consolidations (LIQ_SHOCK_20, CROWD_TURNOVER_60)
# deprecated.
FEATURE_REGISTRY_VERSION = 5

# Default correlation threshold for cluster membership.
CORR_THRESHOLD = 0.9


class PitLevel(str, enum.Enum):
    """Data tier of a feature: the weakest source it consumes."""

    PIT_DAILY = "pit_daily"  # local daily-bar computation
    PIT_FUNDAMENTAL = "pit_fundamental"  # point-in-time financial statements
    PIT_CAPITAL = "pit_capital"  # margin / capital-flow feeds
    SNAPSHOT = "snapshot"  # current-snapshot extrapolation (industry membership)
    NEUTRAL = "neutral"  # no live data source; stays neutral


# Deprecated features with their reason (single source: the vocabulary owns
# which names are de-sampled; the reasons ride along for the registry).
# Aliased features (FEATURE_ALIASES) are not part of the live vocabulary and
# therefore not in the registry.
from .vocab import DEPRECATION_REASONS as _DEPRECATION_REASONS  # noqa: E402


@dataclass(frozen=True)
class FeatureRecord:
    """Declarative metadata for one vocabulary feature."""

    name: str
    family: str
    pit_level: PitLevel
    coverage: float
    correlation_cluster: str
    deprecated: bool
    deprecation_reason: str | None
    # P7 D1 authored research metadata (descriptive; contract §6.1).
    availability_rule: str
    hypothesis: str
    expected_direction: int
    semantic_type: SemanticType
    expected_horizon: RecommendedHorizon | None
    promotion_allowed: bool
    compute_cost: ComputeCost
    depends_on: tuple[str, ...]


def _family_of(name: str) -> str:
    entry = FACTOR_REGISTRY.get(name)
    if entry is not None:
        return entry[0].family
    if name in FUNDAMENTAL_PIT_NAMES:
        return "fundamental"
    if name in EXTERNAL_FACTOR_NAMES:
        return "external"
    if name in NEUTRAL_FEATURE_NAMES:
        return "neutral"
    return ""


def _pit_level_of(name: str, family: str) -> PitLevel:
    if name in NEUTRAL_FEATURE_NAMES:
        return PitLevel.NEUTRAL
    if name in FUNDAMENTAL_PIT_NAMES:
        return PitLevel.PIT_FUNDAMENTAL
    if name in EXTERNAL_FACTOR_NAMES:
        # INDUSTRY_MOMENTUM is built on the industry membership snapshot;
        # MARGIN_BALANCE_CHG is a PIT capital-flow feed.
        return (
            PitLevel.SNAPSHOT
            if name == "INDUSTRY_MOMENTUM"
            else PitLevel.PIT_CAPITAL
        )
    if family == "industry":
        # IND_REL_* demean against the Shenwan membership snapshot.
        return PitLevel.SNAPSHOT
    return PitLevel.PIT_DAILY


def pit_level_of(name: str) -> PitLevel:
    """Public PIT level (weakest data source) of one vocabulary feature.

    The single resolution path for tier mapping (P2): ``data_tier`` derives
    the A/B/C credibility tier from this level.
    """

    return _pit_level_of(name, _family_of(name))


def family_of(name: str) -> str:
    """Public family derivation (single path; shared with the generated
    registry docs, which must not keep a second copy)."""

    return _family_of(name)


def _zscore_per_date(tensor: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Per-date cross-sectional z-score of every feature over eligible
    cells; non-eligible cells become 0 (the neutral convention)."""

    finite = np.isfinite(tensor)
    ref = finite & eligible
    safe = np.where(ref, tensor, 0.0)
    n = ref.sum(axis=1, keepdims=True).clip(min=1)
    mean = safe.sum(axis=1, keepdims=True) / n
    dev = np.where(ref, tensor - mean, 0.0)
    var = (dev * dev).sum(axis=1, keepdims=True) / n
    std = np.sqrt(var)
    return np.where(ref, (tensor - mean) / (std + 1e-9), 0.0)


def _correlation_clusters(
    tensor: np.ndarray, eligible: np.ndarray, threshold: float
) -> dict[str, str]:
    """Connected components of features at ``|corr| >= threshold``.

    Feature similarity is the Pearson correlation of the per-date
    z-scored signals across all calibration cells.  Components are sorted
    by their lexicographically smallest member and id'd ``c0, c1, ...``,
    so the ids are deterministic in the data.  A degenerate feature (no
    variance) correlates 0 with everything and forms its own cluster.
    """

    n = tensor.shape[0]
    z = _zscore_per_date(tensor, eligible).reshape(n, -1)
    # Union-find over the feature indices.
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        zi = z[i]
        denom = float(np.sqrt(zi @ zi))
        if denom <= 0.0:
            continue
        for j in range(i + 1, n):
            zj = z[j]
            denom_j = float(np.sqrt(zj @ zj))
            if denom_j <= 0.0:
                continue
            corr = float(zi @ zj) / (denom * denom_j)
            if abs(corr) >= threshold:
                union(i, j)

    members: dict[int, list[str]] = {}
    for i, name in enumerate(FEATURE_NAMES):
        members.setdefault(find(i), []).append(name)
    groups = sorted(
        (sorted(names) for names in members.values()),
        key=lambda names: names[0],
    )
    ids = {name: f"c{idx}" for idx, names in enumerate(groups) for name in names}
    return ids


def _counts_by(records, key) -> dict[str, int]:
    """Deterministic value counts over records (summary helper)."""

    out: dict[str, int] = {}
    for record in records:
        name = key(record)
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


@dataclass
class FeatureRegistry:
    """Deterministic per-feature metadata derived from the factor tensor."""

    _records: dict[str, FeatureRecord]
    slice_spec: str
    source: str

    def records(self) -> dict[str, FeatureRecord]:
        return dict(self._records)

    def summary(self) -> dict[str, object]:
        # Lazy import: ``data_tier`` resolves features back through this
        # module, so the dependency cannot be module-level.
        from .data_tier import tier_of_pit_level

        by_level: dict[str, int] = {}
        by_tier: dict[str, int] = {}
        for record in self._records.values():
            by_level[record.pit_level.value] = by_level.get(record.pit_level.value, 0) + 1
            tier = tier_of_pit_level(record.pit_level).value
            by_tier[tier] = by_tier.get(tier, 0) + 1
        # P9 §6: pending_data placeholders (family ⑤) are documented gaps,
        # never vocabulary members and never search-budget consumers.
        from .feature_metadata import PENDING_DATA_FEATURES

        return {
            "n_features": len(self._records),
            "by_pit_level": by_level,
            "by_data_tier": by_tier,
            "deprecated": sorted(
                name for name, r in self._records.items() if r.deprecated
            ),
            "pending_data": sorted(
                entry.name for entry in PENDING_DATA_FEATURES
            ),
            "n_clusters": len(
                {r.correlation_cluster for r in self._records.values()}
            ),
            "by_semantic_type": _counts_by(
                self._records.values(), lambda r: r.semantic_type.value
            ),
            "by_expected_horizon": _counts_by(
                self._records.values(),
                lambda r: r.expected_horizon.value if r.expected_horizon else "none",
            ),
        }

    def to_dict(self) -> dict[str, object]:
        # Lazy import: see :meth:`summary`.
        from .data_tier import DATA_TIER_VERSION, tier_of_pit_level

        return {
            "version": FEATURE_REGISTRY_VERSION,
            "data_tier_version": DATA_TIER_VERSION,
            "slice": self.slice_spec,
            "source": self.source,
            "features": [
                {
                    "name": r.name,
                    "family": r.family,
                    "pit_level": r.pit_level.value,
                    "data_tier": tier_of_pit_level(r.pit_level).value,
                    "coverage": r.coverage,
                    "correlation_cluster": r.correlation_cluster,
                    "deprecated": r.deprecated,
                    "deprecation_reason": r.deprecation_reason,
                    # P7 D1 research metadata (additive; contract §6.1).
                    "availability_rule": r.availability_rule,
                    "hypothesis": r.hypothesis,
                    "expected_direction": r.expected_direction,
                    "semantic_type": r.semantic_type.value,
                    "expected_horizon": (
                        r.expected_horizon.value if r.expected_horizon else None
                    ),
                    "promotion_allowed": r.promotion_allowed,
                    "compute_cost": r.compute_cost.value,
                    "depends_on": list(r.depends_on),
                }
                for r in self._records.values()
            ],
        }

    @classmethod
    def build(
        cls,
        factor_tensor: np.ndarray,
        universe_mask: np.ndarray,
        *,
        calibration_slice: CalibrationSlice,
        families: dict[str, str] | None = None,
        corr_threshold: float = CORR_THRESHOLD,
    ) -> "FeatureRegistry":
        """Build the registry from a ``[feature, stock, date]`` tensor and
        the aligned PIT eligibility mask, measured on the calibration
        slice.  Deterministic in its inputs."""

        tensor = np.asarray(factor_tensor, dtype=np.float64)
        mask = np.asarray(universe_mask, dtype=bool)
        if tensor.ndim != 3:
            raise ValueError(f"factor tensor must be 3-D, got {tensor.ndim}-D")
        if mask.shape != tensor.shape[1:]:
            raise ValueError(
                f"universe_mask shape {mask.shape} does not match tensor "
                f"(stock, date) shape {tensor.shape[1:]}"
            )
        if tensor.shape[0] != len(FEATURE_NAMES):
            raise ValueError(
                f"factor tensor has {tensor.shape[0]} rows; the vocabulary "
                f"has {len(FEATURE_NAMES)} features"
            )

        dates = slice(calibration_slice.date_start, calibration_slice.date_end)
        tens = tensor[:, :, dates]
        elig = mask[:, dates]

        n_eligible = int(elig.sum())
        coverage: dict[str, float] = {}
        for i, name in enumerate(FEATURE_NAMES):
            covered = int((np.isfinite(tens[i]) & elig).sum())
            coverage[name] = float(covered / max(n_eligible, 1))

        clusters = _correlation_clusters(tens, elig, float(corr_threshold))

        records: dict[str, FeatureRecord] = {}
        for i, name in enumerate(FEATURE_NAMES):
            family = (
                families.get(name, "") if families is not None else _family_of(name)
            )
            # Authored research metadata (P7 D1): fails closed when a
            # vocabulary feature was added without its declaration.
            authored = authored_metadata_of(name)
            deprecated = name in _DEPRECATION_REASONS
            if deprecated and authored.promotion_allowed:
                raise ValueError(
                    f"deprecated feature {name} must set promotion_allowed=False"
                )
            records[name] = FeatureRecord(
                name=name,
                family=family,
                pit_level=_pit_level_of(name, family),
                coverage=coverage[name],
                correlation_cluster=clusters[name],
                deprecated=deprecated,
                deprecation_reason=_DEPRECATION_REASONS.get(name),
                availability_rule=authored.availability_rule,
                hypothesis=authored.hypothesis,
                expected_direction=authored.expected_direction,
                semantic_type=authored.semantic_type,
                expected_horizon=resolve_expected_horizon(name),
                promotion_allowed=authored.promotion_allowed,
                compute_cost=resolve_cost(name, authored),
                depends_on=resolve_depends_on(name, authored),
            )
        return cls(records, str(calibration_slice), source="factor_tensor")


def formula_registry_status_report(
    tokens=None,
    feature_name: str | None = None,
) -> dict:
    """Registry promotability of a formula's features (P12 promotion G7).

    Single decode path: reuses :func:`ashare_model.data_tier.formula_feature_names`
    (the route the G6 data-tier gate already uses); ``tokens`` is the
    canonical postfix token list and ``feature_name`` covers bare-factor
    baseline rows (``formula=None``, ``formula_text=NAME``), symmetric to
    :func:`ashare_model.data_tier.formula_data_tier_report`.  The
    per-feature status is read from the single registry authorities
    (``feature_metadata`` authored flags and ``vocab.DEPRECATION_REASONS``)
    — never copied.

    Returns ``{"feature_registry_version", "per_feature": {name:
    {"promotion_allowed": bool, "deprecated": bool}}, "traceable": bool}``.
    An untraceable input (no tokens, an undecodable token list, a name
    outside the vocabulary, or an authored-metadata gap) yields
    ``traceable=False`` with an empty ``per_feature`` — the promotion gate
    must reject (fail closed, docs/p12_promotion_enforcement_contract.md §4).
    """

    # Lazy import: ``data_tier`` imports this module at module level
    # (PitLevel / pit_level_of), so the dependency cannot be module-level
    # here either (same pattern as :meth:`FeatureRegistry.summary`).
    from .data_tier import formula_feature_names

    untraceable = {
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "per_feature": {},
        "traceable": False,
    }
    if tokens is not None:
        names = formula_feature_names(tokens)
        if names is None:
            return untraceable
    elif feature_name is not None:
        if feature_name not in FEATURE_NAMES:
            return untraceable
        names = [feature_name]
    else:
        return untraceable

    per_feature: dict[str, dict[str, bool]] = {}
    for name in names:
        try:
            allowed = bool(authored_metadata_of(name).promotion_allowed)
        except ValueError:
            # Authored-metadata gap: fail closed — the gate rejects via
            # ``traceable=False`` instead of guessing a status.
            return untraceable
        per_feature[name] = {
            "promotion_allowed": allowed,
            "deprecated": name in _DEPRECATION_REASONS,
        }
    return {
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "per_feature": per_feature,
        "traceable": True,
    }
