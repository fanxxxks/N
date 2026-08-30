"""Operator metadata registry (P7 D2).

``OPS_CONFIG`` (``ashare_model.ops``) stays the single source for the
implementation and arity; this module adds the descriptive metadata layer
on top — category, per-argument input semantics, output semantics, compute
cost and numerical-stability notes — resolved **by name/arity from
``OPS_CONFIG``** so the two can never drift (arity is derived, never
copied; see :func:`arity_of`).

Fields are descriptive only: nothing here changes sampling, evaluation or
canonicalization semantics.  ``output`` is ``None`` where the six-type
semantic lattice (P7 D1) has no committed home yet (scale-free bounded
statistics: TS_RANK/CORR); the Phase E contract assigns those when it
wires semantic types into search legality — they are never guessed here.

Input-semantics conventions (the Phase E direction, recorded early):

* boolean/event signals may not flow through windowed numeric reducers
  (MA/STD/TS_RANK/DOWNVOL/DELTA/DECAY/...) or correlations;
* correlations need two continuous signals (both arguments exclude
  boolean/event);
* ``CS_NEUTRALIZE`` accepts only cross-sectional signals (industry
  demeaning is meaningless for non-cross-sectional inputs);
* arithmetic ops accept any semantic type today — narrowing them is a
  Phase E contract decision.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .feature_metadata import ComputeCost, SemanticType
from .ops import OPS_BY_NAME

_SIX = tuple(SemanticType)
_CONT = tuple(t for t in SemanticType if t is not SemanticType.BOOLEAN_EVENT_SIGNAL)
_XS = (SemanticType.CROSS_SECTIONAL_SIGNAL,)


class OperatorCategory(str, enum.Enum):
    ARITHMETIC = "arithmetic"
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"
    EVENT = "event"


@dataclass(frozen=True)
class OperatorMeta:
    """Descriptive metadata for one registered operator.

    ``inputs`` carries one tuple of allowed semantic types per argument
    (argument order matches the operator signature; commutative pairs
    repeat the same tuple).  ``output`` is the committed semantic type of
    the result, or ``None`` when the lattice has no home for it yet.
    """

    category: OperatorCategory
    inputs: tuple[tuple[SemanticType, ...], ...]
    output: SemanticType | None
    cost: ComputeCost
    stability: str


OPERATOR_REGISTRY: dict[str, OperatorMeta] = {
    # --- arithmetic -------------------------------------------------------
    "ADD": OperatorMeta(
        OperatorCategory.ARITHMETIC, (_SIX, _SIX), None, ComputeCost.LOW,
        "exact float addition; float32 rounding only",
    ),
    "SUB": OperatorMeta(
        OperatorCategory.ARITHMETIC, (_SIX, _SIX), None, ComputeCost.LOW,
        "exact float subtraction; float32 rounding only",
    ),
    "MUL": OperatorMeta(
        OperatorCategory.ARITHMETIC, (_SIX, _SIX), None, ComputeCost.LOW,
        "exact float multiplication; magnitudes can overflow to inf in extreme tails",
    ),
    "DIV": OperatorMeta(
        OperatorCategory.ARITHMETIC, (_SIX, _SIX), None, ComputeCost.LOW,
        "zero-denominator guard: denom == 0 maps to eps with sign, so the result stays finite",
    ),
    "NEG": OperatorMeta(
        OperatorCategory.ARITHMETIC, (_SIX,), None, ComputeCost.LOW,
        "exact negation",
    ),
    "ABS": OperatorMeta(
        OperatorCategory.ARITHMETIC, (_SIX,), None, ComputeCost.LOW,
        "exact absolute value",
    ),
    "SIGN": OperatorMeta(
        OperatorCategory.ARITHMETIC, (_SIX,), None, ComputeCost.LOW,
        "exact sign; NaN inputs cannot reach the VM (clamped upstream)",
    ),
    # --- event / gating ----------------------------------------------------
    "GATE": OperatorMeta(
        OperatorCategory.EVENT, (_SIX, _SIX, _SIX), None, ComputeCost.LOW,
        "condition > 0 selector; exact mask arithmetic",
    ),
    "JUMP": OperatorMeta(
        OperatorCategory.EVENT, (_CONT,), SemanticType.BOOLEAN_EVENT_SIGNAL, ComputeCost.MEDIUM,
        "causal trailing 60-session z-baseline (v9); z bounded by sqrt(n-1), "
        "expanding head windows < 10 sessions can never fire",
    ),
    # --- time series -------------------------------------------------------
    "DECAY": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "constant extension at the leading edge (no fabricated decay)",
    ),
    "DELAY1": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.LOW,
        "constant extension at the leading edge",
    ),
    "MAX3": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.LOW,
        "trailing 3-session max with constant extension",
    ),
    "DELTA5": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "difference over 5 sessions; constant extension avoids spurious jumps",
    ),
    "DELTA10": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "difference over 10 sessions; constant extension",
    ),
    "DELTA20": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "difference over 20 sessions; constant extension",
    ),
    "MA5": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "NaN-aware trailing mean; expanding window for short series",
    ),
    "MA10": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "NaN-aware trailing mean; expanding window for short series",
    ),
    "MA20": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "NaN-aware trailing mean; expanding window for short series",
    ),
    "MA60": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "NaN-aware trailing mean; expanding window for short series",
    ),
    "STD5": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "NaN-aware trailing std; expanding window for short series",
    ),
    "STD10": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "NaN-aware trailing std; expanding window for short series",
    ),
    "STD20": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "NaN-aware trailing std; expanding window for short series",
    ),
    "STD60": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "NaN-aware trailing std; expanding window for short series",
    ),
    "TS_RANK5": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "trailing percentile rank; NaN padding excluded from the denominator (clamped >= 1)",
    ),
    "TS_RANK10": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "trailing percentile rank; NaN padding excluded from the denominator (clamped >= 1)",
    ),
    "TS_RANK20": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "trailing percentile rank; NaN padding excluded from the denominator (clamped >= 1)",
    ),
    "TS_RANK60": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), None, ComputeCost.MEDIUM,
        "trailing percentile rank; NaN padding excluded from the denominator (clamped >= 1)",
    ),
    "CORR5": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT, _CONT), None, ComputeCost.MEDIUM,
        "prefix-sum trailing Pearson correlation; degenerate window yields 0",
    ),
    "CORR10": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT, _CONT), None, ComputeCost.MEDIUM,
        "prefix-sum trailing Pearson correlation; degenerate window yields 0",
    ),
    "CORR20": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT, _CONT), None, ComputeCost.MEDIUM,
        "prefix-sum trailing Pearson correlation; degenerate window yields 0",
    ),
    "CORR60": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT, _CONT), None, ComputeCost.MEDIUM,
        "prefix-sum trailing Pearson correlation; degenerate window yields 0",
    ),
    "DOWNVOL5": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "sqrt of mean squared negative values; matches the neutral-0 convention",
    ),
    "DOWNVOL10": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "sqrt of mean squared negative values; matches the neutral-0 convention",
    ),
    "DOWNVOL20": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "sqrt of mean squared negative values; matches the neutral-0 convention",
    ),
    "DOWNVOL60": OperatorMeta(
        OperatorCategory.TIME_SERIES, (_CONT,), SemanticType.RETURN_LIKE, ComputeCost.MEDIUM,
        "sqrt of mean squared negative values; matches the neutral-0 convention",
    ),
    # --- cross-sectional ---------------------------------------------------
    "CS_RANK": OperatorMeta(
        OperatorCategory.CROSS_SECTIONAL, (_CONT,), SemanticType.CROSS_SECTIONAL_SIGNAL, ComputeCost.MEDIUM,
        "reference cells = finite & PIT-eligible; degenerate cross-section -> neutral 0",
    ),
    "CS_ZSCORE": OperatorMeta(
        OperatorCategory.CROSS_SECTIONAL, (_CONT,), SemanticType.CROSS_SECTIONAL_SIGNAL, ComputeCost.MEDIUM,
        "reference cells = finite & PIT-eligible; count clamped >= 1",
    ),
    "CS_DEMEAN": OperatorMeta(
        OperatorCategory.CROSS_SECTIONAL, (_CONT,), SemanticType.CROSS_SECTIONAL_SIGNAL, ComputeCost.MEDIUM,
        "reference cells = finite & PIT-eligible; degenerate cross-section -> neutral 0",
    ),
    "CS_NEUTRALIZE": OperatorMeta(
        OperatorCategory.CROSS_SECTIONAL, (_XS,), SemanticType.CROSS_SECTIONAL_SIGNAL, ComputeCost.MEDIUM,
        "industry-group demean on the VM's industry codes; degrades to CS_DEMEAN without them "
        "(the group substitution is recorded, never silent in artifacts)",
    ),
}


def operator_meta_of(name: str) -> OperatorMeta:
    """Metadata of one operator; unknown names fail loudly."""

    try:
        return OPERATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"operator {name!r} has no registry metadata; every OPS_CONFIG "
            "operator must be declared in ashare_model.operator_registry"
        ) from None


def arity_of(name: str) -> int:
    """Arity derived from OPS_CONFIG (single authority; never duplicated)."""

    return OPS_BY_NAME[name][1]


# Import-time invariant: the metadata layer covers exactly the implementation
# layer, and the authored argument counts always match the derived arity.
assert set(OPERATOR_REGISTRY) == set(OPS_BY_NAME), (
    "OPERATOR_REGISTRY must cover exactly the OPS_CONFIG operators"
)
for _name, _meta in OPERATOR_REGISTRY.items():
    assert len(_meta.inputs) == arity_of(_name), _name
