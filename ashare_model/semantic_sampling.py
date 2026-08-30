"""Semantic-type legality for formula sampling (P7-E).

Contract: ``docs/p7_semantic_types_contract.md`` (pre-registered).  This
module is the **single** implementation of the semantic-type lattice over
the formula vocabulary:

* terminal types come from the authored feature metadata
  (:data:`ashare_model.feature_metadata.FEATURE_METADATA`);
* per-argument allowed types come from the P7-D2
  :data:`ashare_model.operator_registry.OPERATOR_REGISTRY`;
* the output-type rules the D2 registry deliberately left open
  (family-preserving transforms, cross-sectional ratios/correlations,
  volatility scales, the JUMP event detector) are resolved here — one
  place, consumed by the action mask (RL/Random/TPE) and the typed GP
  primitive set alike.

``formula_types_legal`` is the shared sequence checker: a token sequence
is type-legal when every operator application satisfies its signature and
exactly one well-typed value remains.  Structural postfix legality
(arity/termination) stays with the existing grammar; this layer only
adds the typing dimension — VM execution and canonicalization semantics
are untouched.
"""

from __future__ import annotations

from ashare_model.feature_metadata import FEATURE_METADATA, SemanticType
from ashare_model.ops import OPS_BY_NAME
from ashare_model.vocab import FORMULA_VOCAB, FormulaVocab

_CONTINUOUS = frozenset(SemanticType) - {SemanticType.BOOLEAN_EVENT_SIGNAL}

# Contract §1: operators whose output is committed by this contract (the
# rest are family-preserving and return the shared input family).
_OUTPUT_CROSS_SECTIONAL = frozenset({
    "DIV",
    "CORR5", "CORR10", "CORR20", "CORR60",
    "CS_RANK", "CS_ZSCORE", "CS_DEMEAN", "CS_NEUTRALIZE",
})
_OUTPUT_RETURN_LIKE = frozenset({
    "STD5", "STD10", "STD20", "STD60",
    "DOWNVOL5", "DOWNVOL10", "DOWNVOL20", "DOWNVOL60",
    "DELTA5", "DELTA10", "DELTA20",
})
_OUTPUT_EVENT = frozenset({"JUMP"})
# Family-preserving: all non-boolean args must share one type T, output T.
# GATE's condition argument is exempt (any type, its intended use); the
# branches (args 1..n) carry the shared type.
_FAMILY_PRESERVING = frozenset(OPS_BY_NAME) - (
    _OUTPUT_CROSS_SECTIONAL | _OUTPUT_RETURN_LIKE | _OUTPUT_EVENT
)


def _feature_type(name: str) -> SemanticType:
    try:
        return FEATURE_METADATA[name].semantic_type
    except KeyError:
        raise ValueError(
            f"feature {name!r} has no authored semantic type; every "
            "vocabulary feature must be declared in "
            "ashare_model.feature_metadata"
        ) from None


def signature_legal(name: str, arg_types: tuple[SemanticType, ...]) -> bool:
    """Whether applying operator ``name`` to these argument types is legal.

    Combines the registry's per-argument allowed types with the contract's
    same-type rule for family-preserving operators (GATE's condition
    argument exempt).  Raises ``ValueError`` on an arity mismatch — a
    caller bug, not a semantic violation.
    """

    fn, arity = OPS_BY_NAME[name]
    del fn
    if len(arg_types) != arity:
        raise ValueError(
            f"operator {name} expects {arity} arguments, got {len(arg_types)}"
        )
    from ashare_model.operator_registry import OPERATOR_REGISTRY

    meta = OPERATOR_REGISTRY[name]
    for position, (allowed, actual) in enumerate(zip(meta.inputs, arg_types)):
        if actual not in allowed:
            return False
        # Contract §1 narrowing: every argument is continuous except the
        # GATE condition (an event signal's intended use).  The D2 registry
        # deliberately recorded arithmetic as accepting any type today;
        # this line is where the Phase-E contract tightens it.
        if not (name == "GATE" and position == 0):
            if actual not in _CONTINUOUS:
                return False
    if name in _FAMILY_PRESERVING:
        branch_types = arg_types[1:] if name == "GATE" else arg_types
        if len(set(branch_types)) > 1:
            return False
    return True


def resolve_output(name: str, arg_types: tuple[SemanticType, ...]) -> SemanticType:
    """Output semantic type of one legal operator application (contract
    §1).  Callers must have checked :func:`signature_legal` first; the
    family-preserving branch relies on the shared input type."""

    if name in _OUTPUT_CROSS_SECTIONAL:
        return SemanticType.CROSS_SECTIONAL_SIGNAL
    if name in _OUTPUT_RETURN_LIKE:
        return SemanticType.RETURN_LIKE
    if name in _OUTPUT_EVENT:
        return SemanticType.BOOLEAN_EVENT_SIGNAL
    # Family-preserving: GATE returns the branch type, everything else the
    # shared (single) argument type.
    return arg_types[1] if name == "GATE" else arg_types[0]


def formula_types_legal(tokens, vocab: FormulaVocab | None = None) -> bool:
    """Type-legality of a full token sequence (EOS/PAD included or not).

    Mirrors the stack machine: features push their semantic type,
    operators pop their arity and push the resolved output; the sequence
    is legal when every application is signature-legal and exactly one
    value remains.  Non-finite token ids are the grammar's business and
    simply fail here.
    """

    vocab = vocab or FORMULA_VOCAB
    stack: list[SemanticType] = []
    for raw in tokens:
        token = int(raw)
        if token == vocab.pad_token_id:
            break
        if vocab.eos_token_id is not None and token == vocab.eos_token_id:
            break
        if token < vocab.operator_offset:
            name = vocab.feature_names[token - vocab.feature_offset]
            stack.append(_feature_type(name))
            continue
        name = vocab.operator_names[token - vocab.operator_offset]
        arity = OPS_BY_NAME[name][1]
        if len(stack) < arity:
            return False
        arg_types = tuple(stack[len(stack) - arity :])
        del stack[len(stack) - arity :]
        if not signature_legal(name, arg_types):
            return False
        stack.append(resolve_output(name, arg_types))
    return len(stack) == 1


def feature_types(vocab: FormulaVocab) -> list[SemanticType]:
    """Semantic type per feature token of a vocabulary (mask/pset builders
    consume this; unknown names fail closed)."""

    return [_feature_type(name) for name in vocab.feature_names]
