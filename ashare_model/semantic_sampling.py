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

import torch

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


def type_id(t: SemanticType) -> int:
    """Tensor-friendly id of a semantic type (1-based; 0 stays reserved
    for 'empty stack slot' in the mask/state layers)."""

    return list(SemanticType).index(t) + 1


def from_id(i: int) -> SemanticType:
    """Inverse of :func:`type_id` (0 is 'empty' and never valid)."""

    if i < 1:
        raise ValueError(f"semantic type id {i} is the empty marker")
    return list(SemanticType)[i - 1]


def fixed_output(name: str) -> SemanticType | None:
    """The contract-committed output type of operators whose result does
    not depend on the argument family (cross-sectional ratios/correlations,
    volatility scales, the JUMP detector); ``None`` for family-preserving
    operators (output = shared input family)."""

    if name in _OUTPUT_CROSS_SECTIONAL:
        return SemanticType.CROSS_SECTIONAL_SIGNAL
    if name in _OUTPUT_RETURN_LIKE:
        return SemanticType.RETURN_LIKE
    if name in _OUTPUT_EVENT:
        return SemanticType.BOOLEAN_EVENT_SIGNAL
    return None


def is_gate(name: str) -> bool:
    """Whether ``name`` is the GATE operator (its condition argument is
    exempt from the continuous-type rule and its output is the branch
    family)."""

    return name == "GATE"


def same_type_args(name: str) -> bool:
    """Whether the contract requires the (branch) arguments of ``name``
    to share one semantic type (family-preserving operators)."""

    return name in _FAMILY_PRESERVING


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


# Tensor tables used to advance the shared sampler state.  The key carries
# the vocabulary and authored feature types, so a metadata monkeypatch or a
# different toy/research-domain vocabulary cannot reuse stale token offsets.
_STATE_TABLE_CACHE: dict[
    tuple,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _state_tables(
    vocab: FormulaVocab, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    authored_types = tuple(feature_types(vocab))
    fingerprint = (
        tuple(vocab.feature_names),
        tuple(vocab.operator_names),
        authored_types,
        device.type,
        device.index,
    )
    cached = _STATE_TABLE_CACHE.get(fingerprint)
    if cached is not None:
        return cached

    arities = torch.zeros(vocab.size, dtype=torch.long, device=device)
    feature_type_ids = torch.zeros(
        vocab.size, dtype=torch.long, device=device
    )
    fixed_output_ids = torch.full(
        (vocab.size,), -1, dtype=torch.long, device=device
    )
    gate_tokens = torch.zeros(vocab.size, dtype=torch.bool, device=device)

    for index, semantic_type in enumerate(authored_types):
        feature_type_ids[vocab.feature_offset + index] = type_id(semantic_type)
    for index, name in enumerate(vocab.operator_names):
        try:
            arity = OPS_BY_NAME[name][1]
        except KeyError:
            raise ValueError(
                f"operator {name!r} has no implementation/signature in OPS_BY_NAME"
            ) from None
        token = vocab.operator_offset + index
        arities[token] = arity
        output = fixed_output(name)
        if output is not None:
            fixed_output_ids[token] = type_id(output)
        gate_tokens[token] = is_gate(name)

    tables = (arities, feature_type_ids, fixed_output_ids, gate_tokens)
    _STATE_TABLE_CACHE.clear()
    _STATE_TABLE_CACHE[fingerprint] = tables
    return tables


def advance_stack_state(
    action: torch.Tensor,
    stack_sizes: torch.Tensor,
    stack_types: torch.Tensor,
    done: torch.Tensor,
    *,
    vocab: FormulaVocab | None = None,
) -> None:
    """Advance structural and semantic tensor state for one sampled token.

    This is the one vocabulary-aware state transition used by RL and the
    random baseline.  Features push their authored type; an operator pops
    its arguments and pushes either its fixed output or its preserved input
    family (GATE uses its first branch); EOS/PAD latch ``done``.  The action
    mask is responsible for signature/arity legality before this transition.
    """

    vocab = vocab or FORMULA_VOCAB
    if action.shape != stack_sizes.shape or done.shape != stack_sizes.shape:
        raise ValueError("action, stack_sizes and done must have the same shape")
    if stack_types.ndim != 2 or stack_types.shape[0] != action.shape[0]:
        raise ValueError(
            "stack_types must be a [batch, capacity] tensor aligned with action"
        )
    if (
        action.device != stack_sizes.device
        or stack_types.device != stack_sizes.device
        or done.device != stack_sizes.device
    ):
        raise ValueError("all sampling-state tensors must share one device")
    if action.dtype != torch.long or stack_types.dtype != torch.long:
        raise ValueError("action and stack_types must be LongTensors")
    if bool(((action < 0) | (action >= vocab.size)).any()):
        raise ValueError("sampled action is outside the vocabulary")

    arities, feature_type_ids, fixed_output_ids, gate_tokens = _state_tables(
        vocab, action.device
    )
    is_pad = action == vocab.pad_token_id
    if vocab.eos_token_id is None:
        is_eos = torch.zeros_like(is_pad)
    else:
        is_eos = action == vocab.eos_token_id
    is_feature = (action >= vocab.feature_offset) & (
        action < vocab.operator_offset
    )
    operator_end = vocab.operator_offset + len(vocab.operator_names)
    is_operator = (action >= vocab.operator_offset) & (action < operator_end)
    if bool((~(is_pad | is_eos | is_feature | is_operator)).any()):
        raise ValueError("sampled action does not map to a vocabulary token kind")

    old_sizes = stack_sizes.clone()
    if bool((is_feature & (old_sizes >= stack_types.shape[1])).any()):
        raise ValueError("feature push would exceed stack_types capacity")
    action_arities = arities[action]
    new_sizes = torch.where(is_feature, old_sizes + 1, old_sizes)
    new_sizes = torch.where(
        is_operator,
        old_sizes - action_arities + 1,
        new_sizes,
    ).clamp(min=0)
    stack_sizes.copy_(new_sizes)
    done.copy_(done | is_eos | is_pad)

    if is_feature.any():
        rows = torch.nonzero(is_feature, as_tuple=False).flatten()
        stack_types[rows, old_sizes[rows]] = feature_type_ids[action[rows]]
    if is_operator.any():
        rows = torch.nonzero(is_operator, as_tuple=False).flatten()
        arity = action_arities[rows]
        first_slot = (old_sizes[rows] - arity).clamp(min=0)
        family_slot = first_slot + gate_tokens[action[rows]].long()
        family_type = stack_types[rows, family_slot]
        fixed = fixed_output_ids[action[rows]]
        output_type = torch.where(fixed >= 0, fixed, family_type)
        stack_types[rows, first_slot] = output_type
