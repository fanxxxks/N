"""Formula intermediate representation (AST) and the postfix bytecode grammar.

The AST is the **single source of truth** for formula semantics.  Postfix
token sequences are only two things: the StackVM bytecode and the model
serialization format.  Every consumer that needs formula semantics decodes
bytecode to the AST once and derives everything (infix text, operator
coverage, deduplication) from the AST.

Grammar (postfix, stack-only; the legacy ``open_slots`` hybrid is gone):

* ``PAD``   — only after ``EOS`` (termination).  Legacy sequences that pad
  without an explicit EOS are accepted only when the stack is exactly 1
  (the formula is complete), and act as an implicit terminator.
* ``EOS``   — only when ``stack == 1``; terminates the formula.
* feature  — ``stack + 1``.
* unary    — requires ``stack >= 1``; stack unchanged.
* binary   — requires ``stack >= 2``; ``stack - 1`` after execution.
* ternary  — requires ``stack >= 3``; ``stack - 2`` after execution.
* any token after termination that is not ``PAD`` is invalid.

``decode``/``encode`` are exact inverses: ``encode(decode(tokens))`` is the
canonical bytecode (EOS-terminated) of the same AST, and
``decode(encode(ast))`` returns an equal AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

from .ops import OPS_CONFIG
from .vocab import FormulaVocab


class FormulaSyntaxError(ValueError):
    """The token sequence is not a valid postfix formula under the grammar."""


@dataclass(frozen=True)
class Feature:
    """A raw factor reference ``Feature(name)``."""

    name: str


@dataclass(frozen=True)
class Unary:
    op: str
    operand: "FormulaIR"


@dataclass(frozen=True)
class Binary:
    op: str
    left: "FormulaIR"
    right: "FormulaIR"


@dataclass(frozen=True)
class Ternary:
    op: str
    a: "FormulaIR"
    b: "FormulaIR"
    c: "FormulaIR"


FormulaIR = Union[Feature, Unary, Binary, Ternary]

# Operator name -> arity, from the single operator table (ops.OPS_CONFIG).
_OPERATOR_ARITY = {name: arity for name, _, arity in OPS_CONFIG}


def _arity_of(op: str) -> int:
    try:
        return _OPERATOR_ARITY[op]
    except KeyError as exc:
        raise FormulaSyntaxError(f"unknown operator {op!r}") from exc


def decode(tokens: Iterable[int], vocab: FormulaVocab | None = None) -> FormulaIR:
    """Parse postfix bytecode into the AST, or raise :class:`FormulaSyntaxError`.

    Rules are the grammar above.  ``PAD`` at ``stack == 1`` is accepted as a
    legacy implicit terminator so pre-EOS artifacts keep loading; anything
    else that violates the grammar is rejected.
    """

    vocab = vocab or _default_vocab()
    stack: list[FormulaIR] = []
    terminated = False
    for raw_token in tokens:
        token = int(raw_token)
        if token == vocab.pad_token_id:
            if terminated:
                continue
            if len(stack) != 1:
                raise FormulaSyntaxError("PAD before the formula is complete")
            terminated = True
            continue
        if terminated:
            raise FormulaSyntaxError("token after EOS/PAD termination")
        if vocab.eos_token_id is not None and token == vocab.eos_token_id:
            if len(stack) != 1:
                raise FormulaSyntaxError(
                    f"EOS requires exactly one value on the stack, got {len(stack)}"
                )
            terminated = True
            continue
        if token < vocab.operator_offset:
            feature_idx = token - vocab.feature_offset
            if feature_idx < 0 or feature_idx >= len(vocab.feature_names):
                raise FormulaSyntaxError(f"feature token out of range: {token}")
            stack.append(Feature(vocab.feature_names[feature_idx]))
            continue
        op_index = token - vocab.operator_offset
        if op_index < 0 or op_index >= len(vocab.operator_names):
            raise FormulaSyntaxError(f"operator token out of range: {token}")
        op = vocab.operator_names[op_index]
        arity = _arity_of(op)
        if len(stack) < arity:
            raise FormulaSyntaxError(
                f"{op} needs {arity} operand(s), got {len(stack)}"
            )
        args = [stack.pop() for _ in range(arity)]
        args.reverse()
        if arity == 1:
            stack.append(Unary(op, args[0]))
        elif arity == 2:
            stack.append(Binary(op, args[0], args[1]))
        else:
            stack.append(Ternary(op, args[0], args[1], args[2]))
    if len(stack) != 1:
        raise FormulaSyntaxError(
            "formula must leave exactly one value on the stack, "
            f"got {len(stack)}"
        )
    return stack[0]


def encode(ir: FormulaIR, vocab: FormulaVocab | None = None) -> list[int]:
    """Serialize the AST to canonical postfix bytecode (EOS-terminated)."""

    vocab = vocab or _default_vocab()
    tokens: list[int] = []

    def emit(node: FormulaIR) -> None:
        if isinstance(node, Feature):
            tokens.append(vocab.feature_offset + vocab.feature_names.index(node.name))
        elif isinstance(node, Unary):
            emit(node.operand)
            tokens.append(vocab.operator_offset + vocab.operator_names.index(node.op))
        elif isinstance(node, Binary):
            emit(node.left)
            emit(node.right)
            tokens.append(vocab.operator_offset + vocab.operator_names.index(node.op))
        else:
            emit(node.a)
            emit(node.b)
            emit(node.c)
            tokens.append(vocab.operator_offset + vocab.operator_names.index(node.op))

    emit(ir)
    if vocab.eos_token_id is not None:
        tokens.append(vocab.eos_token_id)
    return tokens


def infix(ir: FormulaIR) -> str:
    """Render the AST as human-readable infix text."""

    if isinstance(ir, Feature):
        return ir.name
    if isinstance(ir, Unary):
        return f"{ir.op}({infix(ir.operand)})"
    if isinstance(ir, Binary):
        return f"({infix(ir.left)} {ir.op} {infix(ir.right)})"
    return f"{ir.op}({infix(ir.a)}, {infix(ir.b)}, {infix(ir.c)})"


def operator_names(ir: FormulaIR) -> set[str]:
    """Distinct operator names used anywhere in the AST."""

    if isinstance(ir, Feature):
        return set()
    found = {ir.op}
    for child in _children(ir):
        found |= operator_names(child)
    return found


def _children(ir: FormulaIR) -> tuple[FormulaIR, ...]:
    if isinstance(ir, Unary):
        return (ir.operand,)
    if isinstance(ir, Binary):
        return (ir.left, ir.right)
    if isinstance(ir, Ternary):
        return (ir.a, ir.b, ir.c)
    return ()


def postfix_valid(tokens: Iterable[int], vocab: FormulaVocab | None = None) -> bool:
    """Pure stack-rule validity check (no AST construction).

    Returns ``True`` exactly when :func:`decode` would succeed.  Kept
    separate so the sampling mask and the VM share one cheap validator.
    """

    vocab = vocab or _default_vocab()
    stack = 0
    terminated = False
    for raw_token in tokens:
        token = int(raw_token)
        if token == vocab.pad_token_id:
            if terminated:
                continue
            if stack != 1:
                return False
            terminated = True
            continue
        if terminated:
            return False
        if vocab.eos_token_id is not None and token == vocab.eos_token_id:
            if stack != 1:
                return False
            terminated = True
            continue
        if token < vocab.operator_offset:
            if not (vocab.feature_offset <= token < vocab.operator_offset):
                return False
            stack += 1
            continue
        op_index = token - vocab.operator_offset
        if not (0 <= op_index < len(vocab.operator_names)):
            return False
        arity = _OPERATOR_ARITY[vocab.operator_names[op_index]]
        if stack < arity:
            return False
        stack = stack - arity + 1
    return stack == 1


def formula_length(tokens: Iterable[int], vocab: FormulaVocab | None = None) -> int:
    """Effective length of a formula sequence.

    Counts every token up to and including the terminator: the EOS token
    when present, otherwise the first PAD (a legacy implicit terminator).
    A sequence without any terminator counts its full length.  Used by the
    training runtime to report formula-length statistics.
    """

    vocab = vocab or _default_vocab()
    length = 0
    for raw_token in tokens:
        token = int(raw_token)
        length += 1
        if token == vocab.pad_token_id:
            return length
        if vocab.eos_token_id is not None and token == vocab.eos_token_id:
            return length
    return length


def _default_vocab() -> FormulaVocab:
    # Imported lazily so the module stays importable without torch-side
    # initialization order concerns (vocab imports ops only).
    from .vocab import FORMULA_VOCAB

    return FORMULA_VOCAB
