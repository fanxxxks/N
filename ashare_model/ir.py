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

Canonicalization (T2-01): :func:`canonicalize` reduces an AST to its
canonical form — the deduplication unit of the search pipeline.  Rules:

* ADD/MUL children are sorted by their canonical structural key
  (floating-point addition and multiplication are commutative, so
  ``x + y`` and ``y + x`` are the same formula);
* ``NEG(NEG(x))`` collapses to ``x`` and ``ABS(NEG(x))`` to ``ABS(x)``;
* identity formulas producing a constant are **degenerate** — they carry
  no cross-sectional information and are never evaluated:
  ``SUB(x, x)`` (zero), ``DIV(x, x)`` and ``CORR*(x, x)`` (one),
  ``ADD(x, NEG(x))`` (zero);
* ``GATE(c, x, x)`` collapses to ``x``;
* degeneracy propagates through enclosing operators.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Union

from .ops import OPERATOR_ARITY
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


def _arity_of(op: str) -> int:
    try:
        return OPERATOR_ARITY[op]
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


def feature_names(ir: FormulaIR) -> set[str]:
    """Distinct feature names referenced anywhere in the AST.

    The traceability hook for data-credibility tiering (P2): a formula's
    features are resolved here once, then each feature maps to its tier.
    """

    if isinstance(ir, Feature):
        return {ir.name}
    found: set[str] = set()
    for child in _children(ir):
        found |= feature_names(child)
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
        arity = OPERATOR_ARITY[vocab.operator_names[op_index]]
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


# --- canonicalization (T2-01) ---------------------------------------------

# Binary operators whose ``(x, x)`` application is a constant: SUB (zero),
# DIV (one) and the CORR window family (one).  Such formulas carry no
# cross-sectional information and are never evaluated (degenerate).
_SELF_CONSTANT_BINARY = frozenset({"SUB", "DIV", "CORR5", "CORR10", "CORR20", "CORR60"})
# Floating-point ADD/MUL are commutative, so their children are sorted.
_COMMUTATIVE_BINARY = frozenset({"ADD", "MUL"})


def _same(left: FormulaIR, right: FormulaIR) -> bool:
    """Structural equality of two canonical ASTs (key comparison)."""
    return structural_key(left) == structural_key(right)


def _is_neg_of(x: FormulaIR, y: FormulaIR) -> bool:
    """True when ``x`` is ``NEG(y)`` with canonical operands."""
    return isinstance(x, Unary) and x.op == "NEG" and _same(x.operand, y)


def canonicalize(ir: FormulaIR) -> FormulaIR | None:
    """Reduce an AST to its canonical form, or ``None`` for a degenerate
    (constant-producing) formula.

    Rules (see the module docstring): children are canonicalized first
    (post-order); ADD/MUL children are sorted by structural key; double
    NEG and ``ABS(NEG(x))`` collapse; ``SUB/DIV/CORR*(x, x)`` and
    ``ADD(x, NEG(x))`` are degenerate; ``GATE(c, x, x)`` collapses to
    ``x``; degeneracy propagates through enclosing operators.  The
    canonical form is idempotent.
    """

    if isinstance(ir, Feature):
        return ir
    if isinstance(ir, Unary):
        operand = canonicalize(ir.operand)
        if operand is None:
            return None  # degenerate operand propagates
        if ir.op == "NEG":
            if isinstance(operand, Unary) and operand.op == "NEG":
                return operand.operand  # NEG(NEG(x)) -> x
            return Unary("NEG", operand)
        if ir.op == "ABS" and isinstance(operand, Unary) and operand.op == "NEG":
            return Unary("ABS", operand.operand)  # ABS(NEG(x)) -> ABS(x)
        return Unary(ir.op, operand)
    if isinstance(ir, Binary):
        left = canonicalize(ir.left)
        right = canonicalize(ir.right)
        if left is None or right is None:
            return None
        if ir.op in _SELF_CONSTANT_BINARY and _same(left, right):
            return None  # SUB(x, x) == 0, DIV(x, x) == CORR*(x, x) == 1
        if ir.op == "ADD" and (_is_neg_of(left, right) or _is_neg_of(right, left)):
            return None  # ADD(x, NEG(x)) == 0
        if ir.op in _COMMUTATIVE_BINARY and structural_key(right) < structural_key(left):
            left, right = right, left
        return Binary(ir.op, left, right)
    a = canonicalize(ir.a)
    b = canonicalize(ir.b)
    c = canonicalize(ir.c)
    if a is None or b is None or c is None:
        return None
    if ir.op == "GATE" and _same(b, c):
        return b  # GATE(cond, x, x) == x
    return Ternary(ir.op, a, b, c)


def structural_key(ir: FormulaIR) -> str:
    """Canonical string key of an AST (children in canonical order).

    Distinct canonical ASTs map to distinct keys, so the key is a valid
    equality proxy and the basis of the structural hash.
    """

    if isinstance(ir, Feature):
        return f"f:{ir.name}"
    if isinstance(ir, Unary):
        return f"u:{ir.op}:{structural_key(ir.operand)}"
    if isinstance(ir, Binary):
        return (
            f"b:{ir.op}:{structural_key(ir.left)}:{structural_key(ir.right)}"
        )
    return (
        f"t:{ir.op}:{structural_key(ir.a)}:{structural_key(ir.b)}:"
        f"{structural_key(ir.c)}"
    )


def structural_hash(ir: FormulaIR) -> str:
    """Short deterministic hash of the canonical structural key (12 hex)."""

    return hashlib.sha256(structural_key(ir).encode("utf-8")).hexdigest()[:12]


def is_degenerate(ir: FormulaIR) -> bool:
    """True when the formula is a degenerate constant (canonicalizes to
    ``None``) — it carries no cross-sectional information."""

    return canonicalize(ir) is None


def canonical_ast(
    tokens: Iterable[int], vocab: FormulaVocab | None = None
) -> FormulaIR | None:
    """Canonical AST of a token sequence, or ``None``.

    ``None`` covers both structurally invalid sequences and degenerate
    (constant-producing) formulas: neither is ever evaluated.  Raises
    nothing — callers that need the strict grammar contract use
    :func:`decode` directly.
    """

    try:
        return canonicalize(decode(tokens, vocab))
    except FormulaSyntaxError:
        return None


def canonical_tokens(
    tokens: Iterable[int], vocab: FormulaVocab | None = None
) -> list[int] | None:
    """Canonical EOS-terminated bytecode of the canonical AST.

    Raises :class:`FormulaSyntaxError` for structurally invalid sequences
    (like :func:`decode`) and returns ``None`` for degenerate
    (constant-producing) formulas.  Canonical bytecode is the deduplication
    key of the search pipeline: any two inputs with the same canonical
    bytecode are the same formula.
    """

    canonical = canonicalize(decode(tokens, vocab))
    if canonical is None:
        return None
    return encode(canonical, vocab)


def _default_vocab() -> FormulaVocab:
    # Imported lazily so the module stays importable without torch-side
    # initialization order concerns (vocab imports ops only).
    from .vocab import FORMULA_VOCAB

    return FORMULA_VOCAB
