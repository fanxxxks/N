"""T2-01 contracts: AST canonicalization.

The canonical form is the deduplication unit of the search pipeline:

* ADD/MUL children are sorted by their canonical structural key
  (commutativity: ``x + y`` and ``y + x`` are the same formula);
* double NEG and ``ABS(NEG(x))`` collapse;
* identity formulas that produce a constant (``SUB(x, x)``,
  ``DIV(x, x)``, ``ADD(x, NEG(x))``) are **degenerate**: they carry no
  cross-sectional information and are never evaluated;
* ``GATE(c, x, x)`` collapses to ``x``;
* degeneracy propagates through enclosing operators;
* the canonical form is idempotent and structurally hashed.
"""

from __future__ import annotations

import pytest

from ashare_model.ir import (
    Binary,
    Feature,
    FormulaSyntaxError,
    Ternary,
    Unary,
    canonical_tokens,
    canonicalize,
    decode,
    encode,
    is_degenerate,
    structural_hash,
    structural_key,
)
from ashare_model.ops import OPS_CONFIG
from ashare_model.vocab import FORMULA_VOCAB


def _op(name: str) -> int:
    return FORMULA_VOCAB.operator_offset + [
        cfg[0] for cfg in OPS_CONFIG
    ].index(name)


def _eos() -> int:
    return FORMULA_VOCAB.eos_token_id


def _tok(name: str) -> list[int]:
    """Bare feature token (no terminator) for building postfix sequences."""
    return [FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name)]


def _tokens(*names: str) -> list[int]:
    """Token sequence for bare feature names (postfix, EOS-terminated)."""
    out = [_tok(n)[0] for n in names]
    return out + [_eos()]


# --- commutativity: ADD/MUL child ordering -------------------------------


def test_add_commutes_canonically():
    a, b = _tok("RET_1"), _tok("RET_5")
    add_ab = a + b + [_op("ADD")]
    add_ba = b + a + [_op("ADD")]
    assert canonicalize(decode(add_ab)) == canonicalize(decode(add_ba))
    assert structural_key(canonicalize(decode(add_ab))) == structural_key(
        canonicalize(decode(add_ba))
    )
    assert canonical_tokens(add_ab) == canonical_tokens(add_ba)


def test_mul_commutes_canonically():
    a, b = _tok("RET_1"), _tok("RET_10")
    mul_ab = a + b + [_op("MUL")]
    mul_ba = b + a + [_op("MUL")]
    assert canonical_tokens(mul_ab) == canonical_tokens(mul_ba)


def test_sub_is_not_reordered():
    a, b = _tok("RET_1"), _tok("RET_5")
    sub_ab = a + b + [_op("SUB")]
    sub_ba = b + a + [_op("SUB")]
    assert canonical_tokens(sub_ab) != canonical_tokens(sub_ba)
    assert structural_key(decode(sub_ab)) != structural_key(decode(sub_ba))


def test_sorting_is_recursive():
    # ADD(ADD(a, b), c) and ADD(ADD(b, a), c) canonicalize to the same AST,
    # but different association shapes stay distinct (no reassociation).
    a, b, c = _tok("RET_1"), _tok("RET_5"), _tok("RET_10")
    left_nested = a + b + [_op("ADD")] + c + [_op("ADD")]
    left_nested_swapped = b + a + [_op("ADD")] + c + [_op("ADD")]
    right_nested = a + b + c + [_op("ADD")] + [_op("ADD")]
    assert canonical_tokens(left_nested) == canonical_tokens(left_nested_swapped)
    assert canonical_tokens(left_nested) != canonical_tokens(right_nested)


def test_sorting_mixes_operator_shapes_stably():
    # ADD(NEG(a), b) and ADD(b, NEG(a)) canonicalize to the same AST.
    a, b = _tok("RET_1"), _tok("RET_5")
    neg = [_op("NEG")]
    left = a + neg + b + [_op("ADD")]
    right = b + a + neg + [_op("ADD")]
    assert canonical_tokens(left) == canonical_tokens(right)


# --- double NEG and ABS ---------------------------------------------------


def test_double_neg_collapses():
    a = _tokens("RET_1")[:-1]
    neg = [_op("NEG")]
    ast = decode(a + neg + neg)
    assert canonicalize(ast) == Feature("RET_1")
    assert canonical_tokens(a + neg + neg) == _tokens("RET_1")


def test_abs_of_neg_collapses():
    a = _tokens("RET_1")[:-1]
    neg = [_op("NEG")]
    abs_ = [_op("ABS")]
    assert canonical_tokens(a + neg + abs_) == canonical_tokens(a + abs_)
    assert canonicalize(decode(a + neg + abs_)) == Unary("ABS", Feature("RET_1"))


# --- degenerate (constant-producing) formulas ----------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda a: a + a + [_op("SUB")],  # SUB(x, x) == 0
        lambda a: a + a + [_op("DIV")],  # DIV(x, x) == 1
        lambda a: a + a + [_op("NEG")] + [_op("ADD")],  # ADD(x, NEG(x)) == 0
        lambda a: a + [_op("NEG")] + a + [_op("ADD")],  # ADD(NEG(x), x) == 0
    ],
    ids=["sub-self", "div-self", "add-neg", "neg-add"],
)
def test_identity_formulas_are_degenerate(build):
    a = _tokens("RET_1")[:-1]
    tokens = build(a)
    assert is_degenerate(decode(tokens))
    assert canonicalize(decode(tokens)) is None
    assert canonical_tokens(tokens) is None


def test_gate_with_equal_branches_collapses_to_operand():
    a, b = (_tokens(n)[:-1] for n in ("RET_1", "RET_5"))
    gate = a + b + b + [_op("GATE")]
    assert canonicalize(decode(gate)) == Feature("RET_5")
    assert canonical_tokens(gate) == _tokens("RET_5")


def test_gate_with_distinct_branches_stays():
    a, b = (_tokens(n)[:-1] for n in ("RET_1", "RET_5"))
    # GATE(cond=a, x=b, y=a): distinct branches keep the formula.
    gate = a + b + a + [_op("GATE")]
    assert canonicalize(decode(gate)) is not None
    assert not is_degenerate(decode(gate))


def test_degeneracy_propagates_through_unary():
    a = _tokens("RET_1")[:-1]
    tokens = a + a + [_op("SUB")] + [_op("MA20")]
    assert canonicalize(decode(tokens)) is None


def test_degeneracy_propagates_through_binary():
    a, b = (_tokens(n)[:-1] for n in ("RET_1", "RET_5"))
    tokens = a + a + [_op("SUB")] + b + [_op("ADD")]
    assert canonicalize(decode(tokens)) is None


def test_degeneracy_does_not_claim_distinct_operands():
    a, b = (_tokens(n)[:-1] for n in ("RET_1", "RET_5"))
    for op in ("SUB", "DIV"):
        tokens = a + b + [_op(op)]
        assert not is_degenerate(decode(tokens))
        assert canonical_tokens(tokens) is not None


# --- idempotence and hashing ---------------------------------------------


def test_canonicalize_is_idempotent():
    a, b, c = (_tokens(n)[:-1] for n in ("RET_1", "RET_5", "RET_10"))
    samples = [
        a + b + [_op("ADD")],
        b + a + [_op("ADD")],
        a + [_op("NEG")] + [_op("NEG")],
        a + b + [_op("ADD")] + [_op("NEG")] + [_op("NEG")],
        a + b + [_op("MUL")] + c + [_op("MUL")],
        c + a + b + [_op("MUL")] + [_op("MUL")],
        a + b + [_op("CORR20")],
        a + b + b + [_op("GATE")],
    ]
    for tokens in samples:
        ast = decode(tokens)
        once = canonicalize(ast)
        assert once is not None, tokens
        assert canonicalize(once) == once, tokens


def test_structural_hash_is_stable_and_distinct():
    a, b = (_tokens(n)[:-1] for n in ("RET_1", "RET_5"))
    add_ab = canonicalize(decode(a + b + [_op("ADD")]))
    add_ba = canonicalize(decode(b + a + [_op("ADD")]))
    sub_ab = canonicalize(decode(a + b + [_op("SUB")]))
    assert add_ab is not None and add_ba is not None and sub_ab is not None
    assert structural_hash(add_ab) == structural_hash(add_ba)
    assert structural_hash(add_ab) != structural_hash(sub_ab)
    assert len(structural_hash(add_ab)) == 12
    assert structural_hash(add_ab) == structural_hash(add_ba)  # deterministic


def test_structural_key_is_bijective_on_canonical_asts():
    a, b = (_tokens(n)[:-1] for n in ("RET_1", "RET_5"))
    forms = [
        decode(a + b + [_op("ADD")]),
        decode(a + b + [_op("SUB")]),
        decode(a + b + [_op("MUL")]),
        decode(a + b + [_op("DIV")]),
        decode(a + [_op("NEG")]),
        decode(b + [_op("NEG")]),
    ]
    keys = {structural_key(f) for f in forms}
    assert len(keys) == len(forms)


# --- integration with the existing grammar --------------------------------


def test_canonical_tokens_rejects_invalid_and_degenerate():
    with pytest.raises(FormulaSyntaxError):
        canonical_tokens([_eos()])  # invalid: EOS on empty stack
    assert canonical_tokens([1, 1, _op("SUB")]) is None  # degenerate
    assert canonical_tokens(_tokens("RET_1")) is not None


def test_canonical_form_stays_inside_the_grammar():
    # The canonical bytecode must itself decode to the canonical AST.
    a, b = (_tokens(n)[:-1] for n in ("RET_1", "RET_5"))
    tokens = b + a + [_op("ADD")]
    canonical = canonical_tokens(tokens)
    assert canonical is not None
    assert decode(canonical) == canonicalize(decode(tokens))
    assert encode(decode(canonical)) == canonical


def test_existing_grammar_contracts_unchanged():
    # encode(decode(tokens)) stays the canonical bytecode of the same AST.
    add = _op("ADD")
    assert encode(decode([1, 2, add])) == [1, 2, add, _eos()]
    assert decode(encode(decode([1, 2, add]))) == decode([1, 2, add])
