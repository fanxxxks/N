"""Unit tests for the FormulaIR AST and the stack-only postfix grammar."""

from __future__ import annotations

import pytest

from ashare_model.ir import (
    Binary,
    Feature,
    FormulaSyntaxError,
    Ternary,
    Unary,
    decode,
    encode,
    infix,
    operator_names,
    postfix_valid,
)
from ashare_model.ops import OPS_CONFIG
from ashare_model.vocab import FORMULA_VOCAB


def _op(name: str) -> int:
    return FORMULA_VOCAB.operator_offset + [
        cfg[0] for cfg in OPS_CONFIG
    ].index(name)


def _eos() -> int:
    return FORMULA_VOCAB.eos_token_id


def test_decode_feature_and_infix():
    assert decode([1]) == Feature(FORMULA_VOCAB.feature_names[0])
    assert infix(decode([1])) == FORMULA_VOCAB.feature_names[0]


def test_decode_binary_unary_ternary():
    add = _op("ADD")
    neg = _op("NEG")
    gate = _op("GATE")
    ast = decode([1, 2, add])
    assert ast == Binary("ADD", Feature("RET_1"), Feature("RET_5"))
    assert decode([1, neg]) == Unary("NEG", Feature("RET_1"))
    assert decode([1, 2, 3, gate]) == Ternary(
        "GATE", Feature("RET_1"), Feature("RET_5"), Feature("RET_10")
    )


def test_decode_ignores_padding_after_eos_and_stops():
    ast = decode([1, _eos(), 0, 0])
    assert ast == Feature("RET_1")
    # Legacy padding without EOS is accepted only at stack == 1.
    assert decode([1, 2, _op("ADD"), 0, 0]) == Binary(
        "ADD", Feature("RET_1"), Feature("RET_5")
    )


def test_decode_rejects_violations():
    with pytest.raises(FormulaSyntaxError):
        decode([])  # nothing on the stack
    with pytest.raises(FormulaSyntaxError):
        decode([_eos()])  # EOS with an empty stack
    with pytest.raises(FormulaSyntaxError):
        decode([1, 2])  # two values, no operator
    with pytest.raises(FormulaSyntaxError):
        decode([1, _op("ADD")])  # binary with one operand
    with pytest.raises(FormulaSyntaxError):
        decode([1, 2, 0])  # PAD before the formula is complete
    with pytest.raises(FormulaSyntaxError):
        decode([1, _eos(), 2])  # token after EOS is not PAD
    with pytest.raises(FormulaSyntaxError):
        decode([1, _eos(), _eos()])  # second EOS
    with pytest.raises(FormulaSyntaxError):
        decode([FORMULA_VOCAB.size])  # out-of-range token


def test_encode_appends_eos_and_roundtrips():
    add = _op("ADD")
    canonical = encode(decode([1, 2, add]))
    assert canonical == [1, 2, add, _eos()]
    assert decode(canonical) == decode([1, 2, add])
    # encode(decode(tokens)) is canonical even for legacy inputs.
    assert encode(decode([1, 2, add, 0, 0])) == [1, 2, add, _eos()]


def test_postfix_valid_matches_decode():
    good = [
        [1, _eos()],
        [1, 2, _op("ADD"), _eos()],
        [1, 2, 3, _op("GATE"), _eos()],
        [1, _op("MA20"), _eos()],
        [1, 0, 0],  # legacy padding at stack == 1
    ]
    bad = [
        [],
        [_eos()],
        [1, 2],
        [1, _op("ADD")],
        [1, 2, 0],
        [1, _eos(), 2],
        [1, 1, 1],
    ]
    for tokens in good:
        assert postfix_valid(tokens), tokens
        decode(tokens)  # must not raise
    for tokens in bad:
        assert not postfix_valid(tokens), tokens
        with pytest.raises(FormulaSyntaxError):
            decode(tokens)


def test_operator_names_collects_distinct_operators():
    add = _op("ADD")
    neg = _op("NEG")
    ast = decode([1, neg, 2, neg, add])  # ADD(NEG(RET_1), NEG(RET_5))
    assert operator_names(ast) == {"ADD", "NEG"}
    assert operator_names(Feature("RET_1")) == set()


def test_eos_grammar_roundtrip_all_arities():
    # Every operator family encodes and decodes through the EOS grammar.
    for name, _, arity in OPS_CONFIG:
        features = list(range(1, arity + 1))
        ast = decode(features + [_op(name), _eos()])
        assert operator_names(ast) == {name}
        assert encode(decode(encode(ast))) == encode(ast)
