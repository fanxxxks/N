"""Semantic-type sampling lattice (P7-E, docs/p7_semantic_types_contract.md).

The shared legality layer for every searcher: ``formula_types_legal`` is
the single type checker (RL/Random/TPE via the action mask, GP via the
typed pset) and reads its operator signatures from the P7-D2
``OPERATOR_REGISTRY`` — no second hardcoded type table is allowed (the
registry-driven test mutates a signature and expects the checker to
follow).
"""

from __future__ import annotations

import pytest

from ashare_model.feature_metadata import SemanticType
from ashare_model.semantic_sampling import (
    formula_types_legal,
    resolve_output,
    signature_legal,
)
from ashare_model.vocab import FORMULA_VOCAB

_V = FORMULA_VOCAB


def _seq(*names: str) -> list[int]:
    """Token ids from token names (EOS appended explicitly by callers)."""
    table = {
        "PAD": _V.pad_token_id,
        "EOS": _V.eos_token_id,
        **{n: _V.feature_offset + _V.feature_names.index(n) for n in _V.feature_names},
        **{n: _V.operator_offset + _V.operator_names.index(n) for n in _V.operator_names},
    }
    return [table[n] for n in names]


# --- legal formulas ----------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [
        # Same-family arithmetic.
        ("RET_1", "RET_5", "SUB", "EOS"),
        # Cross-family DIV (Amihud-style) is legal.
        ("PE_TTM", "VOL_20", "DIV", "EOS"),
        # Event signal as the GATE condition is its intended use.
        ("LIMIT_UP_EVENT", "RET_1", "RET_5", "GATE", "EOS"),
        # CORR across two continuous families is legal.
        ("CLOSE_POSITION", "ILLIQ_20", "CORR20", "EOS"),
        # Industry neutralize on a cross-sectional signal.
        ("IND_REL_RET_5", "CS_NEUTRALIZE", "EOS"),
        # Volatility of returns is return-scale.
        ("RET_1", "STD20", "EOS"),
        # Family-preserving smoothing.
        ("VOL_20", "MA10", "TS_RANK10", "EOS"),
    ],
)
def test_legal_formulas(names):
    assert formula_types_legal(_seq(*names))


def test_incomplete_formula_is_illegal():
    # Stack of 2 at EOS: not a single well-typed result.
    assert not formula_types_legal(_seq("RET_1", "RET_5", "EOS"))


# --- illegal formulas (contract §6.1 examples) --------------------------------


@pytest.mark.parametrize(
    "names",
    [
        # Event signals may not participate in division.
        ("LIMIT_UP_EVENT", "LIMIT_DOWN_EVENT", "DIV", "EOS"),
        # Event signals may not enter windowed reducers.
        ("LIMIT_UP_EVENT", "MA20", "EOS"),
        # Industry neutralization only accepts cross-sectional signals.
        ("PE_TTM", "CS_NEUTRALIZE", "EOS"),
        # GATE branches must share one type.
        ("LIMIT_UP_EVENT", "RET_1", "PE_TTM", "GATE", "EOS"),
        # CORR arguments must be continuous.
        ("LIMIT_UP_EVENT", "RET_1", "CORR20", "EOS"),
        # Mixed-branch arithmetic within GATE stays illegal.
        ("TURNOVER", "RET_1", "PE_TTM", "ADD", "GATE", "EOS"),
    ],
)
def test_illegal_formulas(names):
    assert not formula_types_legal(_seq(*names))


# --- output resolution (contract §1 table) -------------------------------------


def test_output_resolution():
    assert resolve_output("DIV", (SemanticType.RETURN_LIKE, SemanticType.VOLUME_LIKE)) is SemanticType.CROSS_SECTIONAL_SIGNAL
    assert resolve_output("CORR20", (SemanticType.PRICE_LIKE, SemanticType.VOLUME_LIKE)) is SemanticType.CROSS_SECTIONAL_SIGNAL
    assert resolve_output("STD20", (SemanticType.RETURN_LIKE,)) is SemanticType.RETURN_LIKE
    assert resolve_output("DELTA5", (SemanticType.PRICE_LIKE,)) is SemanticType.RETURN_LIKE
    assert resolve_output("JUMP", (SemanticType.RETURN_LIKE,)) is SemanticType.BOOLEAN_EVENT_SIGNAL
    # Family-preserving ops return the (shared) input family.
    assert resolve_output("MA60", (SemanticType.FUNDAMENTAL_LIKE,)) is SemanticType.FUNDAMENTAL_LIKE
    assert resolve_output("ADD", (SemanticType.VOLUME_LIKE, SemanticType.VOLUME_LIKE)) is SemanticType.VOLUME_LIKE
    assert resolve_output("GATE", (SemanticType.BOOLEAN_EVENT_SIGNAL, SemanticType.RETURN_LIKE, SemanticType.RETURN_LIKE)) is SemanticType.RETURN_LIKE


def test_signature_legality_mirrors_registry_inputs():
    # CS_NEUTRALIZE: only cross-sectional (registry-recorded and enforced).
    assert signature_legal("CS_NEUTRALIZE", (SemanticType.CROSS_SECTIONAL_SIGNAL,))
    assert not signature_legal("CS_NEUTRALIZE", (SemanticType.PRICE_LIKE,))
    # Arity mismatch fails.
    with pytest.raises(ValueError):
        signature_legal("ADD", (SemanticType.RETURN_LIKE,))


def test_checker_is_registry_driven(monkeypatch):
    """No second type table: the checker reads each operator's allowed
    argument types from the registry, so widening a signature there flips
    the verdict (contract §3.4)."""
    from ashare_model.operator_registry import OPERATOR_REGISTRY

    seq = _seq("PE_TTM", "CS_NEUTRALIZE", "EOS")
    assert not formula_types_legal(seq)
    meta = OPERATOR_REGISTRY["CS_NEUTRALIZE"]
    monkeypatch.setitem(
        OPERATOR_REGISTRY,
        "CS_NEUTRALIZE",
        meta.__class__(
            category=meta.category,
            inputs=(tuple(SemanticType),),
            output=meta.output,
            cost=meta.cost,
            stability=meta.stability,
        ),
    )
    assert formula_types_legal(seq)
