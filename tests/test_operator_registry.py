"""Operator metadata registry (P7 D2, plan §6).

``OPS_CONFIG`` stays the single implementation/arity source; this registry
adds the descriptive metadata layer (category, per-argument input
semantics, output semantics, cost, numerical-stability notes) derived by
name/arity from ``OPS_CONFIG`` — never duplicated.  Fields are
descriptive only: Phase E's pre-registered contract will wire semantic
types into search legality.
"""

from __future__ import annotations

import pytest

from ashare_model.feature_metadata import ComputeCost, SemanticType
from ashare_model.operator_registry import (
    OPERATOR_REGISTRY,
    OperatorMeta,
    operator_meta_of,
)
from ashare_model.ops import OPS_CONFIG

_ANY = set(SemanticType)
_CONT = _ANY - {SemanticType.BOOLEAN_EVENT_SIGNAL}


def test_registry_covers_exactly_the_ops_config_names():
    assert set(OPERATOR_REGISTRY) == {name for name, _, _ in OPS_CONFIG}


def test_arity_is_derived_not_duplicated():
    for name, _, arity in OPS_CONFIG:
        meta = operator_meta_of(name)
        assert len(meta.inputs) == arity, name


def test_all_entries_are_complete():
    for name, meta in OPERATOR_REGISTRY.items():
        assert isinstance(meta, OperatorMeta), name
        assert meta.category in {
            "arithmetic",
            "time_series",
            "cross_sectional",
            "event",
        }, name
        assert isinstance(meta.cost, ComputeCost), name
        assert meta.stability, name
        for arg_types in meta.inputs:
            assert arg_types, name
            assert all(isinstance(t, SemanticType) for t in arg_types), name
        assert meta.output is None or isinstance(meta.output, SemanticType), name


def test_correlation_requires_two_continuous_signals():
    # The contract example: correlation needs two continuous (non-event)
    # signals; event signals may not enter it.
    for name in ("CORR5", "CORR10", "CORR20", "CORR60"):
        meta = operator_meta_of(name)
        assert len(meta.inputs) == 2
        for arg_types in meta.inputs:
            assert SemanticType.BOOLEAN_EVENT_SIGNAL not in arg_types, name


def test_industry_neutralize_accepts_cross_sectional_only():
    meta = operator_meta_of("CS_NEUTRALIZE")
    assert meta.inputs == ((SemanticType.CROSS_SECTIONAL_SIGNAL,),)
    assert meta.output == SemanticType.CROSS_SECTIONAL_SIGNAL


def test_cross_sectional_family_outputs_cross_sectional():
    for name in ("CS_RANK", "CS_ZSCORE", "CS_DEMEAN"):
        assert operator_meta_of(name).output == SemanticType.CROSS_SECTIONAL_SIGNAL


def test_jump_outputs_event_signal():
    meta = operator_meta_of("JUMP")
    assert meta.output == SemanticType.BOOLEAN_EVENT_SIGNAL
    # The z-score baseline needs a non-degenerate continuous input.
    assert SemanticType.BOOLEAN_EVENT_SIGNAL not in meta.inputs[0]


def test_arithmetic_ops_accept_any_semantic_type():
    for name in ("ADD", "SUB", "MUL", "DIV", "NEG", "ABS", "SIGN"):
        meta = operator_meta_of(name)
        for arg_types in meta.inputs:
            assert set(arg_types) == _ANY, name


def test_windowed_ops_reject_event_inputs():
    # Boolean/event signals may not flow through windowed numeric reducers.
    for name in (
        "MA5", "MA10", "MA20", "MA60",
        "STD5", "STD10", "STD20", "STD60",
        "TS_RANK5", "TS_RANK10", "TS_RANK20", "TS_RANK60",
        "DOWNVOL5", "DOWNVOL10", "DOWNVOL20", "DOWNVOL60",
        "DECAY", "DELTA5", "DELTA10", "DELTA20",
        "DELAY1", "MAX3",
    ):
        for arg_types in operator_meta_of(name).inputs:
            assert SemanticType.BOOLEAN_EVENT_SIGNAL not in arg_types, name


def test_volatility_family_outputs_return_scale():
    for name in ("STD5", "STD10", "STD20", "STD60",
                 "DOWNVOL5", "DOWNVOL10", "DOWNVOL20", "DOWNVOL60",
                 "DELTA5", "DELTA10", "DELTA20"):
        assert operator_meta_of(name).output == SemanticType.RETURN_LIKE, name


def test_scale_free_outputs_defer_to_phase_e():
    # TS_RANK / CORR outputs are scale-free bounded statistics with no
    # committed home in the six-type lattice yet; the Phase E contract
    # assigns them (recorded as None, never guessed).
    for name in ("TS_RANK5", "TS_RANK10", "TS_RANK20", "TS_RANK60",
                 "CORR5", "CORR10", "CORR20", "CORR60"):
        assert operator_meta_of(name).output is None, name


def test_unknown_operator_fails_loudly():
    with pytest.raises(ValueError, match="NOPE"):
        operator_meta_of("NOPE")
