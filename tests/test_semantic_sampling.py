"""Semantic-type sampling lattice (P7-E, docs/p7_semantic_types_contract.md).

The shared legality layer for every searcher: ``formula_types_legal`` is
the single type checker (RL/Random/TPE via the action mask, GP via the
typed pset) and reads its operator signatures from the P7-D2
``OPERATOR_REGISTRY`` — no second hardcoded type table is allowed (the
registry-driven test mutates a signature and expects the checker to
follow).
"""

from __future__ import annotations

import random as py_random

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
    from ashare_model.alphagpt import build_action_mask
    from ashare_model.semantic_sampling import type_id
    import torch

    seq = _seq("PE_TTM", "CS_NEUTRALIZE", "EOS")
    assert not formula_types_legal(seq)
    neutralize = _seq("CS_NEUTRALIZE")[0]
    stack_sizes = torch.ones(1, dtype=torch.long)
    stack_types = torch.zeros(1, 4, dtype=torch.long)
    stack_types[0, 0] = type_id(SemanticType.FUNDAMENTAL_LIKE)
    before = build_action_mask(
        stack_sizes,
        torch.zeros(1, dtype=torch.bool),
        0,
        4,
        FORMULA_VOCAB,
        stack_types=stack_types,
    )
    assert not torch.isfinite(before[0, neutralize])
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
    after = build_action_mask(
        stack_sizes,
        torch.zeros(1, dtype=torch.bool),
        0,
        4,
        FORMULA_VOCAB,
        stack_types=stack_types,
    )
    assert after[0, neutralize] == 0.0


# --- contract §6: end-to-end sampling properties (P7-E) ----------------------


def _eos():
    return FORMULA_VOCAB.eos_token_id


def test_random_sampling_is_type_legal():
    """§6.3: the shared mask path (random baseline / RL / TPE) only
    produces type-legal formulas."""
    from ashare_model.train_windows import sample_random_formulas

    samples = sample_random_formulas(seed=11, vocab=FORMULA_VOCAB, max_len=12, n=200)
    for seq in samples:
        assert formula_types_legal(seq), seq


def test_random_sampling_is_deterministic():
    from ashare_model.train_windows import sample_random_formulas

    first = sample_random_formulas(seed=3, vocab=FORMULA_VOCAB, max_len=10, n=64)
    second = sample_random_formulas(seed=3, vocab=FORMULA_VOCAB, max_len=10, n=64)
    assert first == second


def test_random_sampling_fails_closed_when_no_token_is_legal():
    """Contract §5: an empty effective feature domain must not trigger the
    legacy all-vocabulary uniform fallback (which emitted garbage tokens)."""
    from ashare_model.train_windows import sample_random_formulas

    with pytest.raises(RuntimeError, match=r"no legal token.*step 0.*rows \[0\]"):
        sample_random_formulas(
            seed=1,
            vocab=FORMULA_VOCAB,
            max_len=4,
            n=1,
            feature_ids=[],
        )


def test_action_mask_rejects_feature_without_authored_type():
    """Contract §5/§6.3: toy vocabularies must use registered feature
    names; missing metadata is an explicit boundary error."""
    import torch

    from ashare_model.alphagpt import build_action_mask
    from ashare_model.vocab import FormulaVocab

    vocab = FormulaVocab(
        feature_names=("NOT_IN_FEATURE_METADATA",),
        operator_names=("NEG",),
    )
    with pytest.raises(ValueError, match="has no authored semantic type"):
        build_action_mask(
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.bool),
            0,
            4,
            vocab,
            stack_types=torch.zeros(1, 4, dtype=torch.long),
        )


def test_random_sampling_supports_registered_toy_vocabulary():
    """Contract §6.3: token-state advancement is vocabulary-local; a toy
    vocabulary made from real metadata must not be interpreted with the
    production vocabulary's offsets."""
    from ashare_model.train_windows import sample_random_formulas
    from ashare_model.vocab import FormulaVocab

    vocab = FormulaVocab(
        feature_names=("RET_1", "RET_5"),
        operator_names=("NEG", "ADD", "GATE"),
    )
    samples = sample_random_formulas(
        seed=19, vocab=vocab, max_len=8, n=100
    )

    assert len(samples) == 100
    for sequence in samples:
        assert formula_types_legal(sequence, vocab), sequence


def test_gp_typed_trees_are_type_legal_and_length_bounded():
    """§6.2: typed-GP trees satisfy the signature lattice and the
    ``len(tree_to_tokens(tree)) == len(tree)`` invariant."""
    from ashare_model.gp_search import _typed_expr, build_pset, tree_to_tokens

    pset = build_pset(FORMULA_VOCAB)
    py_random.seed(5)
    for _ in range(200):
        tree = _typed_expr(pset, 1, 2)
        tokens = tree_to_tokens(tree, FORMULA_VOCAB)
        assert len(tokens) == len(tree)
        assert formula_types_legal(tokens + [_eos()]), tokens


def test_feature_id_restriction_still_applies():
    from ashare_model.gp_search import _typed_expr, build_pset, tree_to_tokens
    from ashare_model.vocab import FEATURE_NAMES

    allowed = [FEATURE_NAMES.index("ROE"), FEATURE_NAMES.index("PE_TTM")]
    ids = [FORMULA_VOCAB.feature_offset + i for i in allowed]
    pset = build_pset(FORMULA_VOCAB, feature_ids=ids)
    py_random.seed(9)
    for _ in range(50):
        tree = _typed_expr(pset, 1, 2)
        for token in tree_to_tokens(tree, FORMULA_VOCAB):
            if token < FORMULA_VOCAB.operator_offset:
                assert token in ids


def test_version_pins():
    """§4: the coordinated generation bumps (requirement-change path,
    arbitration source: docs/p7_semantic_types_contract.md; the
    SEARCH_CONTRACT pin moves to 3 per the approved P10 contract
    §9 — docs/p10_searcher_fairness_contract.md — GP node-cap
    alignment changes the matched effective space, AGENTS.md §10.1
    case 2)."""
    from ashare_model.evaluation import PROTOCOL_VERSION
    from ashare_model.search_contract import SEARCH_CONTRACT_VERSION
    from ashare_model.vocab import GRAMMAR_VERSION

    assert GRAMMAR_VERSION == 5
    assert SEARCH_CONTRACT_VERSION == 3
    assert PROTOCOL_VERSION == "25"


def test_legacy_formula_resolution_unaffected():
    """§5: a type-violating pre-P7E formula still resolves by name (and
    would execute); only *sampling* is constrained."""
    import torch

    from ashare_model.vocab import resolve_formula_tokens
    from ashare_model.vm import StackVM

    # LIMIT_UP_EVENT LIMIT_DOWN_EVENT DIV is type-illegal now, but it is a
    # structurally valid postfix formula a legacy artifact could carry.
    names = ["LIMIT_UP_EVENT", "LIMIT_DOWN_EVENT", "DIV"]
    tokens = [
        FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(n)
        for n in names[:2]
    ]
    tokens.append(
        FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index(names[2])
    )
    payload = {
        "formula": tokens,
        "feature_names": list(FORMULA_VOCAB.feature_names),
        "operator_names": list(FORMULA_VOCAB.operator_names),
        "grammar_version": 2,
    }
    resolved = resolve_formula_tokens(payload, FORMULA_VOCAB)
    assert resolved
    assert not formula_types_legal(resolved + [_eos()])
    factor = torch.randn(FORMULA_VOCAB.feature_count, 2, 4)
    vm = StackVM(
        FORMULA_VOCAB,
        universe_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    assert vm.execute(resolved, factor) is not None
