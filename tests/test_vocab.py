from __future__ import annotations

import pytest

from ashare_model.vocab import (
    FEATURE_ALIASES,
    FEATURE_NAMES,
    FORMULA_VOCAB,
    LEGACY_FEATURE_NAMES,
    LEGACY_OPERATOR_NAMES,
    FormulaVocab,
    resolve_formula_tokens,
    tokens_to_names,
)


def test_formula_vocab_consistency():
    assert FORMULA_VOCAB.feature_count == len(FEATURE_NAMES)
    assert FORMULA_VOCAB.operator_offset == 1 + len(FEATURE_NAMES)
    assert FORMULA_VOCAB.token_names[0] == "PAD"
    assert FORMULA_VOCAB.token_names[1] == FEATURE_NAMES[0]
    assert FORMULA_VOCAB.size == len(FORMULA_VOCAB.token_names)
    assert len(FORMULA_VOCAB.operator_names) == 18


def test_feature_version_pinned():
    # Changing the feature or operator name lists must change this version;
    # the test pins the released vocabulary generation so accidental edits
    # to FEATURE_NAMES are caught in review, not silently in production.
    assert FORMULA_VOCAB.feature_version == "cf5116134c84"
    assert len(FORMULA_VOCAB.feature_version) == 12


def test_legacy_lists_pinned_to_first_generation():
    # The first generation had 34 features and 16 operators; the pins must
    # survive later vocabulary growth because legacy formulas remap against
    # them.  The v3 generation retired one duplicate (RET_20 -> alias), so
    # every legacy name must resolve either directly or through an alias.
    assert len(LEGACY_FEATURE_NAMES) == 34
    assert len(LEGACY_OPERATOR_NAMES) == 16
    assert tuple(FEATURE_NAMES) != tuple(LEGACY_FEATURE_NAMES)
    assert LEGACY_OPERATOR_NAMES == FORMULA_VOCAB.operator_names[:16]
    missing = [
        name
        for name in LEGACY_FEATURE_NAMES
        if name not in FEATURE_NAMES and name not in FEATURE_ALIASES
    ]
    assert missing == []


def test_feature_aliases_are_consistent():
    # Aliases retire duplicate features: the alias itself must not be
    # offered to the policy, and its target must be a live feature.
    assert set(FEATURE_ALIASES).isdisjoint(FEATURE_NAMES)
    assert set(FEATURE_ALIASES.values()) <= set(FEATURE_NAMES)
    assert "RET_20" in FEATURE_ALIASES
    assert "MOMENTUM_20" in FEATURE_NAMES


def test_resolve_formula_tokens_aliases_retired_features():
    # RET_20 was identical to MOMENTUM_20; a formula that references it
    # (legacy token layout or recorded metadata) resolves to MOMENTUM_20's
    # current token id with unchanged semantics.
    legacy_token = 1 + LEGACY_FEATURE_NAMES.index("RET_20")
    resolved = resolve_formula_tokens({"formula": [legacy_token]})
    assert resolved == [1 + FEATURE_NAMES.index("MOMENTUM_20")]
    payload = {
        "formula": [1],
        "feature_names": ["RET_20", "MOMENTUM_20"],
    }
    assert resolve_formula_tokens(payload) == [1 + FEATURE_NAMES.index("MOMENTUM_20")]


def test_resolve_formula_tokens_alias_target_must_exist():
    # An alias whose target is absent from the active vocabulary is not a
    # silent substitution: it fails like any other unknown name.
    src = FormulaVocab(
        feature_names=tuple(FEATURE_NAMES),
        operator_names=FORMULA_VOCAB.operator_names,
    )
    assert "MOMENTUM_20" in src.feature_names
    stripped = FormulaVocab(
        feature_names=tuple(n for n in src.feature_names if n != "MOMENTUM_20"),
        operator_names=FORMULA_VOCAB.operator_names,
    )
    payload = {"formula": [1], "feature_names": ["RET_20"]}
    with pytest.raises(ValueError, match="RET_20"):
        resolve_formula_tokens(payload, vocab=stripped)


def test_tokens_to_names_roundtrip():
    tokens = [1, 2, FORMULA_VOCAB.operator_offset, FORMULA_VOCAB.pad_token_id]
    names = tokens_to_names(tokens)
    assert names == [
        FEATURE_NAMES[0],
        FEATURE_NAMES[1],
        FORMULA_VOCAB.operator_names[0],
        "PAD",
    ]


def test_tokens_to_names_rejects_out_of_range():
    assert tokens_to_names([1, FORMULA_VOCAB.size]) is None
    assert tokens_to_names([1, -1]) is None


def _sample_payload():
    return {
        "formula": [1, 2, FORMULA_VOCAB.operator_offset],
        "feature_names": list(FEATURE_NAMES),
        "operator_names": list(FORMULA_VOCAB.operator_names),
        "feature_version": FORMULA_VOCAB.feature_version,
    }


def test_resolve_formula_tokens_identity_with_metadata():
    tokens = resolve_formula_tokens(_sample_payload())
    assert tokens == [1, 2, FORMULA_VOCAB.operator_offset]


def test_resolve_formula_tokens_remaps_by_name():
    # A formula trained on a reordered feature list must resolve to the ids
    # of the same *names* in the current vocabulary, never the same ids.
    reordered = [FEATURE_NAMES[-1]] + list(FEATURE_NAMES[:-1])
    payload = _sample_payload()
    payload["feature_names"] = reordered
    payload["formula"] = [1, 2, FORMULA_VOCAB.operator_offset]
    resolved = resolve_formula_tokens(payload)
    assert resolved[0] == 1 + FEATURE_NAMES.index(FEATURE_NAMES[-1])
    assert resolved[1] == 1 + FEATURE_NAMES.index(FEATURE_NAMES[0])


def test_resolve_formula_tokens_legacy_payload_without_metadata():
    # Legacy artifacts carry no feature list: they resolve against the
    # pinned first-generation vocabulary by name, so the old feature ids
    # keep their meaning and the old operator ids shift to the new offset.
    legacy_op = 1 + len(LEGACY_FEATURE_NAMES)  # first operator id pre-growth
    payload = {"formula": [1, 2, legacy_op]}
    tokens = resolve_formula_tokens(payload)
    assert tokens == [1, 2, FORMULA_VOCAB.operator_offset]


def test_resolve_formula_tokens_remaps_legacy_under_future_vocab():
    # Even against a hypothetical further-grown vocabulary the legacy
    # formula remaps by name: feature ids stay, operator ids shift.
    future = FormulaVocab(
        feature_names=tuple(FEATURE_NAMES) + ("NEW_FACTOR",),
        operator_names=tuple(FORMULA_VOCAB.operator_names) + ("NEWOP20",),
    )
    legacy_op = 1 + len(LEGACY_FEATURE_NAMES)
    tokens = resolve_formula_tokens({"formula": [1, legacy_op]}, vocab=future)
    assert tokens == [1, future.operator_offset]


def test_resolve_formula_tokens_accepts_bare_token_list():
    assert resolve_formula_tokens([1, 2]) == [1, 2]


def test_resolve_formula_tokens_rejects_unknown_feature_name():
    payload = _sample_payload()
    payload["feature_names"] = list(FEATURE_NAMES) + ["BRAND_NEW_FACTOR"]
    payload["formula"] = [1 + len(FEATURE_NAMES)]  # points at the extra name
    with pytest.raises(ValueError, match="BRAND_NEW_FACTOR"):
        resolve_formula_tokens(payload)


def test_resolve_formula_tokens_rejects_missing_feature_in_current_vocab():
    src = FormulaVocab(
        feature_names=tuple(FEATURE_NAMES) + ("EXTRA",),
        operator_names=FORMULA_VOCAB.operator_names,
    )
    payload = {
        "formula": [1 + len(FEATURE_NAMES)],
        "feature_names": list(src.feature_names),
    }
    with pytest.raises(ValueError, match="EXTRA"):
        resolve_formula_tokens(payload)


def test_resolve_formula_tokens_rejects_out_of_range_tokens():
    payload = _sample_payload()
    payload["formula"] = [FORMULA_VOCAB.size]
    with pytest.raises(ValueError, match="out of range"):
        resolve_formula_tokens(payload)


def test_resolve_formula_tokens_rejects_missing_formula_field():
    with pytest.raises(ValueError, match="no 'formula' field"):
        resolve_formula_tokens({"feature_names": list(FEATURE_NAMES)})
