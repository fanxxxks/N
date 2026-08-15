from __future__ import annotations

import pytest

from ashare_model.vocab import (
    FEATURE_NAMES,
    FORMULA_VOCAB,
    LEGACY_FEATURE_NAMES,
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
    assert len(FORMULA_VOCAB.operator_names) == 16


def test_feature_version_pinned():
    # Changing the feature or operator name lists must change this version;
    # the test pins the released vocabulary generation so accidental edits
    # to FEATURE_NAMES are caught in review, not silently in production.
    assert FORMULA_VOCAB.feature_version == "299902b75579"
    assert len(FORMULA_VOCAB.feature_version) == 12


def test_legacy_feature_names_pinned_to_first_generation():
    assert tuple(LEGACY_FEATURE_NAMES) == tuple(FEATURE_NAMES)
    assert len(LEGACY_FEATURE_NAMES) == 34


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
    # pinned first-generation vocabulary while it still matches.
    payload = {"formula": [1, 2, FORMULA_VOCAB.operator_offset]}
    tokens = resolve_formula_tokens(payload)
    assert tokens == [1, 2, FORMULA_VOCAB.operator_offset]


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


def test_resolve_formula_tokens_rejects_legacy_after_vocab_change():
    # Once the current vocabulary diverges from the pinned legacy list,
    # metadata-free formulas must be refused rather than misread.
    changed = FormulaVocab(
        feature_names=tuple(FEATURE_NAMES) + ("NEW_FACTOR",),
        operator_names=FORMULA_VOCAB.operator_names,
    )
    with pytest.raises(ValueError, match="legacy"):
        resolve_formula_tokens({"formula": [1, 2]}, vocab=changed)
