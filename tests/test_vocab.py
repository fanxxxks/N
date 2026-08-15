from __future__ import annotations

from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB


def test_formula_vocab_consistency():
    assert FORMULA_VOCAB.feature_count == len(FEATURE_NAMES)
    assert FORMULA_VOCAB.operator_offset == 1 + len(FEATURE_NAMES)
    assert FORMULA_VOCAB.token_names[0] == "PAD"
    assert FORMULA_VOCAB.token_names[1] == FEATURE_NAMES[0]
    assert FORMULA_VOCAB.size == len(FORMULA_VOCAB.token_names)
    assert len(FORMULA_VOCAB.operator_names) == 16
