from __future__ import annotations

import pytest

from ashare_model.ir import Feature, decode
from ashare_model.vocab import (
    FEATURE_ALIASES,
    FEATURE_NAMES,
    FORMULA_VOCAB,
    GRAMMAR_VERSION,
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
    # The independent EOS token sits at the end of the token space so the
    # pre-EOS feature/operator ids never shift.
    assert FORMULA_VOCAB.has_eos
    assert FORMULA_VOCAB.eos_token_id == FORMULA_VOCAB.size - 1
    assert FORMULA_VOCAB.token_names[-1] == "EOS"
    # v6 vocabulary: the original 18 operators plus 4 cross-sectional
    # operators and 17 enumerated windows (5/10/60 + DELTA10/20).
    assert len(FORMULA_VOCAB.operator_names) == 39
    assert FORMULA_VOCAB.operator_names[:18] == LEGACY_OPERATOR_NAMES + (
        "CORR20",
        "DOWNVOL20",
    )


def test_feature_version_pinned():
    # Changing the feature/operator name lists or the grammar generation
    # must change this version; the test pins the released vocabulary
    # generation so accidental edits are caught in review, not silently
    # in production.  v4 (P9, docs/p9_factor_family_contract.md §9 --
    # requirement-change path, APPROVED 2026-09-01): the four orthogonal
    # families append 11 feature names and the deprecated features leave
    # the sampling space; saved formulas still resolve by name.
    # v6 (P13 §5.3/§6.1, docs/p13_fundamental_fields_contract.md
    # APPROVED): family ⑤ appends 4 slow-fundamental names (73 -> 77);
    # whitelist §10.1 case 2.
    assert FORMULA_VOCAB.feature_version == "0e64ad614bfd"
    assert len(FORMULA_VOCAB.feature_version) == 12
    # v6 grammar: the v5 layout plus the P13 family-⑤ feature append.
    assert GRAMMAR_VERSION == 6
    # A grammar bump changes the version even with identical name lists.
    legacy = FormulaVocab(
        feature_names=FORMULA_VOCAB.feature_names,
        operator_names=FORMULA_VOCAB.operator_names,
        has_eos=False,
    )
    assert legacy.feature_version != FORMULA_VOCAB.feature_version


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
    expected = [1 + FEATURE_NAMES.index("MOMENTUM_20")]
    resolved = resolve_formula_tokens({"formula": [legacy_token]})
    assert resolved == expected
    payload = {
        "formula": [1],
        "feature_names": ["RET_20", "MOMENTUM_20"],
    }
    assert resolve_formula_tokens(payload) == expected


def test_legacy_bare_factor_migrates_to_feature_ast():
    # Old bare-factor artifacts ([1] = first feature) migrate by name and
    # decode to Feature(name): the AST is the single source of truth.
    canonical = resolve_formula_tokens({"formula": [1]})
    assert canonical == [1]
    ast = decode(canonical)
    assert ast == Feature(FEATURE_NAMES[0])


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
    tokens = [
        1,
        2,
        FORMULA_VOCAB.operator_offset,
        FORMULA_VOCAB.eos_token_id,
        FORMULA_VOCAB.pad_token_id,
    ]
    names = tokens_to_names(tokens)
    assert names == [
        FEATURE_NAMES[0],
        FEATURE_NAMES[1],
        FORMULA_VOCAB.operator_names[0],
        "EOS",
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


def test_resolve_formula_tokens_v2_payload_keeps_eos_token():
    # Payloads written under the v2 grammar record ``grammar_version`` and
    # carry the EOS token; resolution must remap EOS by name, not misread
    # it as a feature of the legacy layout.
    payload = _sample_payload()
    payload["grammar_version"] = 2
    payload["formula"] = [1, 2, FORMULA_VOCAB.operator_offset, FORMULA_VOCAB.eos_token_id]
    assert resolve_formula_tokens(payload) == payload["formula"]


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


# --- P9 (docs/p9_factor_family_contract.md, APPROVED): v4 grammar, additive
# feature generation, deprecation registry. RED tests written before impl. ---

P9_NEW_FEATURES = (
    "IND_REL_RET_60",
    "IND_REL_RET_120",
    "LIQ_SHOCK_20",
    "VOLUME_SHRINK_5_20",
    "PV_DIV_20",
    "LIMIT_UP_CNT_5",
    "LIMIT_DOWN_STREAK",
    "LIMIT_BREAK_5",
    "CROWD_TURNOVER_60",
    "CROWD_AMOUNT_60",
    "MARGIN_CROWD_60",
)
P9_DEPRECATED_FEATURES = (
    "RET_5",
    "RET_10",
    "RET_120",
    "MOMENTUM_60",
    "VOLUME_RATIO",
    "TURNOVER_MA5",
    "MACD_DEA",
    "VOL_60",
)


def test_p9_grammar_v4_with_additive_feature_generation():
    # v5 = the v4 P9 layout + the §7 adjudicated deprecations.
    # v6 (P13 §5.3, whitelist §10.1 case 2): family ⑤ appends 4 more.
    assert GRAMMAR_VERSION == 6
    for name in P9_NEW_FEATURES:
        assert name in FEATURE_NAMES
    # v4 names are appended, so every pre-v4 token id stays stable; the
    # P13 family-⑤ names append after them, so every pre-v5 token id
    # (including the deprecated v4 names) stays stable too.
    assert len(FEATURE_NAMES) == 77
    assert list(FEATURE_NAMES[62:73]) == list(P9_NEW_FEATURES)
    assert list(FEATURE_NAMES[73:]) == [
        "CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL",
    ]


def test_p9_deprecated_names_stay_in_the_token_space():
    from ashare_model.vocab import DEPRECATED_FEATURE_NAMES

    # NORTHBOUND_CHG (the pre-P9 neutral deprecation) joins the P9 window
    # variants: deprecated means "never sampled", uniformly.
    assert set(P9_DEPRECATED_FEATURES) <= set(DEPRECATED_FEATURE_NAMES)
    assert "NORTHBOUND_CHG" in DEPRECATED_FEATURE_NAMES
    for name in P9_DEPRECATED_FEATURES:
        # Deprecation never removes a name: legacy formulas must keep
        # resolving by name (same policy as the RET_20 alias precedent).
        assert name in FEATURE_NAMES
        assert name in FORMULA_VOCAB.feature_names
