from __future__ import annotations

import pytest

from ashare_model.ir import Feature, decode
from ashare_model.vocab import (
    FEATURE_ALIASES,
    FEATURE_NAMES,
    FORMULA_VOCAB,
    GRAMMAR_VERSION,
    GRAMMAR_V5_TOKEN_NAMES,
    GRAMMAR_V5_VOCAB,
    LEGACY_FEATURE_NAMES,
    LEGACY_OPERATOR_NAMES,
    FormulaVocab,
    resolve_formula_tokens,
    tokens_to_names,
)


def test_formula_vocab_consistency():
    # t46 Plan A (captain ruling): the grammar-6 feature append keeps the
    # index-contiguous derivation -- features 1-77, operators 78-116,
    # EOS 117 -- and legacy grammar-5 formulas are kept decodable through
    # the frozen grammar-5 layout + by-name remap (see
    # test_grammar5_seed7_decodes_via_frozen_layout_and_remap).
    assert FORMULA_VOCAB.feature_count == len(FEATURE_NAMES) == 77
    assert FORMULA_VOCAB.operator_offset == 78
    assert FORMULA_VOCAB.token_names[0] == "PAD"
    assert FORMULA_VOCAB.token_names[1] == FEATURE_NAMES[0]
    assert FORMULA_VOCAB.size == len(FORMULA_VOCAB.token_names) == 118
    assert FORMULA_VOCAB.has_eos
    assert FORMULA_VOCAB.eos_token_id == FORMULA_VOCAB.size - 1 == 117
    assert FORMULA_VOCAB.token_names[-1] == "EOS"
    # Probe (a): the four family-⑤ features sit at token ids 74-77.
    family5 = ("CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL")
    for offset, name in enumerate(family5):
        assert FEATURE_NAMES[73 + offset] == name
        assert FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name) == 74 + offset
    # v6 vocabulary: the original 18 operators plus 4 cross-sectional
    # operators and 17 enumerated windows (5/10/60 + DELTA10/20).
    assert len(FORMULA_VOCAB.operator_names) == 39
    assert FORMULA_VOCAB.operator_names[:18] == LEGACY_OPERATOR_NAMES + (
        "CORR20",
        "DOWNVOL20",
    )


def test_grammar5_seed7_decodes_via_frozen_layout_and_remap():
    """P13 amendment Plan A (t46): grammar-6 appended the family-⑤ features
    to the feature block, so the operator/EOS ids shifted (74->78,
    113->117).  grammar-5-era artifacts (the P10 campaign, seed-7) stay
    decodable through the FROZEN grammar-5 layout + the by-name remap (P9
    precedent); the frozen name lists are data, never a second semantic
    path."""
    from ashare_model.ir import decode as ir_decode
    from ashare_model.vocab import GRAMMAR_V5_TOKEN_NAMES, GRAMMAR_V5_VOCAB

    seed7 = [6, 99, 45, 68, 72, 73, 81, 83, 88, 105, 77, 113]

    # The frozen layout reproduces the released grammar-5 shape exactly:
    # 73 features + 39 operators + EOS = 114 tokens, EOS at 113.
    assert len(GRAMMAR_V5_TOKEN_NAMES) == 114
    assert GRAMMAR_V5_VOCAB.eos_token_id == 113
    assert GRAMMAR_V5_VOCAB.feature_count == 73
    assert GRAMMAR_V5_VOCAB.feature_names == tuple(FEATURE_NAMES[:73])

    # Probe (d): seed-7 decodes against the frozen layout (never None), and
    # the decoded names are exactly the pre-bump decode.
    v5_names = tokens_to_names(seed7, GRAMMAR_V5_VOCAB)
    assert v5_names is not None
    assert "EOS" in v5_names

    # Under the live grammar-6 layout the same integers decode DIFFERENTLY
    # (the shift is real) -- which is why the frozen dispatch exists.
    live_names = tokens_to_names(seed7, FORMULA_VOCAB)
    assert live_names != v5_names

    # Probe (e): the by-name remap upgrades the grammar-5 formula into the
    # live layout; the decoded AST is semantically identical (Gap 4: full
    # AST equality, not just length).
    resolved = resolve_formula_tokens(
        {"formula": seed7, "grammar_version": 5}, FORMULA_VOCAB
    )
    assert resolved != seed7  # operator/EOS ids moved (74->78, 113->117)
    assert ir_decode(seed7, vocab=GRAMMAR_V5_VOCAB) == ir_decode(
        resolved, vocab=FORMULA_VOCAB
    )


def test_action_mask_reaches_family5_features():
    """Probes (b)+(c) (t46 Plan A): the live action mask treats the appended
    family-⑤ feature ids as first-class features -- a mask restricted to
    them is legal and a random walk under it emits those tokens."""
    import random as py_random

    import torch

    from ashare_model.alphagpt import build_action_mask
    from ashare_model.gp_search import (
        _typed_expr,
        build_pset,
        tree_to_tokens,
    )

    from ashare_model.train_windows import sample_random_formulas

    family_ids = [
        FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name)
        for name in ("CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL")
    ]
    assert family_ids == [74, 75, 76, 77]  # probe (a): contiguous append

    pset = build_pset(FORMULA_VOCAB, feature_ids=family_ids)
    py_random.seed(11)
    for _ in range(50):
        tree = _typed_expr(pset, 1, 2)
        tokens = tree_to_tokens(tree, FORMULA_VOCAB)
        for token in tokens:
            if token < FORMULA_VOCAB.operator_offset:
                assert token in family_ids  # probe (c): only family-⑤ features

    # Probe (b): build_action_mask accepts the appended ids as feature_ids.
    stack_sizes = torch.zeros(4, dtype=torch.long)
    done = torch.zeros(4, dtype=torch.bool)
    stack_types = torch.zeros(4, 6, dtype=torch.long)
    mask = build_action_mask(
        stack_sizes,
        done,
        0,
        6,
        FORMULA_VOCAB,
        feature_ids=family_ids,
        stack_types=stack_types,
    )
    allowed = (mask == 0.0)[0]  # first (only) batch row over the token axis
    for token in family_ids:
        assert bool(allowed[token]), token

    # The random baseline sampler shares the same legality rules and never
    # emits an out-of-scope feature token (PAD-after-EOS and operators/EOS
    # are legal non-feature tokens).  Probe (c): the family-⑤ features are
    # genuinely emitted.
    sequences = sample_random_formulas(
        42, FORMULA_VOCAB, 12, 300, feature_ids=family_ids
    )
    assert sequences
    operator_offset = FORMULA_VOCAB.operator_offset
    emitted_features = {
        token
        for seq in sequences
        for token in seq
        if 1 <= token < operator_offset
    }
    assert emitted_features, "the walk never emitted a family-⑤ feature"
    assert emitted_features <= set(family_ids)


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
    # t46 Plan A: the appended names keep the contiguous feature block
    # (ids 74-77) and legacy grammar-5 decoding goes through the frozen
    # grammar-5 layout -- the t14 feature_version hash is restored.
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
    # RET_20 was identical to MOMENTUM_20; a formula that references it via
    # recorded metadata resolves to MOMENTUM_20's current token id with
    # unchanged semantics.
    expected = [1 + FEATURE_NAMES.index("MOMENTUM_20")]
    payload = {
        "formula": [1],
        "feature_names": ["RET_20", "MOMENTUM_20"],
        "grammar_version": 6,
    }
    assert resolve_formula_tokens(payload) == expected


def test_resolve_formula_tokens_fails_closed_without_grammar_version():
    """t46 captain pre-ruling (AGENTS §4.3/§4.6): a payload without a
    declared grammar generation is fail-closed rejected from remap
    decoding -- silently decoding it against the latest table was the
    t28 incident mechanism.  Display layers mark such artifacts
    legacy/unavailable; pre-P7 artifacts need an explicit auditable
    migration."""
    legacy_token = 1 + LEGACY_FEATURE_NAMES.index("RET_20")
    with pytest.raises(ValueError, match="grammar_version"):
        resolve_formula_tokens({"formula": [legacy_token]})
    with pytest.raises(ValueError, match="grammar_version"):
        resolve_formula_tokens({"formula": [1]})


def test_resolve_formula_tokens_fails_closed_on_unknown_grammar_generation():
    """t46 captain pre-ruling + t57 F2: only known generations (2-6) enter
    remap decoding; anything else fails closed."""
    payload = _sample_payload()
    payload["grammar_version"] = 1
    with pytest.raises(ValueError, match="unsupported grammar generation 1"):
        resolve_formula_tokens(payload)
    payload["grammar_version"] = 7
    with pytest.raises(ValueError, match="unsupported grammar generation 7"):
        resolve_formula_tokens(payload)


def test_grammar2_3_payloads_decode_via_frozen_v3_layout():
    """t57 F2 (contract-a ruling): grammar-2/3 payloads decode against the
    frozen grammar-3 layout (62 features, EOS 102) -- under the live
    layout token 102 is TS_RANK5, so the per-generation dispatch is the
    corruption guard; the by-name remap upgrades the formula into the live
    vocabulary (EOS 102 -> 117)."""
    v3_payload = {
        "formula": [1, 102],  # RET_1 + EOS(102 under grammar 3)
        "grammar_version": 3,
    }
    resolved = resolve_formula_tokens(v3_payload, FORMULA_VOCAB)
    assert resolved == [1, FORMULA_VOCAB.eos_token_id]
    # A grammar-2 payload dispatches to the same frozen layout.
    v2_payload = {
        "formula": [1, 102],
        "grammar_version": 2,
    }
    assert resolve_formula_tokens(v2_payload, FORMULA_VOCAB) == [
        1,
        FORMULA_VOCAB.eos_token_id,
    ]


def test_legacy_bare_factor_requires_declared_generation():
    # t46 captain pre-ruling: the old bare-factor migration path ([1] with
    # no metadata) is superseded by the fail-closed grammar gate -- the
    # bare token alone no longer declares which layout it was sampled
    # against.  With the generation declared, the migration still works
    # and the AST remains the single source of truth.
    with pytest.raises(ValueError, match="grammar_version"):
        resolve_formula_tokens({"formula": [1]})
    canonical = resolve_formula_tokens({"formula": [1], "grammar_version": 6})
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
    payload = {"formula": [1], "feature_names": ["RET_20"], "grammar_version": 6}
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
        # t46 captain pre-ruling: every formal artifact declares its grammar
        # generation; payloads without it are fail-closed rejected from
        # remap decoding (AGENTS §4.3/§4.6).
        "grammar_version": GRAMMAR_VERSION,
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
    # t46 captain pre-ruling: a bare payload with no declared grammar
    # generation is fail-closed rejected from remap decoding (the
    # pre-t46 behavior silently decoded it against the LEGACY v1 layout,
    # which misreads every post-v1 token id -- the t28 incident class).
    legacy_op = 1 + len(LEGACY_FEATURE_NAMES)  # first operator id pre-growth
    payload = {"formula": [1, 2, legacy_op]}
    with pytest.raises(ValueError, match="grammar_version"):
        resolve_formula_tokens(payload)


def test_resolve_formula_tokens_remaps_legacy_under_future_vocab():
    # Even against a hypothetical further-grown vocabulary a formula with
    # a DECLARED generation remaps by name: feature ids stay, operator ids
    # shift (t46 captain pre-ruling: the declared generation must be a
    # known one -- here the live grammar 6).
    future = FormulaVocab(
        feature_names=tuple(FEATURE_NAMES) + ("NEW_FACTOR",),
        operator_names=tuple(FORMULA_VOCAB.operator_names) + ("NEWOP20",),
    )
    tokens = [1, 2, FORMULA_VOCAB.operator_offset]
    payload = {
        "formula": tokens,
        "grammar_version": 6,
    }
    resolved = resolve_formula_tokens(payload, vocab=future)
    assert resolved == [1, 2, future.operator_offset]


def test_resolve_formula_tokens_accepts_bare_token_list():
    # A bare token list is accepted only when the generation is declared
    # (t46 captain pre-ruling): with the live generation the ids are
    # identity-mapped.
    with pytest.raises(ValueError, match="grammar_version"):
        resolve_formula_tokens([1, 2])
    assert resolve_formula_tokens(
        {"formula": [1, 2], "grammar_version": 6}
    ) == [1, 2]


def test_resolve_formula_tokens_grammar5_payload_remaps_operators_and_eos():
    """t46 Plan A Gap-1 evidence: the by-name remap covers ALL token
    dimensions across the grammar-5 -> grammar-6 shift -- an operator id
    (74..112 under grammar 5) resolves to its shifted operator id
    (78..116) and EOS 113 resolves to 117, while feature ids 1-73 stay
    bit-stable.  t57 F2: grammar-2 payloads now dispatch to the frozen
    grammar-3 layout; unknown generations still fail closed."""
    feature = 1 + FEATURE_NAMES.index("ROE")  # grammar-5 feature id (bit-stable)
    operator = 1 + 73 + 3  # grammar-5 operator id 77 (feature block grew)
    payload = {
        "formula": [feature, operator, 113],
        "feature_names": list(GRAMMAR_V5_VOCAB.feature_names),
        "operator_names": list(FORMULA_VOCAB.operator_names),
        "grammar_version": 5,
    }
    resolved = resolve_formula_tokens(payload, FORMULA_VOCAB)
    assert resolved == [feature, 78 + 3, 117]
    # The names survive: same AST under both layouts.
    assert tokens_to_names(resolved, FORMULA_VOCAB) == tokens_to_names(
        [feature, operator, 113], GRAMMAR_V5_VOCAB
    )
    # An unknown generation fails closed (t57 F2: known = 2-6).
    v1_payload = dict(payload, grammar_version=1)
    with pytest.raises(ValueError, match="unsupported grammar generation 1"):
        resolve_formula_tokens(v1_payload, FORMULA_VOCAB)


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
        "grammar_version": 6,
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
