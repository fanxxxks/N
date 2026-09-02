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
    # t46 accepted Plan B (captain final ruling): the family-⑤ tail rides
    # AFTER EOS -- the layout invariant is "features 1-73 / operators
    # 74-112 / EOS 113 / tail 114-117"; every grammar-5-era token id
    # (0-113) keeps its exact meaning with zero remap cost.
    assert FORMULA_VOCAB.feature_count == 73
    assert FORMULA_VOCAB.feature_count == len(FORMULA_VOCAB.feature_names)
    assert len(FEATURE_NAMES) == 77
    assert FORMULA_VOCAB.operator_offset == 74
    assert FORMULA_VOCAB.token_names[0] == "PAD"
    assert FORMULA_VOCAB.token_names[1] == FEATURE_NAMES[0]
    assert FORMULA_VOCAB.size == len(FORMULA_VOCAB.token_names) == 118
    assert FORMULA_VOCAB.has_eos
    assert FORMULA_VOCAB.eos_token_id == 113
    assert FORMULA_VOCAB.token_names[113] == "EOS"
    assert FORMULA_VOCAB.token_names[114:] == FORMULA_VOCAB.tail_feature_names
    assert len(FORMULA_VOCAB.tail_feature_names) == 4
    assert tuple(FORMULA_VOCAB.tail_feature_names) == tuple(FEATURE_NAMES[73:])
    # v6 vocabulary: the original 18 operators plus 4 cross-sectional
    # operators and 17 enumerated windows (5/10/60 + DELTA10/20).
    assert len(FORMULA_VOCAB.operator_names) == 39
    assert FORMULA_VOCAB.operator_names[:18] == LEGACY_OPERATOR_NAMES + (
        "CORR20",
        "DOWNVOL20",
    )


def test_grammar5_seed7_decodes_bit_stable_under_live_vocab():
    """P13 amendment Plan B (accepted): grammar-5-era tokens (the P10
    seed-7 formula) decode IDENTICALLY against the live grammar-6
    vocabulary -- ids 0-113 are bit-stable, so legacy decodability costs
    zero remapping (p12 migration promise directly satisfied).  The
    expected names are the released P10 formula's decode (provenance:
    docs/p10_searcher_comparison_20260901/summary.json)."""
    seed7 = [6, 99, 45, 68, 72, 73, 81, 83, 88, 105, 77, 113]
    names = tokens_to_names(seed7, FORMULA_VOCAB)
    assert names == [
        "TURNOVER",
        "STD5",
        "ATR_14",
        "LIMIT_UP_CNT_5",
        "CROWD_AMOUNT_60",
        "MARGIN_CROWD_60",
        "GATE",
        "DECAY",
        "STD20",
        "CORR5",
        "DIV",
        "EOS",
    ]
    # By-name resolve round-trips to the identical ids (zero remap).
    assert (
        resolve_formula_tokens(
            {"formula": seed7, "grammar_version": 5}, FORMULA_VOCAB
        )
        == seed7
    )


def test_action_mask_rejects_tail_feature_ids_fail_closed():
    """t46 seam documentation (captain: the two research_domain REDs and
    this pin stay RED/fail-closed until t52 wires the tail): the sampling
    mask classifies features by the contiguous block
    [feature_offset, operator_offset), so the family-⑤ tail ids
    (114-117) are unreachable at the sampling layer -- build_action_mask
    must reject them loudly instead of silently misindexing.  t52
    (impl-b, alphagpt.py ownership) turns this and the research_domain
    tail-sampling tests green."""
    import pytest

    from ashare_model.alphagpt import build_action_mask

    import torch

    tail_ids = [
        FORMULA_VOCAB.eos_token_id + 1 + index
        for index in range(len(FORMULA_VOCAB.tail_feature_names))
    ]
    assert tail_ids == [114, 115, 116, 117]
    stack_sizes = torch.zeros(4, dtype=torch.long)
    done = torch.zeros(4, dtype=torch.bool)
    stack_types = torch.zeros(4, 6, dtype=torch.long)
    with pytest.raises(ValueError):
        build_action_mask(
            stack_sizes,
            done,
            0,
            6,
            FORMULA_VOCAB,
            feature_ids=tail_ids,
            stack_types=stack_types,
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
    # t46 accepted Plan B: the four names ride AFTER EOS (tail_feature_names
    # is part of the feature_version payload); the t14-era hash changes to
    # the tail-layout hash -- re-pinned, no grammar bump.
    assert FORMULA_VOCAB.feature_version == "afbf06f3ba54"
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
    """t46 captain pre-ruling: only known generations (5 = frozen layout,
    6 = live) enter remap decoding; anything else fails closed."""
    payload = _sample_payload()
    payload["grammar_version"] = 2
    with pytest.raises(ValueError, match="unsupported grammar generation 2"):
        resolve_formula_tokens(payload)
    payload["grammar_version"] = 7
    with pytest.raises(ValueError, match="unsupported grammar generation 7"):
        resolve_formula_tokens(payload)


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
    # Mirrors the real artifact writer (train.py:933): ``feature_names`` is
    # the vocabulary's contiguous pre-operator feature block; the family-⑤
    # tail rides after EOS and is recorded via ``tail_feature_names`` once
    # tail formulas enter artifacts (t52 wiring).
    return {
        "formula": [1, 2, FORMULA_VOCAB.operator_offset],
        "feature_names": list(FORMULA_VOCAB.feature_names),
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
    # (The recorded list is the contiguous head block; the family-⑤ tail
    # rides after EOS and is exercised via the tail payload in the layout
    # tests.)
    head = list(FORMULA_VOCAB.feature_names)
    reordered = [head[-1]] + head[:-1]
    payload = _sample_payload()
    payload["feature_names"] = reordered
    payload["formula"] = [1, 2, FORMULA_VOCAB.operator_offset]
    resolved = resolve_formula_tokens(payload)
    assert resolved[0] == 1 + head.index(head[-1])
    assert resolved[1] == 1 + head.index(head[0])


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


def test_resolve_formula_tokens_grammar5_payload_roundtrip_is_identity():
    """t46 Plan B zero-remap-cost evidence: a grammar-5 payload (recorded
    73-feature layout, operator ids 74-112, EOS 113) resolves to the
    IDENTICAL ids under the live grammar-6 vocabulary -- feature ids are
    bit-stable and the operator/EOS ids never moved.  An unknown
    historical generation (2) is fail-closed rejected instead of silently
    misdecoded."""
    feature = 1 + FORMULA_VOCAB.feature_names.index("ROE")
    operator = 74 + 3  # grammar-5 operator id (operator_names[3] = DIV)
    payload = {
        "formula": [feature, operator, 113],
        "feature_names": list(FORMULA_VOCAB.feature_names),
        "operator_names": list(FORMULA_VOCAB.operator_names),
        "grammar_version": 5,
    }
    resolved = resolve_formula_tokens(payload, FORMULA_VOCAB)
    assert resolved == [feature, operator, 113]  # identity: zero remap
    # The names survive: same AST under the live vocabulary.
    assert tokens_to_names(resolved, FORMULA_VOCAB) == tokens_to_names(
        [feature, operator, 113], FORMULA_VOCAB
    )
    # An unknown historical generation fails closed.
    v2_payload = dict(payload, grammar_version=2)
    with pytest.raises(ValueError, match="unsupported grammar generation 2"):
        resolve_formula_tokens(v2_payload, FORMULA_VOCAB)


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
