"""P28 A04 contract tests: factor metadata, semantic type, FO-01 manifest.

Contract: ``docs/p28_a04_feature_metadata_semantic_type_draft.md``
(APPROVED — approval record: ``reports/alphagpt-ab/00_authority_exception_record.md``
§11 t6 Ruling 2 + §4ter user confirmation; the user's direct A04 execution
order is the session-level authority).  The assertions below come from the
contract's invariants (I1-I12) and RED list (R1-R14) plus the approved data
contracts (p17 §5.2/§6.1/§6.2 units and visibility; p24 §2 F2
pre-registration) — never from the implementation's current output.

Layer convention for the NaN semantics (I6/R4): the raw factor frames keep
``NaN`` for missing source data (never a fake 0 before cross-sectional
standardization); after the engine-wide standardization the value 0 is the
neutral (factors.py — applies uniformly to every feature).  The neutral
placeholders (NEUTRAL_FEATURE_NAMES) have no live source at all.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data.capital_flow import EXTERNAL_FACTOR_NAMES
from ashare_model.feature_metadata import (
    FEATURE_METADATA,
    SemanticType,
    resolve_depends_on,
)
from ashare_model.feature_registry import (
    FEATURE_REGISTRY_VERSION,
    family_of,
    pit_level_of,
)
from ashare_model.data_tier import tier_of_pit_level
from ashare_model.factors import NEUTRAL_FEATURE_NAMES
from ashare_model.vocab import (
    DEPRECATED_FEATURE_NAMES,
    FEATURE_NAMES,
    FORMULA_VOCAB,
    GRAMMAR_VERSION,
)

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "feature_registry.md"

F2_NAMES = ("MARGIN_BALANCE_TO_CIRC_MV", "MARGIN_BUY_PRESSURE_5")
MARGIN_FAMILY = F2_NAMES + ("MARGIN_BALANCE_CHG", "MARGIN_CROWD_60")
SAMPLABLE = [n for n in FEATURE_NAMES if n not in DEPRECATED_FEATURE_NAMES]

# The controlled unit vocabulary (I6): short economic-dimension tokens;
# the per-feature source-scale chains (yuan vs wan_yuan vs qian_yuan) are
# audited in the p28 measurement log's declaration-source table.
_UNITS = frozenset({
    "ratio", "return", "pct", "price_adj", "cap_proxy", "per_amount",
    "log_volume", "count", "sessions", "boolean", "score_0_100", "neutral",
})


def _seq(*names: str) -> list[int]:
    """Token ids from token names (EOS appended explicitly by callers)."""

    table = {
        "EOS": FORMULA_VOCAB.eos_token_id,
        **{
            n: FORMULA_VOCAB.feature_offset + i
            for i, n in enumerate(FORMULA_VOCAB.feature_names)
        },
        **{
            n: FORMULA_VOCAB.operator_offset + i
            for i, n in enumerate(FORMULA_VOCAB.operator_names)
        },
    }
    return [table[n] for n in names]


def _manifest_block() -> dict:
    """Parse the FO-01 JSON manifest out of the generated registry doc."""

    text = DOC_PATH.read_text(encoding="utf-8")
    marker = "## 4. FO-01"
    start = text.find(marker)
    assert start != -1, (
        "docs/feature_registry.md carries no FO-01 manifest section "
        "(p28 I5/R9); regenerate with scripts/generate_registry_docs.py"
    )
    fence_start = text.find("```json", start)
    fence_end = text.find("```", fence_start + 7)
    assert fence_start != -1 and fence_end != -1, "FO-01 manifest JSON fence"
    return json.loads(text[fence_start + 7:fence_end])


# ---------------------------------------------------------------------------
# R1/R3 — the F2 availability rule expresses the JOINT visibility
# (margin T+1 + daily_basic same-session), never the quarterly template.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", F2_NAMES)
def test_r1_f2_availability_rule_expresses_joint_visibility(name):
    rule = FEATURE_METADATA[name].availability_rule
    # The margin leg: the p17 §6.1 T+1 visibility of margin_detail_pit.
    assert "margin_detail_pit" in rule, name
    assert "T+1" in rule, name
    # The denominator leg: the daily_basic trade-date same-session close.
    assert "daily_basic" in rule, name
    # The joint AND-condition must be explicit (I2: never only one half).
    assert re.search(r"[Bb]oth", rule), name
    # The quarterly-TTM template must be gone (F04).
    assert "quarterly report" not in rule, name
    assert "statutory disclosure season" not in rule, name
    assert "TTM" not in rule, name


@pytest.mark.parametrize("name", F2_NAMES)
def test_r3_joint_rule_names_both_arrival_conventions(name):
    rule = FEATURE_METADATA[name].availability_rule
    # The two sources arrive on different clocks (R3): the margin detail
    # keys its visibility on available_date (T+1), the daily_basic
    # denominator on the trade-date session-t close convention.
    assert "available_date" in rule, name
    assert "trade_date" in rule, name


# ---------------------------------------------------------------------------
# R2 — units are declared for every samplable feature and match the source
# field units (p17 §5.2: rzye/rzmre=yuan, circ_mv=wan_yuan; the ratio is
# scale-invariant under the per-date xcs-z).
# ---------------------------------------------------------------------------


def test_r2_units_declared_for_every_samplable_feature():
    for name in SAMPLABLE:
        meta = FEATURE_METADATA[name]
        unit = getattr(meta, "unit", None)
        assert unit, f"{name}: no unit declared (p28 I6/R2)"
        assert unit in _UNITS, f"{name}: unit {unit!r} outside the controlled set"


@pytest.mark.parametrize(
    ("name", "unit"),
    [
        ("MARGIN_BALANCE_TO_CIRC_MV", "ratio"),
        ("MARGIN_BUY_PRESSURE_5", "ratio"),
        # The 20-session change of the balance is a return-scale quantity.
        ("MARGIN_BALANCE_CHG", "return"),
        ("MARGIN_CROWD_60", "ratio"),
    ],
)
def test_r2_margin_family_units_are_declared(name, unit):
    assert FEATURE_METADATA[name].unit == unit, name


def test_r2_f2_unit_chain_is_the_yuan_over_wanyuan_ratio():
    """The F2 ratio consumes rzye/rzmre [yuan] over circ_mv [wan_yuan]
    (p17 §5.2 tables 4/18, L1 no-conversion); the constant 1e4 scale
    factor is absorbed by the per-date cross-sectional z (scale
    invariant), so the declared economic unit is the ratio itself."""

    for name, margin_field in (
        ("MARGIN_BALANCE_TO_CIRC_MV", "rzye"),
        ("MARGIN_BUY_PRESSURE_5", "rzmre"),
    ):
        deps = resolve_depends_on(name, FEATURE_METADATA[name])
        assert f"margin_detail_pit.{margin_field}" in deps, name
        assert "daily_basic_pit.circ_mv" in deps, name


# ---------------------------------------------------------------------------
# R4 — a missing dependency stays NaN (never neutral-0 before the
# standardization) and the NaN semantics are declared.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", F2_NAMES)
def test_r4_f2_nan_semantics_declared(name):
    from ashare_model.feature_metadata import NaNSemantics

    assert FEATURE_METADATA[name].nan_policy is NaNSemantics.MISSING_NAN, name
    assert (
        FEATURE_METADATA["NORTHBOUND_CHG"].nan_policy
        is NaNSemantics.NEUTRAL_PLACEHOLDER
    )


def test_r4_xcs_z_preserves_missing_cells():
    """Behavioral oracle (hand-computed): the single xcs-z implementation
    standardizes only finite cells and restores NaN — missing is never a
    0 that would distort the cross-section (p24 §2 NaN guards)."""

    from ashare_model.data_loader import apply_cross_sectional_z

    frames = {
        "F": pd.DataFrame(
            np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]], dtype=np.float64)
        )
    }
    apply_cross_sectional_z(frames, ("F",))
    out = frames["F"].to_numpy(dtype=np.float64)
    assert np.isnan(out[0, 2]), "a missing cell must stay NaN"
    # Column 0: values [1, 3] -> mean 2, std 1 -> z [-1, 1] (hand-computed;
    # the implementation guards the division with std + 1e-9, so the
    # tolerance sits one order above that epsilon — not a weaker assert).
    assert out[0, 0] == pytest.approx(-1.0, abs=1e-6)
    assert out[1, 0] == pytest.approx(1.0, abs=1e-6)
    # Column 1: values [2, 4] -> mean 3, std 1 -> z [-1, 1].
    assert out[0, 1] == pytest.approx(-1.0, abs=1e-6)
    assert out[1, 1] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# R5/R7 — the corrected semantic type is executable consistently across
# backends (I3 + t6 Ruling 2: CROSS_SECTIONAL_SIGNAL).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", F2_NAMES)
def test_r7_f2_semantic_type_is_cross_sectional_signal(name):
    assert (
        FEATURE_METADATA[name].semantic_type is SemanticType.CROSS_SECTIONAL_SIGNAL
    ), name


def test_r7_cs_neutralize_is_the_only_single_type_op():
    """The contract's §1.4 measured fact: CS_NEUTRALIZE is the only
    operator restricted to a single input family — this is what makes the
    type correction a real (measurable) search-space change."""

    from ashare_model.operator_registry import OPERATOR_REGISTRY

    single = [
        name
        for name, meta in OPERATOR_REGISTRY.items()
        if all(len({t for t in arg}) == 1 for arg in meta.inputs)
    ]
    assert single == ["CS_NEUTRALIZE"], single


def test_r5_signature_checker_matches_typed_pset_everywhere():
    """I8: for every operator and every argument-type tuple, the semantic
    checker and the GP typed primitive set must agree exactly — a
    signature legal for one backend is legal for all of them."""

    from ashare_model.gp_search import _type_class, build_pset
    from ashare_model.ops import OPS_BY_NAME
    from ashare_model.semantic_sampling import (
        fixed_output,
        is_gate,
        same_type_args,
        signature_legal,
    )

    pset = build_pset(FORMULA_VOCAB)
    registered: set[tuple] = set()
    for ret_cls, prims in pset.primitives.items():
        for prim in prims:
            registered.add((prim.name, tuple(prim.args), ret_cls))

    types = list(SemanticType)
    mismatches: list[str] = []
    for op_name in FORMULA_VOCAB.operator_names:
        arity = OPS_BY_NAME[op_name][1]
        # Enumerate the families per position (the boolean family only
        # enters through GATE's condition argument).
        choices = []
        for j in range(arity):
            if is_gate(op_name) and j == 0:
                choices.append(types)
            else:
                choices.append(
                    [t for t in types if t is not SemanticType.BOOLEAN_EVENT_SIGNAL]
                )
        # Bounded enumeration: arity <= 3, <= 6 families per position.
        for combo in itertools.product(*choices):
            args = tuple(combo)
            legal = signature_legal(op_name, args)
            if is_gate(op_name):
                out = args[1]
            elif same_type_args(op_name):
                out = args[0]
            else:
                out = fixed_output(op_name) or args[0]
            in_pset = (
                op_name,
                tuple(_type_class(t) for t in args),
                _type_class(out),
            ) in registered
            if legal != in_pset:
                mismatches.append(
                    f"{op_name}{args}: checker={legal} pset={in_pset}"
                )
    assert not mismatches, mismatches[:10]


def test_r5_f2_legality_flips_are_backend_consistent():
    """The Ruling-2 flips on the shared checker (the mask/state layer the
    RL/Random/TPE action masks and the GP pset both derive from — the
    checker↔pset bi-implication is pinned by the enumeration test):
    CS_NEUTRALIZE becomes legal over F2; same-family composition with the
    cross-sectional family becomes legal; composition with the
    fundamental family becomes illegal (family-preserving ops share one
    type)."""

    from ashare_model.semantic_sampling import formula_types_legal

    # CS_NEUTRALIZE over F2: legal (was illegal under FUNDAMENTAL_LIKE).
    assert formula_types_legal(_seq("MARGIN_BALANCE_TO_CIRC_MV", "CS_NEUTRALIZE", "EOS"))
    assert formula_types_legal(_seq("MARGIN_BUY_PRESSURE_5", "CS_NEUTRALIZE", "EOS"))
    # F2 + IND_REL_RET_5 (both cross-sectional): legal.
    assert formula_types_legal(_seq("MARGIN_BALANCE_TO_CIRC_MV", "IND_REL_RET_5", "ADD", "EOS"))
    # F2 + PE_TTM (fundamental): illegal after the correction (was legal).
    assert not formula_types_legal(_seq("MARGIN_BALANCE_TO_CIRC_MV", "PE_TTM", "ADD", "EOS"))
    assert not formula_types_legal(_seq("MARGIN_BUY_PRESSURE_5", "LEVERAGE_CHANGE", "ADD", "EOS"))
    # F2 stays a legal root on its own.
    assert formula_types_legal(_seq("MARGIN_BALANCE_TO_CIRC_MV", "EOS"))


# ---------------------------------------------------------------------------
# R6 — the cache identities carry the registry generation (future-facing
# invalidation); legacy by-name decode is untouched.
# ---------------------------------------------------------------------------


def test_r6_semantic_cache_key_carries_registry_generation():
    from ashare_model.semantic_cache import SemanticCache

    cache = SemanticCache(
        dataset_id="d", reward_version=15, protocol_version=27, window_id="w"
    )
    key = cache.key_for(_seq("RET_1"))
    assert key is not None
    # The key layout includes the feature-registry generation (p28 §4.2:
    # SEMANTIC_CACHE_VERSION bump + key field), so entries cached under a
    # different registry generation can never be reused.
    assert key.feature_registry_version == str(FEATURE_REGISTRY_VERSION)


def test_r6_legacy_formula_resolution_unaffected():
    """I12/R6: the metadata correction is future-facing — legacy payloads
    (grammar 9, registry 14 era) still decode by id/name unchanged."""

    from ashare_model.vocab import resolve_formula_tokens

    v = FORMULA_VOCAB
    f2_id = v.feature_offset + FEATURE_NAMES.index("MARGIN_BALANCE_TO_CIRC_MV")
    payload = {
        "grammar_version": 9,
        "formula": [f2_id, v.eos_token_id],
    }
    resolved = resolve_formula_tokens(payload)
    assert resolved == [f2_id, v.eos_token_id]


# ---------------------------------------------------------------------------
# R8 — the depends_on convention is uniform (I4): the margin family is
# fully table.field; no feature mixes table-level and field-level forms.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MARGIN_FAMILY)
def test_r8_margin_family_deps_are_table_field(name):
    deps = resolve_depends_on(name, FEATURE_METADATA[name])
    assert deps, name
    for dep in deps:
        assert re.fullmatch(r"[a-z_]+\.[a-z_]+", dep), (name, dep)


def test_r8_no_feature_mixes_dep_forms():
    """Within one feature, a table.field dependency never coexists with a
    bare table name or a bare field name (I4's mixing ban)."""

    for name in SAMPLABLE:
        deps = resolve_depends_on(name, FEATURE_METADATA[name])
        forms = {"dot" if "." in d else "bare" for d in deps}
        assert len(forms) <= 1, (name, deps)


# ---------------------------------------------------------------------------
# R9/R10 — the FO-01 manifest exists, is complete, and is generated.
# ---------------------------------------------------------------------------


def test_r9_fo01_manifest_counts_and_feature_fields():
    manifest = _manifest_block()
    counts = manifest["counts"]
    assert counts == {
        "feature_slots": 83,
        "samplable": 70,
        "deprecated": 13,
        "operators": 39,
    }, counts
    by_name = {f["name"]: f for f in manifest["features"]}
    assert set(by_name) == set(FEATURE_NAMES)
    required = {
        "name", "vocab_index", "family", "domain", "pit_level", "tier",
        "semantic_type", "availability_rule", "unit", "nan_policy",
        "depends_on", "promotion_allowed", "implementation_status",
        "research_eligibility", "evidence_status", "deprecated",
    }
    for name in SAMPLABLE:
        missing = required - set(by_name[name])
        assert not missing, (name, missing)
    # Operators: the full 39 with their contract fields.
    ops = {o["name"]: o for o in manifest["operators"]}
    assert len(ops) == 39
    for op in ops.values():
        assert {"name", "arity", "inputs", "output", "cost"} <= set(op)


def test_r9_manifest_status_fields_are_distinct():
    """I10/R14: implementation / research-eligibility / evidence /
    promotion are four different answers — the manifest must carry them
    separately and the legend must forbid conflating them."""

    manifest = _manifest_block()
    by_name = {f["name"]: f for f in manifest["features"]}
    nb = by_name["NORTHBOUND_CHG"]
    assert nb["implementation_status"] == "neutral_placeholder"
    assert nb["research_eligibility"] == "deprecated"
    assert nb["evidence_status"] != by_name["RET_1"]["evidence_status"]
    assert nb["promotion_allowed"] is False
    f2 = by_name["MARGIN_BALANCE_TO_CIRC_MV"]
    assert f2["research_eligibility"] == "samplable"
    assert f2["implementation_status"] == "computed_live"
    # The legend states the distinction (candidate admission is not
    # single-factor effectiveness is not family-gate passage).
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "候选池准入" in text and "单因子有效" in text


def test_r10_manifest_is_generated_and_drift_guarded():
    """The manifest is a projection of the registries: regenerating the
    doc must reproduce the committed content byte-for-byte (the drift
    guard), and the regenerated manifest must parse."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_registry_docs", ROOT / "scripts" / "generate_registry_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regenerated = module.build_document()
    committed = DOC_PATH.read_text(encoding="utf-8")
    assert committed == regenerated, (
        "docs/feature_registry.md is stale (p28 R10); regenerate and "
        "commit code+doc together"
    )
    fence_start = regenerated.find("```json")
    parsed = json.loads(
        regenerated[fence_start + 7:regenerated.find("```", fence_start + 7)]
    )
    assert parsed["counts"]["samplable"] == 70


# ---------------------------------------------------------------------------
# R11-R14 — the non-goal guards (expected green before AND after; they pin
# what the correction must NOT change).
# ---------------------------------------------------------------------------


def test_r11_vocab_and_grammar_untouched():
    assert len(FEATURE_NAMES) == 83
    assert FEATURE_NAMES[81] == "MARGIN_BALANCE_TO_CIRC_MV"
    assert FEATURE_NAMES[82] == "MARGIN_BUY_PRESSURE_5"
    assert len(DEPRECATED_FEATURE_NAMES) == 13
    assert GRAMMAR_VERSION == 9  # t6 Ruling 2/8: no grammar bump


def test_r12_factor_values_are_invariant(tmp_path):
    """I9: the correction is metadata-only — the capital frames' numeric
    witness (hand-computed oracle on a fresh fixture, distinct from
    test_capital_flow's) must hold unchanged."""

    from ashare_data.capital_flow import build_capital_frames
    from ashare_data.db import AshareDB
    from tests.conftest import _V2_PRICE_DDL, populate_v2_margin_table
    from tests.test_capital_flow import _data_config, _dates

    cfg = _data_config(tmp_path)
    dates = _dates(4)
    codes = ["000001.SZ", "600000.SH"]
    v2_path = cfg.duckdb_path.parent / "ashare_v2.duckdb"
    margin_rows = [
        {"ts_code": "000001.SZ", "trade_date": dates[0],
         "available_date": dates[1], "rzye": 200.0, "rzmre": 40.0},
        {"ts_code": "600000.SH", "trade_date": dates[0],
         "available_date": dates[1], "rzye": 400.0, "rzmre": 80.0},
    ]
    populate_v2_margin_table(v2_path, margin_rows)
    with AshareDB(v2_path) as db:
        for ddl in _V2_PRICE_DDL.values():
            if "daily_basic_pit" in ddl:
                db.execute(ddl)
        db.execute("DELETE FROM daily_basic_pit")
        for code in codes:
            db.execute(
                "INSERT INTO daily_basic_pit (ts_code, trade_date, "
                "turnover_rate, circ_mv) VALUES (?, ?, ?, ?)",
                [code, dates[1], 1.0, 800.0],
            )
    frames = build_capital_frames(cfg, codes, dates)
    # Hand-computed: 200/800 = 0.25 and 400/800 = 0.5 (raw ratios; the
    # metadata correction must never move these numbers).
    assert frames["MARGIN_BALANCE_TO_CIRC_MV"].loc["000001.SZ", dates[1]] == pytest.approx(0.25)
    assert frames["MARGIN_BALANCE_TO_CIRC_MV"].loc["600000.SH", dates[1]] == pytest.approx(0.5)
    # The no-coverage cell stays NaN in the raw frame (R4's frame layer).
    assert np.isnan(frames["MARGIN_BALANCE_TO_CIRC_MV"].loc["000001.SZ", dates[-1]])


def test_r13_d7_source_routing_unchanged():
    assert family_of("MARGIN_BALANCE_TO_CIRC_MV") == "external"
    assert pit_level_of("MARGIN_BUY_PRESSURE_5").value == "pit_capital"
    assert tier_of_pit_level(pit_level_of("MARGIN_BALANCE_TO_CIRC_MV")).value == "B"
    from ashare_model.data_eligibility import (
        SOURCE_MARGIN,
        required_source_families,
    )

    assert required_source_families(list(F2_NAMES)) == frozenset({SOURCE_MARGIN})


def test_r14_family_verdict_and_promotion_unchanged():
    """p24's candidate-pool admission (promotion_allowed=True) survives
    the correction; the evidence states stay distinguishable (I10)."""

    for name in F2_NAMES:
        assert FEATURE_METADATA[name].promotion_allowed is True, name
    # The withdrawn members keep their withdrawn status (the correction
    # must not silently rehabilitate anyone).
    assert FEATURE_METADATA["INDUSTRY_MOMENTUM"].promotion_allowed is False
    assert not (set(NEUTRAL_FEATURE_NAMES) & set(EXTERNAL_FACTOR_NAMES))
