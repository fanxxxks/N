"""Unified content-identity primitives (P8-02, lifecycle contract §2).

``ashare_model.identity`` is the single canonical-JSON + SHA-256
implementation for every content identity in the repository (spec_id,
candidate_id, artifact_id, ledger entry hashes, formula hashes). The
historical byte behavior of ``ledger.canonical_json`` and
``promotion.formula_hash`` is preserved through explicit compatibility
entry points of the same module — parameterization, not a second
implementation. Oracles in these tests are independent stdlib calls,
never the implementation under test.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest

import ashare_model.identity as identity
from ashare_model.ledger import ExperimentLedger, LedgerEntry
from ashare_model.promotion import formula_hash as promotion_formula_hash


# -- the unified module exists and owns the single implementation -----------


def test_unified_module_exposes_contract_surface():
    for name in (
        "canonical_json",
        "canonical_json_strict",
        "content_hash",
        "sha256_hex",
        "formula_hash",
        "CanonicalJSONError",
    ):
        assert hasattr(identity, name), f"identity.{name} missing"


def test_ledger_and_promotion_reexport_the_single_implementation():
    # No second implementation: the historical names ARE the unified ones.
    from ashare_model.ledger import canonical_json as ledger_canonical_json

    assert ledger_canonical_json is identity.canonical_json
    assert promotion_formula_hash is identity.formula_hash


# -- ledger-compatible canonical JSON: byte-identical to the old oracle -----


_LEGACY_ARGS = dict(sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize(
    "value",
    [
        {"b": 1, "a": 2},
        {"z": [1, 2, {"nested": True}], "text": "中文", "none": None},
        {"floats": [0.5, -1.25, 1e-9], "empty": {}, "list": []},
        {"unicode_key": {"内": "值"}},
        [{}, [1, [2, [3]]], "x", 3, False],
    ],
)
def test_canonical_json_matches_legacy_byte_oracle(value):
    assert identity.canonical_json(value) == json.dumps(value, **_LEGACY_ARGS)


def test_canonical_json_is_dict_order_independent():
    assert identity.canonical_json({"a": 1, "b": [2, 3]}) == identity.canonical_json(
        {"b": [2, 3], "a": 1}
    )


def test_ledger_entry_hash_is_the_unified_hash_of_canonical_payload(tmp_path):
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl", run_id="run-identity")
    ledger.record_trial(algorithm="trained", candidate="trained", seed=42)
    line = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
    entry = LedgerEntry.from_json(line)
    payload = {
        name: getattr(entry, name)
        for name in LedgerEntry._fields_except("entry_hash")
    }
    expected = hashlib.sha256(
        json.dumps(payload, **_LEGACY_ARGS).encode("utf-8")
    ).hexdigest()
    assert entry.entry_hash == expected == identity.sha256_hex(
        identity.canonical_json(payload)
    )


# -- strict content-identity mode: rejects NaN/Inf/implicit types -----------


def test_strict_mode_accepts_plain_json_structures():
    value = {"kind": "x", "list": [1, 2.5, None, True, "中文"], "nested": {"a": {}}}
    text = identity.canonical_json_strict(value)
    assert json.loads(text) == value
    assert identity.canonical_json_strict({"b": 1, "a": 2}) == '{"a":2,"b":1}'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_strict_mode_rejects_non_finite_floats(bad):
    with pytest.raises(identity.CanonicalJSONError):
        identity.canonical_json_strict({"x": bad})
    with pytest.raises(identity.CanonicalJSONError):
        identity.canonical_json_strict([bad])


@pytest.mark.parametrize(
    "bad",
    [
        (1, 2),                      # tuple would implicitly serialize as array
        {1, 2},                      # set
        b"bytes",                    # bytes
        object(),                    # arbitrary object
        {"key": object()},           # nested unsupported value
        {1: "int-key"},              # non-string dict key would be coerced
    ],
)
def test_strict_mode_rejects_implicit_unsupported_types(bad):
    with pytest.raises(identity.CanonicalJSONError):
        identity.canonical_json_strict(bad)


def test_strict_mode_rejects_nested_non_finite_deep_in_structure():
    with pytest.raises(identity.CanonicalJSONError):
        identity.canonical_json_strict({"a": [{"b": [math.nan]}]})


# -- content_hash: kind-domain-separated, deterministic ---------------------


def test_content_hash_is_deterministic_and_order_independent():
    first = identity.content_hash("runspec", {"a": 1, "b": [2, 3]})
    second = identity.content_hash("runspec", {"b": [2, 3], "a": 1})
    assert first == second
    assert first == hashlib.sha256(
        identity.canonical_json_strict({"kind": "runspec", "value": {"a": 1, "b": [2, 3]}}).encode("utf-8")
    ).hexdigest()


def test_content_hash_kind_separates_domains():
    value = {"same": "payload"}
    assert identity.content_hash("runspec", value) != identity.content_hash(
        "candidate", value
    )
    with pytest.raises(identity.CanonicalJSONError):
        identity.content_hash("", value)


# -- formula_hash: promotion compatibility is byte-identical ----------------


def test_formula_hash_token_path_matches_legacy_oracle():
    tokens = [3, 1, 2, 40, 41]
    expected = hashlib.sha256(
        json.dumps([int(t) for t in tokens], sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert identity.formula_hash(tokens) == expected


def test_formula_hash_token_coercion_is_legacy_identical():
    # legacy behavior: tokens pass through int(t) before serialization
    assert identity.formula_hash(["3", "1"]) == identity.formula_hash([3, 1])


def test_formula_hash_text_path_is_raw_without_normalization():
    assert identity.formula_hash(None, "a + b") == hashlib.sha256(
        b"a + b"
    ).hexdigest()
    assert identity.formula_hash(None, "a + b") != identity.formula_hash(
        None, "a+b"
    )
    assert identity.formula_hash(None, " a + b ") != identity.formula_hash(
        None, "a + b"
    )


def test_formula_hash_requires_tokens_or_text():
    with pytest.raises(ValueError):
        identity.formula_hash(None, None)
    with pytest.raises(ValueError):
        identity.formula_hash([], None)


def test_formula_hash_text_path_loses_to_tokens_when_both_given():
    # legacy precedence: tokens win when truthy
    assert identity.formula_hash([7], "ignored") == identity.formula_hash([7])
