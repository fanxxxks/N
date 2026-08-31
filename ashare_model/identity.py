"""Unified content-identity primitives (P8-02).

This module is the **single** canonical-JSON + SHA-256 implementation for
every content identity in the repository (lifecycle contract §2,
``docs/p8_research_lifecycle_contract.md``): ledger entry hashes, promotion
formula hashes, and (from P8-03 onward) spec_id / candidate_id /
artifact_id. No other module may implement its own canonicalization or
hashing of content identities.

Two explicit modes, one implementation:

* **strict content-identity mode** (:func:`canonical_json_strict`,
  :func:`content_hash`) — used for all *new* content identities. Rejects
  NaN/Infinity, non-string dict keys and every type that JSON does not
  natively represent (tuples, sets, bytes, arbitrary objects) instead of
  silently coercing them; floats must be finite.
* **compatibility mode** (:func:`canonical_json`, :func:`formula_hash`) —
  the historical byte behavior of ``ashare_model.ledger`` and
  ``ashare_model.promotion``, preserved so existing ledgers and promotion
  records hash identically. These are parameterizations of the same
  implementation, tested byte-pinned against independent stdlib oracles in
  ``tests/test_identity.py``; they are not a second semantic path.

Hashes are lowercase full-length hex SHA-256 over UTF-8 text.
``content_hash`` domain-separates by ``kind`` (``"runspec"``,
``"candidate"``, ``"artifact"``, ...) so a payload of one identity type can
never collide with the same payload under another type.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = [
    "CanonicalJSONError",
    "candidate_id",
    "canonical_json",
    "canonical_json_strict",
    "content_hash",
    "formula_hash",
    "sha256_hex",
]


class CanonicalJSONError(ValueError):
    """A value cannot be canonicalized for content hashing (fail-closed)."""


def _validate_strict(value: Any, path: str = "value") -> None:
    """Recursively whitelist exactly the JSON-representable Python types."""

    if value is None or type(value) is bool:
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalJSONError(
                f"{path}: non-finite float {value!r} is not allowed in "
                "content identity (fail-closed; sanitize or reject upstream)"
            )
        return
    if type(value) is str:
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_strict(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CanonicalJSONError(
                    f"{path}: dict keys must be str for content identity, "
                    f"got {type(key).__name__} ({key!r})"
                )
            _validate_strict(item, f"{path}.{key}")
        return
    raise CanonicalJSONError(
        f"{path}: type {type(value).__name__} is not JSON-representable; "
        "implicit coercion is forbidden in content identity "
        "(convert explicitly upstream)"
    )


def canonical_json_strict(value: Any) -> str:
    """Deterministic JSON text for content identities (strict mode).

    UTF-8, sorted keys, compact separators, non-ASCII preserved; rejects
    non-finite floats and implicit types instead of coercing them.
    """

    _validate_strict(value)
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def content_hash(kind: str, value: Any) -> str:
    """Kind-domain-separated content hash (strict mode).

    The hash input is ``{"kind": kind, "value": value}`` in strict
    canonical form, so the same payload under different identity kinds
    never produces the same digest.
    """

    if type(kind) is not str or not kind:
        raise CanonicalJSONError("content_hash kind must be a non-empty str")
    return sha256_hex(canonical_json_strict({"kind": kind, "value": value}))


def candidate_id(spec_id: str, tokens, direction: int) -> str:
    """Lifecycle content identity of one candidate formula (contract §2).

    ``H(kind="candidate", {spec_id, canonical tokens, direction})`` —
    unique within a spec, never reusable across specs. This is the single
    candidate-identity implementation: searcher-internal candidate labels
    are diagnostics, never lifecycle identity.
    """

    if type(spec_id) is not str or not spec_id:
        raise CanonicalJSONError("candidate_id requires a non-empty spec_id")
    return content_hash(
        "candidate",
        {
            "spec_id": spec_id,
            "tokens": [int(t) for t in tokens],
            "direction": int(direction),
        },
    )


def sha256_hex(text: str) -> str:
    """Lowercase full-length hex SHA-256 of the UTF-8 encoding of ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Ledger-compatible deterministic JSON text (compatibility mode).

    Byte-identical to the historical ``ashare_model.ledger.canonical_json``:
    sorted keys, compact separators, non-ASCII preserved. Kept as an
    explicit compatibility parameterization so existing ledger lines
    re-hash identically; new content identities must use strict mode.
    """

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def formula_hash(tokens, formula_text: str | None = None) -> str:
    """Content hash identifying one formula across artifacts and paper
    windows: the canonical token list when available, else the text.

    Compatibility mode: byte-identical to the historical
    ``ashare_model.promotion.formula_hash`` (token path serializes the
    int-coerced token list with ``sort_keys=True``; text path is raw,
    without normalization; tokens win when both are given).
    """

    if tokens:
        text = json.dumps([int(t) for t in tokens], sort_keys=True)
    elif formula_text:
        text = formula_text
    else:
        raise ValueError("formula_hash needs tokens or formula_text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
