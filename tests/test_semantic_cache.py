"""T2-01 contracts: semantic cache and the unique-semantic-evaluation budget.

The budget unit of Phase 2 is the **unique semantic formula evaluation**:
structurally identical formulas (same canonical AST hash) and numerically
equivalent formulas (same rank fingerprint on the fixed calibration slice)
never consume budget twice.  The cache key carries the AST hash plus the
full evaluation context (dataset_id, reward version, protocol version,
window), so scores are never reused across contexts.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_model.ir import FormulaSyntaxError, decode
from ashare_model.semantic_cache import (
    CALIBRATION_DATES,
    SEMANTIC_CACHE_VERSION,
    CalibrationSlice,
    SemanticCache,
)
from ashare_model.vocab import FORMULA_VOCAB


def _tok(*names: str) -> tuple[int, ...]:
    return tuple(
        FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(n)
        for n in names
    )


def _op(name: str) -> int:
    from ashare_model.ops import OPS_CONFIG

    return FORMULA_VOCAB.operator_offset + [
        cfg[0] for cfg in OPS_CONFIG
    ].index(name)


def _add(*names: str) -> tuple[int, ...]:
    return _tok(*names) + (_op("ADD"),)


def _neg(name: str) -> tuple[int, ...]:
    return _tok(name) + (_op("NEG"),)


class ToyExecutor:
    """Deterministic toy formula semantics over a base tensor.

    ``base`` maps feature names to ``[stock x date]`` arrays; ADD/SUB/MUL
    are elementwise, NEG negates, ABS takes absolute values.  Unknown
    operators and malformed sequences return ``None`` (execution failure).
    """

    def __init__(self, base: dict[str, np.ndarray]):
        self.base = {k: np.asarray(v, dtype=np.float64) for k, v in base.items()}

    def __call__(self, tokens: tuple[int, ...]) -> np.ndarray | None:
        try:
            ir = decode(tokens)
        except FormulaSyntaxError:
            return None
        return self._eval(ir)

    def _eval(self, node):
        from ashare_model.ir import Binary, Feature, Ternary, Unary

        if isinstance(node, Feature):
            return self.base.get(node.name)
        if isinstance(node, Unary):
            x = self._eval(node.operand)
            if x is None:
                return None
            if node.op == "NEG":
                return -x
            if node.op == "ABS":
                return np.abs(x)
            return None
        if isinstance(node, Binary):
            l, r = self._eval(node.left), self._eval(node.right)
            if l is None or r is None:
                return None
            if node.op == "ADD":
                return l + r
            if node.op == "SUB":
                return l - r
            if node.op == "MUL":
                return l * r
            return None
        if isinstance(node, Ternary):
            return None
        return None


@pytest.fixture()
def base() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    n_stocks, n_dates = 24, 40
    ret1 = np.cumsum(rng.normal(0, 1, size=(n_stocks, n_dates)), axis=1)
    ret5 = ret1[:, ::2]  # correlated with RET_1 (same cross-section pattern)
    ret5 = np.repeat(ret5, 2, axis=1)[:, :n_dates]
    ret10 = rng.normal(0, 1, size=(n_stocks, n_dates))
    return {"RET_1": ret1, "RET_5": ret5, "RET_10": ret10}


def _cache(**overrides) -> SemanticCache:
    kwargs = dict(
        dataset_id="dataset-a",
        reward_version=13,
        protocol_version=18,
        window_id="train:0:100",
    )
    kwargs.update(overrides)
    return SemanticCache(**kwargs)


# --- calibration slice ----------------------------------------------------


def test_calibration_slice_is_fixed_head_window():
    assert CALIBRATION_DATES == 60
    assert CalibrationSlice.of(n_dates=2800) == CalibrationSlice(0, 60)
    assert CalibrationSlice.of(n_dates=30) == CalibrationSlice(0, 30)
    assert str(CalibrationSlice(0, 60)) == "dates:0:60"


# --- cache keys -----------------------------------------------------------


def test_key_for_valid_formula():
    cache = _cache()
    key = cache.key_for(_tok("RET_1"))
    assert key is not None
    assert key.structural_hash  # canonical AST hash
    assert key.dataset_id == "dataset-a"
    assert key.reward_version == 13
    assert key.protocol_version == 18
    assert key.window_id == "train:0:100"


def test_key_is_canonical_across_commuted_forms():
    cache = _cache()
    assert cache.key_for(_add("RET_1", "RET_5")) == cache.key_for(
        _add("RET_5", "RET_1")
    )


def test_key_for_invalid_and_degenerate_is_none():
    cache = _cache()
    assert cache.key_for(()) is None  # invalid
    # SUB(x, x) is a degenerate constant formula: never evaluated.
    assert cache.key_for(_tok("RET_1", "RET_1") + (_op("SUB"),)) is None


def test_key_changes_with_every_context_field():
    base_key = _cache().key_for(_tok("RET_1"))
    for overrides in (
        {"dataset_id": "dataset-b"},
        {"reward_version": 14},
        {"protocol_version": 17},
        {"window_id": "train:0:50"},
    ):
        other = _cache(**overrides).key_for(_tok("RET_1"))
        assert other is not None and other != base_key, overrides


# --- semantic fingerprints ------------------------------------------------


def test_fingerprint_is_deterministic(base):
    cache = _cache()
    ex = ToyExecutor(base)
    first = cache.fingerprint(_tok("RET_1"), ex)
    second = cache.fingerprint(_tok("RET_1"), ex)
    assert first is not None
    assert first == second


def test_fingerprint_dedups_monotone_equivalents(base):
    # ADD(x, x) == 2x has the same cross-sectional rank pattern as x.
    cache = _cache()
    ex = ToyExecutor(base)
    assert cache.fingerprint(_add("RET_1", "RET_1"), ex) == cache.fingerprint(
        _tok("RET_1"), ex
    )


def test_fingerprint_distinguishes_sign_flip(base):
    cache = _cache()
    ex = ToyExecutor(base)
    assert cache.fingerprint(_neg("RET_1"), ex) != cache.fingerprint(
        _tok("RET_1"), ex
    )


def test_fingerprint_of_constants_collapses(base):
    # Any constant signal ranks identically: one degenerate semantic class.
    cache = _cache()
    ex = ToyExecutor(base)
    const_a = cache.fingerprint(_tok("RET_1", "RET_1") + (_op("SUB"),), ex)
    const_b = cache.fingerprint(_tok("RET_5", "RET_5") + (_op("SUB"),), ex)
    assert const_a is not None and const_a == const_b


def test_fingerprint_failure_is_none(base):
    cache = _cache()
    ex = ToyExecutor(base)
    assert cache.fingerprint(_tok("RET_1") + (_op("MA20"),), ex) is None
    assert cache.fingerprint((), ex) is None


# --- budget accounting -----------------------------------------------------


def _score(value: float):
    return {"reward": value}


def test_first_evaluation_consumes_budget(base):
    cache = _cache(cap=64)
    ex = ToyExecutor(base)
    key = cache.key_for(_tok("RET_1"))
    fp = cache.fingerprint(_tok("RET_1"), ex)
    cache.put(key, _score(1.0), fp)
    assert cache.budget_used == 1
    assert cache.n_semantic_dedups == 0


def test_structural_duplicate_reuses_without_budget(base):
    cache = _cache(cap=64)
    ex = ToyExecutor(base)
    # ADD(a, b) and ADD(b, a) are the same canonical formula.
    k1 = cache.key_for(_add("RET_1", "RET_5"))
    k2 = cache.key_for(_add("RET_5", "RET_1"))
    assert k1 == k2
    cache.put(k1, _score(1.0), cache.fingerprint(_add("RET_1", "RET_5"), ex))
    assert cache.get(k2) == _score(1.0)
    assert cache.budget_used == 1
    assert cache.n_hits == 1


def test_semantic_duplicate_reuses_without_budget(base):
    cache = _cache(cap=64)
    ex = ToyExecutor(base)
    # x and ADD(x, x) differ structurally but share a fingerprint — and, at
    # equal complexity bills, the score including its penalty is exactly
    # reusable.
    k1 = cache.key_for(_tok("RET_1"))
    k2 = cache.key_for(_add("RET_1", "RET_1"))
    assert k1 != k2
    fp = cache.fingerprint(_tok("RET_1"), ex)
    cache.put(k1, _score(1.0), fp, complexity_bill=1.0)
    assert cache.budget_used == 1
    cache.put(k2, _score(1.0), fp, complexity_bill=1.0)
    assert cache.budget_used == 1  # no second evaluation
    assert cache.n_semantic_dedups == 1
    assert cache.score_by_fingerprint(fp, complexity_bill=1.0) == _score(1.0)


def test_semantic_class_includes_complexity_bill(base):
    # Two numerically equivalent formulas with different complexity bills
    # carry different penalties: their scores are not interchangeable, so
    # they are distinct semantic classes and both consume budget.
    cache = _cache(cap=64)
    ex = ToyExecutor(base)
    fp = cache.fingerprint(_tok("RET_1"), ex)
    k1 = cache.key_for(_tok("RET_1"))
    k2 = cache.key_for(_add("RET_1", "RET_1"))
    cache.put(k1, _score(1.0), fp, complexity_bill=1.0)
    cache.put(k2, _score(0.8), fp, complexity_bill=1.7)
    assert cache.budget_used == 2
    assert cache.n_semantic_dedups == 0
    assert cache.score_by_fingerprint(fp, complexity_bill=1.0) == _score(1.0)
    assert cache.score_by_fingerprint(fp, complexity_bill=1.7) == _score(0.8)
    assert cache.score_by_fingerprint(fp) is None  # no bill -> distinct class


def test_is_claimed_tracks_in_flight_classes(base):
    cache = _cache(cap=64)
    ex = ToyExecutor(base)
    fp = cache.fingerprint(_tok("RET_1"), ex)
    assert not cache.is_claimed(fp, complexity_bill=1.0)
    key = cache.key_for(_tok("RET_1"))
    cache.put(key, None, fp, complexity_bill=1.0)  # placeholder claim
    assert cache.is_claimed(fp, complexity_bill=1.0)
    assert cache.budget_used == 1
    cache.put(key, _score(1.0), None)  # real score overwrites the placeholder
    assert cache.get(key) == _score(1.0)
    assert cache.budget_used == 1


def test_put_without_fingerprint_does_not_consume_budget():
    cache = _cache(cap=64)
    key = cache.key_for(_tok("RET_1"))
    cache.put(key, _score(-2.0), fingerprint=None)
    assert cache.budget_used == 0
    assert cache.get(key) == _score(-2.0)


def test_budget_survives_lru_eviction(base):
    cache = _cache(cap=2)
    ex = ToyExecutor(base)
    for name, value in (("RET_1", 1.0), ("RET_5", 2.0), ("RET_10", 3.0)):
        fp = cache.fingerprint(_tok(name), ex)
        cache.put(cache.key_for(_tok(name)), _score(value), fp)
    assert cache.budget_used == 3  # evictions never refund budget
    # The oldest entry is gone; the newest is still cached.
    assert cache.get(cache.key_for(_tok("RET_1"))) is None
    assert cache.get(cache.key_for(_tok("RET_10"))) == _score(3.0)


def test_stats_shape(base):
    cache = _cache(cap=64)
    ex = ToyExecutor(base)
    fp = cache.fingerprint(_tok("RET_1"), ex)
    cache.put(cache.key_for(_tok("RET_1")), _score(1.0), fp)
    stats = cache.stats()
    assert stats["budget_used"] == 1
    assert stats["n_hits"] == 0
    assert stats["n_semantic_dedups"] == 0
    assert stats["cap"] == 64


def test_cache_version_is_pinned():
    assert SEMANTIC_CACHE_VERSION == 1
