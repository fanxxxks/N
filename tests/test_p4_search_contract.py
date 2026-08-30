"""P4 contract tests.

Assertion source: ``docs/p4_search_transformer_contract.md`` sections 1--3
and 6.  These tests intentionally precede the implementation.
"""

from __future__ import annotations

import pytest

from ashare_data.config import make_model_config
from ashare_model.alphagpt import MODEL_VERSION
from ashare_model.evaluation import PROTOCOL_VERSION
from ashare_model.search_contract import (
    SEARCH_CONTRACT_VERSION,
    SearchBackend,
    SearchRequest,
    SearchResult,
)
from ashare_model.search_backends import get_search_backend, registered_backend_names
from ashare_model.searcher_bench import SEARCHER_BENCH_VERSION


def test_p4_semantic_versions_are_pinned_by_contract():
    # P6 contract §5 bumped PROTOCOL_VERSION 23 -> 24; P7-E contract §4
    # bumps 24 -> 25 and SEARCH_CONTRACT_VERSION 1 -> 2 (semantic-type
    # sampling legality narrows the effective search space; pre/post
    # results are not matched comparisons).
    assert SEARCH_CONTRACT_VERSION == 2
    assert PROTOCOL_VERSION == "25"
    assert MODEL_VERSION == 3
    assert SEARCHER_BENCH_VERSION == 2


def test_model_searcher_accepts_exactly_the_four_registered_backends():
    assert registered_backend_names() == ("gp", "tpe", "random", "rl")
    for name in registered_backend_names():
        cfg = make_model_config({"model": {"searcher": name}})
        assert cfg.searcher == name
        assert isinstance(get_search_backend(name), SearchBackend)
    with pytest.raises(ValueError, match="gp.*tpe.*random.*rl"):
        make_model_config({"model": {"searcher": "unknown"}})


def test_search_request_pins_budget_unit_and_rl_split():
    request = SearchRequest(
        seed=7,
        budget=16,
        max_formula_len=12,
        steps=4,
        batch_size=4,
    )
    assert request.budget == 16
    with pytest.raises(ValueError, match="positive"):
        SearchRequest(seed=7, budget=0, max_formula_len=12)
    with pytest.raises(ValueError, match="both"):
        SearchRequest(seed=7, budget=16, max_formula_len=12, steps=4)
    with pytest.raises(ValueError, match="multiply"):
        SearchRequest(
            seed=7,
            budget=16,
            max_formula_len=12,
            steps=2,
            batch_size=4,
        )


def test_search_result_records_requested_consumed_stagnation_and_curve():
    result = SearchResult(
        backend="gp",
        seed=7,
        requested_budget=4,
        consumed_budget=2,
        termination_reason="proposal_stagnation",
        stagnation_reason="three_generations_without_new_semantic_class",
        best_so_far=((1, -0.4), (2, 0.2)),
    )
    payload = result.to_dict()
    assert payload["contract_version"] == SEARCH_CONTRACT_VERSION
    assert payload["requested_budget"] == 4
    assert payload["consumed_budget"] == 2
    assert payload["termination_reason"] == "proposal_stagnation"
    assert payload["stagnation_reason"] == (
        "three_generations_without_new_semantic_class"
    )
    assert payload["best_so_far"] == [[1, -0.4], [2, 0.2]]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"consumed_budget": 5}, "exceed"),
        ({"best_so_far": ((2, 0.1), (1, 0.2))}, "strictly increasing"),
        ({"best_so_far": ((1, 0.2), (2, 0.1))}, "non-decreasing"),
        ({"best_so_far": ((1, 0.2),)}, "end at consumed budget"),
        ({"stagnation_reason": None}, "stagnation_reason"),
    ],
)
def test_search_result_rejects_untruthful_accounting(overrides, message):
    values = {
        "backend": "gp",
        "seed": 7,
        "requested_budget": 4,
        "consumed_budget": 2,
        "termination_reason": "proposal_stagnation",
        "stagnation_reason": "no_new_class",
        "best_so_far": ((1, 0.1), (2, 0.2)),
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        SearchResult(**values)


def test_zero_consumption_requires_an_empty_curve():
    result = SearchResult(
        backend="random",
        seed=1,
        requested_budget=4,
        consumed_budget=0,
        termination_reason="candidate_pool_exhausted",
        best_so_far=(),
    )
    assert result.best_so_far == ()
    with pytest.raises(ValueError, match="zero consumed"):
        SearchResult(
            backend="random",
            seed=1,
            requested_budget=4,
            consumed_budget=0,
            termination_reason="candidate_pool_exhausted",
            best_so_far=((1, 0.1),),
        )
