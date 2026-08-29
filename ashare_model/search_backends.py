"""Lazy registry of the four formal formula-search backends."""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from .search_contract import SearchBackend, SearchRequest, SearchResult

_BACKEND_NAMES = ("gp", "tpe", "random", "rl")


def _require_matching_budget(request: SearchRequest, evaluator: object) -> None:
    evaluator_budget = int(getattr(evaluator, "budget"))
    if evaluator_budget != int(request.budget):
        raise ValueError(
            f"evaluator budget {evaluator_budget} does not match request "
            f"budget {request.budget}"
        )


class GPSearchBackend:
    name = "gp"

    def search(self, request, evaluator, **backend_context) -> SearchResult:
        if evaluator is None:
            raise ValueError("gp requires a SemanticBudgetEvaluator")
        _require_matching_budget(request, evaluator)
        from .gp_search import run_gp_baseline

        return run_gp_baseline(
            seed=request.seed,
            evaluator=evaluator,
            max_formula_len=request.max_formula_len,
            vocab=backend_context.get("vocab"),
        )


class TPESearchBackend:
    name = "tpe"

    def search(self, request, evaluator, **backend_context) -> SearchResult:
        if evaluator is None:
            raise ValueError("tpe requires a SemanticBudgetEvaluator")
        _require_matching_budget(request, evaluator)
        # Optuna stays lazy: selecting GP/Random/RL does not require importing
        # the optional TPE dependency.
        from .tpe_search import run_tpe_baseline

        return run_tpe_baseline(
            seed=request.seed,
            evaluator=evaluator,
            max_formula_len=request.max_formula_len,
            vocab=backend_context.get("vocab"),
        )


class RandomSearchBackend:
    name = "random"

    def search(self, request, evaluator, **backend_context) -> SearchResult:
        if evaluator is None:
            raise ValueError("random requires a SemanticBudgetEvaluator")
        _require_matching_budget(request, evaluator)
        from .baseline_harness import canonical_form_pool

        # Semantic duplicates can consume proposal headroom without consuming
        # budget.  A bounded 8x pool is the established random-search policy.
        pool = canonical_form_pool(
            request.seed,
            backend_context.get("vocab") or evaluator.vocab,
            request.max_formula_len,
            request.budget * 8,
        )
        for key in pool:
            evaluator.propose(key)
            if evaluator.budget_used >= evaluator.budget:
                break
        reason = (
            "budget_exhausted"
            if evaluator.budget_used >= evaluator.budget
            else "candidate_pool_exhausted"
        )
        return evaluator.finish(
            backend=self.name,
            seed=request.seed,
            termination_reason=reason,
        )


class RLSearchBackend:
    name = "rl"

    def search(self, request, evaluator, **backend_context) -> SearchResult:
        runner = backend_context.get("runner")
        if not isinstance(runner, Callable):
            raise ValueError("rl backend requires a callable runner context")
        result = runner(request, evaluator)
        if not isinstance(result, SearchResult) or result.backend != self.name:
            raise TypeError("rl runner must return an rl SearchResult")
        return result


_BACKENDS: dict[str, SearchBackend] = {
    "gp": GPSearchBackend(),
    "tpe": TPESearchBackend(),
    "random": RandomSearchBackend(),
    "rl": RLSearchBackend(),
}


def registered_backend_names() -> tuple[str, ...]:
    return _BACKEND_NAMES


def get_search_backend(name: str) -> SearchBackend:
    try:
        return _BACKENDS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown search backend {name!r}; choose from {_BACKEND_NAMES}"
        ) from exc


def log_search_start(backend: str, request: SearchRequest) -> None:
    logger.info(
        "search.start backend={} seed={} requested_budget={}",
        backend,
        request.seed,
        request.budget,
    )


def log_search_stop(result: SearchResult) -> None:
    best = result.best_so_far[-1][1] if result.best_so_far else None
    logger.info(
        "search.stop backend={} seed={} requested_budget={} consumed_budget={} "
        "termination_reason={} stagnation_reason={} best_val_reward={}",
        result.backend,
        result.seed,
        result.requested_budget,
        result.consumed_budget,
        result.termination_reason,
        result.stagnation_reason,
        best,
    )
