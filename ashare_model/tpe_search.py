"""Optuna TPE baseline (T2-02).

The baseline searches the policy's exact search space with Optuna's
TPESampler — a mature, maintained implementation of tree-structured
Parzen-estimator search — instead of a hand-rolled surrogate.

Parameterization: a trial is a full ``max_formula_len`` token sequence,
suggested position by position as categoricals over the whole vocabulary
(a fixed choice set, as Optuna requires).  The suggested index is mapped
into the **legal action set of the current stack state** (the same
``build_action_mask`` legality rules the policy samples under), so every
proposal is a structurally valid formula: no trial is wasted on invalid
sequences and the surrogate learns the same space the RL policy and the
random baseline search.  Early positions have path-independent legal sets,
so the index mapping is stable there; later positions vary over the few
stack-size paths.

Evaluation is billed through the shared
:class:`~ashare_model.baseline_harness.SemanticBudgetEvaluator`: only
unique semantic evaluations consume budget; invalid, degenerate,
duplicate and claimed proposals get the current best reward as a neutral
value.  The search stops when the budget is exhausted or the proposal
stream stalls (repeatedly re-proposing already-evaluated classes).
"""

from __future__ import annotations

import math

import torch
import optuna

from .alphagpt import build_action_mask
from .baseline_harness import SemanticBudgetEvaluator
from .ops import OPS_CONFIG
from .search_contract import SearchResult
from .vocab import FormulaVocab

_OPERATOR_ARITY = {name: arity for name, _, arity in OPS_CONFIG}


def _legal_tokens(
    vocab: FormulaVocab, stack: int, done: bool, step: int, max_len: int
) -> list[int]:
    """Legal token ids at one position of the mask-legal sampling walk."""

    mask = build_action_mask(
        torch.tensor([stack], dtype=torch.long),
        torch.tensor([done], dtype=torch.bool),
        step,
        max_len,
        vocab,
    )
    return [t for t in range(vocab.size) if mask[0, t] == 0.0]


def _advance(vocab: FormulaVocab, token: int, stack: int, done: bool):
    """Advance the sampling state by one token (stack-only postfix rules)."""

    if done:
        return stack, done
    if token == vocab.pad_token_id:
        return stack, True
    if vocab.eos_token_id is not None and token == vocab.eos_token_id:
        return stack, True
    if token < vocab.operator_offset:
        return stack + 1, done
    op_index = token - vocab.operator_offset
    arity = _OPERATOR_ARITY[vocab.operator_names[op_index]]
    return stack - arity + 1, done


def run_tpe_baseline(
    *,
    seed: int,
    evaluator: SemanticBudgetEvaluator,
    max_formula_len: int,
    vocab: FormulaVocab | None = None,
    n_startup_trials: int = 10,
    stall_limit: int = 50,
    tell_batch: int = 16,
) -> SearchResult:
    """TPE search over mask-legal token sequences under the semantic budget.

    ``n_startup_trials`` keeps Optuna's default random-startup phase
    (uniform proposals before the surrogate kicks in).  Trials are asked
    in batches of ``tell_batch`` and told after the evaluator flushed the
    batched scoring, so the search keeps the memory-bounded chunked
    evaluation instead of paying per-proposal scoring overhead.  The
    search stops when the evaluation budget is exhausted or
    ``stall_limit`` consecutive trials produced no new evaluation.
    """

    vocab = vocab or evaluator.vocab
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        n_ei_candidates=24,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    choices = list(range(vocab.size))

    def propose(trial) -> tuple[int, ...]:
        tokens: list[int] = []
        stack, done = 0, False
        for step in range(int(max_formula_len)):
            legal = _legal_tokens(vocab, stack, done, step, max_formula_len)
            if not legal:
                break  # the mask contract guarantees at least one legal token
            index = int(trial.suggest_categorical(f"pos_{step}", choices))
            token = legal[index % len(legal)]
            tokens.append(token)
            stack, done = _advance(vocab, token, stack, done)
        return tuple(tokens)

    def tell(trial, tokens) -> None:
        score = evaluator.score_of(tokens)
        if score is not None and math.isfinite(float(score.val_reward)):
            study.tell(trial, float(score.val_reward))
        else:
            # Skipped proposals (duplicate, degenerate or claimed classes)
            # and non-finite rewards get the current best as a neutral
            # value: they neither help nor hurt the surrogate (Optuna
            # rejects NaN objectives outright).
            study.tell(trial, evaluator.best_reward)

    stall = 0
    batch: list = []
    while evaluator.budget_used < evaluator.budget:
        budget_before = evaluator.budget_used
        trial = study.ask()
        tokens = propose(trial)
        evaluator.propose(tokens)
        batch.append((trial, tokens))
        if len(batch) >= tell_batch or evaluator.budget_used >= evaluator.budget:
            evaluator.flush()
            for trial, tokens in batch:
                tell(trial, tokens)
            batch = []
        if evaluator.budget_used == budget_before:
            stall += 1
            if stall >= stall_limit:
                break  # the proposal stream stopped yielding new classes
        else:
            stall = 0
    if batch:
        evaluator.flush()
        for trial, tokens in batch:
            tell(trial, tokens)
    stalled = evaluator.budget_used < evaluator.budget
    return evaluator.finish(
        backend="tpe",
        seed=seed,
        termination_reason=(
            "proposal_stagnation" if stalled else "budget_exhausted"
        ),
        stagnation_reason=(
            f"{stall_limit}_trials_without_new_semantic_class" if stalled else None
        ),
        diagnostics={"consecutive_stalled_trials": stall},
    )
