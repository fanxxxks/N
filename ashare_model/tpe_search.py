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
duplicate and claimed proposals get the run's punitive worst-so-far
anchor (p14 §5.2 — never the current best, which poisoned the surrogate
at the reward plateau).  The proposal distribution carries the per-
position EOS length prior (p14 §5.3, profile ``p14-uniform-2-11-v1``).
The search stops when the budget is exhausted or the proposal
stream stalls (repeatedly re-proposing already-evaluated classes).
"""

from __future__ import annotations

import math
import random

import torch
import optuna

from .alphagpt import build_action_mask
from .baseline_harness import SemanticBudgetEvaluator
from .feature_metadata import FEATURE_METADATA, SemanticType
from .ops import OPS_CONFIG
from .search_contract import SearchResult
from .search_length_prior import sample_target_content_length
from .semantic_sampling import from_id, resolve_output, type_id
from .vocab import FormulaVocab

_OPERATOR_ARITY = {name: arity for name, _, arity in OPS_CONFIG}


def _legal_tokens(
    vocab: FormulaVocab,
    stack: int,
    done: bool,
    step: int,
    max_len: int,
    feature_ids: list[int] | None = None,
    stack_types: list[int] | None = None,
) -> list[int]:
    """Legal token ids at one position of the mask-legal sampling walk.
    ``feature_ids`` (P6 §4.2) restricts the open feature tokens."""

    capacity = max(max_len, len(stack_types or []) + 1)
    types_tensor = torch.zeros(1, capacity, dtype=torch.long)
    if stack_types:
        types_tensor[0, : len(stack_types)] = torch.tensor(
            stack_types, dtype=torch.long
        )
    mask = build_action_mask(
        torch.tensor([stack], dtype=torch.long),
        torch.tensor([done], dtype=torch.bool),
        step,
        max_len,
        vocab,
        feature_ids=feature_ids,
        stack_types=types_tensor,
    )
    return [t for t in range(vocab.size) if mask[0, t] == 0.0]


def _advance(
    vocab: FormulaVocab,
    token: int,
    stack: int,
    types: list[int],
    done: bool,
):
    """Advance the sampling state by one token (stack-only postfix rules
    plus the P7-E semantic-type state, rules from
    :mod:`ashare_model.semantic_sampling`)."""

    if done:
        return stack, types, done
    if token == vocab.pad_token_id:
        return stack, types, True
    if vocab.eos_token_id is not None and token == vocab.eos_token_id:
        return stack, types, True
    if token < vocab.operator_offset:
        name = vocab.feature_names[token - vocab.feature_offset]
        types.append(type_id(FEATURE_METADATA[name].semantic_type))
        return stack + 1, types, done
    op_index = token - vocab.operator_offset
    name = vocab.operator_names[op_index]
    arity = _OPERATOR_ARITY[name]
    arg_types = tuple(from_id(i) for i in types[len(types) - arity :])
    del types[len(types) - arity :]
    types.append(type_id(resolve_output(name, arg_types)))
    return stack - arity + 1, types, done


def run_tpe_baseline(
    *,
    seed: int,
    evaluator: SemanticBudgetEvaluator,
    max_formula_len: int,
    vocab: FormulaVocab | None = None,
    feature_ids: list[int] | None = None,
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
    ``feature_ids`` (P6 §4.2) restricts the open feature tokens.
    """

    vocab = vocab or evaluator.vocab
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        n_ei_candidates=24,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    choices = list(range(vocab.size))
    # P14 §5.3: the per-trial target content length is drawn from the
    # preregistered profile (p14-uniform-2-11-v1) on a run-local seeded
    # stream — deterministic in the run seed, independent of Optuna's
    # surrogate sampling.
    length_rng = random.Random(seed)

    def propose(trial) -> tuple[int, ...]:
        tokens: list[int] = []
        stack, done = 0, False
        types: list[int] = []
        # P14 §5.3: each trial draws a target content length from the
        # preregistered profile (p14-uniform-2-11-v1) and walks under the
        # matching effective position bound, so the mask forces EOS
        # termination at the target — the induced proposal length
        # distribution matches the profile while the legal set, the
        # EOS-inclusive cap and the surrogate's index domain are untouched.
        target_content = sample_target_content_length(
            length_rng, max_formula_len
        )
        trial_max_len = (
            max_formula_len if target_content is None else target_content + 1
        )
        for step in range(int(trial_max_len)):
            legal = _legal_tokens(
                vocab, stack, done, step, trial_max_len, feature_ids,
                stack_types=types,
            )
            if not legal:
                break  # the mask contract guarantees at least one legal token
            index = int(trial.suggest_categorical(f"pos_{step}", choices))
            token = legal[index % len(legal)]
            tokens.append(token)
            stack, types, done = _advance(vocab, token, stack, types, done)
        return tuple(tokens)

    def tell(trial, tokens) -> None:
        score = evaluator.score_of(tokens)
        if score is not None and math.isfinite(float(score.val_reward)):
            study.tell(trial, float(score.val_reward))
        else:
            # Skipped proposals (duplicate, degenerate or claimed classes)
            # and non-finite rewards get the run's punitive worst-so-far
            # anchor — strictly no better than every evaluated proposal —
            # so the surrogate learns to avoid duplicate-generating regions
            # (p14 §5.2; the old "current best" tell poisoned the surrogate
            # at the reward plateau).  Optuna 4.9.0 accepts ±inf objectives
            # (telling NaN would fail the trial with a UserWarning) — the
            # punitive path never sends NaN: non-finite real scores route
            # here too.
            study.tell(trial, evaluator.worst_reward)

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
