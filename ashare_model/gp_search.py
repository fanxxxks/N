"""DEAP strongly-typed genetic-programming baseline (T2-02).

The baseline reuses DEAP's official typed-tree machinery (primitive set,
half-and-half generation, one-point crossover, uniform mutation,
tournament selection) instead of reimplementing a GP engine.  What this
module adds is the mapping between DEAP trees and the formula grammar:

* terminals are the vocabulary's features, primitives are the operators
  with their exact arities — every tree is a strongly-typed formula;
* a tree maps to canonical postfix token sequences and back
  (:func:`tree_to_tokens` / :func:`tokens_to_tree`);
* the tree size is capped so every proposal fits the policy's
  ``max_formula_len`` (``2 * nodes <= max_formula_len``, EOS included);
* evaluation is billed through the shared
  :class:`~ashare_model.baseline_harness.SemanticBudgetEvaluator`, so the
  GP consumes the same unique-semantic-evaluation budget as every other
  searcher.

The search loop is a classic generational GP: tournament selection,
one-point crossover and uniform mutation with depth-bounded re-generation,
running until the evaluation budget is exhausted.  Deterministic in
``seed`` (DEAP draws from Python's ``random`` module).
"""

from __future__ import annotations

import random as py_random

import numpy as np
from deap import base, creator, gp, tools

from .baseline_harness import SemanticBudgetEvaluator
from .feature_metadata import FEATURE_METADATA, SemanticType
from .operator_registry import OPERATOR_REGISTRY
from .ops import OPS_BY_NAME, OPS_CONFIG
from .search_contract import SearchResult
from .semantic_sampling import fixed_output, is_gate, same_type_args
from .vocab import FormulaVocab

# DEAP creator classes are process-global; create once, reuse forever.
_CREATOR_READY = False


def _ensure_creator() -> None:
    global _CREATOR_READY
    if _CREATOR_READY:
        return
    if not hasattr(creator, "FitnessMaxFormula"):
        creator.create("FitnessMaxFormula", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualFormula"):
        creator.create(
            "IndividualFormula",
            gp.PrimitiveTree,
            fitness=creator.FitnessMaxFormula,
        )
    _CREATOR_READY = True


# DEAP typed primitives need a real class as the type token (it calls
# ``issubclass`` on it).  P7-E: one class per semantic type (contract:
# docs/p7_semantic_types_contract.md) instead of the single pre-P7
# ``Signal`` type, so a tree can only combine operator applications whose
# registered signatures match.
_TYPE_CLASSES: dict[SemanticType, type] = {}


def _type_class(semantic_type: SemanticType) -> type:
    cls = _TYPE_CLASSES.get(semantic_type)
    if cls is None:
        cls = type(
            f"Signal_{semantic_type.value}", (), {"__slots__": ()}
        )
        _TYPE_CLASSES[semantic_type] = cls
    return cls


class Signal:
    """Root type token of the primitive set (generation always passes an
    explicit continuous root family via :func:`_typed_expr`, so this root
    is never produced by a primitive)."""


def build_pset(vocab: FormulaVocab, feature_ids: list[int] | None = None):
    """Strongly-typed primitive set over the formula vocabulary.

    Every feature is a terminal typed by its authored semantic type
    (:mod:`ashare_model.feature_metadata`) and every operator is
    registered once per legal signature from
    :mod:`ashare_model.operator_registry` plus the contract's output
    rules — the token mapping only reads the primitive *name*, so the
    per-signature variants map to the same vocabulary token and the
    search space never drifts from the registry.  ``feature_ids`` (P6
    §4.2) restricts the terminal set to the given feature tokens;
    ``None`` keeps every vocabulary feature.
    """

    pset = gp.PrimitiveSetTyped("FORMULA", [], Signal)
    # P9 §4.2: deprecated features never enter the GP terminal set (they
    # keep their token ids for legacy formula resolution, but they are not
    # samplable — the same policy the action mask enforces for the other
    # backends).
    deprecated = vocab.deprecated_names
    if feature_ids is None:
        feature_names = [
            name for name in vocab.feature_names if name not in deprecated
        ]
    else:
        allowed = {int(token) for token in feature_ids}
        feature_names = [
            name
            for token, name in enumerate(
                vocab.feature_names, start=vocab.feature_offset
            )
            if token in allowed and name not in deprecated
        ]
    for name in feature_names:
        pset.addTerminal(name, _type_class(FEATURE_METADATA[name].semantic_type))
    for name in vocab.operator_names:
        try:
            arity = OPS_BY_NAME[name][1]
        except KeyError:
            raise ValueError(
                f"operator {name!r} has no implementation/signature in OPS_BY_NAME"
            ) from None
        meta = OPERATOR_REGISTRY[name]
        fixed = fixed_output(name)
        arg_sets = [
            set(meta.inputs[j])
            - ({SemanticType.BOOLEAN_EVENT_SIGNAL} if not is_gate(name) or j else set())
            for j in range(arity)
        ]
        order = tuple(SemanticType)
        if same_type_args(name):
            # One shared family T across the branch arguments; the GATE
            # condition varies over its (any-type) allowed set.
            branch_sets = arg_sets[1:] if is_gate(name) else arg_sets
            common = set.intersection(*branch_sets) if branch_sets else set()
            conditions = (
                [semantic_type for semantic_type in order if semantic_type in arg_sets[0]]
                if is_gate(name)
                else [None]
            )
            for cond in conditions:
                for semantic_type in [t for t in order if t in common]:
                    in_types = []
                    if is_gate(name):
                        in_types.append(_type_class(cond))
                    in_types.extend(
                        [_type_class(semantic_type)]
                        * (arity - (1 if is_gate(name) else 0))
                    )
                    pset.addPrimitive(
                        _op_placeholder(name),
                        in_types,
                        _type_class(semantic_type),
                        name=name,
                    )
        elif fixed is not None and arity == 2:
            # Cross-type fixed-output ops (DIV/CORR): product of the
            # per-argument continuous sets.
            for first_type in order:
                if first_type not in arg_sets[0]:
                    continue
                for second_type in order:
                    if second_type not in arg_sets[1]:
                        continue
                    pset.addPrimitive(
                        _op_placeholder(name),
                        [_type_class(first_type), _type_class(second_type)],
                        _type_class(fixed),
                        name=name,
                    )
        else:
            for semantic_type in [t for t in order if t in arg_sets[0]]:
                pset.addPrimitive(
                    _op_placeholder(name),
                    [_type_class(semantic_type)] * arity,
                    _type_class(fixed)
                    if fixed is not None
                    else _type_class(semantic_type),
                    name=name,
                )
    return pset


# Root families a typed tree may produce (the VM z-scores any family, so
# every continuous root is a legal formula result).
ROOT_TYPES = tuple(t for t in SemanticType if t is not SemanticType.BOOLEAN_EVENT_SIGNAL)


def _typed_expr(pset, min_: int, max_: int):
    """Generate one typed tree, retrying DEAP's explicit no-terminal
    failure for feature-restricted primitive sets.

    With the full vocabulary the first draw chooses uniformly from all five
    continuous root families.  A restricted research domain can make a
    drawn family impossible at the sampled depth; retrying conditions on a
    tree that can actually be expressed without opening out-of-domain
    terminals.  The retry stream is seed-deterministic and bounded.
    """

    error: IndexError | None = None
    for _ in range(100):
        type_ = _type_class(py_random.choice(ROOT_TYPES))
        try:
            return gp.genHalfAndHalf(
                pset, min_=min_, max_=max_, type_=type_
            )
        except IndexError as exc:
            error = exc
    raise ValueError(
        "typed GP could not generate an expressible tree in 100 attempts"
    ) from error


def _typed_mutation_expr(pset, min_: int, max_: int, type_):
    """Generate a type-compatible mutation subtree without silently
    widening a feature-restricted terminal set."""

    error: IndexError | None = None
    for _ in range(100):
        try:
            return gp.genFull(pset, min_=min_, max_=max_, type_=type_)
        except IndexError as exc:
            error = exc
    raise ValueError(
        "typed GP could not generate a mutation subtree in 100 attempts"
    ) from error


_PLACEHOLDERS: dict[str, callable] = {}


def _op_placeholder(name: str):
    """Placeholder callable for a primitive (DEAP never invokes it: tree
    evaluation is our token mapping, not DEAP's compiled evaluator).

    Memoized per name: DEAP registers each signature variant under the
    same operator name and requires the stored callable to be identical
    (``context[name] is primitive``), so every variant shares one object.
    """

    fn = _PLACEHOLDERS.get(name)
    if fn is None:
        def _unused(*args):
            raise RuntimeError(f"DEAP evaluator must not run; operator {name}")

        _PLACEHOLDERS[name] = _unused
        return _unused
    return fn


def tree_to_tokens(tree, vocab: FormulaVocab) -> list[int]:
    """Postfix token sequence of a DEAP tree (no EOS: the caller appends
    the terminator when a full sequence is needed).

    DEAP stores the tree in prefix order (a Primitive's ``args`` are its
    type tokens, not its children), so the postfix emission is an
    arity-driven recursive walk over the prefix list.
    """

    nodes = list(tree)

    def walk(index: int) -> tuple[list[int], int]:
        node = nodes[index]
        if isinstance(node, gp.Terminal):
            return [
                vocab.feature_offset + vocab.feature_names.index(node.value)
            ], index + 1
        tokens: list[int] = []
        pos = index + 1
        for _ in range(node.arity):
            child_tokens, pos = walk(pos)
            tokens.extend(child_tokens)
        tokens.append(
            vocab.operator_offset + vocab.operator_names.index(node.name)
        )
        return tokens, pos

    tokens, _ = walk(0)
    return tokens


def _prefix_nodes(node) -> list:
    """Prefix-order node list of a primitive subtree (DEAP's layout)."""

    out = [node]
    for child in getattr(node, "args", ()):
        out.extend(_prefix_nodes(child))
    return out


def tokens_to_tree(tokens, vocab: FormulaVocab):
    """DEAP tree for a canonical postfix token sequence (no EOS), or
    ``None`` when the sequence is not a well-typed tree (e.g. it contains
    operators this grammar cannot express as a tree — the grammar is
    tree-shaped, so this only fails on invalid input)."""

    _ensure_creator()
    stack: list = []
    for token in tokens:
        token = int(token)
        if token < vocab.operator_offset:
            feature_idx = token - vocab.feature_offset
            if not (0 <= feature_idx < len(vocab.feature_names)):
                return None
            stack.append(
                gp.Terminal(vocab.feature_names[feature_idx], False, Signal)
            )
            continue
        op_index = token - vocab.operator_offset
        if not (0 <= op_index < len(vocab.operator_names)):
            return None
        name = vocab.operator_names[op_index]
        arity = _op_arity(name)
        if len(stack) < arity:
            return None
        args = [stack.pop() for _ in range(arity)]
        args.reverse()
        stack.append(gp.Primitive(name, args, arity))
    if len(stack) != 1:
        return None
    tree = gp.PrimitiveTree(_prefix_nodes(stack[0]))
    if len(tree) != len(tokens):
        return None
    return tree


def _op_arity(name: str) -> int:
    for op_name, _, arity in OPS_CONFIG:
        if op_name == name:
            return arity
    raise KeyError(name)


def _max_nodes(max_formula_len: int) -> int:
    """Largest tree that fits the policy's length budget: a tree of ``n``
    nodes serializes to ``2n - 1`` postfix tokens plus EOS <= max_len."""

    return max(1, int(max_formula_len) // 2)


def run_gp_baseline(
    *,
    seed: int,
    evaluator: SemanticBudgetEvaluator,
    max_formula_len: int,
    vocab: FormulaVocab | None = None,
    feature_ids: list[int] | None = None,
    pop_size: int = 40,
    cxpb: float = 0.6,
    mutpb: float = 0.3,
    tournsize: int = 3,
) -> SearchResult:
    """Generational strongly-typed GP search under the semantic budget.

    Proposals that the evaluator skips (invalid, degenerate, duplicate or
    claimed classes) receive the current best reward as a neutral fitness;
    only full evaluations consume budget.  The search stops when the
    budget is exhausted or a generation produced no new evaluation.
    ``feature_ids`` (P6 §4.2) restricts the terminal set.
    """

    _ensure_creator()
    vocab = vocab or evaluator.vocab
    py_random.seed(seed)
    np.random.seed(seed)
    pset = build_pset(vocab, feature_ids=feature_ids)
    node_cap = _max_nodes(max_formula_len)

    toolbox = base.Toolbox()
    toolbox.register(
        "expr", _typed_expr, pset=pset, min_=1, max_=2
    )
    toolbox.register(
        "individual", tools.initIterate, creator.IndividualFormula, toolbox.expr
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "select", tools.selTournament, tournsize=tournsize
    )
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", _typed_mutation_expr, min_=0, max_=2)
    toolbox.register(
        "mutate",
        gp.mutUniform,
        expr=toolbox.expr_mut,
        pset=pset,
    )

    def fresh_individual() -> "creator.IndividualFormula":
        for _ in range(50):
            ind = toolbox.individual()
            if len(ind) <= node_cap:
                return ind
        # Extremely unlikely: shrink the oversized tree by truncation.
        return toolbox.individual()

    def fitness_of(ind) -> float:
        tokens = tuple(tree_to_tokens(ind, vocab))
        score = evaluator.score_of(tokens)
        if score is not None:
            return float(score.val_reward)
        return evaluator.best_reward  # neutral fitness for skipped proposals

    population = [fresh_individual() for _ in range(pop_size)]
    stall_generations = 0
    while evaluator.budget_used < evaluator.budget:
        budget_before = evaluator.budget_used
        # Propose the whole generation, then flush the batched scoring so
        # every individual's fitness is the real reward (the evaluator
        # scores in memory-bounded chunks; chunk=1 callers get eager
        # scores and the flush is a no-op).
        for ind in population:
            evaluator.propose(tuple(tree_to_tokens(ind, vocab)))
            if evaluator.budget_used >= evaluator.budget:
                break  # never overshoot the budget by more than one
        evaluator.flush()
        for ind in population:
            ind.fitness.values = (fitness_of(ind),)
        # A generation that cannot produce any new evaluation is stalled:
        # the search has converged on the already-evaluated classes.
        if evaluator.budget_used == budget_before:
            stall_generations += 1
            if stall_generations >= 3:
                break
        else:
            stall_generations = 0
        if evaluator.budget_used >= evaluator.budget:
            break

        offspring = [toolbox.clone(ind) for ind in population]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if py_random.random() < cxpb:
                toolbox.mate(child1, child2)
            if py_random.random() < mutpb:
                toolbox.mutate(child1)
        # Repair oversize children by regeneration (node cap mirrors the
        # policy's max formula length).
        for index, ind in enumerate(offspring):
            if len(ind) > node_cap:
                offspring[index] = fresh_individual()
        population = toolbox.select(offspring + population, k=pop_size)

    stalled = evaluator.budget_used < evaluator.budget
    return evaluator.finish(
        backend="gp",
        seed=seed,
        termination_reason=(
            "proposal_stagnation" if stalled else "budget_exhausted"
        ),
        stagnation_reason=(
            "three_generations_without_new_semantic_class" if stalled else None
        ),
        diagnostics={"stall_generations": stall_generations},
    )
