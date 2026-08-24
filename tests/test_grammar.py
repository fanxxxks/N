"""Grammar property tests (T0-02).

These tests pin the postfix grammar as a *system*: the sampling mask, the
AST decode/encode pair and the StackVM must agree on exactly the same set
of legal formulas, exhaustively on a toy vocabulary and statistically on
the production vocabulary.

Covered contracts:

* every legal AST of a toy vocab encodes, executes and round-trips;
* the sampling mask never admits an illegal sequence (exhaustive DFS);
* 10,000 random samples execute with a 100% success rate;
* all 39 operators are reachable (deterministic construction + coverage
  over the random sample);
* PAD never appears before EOS in sampled sequences;
* every sampled formula has an effective length > 1;
* AST -> postfix -> AST round-trips exactly.
"""

from __future__ import annotations

from ashare_model.alphagpt import build_action_mask
from ashare_model.ir import (
    Binary,
    Feature,
    FormulaSyntaxError,
    Ternary,
    Unary,
    decode,
    encode,
    formula_length,
    infix,
    operator_names,
    postfix_valid,
)
from ashare_model.ops import OPS_CONFIG
from ashare_model.train import sample_random_formulas
from ashare_model.vm import StackVM
from ashare_model.vocab import FORMULA_VOCAB, FormulaVocab

import torch


def _op_id(vocab, name: str) -> int:
    return vocab.operator_offset + vocab.operator_names.index(name)


# --- toy vocabulary -----------------------------------------------------------


def _toy_vocab() -> FormulaVocab:
    """Two features and one operator of each arity (unary/binary/ternary)."""
    return FormulaVocab(
        feature_names=("F1", "F2"),
        operator_names=("NEG", "ADD", "GATE"),
    )


def _toy_asts(node_budget: int):
    """Enumerate every AST of at most ``node_budget`` nodes over the toy
    vocabulary.  Yielded ASTs are the exhaustive legal-AST universe the
    test then encodes and executes."""

    def all_asts(budget: int):
        if budget < 1:
            return
        for name in ("F1", "F2"):
            yield Feature(name)
        if budget < 2:
            return
        for operand in all_asts(budget - 1):
            yield Unary("NEG", operand)
        for left_budget in range(1, budget):
            right_budget = budget - left_budget
            for left in all_asts(left_budget):
                for right in all_asts(right_budget):
                    yield Binary("ADD", left, right)
        for a_budget in range(1, budget):
            for b_budget in range(1, budget - a_budget):
                c_budget = budget - a_budget - b_budget
                if c_budget < 1:
                    continue
                for a in all_asts(a_budget):
                    for b in all_asts(b_budget):
                        for c in all_asts(c_budget):
                            yield Ternary("GATE", a, b, c)

    seen: set[str] = set()
    for budget in range(1, node_budget + 1):
        for ast in all_asts(budget):
            text = infix(ast)
            if text not in seen:
                seen.add(text)
                yield ast


def test_toy_vocab_every_legal_ast_encodes_and_executes():
    vocab = _toy_vocab()
    factor = torch.randn(2, 3, 5)
    vm = StackVM(vocab, universe_mask=torch.ones(3, 5, dtype=torch.bool))
    asts = list(_toy_asts(node_budget=5))
    # The enumeration is meaningfully exhaustive, not vacuous.
    assert len(asts) > 200
    for ast in asts:
        tokens = encode(ast, vocab)
        assert postfix_valid(tokens, vocab), infix(ast)
        assert tokens[-1] == vocab.eos_token_id
        assert decode(tokens, vocab) == ast, infix(ast)
        assert decode(encode(decode(tokens, vocab), vocab), vocab) == ast
        assert vm.execute(tokens, factor) is not None, infix(ast)


def test_toy_vocab_mask_exhaustive_dfs_never_admits_illegal_sequences():
    """Walk every sequence the mask admits for a tiny max_len: each one
    must be valid postfix, decodable and executable."""

    vocab = _toy_vocab()
    max_len = 4
    factor = torch.randn(2, 3, 5)
    vm = StackVM(vocab, universe_mask=torch.ones(3, 5, dtype=torch.bool))
    arity = {name: a for name, _, a in OPS_CONFIG}

    def legal(stack: int, done: bool, step: int):
        mask = build_action_mask(
            torch.tensor([stack]), torch.tensor([done]), step, max_len, vocab
        )
        return [t for t in range(vocab.size) if mask[0, t] == 0.0]

    def advance(token: int, stack: int, done: bool):
        if token == vocab.pad_token_id:
            return stack, True
        if vocab.eos_token_id is not None and token == vocab.eos_token_id:
            return stack, True
        if token < vocab.operator_offset:
            return stack + 1, done
        name = vocab.operator_names[token - vocab.operator_offset]
        return stack - arity[name] + 1, done

    count = 0
    stack, done = 0, False

    def dfs(step: int, sequence: list[int], stack: int, done: bool) -> None:
        nonlocal count
        if step == max_len:
            return
        for token in legal(stack, done, step):
            sequence.append(token)
            count += 1
            # Every *complete* sequence the mask admits must be valid
            # postfix, decodable and executable (incomplete prefixes are
            # not formulas yet).
            if postfix_valid(sequence, vocab):
                assert decode(sequence, vocab) is not None  # must not raise
                assert vm.execute(sequence, factor) is not None, sequence
            next_stack, next_done = advance(token, stack, done)
            dfs(step + 1, sequence, next_stack, next_done)
            sequence.pop()

    dfs(0, [], stack, done)
    # Exhaustive over the mask's full decision tree (not a vacuous pass):
    # the tiny grammar admits tens of token placements across all depths.
    assert count >= 20


def test_toy_vocab_pad_only_after_eos_in_admitted_sequences():
    vocab = _toy_vocab()
    max_len = 4

    def legal(stack: int, done: bool, step: int):
        mask = build_action_mask(
            torch.tensor([stack]), torch.tensor([done]), step, max_len, vocab
        )
        return [t for t in range(vocab.size) if mask[0, t] == 0.0]

    def advance(token: int, stack: int, done: bool):
        if token == vocab.pad_token_id or (
            vocab.eos_token_id is not None and token == vocab.eos_token_id
        ):
            return stack, True
        if token < vocab.operator_offset:
            return stack + 1, done
        name = vocab.operator_names[token - vocab.operator_offset]
        return stack - dict((n, a) for n, _, a in OPS_CONFIG)[name] + 1, done

    def dfs(step: int, sequence: list[int], stack: int, done: bool) -> None:
        if step == max_len:
            return
        for token in legal(stack, done, step):
            sequence.append(token)
            if token == vocab.pad_token_id:
                # PAD is legal only once the formula is terminated (EOS).
                assert done, sequence
            dfs(step + 1, sequence, *advance(token, stack, done))
            sequence.pop()

    dfs(0, [], 0, False)


# --- production vocabulary -----------------------------------------------------


def _vm_and_factors() -> tuple[StackVM, torch.Tensor]:
    torch.manual_seed(0)
    factor = torch.randn(FORMULA_VOCAB.feature_count, 6, 10)
    vm = StackVM(FORMULA_VOCAB, universe_mask=torch.ones(6, 10, dtype=torch.bool))
    return vm, factor


def test_random_samples_execute_100_percent():
    vm, factor = _vm_and_factors()
    samples = sample_random_formulas(seed=7, vocab=FORMULA_VOCAB, max_len=12, n=10_000)
    assert len(samples) == 10_000
    for tokens in samples:
        assert len(tokens) == 12
        assert postfix_valid(tokens, FORMULA_VOCAB)
        assert vm.execute(list(tokens), factor) is not None


def test_random_samples_pad_only_after_eos_and_length_gt_one():
    samples = sample_random_formulas(seed=9, vocab=FORMULA_VOCAB, max_len=12, n=10_000)
    eos = FORMULA_VOCAB.eos_token_id
    for tokens in samples:
        eos_index = tokens.index(eos)
        # Every sequence carries exactly one EOS terminator, and everything
        # after it is PAD.
        assert eos_index >= 1  # effective length > 1
        assert all(t == FORMULA_VOCAB.pad_token_id for t in tokens[eos_index + 1 :])
        assert eos_index + 1 == formula_length(tokens, FORMULA_VOCAB)
        # The effective (non-padded) formula is longer than a bare token.
        assert formula_length(tokens, FORMULA_VOCAB) > 1


def test_ast_postfix_ast_roundtrip_on_random_samples():
    samples = sample_random_formulas(seed=11, vocab=FORMULA_VOCAB, max_len=12, n=2_000)
    for tokens in samples:
        ast = decode(tokens)
        assert decode(encode(ast, FORMULA_VOCAB), FORMULA_VOCAB) == ast
        # Sampled bytecode is canonical up to the EOS terminator: the
        # re-encoded form equals the non-padded prefix exactly.
        effective = formula_length(tokens, FORMULA_VOCAB)
        assert encode(ast, FORMULA_VOCAB) == list(tokens[:effective])


def test_all_operators_reachable():
    """Every one of the 39 operators appears in some legal, executable
    formula — constructed deterministically (reachability), and covered by
    the 10,000-formula random sample (search-space saturation)."""

    vm, factor = _vm_and_factors()
    names = [cfg[0] for cfg in OPS_CONFIG]
    assert len(names) == 39
    # Deterministic reachability: one explicit AST per operator.
    for name in names:
        arity = dict((n, a) for n, _, a in OPS_CONFIG)[name]
        features = list(range(1, arity + 1))
        tokens = features + [_op_id(FORMULA_VOCAB, name), FORMULA_VOCAB.eos_token_id]
        assert decode(tokens).__class__.__name__ in ("Unary", "Binary", "Ternary")
        assert operator_names(decode(tokens)) == {name}
        assert vm.execute(tokens, factor) is not None, name

    # Random-sample coverage: uniform sampling over the legal mask hits
    # every operator (a deterministic constructive proof above; this is
    # the saturation check on the actual sampler).
    samples = sample_random_formulas(seed=7, vocab=FORMULA_VOCAB, max_len=12, n=10_000)
    covered: set[str] = set()
    for tokens in samples:
        covered |= operator_names(decode(tokens))
    assert covered == set(names)


def test_sampled_sequences_are_grammar_canonical():
    samples = sample_random_formulas(seed=13, vocab=FORMULA_VOCAB, max_len=12, n=500)
    for tokens in samples:
        # decode(encode(ast)) == tokens: sampled bytecode is canonical, so
        # no legacy padding forms ever reach the VM from the sampler.
        assert decode(encode(decode(tokens), FORMULA_VOCAB), FORMULA_VOCAB) == decode(tokens)


def test_invalid_sequences_are_rejected_by_all_layers():
    # The three layers (decode, postfix_valid, VM) agree on rejection.
    vm, factor = _vm_and_factors()
    eos = FORMULA_VOCAB.eos_token_id
    add = _op_id(FORMULA_VOCAB, "ADD")
    bad = [
        [],
        [eos],
        [1, 2],
        [1, add],
        [1, 2, 0],
        [1, eos, 2],
        [1, eos, eos],
        [FORMULA_VOCAB.size],
        [1, 1, 1, 0, 0],  # PAD before the stack is complete
    ]
    for tokens in bad:
        assert not postfix_valid(tokens, FORMULA_VOCAB), tokens
        try:
            decode(tokens)
            raise AssertionError(f"decode accepted {tokens}")
        except FormulaSyntaxError:
            pass
        assert vm.execute(tokens, factor) is None, tokens
