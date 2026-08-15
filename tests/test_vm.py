from __future__ import annotations

import torch

from ashare_model.vm import StackVM, formula_decode
from ashare_model.vocab import FORMULA_VOCAB


def test_vm_executes_binary_unary_and_ternary():
    vm = StackVM(FORMULA_VOCAB)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 2, 4)
    add = FORMULA_VOCAB.operator_offset
    neg = add + 4  # NEG follows ADD/SUB/MUL/DIV in OPS_CONFIG
    gate = add + 7  # GATE follows NEG/ABS/SIGN
    assert vm.execute([1, 2, add], feature).shape == (2, 4)
    assert vm.execute([1, neg], feature).shape == (2, 4)
    assert vm.execute([1, 2, 3, gate], feature).shape == (2, 4)


def test_vm_returns_none_for_illegal_tokens():
    vm = StackVM(FORMULA_VOCAB)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 2, 4)
    assert vm.execute([1, 2], feature) is None
    assert vm.execute([-1], feature) is None
    assert vm.execute([FORMULA_VOCAB.size + 10], feature) is None


def test_formula_decode_variants_and_errors():
    add = FORMULA_VOCAB.operator_offset
    assert formula_decode([1, 2, add], FORMULA_VOCAB).startswith("(")
    assert formula_decode([1], FORMULA_VOCAB) == "RET_1"
    assert formula_decode([], FORMULA_VOCAB) == "INVALID"
    assert formula_decode([1, add], FORMULA_VOCAB) == "INVALID"
    assert formula_decode([FORMULA_VOCAB.size + 5], FORMULA_VOCAB) == "INVALID"


def test_vm_clamps_nonfinite_results_to_neutral_zero():
    vm = StackVM(FORMULA_VOCAB)
    feature = torch.zeros(FORMULA_VOCAB.feature_count, 2, 4)
    feature[0] = 1e30
    feature[1] = 1e30
    mul = FORMULA_VOCAB.operator_offset + 2  # MUL is the third operator
    out = vm.execute([1, 2, mul], feature)
    assert out is not None
    assert torch.isfinite(out).all()
    assert (out == 0).all()
