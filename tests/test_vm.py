from __future__ import annotations

import pytest
import torch

from ashare_model.ops import OPS_CONFIG
from ashare_model.vm import StackVM, cross_sectional_zscore, formula_decode
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vm_matches_cpu_on_cuda():
    # The GPU path only accelerates the VM operators; results must agree
    # with CPU execution within float32 tolerance.
    vm = StackVM(FORMULA_VOCAB)
    torch.manual_seed(3)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 3, 16)
    op_id = {name: FORMULA_VOCAB.operator_offset + i for i, (name, _, _) in enumerate(OPS_CONFIG)}
    formulas = [
        [1],
        [1, op_id["MA20"]],
        [1, op_id["TS_RANK20"]],
        [1, 2, op_id["CORR20"]],
        [1, op_id["MA20"], 1, op_id["STD20"], op_id["SUB"]],
        [1, op_id["DELAY1"], 2, op_id["DECAY"], op_id["ADD"]],
    ]
    for tokens in formulas:
        cpu_out = vm.execute(tokens, feature)
        assert cpu_out is not None
        gpu_out = vm.execute(tokens, feature.cuda())
        assert gpu_out is not None
        assert torch.allclose(cpu_out, gpu_out.cpu(), atol=1e-4, rtol=1e-4)


# --- terminal cross-sectional standardization ---------------------------------


def test_cross_sectional_zscore_standardizes_per_date():
    signal = torch.tensor(
        [
            [1.0, 10.0, 2.0],
            [3.0, 10.0, 6.0],
            [5.0, 10.0, 4.0],
        ]
    )
    out = cross_sectional_zscore(signal)
    assert out.shape == signal.shape
    # Non-degenerate columns are mean-zero / unit-std per date.
    assert torch.allclose(out.mean(dim=0)[[0, 2]], torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        out.std(dim=0, unbiased=False)[[0, 2]], torch.ones(2), atol=1e-5
    )
    # A degenerate (constant) column maps to the neutral 0, not ±inf.
    assert (out[:, 1] == 0.0).all()


def test_cross_sectional_zscore_is_scale_invariant_and_order_preserving():
    rng = torch.Generator().manual_seed(11)
    signal = torch.randn(6, 5, generator=rng)
    scaled = signal * 37.0 + 4.0  # affine drift must cancel out entirely
    out = cross_sectional_zscore(signal)
    out_scaled = cross_sectional_zscore(scaled)
    assert torch.allclose(out, out_scaled, atol=1e-5, rtol=1e-5)
    # Monotone per column: ranks are preserved, so rank IC and top-n
    # selection are unchanged by the terminal standardization.
    assert torch.equal(out.argsort(dim=0), signal.argsort(dim=0))


def test_vm_output_is_cross_sectionally_standardized():
    vm = StackVM(FORMULA_VOCAB)
    torch.manual_seed(5)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 4, 8)
    out = vm.execute([1], feature)
    assert out is not None
    assert torch.allclose(out.mean(dim=0), torch.zeros(8), atol=1e-6)
    assert torch.allclose(out.std(dim=0, unbiased=False), torch.ones(8), atol=1e-5)
    # The raw feature itself is not standardized: the VM output is the
    # z-scored form of it, column by column.
    assert not torch.allclose(out, feature[0], atol=1e-6)
    assert torch.equal(out.argsort(dim=0), feature[0].argsort(dim=0))


def test_vm_standardizes_stacked_arithmetic_scale():
    # MUL-chained formulas drift the raw scale; the terminal standardization
    # must keep the output scale stable regardless of the arithmetic path.
    vm = StackVM(FORMULA_VOCAB)
    torch.manual_seed(7)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 3, 10).abs() + 1.0
    mul = FORMULA_VOCAB.operator_offset + 2
    chained = vm.execute([1, 2, mul, 2, mul, 2, mul], feature)
    assert chained is not None
    assert chained.std(dim=0).max().item() < 2.0  # scale bounded, no explosion
    assert torch.isfinite(chained).all()
