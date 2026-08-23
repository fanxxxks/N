from __future__ import annotations

import pytest
import torch

from ashare_model.ops import OPS_CONFIG
from ashare_model.vm import StackVM, cross_sectional_zscore, formula_decode
from ashare_model.vocab import FORMULA_VOCAB


def _vm(feature: torch.Tensor) -> StackVM:
    """A VM whose mask marks every cell eligible (the test then overrides
    ``vm.universe_mask`` when it needs a real PIT mask)."""
    return StackVM(
        FORMULA_VOCAB,
        universe_mask=torch.ones(feature.shape[1:], dtype=torch.bool),
    )


def test_vm_executes_binary_unary_and_ternary():
    feature = torch.randn(FORMULA_VOCAB.feature_count, 2, 4)
    vm = _vm(feature)
    add = FORMULA_VOCAB.operator_offset
    neg = add + 4  # NEG follows ADD/SUB/MUL/DIV in OPS_CONFIG
    gate = add + 7  # GATE follows NEG/ABS/SIGN
    assert vm.execute([1, 2, add], feature).shape == (2, 4)
    assert vm.execute([1, neg], feature).shape == (2, 4)
    assert vm.execute([1, 2, 3, gate], feature).shape == (2, 4)


def test_vm_returns_none_for_illegal_tokens():
    feature = torch.randn(FORMULA_VOCAB.feature_count, 2, 4)
    vm = _vm(feature)
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
    feature = torch.zeros(FORMULA_VOCAB.feature_count, 2, 4)
    vm = _vm(feature)
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
    torch.manual_seed(3)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 3, 16)
    vm = _vm(feature)
    op_id = {name: FORMULA_VOCAB.operator_offset + i for i, (name, _, _) in enumerate(OPS_CONFIG)}
    formulas = [
        [1],
        [1, op_id["MA20"]],
        [1, op_id["TS_RANK20"]],
        [1, 2, op_id["CORR20"]],
        [1, op_id["MA20"], 1, op_id["STD20"], op_id["SUB"]],
        [1, op_id["DELAY1"], 2, op_id["DECAY"], op_id["ADD"]],
        # Cross-sectional operators and enumerated windows.
        [1, op_id["CS_RANK"]],
        [1, op_id["CS_ZSCORE"]],
        [1, op_id["CS_DEMEAN"]],
        [1, 2, op_id["CS_RANK"], op_id["ADD"]],
        [1, op_id["MA5"], 1, op_id["STD60"], op_id["SUB"]],
        [1, 2, op_id["CORR10"]],
        [1, op_id["DOWNVOL60"], 1, op_id["DELTA20"], op_id["ADD"]],
    ]
    for tokens in formulas:
        cpu_out = vm.execute(tokens, feature)
        assert cpu_out is not None
        # The mask must travel with the feature tensor: the terminal
        # cross-sectional z-score reads both on the same device, and a
        # CPU mask against a CUDA feature raises a device mismatch.
        vm.universe_mask = vm.universe_mask.cuda()
        gpu_out = vm.execute(tokens, feature.cuda())
        vm.universe_mask = vm.universe_mask.cpu()
        assert gpu_out is not None
        assert torch.allclose(cpu_out, gpu_out.cpu(), atol=1e-4, rtol=1e-4)


def test_vm_cs_neutralize_uses_industry_codes():
    feature = torch.zeros(FORMULA_VOCAB.feature_count, 4, 3)
    vm = _vm(feature)
    op_id = {
        name: FORMULA_VOCAB.operator_offset + i
        for i, (name, _, _) in enumerate(OPS_CONFIG)
    }
    # Two industries (rows 0-1, rows 2-3) at distinct levels: the raw
    # neutralized signal is industry-demeaned before the terminal z-score.
    feature[0, :, :] = torch.tensor(
        [[10.0, 2.0, 4.0], [20.0, 4.0, 8.0], [100.0, 40.0, 50.0], [140.0, 60.0, 70.0]]
    )
    group = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )
    vm.industry_codes = group
    out = vm.execute([1, op_id["CS_NEUTRALIZE"]], feature)
    assert out is not None
    # Within each industry the demeaned rows are exact opposites, so the
    # standardized columns of each industry pair cancel out.
    assert torch.allclose(out[0] + out[1], torch.zeros(3), atol=1e-6)
    assert torch.allclose(out[2] + out[3], torch.zeros(3), atol=1e-6)
    # Without groups the operator is the full-market demean: industry 1's
    # level dominates the market mean, so the standardized outputs differ.
    vm.industry_codes = None
    out_demean = vm.execute([1, op_id["CS_NEUTRALIZE"]], feature)
    assert out_demean is not None
    assert not torch.allclose(out, out_demean, atol=1e-3)


def test_vm_new_operators_decode_and_execute():
    torch.manual_seed(41)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 3, 12)
    vm = _vm(feature)
    op_id = {
        name: FORMULA_VOCAB.operator_offset + i
        for i, (name, _, _) in enumerate(OPS_CONFIG)
    }
    for name in (
        "CS_RANK", "CS_ZSCORE", "CS_DEMEAN", "CS_NEUTRALIZE",
        "MA5", "MA10", "MA60", "STD5", "STD10", "STD60",
        "TS_RANK5", "TS_RANK10", "TS_RANK60",
        "CORR5", "CORR10", "CORR60",
        "DOWNVOL5", "DOWNVOL10", "DOWNVOL60", "DELTA10", "DELTA20",
    ):
        tokens = [1, 2, op_id[name]] if name.startswith("CORR") else [1, op_id[name]]
        out = vm.execute(tokens, feature)
        assert out is not None, name
        assert torch.isfinite(out).all(), name
        assert name in formula_decode(tokens, FORMULA_VOCAB), name


# --- terminal cross-sectional standardization ---------------------------------


def test_cross_sectional_zscore_standardizes_per_date():
    signal = torch.tensor(
        [
            [1.0, 10.0, 2.0],
            [3.0, 10.0, 6.0],
            [5.0, 10.0, 4.0],
        ]
    )
    out = cross_sectional_zscore(signal, torch.ones_like(signal, dtype=torch.bool))
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
    eligible = torch.ones_like(signal, dtype=torch.bool)
    out = cross_sectional_zscore(signal, eligible)
    out_scaled = cross_sectional_zscore(scaled, eligible)
    assert torch.allclose(out, out_scaled, atol=1e-5, rtol=1e-5)
    # Monotone per column: ranks are preserved, so rank IC and top-n
    # selection are unchanged by the terminal standardization.
    assert torch.equal(out.argsort(dim=0), signal.argsort(dim=0))


def test_vm_output_is_cross_sectionally_standardized():
    torch.manual_seed(5)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 4, 8)
    vm = _vm(feature)
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
    torch.manual_seed(7)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 3, 10).abs() + 1.0
    vm = _vm(feature)
    mul = FORMULA_VOCAB.operator_offset + 2
    chained = vm.execute([1, 2, mul, 2, mul, 2, mul], feature)
    assert chained is not None
    assert chained.std(dim=0).max().item() < 2.0  # scale bounded, no explosion
    assert torch.isfinite(chained).all()


# --- PIT universe mask on the VM ----------------------------------------------


def _vm_op_ids():
    return {
        name: FORMULA_VOCAB.operator_offset + i
        for i, (name, _, _) in enumerate(OPS_CONFIG)
    }


def test_vm_terminal_zscore_excludes_ineligible_stocks():
    torch.manual_seed(9)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 4, 6)
    vm = _vm(feature)
    eligible = torch.ones(4, 6, dtype=torch.bool)
    eligible[3] = False
    vm.universe_mask = eligible
    out = vm.execute([1], feature)
    assert out is not None
    # Ineligible cells are non-participating (NaN), not zeros in the sort.
    assert torch.isnan(out[3]).all()
    # The eligible rows are z-scored over the eligible reference set only.
    ref = cross_sectional_zscore(
        feature[0][:3], torch.ones(3, 6, dtype=torch.bool)
    )
    assert torch.allclose(out[:3], ref, atol=1e-5, rtol=1e-5)


def test_vm_extreme_ineligible_values_do_not_move_eligible_signals():
    base = torch.randn(FORMULA_VOCAB.feature_count, 4, 2)
    vm = _vm(base)
    eligible = torch.tensor(
        [[True, True], [True, True], [True, True], [False, False]]
    )
    vm.universe_mask = eligible
    extreme = base.clone()
    extreme[0, 3] = 1e9  # future member with extreme values
    out_base = vm.execute([1], base)
    out_extreme = vm.execute([1], extreme)
    assert out_base is not None and out_extreme is not None
    # The eligible rows are identical: the extreme ineligible row cannot
    # shift their z-scores (it also cannot enter as a zero).
    assert torch.allclose(out_base[:3], out_extreme[:3], atol=1e-5, rtol=1e-5)
    assert torch.isnan(out_extreme[3]).all()


def test_vm_cs_operators_honor_universe_mask():
    op_id = _vm_op_ids()
    feature = torch.zeros(FORMULA_VOCAB.feature_count, 4, 3)
    vm = _vm(feature)
    feature[0] = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [1e6, 1e6, 1e6]]
    )
    eligible = torch.tensor(
        [[True, True, True], [True, True, True], [True, True, True], [False, False, False]]
    )
    vm.universe_mask = eligible
    for name in ("CS_RANK", "CS_ZSCORE", "CS_DEMEAN"):
        out = vm.execute([1, op_id[name]], feature)
        assert out is not None, name
        # The ineligible row is non-participating in the final signal...
        assert torch.isnan(out[3]).all(), name
        # ...and the eligible rows match the eligible-only execution.
        vm.universe_mask = torch.ones(3, 3, dtype=torch.bool)
        eligible_only = vm.execute(
            [1, op_id[name]], feature[:, :3]
        )
        vm.universe_mask = eligible
        assert eligible_only is not None
        assert torch.allclose(out[:3], eligible_only, atol=1e-5, rtol=1e-5), name


def test_vm_cs_neutralize_honors_mask_through_industry_codes():
    feature = torch.zeros(FORMULA_VOCAB.feature_count, 4, 2)
    vm = _vm(feature)
    op_id = _vm_op_ids()
    feature[0] = torch.tensor(
        [[10.0, 2.0], [20.0, 4.0], [1000.0, 2000.0], [7.0, 5.0]]
    )
    vm.industry_codes = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [float("nan"), float("nan")]]
    )
    vm.universe_mask = torch.tensor(
        [[True, True], [True, True], [False, False], [True, True]]
    )
    out = vm.execute([1, op_id["CS_NEUTRALIZE"]], feature)
    assert out is not None
    # The terminal signal is z-scored; the industry pair is neutralized
    # against the eligible members only, so the ineligible extreme could
    # not move the industry mean.
    assert not torch.isnan(out[:2]).any()
    assert torch.isnan(out[2]).all()
    # Unmapped row stays raw inside the operator and is standardized as the
    # sole valid member of no group at the terminal step.
    assert torch.isfinite(out[3]).all()


def test_vm_day_without_eligible_stocks_is_stable():
    feature = torch.randn(FORMULA_VOCAB.feature_count, 3, 4)
    vm = _vm(feature)
    eligible = torch.ones(3, 4, dtype=torch.bool)
    eligible[:, 2] = False
    vm.universe_mask = eligible
    out = vm.execute([1], feature)
    assert out is not None
    # The empty day collapses to the stable neutral 0 (no NaN spread, no
    # extreme values) while the eligible days keep their z-scores.
    assert (out[:, 2] == 0.0).all()
    assert torch.isfinite(out[:, :2]).all() and torch.isfinite(out[:, 3]).all()


def test_vm_single_eligible_stock_is_neutral_but_valid():
    feature = torch.zeros(FORMULA_VOCAB.feature_count, 3, 2)
    vm = _vm(feature)
    feature[0] = torch.tensor([[5.0, 7.0], [1e9, 1e9], [1e9, 1e9]])
    eligible = torch.tensor([[True, True], [False, False], [False, False]])
    vm.universe_mask = eligible
    out = vm.execute([1], feature)
    assert out is not None
    # A single eligible member has zero dispersion: neutral z-score, but it
    # stays the only valid cell of its days.
    assert torch.allclose(out[0], torch.zeros(2), atol=1e-6)
    assert torch.isnan(out[1]).all() and torch.isnan(out[2]).all()


def test_vm_mask_shape_mismatch_returns_none():
    feature = torch.randn(FORMULA_VOCAB.feature_count, 3, 4)
    vm = _vm(feature)
    vm.universe_mask = torch.ones(2, 4, dtype=torch.bool)
    # A misaligned mask makes the execution invalid (the VM's None contract
    # for unusable executions); the underlying operators raise a clear
    # ValueError for direct callers.
    assert vm.execute([1], feature) is None
    with pytest.raises(ValueError, match="universe_mask shape"):
        cross_sectional_zscore(feature[0], vm.universe_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vm_masked_execution_matches_cpu_on_cuda():
    torch.manual_seed(7)
    feature = torch.randn(FORMULA_VOCAB.feature_count, 4, 16)
    vm = _vm(feature)
    eligible = torch.rand(4, 16) > 0.3
    industry = torch.randint(0, 2, (4, 16)).float()
    vm.universe_mask = eligible
    vm.industry_codes = industry
    op_id = _vm_op_ids()
    formulas = [
        [1],
        [1, op_id["CS_RANK"]],
        [1, op_id["CS_ZSCORE"]],
        [1, op_id["CS_DEMEAN"]],
        [1, op_id["CS_NEUTRALIZE"]],
        [1, 2, op_id["CS_RANK"], op_id["ADD"]],
    ]
    for tokens in formulas:
        cpu_out = vm.execute(tokens, feature)
        assert cpu_out is not None
        vm.universe_mask = eligible.cuda()
        vm.industry_codes = industry.cuda()
        gpu_out = vm.execute(tokens, feature.cuda())
        vm.universe_mask = eligible
        vm.industry_codes = industry
        assert gpu_out is not None
        # The NaN (non-participating) pattern and the finite values agree
        # within the established float32 tolerance contract.
        assert torch.equal(torch.isnan(cpu_out), torch.isnan(gpu_out.cpu()))
        finite = ~torch.isnan(cpu_out)
        assert torch.allclose(
            cpu_out[finite], gpu_out.cpu()[finite], atol=1e-4, rtol=1e-4
        )
