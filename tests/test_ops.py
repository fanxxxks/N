from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ashare_model.ops import (
    OPS_CONFIG,
    _cs_demean,
    _cs_neutralize,
    _cs_rank,
    _op_jump,
    _ts_corr,
    _ts_delay,
    _ts_delta,
    _ts_downvol,
    _ts_ma,
    _ts_rank,
    _ts_std,
    _ts_window,
    cross_sectional_zscore,
)


def test_all_operators_preserve_batch_and_time_shape():
    x = torch.randn(2, 6)
    y = torch.randn(2, 6)
    eligible = torch.ones_like(x, dtype=torch.bool)
    for name, fn, arity in OPS_CONFIG:
        if name == "CS_NEUTRALIZE":
            # The VM dispatches this operator by name (industry groups live
            # on the VM); the registry slot is a bare placeholder.
            out = _cs_neutralize(x, eligible=eligible)
        elif name.startswith("CS_"):
            out = fn(x, eligible)
        elif arity == 1:
            out = fn(x)
        elif arity == 2:
            out = fn(x, y)
        else:
            out = fn(x, x, x)
        assert out.shape == x.shape, name
        assert torch.isfinite(out).all(), name


def test_all_operators_are_causal():
    """No-leakage contract over the whole operator registry.

    Perturbing any input column at or after ``cut`` must not change any
    earlier output column.  This is the regression test for the JUMP
    look-ahead (it standardized by the full-timeline mean/std) and a
    permanent guard for every operator added to OPS_CONFIG afterwards.

    The past region carries a planted outlier so saturating operators
    (JUMP's relu(z - 3) maps most values to exactly 0) have a nonzero past
    output to move; the perturbation is a deterministic large constant so
    any statistic that reads the future shifts visibly.
    """
    torch.manual_seed(41)
    b, t = 6, 40
    cut = t // 2
    x = torch.randn(b, t)
    x[:, cut - 1] = 50.0  # fires JUMP at a past column (z = sqrt(n-1) > 3)
    y = torch.randn(b, t)
    cond = torch.randn(b, t)
    group = torch.randint(0, 3, (b, t)).float()
    eligible = torch.ones(b, t, dtype=torch.bool)

    def perturbed(src: torch.Tensor) -> torch.Tensor:
        dst = src.clone()
        if dst.dtype == torch.bool:
            dst[:, cut:] = False
        else:
            dst[:, cut:] = 1000.0
        return dst

    for name, fn, arity in OPS_CONFIG:
        if name == "CS_NEUTRALIZE":
            # The registry slot is a bare placeholder (the VM dispatches
            # this operator by name); call the real implementation.
            fn = _cs_neutralize
            args, kwargs = (x, group), {"eligible": eligible}
        elif name.startswith("CS_"):
            args, kwargs = (x,), {"eligible": eligible}
        elif arity == 1:
            args, kwargs = (x,), {}
        elif arity == 2:
            args, kwargs = (x, y), {}
        else:  # GATE
            args, kwargs = (cond, x, y), {}
        base = fn(*args, **kwargs)
        out = fn(
            *(perturbed(a) for a in args),
            **{k: perturbed(v) for k, v in kwargs.items()},
        )
        assert torch.allclose(
            base[:, :cut], out[:, :cut], atol=1e-6, equal_nan=True
        ), name


def test_ts_window_expanding_for_short_series():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = _ts_window(x, 10, lambda w, dim: w.mean(dim=dim))
    assert out.shape == x.shape
    assert out[0, 0].item() == 1.0
    assert out[0, 1].item() == 1.5
    assert out[0, 2].item() == 2.0


def test_ts_rank_bounds():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = _ts_rank(x, 20)
    assert out.shape == x.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
    assert out[0, -1].item() == 1.0


def _ts_rank_reference(x: torch.Tensor, d: int) -> torch.Tensor:
    """The previous per-column implementation, kept as the test oracle."""
    if d <= 1:
        return torch.zeros_like(x)
    out = torch.zeros_like(x)
    for i in range(x.shape[1]):
        start = max(0, i - d + 1)
        window = x[:, start : i + 1]
        out[:, i] = (window <= x[:, i : i + 1]).float().mean(dim=1)
    return out


def test_ts_rank_matches_per_column_reference():
    torch.manual_seed(11)
    for b, t, d in ((1, 50, 20), (3, 7, 20), (2, 40, 5), (1, 25, 1), (4, 60, 20)):
        x = torch.randn(b, t)
        assert torch.allclose(
            _ts_rank(x, d), _ts_rank_reference(x, d), atol=1e-6
        )
        # Every column is a valid fraction of its expanding window.
        out = _ts_rank(x, d)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_div_zero_denominator_is_finite():
    div = dict((cfg[0], cfg[1]) for cfg in OPS_CONFIG)["DIV"]
    x = torch.tensor([[2.0, -2.0, 5.0]])
    y = torch.tensor([[0.0, 0.0, 0.0]])
    out = div(x, y)
    assert torch.isfinite(out).all()
    # Ordinary division is unchanged.
    assert div(torch.tensor([[-2.0]]), torch.tensor([[-3.0]])).item() > 0.0


def test_ma20_has_no_zero_padding_bias():
    x = torch.full((1, 25), 10.0)
    ma = _ts_ma(x, 20)
    assert torch.allclose(ma[0, 18:], torch.full((7,), 10.0))

    x2 = torch.arange(1, 26, dtype=torch.float).unsqueeze(0)
    ma2 = _ts_ma(x2, 20)
    # Position 18 (0-based) only has 19 real values: the expanding mean of
    # 1..19 is 10.0, not the zero-diluted 9.5 the old implementation gave.
    assert abs(ma2[0, 18].item() - 10.0) < 1e-5
    assert abs(ma2[0, 19].item() - 10.5) < 1e-5
    assert abs(ma2[0, 24].item() - 15.5) < 1e-5


def test_std20_of_constant_series_is_zero():
    x = torch.full((1, 25), 10.0)
    out = _ts_std(x, 20)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_delay1_extends_with_first_value():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    out = _ts_delay(x, 1)
    assert out[0, 0].item() == 1.0
    assert out[0, 1].item() == 1.0
    assert out[0, 2].item() == 2.0


def test_jump_detects_spike_against_trailing_window_only():
    # An isolated spike fires exactly at the spike: with a single outlier
    # in a window of n values the population-std z-score is sqrt(n-1), so
    # at position 40 (expanding window of 41 values) z = sqrt(40) > 3.
    # Before the spike the window is constant (z = 0); after it the mean is
    # dragged up so the neutral value has a negative z (relu keeps 0).
    x = torch.zeros(1, 80)
    x[0, 40] = 10.0
    out = _op_jump(x)
    assert torch.allclose(out[0, :40], torch.zeros(40), atol=1e-6)
    assert out[0, 40].item() == pytest.approx(math.sqrt(40.0) - 3.0, abs=1e-4)
    assert torch.allclose(out[0, 41:], torch.zeros(39), atol=1e-6)


def test_jump_baseline_is_limited_to_trailing_window():
    # A sustained level shift is a jump only on the day it happens: 60
    # sessions later the shifted level has filled the trailing baseline,
    # so the z-score of the (now constant) window is neutral again.  Under
    # the old full-timeline standardization the shift day itself scored
    # z = 1 against the forever-mixed distribution and never fired.
    x = torch.cat([torch.zeros(1, 60), torch.full((1, 60), 5.0)], dim=1)
    out = _op_jump(x)
    assert out[0, 60].item() > 0.0
    assert torch.allclose(out[0, -20:], torch.zeros(20), atol=1e-6)


def _op(name):
    return dict((cfg[0], cfg[1]) for cfg in OPS_CONFIG)[name]


def test_corr20_of_identical_series_is_one():
    x = torch.randn(2, 30)
    out = _op("CORR20")(x, x)
    assert torch.isfinite(out).all()
    # Position 0 has a single-sample window (correlation 0); after that the
    # identical series correlate at exactly 1.
    assert torch.allclose(out[:, 0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(out[:, 1:], torch.ones(2, 29), atol=1e-5)


def test_corr20_of_anticorrelated_series_is_minus_one():
    x = torch.randn(1, 30)
    out = _op("CORR20")(x, -x)
    assert torch.allclose(out[:, 1:], -torch.ones(1, 29), atol=1e-5)


def test_corr20_of_constant_series_is_zero():
    x = torch.full((1, 25), 3.0)
    y = torch.arange(25, dtype=torch.float).unsqueeze(0)
    out = _op("CORR20")(x, y)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_corr20_respects_trailing_window():
    # After a regime change the trailing correlation must reflect only the
    # most recent 20 values, not the whole history.
    x = torch.cat([torch.full((1, 10), 1.0), torch.full((1, 10), -1.0)], dim=1)
    out = _op("CORR20")(x, x)
    assert torch.allclose(out[:, -1], torch.ones(1), atol=1e-5)


def test_downvol20_only_penalizes_negative_values():
    x = torch.cat([torch.full((1, 10), 1.0), torch.full((1, 10), -2.0)], dim=1)
    out = _op("DOWNVOL20")(x)
    # Trailing 20-window: 10 zeros and 10 squares of 4 -> sqrt(40/20).
    assert abs(out[0, -1].item() - 2.0**0.5) < 1e-5
    # All-positive series has zero downside volatility.
    pos = torch.full((1, 25), 3.0)
    out_pos = _op("DOWNVOL20")(pos)
    assert torch.allclose(out_pos, torch.zeros_like(out_pos), atol=1e-6)


# --- cross-sectional operators ------------------------------------------------


def _numpy_avg_rank(col: np.ndarray) -> np.ndarray:
    """Average-rank oracle per column (ties share the mean of their ranks)."""
    n = len(col)
    order = np.argsort(col, kind="stable")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and col[order[j + 1]] == col[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks / n


def test_cs_rank_matches_average_rank_oracle():
    torch.manual_seed(17)
    for n, t in ((8, 5), (30, 4), (5, 9)):
        x = torch.randn(n, t)
        # Force tie groups: quantize every other column.
        x[:, 1::2] = (x[:, 1::2] * 2).round() / 2
        out = _cs_rank(x, torch.ones_like(x, dtype=torch.bool)).numpy()
        for day in range(t):
            expected = _numpy_avg_rank(x[:, day].numpy())
            assert np.allclose(out[:, day], expected, atol=1e-6)


def test_cs_rank_is_monotone_invariant_and_bounded():
    torch.manual_seed(19)
    x = torch.randn(12, 6)
    eligible = torch.ones_like(x, dtype=torch.bool)
    out = _cs_rank(x, eligible)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
    # Affine transforms do not change ranks.
    assert torch.allclose(_cs_rank(x * 3.0 - 2.0, eligible), out, atol=1e-6)
    # Rank preserves order: the largest value gets rank 1 (or the shared
    # top rank under ties).
    assert (out[x == x.max(dim=0, keepdim=True).values] >= out.max() - 1e-6).all()


def test_cs_operators_see_only_the_current_cross_section():
    # No-leakage contract: changing a future column must not change any
    # earlier column of any cross-sectional operator.
    torch.manual_seed(23)
    x = torch.randn(10, 8)
    eligible = torch.ones_like(x, dtype=torch.bool)
    base = {
        "rank": _cs_rank(x, eligible),
        "z": cross_sectional_zscore(x, eligible),
        "dm": _cs_demean(x, eligible),
    }
    changed = x.clone()
    changed[:, 5] = torch.randn(10) * 100.0
    for name, out in (
        ("rank", _cs_rank(changed, eligible)),
        ("z", cross_sectional_zscore(changed, eligible)),
        ("dm", _cs_demean(changed, eligible)),
    ):
        assert torch.allclose(out[:, :5], base[name][:, :5], atol=1e-6), name


def test_cs_zscore_and_demean_numerics():
    x = torch.tensor([[1.0, 4.0, 9.0], [2.0, 4.0, 6.0], [3.0, 4.0, 3.0]])
    eligible = torch.ones_like(x, dtype=torch.bool)
    z = cross_sectional_zscore(x, eligible)
    assert torch.allclose(z.mean(dim=0)[[0, 2]], torch.zeros(2), atol=1e-6)
    assert torch.allclose(z.std(dim=0, unbiased=False)[[0, 2]], torch.ones(2), atol=1e-5)
    # The constant column is degenerate: it maps to neutral 0.
    assert (z[:, 1] == 0.0).all()
    dm = _cs_demean(x, eligible)
    assert torch.allclose(dm.mean(dim=0), torch.zeros(3), atol=1e-6)
    assert (dm[:, 1] == 0.0).all()


def test_cs_neutralize_subtracts_group_means():
    x = torch.tensor(
        [
            [10.0, 2.0],
            [20.0, 4.0],
            [100.0, 40.0],
            [140.0, 60.0],
            [7.0, 5.0],
        ]
    )
    group = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [float("nan"), float("nan")],
        ]
    )
    out = _cs_neutralize(x, group, eligible=torch.ones_like(x, dtype=torch.bool))
    # Group 0 mean: day0 (10+20)/2=15 -> [-5, +5]; day1 (2+4)/2=3 -> [-1, +1].
    assert out[0, 0].item() == -5.0 and out[1, 0].item() == 5.0
    assert out[0, 1].item() == -1.0 and out[1, 1].item() == 1.0
    # Group 1 mean: day0 120 -> [-20, +20]; day1 50 -> [-10, +10].
    assert out[2, 0].item() == -20.0 and out[3, 0].item() == 20.0
    assert out[2, 1].item() == -10.0 and out[3, 1].item() == 10.0
    # Unmapped (NaN) rows belong to no group and stay untouched.
    assert torch.allclose(out[4], x[4])


def test_cs_neutralize_falls_back_to_demean_without_groups():
    x = torch.tensor([[1.0, 3.0], [3.0, 9.0], [5.0, 6.0]])
    eligible = torch.ones_like(x, dtype=torch.bool)
    assert torch.allclose(_cs_neutralize(x, eligible=eligible), _cs_demean(x, eligible))
    assert torch.allclose(
        _cs_neutralize(x, torch.full_like(x, float("nan")), eligible=eligible),
        _cs_demean(x, eligible),
    )


# --- enumerated windows -------------------------------------------------------


def test_enumerated_windows_match_their_parameterized_cores():
    torch.manual_seed(29)
    x = torch.randn(4, 80)
    y = torch.randn(4, 80)
    assert torch.allclose(_op("MA5")(x), _ts_ma(x, 5), atol=1e-6)
    assert torch.allclose(_op("MA10")(x), _ts_ma(x, 10), atol=1e-6)
    assert torch.allclose(_op("MA60")(x), _ts_ma(x, 60), atol=1e-6)
    assert torch.allclose(_op("STD5")(x), _ts_std(x, 5), atol=1e-6)
    assert torch.allclose(_op("STD10")(x), _ts_std(x, 10), atol=1e-6)
    assert torch.allclose(_op("STD60")(x), _ts_std(x, 60), atol=1e-6)
    assert torch.allclose(_op("TS_RANK5")(x), _ts_rank(x, 5), atol=1e-6)
    assert torch.allclose(_op("TS_RANK10")(x), _ts_rank(x, 10), atol=1e-6)
    assert torch.allclose(_op("TS_RANK60")(x), _ts_rank(x, 60), atol=1e-6)
    assert torch.allclose(_op("CORR5")(x, y), _ts_corr(x, y, 5), atol=1e-6)
    assert torch.allclose(_op("CORR10")(x, y), _ts_corr(x, y, 10), atol=1e-6)
    assert torch.allclose(_op("CORR60")(x, y), _ts_corr(x, y, 60), atol=1e-6)
    assert torch.allclose(_op("DOWNVOL5")(x), _ts_downvol(x, 5), atol=1e-6)
    assert torch.allclose(_op("DOWNVOL10")(x), _ts_downvol(x, 10), atol=1e-6)
    assert torch.allclose(_op("DOWNVOL60")(x), _ts_downvol(x, 60), atol=1e-6)
    assert torch.allclose(_op("DELTA10")(x), _ts_delta(x, 10), atol=1e-6)
    assert torch.allclose(_op("DELTA20")(x), _ts_delta(x, 20), atol=1e-6)


# --- PIT universe mask in the cross-sectional operators -----------------------


def _masked_x():
    x = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [1e6, 1e6],  # future member with extreme values
        ]
    )
    eligible = torch.tensor(
        [
            [True, True],
            [True, True],
            [True, True],
            [False, False],
        ]
    )
    return x, eligible


def test_cs_zscore_reference_is_eligible_only():
    x, eligible = _masked_x()
    out = cross_sectional_zscore(x, eligible)
    # The eligible rows match the z-score of the eligible-only matrix: the
    # extreme ineligible row contributes nothing.
    baseline = cross_sectional_zscore(
        x[:3], torch.ones_like(x[:3], dtype=torch.bool)
    )
    assert torch.allclose(out[:3], baseline, atol=1e-6)
    # Ineligible cells are non-participating (NaN), not zeros in the sort.
    assert torch.isnan(out[3]).all()


def test_cs_demean_reference_is_eligible_only():
    x, eligible = _masked_x()
    out = _cs_demean(x, eligible)
    baseline = _cs_demean(x[:3], torch.ones_like(x[:3], dtype=torch.bool))
    assert torch.allclose(out[:3], baseline, atol=1e-6)
    assert torch.isnan(out[3]).all()


def test_cs_rank_reference_is_eligible_only():
    x, eligible = _masked_x()
    out = _cs_rank(x, eligible)
    baseline = _cs_rank(x[:3], torch.ones_like(x[:3], dtype=torch.bool))
    assert torch.allclose(out[:3], baseline, atol=1e-6)
    assert torch.isnan(out[3]).all()
    # Eligible ranks stay in [0, 1], monotone in value, with the top
    # eligible cell at (n_eligible - 1) / n_eligible.
    assert torch.all(out[:3] >= 0.0) and torch.all(out[:3] <= 1.0)
    assert out[2, 0].item() == pytest.approx(2.0 / 3.0)


def test_cs_ops_single_eligible_stock_is_neutral_but_valid():
    x = torch.tensor([[5.0, 7.0], [100.0, 200.0]])
    eligible = torch.tensor([[True, True], [False, False]])
    z = cross_sectional_zscore(x, eligible)
    dm = _cs_demean(x, eligible)
    # A single eligible member has zero dispersion: z-score and demean are
    # exactly neutral, but the member stays the only valid (finite) cell.
    assert torch.allclose(z[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(dm[0], torch.zeros(2), atol=1e-6)
    assert torch.isnan(z[1]).all() and torch.isnan(dm[1]).all()


def test_cs_ops_day_without_eligible_stocks_is_stable():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    eligible = torch.tensor([[True, False], [True, False]])
    for out in (
        cross_sectional_zscore(x, eligible),
        _cs_demean(x, eligible),
        _cs_rank(x, eligible),
    ):
        # The empty day collapses to the stable neutral 0: no NaN spread,
        # no extreme values.
        assert (out[:, 1] == 0.0).all()
        assert torch.isfinite(out).all()


def test_cs_neutralize_industry_means_use_eligible_finite_members():
    x = torch.tensor(
        [
            [10.0, 2.0],
            [20.0, 4.0],
            [1000.0, 2000.0],  # ineligible same-industry extreme
            [7.0, 5.0],  # unmapped
        ]
    )
    group = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [float("nan"), float("nan")],
        ]
    )
    eligible = torch.tensor(
        [
            [True, True],
            [True, True],
            [False, False],
            [True, True],
        ]
    )
    out = _cs_neutralize(x, group, eligible=eligible)
    # Group 0's mean counts only eligible members: (10+20)/2=15 -> [-5, +5]
    # on day 0 and (2+4)/2=3 -> [-1, +1] on day 1.
    assert out[0, 0].item() == -5.0 and out[1, 0].item() == 5.0
    assert out[0, 1].item() == -1.0 and out[1, 1].item() == 1.0
    # The ineligible member is non-participating; the unmapped eligible
    # member keeps its raw value and never entered group 0's mean.
    assert torch.isnan(out[2]).all()
    assert torch.allclose(out[3], x[3])


def test_cs_neutralize_unmapped_never_enters_other_industry_means():
    x = torch.tensor([[10.0], [20.0], [1000.0]])
    group = torch.tensor([[0.0], [0.0], [float("nan")]])
    eligible = torch.ones_like(x, dtype=torch.bool)
    out = _cs_neutralize(x, group, eligible=eligible)
    # The unmapped extreme must not move industry 0's mean (still 15).
    assert out[0, 0].item() == -5.0 and out[1, 0].item() == 5.0
    assert out[2, 0].item() == 1000.0


def test_cs_ops_reject_mask_shape_mismatch():
    x = torch.randn(3, 4)
    bad = torch.ones(4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="universe_mask shape"):
        cross_sectional_zscore(x, bad)
    with pytest.raises(ValueError, match="universe_mask shape"):
        _cs_demean(x, bad)
    with pytest.raises(ValueError, match="universe_mask shape"):
        _cs_rank(x, bad)
    with pytest.raises(ValueError, match="universe_mask shape"):
        _cs_neutralize(x, torch.zeros(3, 4), eligible=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_masked_cs_ops_match_cpu_on_cuda():
    torch.manual_seed(51)
    x = torch.randn(8, 12)
    eligible = torch.rand(8, 12) > 0.3
    group = torch.randint(0, 3, (8, 12)).float()
    group[torch.rand(8, 12) < 0.2] = float("nan")
    for fn, args, kwargs in (
        (cross_sectional_zscore, (x, eligible), {}),
        (_cs_demean, (x, eligible), {}),
        (_cs_rank, (x, eligible), {}),
        (_cs_neutralize, (x, group), {"eligible": eligible}),
    ):
        cpu = fn(*args, **kwargs)
        gpu = fn(*[a.cuda() for a in args], **{k: v.cuda() for k, v in kwargs.items()})
        # The NaN (non-participating) pattern and the finite values must
        # agree within the established float32 tolerance contract.
        assert torch.equal(torch.isnan(cpu), torch.isnan(gpu.cpu()))
        finite = ~torch.isnan(cpu)
        assert torch.allclose(cpu[finite], gpu.cpu()[finite], atol=1e-4, rtol=1e-4)
