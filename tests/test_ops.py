from __future__ import annotations

import numpy as np
import torch

from ashare_model.ops import (
    OPS_CONFIG,
    _cs_demean,
    _cs_neutralize,
    _cs_rank,
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
    for name, fn, arity in OPS_CONFIG:
        if arity == 1:
            out = fn(x)
        elif arity == 2:
            out = fn(x, y)
        else:
            out = fn(x, x, x)
        assert out.shape == x.shape, name
        assert torch.isfinite(out).all(), name


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
        out = _cs_rank(x).numpy()
        for day in range(t):
            expected = _numpy_avg_rank(x[:, day].numpy())
            assert np.allclose(out[:, day], expected, atol=1e-6)


def test_cs_rank_is_monotone_invariant_and_bounded():
    torch.manual_seed(19)
    x = torch.randn(12, 6)
    out = _cs_rank(x)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)
    # Affine transforms do not change ranks.
    assert torch.allclose(_cs_rank(x * 3.0 - 2.0), out, atol=1e-6)
    # Rank preserves order: the largest value gets rank 1 (or the shared
    # top rank under ties).
    assert (out[x == x.max(dim=0, keepdim=True).values] >= out.max() - 1e-6).all()


def test_cs_operators_see_only_the_current_cross_section():
    # No-leakage contract: changing a future column must not change any
    # earlier column of any cross-sectional operator.
    torch.manual_seed(23)
    x = torch.randn(10, 8)
    base = {"rank": _cs_rank(x), "z": cross_sectional_zscore(x), "dm": _cs_demean(x)}
    changed = x.clone()
    changed[:, 5] = torch.randn(10) * 100.0
    for name, out in (
        ("rank", _cs_rank(changed)),
        ("z", cross_sectional_zscore(changed)),
        ("dm", _cs_demean(changed)),
    ):
        assert torch.allclose(out[:, :5], base[name][:, :5], atol=1e-6), name


def test_cs_zscore_and_demean_numerics():
    x = torch.tensor([[1.0, 4.0, 9.0], [2.0, 4.0, 6.0], [3.0, 4.0, 3.0]])
    z = cross_sectional_zscore(x)
    assert torch.allclose(z.mean(dim=0)[[0, 2]], torch.zeros(2), atol=1e-6)
    assert torch.allclose(z.std(dim=0, unbiased=False)[[0, 2]], torch.ones(2), atol=1e-5)
    # The constant column is degenerate: it maps to neutral 0.
    assert (z[:, 1] == 0.0).all()
    dm = _cs_demean(x)
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
    out = _cs_neutralize(x, group)
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
    assert torch.allclose(_cs_neutralize(x), _cs_demean(x))
    assert torch.allclose(
        _cs_neutralize(x, torch.full_like(x, float("nan"))), _cs_demean(x)
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
