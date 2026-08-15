from __future__ import annotations

import torch

from ashare_model.ops import OPS_CONFIG, _ts_delay, _ts_ma, _ts_std, _ts_window, _ts_rank


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
