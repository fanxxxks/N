"""Tensor operators used by the A-share StackVM.

Conventions
-----------
* Windowed operators (MA20/STD20/TS_RANK20) only look backwards.  Leading
  positions use the values actually available (an expanding window), so a
  constant series has a constant moving average and zero standard deviation.
* Delay-based operators (DELAY1/MAX3/DECAY/DELTA5) extend the series with
  the first available value (constant extension) instead of zeros, so the
  leading positions do not fabricate spurious jumps or decay.
* DIV is guarded against a zero denominator.
"""

from __future__ import annotations

import torch


def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    """Shift a series forward by ``d`` steps, extending with the first value."""
    if d <= 0:
        return x
    if d >= x.shape[1]:
        return x[:, :1].expand(-1, x.shape[1])
    pad = x[:, :1].expand(-1, d)
    return torch.cat([pad, x[:, :-d]], dim=1)


def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)


def _ts_window(
    x: torch.Tensor,
    d: int,
    reducer,
) -> torch.Tensor:
    """Apply a trailing-window reducer while avoiding future leakage.

    Windows are padded with NaN on the left and the reducer must be
    NaN-aware (e.g. ``torch.nanmean``).  Every window contains at least one
    real value, so the output stays finite; for series shorter than ``d`` an
    expanding window over the available values is used, which keeps the
    short-series semantics explicit.
    """

    if d <= 1:
        return x
    b, t = x.shape
    if t < d:
        # Not enough history: use an expanding window for the available part.
        out = torch.zeros_like(x)
        for i in range(t):
            out[:, i] = reducer(x[:, : i + 1], dim=1)
        return out
    pad = torch.full((b, d - 1), float("nan"), device=x.device, dtype=x.dtype)
    padded = torch.cat([pad, x], dim=1)
    windows = padded.unfold(1, d, 1)
    return reducer(windows, dim=-1)


def _ts_ma(x: torch.Tensor, d: int) -> torch.Tensor:
    return _ts_window(x, d, lambda w, dim: torch.nanmean(w, dim=dim))


def _ts_std(x: torch.Tensor, d: int) -> torch.Tensor:
    def reducer(w: torch.Tensor, dim: int) -> torch.Tensor:
        mean = torch.nanmean(w, dim=dim, keepdim=True)
        var = torch.nanmean((w - mean) ** 2, dim=dim)
        return torch.sqrt(var)

    return _ts_window(x, d, reducer)


def _ts_rank(x: torch.Tensor, d: int) -> torch.Tensor:
    if d <= 1:
        return torch.zeros_like(x)
    b, t = x.shape
    out = torch.zeros_like(x)
    for i in range(t):
        start = max(0, i - d + 1)
        window = x[:, start : i + 1]  # [B, W]
        last = x[:, i : i + 1]
        # Fraction of window values below or equal to the latest value.
        rank = (window <= last).float().mean(dim=1)
        out[:, i] = rank
    return out


def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y


def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)


def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)


def _op_max3(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.maximum(_ts_delay(x, 1), _ts_delay(x, 2)))


def _safe_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Divide with a guard that also handles a zero denominator."""
    eps = 1e-6
    denom = torch.where(y == 0, torch.full_like(y, eps), y + eps * torch.sign(y))
    return x / denom


OPS_CONFIG = [
    ("ADD", lambda x, y: x + y, 2),
    ("SUB", lambda x, y: x - y, 2),
    ("MUL", lambda x, y: x * y, 2),
    ("DIV", _safe_div, 2),
    ("NEG", lambda x: -x, 1),
    ("ABS", torch.abs, 1),
    ("SIGN", torch.sign, 1),
    ("GATE", _op_gate, 3),
    ("JUMP", _op_jump, 1),
    ("DECAY", _op_decay, 1),
    ("DELAY1", lambda x: _ts_delay(x, 1), 1),
    ("MAX3", _op_max3, 1),
    ("DELTA5", lambda x: _ts_delta(x, 5), 1),
    ("MA20", lambda x: _ts_ma(x, 20), 1),
    ("STD20", lambda x: _ts_std(x, 20), 1),
    ("TS_RANK20", lambda x: _ts_rank(x, 20), 1),
]


OP_ARITY = {cfg[0]: cfg[2] for cfg in OPS_CONFIG}
