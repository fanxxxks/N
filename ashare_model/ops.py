"""Tensor operators used by the A-share StackVM.

Conventions
-----------
* Windowed operators (MA20/STD20/TS_RANK20/CORR20/DOWNVOL20) only look
  backwards.  Leading positions use the values actually available (an
  expanding window), so a constant series has a constant moving average and
  zero standard deviation; CORR20 with a degenerate window yields 0.
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
    """Trailing-window percentile rank of the latest value.

    Vectorized with ``unfold`` (chunked over time to bound the window
    intermediate), producing the same values as the previous per-column
    loop: leading columns rank inside the expanding window of real values
    (NaN padding is excluded from the denominator), later columns rank
    inside the trailing ``d``-window.  The VM clamps non-finite values to
    0, so NaN inputs are unreachable through the normal path.
    """
    if d <= 1:
        return torch.zeros_like(x)
    b, t = x.shape
    out = torch.empty_like(x)
    # Bound the [b, block, d] unfold: budget in float32 elements (~256 MB).
    block = max(1, (1 << 26) // max(b * d, 1))
    nan_pad = torch.full((b, d - 1), float("nan"), device=x.device, dtype=x.dtype)
    for start in range(0, t, block):
        end = min(start + block, t)
        seg = x[:, max(0, start - d + 1) : end]
        lead_pad = max(d - 1 - start, 0)
        if lead_pad:
            padded = torch.cat([nan_pad[:, :lead_pad], seg], dim=1)
        else:
            padded = seg
        windows = padded.unfold(1, d, 1)  # [b, end-start, d]
        last = x[:, start:end].unsqueeze(-1)
        # NaN comparisons are False (same as the previous loop), and NaN
        # padding never dilutes the denominator of the expanding window.
        leq = (windows <= last).float().sum(dim=-1)
        den = torch.isfinite(windows).sum(dim=-1).clamp(min=1).float()
        out[:, start:end] = leq / den
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


def _ts_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """Trailing-window Pearson correlation between two series.

    Implemented with prefix sums so it is fully vectorized over batch and
    time (the VM executes on the full [stock x date] tensor inside the RL
    loop).  Expanding windows are used for the leading positions; a
    degenerate window (zero variance) yields correlation 0.
    """

    if d <= 1:
        return torch.zeros_like(x)
    b, t = x.shape
    pad = torch.zeros(b, 1, device=x.device, dtype=x.dtype)
    sx = torch.cumsum(torch.cat([pad, x], dim=1), dim=1)
    sy = torch.cumsum(torch.cat([pad, y], dim=1), dim=1)
    sxx = torch.cumsum(torch.cat([pad, x * x], dim=1), dim=1)
    syy = torch.cumsum(torch.cat([pad, y * y], dim=1), dim=1)
    sxy = torch.cumsum(torch.cat([pad, x * y], dim=1), dim=1)
    idx = torch.arange(t, device=x.device)
    start = torch.clamp(idx - d + 1, min=0)  # inclusive window start

    def window_sum(prefix: torch.Tensor) -> torch.Tensor:
        return prefix[:, idx + 1] - prefix[:, start]

    n = (idx - start + 1).to(x.dtype)
    xs, ys = window_sum(sx), window_sum(sy)
    cov = window_sum(sxy) - xs * ys / n
    vx = torch.clamp(window_sum(sxx) - xs * xs / n, min=0.0)
    vy = torch.clamp(window_sum(syy) - ys * ys / n, min=0.0)
    denom = torch.sqrt(vx * vy)
    return torch.where(
        denom > 1e-9,
        cov / torch.clamp(denom, min=1e-9),
        torch.zeros_like(cov),
    )


def _ts_downvol(x: torch.Tensor, d: int) -> torch.Tensor:
    """Trailing-window downside volatility: sqrt of the mean of squared
    negative values (positive values contribute 0, matching the neutral-0
    convention of the standardized factor stack)."""

    def reducer(w: torch.Tensor, dim: int) -> torch.Tensor:
        down = torch.clamp(w, max=0.0)
        return torch.sqrt(torch.nanmean(down * down, dim=dim))

    return _ts_window(x, d, reducer)


# --- cross-sectional operators -----------------------------------------------


def _validate_eligible(
    x: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    """Return ``eligible`` after enforcing the ``[stock, date]`` alignment."""

    if eligible.shape != x.shape:
        raise ValueError(
            f"universe_mask shape {tuple(eligible.shape)} does not match "
            f"signal shape {tuple(x.shape)}"
        )
    return eligible


def _cs_reference(
    x: torch.Tensor, eligible: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(ref, count)``: the per-cell reference mask and its per-date count.

    A cell is a reference cell when it is finite and PIT-eligible.
    ``count`` has shape ``[T]``.
    """

    eligible = _validate_eligible(x, eligible)
    finite = torch.isfinite(x)
    ref = finite & eligible
    return ref, ref.sum(dim=0)


def _masked_output(
    out: torch.Tensor,
    ref: torch.Tensor,
    count: torch.Tensor,
) -> torch.Tensor:
    """Finalize a cross-sectional operator's output.

    Non-reference cells become NaN (non-participating) so they can never
    enter a downstream sort; a date with no reference cell at all collapses
    to the stable neutral 0 instead of spreading NaN statistics.
    """

    out = torch.where(ref, out, torch.full_like(out, float("nan")))
    out[:, count == 0] = 0.0
    return out


def cross_sectional_zscore(
    signal: torch.Tensor,
    eligible: torch.Tensor,
) -> torch.Tensor:
    """Per-date cross-sectional z-score of a ``[stock, date]`` signal.

    The mean and standard deviation are taken over the *reference* cells of
    each date column — finite and PIT-eligible (``eligible`` is the
    mandatory ``[stock, date]`` bool mask) — so only the current
    cross-section counts and no future information enters.  Reference cells
    get the z-score; non-reference cells map to NaN (non-participating).  A
    degenerate column (zero standard deviation, e.g. an all-neutral date)
    maps to 0: the numerator is 0 there as well, so the ``eps`` guard alone
    is sufficient; a column with no reference cell at all also maps to the
    stable neutral 0.  Shared by the ``CS_ZSCORE`` operator and the VM's
    terminal standardization.
    """

    ref, count = _cs_reference(signal, eligible)
    safe = torch.where(ref, signal, torch.zeros_like(signal))
    mean = safe.sum(dim=0) / count.clamp(min=1)  # [T]
    dev = torch.where(ref, signal - mean, torch.zeros_like(signal))
    var = (dev * dev).sum(dim=0) / count.clamp(min=1)
    std = torch.sqrt(var)
    out = (signal - mean) / (std + 1e-6)
    return _masked_output(out, ref, count)


def _cs_demean(
    x: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    """Subtract the per-date eligible-only reference mean.  Non-reference
    cells stay NaN."""

    ref, count = _cs_reference(x, eligible)
    safe = torch.where(ref, x, torch.zeros_like(x))
    mean = safe.sum(dim=0) / count.clamp(min=1)  # [T]
    return _masked_output(x - mean, ref, count)


def _cs_rank(
    x: torch.Tensor, eligible: torch.Tensor
) -> torch.Tensor:
    """Per-date cross-sectional percentile rank in ``[0, 1]``.

    Ranks are computed among the reference cells (finite and PIT-eligible);
    non-reference cells map to NaN.  Ties share
    the *average* rank of their group, so the result depends only on the
    values — never on the sort order of equal elements — and is therefore
    identical on CPU and CUDA.  Implemented with a stable argsort and
    per-group start/end reductions (no quadratic pairwise comparisons, no
    reliance on running-min scans).  Non-reference cells are pushed to the
    bottom with a ``-inf`` sentinel so their positions can be subtracted in
    one vectorized pass.
    """

    ref, count = _cs_reference(x, eligible)
    n, t = x.shape
    ranked = torch.where(ref, x, torch.full_like(x, float("-inf")))
    order = torch.argsort(ranked, dim=0, stable=True)  # [n, t]
    sorted_x = ranked.gather(0, order)
    # boundary[p] marks the first sorted position of each equal-value group.
    diff = sorted_x[1:] != sorted_x[:-1]  # [n-1, t]
    boundary = torch.cat([diff.new_ones((1, t)), diff], dim=0)
    gid = boundary.cumsum(dim=0) - 1  # [n, t] dense group ids
    n_groups = int(gid.max()) + 1
    pos = (
        torch.arange(n, device=x.device, dtype=x.dtype)
        .unsqueeze(1)
        .expand(n, t)
    )
    starts = torch.full(
        (n_groups, t), float("inf"), device=x.device, dtype=x.dtype
    )
    starts.scatter_reduce_(0, gid, pos, reduce="amin", include_self=False)
    ends = torch.full(
        (n_groups, t), float("-inf"), device=x.device, dtype=x.dtype
    )
    ends.scatter_reduce_(0, gid, pos, reduce="amax", include_self=False)
    # Average rank of the group = mean of its first and last sorted position.
    avg_sorted = ((starts + ends) / 2.0).gather(0, gid)
    full = torch.zeros_like(x)
    full.scatter_(0, order, avg_sorted)
    # The non-reference cells occupy the bottom ``n - count`` positions of
    # each column, so the reference ranks are full-rank minus that offset,
    # normalized by the number of reference cells.
    excluded = n - count  # [T]
    return _masked_output((full - excluded) / count.clamp(min=1), ref, count)


def _cs_neutralize(
    x: torch.Tensor,
    group: torch.Tensor | None = None,
    *,
    eligible: torch.Tensor,
) -> torch.Tensor:
    """Subtract the per-date mean within each group (industry neutralization).

    ``group`` is a ``[stock, date]`` tensor of discrete group ids aligned
    with ``x``; non-finite cells (stocks without a mapping) belong to no
    group, so their values are untouched — and never enter another group's
    mean.  ``eligible`` is the mandatory ``[stock, date]`` PIT universe mask:
    a group's mean uses only the same-day members that are eligible *and*
    finite, and ineligible cells map to NaN (non-participating).  Without
    ``group`` — or when no finite group id exists — the operator degrades
    to the full-market demean, so a missing industry source never
    fabricates a grouping.
    """

    if group is None:
        return _cs_demean(x, eligible)
    ids = torch.unique(group[torch.isfinite(group)])
    if ids.numel() == 0:
        return _cs_demean(x, eligible)
    ref, count = _cs_reference(x, eligible)
    out = x.clone()
    for gid in ids:
        member = (group == gid) & ref
        group_count = member.sum(dim=0, keepdim=True)
        total = torch.where(member, x, torch.zeros_like(x)).sum(
            dim=0, keepdim=True
        )
        mean = total / group_count.clamp(min=1)
        out = torch.where(group == gid, x - mean, out)
    return _masked_output(out, ref, count)


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
    ("CORR20", lambda x, y: _ts_corr(x, y, 20), 2),
    ("DOWNVOL20", lambda x: _ts_downvol(x, 20), 1),
    # Cross-sectional operators: the "relative strength" family, enabling
    # formulas to express rank/zscore/neutral semantics inside the stack.
    ("CS_RANK", _cs_rank, 1),
    ("CS_ZSCORE", cross_sectional_zscore, 1),
    ("CS_DEMEAN", _cs_demean, 1),
    # CS_NEUTRALIZE groups by the industry codes carried on the VM; without
    # them it degrades to CS_DEMEAN (the VM substitutes the group tensor).
    ("CS_NEUTRALIZE", lambda x: _cs_neutralize(x), 1),
    # Parameterized-window family, enumerated: the 20-day versions above are
    # the original operators; 5/10/60 give the policy short and long
    # horizons without changing the sampling grammar.
    ("MA5", lambda x: _ts_ma(x, 5), 1),
    ("MA10", lambda x: _ts_ma(x, 10), 1),
    ("MA60", lambda x: _ts_ma(x, 60), 1),
    ("STD5", lambda x: _ts_std(x, 5), 1),
    ("STD10", lambda x: _ts_std(x, 10), 1),
    ("STD60", lambda x: _ts_std(x, 60), 1),
    ("TS_RANK5", lambda x: _ts_rank(x, 5), 1),
    ("TS_RANK10", lambda x: _ts_rank(x, 10), 1),
    ("TS_RANK60", lambda x: _ts_rank(x, 60), 1),
    ("CORR5", lambda x, y: _ts_corr(x, y, 5), 2),
    ("CORR10", lambda x, y: _ts_corr(x, y, 10), 2),
    ("CORR60", lambda x, y: _ts_corr(x, y, 60), 2),
    ("DOWNVOL5", lambda x: _ts_downvol(x, 5), 1),
    ("DOWNVOL10", lambda x: _ts_downvol(x, 10), 1),
    ("DOWNVOL60", lambda x: _ts_downvol(x, 60), 1),
    ("DELTA10", lambda x: _ts_delta(x, 10), 1),
    ("DELTA20", lambda x: _ts_delta(x, 20), 1),
]
