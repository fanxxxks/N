"""Research targets with explicit signal/entry/holding causality (P3-02)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ashare_portfolio.rebalance import RebalancePolicy


TARGET_CONTRACT_VERSION = 1


def causal_target_returns(
    open_: np.ndarray,
    dates: Sequence[str],
    policy: RebalancePolicy,
    *,
    rebalance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return sparse forward-open labels under ``policy``.

    At a rebalance signal ``t`` the label is
    ``open[t + 1 + horizon] / open[t + 1] - 1``. Non-rebalance columns,
    incomplete labels and missing/non-positive endpoints are ``NaN`` so
    research statistics cannot mistake them for zero-return observations.
    """

    open_arr = np.asarray(open_, dtype=np.float64)
    if open_arr.ndim != 2:
        raise ValueError("open_ must be [stock, date]")
    if open_arr.shape[1] != len(dates):
        raise ValueError(
            f"open_ has {open_arr.shape[1]} dates but date axis has {len(dates)}"
        )
    if rebalance_mask is None:
        rebalance_mask = policy.rebalance_mask(dates)
    rebalance_mask = np.asarray(rebalance_mask, dtype=bool)
    if rebalance_mask.shape != (len(dates),):
        raise ValueError(
            f"rebalance_mask shape {rebalance_mask.shape} does not match "
            f"date axis ({len(dates)},)"
        )
    executable_end = max(len(dates) - policy.exit_offset, 0)
    signal_indices = np.flatnonzero(rebalance_mask[:executable_end])
    target = np.full(open_arr.shape, np.nan, dtype=np.float64)
    for signal_index in signal_indices:
        entry = policy.entry_index(signal_index)
        exit_ = policy.exit_index(signal_index)
        entry_open = open_arr[:, entry]
        exit_open = open_arr[:, exit_]
        valid = (
            np.isfinite(entry_open)
            & np.isfinite(exit_open)
            & (entry_open > 0.0)
            & (exit_open > 0.0)
        )
        target[valid, signal_index] = (
            exit_open[valid] / entry_open[valid] - 1.0
        )
    return target
