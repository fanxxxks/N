"""Causal rebalance calendars (P3-01/P3-02)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np


REBALANCE_POLICY_VERSION = 1

_MAX_NON_OVERLAPPING_HORIZON = {
    "daily": 1,
    "weekly": 1,
    "every_5_days": 5,
    "every_10_days": 10,
}


def _date_key(value: str) -> str:
    return str(value).replace("-", "")


def _validated_dates(dates: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in dates)
    keys = tuple(_date_key(value) for value in values)
    if any(left >= right for left, right in zip(keys, keys[1:])):
        raise ValueError("date axis must be strictly increasing")
    return values


@dataclass(frozen=True)
class RebalancePolicy:
    """One validated rebalance schedule and target horizon.

    The schedule is always resolved on the complete date axis. Consumers
    slice the returned mask; they must never restart a 5/10-day cadence at a
    fold boundary.
    """

    frequency: str = "daily"
    horizon: int = 1

    def __post_init__(self) -> None:
        frequency = str(self.frequency)
        if frequency not in _MAX_NON_OVERLAPPING_HORIZON:
            allowed = ", ".join(_MAX_NON_OVERLAPPING_HORIZON)
            raise ValueError(
                f"frequency must be one of {allowed}, got {frequency!r}"
            )
        horizon = int(self.horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")
        maximum = _MAX_NON_OVERLAPPING_HORIZON[frequency]
        if horizon > maximum:
            raise ValueError(
                f"frequency={frequency!r}, horizon={horizon} would create "
                f"overlapping labels; maximum non-overlapping horizon is {maximum}"
            )
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "horizon", horizon)

    @classmethod
    def from_config(cls, config) -> "RebalancePolicy":
        frequency = getattr(
            config,
            "rebalance_frequency",
            getattr(config, "frequency", "daily"),
        )
        horizon = getattr(
            config,
            "target_horizon",
            getattr(config, "horizon", 1),
        )
        return cls(str(frequency), int(horizon))

    @property
    def entry_offset(self) -> int:
        return 1

    @property
    def exit_offset(self) -> int:
        return self.entry_offset + self.horizon

    def entry_index(self, signal_index: int) -> int:
        return int(signal_index) + self.entry_offset

    def exit_index(self, signal_index: int) -> int:
        return int(signal_index) + self.exit_offset

    def rebalance_mask(self, dates: Sequence[str]) -> np.ndarray:
        values = _validated_dates(dates)
        count = len(values)
        mask = np.zeros(count, dtype=bool)
        if count == 0:
            return mask
        if self.frequency == "daily":
            mask[:] = True
        elif self.frequency == "every_5_days":
            mask[::5] = True
        elif self.frequency == "every_10_days":
            mask[::10] = True
        else:
            previous_week: tuple[int, int] | None = None
            previous_index = -1
            for index, value in enumerate(values):
                try:
                    iso = datetime.strptime(
                        _date_key(value), "%Y%m%d"
                    ).isocalendar()
                except ValueError as exc:
                    raise ValueError(f"invalid weekly trade date {value!r}") from exc
                week = (iso.year, iso.week)
                if previous_week is not None and week != previous_week:
                    mask[previous_index] = True
                previous_week = week
                previous_index = index
            mask[previous_index] = True
        return mask

    def executable_signal_indices(self, dates: Sequence[str]) -> list[int]:
        mask = self.rebalance_mask(dates)
        executable_end = max(len(mask) - self.exit_offset, 0)
        return np.flatnonzero(mask[:executable_end]).astype(int).tolist()
