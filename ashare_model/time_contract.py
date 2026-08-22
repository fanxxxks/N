"""Canonical t+2 date and fold boundaries.

All ranges are half-open. A signal at column ``t`` enters at ``t+1`` and
exits at ``t+2``; configured date anchors are inclusive and therefore use
``bisect_right``.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


ENTRY_OFFSET = 1
EXIT_OFFSET = 2


def _date_key(value: str) -> str:
    return str(value).replace("-", "")


def _normalized_dates(dates: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(str(value) for value in dates)
    keys = tuple(_date_key(value) for value in values)
    if not values:
        raise ValueError("date axis is empty")
    if any(left >= right for left, right in zip(keys, keys[1:])):
        raise ValueError("date axis must be strictly increasing")
    return values


@dataclass(frozen=True)
class TrainingTimeContract:
    """Resolved training cutoff, including the price context for t+2 labels."""

    dates: tuple[str, ...]
    train_end: str
    train_anchor_end_exclusive: int
    train_signal_start: int
    train_signal_end: int
    train_label_end: int

    @classmethod
    def resolve(
        cls,
        dates: list[str] | tuple[str, ...],
        train_end: str,
    ) -> "TrainingTimeContract":
        values = _normalized_dates(dates)
        keys = [_date_key(value) for value in values]
        anchor_end = bisect_right(keys, _date_key(train_end))
        signal_start = 0
        signal_end = anchor_end - EXIT_OFFSET
        if signal_end <= signal_start:
            raise ValueError(
                f"training window through {train_end} has no complete t+2 labels"
            )
        return cls(
            dates=values,
            train_end=train_end,
            train_anchor_end_exclusive=anchor_end,
            train_signal_start=signal_start,
            train_signal_end=signal_end,
            # Prices through the inclusive anchor are the complete, permitted
            # context from which training labels are recomputed.
            train_label_end=anchor_end,
        )

    @property
    def train_signal_range(self) -> range:
        return range(self.train_signal_start, self.train_signal_end)

    @property
    def train_signal_count(self) -> int:
        return self.train_signal_end - self.train_signal_start

    def signal_date(self, signal_index: int) -> str:
        self._validate_signal_index(signal_index)
        return self.dates[signal_index]

    def entry_date(self, signal_index: int) -> str:
        self._validate_signal_index(signal_index)
        return self.dates[signal_index + ENTRY_OFFSET]

    def exit_date(self, signal_index: int) -> str:
        self._validate_signal_index(signal_index)
        return self.dates[signal_index + EXIT_OFFSET]

    def _validate_signal_index(self, signal_index: int) -> None:
        if signal_index not in self.train_signal_range:
            raise IndexError(f"signal index {signal_index} is outside the training range")


@dataclass(frozen=True)
class FoldTimeContract:
    """One immutable train/test fold resolved against a concrete date axis."""

    dates: tuple[str, ...]
    train_end: str
    test_end: str
    train_anchor_end_exclusive: int
    train_signal_start: int
    train_signal_end: int
    train_label_end: int
    test_signal_start: int
    test_signal_end: int
    test_price_end: int

    @classmethod
    def resolve(
        cls,
        dates: list[str] | tuple[str, ...],
        train_end: str,
        test_end: str,
    ) -> "FoldTimeContract":
        values = _normalized_dates(dates)
        keys = [_date_key(value) for value in values]
        train_anchor_end = bisect_right(keys, _date_key(train_end))
        test_anchor_end = bisect_right(keys, _date_key(test_end))
        train_signal_start = 0
        train_signal_end = train_anchor_end - EXIT_OFFSET
        test_signal_start = train_anchor_end
        test_signal_end = test_anchor_end - EXIT_OFFSET
        if train_signal_end <= train_signal_start:
            raise ValueError(
                f"fold {train_end} -> {test_end}: train window has no complete t+2 labels"
            )
        if test_signal_end <= test_signal_start:
            raise ValueError(
                f"fold {train_end} -> {test_end}: test window has no complete t+2 returns"
            )
        if test_anchor_end > len(values):
            raise ValueError("test price boundary exceeds the date axis")
        return cls(
            dates=values,
            train_end=train_end,
            test_end=test_end,
            train_anchor_end_exclusive=train_anchor_end,
            train_signal_start=train_signal_start,
            train_signal_end=train_signal_end,
            train_label_end=train_anchor_end,
            test_signal_start=test_signal_start,
            test_signal_end=test_signal_end,
            test_price_end=test_anchor_end,
        )

    @property
    def train_signal_range(self) -> range:
        return range(self.train_signal_start, self.train_signal_end)

    @property
    def test_signal_range(self) -> range:
        return range(self.test_signal_start, self.test_signal_end)

    @property
    def train_signal_count(self) -> int:
        return self.train_signal_end - self.train_signal_start

    @property
    def test_signal_count(self) -> int:
        return self.test_signal_end - self.test_signal_start

    @property
    def test_price_range(self) -> range:
        return range(self.test_signal_start, self.test_price_end)

    def signal_date(self, signal_index: int) -> str:
        self._validate_signal_index(signal_index)
        return self.dates[signal_index]

    def entry_date(self, signal_index: int) -> str:
        self._validate_signal_index(signal_index)
        return self.dates[signal_index + ENTRY_OFFSET]

    def exit_date(self, signal_index: int) -> str:
        self._validate_signal_index(signal_index)
        return self.dates[signal_index + EXIT_OFFSET]

    def _validate_signal_index(self, signal_index: int) -> None:
        if signal_index not in self.train_signal_range and signal_index not in self.test_signal_range:
            raise IndexError(f"signal index {signal_index} is outside this fold")
