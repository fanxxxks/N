from __future__ import annotations

import pytest

from ashare_model.time_contract import (
    ENTRY_OFFSET,
    EXIT_OFFSET,
    FoldTimeContract,
    TrainingTimeContract,
)


DATES = [
    "20240101",
    "20240102",
    "20240103",
    "20240104",
    "20240105",
    "20240108",
    "20240109",
    "20240110",
    "20240111",
    "20240112",
]


def test_offsets_and_inclusive_exact_anchor():
    contract = FoldTimeContract.resolve(DATES, "2024-01-03", "2024-01-10")
    assert (ENTRY_OFFSET, EXIT_OFFSET) == (1, 2)
    assert contract.train_anchor_end_exclusive == 3
    assert contract.train_signal_end == 1
    assert contract.train_label_end == 3
    assert contract.test_signal_start == 3
    assert contract.test_signal_end == 6
    assert contract.test_price_end == 8
    assert contract.signal_date(3) == "20240104"
    assert contract.entry_date(3) == "20240105"
    assert contract.exit_date(3) == "20240108"


def test_weekend_anchor_uses_last_prior_trading_day_inclusively():
    # Saturday 2024-01-06 resolves after Friday, not at Monday.
    contract = FoldTimeContract.resolve(DATES, "2024-01-06", "2024-01-12")
    assert contract.train_anchor_end_exclusive == 5
    assert contract.train_signal_end == 3
    assert contract.test_signal_start == 5
    assert contract.test_price_end == 10


def test_training_contract_separates_signal_and_price_context():
    contract = TrainingTimeContract.resolve(DATES, "2024-01-10")
    assert contract.train_label_end == 8
    assert contract.train_signal_end == 6
    assert list(contract.train_signal_range) == list(range(6))
    assert contract.exit_date(5) == "20240110"
    with pytest.raises(IndexError):
        contract.exit_date(6)


def test_contract_rejects_empty_t2_ranges_and_unsorted_dates():
    with pytest.raises(ValueError, match="training window"):
        TrainingTimeContract.resolve(DATES, "2024-01-02")
    with pytest.raises(ValueError, match="test window"):
        FoldTimeContract.resolve(DATES, "2024-01-10", "2024-01-12")
    with pytest.raises(ValueError, match="strictly increasing"):
        FoldTimeContract.resolve(DATES[::-1], "2024-01-03", "2024-01-10")
