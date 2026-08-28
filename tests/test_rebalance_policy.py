"""P3-01/P3-02 contract tests.

Assertion source: ``docs/p3_portfolio_contract.md`` section 1.  These tests
pin the schedule and label intervals; expected values are derived from the
declared signal/entry/exit relationship, never from implementation output.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import make_backtest_config, make_protocol_config
from ashare_model.targets import causal_target_returns
from ashare_model.time_contract import FoldTimeContract, TrainingTimeContract
from ashare_portfolio.rebalance import RebalancePolicy


DATES = [
    "20240102",
    "20240103",
    "20240105",
    "20240108",
    "20240109",
    "20240112",
    "20240115",
    "20240116",
    "20240117",
    "20240118",
    "20240119",
    "20240122",
    "20240123",
    "20240124",
    "20240125",
    "20240126",
    "20240129",
    "20240130",
    "20240131",
    "20240201",
    "20240202",
    "20240205",
    "20240206",
    "20240207",
    "20240208",
]


@pytest.mark.parametrize(
    ("frequency", "horizon", "expected"),
    [
        ("daily", 1, list(range(len(DATES)))),
        ("weekly", 1, [2, 5, 10, 15, 20, 24]),
        ("every_5_days", 5, [0, 5, 10, 15, 20]),
        ("every_10_days", 10, [0, 10, 20]),
    ],
)
def test_rebalance_policy_schedule_is_exact(
    frequency: str, horizon: int, expected: list[int]
):
    policy = RebalancePolicy(frequency, horizon=horizon)
    assert np.flatnonzero(policy.rebalance_mask(DATES)).tolist() == expected


def test_executable_indices_require_the_declared_exit_open():
    daily = RebalancePolicy("daily", horizon=1)
    every_five = RebalancePolicy("every_5_days", horizon=5)
    assert daily.executable_signal_indices(DATES) == list(range(len(DATES) - 2))
    # t=20 would exit at index 26, outside this 25-session axis.
    assert every_five.executable_signal_indices(DATES) == [0, 5, 10, 15]
    assert daily.entry_index(7) == 8
    assert daily.exit_index(7) == 9
    assert every_five.entry_index(7) == 8
    assert every_five.exit_index(7) == 13


@pytest.mark.parametrize(
    ("frequency", "horizon"),
    [
        ("daily", 2),
        ("weekly", 2),
        ("every_5_days", 6),
        ("every_10_days", 11),
    ],
)
def test_overlap_prone_frequency_horizon_pairs_are_rejected(
    frequency: str, horizon: int
):
    with pytest.raises(ValueError, match="overlap"):
        RebalancePolicy(frequency, horizon=horizon)


def test_unknown_frequency_and_non_positive_horizon_are_rejected():
    with pytest.raises(ValueError, match="frequency"):
        RebalancePolicy("monthly", horizon=1)
    with pytest.raises(ValueError, match="horizon"):
        RebalancePolicy("daily", horizon=0)


@pytest.mark.parametrize(
    ("frequency", "horizon"),
    [
        ("daily", 1),
        ("weekly", 1),
        ("every_5_days", 5),
        ("every_10_days", 10),
    ],
)
def test_all_effective_label_intervals_are_non_overlapping(
    frequency: str, horizon: int
):
    policy = RebalancePolicy(frequency, horizon=horizon)
    spans = [
        (policy.entry_index(t), policy.exit_index(t))
        for t in policy.executable_signal_indices(DATES)
    ]
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_causal_multi_period_target_uses_entry_and_exit_open_only():
    opens = np.asarray(
        [[10.0, 11.0, 12.0, 15.0, 18.0, 21.0, 24.0,
          30.0, 33.0, 36.0, 42.0, 45.0, 48.0]],
        dtype=np.float64,
    )
    dates = DATES[: opens.shape[1]]
    policy = RebalancePolicy("every_5_days", horizon=5)
    target = causal_target_returns(opens, dates, policy)

    # signal 0 -> entry open[1]=11 -> exit open[6]=24
    assert target[0, 0] == pytest.approx(24.0 / 11.0 - 1.0)
    # signal 5 -> entry open[6]=24 -> exit open[11]=45
    assert target[0, 5] == pytest.approx(45.0 / 24.0 - 1.0)
    # Non-rebalance columns and incomplete final labels are research-missing.
    assert np.isnan(target[0, 1:5]).all()
    assert np.isnan(target[0, 10])


def test_causal_target_missing_endpoint_is_nan_not_zero():
    opens = np.full((2, len(DATES)), 10.0, dtype=np.float64)
    opens[0, 1] = np.nan  # entry endpoint of signal 0
    opens[1, 6] = 0.0  # exit endpoint of signal 0, invalid price
    target = causal_target_returns(
        opens, DATES, RebalancePolicy("every_5_days", horizon=5)
    )
    assert np.isnan(target[0, 0])
    assert np.isnan(target[1, 0])


def test_time_contract_offsets_follow_horizon_without_crossing_anchor():
    training = TrainingTimeContract.resolve(DATES, DATES[15], horizon=5)
    # Inclusive anchor end is 16; exit offset is 1 + 5 = 6.
    assert training.train_anchor_end_exclusive == 16
    assert training.train_signal_end == 10
    assert training.train_label_end == 16
    assert training.horizon == 5
    assert training.exit_offset == 6
    assert training.entry_date(9) == DATES[10]
    assert training.exit_date(9) == DATES[15]

    fold = FoldTimeContract.resolve(
        DATES, train_end=DATES[10], test_end=DATES[24], horizon=5
    )
    assert fold.train_signal_end == 5
    assert fold.test_signal_start == 11
    assert fold.test_signal_end == 19
    assert fold.test_price_end == 25
    assert fold.exit_date(18) == DATES[24]


def test_protocol_and_backtest_share_frequency_and_horizon():
    raw = {
        "protocol": {"frequency": "every_5_days", "horizon": 5},
        "backtest": {},
    }
    protocol = make_protocol_config(raw)
    backtest = make_backtest_config(raw)
    assert protocol.frequency == backtest.rebalance_frequency == "every_5_days"
    assert protocol.horizon == backtest.target_horizon == 5


def test_protocol_rejects_overlap_prone_pair_at_config_boundary():
    with pytest.raises(ValueError, match="overlap"):
        make_protocol_config(
            {"protocol": {"frequency": "daily", "horizon": 5}}
        )
