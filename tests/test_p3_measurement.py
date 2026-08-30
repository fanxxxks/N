"""P3 reproducible measurement artifact contract."""

from __future__ import annotations

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig
from ashare_model.data_loader import AshareDataLoader
from ashare_model.p3_measurement import (
    P3_MEASUREMENT_VERSION,
    build_p3_measurement,
)


def test_p3_measurement_schema_version_is_pinned():
    assert P3_MEASUREMENT_VERSION == 3


def test_build_p3_measurement_covers_acceptance_metrics(
    populated_db: DataConfig,
):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    config = BacktestConfig(
        initial_capital=100_000.0,
        top_n=2,
        buy_rank=2,
        sell_rank=3,
        single_weight_cap=0.5,
        train_end_date="2024-02-01",
    )

    payload = build_p3_measurement(
        loader,
        config,
        factor_name="TURNOVER",
        seed=20260829,
        n_stocks=3,
        n_dates=30,
        runtime_repeats=1,
    )

    assert payload["version"] == P3_MEASUREMENT_VERSION
    assert payload["random_seed"] == 20260829
    assert payload["sample"]["n_stocks"] == 3
    assert payload["sample"]["n_dates"] == 30
    sample_start = payload["sample"]["full_start_index"]
    assert sample_start == len(loader.dates) - 30

    parity = payload["parity"]
    assert parity["max_target_weight_diff"] == 0.0
    assert parity["max_buy_weight_diff"] == 0.0
    assert parity["max_sell_weight_diff"] == 0.0
    assert parity["max_order_count_diff"] == 0
    assert parity["max_cost_fraction_diff"] == 0.0

    labels = payload["labels"]
    assert labels["max_overlap_sessions"] == 0
    # P6 contract §2 grows the frequency table with every_20_days; the
    # P3 audit covers every registered cadence (monthly is calendar-gated
    # inside the policy itself and stays out of the audit).
    assert {row["frequency"] for row in labels["policies"]} == {
        "daily",
        "weekly",
        "every_5_days",
        "every_10_days",
        "every_20_days",
    }
    policies = {row["frequency"]: row for row in labels["policies"]}
    for frequency, stride in (
        ("every_5_days", 5),
        ("every_10_days", 10),
        ("every_20_days", 20),
    ):
        row = policies[frequency]
        assert all(index % stride == 0 for index in row["rebalance_indices"])
        assert row["sample_rebalance_indices"] == [
            index - sample_start for index in row["rebalance_indices"]
        ]

    default = payload["default_100k"]
    assert default["initial_order_count"] >= 0
    assert default["subsequent_average_order_count"] >= 0.0
    assert default["days_with_30_or_more_orders"] >= 0
    assert default["total_cost"] >= 0.0

    comparison = payload["pre_post"]
    assert comparison["invariants"]["same_signal"] is True
    assert comparison["invariants"]["same_fees"] is True
    assert comparison["pre_p3_compatible"]["runtime_seconds_median"] >= 0.0
    assert comparison["p3_default"]["runtime_seconds_median"] >= 0.0

    quadrant_rows = payload["bare_factor"]["quadrants"]
    assert len(quadrant_rows) == 4
    assert {
        (row["frequency"], row["horizon"], row["method"])
        for row in quadrant_rows
    } == {
        ("daily", 1, "equal_weight"),
        ("daily", 1, "optimizer"),
        ("weekly", 1, "equal_weight"),
        ("weekly", 1, "optimizer"),
    }
