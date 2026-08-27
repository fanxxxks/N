"""Tests for the capital x positions x turnover fee-matrix report (P1-02).

Contract (docs/phase5_measurement_log.md §2, FEE_MATRIX_VERSION 1):

* One round trip of one position costs: buy commission + sell commission +
  stamp tax (sell side) + 2 x transfer fee + 2 x slippage, where each
  commission leg is ``max(per_order * commission_rate, min_commission)``
  (the minimum-commission floor applies per order, not per capital).
* Annual cost = round-trip cost x N x T with T the annual round trips per
  position (total annual traded notional = T x C).
* drag_pct = annual cost / capital x 100.
* Acceptability (pre-registered): drag_pct <= budget_pct (default 1.5).
* capacity(capital, turnover) = the largest N in the grid whose drag is
  within the budget (0 when none); recommended(capital) = the capacity
  cell at default_turnover (None when capacity is 0);
  feasible_structures = every within-budget cell.

Expected values below are derived by hand from the contract formula with
the default BacktestConfig fee schedule
(commission 0.00025 / min 5 CNY, stamp 0.0005 sell-side, transfer 0.00001,
slippage 0.0005) — they are NOT read from the implementation.
"""
from __future__ import annotations

import json

import pytest

from ashare_data.config import BacktestConfig
from ashare_model.cost_matrix import (
    FEE_MATRIX_VERSION,
    annual_drag,
    build_fee_matrix,
    round_trip_cost,
)

# Hand-derived per-position round-trip costs (default fee schedule).
ROUND_TRIP = {
    # (capital, positions) -> yuan per round trip
    (100_000, 5): 40.4,  # per_order 20000
    (100_000, 10): 25.2,  # per_order 10000 (commission floor binds)
    (100_000, 20): 17.6,  # per_order 5000 (commission floor binds)
    (200_000, 10): 40.4,  # per_order 20000
    (500_000, 10): 101.0,  # per_order 50000 (no floor; transfer 2 sides)
    (1_000_000, 10): 202.0,  # per_order 100000 (no floor; transfer 2 sides)
}


def test_round_trip_cost_matches_hand_derived_values():
    bt = BacktestConfig()
    for (capital, positions), expected in ROUND_TRIP.items():
        assert round_trip_cost(capital, positions, bt) == pytest.approx(
            expected, abs=1e-9
        ), (capital, positions)


def test_round_trip_cost_min_commission_floor_is_per_order():
    # Same per-order notional -> identical round trip, regardless of the
    # capital/position split that produced it.
    bt = BacktestConfig()
    assert round_trip_cost(100_000, 10, bt) == pytest.approx(
        round_trip_cost(200_000, 20, bt), abs=1e-12
    )
    # A per-order notional of 100000 pays 0.00025 x 100000 = 25 per leg
    # (floor not binding): the cost scales with notional.
    assert round_trip_cost(1_000_000, 10, bt) > round_trip_cost(500_000, 10, bt)


def test_annual_drag_hand_derived():
    bt = BacktestConfig()
    # (100k, N=10, T=6): 25.2 x 10 x 6 = 1512 CNY = 1.512% of capital.
    annual, pct = annual_drag(100_000, 10, 6, bt)
    assert annual == pytest.approx(1512.0, abs=1e-9)
    assert pct == pytest.approx(1.512, abs=1e-9)
    # (100k, N=5, T=6): 40.4 x 5 x 6 = 1212 CNY = 1.212%.
    annual, pct = annual_drag(100_000, 5, 6, bt)
    assert annual == pytest.approx(1212.0, abs=1e-9)
    assert pct == pytest.approx(1.212, abs=1e-9)


def test_annual_drag_component_breakdown():
    # (100k, N=10, T=6): commission 10 x 60 = 600; stamp 5 x 60 = 300;
    # transfer 0.2 x 60 = 12; slippage 10 x 60 = 600; total 1512.
    bt = BacktestConfig()
    annual, pct = annual_drag(100_000, 10, 6, bt)
    assert pct == pytest.approx(1.512, abs=1e-9)
    assert annual == pytest.approx(1512.0, abs=1e-9)


def test_drag_monotonicity():
    # Fewer positions, lower turnover and larger capital never increase
    # the drag percentage beyond floating noise.
    bt = BacktestConfig()
    for capital in (100_000, 200_000, 500_000):
        for turnover in (1, 2, 4, 6):
            low = annual_drag(capital, 5, turnover, bt)[1]
            high = annual_drag(capital, 50, turnover, bt)[1]
            assert low <= high + 1e-9
            lower_t = annual_drag(capital, 10, max(1, turnover // 2), bt)[1]
            assert lower_t <= annual_drag(capital, 10, turnover, bt)[1] + 1e-9


def test_build_fee_matrix_default_grid_hand_derived_capacity():
    bt = BacktestConfig()
    report = build_fee_matrix(
        capitals=[100_000, 200_000, 500_000],
        positions=[5, 10, 15, 20, 30, 50],
        turnovers=[1, 2, 4, 6, 12, 26],
        bt_cfg=bt,
        budget_pct=1.5,
        default_turnover=6,
    )
    # Capacity = largest N with drag <= 1.5% at each turnover (0 = none).
    expected = {
        "100000": {"1": 50, "2": 50, "4": 20, "6": 5, "12": 0, "26": 0},
        "200000": {"1": 50, "2": 50, "4": 30, "6": 15, "12": 0, "26": 0},
        "500000": {"1": 50, "2": 50, "4": 50, "6": 30, "12": 0, "26": 0},
    }
    assert report["capacity"] == expected
    # Recommended = capacity cell at the default turnover.
    assert report["recommended"]["100000"] == {
        "positions": 5,
        "turnover": 6,
        "drag_pct": pytest.approx(1.212, abs=1e-9),
        "annual_yuan": pytest.approx(1212.0, abs=1e-9),
    }
    assert report["recommended"]["200000"]["positions"] == 15
    assert report["recommended"]["500000"]["positions"] == 30


def test_build_fee_matrix_feasible_structures_nonempty_and_within_budget():
    bt = BacktestConfig()
    report = build_fee_matrix(
        capitals=[100_000, 200_000, 500_000],
        positions=[5, 10, 15, 20, 30, 50],
        turnovers=[1, 2, 4, 6, 12, 26],
        bt_cfg=bt,
        budget_pct=1.5,
        default_turnover=6,
    )
    structures = report["feasible_structures"]
    assert isinstance(structures, list) and len(structures) >= 10
    for cell in structures:
        assert cell["drag_pct"] <= 1.5 + 1e-9
        assert cell["capital"] in (100_000, 200_000, 500_000)
        assert cell["positions"] in (5, 10, 15, 20, 30, 50)
        assert cell["turnover"] in (1, 2, 4, 6, 12, 26)


def test_build_fee_matrix_cells_match_hand_derived_drag():
    bt = BacktestConfig()
    report = build_fee_matrix(
        capitals=[100_000],
        positions=[10, 20],
        turnovers=[6],
        bt_cfg=bt,
        budget_pct=1.5,
        default_turnover=6,
    )
    assert report["cells"]["100000"]["10"]["6"]["drag_pct"] == pytest.approx(
        1.512, abs=1e-9
    )
    assert report["cells"]["100000"]["20"]["6"]["drag_pct"] == pytest.approx(
        2.112, abs=1e-9
    )
    assert report["cells"]["100000"]["20"]["6"]["annual_yuan"] == pytest.approx(
        2112.0, abs=1e-9
    )


def test_build_fee_matrix_records_fee_params_and_version():
    bt = BacktestConfig()
    report = build_fee_matrix(
        capitals=[100_000], positions=[5], turnovers=[6], bt_cfg=bt
    )
    assert report["version"] == FEE_MATRIX_VERSION
    assert report["fee_params"]["commission_rate"] == 0.00025
    assert report["fee_params"]["min_commission"] == 5.0
    assert report["fee_params"]["stamp_tax_rate"] == 0.0005
    assert report["fee_params"]["transfer_fee_rate"] == 0.00001
    assert report["fee_params"]["slippage_rate"] == 0.0005
    assert report["grid"]["budget_pct"] == 1.5
    assert report["grid"]["default_turnover"] == 6


def test_build_fee_matrix_rejects_invalid_inputs():
    bt = BacktestConfig()
    with pytest.raises(ValueError):
        build_fee_matrix(
            capitals=[100_000], positions=[5], turnovers=[6], bt_cfg=bt,
            budget_pct=0.0,
        )
    with pytest.raises(ValueError):
        build_fee_matrix(
            capitals=[100_000], positions=[5], turnovers=[0], bt_cfg=bt
        )
    with pytest.raises(ValueError):
        build_fee_matrix(
            capitals=[0], positions=[5], turnovers=[6], bt_cfg=bt
        )


def test_cli_writes_fee_matrix_json(tmp_path):
    """CLI smoke: writes the versioned, JSON-serializable report."""
    from ashare_model.cost_matrix import main as cost_matrix_main

    out = tmp_path / "fee_matrix.json"
    code = cost_matrix_main(
        [
            "--config",
            "config/ashare_config.yaml",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == FEE_MATRIX_VERSION
    assert "capacity" in payload and "recommended" in payload
    assert "feasible_structures" in payload
