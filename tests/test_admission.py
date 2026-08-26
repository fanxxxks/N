"""T2-03 contracts: the RL admission decision logic.

The pre-registered rule (also recorded in docs/phase2_measurement_log.md):
RL is admitted only if it beats random *and* GP on both the normalized
best-so-far area and the OOS active IR — median strictly better and at
least 4 of 5 seeds won on each metric — under identical
unique-semantic-evaluation budgets.
"""

from __future__ import annotations

import math

import pytest

from ashare_model.admission import (
    ADMISSION_BATCH,
    ADMISSION_STEPS,
    INIT_SEEDS,
    WIN_FRACTION,
    best_so_far_area,
    best_so_far_curve_values,
    decide_admission,
    default_searcher,
)


def test_init_seeds_are_at_least_five_independent_values():
    assert len(set(INIT_SEEDS)) >= 5
    assert ADMISSION_STEPS > 0 and ADMISSION_BATCH > 0
    assert 0.5 < WIN_FRACTION <= 1.0


def test_best_so_far_curve_values_is_a_step_function():
    curve = [(1.0, 0.1), (3.0, 0.4), (5.0, 0.5)]
    values = best_so_far_curve_values(curve, 6)
    assert values == [0.1, 0.1, 0.4, 0.4, 0.5, 0.5]
    # A curve that never reaches a budget point keeps the last value.
    assert best_so_far_curve_values([(2.0, 0.3)], 4) == [-math.inf, 0.3, 0.3, 0.3]


def test_best_so_far_area_normalizes_by_budget():
    flat = [(b, 0.5) for b in range(1, 11)]
    assert best_so_far_area(flat, 10) == pytest.approx(0.5)
    late = [(1.0, 0.0), (10.0, 1.0)]
    # Average of the step function over 1..10: 0.1 (one point at 1.0).
    assert best_so_far_area(late, 10) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        best_so_far_area([], 0)


def test_decide_admission_rl_dominates_both_baselines():
    rl_areas = [0.5, 0.6, 0.7, 0.8, 0.9]
    rl_irs = [0.9, 1.0, 1.1, 1.2, 1.3]
    weak = [0.3, 0.3, 0.3, 0.3, 0.3]
    verdict = decide_admission(
        rl_areas=rl_areas,
        rl_oos_irs=rl_irs,
        baseline_areas={"random": weak, "gp": weak},
        baseline_oos_irs={"random": weak, "gp": weak},
    )
    assert verdict["admitted"] is True
    assert verdict["default_searcher"] == "rl"
    assert default_searcher(verdict) == "rl"
    assert verdict["evidence"]["random"]["wins_area"] == 5
    assert verdict["evidence"]["gp"]["wins_oos_ir"] == 5


def test_decide_admission_rejects_when_one_metric_loses():
    # RL wins the area but loses the OOS IR median: not admitted.
    rl_areas = [0.5, 0.6, 0.7, 0.8, 0.9]
    rl_irs = [0.5, 0.6, 0.7, 0.8, 0.9]
    baseline_areas = [0.3, 0.3, 0.3, 0.3, 0.3]
    baseline_irs = [1.5, 1.5, 1.5, 1.5, 1.5]
    verdict = decide_admission(
        rl_areas=rl_areas,
        rl_oos_irs=rl_irs,
        baseline_areas={"random": baseline_areas},
        baseline_oos_irs={"random": baseline_irs},
    )
    assert verdict["admitted"] is False
    assert verdict["default_searcher"] == "gp"


def test_decide_admission_rejects_when_seed_wins_insufficient():
    # Median better but only 3 of 5 seeds won: below the 4/5 bar.
    rl_areas = [0.9, 0.1, 0.9, 0.1, 0.9]
    baseline_areas = [0.2, 0.8, 0.2, 0.8, 0.2]
    rl_irs = [0.9, 0.1, 0.9, 0.1, 0.9]
    baseline_irs = [0.2, 0.8, 0.2, 0.8, 0.2]
    verdict = decide_admission(
        rl_areas=rl_areas,
        rl_oos_irs=rl_irs,
        baseline_areas={"random": baseline_areas},
        baseline_oos_irs={"random": baseline_irs},
    )
    assert verdict["admitted"] is False
    assert verdict["evidence"]["random"]["wins_area"] == 3
    assert verdict["evidence"]["random"]["min_wins"] == 4


def test_decide_admission_requires_aligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        decide_admission(
            rl_areas=[0.5, 0.6],
            rl_oos_irs=[0.5],
            baseline_areas={"random": [0.5]},
            baseline_oos_irs={"random": [0.5]},
        )
