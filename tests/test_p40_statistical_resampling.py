"""P40 statistical resampling RED/GREEN tests (WP 5A).

Each test asserts an independent invariant of the p40 preregistration
contract (``docs/p40_statistical_resampling_draft.md`` §2/§5):

* rows/stitches carry the real mark dates (R1/R2, N1/N5);
* exact-duplicate trial columns must not move the max-t p-value (R4, N2) —
  under joint date-block draws duplicate columns produce point-identical
  bootstrap statistics, so their max equals the single-trial statistic;
* the adjudication window is the intersection of current-run trial date
  indexes and historical trials not covering it are ``matrix_excluded``
  yet still counted in the DSR multiplicity (R3, N3);
* the full-rotation degeneracy (block == n giving p = 1/(n_perms+1) on any
  positive-mean noise pool) is gone (R4, N4).

The resampling calibration (C1-C5) lives in
``tests/test_eval_resampling_calibration.py``.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, ModelConfig, ProtocolConfig

from ashare_model.data_loader import AshareDataLoader
from ashare_model.eval_corrections import (
    dsr_from_rows,
    max_t_from_rows,
)
from ashare_model.eval_folds import FoldConfig, resolve_folds
from ashare_model.eval_metrics import (
    benchmark_row,
    evaluate_signal,
    stitch_oos_series,
)


def _iso_dates(n: int, start: str = "2024-01-01") -> list[str]:
    day = _dt.date.fromisoformat(start)
    return [(day + _dt.timedelta(days=i)).isoformat() for i in range(n)]


def _noise_trial_row(
    candidate: str,
    daily: np.ndarray,
    dates: list[str],
    seed: int,
    fold_train_end: str,
) -> dict:
    return {
        "candidate": candidate,
        "failed": False,
        "seed": seed,
        "fold_train_end": fold_train_end,
        "fold_test_end": dates[-1],
        "val_reward": float(seed),
        "daily_returns": [float(x) for x in daily],
        "benchmark_daily_returns": [0.0] * len(daily),
        "daily_dates": list(dates),
    }


def _noise_rows(n: int, t: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    dates = _iso_dates(t)
    return [
        _noise_trial_row(
            f"noise{i}", rng.normal(0.0, 0.01, size=t), dates, i, dates[0]
        )
        for i in range(n)
    ]


# --- N1: stitches carry the real concatenated date index ----------------------


def test_n1_stitch_carries_concatenated_real_dates():
    t = 30
    dates = _iso_dates(t)
    rng = np.random.default_rng(3)
    rows = [
        _noise_trial_row("a", rng.normal(0, 0.01, 10), dates[:10], 0, dates[0]),
        _noise_trial_row(
            "a", rng.normal(0, 0.01, 20), dates[10:], 0, dates[9]
        ),
    ]
    trials = stitch_oos_series(rows)
    assert len(trials) == 1
    trial = trials[0]
    # R2: the trial records its real date index, in fold order, aligned
    # 1:1 with every series.
    assert trial["dates"] == dates
    assert len(trial["dates"]) == len(trial["daily_returns"])
    assert len(trial["dates"]) == len(trial["benchmark_daily_returns"])
    assert len(trial["dates"]) == len(trial["excess_returns"])


def test_n1_undated_succeeded_row_is_an_integrity_error():
    t = 20
    row = _noise_trial_row(
        "a", np.random.default_rng(1).normal(0, 0.01, t), _iso_dates(t), 0, "d"
    )
    del row["daily_dates"]  # a pre-p40 artifact row
    with pytest.raises(ValueError, match="daily_dates"):
        stitch_oos_series([row])


def test_n1_duplicate_date_within_a_trial_is_an_integrity_error():
    t = 10
    dates = _iso_dates(t)
    dates[-1] = dates[-2]  # duplicated mark date
    row = _noise_trial_row(
        "a", np.random.default_rng(2).normal(0, 0.01, t), dates, 0, dates[0]
    )
    with pytest.raises(ValueError, match="duplicate date"):
        stitch_oos_series([row])


# --- N2: exact-duplicate columns must not move the p-value ---------------------


def test_n2_duplicate_trials_do_not_change_p_value():
    t = 200
    rng = np.random.default_rng(11)
    daily = rng.normal(0.0, 0.01, size=t)
    dates = _iso_dates(t)
    single = [_noise_trial_row("solo", daily, dates, 0, dates[0])]
    duplicated = single + [
        _noise_trial_row("solo_copy", daily, dates, 1, dates[0])
    ]
    p_single = max_t_from_rows(single, n_perms=500, seed=0, block_size=10)
    p_dup = max_t_from_rows(duplicated, n_perms=500, seed=0, block_size=10)
    # R4 joint draw: identical columns give point-identical bootstrap
    # statistics, so the max — and therefore the p-value — is unchanged.
    assert p_dup["p_value"] == p_single["p_value"]


# --- N3: window intersection + matrix_excluded historical trials ---------------


def test_n3_window_is_intersection_and_history_is_excluded_but_counted():
    t = 100
    dates = _iso_dates(t)
    rng = np.random.default_rng(5)
    # Trial A covers the full window; trial B lacks dates[60:80] (a failed
    # fold in the middle).  W = A ∩ B has 80 dates.
    rows = [
        _noise_trial_row(
            "a", rng.normal(0, 0.01, t), dates, 0, dates[0]
        ),
        # Trial B, fold 1: the first 60 dates.
        _noise_trial_row(
            "b", rng.normal(0, 0.01, 60), dates[:60], 1, dates[0]
        ),
        # Trial B, fold 2: resumes after the failed fold, dates[80:100].
        {
            "candidate": "b",
            "failed": False,
            "seed": 1,
            "fold_train_end": dates[60],
            "fold_test_end": dates[99],
            "val_reward": 1.0,
            "daily_returns": [float(x) for x in rng.normal(0, 0.01, 20)],
            "benchmark_daily_returns": [0.0] * 20,
            "daily_dates": dates[80:],
        },
        # The failed middle fold itself: recorded, never stitched.
        {
            "candidate": "b",
            "failed": True,
            "seed": 1,
            "fold_train_end": dates[59],
            "fold_test_end": dates[79],
            "val_reward": None,
        },
    ]

    # A historical trial measured one day later: it cannot cover W.
    shifted = _noise_trial_row(
        "hist", rng.normal(0, 0.01, t), _iso_dates(t, start="2024-01-02"), 7,
        "2024-01-02",
    )
    mt = max_t_from_rows(rows, n_perms=300, seed=0, block_size=10,
                         extra_trial_rows=[shifted])
    assert mt["window"]["n_dates"] == 80
    assert mt["window"]["first_date"] == dates[0]
    assert mt["window"]["last_date"] == dates[-1]
    excluded = mt["matrix_excluded"]
    assert len(excluded) == 1
    assert excluded[0]["candidate"] == "hist"
    assert excluded[0]["reason"] == "window_mismatch"
    assert mt["n_current"] == 2
    assert mt["n_extra"] == 1

    # The historical trial still counts towards the DSR multiplicity.
    dsr = dsr_from_rows(rows, extra_trial_rows=[shifted])
    assert dsr["n_trials"] == 3


def test_n3_disjoint_current_trials_are_an_integrity_error():
    t = 10
    rows = [
        _noise_trial_row(
            "a", np.random.default_rng(6).normal(0, 0.01, t), _iso_dates(t), 0,
            "2024-01-01",
        ),
        _noise_trial_row(
            "b", np.random.default_rng(7).normal(0, 0.01, t),
            _iso_dates(t, start="2024-06-01"), 1, "2024-06-01",
        ),
    ]
    with pytest.raises(ValueError, match="common dates"):
        max_t_from_rows(rows, n_perms=100, seed=0, block_size=5)


# --- N4: the small-sample degeneracy is gone (fail closed below the floor) ---


def test_n4_window_below_minimum_refuses_instead_of_garbage_p():
    t = 6
    rng = np.random.default_rng(100)
    rows = [
        _noise_trial_row(
            f"s{i}", rng.normal(0.0, 0.01, size=t), _iso_dates(t), i,
            "2024-01-01",
        )
        for i in range(4)
    ]
    # Old behaviour: block degenerates to n, every bootstrap mean is the
    # centered mean and p == 1/(n_perms+1) on any noise pool.  p40: a
    # window this small cannot calibrate the max-t tail, so the test
    # refuses outright (no p-value at all).
    with pytest.raises(ValueError, match="P40_MIN_WINDOW_DATES"):
        max_t_from_rows(rows, n_perms=500, seed=0, block_size=10)


def test_n4_minimum_window_noise_is_mostly_insignificant():
    t = 30
    insignificant = 0
    reps = 60
    for seed in range(reps):
        rng = np.random.default_rng(200 + seed)
        rows = [
            _noise_trial_row(
                f"s{i}", rng.normal(0.0, 0.01, size=t), _iso_dates(t), i,
                "2024-01-01",
            )
            for i in range(4)
        ]
        mt = max_t_from_rows(rows, n_perms=300, seed=0, block_size=10)
        assert mt["block_size"] <= max(1, t // 6)
        insignificant += mt["p_value"] > 0.05
    # At the minimum legal window the measured size is ~0.08-0.13
    # (contract §2 R4 erratum 2); a rate below 0.75 would signal the old
    # always-significant degeneracy returning.
    assert insignificant >= int(reps * 0.75), insignificant


# --- N5: the two row producers agree on the real mark dates --------------------


def test_n5_evaluate_signal_and_benchmark_row_carry_mark_dates(
    populated_db,
):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    proto = ProtocolConfig(
        folds=[FoldConfig("2024-01-10", "2024-01-25")]
    )
    fold = resolve_folds(
        [proto.folds[0]],
        loader.dates,
        frequency=proto.frequency,
        horizon=proto.horizon,
    )[0]
    from ashare_model.eval_folds import epoch_slice

    _, _, _, dates = epoch_slice(loader, fold)
    rng = np.random.default_rng(1234)
    signal = rng.normal(0.0, 1.0, size=(len(loader.ts_codes), len(dates)))
    metrics = evaluate_signal(signal, loader, fold, BacktestConfig())
    row = benchmark_row(loader, fold)

    # R1: both row producers carry the engine mark dates, aligned 1:1
    # with their daily series — the same engine date authority.
    assert "daily_dates" in metrics
    assert len(metrics["daily_dates"]) == len(metrics["daily_returns"])
    assert "daily_dates" in row
    assert len(row["daily_dates"]) == len(row["daily_returns"])
    assert row["daily_dates"] == metrics["daily_dates"]
    assert len(set(metrics["daily_dates"])) == len(metrics["daily_dates"])
