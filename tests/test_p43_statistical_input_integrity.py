"""P43 statistical-input/numerics/evidence-binding RED/GREEN tests (WP 7).

Each test asserts an independent invariant of the p43 preregistration
contract (``docs/p43_statistical_input_integrity_draft.md`` §2/§5):

* non-finite values never enter significance computation — one NaN in a
  daily/benchmark series or in the joint matrix is an integrity error,
  never the minimum p-value 1/(n_perms+1) (N1/N2);
* the resampling parameters are legality-checked — non-bool integers
  with n_perms >= 1 and block_size >= 1, no silent float truncation (N3);
* the producer's value ranges hold — p in (0,1], significant_95 computed
  from p against the threshold (N4, pinned by the positive controls);
* promotion G2 adjudicates by the numeric threshold, requires finite
  in-range values and refuses a significant_95 that contradicts its own
  p-value (N5);
* G2 verifies the *content* of the statistical identity — run kind,
  window vs the top trial's own date axis, benchmark definition/span,
  trial-set composition vs the artifact's own stitched evidence, and the
  seed class demanded by the candidate family (B5–B9).
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from ashare_model.eval_corrections import (
    dsr_from_rows,
    max_t_from_matrix,
    max_t_from_rows,
)
from ashare_model.eval_metrics import stitch_oos_series


def _iso_dates(n: int, start: str = "2024-01-01") -> list[str]:
    day = _dt.date.fromisoformat(start)
    return [(day + _dt.timedelta(days=i)).isoformat() for i in range(n)]


def _trial_row(
    candidate: str,
    daily: np.ndarray | list[float],
    dates: list[str],
    seed: int,
    bench: np.ndarray | list[float] | None = None,
) -> dict:
    if bench is None:
        bench = np.zeros_like(np.asarray(daily, dtype=np.float64))
    return {
        "candidate": candidate,
        "failed": False,
        "seed": seed,
        "fold_train_end": dates[0],
        "fold_test_end": dates[-1],
        "val_reward": float(seed),
        "daily_returns": [float(x) for x in daily],
        "benchmark_daily_returns": [float(x) for x in bench],
        "daily_dates": list(dates),
    }


def _noise_rows(n: int = 4, t: int = 250, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    dates = _iso_dates(t)
    return [
        _trial_row(f"s{i}", rng.normal(0.0, 0.01, size=t), dates, i)
        for i in range(n)
    ]


def _with_value(rows: list[dict], series: str, index: int, value: float):
    """A copy of ``rows`` with one cell of one row's series replaced."""

    copy = [dict(r) for r in rows]
    copy[0][series] = list(copy[0][series])
    copy[0][series][index] = value
    return copy


# --- N1: non-finite returns never enter the statistics -------------------------


def test_n1a_one_nan_daily_return_is_an_integrity_error():
    rows = _noise_rows()
    poisoned = _with_value(rows, "daily_returns", 7, float("nan"))
    # The p43 counterexample: under the old behaviour one NaN froze every
    # bootstrap comparison and returned the minimum p-value 1/(n_perms+1).
    with pytest.raises(ValueError, match="non-finite"):
        stitch_oos_series(poisoned)
    with pytest.raises(ValueError, match="non-finite"):
        max_t_from_rows(poisoned, n_perms=500, seed=0, block_size=10)
    with pytest.raises(ValueError, match="non-finite"):
        dsr_from_rows(poisoned)


def test_n1b_one_nan_or_inf_benchmark_is_an_integrity_error():
    rows = _noise_rows()
    nan_bench = _with_value(rows, "benchmark_daily_returns", 3, float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        stitch_oos_series(nan_bench)
    inf_daily = _with_value(rows, "daily_returns", 11, float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        stitch_oos_series(inf_daily)


def test_n1c_clean_finite_rows_still_produce_statistics():
    rows = _noise_rows()
    mt = max_t_from_rows(rows, n_perms=500, seed=0, block_size=10)
    assert 0.0 < mt["p_value"] <= 1.0
    assert mt["significant_95"] == (mt["p_value"] <= 0.05)
    dsr = dsr_from_rows(rows)
    assert 0.0 <= dsr["dsr"] <= 1.0


# --- N2: the joint matrix itself is finite-checked -----------------------------


def test_n2_one_nan_in_matrix_refuses_instead_of_minimum_p():
    rng = np.random.default_rng(0)
    m = rng.normal(0.0, 0.01, size=(4, 250))
    ok = max_t_from_matrix(m, n_perms=500, seed=0, block_size=10)
    assert 0.0 < ok["p_value"] <= 1.0
    bad = m.copy()
    bad[0, 7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        max_t_from_matrix(bad, n_perms=500, seed=0, block_size=10)
    bad_inf = m.copy()
    bad_inf[1, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        max_t_from_matrix(bad_inf, n_perms=500, seed=0, block_size=10)


# --- N3: resampling parameter legality -----------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_perms": 0},
        {"n_perms": -5},
        {"n_perms": 500.5},
        {"n_perms": True},
        {"n_perms": "500"},
        {"block_size": 0},
        {"block_size": -3},
        {"block_size": 2.5},
        {"block_size": True},
        {"seed": None},
        {"seed": 0.5},
        {"seed": "0"},
    ],
    ids=[
        "n_perms-zero", "n_perms-negative", "n_perms-float", "n_perms-bool",
        "n_perms-str", "block-zero", "block-negative", "block-float",
        "block-bool", "seed-none", "seed-float", "seed-str",
    ],
)
def test_n3_illegal_resampling_parameters_refuse(kwargs):
    rng = np.random.default_rng(1)
    m = rng.normal(0.0, 0.01, size=(4, 60))
    with pytest.raises(ValueError):
        max_t_from_matrix(m, **kwargs)


def test_n3_legal_integer_parameters_run():
    rng = np.random.default_rng(2)
    m = rng.normal(0.0, 0.01, size=(4, 60))
    mt = max_t_from_matrix(m, n_perms=10, seed=3, block_size=5)
    assert 0.0 < mt["p_value"] <= 1.0
    assert mt["n_perms"] == 10


# --- N5: promotion G2 adjudicates by the numeric threshold ---------------------


def _artifact():
    from test_promotion import _strong_artifact

    return _strong_artifact()


def _significance_gate(artifact):
    # WP17 re-anchor (FROZEN_17 §7.2): G2's positive control now needs
    # the calibration-qualification layer — the helper injects the same
    # in-memory evidence registry as test_promotion._calibration_evidence
    # over the (re-anchored) in-domain fixture (n=250, k=8, n_perms=999),
    # preserving this module's "numeric adjudication positive control"
    # purpose.
    from test_promotion import _calibration_evidence, _verdict

    verdict = _verdict(artifact, calibration_evidence=_calibration_evidence())
    return verdict["gates"]["significance"]


def test_n5_g2_rejects_nonfinite_dsr():
    artifact = _artifact()
    artifact["dsr"]["dsr"] = float("nan")
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("not a finite number" in r for r in gate["reasons"])


def test_n5_g2_rejects_nonfinite_p_value():
    artifact = _artifact()
    artifact["max_t"]["p_value"] = float("nan")
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("not a finite number" in r for r in gate["reasons"])


def test_n5_g2_rejects_contradictory_p_and_significant_flag():
    artifact = _artifact()
    # The p43 counterexample: p=0.90 with significant_95=True passed G2.
    artifact["max_t"]["p_value"] = 0.90
    artifact["max_t"]["significant_95"] = True
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("contradicts" in r for r in gate["reasons"])


def test_n5_g2_rejects_false_negative_contradiction_too():
    artifact = _artifact()
    # p=0.01 claims significance numerically but the flag says False:
    # contradictory evidence is refused in both directions.
    artifact["max_t"]["p_value"] = 0.01
    artifact["max_t"]["significant_95"] = False
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("contradicts" in r for r in gate["reasons"])


def test_n5_g2_rejects_out_of_range_values():
    artifact = _artifact()
    artifact["dsr"]["dsr"] = 1.2
    reasons = _significance_gate(artifact)["reasons"]
    assert any("outside" in r and "dsr" in r.lower() for r in reasons)

    artifact = _artifact()
    artifact["max_t"]["p_value"] = 1.5
    reasons = _significance_gate(artifact)["reasons"]
    assert any("outside" in r and "p_value" in r for r in reasons)

    artifact = _artifact()
    artifact["max_t"]["p_value"] = 0.0
    reasons = _significance_gate(artifact)["reasons"]
    assert any("outside" in r and "p_value" in r for r in reasons)


def test_n5_g2_rejects_non_boolean_significant_flag():
    artifact = _artifact()
    artifact["max_t"]["significant_95"] = "yes"
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("must be a boolean" in r for r in gate["reasons"])


def test_n5_g2_passes_consistent_finite_statistics():
    # Positive control: the untouched build_result artifact (finite inputs,
    # self-consistent blocks) still passes the significance gate.
    gate = _significance_gate(_artifact())
    assert gate["passed"], gate["reasons"]


# --- B5-B9: promotion G2 verifies the identity CONTENT ------------------------
#
# These mutate one identity aspect at a time while keeping the candidate/
# seed binding of p41 B4 intact, so each failure is attributable to the
# content check alone (docs/p43_statistical_input_integrity_draft.md §2).


def _blocks(artifact):
    return artifact["dsr"], artifact["max_t"]


def test_b5_g2_rejects_non_current_run_kind():
    artifact = _artifact()
    for block in _blocks(artifact):
        block["run"] = {"kind": "engineering_run"}
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("run kind" in r for r in gate["reasons"])


def test_b6_g2_rejects_dsr_window_not_matching_top_trial_axis():
    artifact = _artifact()
    top_dates = artifact["top_trial"]["dates"]
    window = artifact["dsr"]["window"]
    assert window["first_date"] == top_dates[0]  # the genuine block binds
    # same span endpoints, wrong length: 499 != 500
    window["n_dates"] = len(top_dates) - 1
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("window" in r for r in gate["reasons"])

    window["n_dates"] = len(top_dates)  # restore, break the first date
    window["first_date"] = "1999-01-01"
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("window" in r for r in gate["reasons"])


def test_b6_g2_rejects_max_t_window_outside_top_trial_axis():
    artifact = _artifact()
    top_dates = artifact["top_trial"]["dates"]
    artifact["max_t"]["window"] = {
        "first_date": "1999-01-01",
        "last_date": top_dates[-1],
        "n_dates": len(top_dates),
    }
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("window" in r for r in gate["reasons"])


def test_b6_g2_rejects_max_t_window_below_p40_floor():
    artifact = _artifact()
    top_dates = artifact["top_trial"]["dates"]
    # A window the producer refuses to run on (n < P40_MIN_WINDOW_DATES):
    # a block claiming one cannot be genuine.
    artifact["max_t"]["window"] = {
        "first_date": top_dates[0],
        "last_date": top_dates[4],
        "n_dates": 5,
    }
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("window" in r for r in gate["reasons"])


def test_b7_g2_rejects_wrong_benchmark_definition():
    artifact = _artifact()
    for block in _blocks(artifact):
        block["benchmark"]["definition"] = "csi300_index"
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("benchmark" in r for r in gate["reasons"])


def test_b7_g2_rejects_benchmark_span_not_matching_window():
    artifact = _artifact()
    artifact["max_t"]["benchmark"]["n_days"] = 3  # window says 500
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("benchmark" in r for r in gate["reasons"])


def test_b8_g2_rejects_trial_set_not_matching_artifact():
    artifact = _artifact()
    # The adjudicated candidate name is intact; only the trial-set
    # composition points somewhere else (the p43 counterexample shape).
    for block in _blocks(artifact):
        block["trial_set"] = {"n_current": 99, "n_extra": 99, "n_trials": 198}
        block["n_trials"] = 198
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("trial" in r for r in gate["reasons"])


def test_b8_g2_rejects_incoherent_trial_set_arithmetic():
    artifact = _artifact()
    artifact["max_t"]["trial_set"] = {
        "n_current": 1, "n_extra": 1, "n_trials": 5,
    }
    artifact["max_t"]["n_trials"] = 5
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("trial" in r for r in gate["reasons"])


def test_b8_g2_rejects_zero_current_trials():
    artifact = _artifact()
    for block in _blocks(artifact):
        block["trial_set"] = {"n_current": 0, "n_extra": 0, "n_trials": 0}
        block["n_trials"] = 0
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("trial" in r for r in gate["reasons"])


# --- B9: the seed class is decided by the candidate family ---------------------


def test_b9_seed_class_by_candidate_family():
    from ashare_model.promotion import candidate_seed_class

    deterministic = [
        "benchmark:equal_weight",
        "baseline:RET_1",
        "selfcheck:noise",
        "control:cash",
        "combo:all3",
        "n10_m2500::combo:fitted_lam050",
        "n10_m2500::baseline:RSQ_60",
    ]
    seeded = [
        "trained",
        "random_search",
        "gp_search",
        "tpe_search",
        "rl:1,2,3",
        "challenger",  # unknown family defaults to the stricter class
    ]
    for name in deterministic:
        assert candidate_seed_class(name) == "deterministic", name
    for name in seeded:
        assert candidate_seed_class(name) == "seeded", name


def _deterministic_artifact():
    """A strong artifact whose top trial is a deterministic baseline arm.

    WP17 re-anchor (FROZEN_17 §3.1 + the §7.2 row-5 pattern): the G2
    positive path now also needs the calibration-qualification layer,
    so the fixture sits at the calibrated operating point — 8 current
    trials (the deterministic baseline arm plus 7 deterministic noise
    pool arms, all seed=None), a 250-common-date stitched axis
    (2 folds x 125 sessions) and max_t_perms=999; _significance_gate
    injects the in-memory covering evidence registry."""

    from ashare_data.config import FoldConfig, ProtocolConfig, TierConfig
    from ashare_model.evaluation import build_result
    from test_stitched_oos import _row

    rng = np.random.default_rng(7)
    rows = []
    for train_end, test_end in [
        ("2026-01-01", "2026-06-30"),
        ("2026-06-30", "2026-12-31"),
    ]:
        daily = [0.004 + x for x in rng.normal(0.0, 0.001, 125)]
        rows.append(
            _row(
                "baseline:RET_1",
                None,
                train_end,
                test_end,
                daily=daily,
                formula_text="RET_1",
                eligible=True,
                rejection_reasons=[],
                direction=1,
                formula=None,
                average_turnover=0.05,
                capacity_utilization=0.05,
            )
        )
    for i in range(7):  # the multiplicity pool (deterministic noise arms)
        for train_end, test_end in [
            ("2026-01-01", "2026-06-30"),
            ("2026-06-30", "2026-12-31"),
        ]:
            daily = [x for x in rng.normal(0.0, 0.001, 125)]
            rows.append(
                _row(
                    f"baseline:pool_{i}",
                    None,
                    train_end,
                    test_end,
                    daily=daily,
                    formula_text="RET_1",
                    eligible=True,
                    rejection_reasons=[],
                    direction=1,
                    formula=None,
                    average_turnover=0.05,
                    capacity_utilization=0.05,
                )
            )
    proto = ProtocolConfig(
        folds=[
            FoldConfig("2026-01-01", "2026-06-30"),
            FoldConfig("2026-06-30", "2026-12-31"),
        ],
        seeds=[42],
        baseline_signals=[],
        random_samples=0,
    )
    return build_result(
        proto, "confirmation", TierConfig(50, 256), rows,
        spec_id="a" * 64,
        run_id="b" * 32,
        data_end_date="2026-12-31",
        dataset_id="ds-current",
        max_t_perms=999,
    )


def test_b9_deterministic_candidate_binds_with_seed_none():
    # seed=None on a deterministic family is the EXPECTED identity, not an
    # accidental None==None match: the genuine build_result blocks bind and
    # the significance gate adjudicates on values.
    artifact = _deterministic_artifact()
    assert artifact["top_trial"]["candidate"] == "baseline:RET_1"
    assert artifact["top_trial"]["seed"] is None
    assert artifact["dsr"]["adjudicated_seed"] is None
    gate = _significance_gate(artifact)
    assert gate["passed"], gate["reasons"]


def test_b9_g2_rejects_fabricated_seed_on_deterministic_candidate():
    # A seed=7 fabricated into the top trial AND both blocks satisfies the
    # p41 equality check; only the candidate-family class check catches it.
    artifact = _deterministic_artifact()
    artifact["top_trial"]["seed"] = 7
    for block in _blocks(artifact):
        block["adjudicated_seed"] = 7
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("seed" in r for r in gate["reasons"])


def test_b9_g2_rejects_none_seed_on_seeded_candidate():
    # "challenger" is an unknown family -> seeded class: a None seed
    # (e.g. wiped to make binding trivially satisfiable) is refused.
    artifact = _artifact()
    artifact["top_trial"]["seed"] = None
    for block in _blocks(artifact):
        block["adjudicated_seed"] = None
    gate = _significance_gate(artifact)
    assert not gate["passed"]
    assert any("seed" in r for r in gate["reasons"])
