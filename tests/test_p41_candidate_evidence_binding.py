"""P41 candidate-evidence binding RED/GREEN tests (WP 5B).

Each test asserts an independent invariant of the p41 preregistration
contract (``docs/p41_candidate_evidence_binding_draft.md`` §2/§5):

* a succeeded row whose benchmark series is missing or length-mismatched
  is an integrity error — never silently zero-padded excess returns (B1);
* the adjudicated candidate of the DSR / max-t result can only come from
  the current run's trials; historical trials count towards multiplicity
  but never lend their significance to another candidate (B3);
* the statistical result carries the five-part identity: adjudicated
  candidate, run kind, window, benchmark, trial-set composition (B2);
* promotion's significance gate enforces the binding between the
  statistical evidence and the artifact's top trial (B4).
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from ashare_model.eval_corrections import dsr_from_rows, max_t_from_rows
from ashare_model.eval_metrics import stitch_oos_series


def _iso_dates(n: int, start: str = "2026-01-01") -> list[str]:
    day = _dt.date.fromisoformat(start)
    return [(day + _dt.timedelta(days=i)).isoformat() for i in range(n)]


def _trial_row(
    candidate: str,
    daily: np.ndarray,
    dates: list[str],
    seed: int,
    bench: np.ndarray | None = None,
) -> dict:
    if bench is None:
        bench = np.zeros_like(daily)
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


def _current_noise_rows(n: int = 3, t: int = 250, seed: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    dates = _iso_dates(t)
    return [
        _trial_row(f"cur{i}", rng.normal(0.0, 0.01, size=t), dates, i)
        for i in range(n)
    ]


def _historical_strong_rows(t: int = 250) -> list[dict]:
    rng = np.random.default_rng(11)
    dates = _iso_dates(t)
    daily = 0.004 + rng.normal(0.0, 0.001, size=t)
    return [_trial_row("hist:strong", daily, dates, 99)]


# --- J1: benchmark missing / mismatched fails closed --------------------------


def test_j1_missing_benchmark_is_an_integrity_error():
    row = _trial_row(
        "a", np.random.default_rng(1).normal(0, 0.01, 30), _iso_dates(30), 0
    )
    del row["benchmark_daily_returns"]
    with pytest.raises(ValueError, match="benchmark"):
        stitch_oos_series([row])


def test_j1_mismatched_benchmark_length_is_an_integrity_error():
    row = _trial_row(
        "a", np.random.default_rng(2).normal(0, 0.01, 30), _iso_dates(30), 0
    )
    row["benchmark_daily_returns"] = [0.0] * 29
    with pytest.raises(ValueError, match="benchmark"):
        stitch_oos_series([row])


# --- J2/J3: historical significance is never granted to another candidate ------


def test_j2_dsr_adjudicates_a_current_candidate_and_counts_history():
    current = _current_noise_rows()
    hist = _historical_strong_rows()

    with_hist = dsr_from_rows(current, extra_trial_rows=hist)
    without_hist = dsr_from_rows(current)

    current_names = {r["candidate"] for r in current}
    # B3: the adjudicated trial is the best CURRENT trial — never the
    # historical one, no matter how strong the historical series is.
    assert with_hist["adjudicated_candidate"] in current_names
    assert with_hist["best_candidate"] == with_hist["adjudicated_candidate"]
    # The historical trial still enters the multiplicity count...
    assert with_hist["n_trials"] == 4
    assert with_hist["trial_set"]["n_current"] == 3
    assert with_hist["trial_set"]["n_extra"] == 1
    # ...which deflates the current candidate harder than a pool without
    # it, and the fact that history outranks the current run is disclosed.
    assert with_hist["dsr"] < without_hist["dsr"]
    assert with_hist["extra_outranks_current"] is True


def test_j3_max_t_observed_statistic_comes_from_current_trials_only():
    current = _current_noise_rows()
    hist = _historical_strong_rows()

    mt = max_t_from_rows(
        current, n_perms=500, seed=0, block_size=10, extra_trial_rows=hist
    )
    current_names = {r["candidate"] for r in current}
    assert mt["adjudicated_candidate"] in current_names
    # The historical trial's huge t-statistic must NOT become this run's
    # observed statistic: the current pool is noise, so the run's own
    # adjudication cannot be significant (history only widens the null,
    # it never supplies the observation).
    assert mt["significant_95"] is False
    assert mt["p_value"] > 0.05


# --- J4: the identity block is complete ---------------------------------------


def test_j4_identity_blocks_carry_window_benchmark_and_trial_set():
    current = _current_noise_rows()
    hist = _historical_strong_rows()
    dates = _iso_dates(250)

    dsr = dsr_from_rows(current, extra_trial_rows=hist)
    mt = max_t_from_rows(
        current, n_perms=300, seed=0, block_size=10, extra_trial_rows=hist
    )
    for result in (dsr, mt):
        assert result["adjudicated_candidate"] is not None
        assert result["adjudicated_seed"] is not None
        assert result["run"] == {"kind": "current_run"}
        assert result["window"]["n_dates"] >= 2
        assert result["window"]["first_date"] == dates[0]
        assert result["window"]["last_date"] == dates[-1]
        assert result["benchmark"]["definition"] == (
            "engine:equal_weight_universe"
        )
        assert result["benchmark"]["n_days"] == result["window"]["n_dates"]
        assert result["trial_set"]["n_trials"] == 4


# --- J5/J6: promotion enforces the binding ------------------------------------


def _bound_stats(candidate: str, seed: int) -> tuple[dict, dict]:
    """Passing statistical values bound to (candidate, seed)."""

    window = {
        "first_date": "2026-01-04",
        "last_date": "2026-12-30",
        "n_dates": 500,
    }
    benchmark = {
        "definition": "engine:equal_weight_universe",
        "n_days": 500,
        "first_date": "2026-01-04",
        "last_date": "2026-12-30",
    }
    trial_set = {"n_current": 1, "n_extra": 0, "n_trials": 1}
    dsr = {
        "dsr": 0.97,
        "n_trials": 1,
        "adjudicated_candidate": candidate,
        "adjudicated_seed": seed,
        "run": {"kind": "current_run"},
        "window": dict(window),
        "benchmark": dict(benchmark),
        "trial_set": dict(trial_set),
    }
    max_t = {
        "p_value": 0.03,
        "significant_95": True,
        "n_trials": 1,
        "adjudicated_candidate": candidate,
        "adjudicated_seed": seed,
        "run": {"kind": "current_run"},
        "window": dict(window),
        "benchmark": dict(benchmark),
        "trial_set": dict(trial_set),
    }
    return dsr, max_t


def test_j5_g2_fails_when_statistics_adjudicate_another_candidate(tmp_path):
    from test_promotion import (
        _future_regime,
        _paper_registry,
        _passing_stress,
        _strong_artifact,
        _verdict,
    )

    artifact = _strong_artifact()
    top = artifact["top_trial"]
    # The statistical evidence adjudicates a DIFFERENT candidate than the
    # artifact's top trial: the promotion's significance gate must refuse
    # (a historical trial's significance cannot be granted to the
    # challenger under review).
    artifact["dsr"], artifact["max_t"] = _bound_stats("someone_else", 7)
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["significance"]
    assert not gate["passed"]
    assert any("bound" in r or "adjudicated" in r for r in gate["reasons"])


def test_j5_g2_fails_without_identity_blocks(tmp_path):
    from test_promotion import (
        _future_regime,
        _paper_registry,
        _passing_stress,
        _strong_artifact,
        _verdict,
    )

    artifact = _strong_artifact()
    # Pre-p41 statistical blocks: values pass, identity absent — refuse.
    artifact["dsr"] = {"dsr": 0.97, "n_trials": 1}
    artifact["max_t"] = {"p_value": 0.03, "significant_95": True}
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["significance"]
    assert not gate["passed"]
    assert any("bound" in r or "identity" in r for r in gate["reasons"])


def test_j6_promotion_rule_version_is_3():
    from ashare_model.promotion import PROMOTION_RULE_VERSION

    assert PROMOTION_RULE_VERSION == "3"
