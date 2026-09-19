"""P42 R03 combination-weight panel RED/GREEN tests (WP 6).

Each test asserts an independent invariant of the p42 preregistration
contract (``docs/p42_r03_combo_weights_draft.md`` §3/§5):

* the weighted combination base is the p34 rank construction with weights
  — equal weights must reproduce ``combo_signal`` exactly (N1);
* the frozen shrinkage fit rule with hand-computed weights, negative
  clipping and the equal-weight fallback (N2);
* the four-way adjudication on synthetic summaries, reusing the p37
  ``_arm_verdict`` rules as one implementation, descriptive lambda arms
  excluded (N3);
* the panel driver structure: composite row keys, per-fold direction and
  weight learning on the reference cell frozen across cells, singles
  measured only on the reference-cell direction path (N4, mini e2e);
* the contract-sha refusal path (N5).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ashare_model.r01_baseline import combo_signal, rank_normalize_signal


def _members():
    """Two direction-signed member signals on a 3-stock x 4-day grid."""

    m1 = np.array(
        [
            [0.3, 0.1, np.nan, 0.2],
            [0.1, 0.3, 0.05, 0.1],
            [0.2, 0.2, 0.1, 0.3],
        ]
    )
    m2 = np.array(
        [
            [0.05, 0.2, 0.3, 0.1],
            [0.3, 0.1, 0.2, 0.2],
            [0.1, 0.3, 0.1, 0.05],
        ]
    )
    mask = np.ones((3, 4), dtype=bool)
    return [(1.0, m1), (-1.0, m2)], mask


# --- N1: the weighted combination base ----------------------------------------


def test_n1_equal_weights_reproduce_combo_signal_exactly():
    from ashare_model.r03_combos import weighted_combo_signal

    members, mask = _members()
    equal = weighted_combo_signal(members, mask, [0.5, 0.5])
    reference = combo_signal(members, mask)
    np.testing.assert_array_equal(equal, reference)


def test_n1_weighted_mean_and_missing_member_renormalized():
    from ashare_model.r03_combos import weighted_combo_signal

    members, mask = _members()
    # Weight 1 on member 0: the combo is member 0's own normalized signal
    # (direction applied), NaN where member 0 is NaN.
    only0 = weighted_combo_signal(members, mask, [1.0, 0.0])
    expected0 = rank_normalize_signal(1.0 * members[0][1], mask)
    np.testing.assert_array_equal(only0, expected0)

    # Day 2 (index 2): member 0 is NaN there, so a 0.5/0.5 combo must fall
    # back to member 1's own normalized value on that day (the available
    # member's renormalized weight is 1).
    both = weighted_combo_signal(members, mask, [0.5, 0.5])
    expected1_day2 = rank_normalize_signal(-1.0 * members[1][1], mask)[0:3, 2]
    np.testing.assert_allclose(both[0:3, 2], expected1_day2)


def test_n1_weights_must_be_a_simplex():
    from ashare_model.r03_combos import weighted_combo_signal

    members, mask = _members()
    with pytest.raises(ValueError, match="weight"):
        weighted_combo_signal(members, mask, [0.7, 0.5])
    with pytest.raises(ValueError, match="weight"):
        weighted_combo_signal(members, mask, [1.2, -0.2])
    with pytest.raises(ValueError, match="member"):
        weighted_combo_signal(members, mask, [1.0])


# --- N2: the frozen shrinkage fit rule -----------------------------------------


def test_n2_fit_weights_hand_computed_with_clipping():
    from ashare_model.r03_combos import fit_combo_weights

    scores = {"MOMENTUM_20": 0.10, "RSQ_60": -0.05, "REVERSAL_60": 0.30}
    out = fit_combo_weights(scores, lam=0.5)
    # Positive part: 0.10 / 0.30 -> shares 0.25 / 0.75 (RSQ clipped to 0).
    # w = (1-lam)/3 + lam * share.
    assert out["weights"]["MOMENTUM_20"] == pytest.approx(
        0.5 / 3 + 0.5 * 0.25
    )
    assert out["weights"]["RSQ_60"] == pytest.approx(0.5 / 3)
    assert out["weights"]["REVERSAL_60"] == pytest.approx(
        0.5 / 3 + 0.5 * 0.75
    )
    assert sum(out["weights"].values()) == pytest.approx(1.0)
    assert out["fallback"] is False
    assert out["lam"] == 0.5
    assert out["inputs"] == scores


def test_n2_all_nonpositive_scores_fall_back_to_equal_weights():
    from ashare_model.r03_combos import fit_combo_weights

    out = fit_combo_weights(
        {"MOMENTUM_20": -0.1, "RSQ_60": 0.0, "REVERSAL_60": -0.3}, lam=0.75
    )
    assert out["weights"] == pytest.approx(
        {"MOMENTUM_20": 1 / 3, "RSQ_60": 1 / 3, "REVERSAL_60": 1 / 3}
    )
    assert out["fallback"] is True


def test_n2_lambda_zero_is_equal_weight_no_fallback():
    from ashare_model.r03_combos import fit_combo_weights

    out = fit_combo_weights(
        {"MOMENTUM_20": 0.2, "RSQ_60": 0.1, "REVERSAL_60": 0.4}, lam=0.0
    )
    assert set(out["weights"].values()) == {pytest.approx(1 / 3)}
    assert out["fallback"] is False


# --- N3: the four-way adjudication ---------------------------------------------


def _summary(excess, ir=0.1, invested=0.7, holdings=8.0, positive=3,
             failed=0):
    return {
        "mean_net_excess_1x": excess,
        "active_ir_1x": ir,
        "invested_ratio_mean_1x": invested,
        "mean_holdings_1x": holdings,
        "positive_folds_1x": positive,
        "n_folds": 5,
        "failed_folds": failed,
    }


def _cells(eq_excess, fitted_excess, extra=None):
    cells = {}
    for cell in ("n10_m1000", "n10_m2500", "n10_m5000"):
        arms = {
            "combo:all3": _summary(eq_excess),
            "combo:fitted_lam050": _summary(fitted_excess),
        }
        if extra:
            arms.update(extra)
        cells[cell] = {"top_n": 10, "arms": arms}
    return cells


def test_n3_increment_verdict_when_margin_met_everywhere():
    from ashare_model.r03_combos import (
        R03_INCREMENT_MARGIN,
        decide_r03_adjudication,
    )

    assert R03_INCREMENT_MARGIN == 0.005
    out = decide_r03_adjudication(_cells(0.02, 0.03))
    assert out["verdict"] == "fitted_increment_region"
    # The descriptive arms never enter the adjudication inputs.
    out2 = decide_r03_adjudication(
        _cells(0.02, 0.03, extra={"combo:fitted_lam025": _summary(9.9)})
    )
    assert out2["verdict"] == "fitted_increment_region"


def test_n3_equal_weight_stands_when_margin_short():
    from ashare_model.r03_combos import decide_r03_adjudication

    out = decide_r03_adjudication(_cells(0.02, 0.023))
    assert out["verdict"] == "equal_weight_stands"
    # Fitted viable but below eq in one cell: still equal_weight_stands
    # (eq replicates, fitted adds nothing everywhere).
    cells = _cells(0.02, 0.03)
    cells["n10_m5000"]["arms"]["combo:fitted_lam050"] = _summary(0.015)
    out = decide_r03_adjudication(cells)
    assert out["verdict"] == "equal_weight_stands"


def test_n3_no_net_space_when_equal_weight_dead_in_two_cells():
    from ashare_model.r03_combos import decide_r03_adjudication

    cells = _cells(0.02, 0.03)
    cells["n10_m1000"]["arms"]["combo:all3"] = _summary(-0.01, ir=-0.05)
    cells["n10_m5000"]["arms"]["combo:all3"] = _summary(-0.02, ir=-0.10)
    out = decide_r03_adjudication(cells)
    assert out["verdict"] == "combo_no_net_space"


def test_n3_mixed_panel_is_no_reliable_info():
    from ashare_model.r03_combos import decide_r03_adjudication

    # Eq viable in exactly two of three cells (dead in one): neither the
    # increment rule, the stands rule (needs all three), nor the
    # no-space rule (needs two dead) applies.
    cells = _cells(0.02, 0.023)
    cells["n10_m2500"]["arms"]["combo:all3"] = _summary(-0.01, ir=-0.05)
    out = decide_r03_adjudication(cells)
    assert out["verdict"] == "panel_no_reliable_info"


def test_n3_arm_rules_are_the_p37_authority():
    from ashare_model import r02_matrix, r03_combos

    # One implementation, never two: the R03 adjudicator must consume the
    # p37 frozen rules dict, and blocking flags (low_activity) veto
    # viability exactly like round 1.
    assert r03_combos.R03_ADJUDICATION_RULES is r02_matrix.R02_ADJUDICATION_RULES
    cells = _cells(0.02, 0.03)
    cells["n10_m1000"]["arms"]["combo:fitted_lam050"] = _summary(
        0.03, invested=0.2
    )
    out = decide_r03_adjudication(cells)
    assert out["verdict"] == "equal_weight_stands"
    assert "low_activity" in out["cells"]["n10_m1000"]["arms"][
        "combo:fitted_lam050"
    ]["flags"]


# --- N4/N5: the panel driver and the refusal path (mini e2e) -------------------


def test_n5_contract_sha_mismatch_refused(mini_env, gate_run_asof):
    tmp_path, _data_config, cfg_path = mini_env
    from test_p34_r01_baseline import _mini_dates

    gate_run_asof(_mini_dates()[-1])
    from ashare_model.r03_combos import R03Refusal, run_r03_matrix

    with pytest.raises(R03Refusal, match="contract sha256"):
        run_r03_matrix(
            cfg_path,
            contract_sha256="0" * 64,
            out_root=tmp_path / "r03_out",
            require_clean_tree=False,
        )


def test_n4_panel_structure_and_frozen_weights(mini_env, gate_run_asof):
    import hashlib
    from pathlib import Path

    tmp_path, _data_config, cfg_path = mini_env
    from test_p34_r01_baseline import _mini_dates

    def _p42_sha() -> str:
        repo = Path(__file__).resolve().parents[1]
        return hashlib.sha256(
            (repo / "docs" / "p42_r03_combo_weights_draft.md").read_bytes()
        ).hexdigest()

    gate_run_asof(_mini_dates()[-1])
    from ashare_model import r03_combos

    summary = r03_combos.run_r03_matrix(
        cfg_path,
        contract_sha256=_p42_sha(),
        out_root=tmp_path / "r03_out",
        require_clean_tree=False,
        min_eligible=3,
        adjudication_rules={
            "min_positive_folds": 1,
            "min_invested_ratio": 0.3,
            "max_failed_folds": 1,
        },
    )
    report = json.loads(
        (tmp_path / "r03_out" / "r03_combo_report.json").read_text(
            encoding="utf-8"
        )
    )
    rows = report["rows"]
    candidates = {str(r["candidate"]) for r in rows}
    cells = ("n10_m1000", "n10_m2500", "n10_m5000")
    combo_arms = {
        "combo:all3",
        "combo:fitted_lam050",
        "combo:fitted_lam025",
        "combo:fitted_lam075",
    }
    # Every cell carries every combo arm at both cost levels.
    for cell in cells:
        for arm in combo_arms:
            for level in ("0x", "1x"):
                assert f"{cell}::{arm}" in candidates, (cell, arm, level)
    # Singles live only on the reference-cell direction path (1x).
    singles = {
        c for c in candidates if "::baseline:" in c
    }
    assert singles and all(
        c.startswith("n10_m2500::") for c in singles
    ), sorted(singles)
    # The benchmark row is the cost-free per-fold reference.
    assert "benchmark:equal_weight" in candidates

    # Directions and weights are learned once per fold on the reference
    # cell and frozen across cells: per-fold weight evidence must be
    # identical for every cell of the same fold.
    evidence = report["fold_evidence"]
    folds = sorted(evidence)
    assert folds
    for fold in folds:
        weights_by_cell = evidence[fold]["weights_by_cell"]
        first = weights_by_cell[cells[0]]["combo:fitted_lam050"]["weights"]
        for cell in cells[1:]:
            assert (
                weights_by_cell[cell]["combo:fitted_lam050"]["weights"]
                == first
            )
    # Descriptive arms are labeled as non-adjudicated.
    adjudicated = report["adjudicated_arms"]
    assert set(adjudicated) == {"combo:all3", "combo:fitted_lam050"}
    # Verdict from the frozen four-way set; honest lifecycle end state.
    assert summary["adjudication"]["verdict"] in {
        "fitted_increment_region",
        "equal_weight_stands",
        "combo_no_net_space",
        "panel_no_reliable_info",
    }
