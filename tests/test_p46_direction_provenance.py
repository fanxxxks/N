"""P46 direction-learning provenance RED/GREEN tests (WP 10).

Each test asserts an independent invariant of the p46 preregistration
contract (``docs/p46_direction_provenance_draft.md`` §2/§3):

* two pre-fixed direction rules exist in one authority - the current
  reference-return rule (delegated, semantics unchanged) and the new
  pure-signal label-IC rule - each returning the five-element provenance
  (N1);
* the label-IC rule is a pure signal rule: no portfolio config in its
  interface, so capital/min-order cannot reach it (N2);
* reference-return decisions are reproducible per config and every pair
  of decisions is attributable to its recorded reference_config (N3);
* neither rule reads the test window: poisoning test-period columns
  leaves decisions and evidence bit-identical (N4);
* baseline rows carry the full provenance (N5);
* the percentile-rank helper is the frozen avg_rank/N_eff form on a
  hand-checked case, and both rules are mirror-stable (N6).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig

from ashare_model.direction import (
    DIRECTION_RULE_VERSION,
    RULE_LABEL_IC,
    RULE_REFERENCE_RETURN,
    DirectionDecision,
    decide_direction,
    label_ic_direction,
    reference_return_direction,
)
from ashare_model.r01_baseline import percentile_normalize_signal

MISSING_HANDLING = "nan-excluded+pit-universe-mask"


def _factor_market(n_stocks: int = 12, n_dates: int = 30, seed: int = 3):
    """A signal that genuinely predicts the label (positive IC)."""

    rng = np.random.default_rng(seed)
    label = rng.normal(0.0, 0.01, size=(n_stocks, n_dates))
    signal = label + rng.normal(0.0, 0.05, size=(n_stocks, n_dates))
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    return signal, label, mask


def _label_def(train_signal_end: int) -> dict:
    return {
        "name": "forward_return_open_open",
        "basis": "adjusted_open",
        "horizon": 1,
        "window": f"[0, {train_signal_end})",
    }


TRAIN_END = 20  # train window columns [0, 20); test window is [22, ...)

# --- N1: the authority exists with both rules -------------------------------


def test_n1_two_rules_with_five_element_provenance():
    signal, label, mask = _factor_market()
    decision_b = label_ic_direction(
        "factor:x",
        signal,
        label,
        universe_mask=mask,
        train_signal_range=(0, TRAIN_END),
        train_cutoff="2023-12-31",
        label_def=_label_def(TRAIN_END),
    )
    assert isinstance(decision_b, DirectionDecision)
    assert decision_b.rule == RULE_LABEL_IC
    assert decision_b.direction in (1, -1)
    # The five elements (B2): rule / train_cutoff / label / missing
    # handling / reference config (A-only) plus rule-specific evidence.
    assert decision_b.train_cutoff == "2023-12-31"
    assert decision_b.label == _label_def(TRAIN_END)
    assert decision_b.missing_handling == MISSING_HANDLING
    assert decision_b.reference_config is None
    assert "ic_mean" in decision_b.evidence and "n_dates" in decision_b.evidence
    assert DIRECTION_RULE_VERSION == 1

    cfg = BacktestConfig(top_n=3, initial_capital=100_000.0)
    decision_a = reference_return_direction(
        "factor:x",
        signal,
        label,
        universe_mask=mask,
        val_windows=[(10, TRAIN_END)],
        train_signal_range=(0, TRAIN_END),
        train_cutoff="2023-12-31",
        label_def=_label_def(TRAIN_END),
        backtest_config=cfg,
    )
    assert decision_a.rule == RULE_REFERENCE_RETURN
    assert decision_a.reference_config is not None
    assert decision_a.reference_config["initial_capital"] == 100_000.0
    for key in ("val_reward", "val_icir", "val_reward_opposite",
                "val_icir_opposite"):
        assert key in decision_a.evidence


def test_n1_dispatch_rejects_unknown_rule():
    signal, label, mask = _factor_market()
    with pytest.raises(ValueError, match="unknown direction rule"):
        decide_direction(
            "astrology",
            "f",
            signal,
            label,
            universe_mask=mask,
            train_signal_range=(0, TRAIN_END),
            train_cutoff="2023-12-31",
            label_def=_label_def(TRAIN_END),
        )


# --- N2: the label-IC rule is a pure signal rule -----------------------------


def test_n2_label_ic_takes_no_portfolio_config_and_decides_by_ic_sign():
    params = inspect.signature(label_ic_direction).parameters
    for name in params:
        assert not any(
            token in name.lower()
            for token in ("config", "capital", "min_trade", "budget")
        ), f"label_ic_direction must stay config-free, got parameter {name!r}"

    signal, label, mask = _factor_market()
    decision = label_ic_direction(
        "factor:x", signal, label,
        universe_mask=mask,
        train_signal_range=(0, TRAIN_END),
        train_cutoff="2023-12-31",
        label_def=_label_def(TRAIN_END),
    )
    # The signal genuinely predicts the label: the IC rule says +1, and a
    # deterministic re-run says the same thing.
    assert decision.direction == 1
    again = label_ic_direction(
        "factor:x", signal, label,
        universe_mask=mask,
        train_signal_range=(0, TRAIN_END),
        train_cutoff="2023-12-31",
        label_def=_label_def(TRAIN_END),
    )
    assert again.direction == decision.direction
    assert again.evidence == decision.evidence


# --- N3: reference-return decisions are attributable per config ---------------


def test_n3_reference_return_reproducible_and_attributable():
    signal, label, mask = _factor_market()
    cfg_small = BacktestConfig(
        top_n=3, initial_capital=100_000.0, min_trade_amount=1000.0,
    )
    cfg_big = BacktestConfig(
        top_n=3, initial_capital=1_000_000.0, min_trade_amount=5000.0,
    )
    kwargs = dict(
        universe_mask=mask,
        val_windows=[(10, TRAIN_END)],
        train_signal_range=(0, TRAIN_END),
        train_cutoff="2023-12-31",
        label_def=_label_def(TRAIN_END),
    )
    d1a = reference_return_direction("f", signal, label,
                                     backtest_config=cfg_small, **kwargs)
    d1b = reference_return_direction("f", signal, label,
                                     backtest_config=cfg_small, **kwargs)
    d2 = reference_return_direction("f", signal, label,
                                    backtest_config=cfg_big, **kwargs)
    # Reproducible per config: same config -> identical decision+evidence.
    assert d1a.direction == d1b.direction
    assert d1a.evidence == d1b.evidence
    # Attributable: whatever the two configs decide, the records identify
    # which training configuration produced each decision.
    assert d1a.reference_config["initial_capital"] == 100_000.0
    assert d2.reference_config["initial_capital"] == 1_000_000.0
    assert d1a.reference_config != d2.reference_config


# --- N4: neither rule reads the test window ----------------------------------


def test_n4_poisoned_test_window_leaves_both_rules_unchanged():
    signal, label, mask = _factor_market()
    n_stocks, n_dates = signal.shape
    kwargs = dict(
        universe_mask=mask,
        train_signal_range=(0, TRAIN_END),
        train_cutoff="2023-12-31",
        label_def=_label_def(TRAIN_END),
    )
    cfg = BacktestConfig(top_n=3)
    base_b = label_ic_direction("f", signal, label, **kwargs)
    base_a = reference_return_direction(
        "f", signal, label,
        val_windows=[(10, TRAIN_END)], backtest_config=cfg, **kwargs,
    )
    # Poison every test-period column: flip the label, blow up the signal.
    poisoned_signal = signal.copy()
    poisoned_label = label.copy()
    poisoned_signal[:, TRAIN_END + 2 :] = -1e9 * poisoned_signal[:, TRAIN_END + 2 :]
    poisoned_label[:, TRAIN_END + 2 :] = -1e9 * label[:, TRAIN_END + 2 :]
    poison_b = label_ic_direction("f", poisoned_signal, poisoned_label, **kwargs)
    poison_a = reference_return_direction(
        "f", poisoned_signal, poisoned_label,
        val_windows=[(10, TRAIN_END)], backtest_config=cfg, **kwargs,
    )
    assert poison_b.direction == base_b.direction
    assert poison_b.evidence == base_b.evidence
    assert poison_a.direction == base_a.direction
    assert poison_a.evidence == base_a.evidence


# --- N6: percentile helper and mirror stability ------------------------------


def test_n6_percentile_normalize_hand_case():
    signal = np.array([[3.0, 1.0], [1.0, np.nan], [2.0, 2.0], [np.nan, 3.0]])
    mask = np.ones(signal.shape, dtype=bool)
    out = percentile_normalize_signal(signal, mask)
    # Column 0: values {3, 1, 2}, N_eff=3, ranks 1..3 -> 1/3, 2/3, 1.
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(1.0 / 3.0)
    assert out[2, 0] == pytest.approx(2.0 / 3.0)
    assert np.isnan(out[3, 0])
    # Column 1: values {1, 2, 3} (NaN row excluded), N_eff=3 again.
    assert out[0, 1] == pytest.approx(1.0 / 3.0)
    assert np.isnan(out[1, 1])
    assert out[2, 1] == pytest.approx(2.0 / 3.0)
    assert out[3, 1] == pytest.approx(1.0)


def test_n6_mirror_stability_both_rules():
    signal, label, mask = _factor_market()
    kwargs = dict(
        universe_mask=mask,
        train_signal_range=(0, TRAIN_END),
        train_cutoff="2023-12-31",
        label_def=_label_def(TRAIN_END),
    )
    d_b = label_ic_direction("f", signal, label, **kwargs)
    d_b_mirror = label_ic_direction("f", -signal, label, **kwargs)
    assert d_b_mirror.direction == -d_b.direction
    d_a = reference_return_direction(
        "f", signal, label,
        val_windows=[(10, TRAIN_END)],
        backtest_config=BacktestConfig(top_n=3), **kwargs,
    )
    d_a_mirror = reference_return_direction(
        "f", -signal, label,
        val_windows=[(10, TRAIN_END)],
        backtest_config=BacktestConfig(top_n=3), **kwargs,
    )
    assert d_a_mirror.direction == -d_a.direction


# --- N5: baseline rows carry the full provenance ------------------------------


def test_n5_baseline_rows_carry_direction_provenance(populated_db):
    from ashare_data.config import ModelConfig, ProtocolConfig
    from ashare_model.data_loader import AshareDataLoader
    from ashare_model.eval_folds import FoldConfig, resolve_folds
    from ashare_model.eval_search import baseline_candidates

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    proto = ProtocolConfig(
        folds=[FoldConfig("2024-01-10", "2024-01-25")],
        baseline_signals=["MOMENTUM_20"],
    )
    fold = resolve_folds(
        proto.folds, loader.dates,
        frequency=proto.frequency, horizon=proto.horizon,
    )[0]
    bt_cfg = BacktestConfig(
        rebalance_frequency=proto.frequency,
        target_horizon=proto.horizon,
    )
    rows = baseline_candidates(
        loader, proto, fold, bt_cfg, ModelConfig(), RewardConfig()
    )
    assert rows
    for row in rows:
        prov = row["direction_provenance"]
        assert prov["rule"] == RULE_REFERENCE_RETURN
        assert prov["train_cutoff"] == fold.train_end
        assert prov["label"]["horizon"] == proto.horizon
        assert prov["missing_handling"] == MISSING_HANDLING
        assert prov["reference_config"]["initial_capital"] == (
            bt_cfg.initial_capital
        )
        assert prov["direction"] in (1, -1)
        assert row["direction"] == prov["direction"]
