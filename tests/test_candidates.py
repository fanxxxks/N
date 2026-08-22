from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_data.processor import open_to_open_returns
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.candidates import (
    CandidateScore,
    CandidateScorer,
    CandidateSelector,
    CandidateSpec,
)
from ashare_model.vocab import FORMULA_VOCAB


def _bt(**kwargs) -> BacktestConfig:
    values = dict(
        initial_capital=100000.0,
        top_n=2,
        single_weight_cap=0.5,
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    values.update(kwargs)
    return BacktestConfig(**values)


def _spec(candidate_id: str, token: int = 1) -> CandidateSpec:
    return CandidateSpec(candidate_id, candidate_id, "test", (token,))


def _score(
    candidate_id: str,
    *,
    val_reward: float,
    val_icir: float,
    full_reward: float,
    full_icir: float,
    eligible: bool,
    token: int,
    reasons: tuple[str, ...] = (),
) -> CandidateScore:
    return CandidateScore(
        tokens=(token,),
        candidate_id=candidate_id,
        formula_text=candidate_id,
        source="test",
        direction=1,
        val_reward=val_reward,
        val_icir=val_icir,
        full_window_reward=full_reward,
        full_window_icir=full_icir,
        complexity_penalty=0.0,
        eligible=eligible,
        rejection_reasons=reasons,
    )


def test_selector_filters_before_ranking_and_keeps_best_rejected():
    rejected_high_reward = _score(
        "rejected",
        val_reward=1.0,
        val_icir=0.01,
        full_reward=1.0,
        full_icir=0.8,
        eligible=False,
        token=1,
        reasons=("val_icir_below_minimum",),
    )
    eligible_lower_reward = _score(
        "eligible",
        val_reward=0.9,
        val_icir=0.2,
        full_reward=0.8,
        full_icir=0.3,
        eligible=True,
        token=2,
    )
    result = CandidateSelector().select(
        [rejected_high_reward, eligible_lower_reward]
    )
    assert result.selected == eligible_lower_reward
    assert result.best_rejected == rejected_high_reward
    assert result.to_dict()["best_rejected"]["rejection_reasons"] == [
        "val_icir_below_minimum"
    ]


def test_selector_uses_all_metrics_then_token_key_for_ties():
    left = _score(
        "left", val_reward=0.5, val_icir=0.4, full_reward=0.3,
        full_icir=0.2, eligible=True, token=2,
    )
    right = _score(
        "right", val_reward=0.5, val_icir=0.4, full_reward=0.3,
        full_icir=0.2, eligible=True, token=1,
    )
    assert CandidateSelector().select([left, right]).selected == right


def test_eligibility_accepts_equal_thresholds_and_rejects_nonfinite():
    def threshold_reward(signals, target, bt, rc, windows, **kwargs):
        n = signals.shape[0]
        return (
            np.full(n, 0.2),
            np.full(n, 0.1),
            np.full(n, 0.4),
            np.full(n, 0.3),
        )

    scorer = CandidateScorer(
        _bt(),
        RewardConfig(
            complexity_penalty=0.0,
            min_val_reward=0.1,
            min_val_icir=0.3,
            ic_min_stocks=2,
        ),
        operator_offset=FORMULA_VOCAB.operator_offset,
        reward_function=threshold_reward,
    )
    signal = np.arange(24, dtype=float).reshape(4, 6)
    target = np.zeros_like(signal)
    score = scorer.score(_spec("edge"), signal, target, [(1, 4)])
    assert score.eligible
    assert score.val_reward == pytest.approx(0.1)
    assert score.val_icir == pytest.approx(0.3)

    def nonfinite_reward(signals, target, bt, rc, windows, **kwargs):
        n = signals.shape[0]
        return (
            np.zeros(n), np.full(n, np.nan), np.zeros(n), np.full(n, np.inf)
        )

    scorer = CandidateScorer(
        _bt(),
        RewardConfig(complexity_penalty=0.0, ic_min_stocks=2),
        operator_offset=FORMULA_VOCAB.operator_offset,
        reward_function=nonfinite_reward,
    )
    score = scorer.score(_spec("bad"), signal, target, [(1, 4)])
    assert not score.eligible
    assert score.rejection_reasons == (
        "val_reward_not_finite",
        "val_icir_not_finite",
    )


def test_invalid_and_near_constant_reasons_are_explicit():
    scorer = CandidateScorer(
        _bt(),
        RewardConfig(ic_min_stocks=2),
        operator_offset=FORMULA_VOCAB.operator_offset,
    )
    target = np.zeros((4, 6))
    invalid = scorer.score(
        _spec("invalid"), None, target, [(1, 4)], formula_valid=False
    )
    constant = scorer.score(_spec("constant"), np.ones_like(target), target, [(1, 4)])
    assert invalid.rejection_reasons == ("invalid_formula",)
    assert constant.rejection_reasons == ("constant_or_near_constant_signal",)


def test_direction_mirror_invariance_in_scoring_and_backtest():
    rng = np.random.default_rng(123)
    n_stocks, n_dates = 8, 12
    target = rng.normal(0.0005, 0.005, size=(n_stocks, n_dates))
    target[:, -2:] = 0.0
    signal = target * 1000.0
    reward_cfg = RewardConfig(
        complexity_penalty=0.0,
        min_val_reward=-1e9,
        min_val_icir=-1e9,
        ic_min_stocks=4,
    )
    scorer = CandidateScorer(
        _bt(), reward_cfg, operator_offset=FORMULA_VOCAB.operator_offset
    )
    windows = [(4, 10)]
    positive, mirrored = scorer.score_many(
        [_spec("positive"), _spec("mirrored")],
        [signal, -signal],
        target,
        windows,
        full_signal_range=(0, 10),
    )
    assert positive.direction == -mirrored.direction
    for field in (
        "val_reward",
        "val_icir",
        "full_window_reward",
        "full_window_icir",
    ):
        assert getattr(positive, field) == pytest.approx(
            getattr(mirrored, field), rel=0.0, abs=0.0
        )
    applied_positive = positive.direction * signal
    applied_mirrored = mirrored.direction * -signal
    assert np.array_equal(applied_positive, applied_mirrored)

    open_ = np.full((n_stocks, n_dates), 10.0)
    for t in range(n_dates - 2):
        open_[:, t + 2] = open_[:, t + 1] * (1.0 + target[:, t])
    assert open_to_open_returns(open_)[:, :10] == pytest.approx(target[:, :10])
    raw = {
        "open": open_,
        "high": open_ * 1.02,
        "low": open_ * 0.98,
        "pre_close": np.roll(open_, 1, axis=1),
        "volume": np.full_like(open_, 1_000_000.0),
    }
    raw["pre_close"][:, 0] = open_[:, 0]
    dates = [f"202401{i + 1:02d}" for i in range(n_dates)]
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    engine = AshareBacktestEngine(_bt())
    left = engine.run(applied_positive, raw, codes, dates)
    right = engine.run(applied_mirrored, raw, codes, dates)
    assert left.daily_returns == right.daily_returns
    assert left.positions == right.positions


def test_direction_tie_break_is_mirror_stable():
    def tied_reward(signals, target, bt, rc, windows, **kwargs):
        n = signals.shape[0]
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    scorer = CandidateScorer(
        _bt(),
        RewardConfig(
            complexity_penalty=0.0,
            min_val_reward=-1.0,
            min_val_icir=-1.0,
            ic_min_stocks=2,
        ),
        operator_offset=FORMULA_VOCAB.operator_offset,
        reward_function=tied_reward,
    )
    signal = np.arange(-12, 12, dtype=float).reshape(4, 6)
    a, b = scorer.score_many(
        [_spec("a"), _spec("b")], [signal, -signal], np.zeros_like(signal), [(1, 4)]
    )
    assert a.direction == -b.direction
    assert np.array_equal(a.direction * signal, b.direction * -signal)


def test_artifact_reader_is_one_way_legacy_compatible():
    legacy = CandidateScore.from_payload(
        {
            "formula": [1],
            "formula_text": "RET_1",
            "best_reward": 0.2,
            "best_icir": 0.3,
        }
    )
    assert legacy.val_reward == 0.2
    assert legacy.full_window_icir == 0.3
    payload = legacy.to_dict()
    assert "best_reward" not in payload and "best_icir" not in payload


# --- signal-date universe eligibility in candidate scoring --------------------


def _future_mask(n_stocks, n_dates, join_day, future_row=-1):
    mask = np.ones((n_stocks, n_dates), dtype=bool)
    mask[future_row, :join_day] = False
    return mask


def _scorer(**reward_kwargs):
    defaults = dict(
        complexity_penalty=0.0,
        min_val_reward=-1e9,
        min_val_icir=-1e9,
        ic_min_stocks=3,
    )
    defaults.update(reward_kwargs)
    return CandidateScorer(
        _bt(),
        RewardConfig(**defaults),
        operator_offset=FORMULA_VOCAB.operator_offset,
    )


def test_scorer_pre_join_extreme_does_not_change_any_field():
    rng = np.random.default_rng(44)
    n_stocks, n_dates = 6, 20
    join_day = 8
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = rng.normal(size=(n_stocks, n_dates))
    extreme = signal.copy()
    extreme[-1, :join_day] = 1e9  # future member's extreme pre-join values
    mask = _future_mask(n_stocks, n_dates, join_day)
    scorer = _scorer()
    windows = [(10, 18)]
    mild = scorer.score(_spec("mild"), signal, target, windows, universe_mask=mask)
    ext = scorer.score(_spec("extreme"), extreme, target, windows, universe_mask=mask)
    # Every quality field is identical: the extreme ineligible values can
    # move neither the score nor the direction nor the rejection decision.
    for field in (
        "direction",
        "val_reward",
        "val_icir",
        "full_window_reward",
        "full_window_icir",
        "complexity_penalty",
        "eligible",
        "rejection_reasons",
    ):
        assert getattr(mild, field) == getattr(ext, field), field
    # Without the mask the same perturbation does move the full-window
    # statistics (sanity: the mask is what neutralizes it).
    mild_open = scorer.score(_spec("mild"), signal, target, windows)
    ext_open = scorer.score(_spec("extreme"), extreme, target, windows)
    assert not (
        mild_open.full_window_icir == ext_open.full_window_icir
        and mild_open.full_window_reward == ext_open.full_window_reward
    )


def test_scorer_join_day_values_enter_the_score():
    rng = np.random.default_rng(46)
    n_stocks, n_dates = 6, 20
    join_day = 8
    target = rng.normal(0.001, 0.01, size=(n_stocks, n_dates))
    signal = rng.normal(size=(n_stocks, n_dates))
    extreme = signal.copy()
    extreme[-1, join_day] = 1e9  # extreme exactly on the join day
    mask = _future_mask(n_stocks, n_dates, join_day)
    scorer = _scorer()
    windows = [(6, 14)]  # validation window straddles the join day
    mild = scorer.score(_spec("mild"), signal, target, windows, universe_mask=mask)
    ext = scorer.score(_spec("extreme"), extreme, target, windows, universe_mask=mask)
    # On the join day the value is eligible: it legitimately moves the
    # validation statistics.
    assert not (
        mild.val_reward == ext.val_reward and mild.val_icir == ext.val_icir
    )


def test_scorer_near_constant_checks_only_eligible_cells():
    n_stocks, n_dates = 4, 8
    target = np.zeros((n_stocks, n_dates))
    signal = np.ones_like(target)  # constant over the eligible cells
    signal[-1, :4] = np.arange(4) * 1e6  # extreme finite ineligible values
    mask = _future_mask(n_stocks, n_dates, join_day=4)
    scorer = _scorer(min_val_reward=0.0, min_val_icir=0.0)
    masked = scorer.score(
        _spec("masked"), signal, target, [(4, 6)], universe_mask=mask
    )
    open_ = scorer.score(_spec("open"), signal, target, [(4, 6)])
    # The eligible cross-section is constant: rejected, regardless of the
    # ineligible extremes...
    assert masked.rejection_reasons == ("constant_or_near_constant_signal",)
    # ...while the unmasked path sees the extreme spread and lets it pass
    # (it then fails on the degenerate IC instead — a different reason).
    assert "constant_or_near_constant_signal" not in open_.rejection_reasons


def test_scorer_direction_tie_break_scans_eligible_only():
    def tied_reward(signals, target, bt, rc, windows, **kwargs):
        n = signals.shape[0]
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    scorer = CandidateScorer(
        _bt(),
        RewardConfig(
            complexity_penalty=0.0,
            min_val_reward=-1.0,
            min_val_icir=-1.0,
            ic_min_stocks=2,
        ),
        operator_offset=FORMULA_VOCAB.operator_offset,
        reward_function=tied_reward,
    )
    # Row 0 (the future member) carries an extreme negative value at the
    # start of the validation window but is ineligible there; the eligible
    # rows carry positive values only.
    signal = np.zeros((3, 8))
    signal[0] = 4.0
    signal[1] = 2.0
    signal[2] = 3.0
    signal[0, 2:5] = -1e6
    mask = _future_mask(3, 8, join_day=5, future_row=0)
    windows = [(2, 6)]
    masked = scorer.score(_spec("masked"), signal, np.zeros_like(signal), windows, universe_mask=mask)
    open_ = scorer.score(_spec("open"), signal, np.zeros_like(signal), windows)
    # With the mask the canonical orientation comes from the first eligible
    # non-zero validation observation (+2/+3) -> +1; without it the extreme
    # ineligible -1e6 decides the direction -> -1.
    assert masked.direction == 1
    assert open_.direction == -1


def test_scorer_direction_mirror_invariance_holds_under_mask():
    rng = np.random.default_rng(47)
    n_stocks, n_dates = 6, 16
    join_day = 6
    target = rng.normal(0.0005, 0.005, size=(n_stocks, n_dates))
    signal = rng.normal(size=(n_stocks, n_dates))
    mask = _future_mask(n_stocks, n_dates, join_day)
    scorer = _scorer()
    windows = [(8, 14)]
    positive, mirrored = scorer.score_many(
        [_spec("positive"), _spec("mirrored")],
        [signal, -signal],
        target,
        windows,
        full_signal_range=(0, 14),
        universe_mask=mask,
    )
    # Both orientations are scored under the identical mask: the applied
    # signal is invariant and the quality fields agree exactly.
    assert positive.direction == -mirrored.direction
    assert np.array_equal(
        positive.direction * signal, mirrored.direction * -signal
    )
    for field in (
        "val_reward",
        "val_icir",
        "full_window_reward",
        "full_window_icir",
    ):
        assert getattr(positive, field) == pytest.approx(
            getattr(mirrored, field), rel=0.0, abs=0.0
        )


def test_scorer_rejects_mask_shape_mismatch():
    scorer = _scorer()
    target = np.zeros((4, 6))
    signal = np.ones((4, 6))
    with pytest.raises(ValueError, match="universe_mask shape"):
        scorer.score(
            _spec("bad"),
            signal,
            target,
            [(1, 4)],
            universe_mask=np.ones((3, 6), dtype=bool),
        )
