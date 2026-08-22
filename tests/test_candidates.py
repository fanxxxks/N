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
