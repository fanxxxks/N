"""p54 / WP16 16-C: both fixed orientations' raw evidence from ONE batch.

Assertion source: ``reports/wp16/FROZEN_C.md`` (frozen 2026-09-23, before
any WP16 code existed; read-only for implementers).  Every assertion cites
the frozen section it pins; nothing below is derived from implementation
output.  The tests were written first and recorded RED in
``reports/p54_red_evidence.txt`` before ``ashare_model/direction.py``,
``ashare_model/candidates.py`` and ``ashare_model/eval_search.py`` were
touched.

What is pinned, per the frozen doc:

* V1 (§3 V1): a synthetic market whose two orientations score differently
  must expose BOTH sides' numbers — equal element-for-element to an
  independently computed interleaved ``reward_function`` batch (the
  anti-forgery anchor of §1's source constraint) — and the loser's true
  value must stop being the winner's value recorded twice (the replaced
  defect, §0.2 ``direction.py:195-197`` / ``eval_search.py:163-179``).
* V2 (§3 V2): mirroring the input exchanges the two sides' evidence
  field-for-field and leaves the applied economic exposure
  (``direction x signal``) unchanged.
* V3 (§3 V3): poisoning every test-period column (the N4手法, incl. a
  NaN variant) leaves the full §1 field pattern bit-identical.
* §2: the selection rule is Rule A verbatim ("参考配置下的验证 reward 择向"),
  its tie-break branches are classifiable from the same batch arrays, and
  the classification agrees with ``CandidateScorer._choose_direction``.
* §0.3 red line 1: exactly ONE ``score_many`` call per decision — the
  second auto-selecting scoring round is gone.
* §5 red line 2: the four legacy evidence keys survive with correct
  chosen/opposite semantics.

Non-finite floats in evidence are stored as ``None`` (the repo's strict
JSON evidence convention, ``r01_baseline.rows_to_jsonable``); the raw
NaN floats stay visible in the batch arrays and in ``reward_clip``
disclosure.

The companion file ``tests/test_p54_wp16c_combo_invariance.py`` pins the
§4 percentile identity and the §4.2.1 measurement-script output contract.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ashare_data.config import BacktestConfig, RewardConfig
from ashare_model.candidates import CandidateScorer
from ashare_model.direction import (
    RULE_REFERENCE_RETURN,
    DirectionDecision,
    reference_return_direction,
)

TRAIN_END = 20  # train window columns [0, 20); test window is [22, ...)
N_STOCKS, N_DATES = 12, 30
VAL_WINDOWS = [(10, TRAIN_END)]
TRAIN_RANGE = (0, TRAIN_END)

DATES = [f"2023-12-{day:02d}" for day in range(1, N_DATES + 1)]

_FEE_FREE = dict(
    commission_rate=0.0,
    min_commission=0.0,
    stamp_tax_rate=0.0,
    transfer_fee_rate=0.0,
    slippage_rate=0.0,
)

_NUMERIC_SIDE_KEYS = (
    "val_reward_plus", "val_reward_minus",
    "val_icir_plus", "val_icir_minus",
    "train_reward_plus", "train_reward_minus",
    "train_icir_plus", "train_icir_minus",
)


# --- fixtures ----------------------------------------------------------------


def _monotone_market(seed: int, slope: float):
    """§3 V1 construction: the label is strictly monotone in the signal
    (plus a tiny noise term) on every window, so the two orientations
    score clearly differently."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, size=(N_STOCKS, N_DATES))
    label = slope * signal + rng.normal(0.0, 1e-4, size=(N_STOCKS, N_DATES))
    mask = np.ones((N_STOCKS, N_DATES), dtype=bool)
    return signal, label, mask


def _canonical_market(seed: int = 5):
    """§2 branch 3 construction: zero returns and a fee-free reference
    config make both sides' ``(val_reward, val_icir)`` keys exactly equal
    (active IR 0, no cost, IC 0), so the decision falls through to the
    mirror-stable canonical orientation; ``signal[0, 10]`` is the first
    observation the val-window scan meets."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.0, size=(N_STOCKS, N_DATES))
    signal[0, 10] = -0.75
    label = np.zeros((N_STOCKS, N_DATES))
    mask = np.ones((N_STOCKS, N_DATES), dtype=bool)
    return signal, label, mask


def _zero_val_window_market(seed: int = 6):
    """§2 branch 3 / "all_zero_nonfinite": variance only before the val
    window; the val window is all zeros so the canonical scan finds no
    finite non-zero observation."""
    rng = np.random.default_rng(seed)
    signal = np.zeros((N_STOCKS, N_DATES))
    signal[:, :10] = rng.normal(0.0, 1.0, size=(N_STOCKS, 10))
    label = np.zeros((N_STOCKS, N_DATES))
    mask = np.ones((N_STOCKS, N_DATES), dtype=bool)
    return signal, label, mask


def _label_def() -> dict:
    return {
        "name": "forward_return_open_open",
        "basis": "adjusted_open",
        "horizon": 1,
        "window": f"[0, {TRAIN_END})",
    }


def _rule_a(signal, label, mask, *, cfg=None, dates=None) -> DirectionDecision:
    return reference_return_direction(
        "factor:x",
        signal,
        label,
        universe_mask=mask,
        val_windows=list(VAL_WINDOWS),
        train_signal_range=TRAIN_RANGE,
        train_cutoff="2023-12-20",
        label_def=_label_def(),
        backtest_config=cfg or BacktestConfig(top_n=3),
        dates=dates,
    )


def _jsonable(value) -> float | None:
    """The evidence convention: non-finite floats are stored as None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _independent_oriented_batch(signal, label, mask, cfg) -> tuple:
    """§1 source constraint / §3 V1: the test's OWN interleaved batch —
    the reward function called directly, never through score_many's
    selection layer — so evidence numbers can be checked against array
    elements rather than against another scorer call."""
    from ashare_model.reward import batched_basket_rewards

    oriented = np.stack([signal, -signal])
    return batched_basket_rewards(
        oriented,
        label,
        cfg,
        RewardConfig(),
        list(VAL_WINDOWS),
        universe_mask=mask,
        train_signal_range=TRAIN_RANGE,
    )


def _side(direction: int) -> str:
    return "plus" if direction == 1 else "minus"


def _other(direction: int) -> str:
    return "minus" if direction == 1 else "plus"


# --- V1: two orientations score differently; both sides' truth is saved ------


def test_v1_positive_slope_plus_side_wins_with_full_field_pattern():
    signal, label, mask = _monotone_market(seed=3, slope=0.02)
    cfg = BacktestConfig(top_n=3)
    decision = _rule_a(signal, label, mask, cfg=cfg, dates=DATES)
    ev = decision.evidence

    # §1: both sides differ (the §0.2 defect recorded one value twice).
    for key in _NUMERIC_SIDE_KEYS:
        assert key in ev, f"missing §1 field {key}"
    assert ev["val_reward_plus"] != ev["val_reward_minus"]
    assert ev["val_icir_plus"] != ev["val_icir_minus"]
    assert ev["train_reward_plus"] != ev["train_reward_minus"]
    assert ev["train_icir_plus"] != ev["train_icir_minus"]

    # §3 V1: direction = larger key side; branch decided on the first key.
    assert decision.direction == 1
    assert ev["direction"] == 1
    assert ev["tie_break_reason"] == "validation_reward"
    assert ev["canonical_orientation_locus"] is None

    # §1 source constraint: every number equals the independently computed
    # interleaved batch's element (no second scorer call, no recomputation).
    rewards, val_rewards, icir, val_icir, objectives = (
        _independent_oriented_batch(signal, label, mask, cfg)
    )
    assert ev["val_reward_plus"] == _jsonable(val_rewards[0])
    assert ev["val_reward_minus"] == _jsonable(val_rewards[1])
    assert ev["val_icir_plus"] == _jsonable(val_icir[0])
    assert ev["val_icir_minus"] == _jsonable(val_icir[1])
    assert ev["train_reward_plus"] == _jsonable(rewards[0])
    assert ev["train_reward_minus"] == _jsonable(rewards[1])
    assert ev["train_icir_plus"] == _jsonable(icir[0])
    assert ev["train_icir_minus"] == _jsonable(icir[1])
    assert ev["objectives_plus"] == [_jsonable(v) for v in objectives[0]]
    assert ev["objectives_minus"] == [_jsonable(v) for v in objectives[1]]
    assert len(ev["objectives_plus"]) == 8  # [B, 8] row stored whole

    # §1 context fields: the ACTUAL train/val dates, not just a cutoff.
    assert ev["train_signal_range"] == [0, TRAIN_END]
    assert ev["train_dates"] == DATES[0:TRAIN_END]
    assert ev["val_windows"] == [[10, TRAIN_END]]
    assert ev["val_dates"] == [DATES[10:TRAIN_END]]
    # §1: train_cutoff stays the existing top-level field.
    assert decision.train_cutoff == "2023-12-20"
    assert ev["reward_clip"] == [-10.0, 10.0]  # RewardConfig defaults

    # §1: applied exposure declaration and the §2 rule naming.
    assert isinstance(ev["applied_signal_basis"], str)
    assert "direction" in ev["applied_signal_basis"]
    assert "验证 reward 择向" in ev["selection_rule"] and "Rule A" in ev["selection_rule"]

    # §5 red line 2: legacy four keys keep chosen/opposite semantics; the
    # loser's true value is no longer the winner's value recorded twice.
    assert ev["val_reward"] == ev["val_reward_plus"]
    assert ev["val_icir"] == ev["val_icir_plus"]
    assert ev["val_reward_opposite"] == ev["val_reward_minus"]
    assert ev["val_icir_opposite"] == ev["val_icir_minus"]
    assert ev["val_reward_opposite"] != ev["val_reward"]

    assert decision.rule == RULE_REFERENCE_RETURN
    assert decision.reference_config is not None


def test_v1_negative_slope_minus_side_wins():
    signal, label, mask = _monotone_market(seed=4, slope=-0.02)
    cfg = BacktestConfig(top_n=3)
    decision = _rule_a(signal, label, mask, cfg=cfg, dates=DATES)
    ev = decision.evidence

    assert decision.direction == -1
    assert ev["tie_break_reason"] == "validation_reward"
    # §3 V1: direction = the larger-key side, whichever it is.
    plus_key = (ev["val_reward_plus"], ev["val_icir_plus"])
    minus_key = (ev["val_reward_minus"], ev["val_icir_minus"])
    assert minus_key > plus_key
    # §5 red line 2: chosen/opposite follow the direction.
    assert ev["val_reward"] == ev["val_reward_minus"]
    assert ev["val_reward_opposite"] == ev["val_reward_plus"]
    assert ev["val_reward_opposite"] != ev["val_reward"]

    rewards, val_rewards, icir, val_icir, objectives = (
        _independent_oriented_batch(signal, label, mask, cfg)
    )
    assert ev["val_reward_plus"] == _jsonable(val_rewards[0])
    assert ev["val_reward_minus"] == _jsonable(val_rewards[1])
    assert ev["objectives_minus"] == [_jsonable(v) for v in objectives[1]]


def test_v1_clip_floor_is_disclosed_not_bare():
    """§1 reward_clip row: the −10.0 the minus side hits in the positive
    fixture is the config clip floor (a per-window-clipped median), so it
    must be presented together with the clip bounds, never as a bare
    number."""
    signal, label, mask = _monotone_market(seed=3, slope=0.02)
    decision = _rule_a(signal, label, mask)
    ev = decision.evidence
    assert ev["reward_clip"] == [-10.0, 10.0]
    assert ev["val_reward_minus"] == ev["reward_clip"][0]


def test_v1_exactly_one_scoring_batch_per_decision(monkeypatch):
    """§0.3 red line 1 / §2: the decision and BOTH sides' evidence come
    from ONE score_many call — the mirrored second auto-selecting round
    (direction.py:195-197, eval_search.py:163-179) is gone."""
    calls: list[int] = []
    original = CandidateScorer.score_many

    def spy(self, specs, signals, target, windows, **kwargs):
        calls.append(len(specs))
        return original(self, specs, signals, target, windows, **kwargs)

    monkeypatch.setattr(CandidateScorer, "score_many", spy)
    signal, label, mask = _monotone_market(seed=3, slope=0.02)
    decision = _rule_a(signal, label, mask)
    assert decision.direction in (1, -1)
    assert len(calls) == 1, (
        f"expected exactly one score_many batch per direction decision, "
        f"got {len(calls)} (§0.3 red line 1)"
    )


# --- V2: mirroring exchanges the sides, applied exposure invariant ------------


def test_v2_mirror_exchanges_both_sides_and_keeps_applied_exposure():
    signal, label, mask = _monotone_market(seed=3, slope=0.02)
    base = _rule_a(signal, label, mask, dates=DATES)
    mirrored = _rule_a(-signal, label, mask, dates=DATES)

    # §3 V2: evidence_plus(s) == evidence_minus(-s) field-for-field (all
    # per-side numbers, objectives and date fields) and symmetrically.
    for key in _NUMERIC_SIDE_KEYS:
        stem = key[: -len("_plus")] if key.endswith("_plus") else key[: -len("_minus")]
        assert base.evidence[f"{stem}_plus"] == mirrored.evidence[f"{stem}_minus"], (
            f"mirror exchange failed for {stem}_plus"
        )
        assert base.evidence[f"{stem}_minus"] == mirrored.evidence[f"{stem}_plus"], (
            f"mirror exchange failed for {stem}_minus"
        )
    assert base.evidence["objectives_plus"] == mirrored.evidence["objectives_minus"]
    assert base.evidence["objectives_minus"] == mirrored.evidence["objectives_plus"]
    for key in (
        "train_signal_range", "train_dates", "val_windows", "val_dates",
        "reward_clip", "tie_break_reason", "selection_rule",
        "applied_signal_basis",
    ):
        assert base.evidence[key] == mirrored.evidence[key], (
            f"mirror-invariant field {key} changed under -s"
        )

    # §3 V2: direction flips; the applied economic exposure is identical.
    assert mirrored.direction == -base.direction
    applied_base = base.direction * signal
    applied_mirrored = mirrored.direction * (-signal)
    assert np.array_equal(applied_base, applied_mirrored)

    # §5 red line 2: the legacy chosen/opposite keys follow the flip —
    # the CHOSEN physical side is the same under mirroring (the applied
    # exposure is invariant), so chosen/opposite values are invariant.
    chosen = f"val_reward_{_side(base.direction)}"
    other = f"val_reward_{_other(base.direction)}"
    assert base.evidence["val_reward"] == base.evidence[chosen]
    assert base.evidence["val_reward_opposite"] == base.evidence[other]
    assert mirrored.evidence["val_reward"] == base.evidence[chosen]
    assert mirrored.evidence["val_reward_opposite"] == base.evidence[other]


def test_v2_mirror_on_the_canonical_branch():
    signal, label, mask = _canonical_market()
    cfg = BacktestConfig(top_n=3, **_FEE_FREE)
    base = _rule_a(signal, label, mask, cfg=cfg, dates=DATES)
    mirrored = _rule_a(-signal, label, mask, cfg=cfg, dates=DATES)

    assert base.evidence["tie_break_reason"] == "canonical_orientation"
    locus = base.evidence["canonical_orientation_locus"]
    assert isinstance(locus, dict)
    assert locus["window"] == [10, TRAIN_END]
    assert locus["scan_index"] == 0
    assert locus["value"] == pytest.approx(signal[0, 10])
    # The canonical branch's direction is the sign of that same first
    # observation (§2's scan order, unchanged).
    assert base.direction == (1 if locus["value"] > 0 else -1)
    assert base.direction == -1

    # Mirror: same locus position, negated value, flipped direction.
    m_locus = mirrored.evidence["canonical_orientation_locus"]
    assert m_locus["window"] == locus["window"]
    assert m_locus["scan_index"] == locus["scan_index"]
    assert m_locus["value"] == pytest.approx(-locus["value"])
    assert mirrored.direction == -base.direction
    assert np.array_equal(base.direction * signal, mirrored.direction * (-signal))


def test_v2_all_zero_canonical_locus():
    signal, label, mask = _zero_val_window_market()
    cfg = BacktestConfig(top_n=3, **_FEE_FREE)
    decision = _rule_a(signal, label, mask, cfg=cfg, dates=DATES)
    ev = decision.evidence
    assert ev["tie_break_reason"] == "canonical_orientation"
    assert ev["canonical_orientation_locus"] == "all_zero_nonfinite"
    assert decision.direction == 1


# --- V3: the test window cannot move any training-side evidence ---------------


def _assert_same_evidence(base: DirectionDecision, poisoned: DirectionDecision) -> None:
    assert poisoned.direction == base.direction
    assert poisoned.evidence == base.evidence, (
        "poisoning the test window changed the training-side evidence "
        "(FROZEN_C §3 V3)"
    )


def test_v3_poisoned_test_window_leaves_evidence_unchanged():
    signal, label, mask = _monotone_market(seed=3, slope=0.02)
    base = _rule_a(signal, label, mask, dates=DATES)

    # N4手法: blow up every test-period column.
    poisoned_signal = signal.copy()
    poisoned_label = label.copy()
    poisoned_signal[:, TRAIN_END + 2:] = -1e9 * poisoned_signal[:, TRAIN_END + 2:]
    poisoned_label[:, TRAIN_END + 2:] = -1e9 * poisoned_label[:, TRAIN_END + 2:]
    _assert_same_evidence(base, _rule_a(poisoned_signal, poisoned_label, mask, dates=DATES))

    # NaN / extreme variant on the same test columns.
    nan_signal = signal.copy()
    nan_label = label.copy()
    nan_signal[:, TRAIN_END + 2:] = np.nan
    nan_label[:, TRAIN_END + 2:] = 1e12
    _assert_same_evidence(base, _rule_a(nan_signal, nan_label, mask, dates=DATES))


def test_v3_poisoning_on_canonical_and_zero_windows():
    for market in (_canonical_market, _zero_val_window_market):
        signal, label, mask = market()
        cfg = BacktestConfig(top_n=3, **_FEE_FREE)
        base = _rule_a(signal, label, mask, cfg=cfg, dates=DATES)
        poisoned_signal = signal.copy()
        poisoned_label = label.copy()
        poisoned_signal[:, TRAIN_END + 2:] = np.nan
        poisoned_label[:, TRAIN_END + 2:] = -1e9
        _assert_same_evidence(base, _rule_a(poisoned_signal, poisoned_label, mask, cfg=cfg, dates=DATES))


# --- §2: the tie-break classifier mirrors the frozen Rule A key rule ----------


def test_rule_a_tie_break_classifier_branches():
    from ashare_model.direction import classify_rule_a_tie_break as classify

    # First key component separates (finite).
    assert classify(1.0, 0.5, 0.9, 0.1) == "validation_reward"
    assert classify(0.9, 0.1, 1.0, 0.5) == "validation_reward"
    # NaN degrades to -inf, still separable on the first component.
    assert classify(float("nan"), 1.0, 0.5, 0.1) == "validation_reward"
    assert classify(0.5, 1.0, float("nan"), 0.1) == "validation_reward"
    # First equal, second separates.
    assert classify(1.0, 0.5, 1.0, 0.4) == "validation_icir"
    assert classify(1.0, float("nan"), 1.0, 0.4) == "validation_icir"
    assert classify(float("-inf"), 0.3, float("nan"), 0.4) == "validation_icir"
    # Both equal (incl. both NaN -> both (-inf, -inf)).
    assert classify(1.0, 0.5, 1.0, 0.5) == "canonical_orientation"
    assert classify(float("nan"), float("nan"), float("nan"), float("nan")) == (
        "canonical_orientation"
    )


def test_rule_a_classifier_agrees_with_choose_direction():
    """§2: the evidence-layer classification never disagrees with the
    (byte-identical, untouched) ``_choose_direction`` comparison."""
    from ashare_model.direction import classify_rule_a_tie_break as classify

    rng = np.random.default_rng(11)
    signal = rng.normal(0.0, 1.0, size=(N_STOCKS, N_DATES))
    mask = np.ones((N_STOCKS, N_DATES), dtype=bool)
    values = [float("nan"), -2.0, -0.5, 0.0, 0.5, 2.0]
    for plus_reward in values:
        for plus_icir in values:
            for minus_reward in values:
                for minus_icir in values:
                    direction = CandidateScorer._choose_direction(
                        signal, list(VAL_WINDOWS), plus_reward, plus_icir,
                        minus_reward, minus_icir, mask,
                    )
                    branch = classify(plus_reward, plus_icir, minus_reward, minus_icir)
                    plus_key = (
                        plus_reward if math.isfinite(plus_reward) else -math.inf,
                        plus_icir if math.isfinite(plus_icir) else -math.inf,
                    )
                    minus_key = (
                        minus_reward if math.isfinite(minus_reward) else -math.inf,
                        minus_icir if math.isfinite(minus_icir) else -math.inf,
                    )
                    if branch == "validation_reward":
                        assert plus_key[0] != minus_key[0]
                        assert direction == (1 if plus_key[0] > minus_key[0] else -1)
                    elif branch == "validation_icir":
                        assert plus_key[0] == minus_key[0]
                        assert direction == (1 if plus_key[1] > minus_key[1] else -1)
                    else:
                        assert plus_key == minus_key


# --- rejected signals keep the direction contract ------------------------------


def test_rejected_near_constant_signal_keeps_direction_and_jsonable_evidence():
    """A near-constant signal never enters a batch; the §1 pattern stays
    present with None numbers and no tie-break, and the direction is the
    same +1 the replaced implementation returned (§0.3 red line 2:
    direction VALUES unchanged)."""
    signal = np.ones((N_STOCKS, N_DATES))
    label = np.zeros((N_STOCKS, N_DATES))
    mask = np.ones((N_STOCKS, N_DATES), dtype=bool)
    decision = _rule_a(signal, label, mask, dates=DATES)
    ev = decision.evidence
    assert decision.direction == 1
    for key in _NUMERIC_SIDE_KEYS:
        assert ev[key] is None
    assert ev["objectives_plus"] is None and ev["objectives_minus"] is None
    assert ev["tie_break_reason"] is None
    for key in ("val_reward", "val_icir", "val_reward_opposite", "val_icir_opposite"):
        assert key in ev  # §5 red line 2: legacy keys survive (value None)
    assert ev["train_dates"] == DATES[0:TRAIN_END]
    assert ev["direction"] == 1


# --- eval_search wiring: baseline provenance from the one batch ----------------


def test_baseline_provenance_carries_both_sides_from_one_batch(
    populated_db, monkeypatch
):
    """§0.2/§0.3 red line 1: ``eval_search.baseline_candidates`` no longer
    runs a whole mirrored second scoring round; its provenance carries
    the §1 pattern sourced from the single batch."""
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

    calls: list[int] = []
    original = CandidateScorer.score_many

    def spy(self, specs, signals, target, windows, **kwargs):
        calls.append(len(specs))
        return original(self, specs, signals, target, windows, **kwargs)

    monkeypatch.setattr(CandidateScorer, "score_many", spy)
    rows = baseline_candidates(loader, proto, fold, bt_cfg, ModelConfig(), RewardConfig())
    assert calls == [len(proto.baseline_signals)], (
        f"baseline_candidates must run exactly ONE scoring batch, got {calls}"
    )

    for row in rows:
        prov = row["direction_provenance"]
        ev = prov["evidence"]
        for key in _NUMERIC_SIDE_KEYS:
            assert key in ev, f"missing §1 field {key} in baseline provenance"
        assert ev["direction"] == prov["direction"] == row["direction"]
        if ev["val_reward_plus"] is None:
            # A rejected candidate never entered the batch (§1 numbers
            # absent, not fabricated): the tie-break is None and the
            # rejection is visible on the row itself.
            assert ev["tie_break_reason"] is None
            assert row["rejection_reasons"]
            continue
        assert ev["tie_break_reason"] in (
            "validation_reward", "validation_icir", "canonical_orientation",
        )
        chosen = f"val_reward_{_side(int(row['direction']))}"
        other = f"val_reward_{_other(int(row['direction']))}"
        assert ev["val_reward"] == ev[chosen]
        assert ev["val_reward_opposite"] == ev[other]
        assert isinstance(ev["train_dates"], list) and ev["train_dates"]
        assert all(isinstance(d, str) for d in ev["train_dates"])
        assert ev["reward_clip"] == [-10.0, 10.0]
