"""Direction-learning authority with provenance (p46, WP 10; 16-C, p54).

Two pre-fixed, independent direction rules over the **train window only**:

* ``reference_return`` (rule A) — the current rule formalized: the
  direction maximizing ``(val_reward, val_icir)`` on the training
  window's validation sub-windows, where ``val_reward`` is the
  reference-config basket return.  The decision is **delegated** to
  :meth:`CandidateScorer.score_many` — one implementation, zero semantic
  change — and the provenance records the exact reference configuration
  (``portfolio_config_provenance``) so any future flip is attributable
  to a named training configuration.  Its optimization target is
  explicit: *net return of the reference portfolio on the training
  window's validation tail*.  Formal name (16-C / FROZEN_C §2):
  **参考配置下的验证 reward 择向（Rule A）**.  Since 16-C the decision and
  BOTH fixed orientations' raw evidence come out of the ONE interleaved
  scoring batch of that call — the loser's true numbers are recorded,
  never a second auto-selecting scorer round (the replaced p46 defect
  where the "opposite" evidence was the same winner value twice).
* ``label_ic`` (rule B) — the pure-signal control: the sign of the mean
  per-date cross-sectional rank IC between the signal and the **fixed
  causal label** on the train window.  The interface accepts no
  portfolio configuration, so capital / minimum order / turnover budget
  / reference cell cannot reach the decision.  IC comes from the single
  IC authority (:func:`ashare_model.diagnostics.rank_ic_stats`).

Both rules are mirror-stable (``-signal -> -direction``), share the
missing-value convention (non-finite cells excluded, PIT universe mask
gated), and never touch test-period columns — the caller slices and the
recorded ``train_cutoff`` names the boundary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig

DIRECTION_RULE_VERSION = 1

RULE_REFERENCE_RETURN = "reference_return"
RULE_LABEL_IC = "label_ic"

#: FROZEN_C §2 formal name of rule A's selection rule (naming discipline:
#: distinct from 16-B's "test-window cumulative-net-return orientation"
#: diagnostic, which never reproduces this production path).
REFERENCE_RETURN_RULE_NAME = "参考配置下的验证 reward 择向（Rule A）"

#: WP18 FROZEN_18_PLAN §2.2 formal name of rule B's selection rule
#: (minted by the WP18 frozen plan, pre-registered 2026-09-27): the sign
#: of the mean cross-sectional rank IC on the pre-validation slice of
#: the training window vs the fixed causal label.  Naming discipline
#: only — the decision itself (``label_ic_direction``) is untouched.
LABEL_IC_RULE_NAME = "训练窗验证段之前切片对固定因果标签的截面秩 IC 符号择向（Rule B）"

#: The applied-exposure declaration of every rule-A record (FROZEN_C §1):
#: the economic exposure actually applied downstream is
#: ``direction x raw signal`` — never a re-scored mirror.
APPLIED_SIGNAL_BASIS = "direction * raw_signal"

#: The shared missing-value convention of both rules (p46 B2): non-finite
#: signal/label cells are excluded and every cross-section is gated by
#: the PIT universe mask.
MISSING_HANDLING = "nan-excluded+pit-universe-mask"

_LABEL_NAME = "causal_target_returns"
_LABEL_BASIS = "adjusted_open_o2o"


@dataclass(frozen=True)
class DirectionDecision:
    """One direction outcome with its five-element provenance (p46 B2).

    ``reference_config`` is rule-A-only (the exact portfolio
    configuration whose validation return the direction optimizes);
    ``evidence`` carries the rule-specific numbers that produced the
    decision.
    """

    candidate: str
    direction: int
    rule: str
    train_cutoff: str
    label: dict[str, Any]
    missing_handling: str
    reference_config: dict[str, Any] | None
    evidence: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_orientation(
    signal: np.ndarray,
    universe_mask: np.ndarray,
    train_signal_range: tuple[int, int],
) -> int:
    """The mirror-stable canonical orientation of
    ``CandidateScorer._choose_direction``'s tie branch: the first finite
    non-zero (mask-gated) train-window observation is positive."""

    start, end = train_signal_range
    values = np.asarray(signal, dtype=np.float64)[:, start:end]
    mask = np.asarray(universe_mask, dtype=bool)[:, start:end]
    for value in np.where(mask, values, np.nan).T.ravel():
        v = float(value)
        if math.isfinite(v) and v != 0.0:
            return 1 if v > 0.0 else -1
    return 1


def label_ic_direction(
    candidate: str,
    signal: np.ndarray,
    target: np.ndarray,
    *,
    universe_mask: np.ndarray,
    train_signal_range: tuple[int, int],
    train_cutoff: str,
    label_def: dict[str, Any],
) -> DirectionDecision:
    """Rule B: direction = sign of the train-window mean rank IC.

    The IC is computed by :func:`diagnostics.rank_ic_stats` (the single
    IC authority) over the train-window columns only; a zero or
    non-finite mean falls back to the canonical orientation.
    """

    from .diagnostics import rank_ic_stats

    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    start, end = (int(train_signal_range[0]), int(train_signal_range[1]))
    stats = rank_ic_stats(
        signal[None, :, start:end],
        target[:, start:end],
        [str(i) for i in range(end - start)],
        names=[candidate],
        eligible=universe_mask[:, start:end],
    )[candidate]
    ic_mean = float(stats["ic_mean"])
    n_dates = int(stats["n_dates"])
    if math.isfinite(ic_mean) and ic_mean != 0.0:
        direction = 1 if ic_mean > 0.0 else -1
    else:
        direction = _canonical_orientation(
            signal, universe_mask, train_signal_range
        )
    return DirectionDecision(
        candidate=candidate,
        direction=int(direction),
        rule=RULE_LABEL_IC,
        train_cutoff=str(train_cutoff),
        label=dict(label_def),
        missing_handling=MISSING_HANDLING,
        reference_config=None,
        evidence={"ic_mean": ic_mean, "n_dates": n_dates},
    )


def _rule_a_key(reward: float, icir: float) -> tuple[float, float]:
    """The lexicographic Rule A key, mirroring
    ``CandidateScorer._choose_direction`` exactly (non-finite degrades to
    ``-inf``).  Kept in lockstep with the frozen rule text (FROZEN_C §2);
    the comparison itself stays in ``_choose_direction``, untouched."""

    return (
        reward if math.isfinite(reward) else -math.inf,
        icir if math.isfinite(icir) else -math.inf,
    )


def classify_rule_a_tie_break(
    plus_reward: float,
    plus_icir: float,
    minus_reward: float,
    minus_icir: float,
) -> str:
    """Which Rule A component decided the direction (FROZEN_C §2).

    ``"validation_reward"`` — the first key component separates (possibly
    after the ``-inf`` degradation of a NaN side); ``"validation_icir"``
    — the first components are equal and the second separates;
    ``"canonical_orientation"`` — both components are equal and the
    mirror-stable canonical-orientation scan decides.  Classification
    only; it never feeds the direction itself.
    """

    plus_key = _rule_a_key(plus_reward, plus_icir)
    minus_key = _rule_a_key(minus_reward, minus_icir)
    if plus_key[0] != minus_key[0]:
        return "validation_reward"
    if plus_key[1] != minus_key[1]:
        return "validation_icir"
    return "canonical_orientation"


def canonical_orientation_locus(
    signal: np.ndarray,
    universe_mask: np.ndarray,
    val_windows: list[tuple[int, int]],
) -> dict[str, Any] | str:
    """Locate the observation Rule A's tie-branch scan lands on
    (FROZEN_C §1 ``canonical_orientation_locus``).

    The scan order is exactly ``CandidateScorer._choose_direction``'s tie
    branch: windows in the given order, each window's mask-gated signal
    columns read ``values.T.ravel()`` (date-major), the first finite
    non-zero observation wins.  Returns ``"all_zero_nonfinite"`` when no
    eligible observation qualifies.
    """

    signal = np.asarray(signal, dtype=np.float64)
    universe_mask = np.asarray(universe_mask, dtype=bool)
    for window_index, (start, end) in enumerate(val_windows):
        values = np.where(
            universe_mask[:, start:end], signal[:, start:end], np.nan
        )
        for scan_index, value in enumerate(values.T.ravel()):
            v = float(value)
            if math.isfinite(v) and v != 0.0:
                return {
                    "val_window_index": int(window_index),
                    "window": [int(start), int(end)],
                    "scan_index": int(scan_index),
                    "value": v,
                }
    return "all_zero_nonfinite"


def _jsonable_float(value: Any) -> float | None:
    """Evidence convention: non-finite floats are stored as ``None`` (the
    repo's strict-JSON artifact convention, ``r01_baseline.rows_to_jsonable``);
    the raw NaN stays visible in the batch arrays and the ``reward_clip``
    disclosure."""

    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


_ORIENTED_NUMERIC_KEYS = (
    "val_reward_plus", "val_reward_minus",
    "val_icir_plus", "val_icir_minus",
    "train_reward_plus", "train_reward_minus",
    "train_icir_plus", "train_icir_minus",
)


def reference_return_provenance(
    *,
    candidate: str,
    direction: int,
    oriented_evidence: dict[str, Any] | None,
    signal: np.ndarray,
    universe_mask: np.ndarray,
    val_windows: list[tuple[int, int]],
    train_signal_range: tuple[int, int],
    train_cutoff: str,
    label_def: dict[str, Any],
    backtest_config: BacktestConfig,
    reward_config: RewardConfig,
    dates: Sequence[str] | None = None,
) -> DirectionDecision:
    """Assemble the rule-A ``DirectionDecision`` from ONE scoring batch.

    ``oriented_evidence`` is the per-candidate row a single
    :meth:`CandidateScorer.score_many` call emitted through
    ``oriented_evidence_out`` (both fixed orientations' raw numbers);
    ``direction`` is the direction that same call decided.  Nothing is
    re-scored and no reward number is recomputed here (FROZEN_C §1
    source constraint); the selection rule itself is untouched — this
    layer only classifies the tie-break and records context.
    """

    from ashare_portfolio.execution_spec import portfolio_config_provenance

    rejected = not oriented_evidence or oriented_evidence.get("rejected_reason")
    evidence: dict[str, Any] = {}
    if rejected:
        for key in _ORIENTED_NUMERIC_KEYS:
            evidence[key] = None
        evidence["objectives_plus"] = None
        evidence["objectives_minus"] = None
        tie_break: str | None = None
        locus: dict[str, Any] | str | None = None
    else:
        for key in _ORIENTED_NUMERIC_KEYS:
            evidence[key] = _jsonable_float(oriented_evidence[key])
        evidence["objectives_plus"] = (
            None
            if oriented_evidence["objectives_plus"] is None
            else [_jsonable_float(v) for v in oriented_evidence["objectives_plus"]]
        )
        evidence["objectives_minus"] = (
            None
            if oriented_evidence["objectives_minus"] is None
            else [_jsonable_float(v) for v in oriented_evidence["objectives_minus"]]
        )
        tie_break = classify_rule_a_tie_break(
            float(oriented_evidence["val_reward_plus"]),
            float(oriented_evidence["val_icir_plus"]),
            float(oriented_evidence["val_reward_minus"]),
            float(oriented_evidence["val_icir_minus"]),
        )
        locus = (
            canonical_orientation_locus(signal, universe_mask, val_windows)
            if tie_break == "canonical_orientation"
            else None
        )

    start, end = (int(train_signal_range[0]), int(train_signal_range[1]))
    chosen = "plus" if direction == 1 else "minus"
    opposite = "minus" if direction == 1 else "plus"
    evidence.update(
        {
            # §1 context fields (the ACTUAL dates, not just a cutoff).
            "train_signal_range": [start, end],
            "train_dates": (
                [str(d) for d in dates[start:end]] if dates is not None else None
            ),
            "val_windows": [[int(a), int(b)] for a, b in val_windows],
            "val_dates": (
                [[str(d) for d in dates[a:b]] for a, b in val_windows]
                if dates is not None
                else None
            ),
            # §1: val_reward is a per-window-clipped median — disclose the
            # bounds so a -10.0 reads as the floor it is.
            "reward_clip": [
                float(reward_config.reward_clip_low),
                float(reward_config.reward_clip_high),
            ],
            "direction": int(direction),
            "applied_signal_basis": APPLIED_SIGNAL_BASIS,
            "selection_rule": REFERENCE_RETURN_RULE_NAME,
            "tie_break_reason": tie_break,
            "canonical_orientation_locus": locus,
            # Legacy four keys (FROZEN_C §5 red line 2) with true
            # chosen/opposite semantics — same-source unpenalized values.
            "val_reward": evidence[f"val_reward_{chosen}"],
            "val_icir": evidence[f"val_icir_{chosen}"],
            "val_reward_opposite": evidence[f"val_reward_{opposite}"],
            "val_icir_opposite": evidence[f"val_icir_{opposite}"],
        }
    )
    return DirectionDecision(
        candidate=candidate,
        direction=int(direction),
        rule=RULE_REFERENCE_RETURN,
        train_cutoff=str(train_cutoff),
        label=dict(label_def),
        missing_handling=MISSING_HANDLING,
        reference_config=portfolio_config_provenance(backtest_config),
        evidence=evidence,
    )


def reference_return_direction(
    candidate: str,
    signal: np.ndarray,
    target: np.ndarray,
    *,
    universe_mask: np.ndarray,
    val_windows: list[tuple[int, int]],
    train_signal_range: tuple[int, int],
    train_cutoff: str,
    label_def: dict[str, Any],
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None = None,
    blocked_buy: np.ndarray | None = None,
    blocked_sell: np.ndarray | None = None,
    realized_ret: np.ndarray | None = None,
    rebalance_mask: np.ndarray | None = None,
    tie_break_keys: np.ndarray | None = None,
    dates: Sequence[str] | None = None,
) -> DirectionDecision:
    """Rule A (参考配置下的验证 reward 择向): delegated, one batch,
    both orientations' evidence kept.

    ONE :meth:`CandidateScorer.score_many` call interleaves the +signal
    and -signal of this candidate into a single reward batch, decides the
    direction with the unchanged Rule A comparison, and emits both
    sides' raw numbers; the decision and the evidence therefore come from
    the same scoring pass (16-C / FROZEN_C §0.3 red line 1).  The old
    second mirrored ``scorer.score`` round — whose "opposite" evidence
    was the same physical winner recorded twice — is gone.  ``dates``
    (optional, aligned with the signal's columns) records the actual
    train/validation dates in the evidence; without it those two fields
    are ``None`` and everything else is unchanged.
    """

    from .candidates import CandidateScorer, CandidateSpec

    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    effective_reward_config = reward_config or RewardConfig()
    scorer = CandidateScorer(backtest_config, effective_reward_config)
    spec = CandidateSpec(
        candidate_id=candidate,
        formula_text=candidate,
        source="direction",
        tokens=(),
    )
    oriented: list[dict[str, Any] | None] = []
    scores = scorer.score_many(
        [spec],
        [signal],
        target,
        val_windows,
        universe_mask=universe_mask,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        train_signal_range=train_signal_range,
        tie_break_keys=tie_break_keys,
        realized_ret=realized_ret,
        rebalance_mask=rebalance_mask,
        oriented_evidence_out=oriented,
    )
    return reference_return_provenance(
        candidate=candidate,
        direction=int(scores[0].direction),
        oriented_evidence=oriented[0],
        signal=signal,
        universe_mask=universe_mask,
        val_windows=val_windows,
        train_signal_range=train_signal_range,
        train_cutoff=train_cutoff,
        label_def=label_def,
        backtest_config=backtest_config,
        reward_config=effective_reward_config,
        dates=dates,
    )


def decide_direction(
    rule: str,
    candidate: str,
    signal: np.ndarray,
    target: np.ndarray,
    *,
    universe_mask: np.ndarray,
    train_signal_range: tuple[int, int],
    train_cutoff: str,
    label_def: dict[str, Any],
    **kwargs,
) -> DirectionDecision:
    """Dispatch the two pre-fixed rules; anything else is refused."""

    if rule == RULE_LABEL_IC:
        return label_ic_direction(
            candidate,
            signal,
            target,
            universe_mask=universe_mask,
            train_signal_range=train_signal_range,
            train_cutoff=train_cutoff,
            label_def=label_def,
        )
    if rule == RULE_REFERENCE_RETURN:
        return reference_return_direction(
            candidate,
            signal,
            target,
            universe_mask=universe_mask,
            train_signal_range=train_signal_range,
            train_cutoff=train_cutoff,
            label_def=label_def,
            **kwargs,
        )
    raise ValueError(
        f"unknown direction rule {rule!r}; the pre-fixed rules are "
        f"{RULE_REFERENCE_RETURN!r} and {RULE_LABEL_IC!r} (p46 B1)"
    )


def label_definition(policy) -> dict[str, Any]:
    """The standard label provenance block for a fold's rebalance policy."""

    return {
        "name": _LABEL_NAME,
        "basis": _LABEL_BASIS,
        "horizon": int(policy.horizon),
    }
