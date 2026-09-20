"""Direction-learning authority with provenance (p46, WP 10).

Two pre-fixed, independent direction rules over the **train window only**:

* ``reference_return`` (rule A) — the current rule formalized: the
  direction maximizing ``(val_reward, val_icir)`` on the training
  window's validation sub-windows, where ``val_reward`` is the
  reference-config basket return.  The decision is **delegated** to
  :meth:`CandidateScorer.score` — one implementation, zero semantic
  change — and the provenance records the exact reference configuration
  (``portfolio_config_provenance``) so any future flip is attributable
  to a named training configuration.  Its optimization target is
  explicit: *net return of the reference portfolio on the training
  window's validation tail*.
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
from typing import Any

import numpy as np

from ashare_data.config import BacktestConfig, RewardConfig

DIRECTION_RULE_VERSION = 1

RULE_REFERENCE_RETURN = "reference_return"
RULE_LABEL_IC = "label_ic"

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
) -> DirectionDecision:
    """Rule A: the current rule, delegated and provenance-wrapped.

    Runs the single scoring authority (:meth:`CandidateScorer.score`)
    twice — once on the signal, once on its mirror — so the decision and
    BOTH orientations' ``(val_reward, val_icir)`` are recorded; by the
    scorer's mirror symmetry the second run's chosen orientation is
    exactly the first run's opposite.
    """

    from .candidates import CandidateScorer, CandidateSpec
    from ashare_portfolio.execution_spec import portfolio_config_provenance

    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scorer = CandidateScorer(
        backtest_config, reward_config or RewardConfig()
    )
    spec = CandidateSpec(
        candidate_id=candidate,
        formula_text=candidate,
        source="direction",
        tokens=(),
    )
    common = dict(
        universe_mask=universe_mask,
        blocked_buy=blocked_buy,
        blocked_sell=blocked_sell,
        train_signal_range=train_signal_range,
        tie_break_keys=tie_break_keys,
        realized_ret=realized_ret,
        rebalance_mask=rebalance_mask,
    )
    chosen = scorer.score(spec, signal, target, val_windows, **common)
    opposite = scorer.score(
        spec, -signal, target, val_windows, **common
    )
    return DirectionDecision(
        candidate=candidate,
        direction=int(chosen.direction),
        rule=RULE_REFERENCE_RETURN,
        train_cutoff=str(train_cutoff),
        label=dict(label_def),
        missing_handling=MISSING_HANDLING,
        reference_config=portfolio_config_provenance(backtest_config),
        evidence={
            "val_reward": float(chosen.val_reward),
            "val_icir": float(chosen.val_icir),
            "val_reward_opposite": float(opposite.val_reward),
            "val_icir_opposite": float(opposite.val_icir),
        },
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
