"""Champion/Challenger promotion gates (T4-01, P2-03/P2-04, P12).

A challenger (the top stitched trial of a current protocol artifact) is
promoted to champion **only** when all seven gates pass at once:

* **G1 data, formula & execution P0** — the artifact is current, bound to the
  immutable dataset, measured under a declared data regime on future or
  strictly locked windows, and the formula passed the hard eligibility
  gates (no rejection reasons) in every succeeded fold;
* **G2 statistical significance** — Deflated Sharpe >= threshold and the
  studentized max-t is significant at 95% (both over the stitched trial
  matrix);
* **G3 excess return & risk** — the stitched top trial satisfies the
  Sharpe / annualized-excess / drawdown / turnover / capacity bounds;
* **G4 cost & capacity stress** — the champion formula is re-scored
  under the documented stress grid (cost multipliers x capital
  multipliers) and every cell still satisfies the G3 risk bounds;
* **G5 future paper-trading window** — at least one complete paper
  window on future/locked data (>= ``paper_min_sessions`` sessions,
  already ended) exists for this exact formula;
* **G6 data credibility tier** (P2) — the challenger formula is traced
  back to the free-data tiers of its features
  (:mod:`ashare_model.data_tier`, ``docs/p2_data_tier_contract.md``):
  by default **only Tier A** features qualify (``allowed_data_tiers``,
  default ``("A",)``); Tier B is promotable only through an explicit
  separate comparison; Tier C (industry snapshot / ST approximation /
  placeholder) can never enter a promotion verdict.
* **G7 feature registry status** (P12) — the challenger formula's
  features must all be registry-promotable: every feature carries
  ``promotion_allowed=True`` and none is ``deprecated``
  (:func:`ashare_model.feature_registry.formula_registry_status_report`,
  ``docs/p12_promotion_enforcement_contract.md``).  Fail-closed: an
  untraceable formula (no tokens / undecodable / feature outside the
  vocabulary) is a rejection, never a pass.

Every gate reports ``passed`` plus human-readable ``reasons``, and the
verdict records the thresholds used, so a refusal is auditable.

Gate-set versions: G1-G6 were the unversioned **v1** era
(``PROMOTION_RULE_VERSION`` did not exist); **v2** (P12,
``docs/p12_promotion_enforcement_contract.md``) adds G7, and v2 verdicts
record ``promotion_rule_version``.  **v3** (P41,
``docs/p41_candidate_evidence_binding_draft.md`` §2 B4) upgrades G2 from
value-only to **binding + value** consumption: the artifact's dsr/max-t
blocks must carry the p41 identity (adjudicated candidate/seed, run,
window, benchmark, trial set) and must adjudicate the artifact's own top
trial — statistical evidence belonging to another candidate (e.g. a
historical trial's significance) is a refusal, never a pass.  The
promotion-gate numbering (G1-G7) is independent of the data-quality gate
numbering in ``ashare_data/gates.py``.
v4 (P43, ``docs/p43_statistical_input_integrity_draft.md``) tightens
G2's consumption to **numeric adjudication** (finite in-range
dsr/p, ``significant_95`` consistent with its own p-value) and to
**identity content verification** (run kind, window vs the top
trial's own date axis, benchmark definition and span, trial-set
composition vs the artifact's own stitched evidence, and the seed
class demanded by the candidate family).

Structure (fail-closed by construction, user ruling 2026-09-21):
each gate G1-G7 is ONE independent :func:`_gate_gN_*` function
returning ``{"passed": bool, "reasons": [...]}`` with its own local
state — :func:`evaluate_challenger` only assembles them.  The
historical single ~360-line inline block shared local variables
across gates, which is exactly how an unchecked NaN slipped through
G3/G4; the split removes the shared state by construction.  Every
numeric gate input is read through :func:`_finite_metric`, the
single fail-closed reader that converts a missing / boolean / NaN /
infinite value into that gate's refusal reason, so no threshold
comparison ever sees a non-finite number.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ashare_data.io_utils import atomic_write_json, read_json_safe

from .eval_corrections import BENCHMARK_DEFINITION, P40_MIN_WINDOW_DATES
from .identity import formula_hash
from .claim_record import (
    CALIBRATION_EVIDENCE_REGISTRATION,
    CalibrationEvidenceRegistry,
    STRING_A_NO_CLAIM_RECORD,
    STRING_C_UNQUALIFIED_TEMPLATE,
    claim_record_consistency_reasons,
    qualify_claim_record,
)

from .artifact_versions import version_matches
from .data_tier import (
    DATA_TIER_VERSION,
    TIER_TIME_RULES,
    DataTier,
    formula_data_tier_report,
)
from .evaluation import (
    PROTOCOL_VERSION,
    evaluate_formula,
    resolve_folds,
    stitch_oos_series,
    stitched_metrics,
)
from .feature_registry import (
    FEATURE_REGISTRY_VERSION,
    formula_registry_status_report,
)
from .reward import REWARD_VERSION
from .regime import HoldoutViolation, RegimeRegistry
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import (
    EXECUTION_SPEC_VERSION,
    portfolio_config_provenance,
    validate_portfolio_config_provenance,
)

# Version of the promotion gate-set semantics.  G1-G6 were the
# unversioned v1 era (no constant existed); v2 (P12,
# docs/p12_promotion_enforcement_contract.md) adds the G7
# feature_registry_status gate; v3 (P41,
# docs/p41_candidate_evidence_binding_draft.md §2 B4) makes G2 consume
# the candidate-evidence binding, not just the dsr/max-t values.
# Recorded in every verdict.  v4 (P43,
# docs/p43_statistical_input_integrity_draft.md) makes G2
# adjudicate numerically (finite in-range values, consistent
# significant_95) and verify the identity content (run kind,
# window, benchmark, trial set, seed class) against the
# artifact's own evidence.  v5 (WP17, reports/wp17/FROZEN_17.md)
# adds G2's fourth layer: the calibration-qualification layer —
# the artifact must carry the five-group claim record, its groups
# must be consistent with the artifact's own statistical blocks,
# and the qualification is RECOMPUTED from the registered
# calibration evidence (fail-closed: p=0.001 with an unqualified
# method cannot support promotion).
PROMOTION_RULE_VERSION = "5"


@dataclass(frozen=True)
class PromotionThresholds:
    """The documented promotion bar (recorded in every verdict)."""

    dsr_min: float = 0.95
    max_t_p_max: float = 0.05
    min_sharpe: float = 0.5
    min_excess_annualized: float = 0.0
    max_drawdown: float = 0.30
    max_average_turnover: float = 0.15  # mean daily turnover fraction
    max_capacity_utilization: float = 0.25
    paper_min_sessions: int = 60
    stress_cost_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    stress_capital_multipliers: tuple[float, ...] = (0.1, 1.0, 10.0)


#: Candidate families whose measurement is deterministic (no RNG in
#: the measurement path): their identity seed is ``None`` by
#: construction (p43 B9).  The r02/r03 cell panels prefix the family
#: with ``<cell>::``; :func:`candidate_seed_class` strips it before
#: reading the family.  Every other family — ``trained`` / the
#: ``*_search`` searchers / ``rl:`` / unknown names — is seeded and
#: demands an integer seed (the stricter default for unknown
#: families: fail closed towards demanding provenance).
DETERMINISTIC_CANDIDATE_FAMILIES = frozenset(
    {"benchmark", "baseline", "selfcheck", "control", "combo"}
)


def candidate_seed_class(candidate) -> str:
    """``"deterministic"`` or ``"seeded"`` — the seed class a
    candidate name demands (p43 B9).

    The class decides what a legal identity seed *is*: ``None``
    for deterministic families, a non-bool integer for seeded
    ones.  Binding a deterministic arm with a fabricated integer
    seed, or a seeded candidate with a wiped seed, is a refusal.
    """

    core = str(candidate).split("::")[-1]
    family = core.split(":", 1)[0]
    return (
        "deterministic"
        if family in DETERMINISTIC_CANDIDATE_FAMILIES
        else "seeded"
    )


def validate_allowed_tiers(allowed: tuple[str, ...]) -> tuple[str, ...]:
    """Validate the promotion's data-tier policy (P2-03/P2-04).

    Only Tier A (default) or explicitly Tier A+B are promotable; Tier C is
    research-display only and **never** enters a promotion verdict, and an
    unknown tier is a contract violation — both raise ``ValueError``
    instead of silently degrading.
    """

    known = {t.value for t in DataTier}
    unknown = sorted(set(allowed) - known)
    if unknown:
        raise ValueError(
            f"unknown data tiers in promotion policy: {unknown}; "
            f"allowed values are {sorted(known)}"
        )
    if "C" in allowed:
        raise ValueError(
            "Tier C data (industry snapshot / ST approximation / placeholder) "
            "is research-display only and can never enter promotion "
            "(docs/p2_data_tier_contract.md)"
        )
    return tuple(dict.fromkeys(allowed))


# --- paper-trading window registry -------------------------------------------


@dataclass(frozen=True)
class PaperWindow:
    """One completed paper-trading observation window (G5 evidence)."""

    window_id: str
    formula_hash: str
    start: str  # YYYY-MM-DD
    end: str  # YYYY-MM-DD
    sessions: int
    config_sha256: str = ""
    equity_path: str = ""
    recorded_at: str = ""


class PaperWindowRegistry:
    """Persisted registry of completed paper windows (atomic JSON).

    A window is registered only after it completed; the recorded
    ``config_sha256`` pins the configuration the window ran under, so a
    window cannot silently mix two configurations.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        payload = read_json_safe(self.path)
        self._windows: list[PaperWindow] = []
        for item in (payload or {}).get("windows", []) or []:
            if isinstance(item, dict) and item.get("window_id"):
                self._windows.append(PaperWindow(**item))

    def register(
        self,
        *,
        formula_hash: str,
        start: str,
        end: str,
        sessions: int,
        config_sha256: str = "",
        equity_path: str = "",
        window_id: str | None = None,
    ) -> PaperWindow:
        window = PaperWindow(
            window_id=window_id or f"paper-{len(self._windows) + 1:04d}",
            formula_hash=formula_hash,
            start=start,
            end=end,
            sessions=int(sessions),
            config_sha256=config_sha256,
            equity_path=equity_path,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self._windows.append(window)
        atomic_write_json(
            self.path,
            {
                "windows": [
                    {
                        "window_id": w.window_id,
                        "formula_hash": w.formula_hash,
                        "start": w.start,
                        "end": w.end,
                        "sessions": w.sessions,
                        "config_sha256": w.config_sha256,
                        "equity_path": w.equity_path,
                        "recorded_at": w.recorded_at,
                    }
                    for w in self._windows
                ]
            },
        )
        return window

    def windows(self) -> list[PaperWindow]:
        return list(self._windows)

    def completed(
        self, formula_hash: str, min_sessions: int, today: str | None = None
    ) -> PaperWindow | None:
        """The completed window for ``formula_hash`` with at least
        ``min_sessions`` sessions (``None`` when absent).  ``today`` is
        injectable for tests; default is the current UTC date."""

        today = today or datetime.now(timezone.utc).date().isoformat()
        for window in self._windows:
            if (
                window.formula_hash == formula_hash
                and window.sessions >= int(min_sessions)
                and window.end <= today
            ):
                return window
        return None


# --- stress grid --------------------------------------------------------------


def stress_champion(
    loader,
    fold_cfgs,
    tokens: list[int],
    direction: int,
    bt_cfg,
    *,
    cost_multipliers=(0.5, 1.0, 2.0),
    capital_multipliers=(0.1, 1.0, 10.0),
) -> dict:
    """Re-score the champion formula under the cost x capital stress grid.

    Each cell re-runs the champion through the identical engine path with
    the stressed ``BacktestConfig`` (all four fee rates multiplied by the
    cost multiplier, ``initial_capital`` by the capital multiplier, via
    ``dataclasses.replace``) and stitches the OOS rows exactly like the
    protocol.  The baseline cell (1.0 x 1.0) reproduces the artifact's
    measurement; the other cells show how the edge and the risk bounds
    hold up under stress.
    """

    import dataclasses

    grid = []
    for cost_mult in cost_multipliers:
        for capital_mult in capital_multipliers:
            stressed = dataclasses.replace(
                bt_cfg,
                commission_rate=float(bt_cfg.commission_rate) * float(cost_mult),
                stamp_tax_rate=float(bt_cfg.stamp_tax_rate) * float(cost_mult),
                transfer_fee_rate=(
                    float(bt_cfg.transfer_fee_rate) * float(cost_mult)
                ),
                slippage_rate=float(bt_cfg.slippage_rate) * float(cost_mult),
                initial_capital=float(bt_cfg.initial_capital) * float(capital_mult),
            )
            rows = []
            for fold_cfg in fold_cfgs:
                fold = resolve_folds(
                    [fold_cfg],
                    loader.dates,
                    frequency=stressed.rebalance_frequency,
                    horizon=stressed.target_horizon,
                )[0]
                metrics = evaluate_formula(
                    list(tokens), loader, fold, stressed, direction=int(direction)
                )
                if metrics is not None:
                    rows.append(
                        {
                            "candidate": "stress",
                            "seed": 0,
                            "fold_train_end": fold_cfg.train_end,
                            "fold_test_end": fold_cfg.test_end,
                            "failed": False,
                            **metrics,
                        }
                    )
            trials = stitch_oos_series(rows)
            cell: dict = {
                "cost_multiplier": float(cost_mult),
                "capital_multiplier": float(capital_mult),
                "n_trials": len(trials),
            }
            if trials:
                metrics = stitched_metrics(trials[0])
                cell.update(
                    {
                        key: metrics.get(key)
                        for key in (
                            "n_days",
                            "total_return",
                            "annual_return",
                            "sharpe",
                            "max_drawdown",
                            "excess_return",
                        )
                    }
                )
            grid.append(cell)
    return {"grid": grid, "n_cells": len(grid)}


def _annualized_excess(excess_return, n_days: int) -> float | None:
    if excess_return is None or not n_days or excess_return <= -1.0:
        return None
    return float((1.0 + excess_return) ** (252.0 / n_days) - 1.0)


def _stress_failures(stress, thresholds: PromotionThresholds) -> list[str]:
    """G4's per-cell adjudication (every metric read fail-closed).

    A cell whose ``n_days`` is absent or below 2 has no stitched OOS
    series (refused); a measured cell's sharpe / max_drawdown /
    excess_return are ALL required finite numbers read through
    :func:`_finite_metric` — a missing or NaN metric is an explicit
    refusal, never a bound that silently holds (``NaN < min`` is
    False, and the pre-2026-09-21 ``is not None and > threshold``
    form let a missing field pass).
    """

    failures: list[str] = []
    for cell in stress.get("grid", []):
        label = (
            f"cost x{cell.get('cost_multiplier')} "
            f"capital x{cell.get('capital_multiplier')}"
        )
        reasons: list[str] = []
        n_days = _finite_metric(
            cell, "n_days", reasons=reasons, required=False,
            what="stress cell",
        )
        if reasons:
            # present-but-non-finite n_days: an integrity refusal
            failures.extend(f"{label}: {reason}" for reason in reasons)
            continue
        if n_days is None or n_days < 2:
            failures.append(f"{label}: no stitched OOS series")
            continue
        sharpe = _finite_metric(
            cell, "sharpe", reasons=reasons, what="stress cell"
        )
        if sharpe is not None and sharpe < thresholds.min_sharpe:
            reasons.append(f"sharpe {sharpe} < {thresholds.min_sharpe}")
        mdd = _finite_metric(
            cell, "max_drawdown", reasons=reasons, what="stress cell"
        )
        if mdd is not None and mdd > thresholds.max_drawdown:
            reasons.append(
                f"max_drawdown {mdd:.3f} > {thresholds.max_drawdown}"
            )
        excess_return = _finite_metric(
            cell, "excess_return", reasons=reasons, what="stress cell"
        )
        excess_ann = (
            _annualized_excess(excess_return, n_days)
            if excess_return is not None
            else None
        )
        if excess_return is not None and (
            excess_ann is None or excess_ann < thresholds.min_excess_annualized
        ):
            reasons.append(
                f"annualized excess {excess_ann} < "
                f"{thresholds.min_excess_annualized}"
            )
        failures.extend(f"{label}: {reason}" for reason in reasons)
    return failures


# --- the promotion verdict -----------------------------------------------------


def _top_rows(artifact: dict) -> list[dict]:
    top = artifact.get("top_trial") or {}
    return [
        r
        for r in artifact.get("rows", [])
        if r.get("candidate") == top.get("candidate")
        and r.get("seed") == top.get("seed")
        and not r.get("failed")
    ]


def _gate(passed: bool, reasons: list[str]) -> dict:
    return {"passed": bool(passed), "reasons": list(reasons)}


def _identity_content_reasons(
    name: str, block: dict, artifact: dict, seed_class: str
) -> list[str]:
    """p43 B5-B9: verify the statistical identity's *content*
    against the artifact's own evidence.

    The p41 B4 gate checked that the identity fields exist and
    point at the artifact's top trial; these checks verify what
    they say — the run kind, the window against the top trial's
    own stitched date axis, the benchmark definition/span, the
    trial-set composition against the artifact's stitched and
    extra-trial counts, and the seed class demanded by the
    candidate family.
    """

    reasons: list[str] = []
    top = artifact.get("top_trial") or {}
    # B5: only current-run statistics can support promotion.
    run = block.get("run") or {}
    if run.get("kind") != "current_run":
        reasons.append(
            f"{name} statistical evidence comes from run kind "
            f"{run.get('kind')!r}, not the current run — only "
            "current-run statistics can support promotion "
            "(p43 B5)"
        )
    # B9: the seed class demanded by the candidate family —
    # seed=None on a deterministic family is the expected
    # identity, a fabricated integer is not; a seeded candidate
    # with a wiped seed is refused.
    top_seed = top.get("seed")
    if seed_class == "deterministic":
        if top_seed is not None:
            reasons.append(
                f"{name}: deterministic candidate "
                f"{top.get('candidate')!r} must carry seed=None "
                f"(got {top_seed!r}); a fabricated seed cannot "
                "satisfy the identity check (p43 B9)"
            )
    elif isinstance(top_seed, bool) or not isinstance(
        top_seed, (int, np.integer)
    ):
        reasons.append(
            f"{name}: seeded candidate {top.get('candidate')!r} "
            f"requires an integer seed, got {top_seed!r} "
            "(p43 B9)"
        )
    # B6: the window against the top trial's own date axis — the
    # dsr window IS that axis; the max_t window (the p40 joint
    # matrix window W, an intersection over current trials)
    # lies inside it and clears the p40 floor.
    window = block.get("window") or {}
    top_dates = top.get("dates")
    if not top_dates:
        reasons.append(
            f"{name}: the artifact's top trial carries no "
            "stitched date axis; the statistical window cannot "
            "be verified (p43 B6)"
        )
    else:
        w_first = window.get("first_date")
        w_last = window.get("last_date")
        w_n = window.get("n_dates")
        if name == "dsr":
            if (
                w_first != top_dates[0]
                or w_last != top_dates[-1]
                or w_n != len(top_dates)
            ):
                reasons.append(
                    f"{name} window {window!r} does not equal the "
                    f"top trial's own date axis ({top_dates[0]}.."
                    f"{top_dates[-1]}, n={len(top_dates)}) — "
                    "statistics measured on another axis are not "
                    "evidence for this challenger (p43 B6)"
                )
        else:
            if not (
                isinstance(w_n, int)
                and not isinstance(w_n, bool)
                and w_n >= P40_MIN_WINDOW_DATES
            ):
                reasons.append(
                    f"{name} window n_dates {w_n!r} is below "
                    f"P40_MIN_WINDOW_DATES={P40_MIN_WINDOW_DATES}: "
                    "the producer refuses such windows, so the "
                    "block cannot be genuine (p43 B6)"
                )
            if not (
                isinstance(w_first, str)
                and isinstance(w_last, str)
                and isinstance(w_n, int)
                and not isinstance(w_n, bool)
                and top_dates[0] <= w_first
                and w_first <= w_last
                and w_last <= top_dates[-1]
                and w_n <= len(top_dates)
            ):
                reasons.append(
                    f"{name} window {window!r} does not lie inside "
                    f"the top trial's date axis ({top_dates[0]}.."
                    f"{top_dates[-1]}, n={len(top_dates)}) — "
                    "statistics measured on another axis are not "
                    "evidence for this challenger (p43 B6)"
                )
    # B7: the benchmark is the engine's equal-weight universe on
    # the same span as the window.
    bench = block.get("benchmark") or {}
    if bench.get("definition") != BENCHMARK_DEFINITION:
        reasons.append(
            f"{name} benchmark definition "
            f"{bench.get('definition')!r} != "
            f"{BENCHMARK_DEFINITION!r} — statistics against "
            "another benchmark are not comparable evidence "
            "(p43 B7)"
        )
    if (
        bench.get("n_days") != window.get("n_dates")
        or bench.get("first_date") != window.get("first_date")
        or bench.get("last_date") != window.get("last_date")
    ):
        reasons.append(
            f"{name} benchmark span {bench!r} does not match its "
            f"own window {window!r} (p43 B7)"
        )
    # B8: the trial-set composition closes internally and against
    # the artifact's own stitched / extra-trial counts.
    ts = block.get("trial_set") or {}
    n_cur = ts.get("n_current")
    n_ext = ts.get("n_extra")
    counts = (n_cur, n_ext, ts.get("n_trials"), block.get("n_trials"))
    if not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0
        for v in counts
    ):
        reasons.append(
            f"{name} trial_set counts must be non-negative "
            f"integers, got n_current={n_cur!r} "
            f"n_extra={n_ext!r} n_trials={ts.get('n_trials')!r} "
            "(p43 B8)"
        )
    else:
        if n_cur < 1:
            reasons.append(
                f"{name} trial_set carries no current-run trial "
                f"(n_current={n_cur}) (p43 B8)"
            )
        if (
            n_cur + n_ext != ts.get("n_trials")
            or block.get("n_trials") != ts.get("n_trials")
        ):
            reasons.append(
                f"{name} trial_set arithmetic is incoherent: "
                f"n_current {n_cur} + n_extra {n_ext} != n_trials "
                f"{ts.get('n_trials')}, or the block's own "
                f"n_trials {block.get('n_trials')!r} disagrees "
                "(p43 B8)"
            )
        stitched = (artifact.get("stitched") or {}).get("n_trials")
        extra_recorded = artifact.get("dsr_extra_trials")
        if not isinstance(stitched, int) or not isinstance(
            extra_recorded, int
        ):
            reasons.append(
                f"{name}: the artifact carries no "
                "stitched/dsr_extra_trials count to verify the "
                "trial set against (p43 B8)"
            )
        elif n_cur != stitched or n_ext != extra_recorded:
            reasons.append(
                f"{name} trial_set (n_current={n_cur}, "
                f"n_extra={n_ext}) does not match the artifact's "
                f"own evidence (stitched n_trials={stitched!r}, "
                f"dsr_extra_trials={extra_recorded!r}) (p43 B8)"
            )
    return reasons


def _is_finite_number(value) -> bool:
    """True for a real int/float with a finite value (bool excluded).

    p43 N5: a NaN slips past every ``<`` comparison in the gates
    (``NaN < threshold`` is False), so gate inputs must be checked
    for finiteness explicitly; booleans are not measurements.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _finite_metric(
    payload,
    key: str,
    *,
    reasons: list[str],
    required: bool = True,
    what: str = "gate input",
) -> float | None:
    """Read one numeric gate input, fail-closed (p43 N5 made structural).

    THE single numeric entry point of the seven promotion gates (user
    ruling 2026-09-21): a value that is missing, ``None``, boolean, NaN
    or infinite is converted into a refusal reason appended to
    ``reasons`` — the caller's gate fails through its normal
    ``not reasons`` rule — and ``None`` is returned, so no threshold
    comparison ever sees a non-finite number and no gate can "forget"
    its own NaN check.  With ``required=False`` a MISSING value is a
    legal ``None`` read (no reason); a present-but-non-finite value is
    always a refusal either way.

    Returns the finite value as a float.
    """

    value = (payload or {}).get(key)
    if _is_finite_number(value):
        return float(value)
    if required or value is not None:
        reasons.append(
            f"{what} {key} is "
            + (
                "missing"
                if value is None
                else f"not a finite number ({value!r})"
            )
            + "; a missing or non-finite risk metric cannot support "
            "promotion (fail-closed)"
        )
    return None


def _top_row_formula_hash(row: dict, reasons: list[str]) -> str | None:
    """The formula identity of a top-trial row, fail-closed (P2-5).

    A row carrying neither ``formula`` tokens nor ``formula_text``
    has no hashable identity: the pre-2026-09-21 code propagated the
    helper's ``ValueError`` out of :func:`evaluate_challenger` (a
    whole-verdict crash), while the same malformed input in G6/G7 was
    a gate refusal.  The unified rule is the gate failure: append the
    refusal reason and return ``None``.
    """

    try:
        return formula_hash(row.get("formula"), row.get("formula_text"))
    except ValueError:
        reasons.append(
            "the top trial row carries neither formula tokens nor "
            "formula_text; its identity cannot be hashed — promotion "
            "requires a traceable formula"
        )
        return None


# --- the seven gate functions (G1-G7) -------------------------------------------
#
# Structural fail-closed reform (user ruling 2026-09-21): each gate is
# one independent function owning ONLY its local ``reasons`` list — no
# shared locals across gates — and every numeric input is read through
# _finite_metric so a missing/NaN metric is a refusal reason, never a
# silently-passed bound.  evaluate_challenger only assembles them.


def _gate_g1_data_formula_p0(
    artifact: dict,
    *,
    regime: RegimeRegistry | None,
    dataset_id: str | None,
    backtest_config,
    top_rows: list[dict],
) -> dict:
    """G1 — data, formula & execution P0 gate (fail-closed bindings).

    Version equality uses the single comparison primitive from
    ashare_model.artifact_versions; a missing field never matches and
    renders as ``None`` in the message (pinned by the F2
    characterization tests).  G1 checks protocol / reward / execution /
    portfolio_constructor only; the wider field set (model_version,
    searcher, imitation) belongs to classify_strategy.
    """

    reasons: list[str] = []
    if artifact.get("legacy") is True:
        reasons.append("artifact is explicitly marked legacy")
    if not version_matches(artifact.get("protocol_version"), PROTOCOL_VERSION):
        reasons.append(
            f"artifact protocol_version {artifact.get('protocol_version')} != "
            f"{PROTOCOL_VERSION}; only current stitched artifacts qualify"
        )
    if not version_matches(artifact.get("reward_version"), REWARD_VERSION):
        reasons.append(
            f"artifact reward_version {artifact.get('reward_version')} != "
            f"{REWARD_VERSION}"
        )
    if not version_matches(
        artifact.get("execution_version"), EXECUTION_SPEC_VERSION
    ):
        reasons.append(
            f"artifact execution_version {artifact.get('execution_version')} != "
            f"{EXECUTION_SPEC_VERSION}; legacy execution semantics cannot "
            "be promoted"
        )
    if not version_matches(
        artifact.get("portfolio_constructor_version"),
        PORTFOLIO_CONSTRUCTOR_VERSION,
    ):
        reasons.append(
            "artifact portfolio_constructor_version "
            f"{artifact.get('portfolio_constructor_version')} != "
            f"{PORTFOLIO_CONSTRUCTOR_VERSION}"
        )
    recorded_portfolio_config = artifact.get("portfolio_config")
    try:
        canonical_portfolio_config = validate_portfolio_config_provenance(
            recorded_portfolio_config
        )
    except ValueError as exc:
        reasons.append(f"artifact {exc}")
    else:
        if backtest_config is not None:
            runtime_portfolio_config = portfolio_config_provenance(
                backtest_config
            )
            if canonical_portfolio_config != runtime_portfolio_config:
                reasons.append(
                    "artifact portfolio_config does not match the current "
                    "promotion BacktestConfig"
                )
    art_dataset = artifact.get("dataset_id")
    if not art_dataset:
        reasons.append("artifact carries no dataset_id; legacy measurements "
                       "cannot be promoted")
    elif dataset_id and art_dataset != dataset_id:
        reasons.append(
            f"artifact dataset {art_dataset} != current dataset {dataset_id}"
        )
    if regime is None or regime.regime is None:
        reasons.append("no data regime declared; a final evaluation must use "
                       "future or strictly locked data")
    else:
        try:
            regime.assert_final_evaluation(artifact.get("folds") or [])
        except HoldoutViolation as exc:
            reasons.append(str(exc))
    top = artifact.get("top_trial")
    if not top:
        reasons.append("no stitched top trial in the artifact")
    elif not top_rows:
        reasons.append("no succeeded rows for the top trial")
    else:
        ineligible = [r for r in top_rows if not r.get("eligible", True)]
        rejected = [
            r for r in top_rows if r.get("rejection_reasons")
        ]
        if ineligible:
            reasons.append("formula failed the hard eligibility gates in "
                           f"{len(ineligible)} fold(s)")
        if rejected:
            reasons.append(
                "formula carried rejection reasons: "
                + "; ".join(
                    ",".join(r.get("rejection_reasons") or [])
                    for r in rejected
                )
            )
    return _gate(not reasons, reasons)


def _gate_g2_significance(
    artifact: dict,
    thresholds: PromotionThresholds,
    calibration_evidence: CalibrationEvidenceRegistry | None = None,
) -> dict:
    """G2 — statistical significance (stitched matrix), p41 B4: binding
    before values.

    The dsr/max-t blocks must carry the p41 identity (adjudicated
    candidate/seed, run kind, window, benchmark, trial set) and must
    adjudicate the artifact's own top trial — a historical trial's
    significance is never granted to the challenger under review, and
    pre-p41 value-only blocks fail closed.  p43 B5-B9: the identity's
    CONTENT is verified too (run kind, window vs the top trial's own
    date axis, benchmark definition/span, trial-set composition vs the
    artifact's own stitched evidence, and the seed class demanded by
    the candidate family), and seed=None is a legal identity value
    decided by that family — presence is checked by key, never by
    None-ness.  The numeric values (dsr, max-t p_value) are read
    through :func:`_finite_metric` (p43 N5): a non-finite statistic is
    a refusal, and a bare boolean significant_95 that contradicts its
    own p-value is not trusted.

    WP17 W3 (FROZEN_17 §3.1) adds the FOURTH layer after the three
    above: the calibration-qualification layer.  (1) the artifact must
    carry the five-group ``claim_record`` (string A); (2) verify-don't-
    trust — its statistics/method/task-actuals groups must match the
    artifact's own dsr/max_t/stitched content field by field (string
    B), and the record's own ``qualification.qualified`` flag is NEVER
    trusted; (3) the qualification is recomputed from the evidence
    object (string C when unqualified).  The numeric layer evaluates
    FIRST but is NOT sufficient: p=0.001 with an unqualified method
    fails G2, and the qualification layer can never rescue a numeric
    failure.
    """

    reasons: list[str] = []
    top = artifact.get("top_trial")
    seed_class = candidate_seed_class((top or {}).get("candidate"))
    for name, block in (("dsr", artifact.get("dsr") or {}),
                        ("max_t", artifact.get("max_t") or {})):
        missing = [
            field for field in (
                "adjudicated_candidate", "run",
                "window", "benchmark", "trial_set",
            )
            if block.get(field) is None
        ]
        if "adjudicated_seed" not in block:
            missing.append("adjudicated_seed")
        if missing:
            reasons.append(
                f"{name} block lacks the statistical identity ({missing}); "
                "unbound statistical evidence cannot support promotion "
                "(p41 §2 B2/B4)"
            )
            continue
        if (block["adjudicated_candidate"] != (top or {}).get("candidate")
                or block["adjudicated_seed"] != (top or {}).get("seed")):
            reasons.append(
                f"{name} adjudicates candidate="
                f"{block['adjudicated_candidate']!r} seed="
                f"{block['adjudicated_seed']!r}, not the artifact's top "
                "trial — statistical evidence bound to another candidate "
                "is not evidence for this challenger (p41 §2 B3/B4)"
            )
            continue
        reasons.extend(
            _identity_content_reasons(name, block, artifact, seed_class)
        )
    dsr = artifact.get("dsr") or {}
    mt = artifact.get("max_t") or {}
    if not reasons:
        # p43 N5: the gate adjudicates by the numeric threshold and
        # requires finite in-range self-consistent values — a NaN
        # slips past every < comparison, and a bare boolean can
        # contradict its own p-value, so neither is trusted.
        if dsr.get("dsr") is None:
            reasons.append("no Deflated Sharpe in the artifact")
        else:
            dsr_value = _finite_metric(
                dsr, "dsr", reasons=reasons, what="dsr block"
            )
            if dsr_value is not None and not 0.0 <= dsr_value <= 1.0:
                reasons.append(
                    f"dsr {dsr_value} outside its [0, 1] value range "
                    "(p43 N5)"
                )
            elif dsr_value is not None and dsr_value < thresholds.dsr_min:
                reasons.append(
                    f"dsr {dsr_value:.3f} < {thresholds.dsr_min}"
                )
        if mt.get("p_value") is None:
            reasons.append("no max-t in the artifact")
        else:
            p_value = _finite_metric(
                mt, "p_value", reasons=reasons, what="max_t block"
            )
            if p_value is not None and not 0.0 < p_value <= 1.0:
                reasons.append(
                    f"max-t p_value {p_value} outside its (0, 1] value "
                    "range (p43 N5)"
                )
            elif p_value is not None:
                significant = mt.get("significant_95")
                if not isinstance(significant, bool):
                    reasons.append(
                        f"significant_95 must be a boolean, got "
                        f"{significant!r} (p43 N5)"
                    )
                elif significant != (p_value <= thresholds.max_t_p_max):
                    reasons.append(
                        f"significant_95={significant} contradicts "
                        f"p_value={p_value} against the threshold "
                        f"{thresholds.max_t_p_max} (p43 N5)"
                    )
                elif p_value > thresholds.max_t_p_max:
                    reasons.append(
                        f"max-t not significant (p={p_value} > "
                        f"{thresholds.max_t_p_max})"
                    )
    if not reasons:
        # WP17 W3 (FROZEN_17 §3.1): the fourth layer — qualification,
        # fail-closed.  A tiny p-value cannot support promotion unless
        # the significance method carries calibrated-evidence
        # qualification for THIS artifact's operating point.
        record = artifact.get("claim_record")
        if not isinstance(record, dict) or any(
            not isinstance(record.get(group), dict)
            for group in (
                "statistics", "method", "task_actuals",
                "calibration_evidence", "qualification",
            )
        ):
            reasons.append(STRING_A_NO_CLAIM_RECORD)
        else:
            # Step 2 (verify-don't-trust): the record's groups 1-3
            # against the artifact's own statistical content.
            reasons.extend(
                claim_record_consistency_reasons(record, artifact)
            )
            if not reasons:
                # Step 3: recompute — the artifact's own qualification
                # flag is never consulted.
                recomputed = qualify_claim_record(
                    record, calibration_evidence
                )
                if not recomputed["qualified"]:
                    task = record.get("task_actuals") or {}
                    method = record.get("method") or {}
                    reasons.append(
                        STRING_C_UNQUALIFIED_TEMPLATE.format(
                            n=task.get("n_common_dates"),
                            k=task.get("n_pool_trials"),
                            np=method.get("n_perms"),
                            bs=method.get("block_size"),
                            policy=task.get("missing_fold_policy"),
                            codes=", ".join(recomputed["reason_codes"]),
                        )
                    )
    return _gate(not reasons, reasons)


def _gate_g3_excess_and_risk(
    artifact: dict, thresholds: PromotionThresholds
) -> dict:
    """G3 — excess return & risk bounds on the stitched top trial.

    Every risk metric is REQUIRED and read through :func:`_finite_metric`
    (user ruling 2026-09-21, unified with the historical sharpe
    behaviour — "risk metric missing = refuse"): sharpe, excess_return,
    n_days, max_drawdown, average_turnover_mean and
    capacity_utilization_max must all be finite numbers.  The
    ProtocolResultArtifact schema types ``top_trial`` as a plain dict
    with no inner schema, so nothing inside it is schema-guaranteed;
    the producer (``eval_metrics.top_trial``) always writes sharpe /
    max_drawdown / excess_return / n_days, and writes
    average_turnover_mean / capacity_utilization_max whenever the fold
    rows carry turnover/capacity measurements — promotion demands the
    complete set, and a NaN (which slips past every ``<`` comparison)
    or an absent metric is an explicit refusal reason.
    """

    top = artifact.get("top_trial")
    if not top:
        return _gate(False, ["no top trial to constrain"])
    reasons: list[str] = []
    sharpe = _finite_metric(
        top, "sharpe", reasons=reasons, what="stitched top trial"
    )
    if sharpe is not None and sharpe < thresholds.min_sharpe:
        reasons.append(
            f"stitched sharpe {sharpe} < {thresholds.min_sharpe}"
        )
    excess_return = _finite_metric(
        top, "excess_return", reasons=reasons, what="stitched top trial"
    )
    n_days = _finite_metric(
        top, "n_days", reasons=reasons, what="stitched top trial"
    )
    if excess_return is not None and n_days is not None:
        excess_ann = _annualized_excess(excess_return, n_days)
        if excess_ann is None or excess_ann < thresholds.min_excess_annualized:
            reasons.append(
                f"annualized excess return {excess_ann} < "
                f"{thresholds.min_excess_annualized}"
            )
    mdd = _finite_metric(
        top, "max_drawdown", reasons=reasons, what="stitched top trial"
    )
    if mdd is not None and mdd > thresholds.max_drawdown:
        reasons.append(f"max drawdown {mdd:.3f} > {thresholds.max_drawdown}")
    turnover = _finite_metric(
        top, "average_turnover_mean", reasons=reasons,
        what="stitched top trial",
    )
    if turnover is not None and turnover > thresholds.max_average_turnover:
        reasons.append(
            f"average turnover {turnover:.4f} > "
            f"{thresholds.max_average_turnover}"
        )
    capacity = _finite_metric(
        top, "capacity_utilization_max", reasons=reasons,
        what="stitched top trial",
    )
    if capacity is not None and capacity > thresholds.max_capacity_utilization:
        reasons.append(
            f"capacity utilization {capacity:.3f} > "
            f"{thresholds.max_capacity_utilization}"
        )
    return _gate(not reasons, reasons)


def _gate_g4_cost_capacity_stress(
    stress: dict | None, thresholds: PromotionThresholds
) -> dict:
    """G4 — cost & capacity stress (per-cell adjudication in
    :func:`_stress_failures`, every metric through
    :func:`_finite_metric`)."""

    if stress is None:
        return _gate(False, ["no cost/capacity stress result provided"])
    failures = _stress_failures(stress, thresholds)
    return _gate(not failures, failures)


def _gate_g5_paper_window(
    top_rows: list[dict],
    paper_registry: PaperWindowRegistry | None,
    regime: RegimeRegistry | None,
    thresholds: PromotionThresholds,
    today: str | None,
) -> dict:
    """G5 — at least one complete future paper-trading window.

    The top-trial row's formula identity is hashed fail-closed (P2-5):
    a row carrying neither ``formula`` tokens nor ``formula_text`` is a
    GATE REFUSAL (the pre-2026-09-21 code crashed the whole evaluation
    with the hasher's ``ValueError``), the same fail-closed semantics
    G6/G7 apply to an untraceable formula.
    """

    reasons: list[str] = []
    if not top_rows:
        reasons.append("no top trial to promote")
        return _gate(False, reasons)
    fh = _top_row_formula_hash(top_rows[-1], reasons)
    if fh is None:
        return _gate(False, reasons)
    if paper_registry is None:
        reasons.append("no paper-window registry provided")
        return _gate(False, reasons)
    window = paper_registry.completed(
        fh, thresholds.paper_min_sessions, today=today
    )
    if window is None:
        reasons.append(
            f"no completed paper window with >= "
            f"{thresholds.paper_min_sessions} sessions for this formula"
        )
    else:
        kind = "dev"
        if regime is not None:
            kind = regime.classify_window(window.start, window.end)
        if kind == "dev":
            reasons.append(
                f"paper window {window.start}..{window.end} ran on "
                "dev/validation data; the observation window must be "
                "future or locked data"
            )
    return _gate(not reasons, reasons)


def _gate_g6_data_tier(
    top_rows: list[dict], allowed: tuple[str, ...]
) -> tuple[dict, dict | None]:
    """G6 — data credibility tier (P2-03/P2-04): the challenger formula
    is traced back to the tiers of its features; every feature must
    belong to the allowed tiers (default: Tier A only).  Tier C can
    never be allowed here (validate_allowed_tiers rejects it up front).

    Returns ``(gate, tier_report)`` — the report is recorded in the
    verdict's ``data_tier_policy`` block.
    """

    reasons: list[str] = []
    tier_report = None
    if top_rows:
        last = top_rows[-1]
        tier_report = formula_data_tier_report(
            tokens=last.get("formula"),
            feature_name=last.get("formula_text"),
            # t46 Plan A: grammar-5-era artifacts decode against the frozen
            # grammar-5 layout; the generation rides on the artifact row.
            grammar_version=last.get("grammar_version"),
        )
    if tier_report is None:
        reasons.append(
            "no traceable formula for the top trial; promotion requires "
            "a formula whose features all belong to the allowed data tiers"
        )
    else:
        for feature, tier in sorted(tier_report["per_feature"].items()):
            if tier not in allowed:
                reasons.append(
                    f"feature {feature} uses data tier {tier}, which is not "
                    f"in the allowed tiers {list(allowed)}"
                )
    return _gate(not reasons, reasons), tier_report


def _gate_g7_feature_registry(
    top_rows: list[dict],
) -> tuple[dict, dict | None]:
    """G7 — feature registry status (P12): the challenger formula's
    features must all be registry-promotable (promotion_allowed, not
    deprecated).  Same top-trial source as G6, so the artifact loading
    path is covered by construction (newly sampled and historical
    formulas both arrive as artifact top rows).  Fail-closed: an
    untraceable formula or a registry gap is a rejection.  The
    promotion-gate numbering (G1-G7) is independent of the data-quality
    gate numbering in ashare_data/gates.py.

    Returns ``(gate, registry_report)`` — the report is recorded in the
    verdict's ``registry_status_policy`` block.
    """

    reasons: list[str] = []
    registry_report = None
    if top_rows:
        last = top_rows[-1]
        registry_report = formula_registry_status_report(
            tokens=last.get("formula"),
            feature_name=last.get("formula_text"),
            grammar_version=last.get("grammar_version"),
        )
    if registry_report is None or not registry_report["traceable"]:
        reasons.append(
            "no traceable formula for registry status; promotion requires "
            "a formula whose every feature is registry-promotable"
        )
    else:
        for name, status in sorted(registry_report["per_feature"].items()):
            if not status["promotion_allowed"]:
                reasons.append(
                    f"feature {name} is not promotion_allowed "
                    "(feature registry status)"
                )
                if status["deprecated"]:
                    reasons.append(
                        f"feature {name} is deprecated "
                        "(feature registry status)"
                    )
    return _gate(not reasons, reasons), registry_report


def _qualification_summary(
    artifact: dict, calibration_evidence: CalibrationEvidenceRegistry | None
) -> dict | None:
    """G2's recomputed qualification summary for the verdict (WP17 W3,
    FROZEN_17 §3.4): qualified / reason codes / covering / blocking —
    recomputed from the evidence object with the same §2 algorithm the
    gate runs, never read off the artifact's own flag.  ``None`` when
    the artifact carries no claim record (string A case)."""

    record = artifact.get("claim_record")
    if not isinstance(record, dict):
        return None
    recomputed = qualify_claim_record(record, calibration_evidence)
    return {
        "qualified": recomputed["qualified"],
        "reason_codes": list(recomputed["reason_codes"]),
        "covering_scenarios": list(recomputed["covering_scenarios"]),
        "blocking_scenarios": list(recomputed["blocking_scenarios"]),
    }


def evaluate_challenger(
    artifact: dict,
    *,
    regime: RegimeRegistry | None = None,
    dataset_id: str | None = None,
    paper_registry: PaperWindowRegistry | None = None,
    stress: dict | None = None,
    thresholds: PromotionThresholds | None = None,
    allowed_data_tiers: tuple[str, ...] = ("A",),
    today: str | None = None,
    backtest_config=None,
    calibration_evidence: CalibrationEvidenceRegistry | None = None,
) -> dict:
    """Apply the seven promotion gates to a current protocol artifact.

    Every gate returns ``{"passed": bool, "reasons": [...]}``; the
    challenger is promoted only when all seven pass.  This function
    ONLY assembles the seven independent :func:`_gate_gN_*` functions
    (the structural fail-closed reform of 2026-09-21) — no gate logic
    lives here.  ``allowed_data_tiers`` is the P2 data-tier policy
    (default Tier A only; Tier B requires an explicit separate
    comparison; Tier C raises).  ``today`` is injectable for tests
    (paper-window completion check).  ``calibration_evidence`` is the
    WP17 W2 object-form evidence registry G2's fourth layer consumes
    (``None``/unavailable => fail-closed, EVIDENCE_NOT_REGISTERED); a
    caller string never grants qualification.
    """

    thresholds = thresholds or PromotionThresholds()
    allowed = validate_allowed_tiers(allowed_data_tiers)
    top = artifact.get("top_trial")
    top_rows = _top_rows(artifact) if top else []

    tier_gate, tier_report = _gate_g6_data_tier(top_rows, allowed)
    registry_gate, registry_report = _gate_g7_feature_registry(top_rows)
    # G5 independently hashes the same top row and reports the failure
    # reason when the identity is missing; the challenger mirror below only
    # needs the hash (None then), so no reason list is carried here.
    challenger_formula_hash = (
        _top_row_formula_hash(top_rows[-1], []) if top_rows else None
    )
    gates: dict[str, dict] = {
        "data_formula_p0": _gate_g1_data_formula_p0(
            artifact,
            regime=regime,
            dataset_id=dataset_id,
            backtest_config=backtest_config,
            top_rows=top_rows,
        ),
        "significance": _gate_g2_significance(
            artifact, thresholds, calibration_evidence
        ),
        "excess_and_risk": _gate_g3_excess_and_risk(artifact, thresholds),
        "cost_capacity_stress": _gate_g4_cost_capacity_stress(
            stress, thresholds
        ),
        "paper_window": _gate_g5_paper_window(
            top_rows, paper_registry, regime, thresholds, today
        ),
        "data_tier": tier_gate,
        "feature_registry_status": registry_gate,
    }

    return {
        "promoted": all(g["passed"] for g in gates.values()),
        "challenger": {
            "candidate": top.get("candidate") if top else None,
            "seed": top.get("seed") if top else None,
            "formula_hash": challenger_formula_hash,
        },
        "data_tier_policy": {
            "data_tier_version": DATA_TIER_VERSION,
            "allowed_tiers": list(allowed),
            "formula_data_tier": tier_report,
            "time_rules": dict(TIER_TIME_RULES),
        },
        "promotion_rule_version": PROMOTION_RULE_VERSION,
        # WP17 W3 (FROZEN_17 §3.4): the verdict carries the artifact's
        # claim record (copy) and G2's recomputed qualification summary
        # — the user-visible formal-conclusion surface.
        "claim_record": (
            json.loads(json.dumps(artifact.get("claim_record")))
            if isinstance(artifact.get("claim_record"), dict)
            else None
        ),
        "qualification": _qualification_summary(artifact, calibration_evidence),
        "registry_status_policy": {
            "feature_registry_version": FEATURE_REGISTRY_VERSION,
            "report": registry_report,
        },
        "gates": gates,
        "thresholds": {
            "dsr_min": thresholds.dsr_min,
            "max_t_p_max": thresholds.max_t_p_max,
            "min_sharpe": thresholds.min_sharpe,
            "min_excess_annualized": thresholds.min_excess_annualized,
            "max_drawdown": thresholds.max_drawdown,
            "max_average_turnover": thresholds.max_average_turnover,
            "max_capacity_utilization": thresholds.max_capacity_utilization,
            "paper_min_sessions": thresholds.paper_min_sessions,
            "stress_cost_multipliers": list(thresholds.stress_cost_multipliers),
            "stress_capital_multipliers": list(
                thresholds.stress_capital_multipliers
            ),
        },
    }


def assert_consumable_protocol_artifact(runs_root, artifact: dict) -> str:
    """Verify a protocol artifact payload is consumable for promotion.

    A03 (p27 read side, user ruling of 2026-09-21): the promotion CLI
    reads the CONVENIENCE MIRROR file, and a mirror is a cache, never
    the formal evidence — exactly like the formal backtest / paper-sim
    entries, the consumed payload must pass BOTH read-side assertions
    before any gate work:

    * the source run named by the artifact's lineage has a lifecycle
      chain at or above SEARCH_PLAN_ADMITTED
      (:func:`ashare_model.lifecycle.assert_qualified_source_run`,
      p27 R7/I10 — a current schema alone is never qualification, and
      engineering/audit products are never promotion evidence);
    * the payload's content id IS registered in that run's index as a
      ``protocol`` artifact
      (:func:`ashare_model.artifact_writer.assert_registered_boundary_artifact`,
      p27 I7 — a tampered, copied or foreign-run mirror is refused).

    Returns the registered artifact id; raises
    ``LifecycleError`` / ``RunStoreError`` / ``ArtifactSchemaError`` on
    refusal.  Extracted as its own function so the consumption rule is
    unit-testable without driving the whole CLI.
    """

    from .artifact_writer import assert_registered_boundary_artifact
    from .lifecycle import assert_qualified_source_run
    from .run_store import RunStore

    root = Path(runs_root)
    spec_id = str(artifact.get("spec_id", ""))
    run_id = str(artifact.get("run_id", ""))
    assert_qualified_source_run(root, spec_id, run_id)
    store = RunStore(root)
    return assert_registered_boundary_artifact(
        store, spec_id, run_id, artifact, artifact_type="protocol"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact", default=None, help="current protocol result JSON"
    )
    parser.add_argument("--config", default=None, help="project config (for stress)")
    parser.add_argument("--output", default="data/promotion_verdict.json")
    parser.add_argument("--regime", default="data/holdout_registry.json")
    parser.add_argument("--paper", default="data/paper_windows.json")
    parser.add_argument(
        "--calibration-evidence",
        default=None,
        help=(
            "calibration evidence JSON for G2's wp17 qualification layer "
            "(default: the registered evidence path; unregistered or "
            "unreadable => fail-closed)"
        ),
    )
    parser.add_argument("--min-eligible", type=int, default=None)
    parser.add_argument(
        "--register-paper",
        nargs=4,
        metavar=("START", "END", "SESSIONS", "EQUITY_PATH"),
        help="register a completed paper window (with --formula-hash) and exit",
    )
    parser.add_argument("--formula-hash", default=None)
    parser.add_argument("--config-sha256", default="")
    parser.add_argument(
        "--allow-tier-b",
        action="store_true",
        help="P2-03: run the separate Tier A+B comparison (default: Tier A only)",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="RunStore root holding the protocol artifact's source run "
        "(default: the configured data_dir; A03 follow-up, user ruling "
        "1a: the source run's lifecycle chain is verified before any "
        "gate evaluation)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]

    if args.register_paper:
        if not args.formula_hash:
            print("--register-paper requires --formula-hash", file=sys.stderr)
            return 1
        PaperWindowRegistry(root / args.paper).register(
            formula_hash=args.formula_hash,
            start=args.register_paper[0],
            end=args.register_paper[1],
            sessions=int(args.register_paper[2]),
            equity_path=args.register_paper[3],
            config_sha256=args.config_sha256,
        )
        print(f"paper window registered in {root / args.paper}")
        return 0

    if not args.artifact:
        print("--artifact is required unless --register-paper is used",
              file=sys.stderr)
        return 1

    artifact_path = Path(args.artifact)
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    from .artifact_schemas import ProtocolResultArtifact
    from .artifact_writer import read_boundary_artifact

    loaded = read_boundary_artifact(
        artifact_path, model_cls=ProtocolResultArtifact, formal=True
    )
    if loaded is None:
        raise FileNotFoundError(f"protocol artifact not found: {artifact_path}")
    artifact, _ = loaded

    # A03 follow-up (user ruling 1a of 2026-09-12; p27 R7/I10 consumption
    # side) + the 2026-09-21 registration check (p27 I7 read side): the
    # source run named by the artifact's lineage must have a lifecycle
    # chain at or above SEARCH_PLAN_ADMITTED AND the consumed mirror
    # payload must be that run's REGISTERED protocol artifact (the
    # mirror is a cache — formal consumption resolves identity from the
    # RunStore).  Read-only replay, before any gate work.
    from .artifact_schemas import ArtifactSchemaError
    from .lifecycle import LifecycleError, RunStoreError

    if args.runs_root:
        runs_root = Path(args.runs_root)
    else:
        try:
            from ashare_data.config import load_config, make_data_config

            data_config = make_data_config(
                load_config(None, project_root=root), root
            )
            runs_root = data_config.data_dir
        except Exception as exc:
            print(
                "promotion: cannot resolve the RunStore root for the "
                "source-run lifecycle check — pass --runs-root "
                f"explicitly ({exc})",
                file=sys.stderr,
            )
            return 1
    try:
        assert_consumable_protocol_artifact(runs_root, artifact)
    except (LifecycleError, RunStoreError, ArtifactSchemaError) as exc:
        print(
            f"promotion: source-run/artifact-registration check refused: {exc}",
            file=sys.stderr,
        )
        return 1

    regime = RegimeRegistry(root / args.regime)
    paper = PaperWindowRegistry(root / args.paper)

    # WP17 W3 (FROZEN_17 §3.1): the calibration evidence is loaded as an
    # OBJECT from a registered path (the G1 RegimeRegistry precedent);
    # None/unavailable => fail-closed (EVIDENCE_NOT_REGISTERED).  A
    # path that is not the registered one builds an unregistered
    # registry object — still fail-closed.
    evidence_path = Path(
        args.calibration_evidence
        or CALIBRATION_EVIDENCE_REGISTRATION[0]["source"]
    )
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    try:
        calibration_evidence = CalibrationEvidenceRegistry.from_file(
            evidence_path
        )
    except (OSError, ValueError):
        calibration_evidence = None

    stress = None
    if args.config:
        from ashare_data.config import (
            load_config,
            make_backtest_config,
            make_data_config,
            make_model_config,
            make_protocol_config,
        )
        from ashare_data.gates import ProductionGateRunner
        from ashare_model.data_loader import AshareDataLoader

        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(
            data_config, min_eligible=args.min_eligible
        ).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        proto_cfg = make_protocol_config(raw)
        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        top_rows = _top_rows(artifact)
        if top_rows:
            last = top_rows[-1]
            tokens = last.get("formula")
            direction = last.get("direction", 1)
            if tokens:
                fold_cfgs = proto_cfg.folds
                stress = stress_champion(
                    loader,
                    fold_cfgs,
                    tokens,
                    direction,
                    backtest_config,
                )
        verdict = evaluate_challenger(
            artifact,
            regime=regime,
            dataset_id=loader.dataset_id,
            paper_registry=paper,
            stress=stress,
            allowed_data_tiers=("A", "B") if args.allow_tier_b else ("A",),
            backtest_config=backtest_config,
            calibration_evidence=calibration_evidence,
        )
    else:
        verdict = evaluate_challenger(
            artifact,
            regime=regime,
            paper_registry=paper,
            stress=stress,
            allowed_data_tiers=("A", "B") if args.allow_tier_b else ("A",),
            calibration_evidence=calibration_evidence,
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, gate in verdict["gates"].items():
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"{status}  {name}")
        for reason in gate["reasons"]:
            print(f"      {reason}")
    print(f"promoted={verdict['promoted']}")
    return 0 if verdict["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
