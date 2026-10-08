"""R03 combination-weight panel (p42, WP 6): fitted shrinkage weights vs
the equal-weight combo on the p37 n10 stable region.

The p42 branch adjudication (``docs/p42_r03_combo_weights_draft.md`` §1)
selected the R03 stage-2 question from the frozen decision table: the
cross-family equal-weight rank combination is the n10 stable region's
ONLY net-positive arm (p37), so the open question is whether a
train-window-fitted shrinkage weight combination adds a stable increment
over it — and whether the equal-weight net space replicates on a new run
identity.

Frozen design (p42 §3): members MOMENTUM_20 / RSQ_60 / REVERSAL_60 with
per-fold train-window directions; weights w_i = (1-lam)/K +
lam*max(val_icir_i, 0)/Sigma learned once per fold on the reference cell
``n10_m2500`` (the SAME ``baseline_candidates`` call that learns the
directions — one authority, train-window only, zero OOS access), frozen
across cells; cells = the p37 n10 stable region; costs 0x/1x; lam=0.50
primary + 0.25/0.75 descriptive-only; the four-way adjudication reuses
the p37 ``_arm_verdict`` / ``_has_gross_info`` single implementation
with the increment margin 0.005 requiring the equal-weight reference to
be alive in all cells (erratum 1).

Import direction: depends on :mod:`ashare_model.r01_baseline` (combo /
cost / fold / summary authorities) and :mod:`ashare_model.r02_matrix`
(the frozen cells, arm-verdict rules and the driver pattern); never
imports the ``evaluation`` facade.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION

from .r01_baseline import (
    COST_LEVELS,
    combo_signal,
    cost_variant,
    rank_normalize_signal,
    rows_to_jsonable,
    summarize_arms,
)
from .r02_matrix import (
    R02_ADJUDICATION_RULES,
    R02_REPORT_SCHEMA_VERSION,  # noqa: F401  (lineage re-export guard)
    _arm_verdict,
    _has_gross_info,
    cell_variant,
)

# --- frozen panel constants (p42 §3) ------------------------------------------

#: The p37 round-1 stable region — the only cells this panel runs.
R03_CELLS: tuple[str, ...] = ("n10_m1000", "n10_m2500", "n10_m5000")

#: Directions AND weights are learned once per fold on this cell and
#: reused frozen across cells (chosen before any R03 result was seen).
R03_REFERENCE_CELL = "n10_m2500"

R03_EQ_ARM = "combo:all3"
R03_PRIMARY_ARM = "combo:fitted_lam050"

#: The fitted arms: the primary adjudicated lambda plus the two
#: descriptive lambdas that never enter the adjudication.
R03_LAMBDAS: dict[str, float] = {
    "combo:fitted_lam050": 0.5,
    "combo:fitted_lam025": 0.25,
    "combo:fitted_lam075": 0.75,
}

#: Only these two arms are adjudicated (p42 §3).
R03_ADJUDICATED_ARMS: tuple[str, str] = (R03_EQ_ARM, R03_PRIMARY_ARM)

#: The preregistered increment margin: the fitted primary arm must beat
#: the equal-weight combo by >= 50bp of mean net excess at 1x in EVERY
#: cell (with the reference alive) for ``fitted_increment_region``.
R03_INCREMENT_MARGIN = 0.005

#: One implementation, never two: the arm-level rules are the p37
#: authority, consumed as the same object (pinned by
#: test_n3_arm_rules_are_the_p37_authority).
R03_ADJUDICATION_RULES = R02_ADJUDICATION_RULES

R03_REPORT_SCHEMA_VERSION = 1

R03_COST_LEVELS: tuple[str, ...] = ("0x", "1x")


class R03Refusal(RuntimeError):
    """A fail-closed refusal of the R03 run (p42 §3/§7)."""


# --- the weighted combination base (p42 §3, N1) --------------------------------


def weighted_combo_signal(
    members: list[tuple[float, np.ndarray]],
    universe_mask: np.ndarray,
    weights: list[float],
) -> np.ndarray:
    """Weighted mean of the rank-normalized signed member signals.

    The p34 §3.2 construction with weights: each member is direction-
    flipped then per-day cross-sectionally rank-normalized over the
    universe (``rank_normalize_signal``, NaN never participates); the
    combo is the weighted average of the members finite on that cell,
    renormalized by the available weight mass (a missing member's weight
    is redistributed — the p34 missing-member semantics generalized from
    counts to weights).  Equal weights reproduce :func:`combo_signal`
    exactly (one construction, not two).
    """

    if not members:
        raise ValueError("weighted_combo_signal needs at least one member")
    if len(weights) != len(members):
        raise ValueError(
            f"weighted_combo_signal: {len(weights)} weights for "
            f"{len(members)} members"
        )
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or (w < 0.0).any() or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError(
            "weights must be non-negative and sum to 1 (a weight simplex)"
        )
    normalized = [
        rank_normalize_signal(
            float(direction) * np.asarray(signal, dtype=np.float64),
            universe_mask,
        )
        for direction, signal in members
    ]
    stack = np.stack(normalized)  # [members, stocks, dates]
    wcol = w[:, None, None]
    finite = np.isfinite(stack)
    wmass = np.where(finite, wcol, 0.0).sum(axis=0)
    totals = np.where(finite, stack * wcol, 0.0).sum(axis=0)
    combo = np.full(wmass.shape, np.nan)
    has = wmass > 0.0
    combo[has] = totals[has] / wmass[has]
    return combo


# --- the frozen shrinkage fit rule (p42 §3, N2) ---------------------------------


def fit_combo_weights(scores: dict[str, float], lam: float) -> dict:
    """Shrinkage weights toward equal weight from member train scores.

    ``w_i = (1-lam)/K + lam * max(s_i, 0) / sum_j max(s_j, 0)``: the
    negative part is clipped, the fit is shrunk toward the equal weight,
    and an all-nonpositive score vector falls back to equal weights with
    the fallback event recorded.  The output carries the inputs and
    ``lam`` so every fitted weight is auditable back to its evidence.
    """

    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must lie in [0, 1], got {lam}")
    inputs = {name: float(value) for name, value in scores.items()}
    if not inputs:
        raise ValueError("fit_combo_weights needs at least one member score")
    k = len(inputs)
    positive = {name: max(value, 0.0) for name, value in inputs.items()}
    total = sum(positive.values())
    if total <= 0.0:
        return {
            "weights": {name: 1.0 / k for name in inputs},
            "fallback": True,
            "lam": lam,
            "inputs": inputs,
        }
    raw = {
        name: (1.0 - lam) / k + lam * value / total
        for name, value in positive.items()
    }
    norm = sum(raw.values())
    return {
        "weights": {name: value / norm for name, value in raw.items()},
        "fallback": False,
        "lam": lam,
        "inputs": inputs,
    }


# --- the four-way adjudication (p42 §3 + erratum 1, N3) -------------------------


def decide_r03_adjudication(
    cell_summaries: dict[str, dict[str, Any]],
    rules: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The p42 frozen four-way verdict (pure function; never sees
    prices, only summarized rows).

    Arm viability per cell comes from the p37 ``_arm_verdict`` rules
    (one implementation); the verdict order is 1 → 2 → 3 → 4 with the
    increment requiring the equal-weight reference to be alive in all
    cells (erratum 1: an increment over a dead baseline is meaningless).
    """

    merged = dict(R03_ADJUDICATION_RULES)
    merged.update(rules or {})
    cells_out: dict[str, dict[str, Any]] = {}
    for name in R03_CELLS:
        cell = cell_summaries.get(name) or {}
        top_n = int(cell.get("top_n") or 10)
        holdings_floor = max(1, top_n // 2)
        arms_out: dict[str, dict[str, Any]] = {}
        for arm in R03_ADJUDICATED_ARMS:
            summary = (cell.get("arms") or {}).get(arm)
            if summary is None:
                arms_out[arm] = {
                    "viable": False,
                    "flags": ["missing"],
                    "evidence": None,
                }
                continue
            verdict = _arm_verdict(summary, holdings_floor, merged)
            verdict["gross_info"] = _has_gross_info(summary)
            arms_out[arm] = verdict
        cells_out[name] = {
            "top_n": top_n,
            "arms": arms_out,
            "net_excess_1x": {
                arm: (cell.get("arms") or {}).get(arm, {}).get(
                    "mean_net_excess_1x"
                )
                for arm in R03_ADJUDICATED_ARMS
            },
        }

    def _alive(arm: str) -> list[str]:
        return [
            name
            for name in R03_CELLS
            if cells_out[name]["arms"][arm]["viable"]
        ]

    eq_alive = _alive(R03_EQ_ARM)
    fitted_alive = _alive(R03_PRIMARY_ARM)
    margin_met = all(
        (
            cells_out[name]["net_excess_1x"][R03_PRIMARY_ARM] is not None
            and cells_out[name]["net_excess_1x"][R03_EQ_ARM] is not None
            and cells_out[name]["net_excess_1x"][R03_PRIMARY_ARM]
            >= cells_out[name]["net_excess_1x"][R03_EQ_ARM]
            + R03_INCREMENT_MARGIN
        )
        for name in R03_CELLS
    )
    if len(eq_alive) == len(R03_CELLS) and len(fitted_alive) == len(
        R03_CELLS
    ) and margin_met:
        verdict = "fitted_increment_region"
        note = (
            "the fitted shrinkage combo is viable and beats the alive "
            "equal-weight combo by >= the frozen 50bp margin in every "
            "cell of the n10 stable region"
        )
    elif len(eq_alive) == len(R03_CELLS):
        verdict = "equal_weight_stands"
        note = (
            "the p37 equal-weight net space replicates on this run "
            "identity and the fitted combo does not clear the frozen "
            "increment margin everywhere"
        )
    elif len(eq_alive) <= len(R03_CELLS) - 2:
        verdict = "combo_no_net_space"
        note = (
            "the equal-weight combo is not viable in "
            f"{len(R03_CELLS) - len(eq_alive)} of {len(R03_CELLS)} cells: "
            "the p37 net space did not replicate"
        )
    else:
        verdict = "panel_no_reliable_info"
        note = (
            "mixed panel: the equal-weight combo is viable in "
            f"{len(eq_alive)} of {len(R03_CELLS)} cells — neither a "
            "replication, an increment, nor a clean negative"
        )
    return {
        "verdict": verdict,
        "note": note,
        "cells": cells_out,
        "rules": merged,
        "increment_margin": R03_INCREMENT_MARGIN,
        "adjudicated_arms": list(R03_ADJUDICATED_ARMS),
        "descriptive_arms": [
            arm for arm in R03_LAMBDAS if arm != R03_PRIMARY_ARM
        ],
    }


# --- the panel (p42 §3, N4) -----------------------------------------------------

R03_MEMBERS: tuple[str, ...] = ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")


def _cell_arm_summaries(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-cell arm summaries through the single ``summarize_arms``
    authority, restricted to the R03 cells and shaped for the
    adjudicator (``{cell: {top_n, arms: {arm: summary}}}``)."""

    arm_summaries = summarize_arms(rows)
    cells: dict[str, dict[str, Any]] = {}
    grid_top_n = {"n10": 10, "n20": 20}
    for key, summary in arm_summaries.items():
        key = str(key)
        if "::" not in key:
            continue
        cell, arm = key.split("::", 1)
        if cell not in R03_CELLS:
            continue
        entry = cells.setdefault(
            cell,
            {
                "top_n": grid_top_n.get(cell.split("_", 1)[0], 10),
                "arms": {},
            },
        )
        entry["arms"][arm] = summary
    return cells


def run_r03_panel(
    loader,
    proto_cfg,
    folds: list,
    backtest_config,
    model_config,
    reward_config,
    *,
    on_fold=None,
) -> tuple[list[dict], dict[str, dict[str, Any]]]:
    """All cells x combo arms x folds x cost levels through the shared
    engine.

    Returns ``(rows, fold_evidence)``.  Row candidates carry the
    ``"<cell>::<arm>"`` composite (the single ``summarize_arms``
    authority computes the per-(cell, arm) summaries); the benchmark is
    the cost-free per-fold reference; the three singles are measured
    once per fold on the reference-cell direction path (1x real panel
    rows — no duplicate engine run); directions and fitted weights are
    learned once per fold on the reference cell and frozen across
    cells.  ``on_fold`` is the budget callback; its exceptions propagate
    (the budget guard must fail the run, not one fold).
    """

    from .eval_folds import epoch_slice
    from .eval_metrics import benchmark_row, evaluate_signal
    from .eval_search import baseline_candidates
    from .vocab import FEATURE_NAMES

    rows: list[dict] = []
    fold_evidence: dict[str, dict[str, Any]] = {}
    total = len(folds)
    for index, fold in enumerate(folds):
        if on_fold is not None:
            on_fold(index, total, fold)

        def _ok(candidate: str, level: str, payload: dict, cell=None) -> dict:
            payload.update(
                {
                    "candidate": candidate,
                    "formula_text": candidate.split("::")[-1],
                    "fold_train_end": fold.train_end,
                    "fold_test_end": fold.test_end,
                    "seed": None,
                    "cost_level": level,
                    "failed": False,
                }
            )
            if cell is not None:
                payload["cell"] = cell
            return payload

        def _failed_row(
            candidate: str, level: str, exc: Exception, cell=None
        ) -> dict:
            row = {
                "candidate": candidate,
                "formula_text": candidate.split("::")[-1],
                "fold_train_end": fold.train_end,
                "fold_test_end": fold.test_end,
                "seed": None,
                "cost_level": level,
                "failed": True,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
            if cell is not None:
                row["cell"] = cell
            return row

        # The cost-free equal-weight reference (cell-independent).
        try:
            row = benchmark_row(loader, fold)
            rows.append(_ok("benchmark:equal_weight", "ref", row))
        except Exception as exc:  # noqa: BLE001 — recorded per p42 §3
            rows.append(_failed_row("benchmark:equal_weight", "ref", exc))

        # Directions AND fit scores: learned once per fold through the
        # one baseline_candidates authority on the reference cell (the
        # reference 1x single-factor rows are real panel rows).
        reference_config = cell_variant(
            backtest_config, R03_REFERENCE_CELL
        )
        singles: list[dict] = []
        singles_failed: Exception | None = None
        try:
            singles = baseline_candidates(
                loader, proto_cfg, fold, reference_config, model_config,
                reward_config,
            )
        except Exception as exc:  # noqa: BLE001 — recorded per p42 §3
            singles_failed = exc
        directions: dict[str, int] = {}
        scores: dict[str, float] = {}
        direction_evidence: dict[str, dict[str, Any]] = {}
        for r in singles:
            arm = str(r["candidate"]).split(":", 1)[1]
            directions[arm] = int(r["direction"])
            scores[arm] = (
                float(r["val_icir"])
                if r.get("val_icir") is not None
                else 0.0
            )
            direction_evidence[arm] = {
                "direction": int(r["direction"]),
                "val_reward": r.get("val_reward"),
                "val_icir": r.get("val_icir"),
            }
            rows.append(
                _ok(
                    f"{R03_REFERENCE_CELL}::{r['candidate']}",
                    "1x",
                    r,
                    R03_REFERENCE_CELL,
                )
            )

        # The frozen weight fits (once per fold, reused verbatim across
        # cells — the "frozen across cells" pin of N4).
        fits: dict[str, dict[str, Any]] = {
            arm: fit_combo_weights(scores, lam)
            if singles_failed is None and scores
            else {
                "weights": {name: 1.0 / len(R03_MEMBERS)
                            for name in R03_MEMBERS},
                "fallback": True,
                "lam": lam,
                "inputs": dict(scores),
            }
            for arm, lam in R03_LAMBDAS.items()
        }
        equal_fit = {
            "weights": {name: 1.0 / len(R03_MEMBERS) for name in R03_MEMBERS},
            "fallback": False,
            "lam": 0.0,
            "inputs": dict(scores),
            "note": "equal weight (the p37 reference construction)",
        }

        fold_data = epoch_slice(loader, fold)
        mask = fold_data.universe_mask
        member_signals: dict[str, np.ndarray] = {
            name: np.asarray(
                fold_data.factors[FEATURE_NAMES.index(name)],
                dtype=np.float64,
            )
            for name in R03_MEMBERS
        }

        def _direction_or_failed(name: str) -> int | None:
            """The r02 fail-closed seam, same semantics as
            ``r02_matrix.run_r02_panel``'s helper of the same name: a
            member without train-window direction evidence fails the
            row — a silent ``+1`` default would fabricate the p42
            frozen "per-fold train-window directions" and run the whole
            panel on an invented side (the old ``directions.get(name,
            1)`` did exactly that).  ``None`` = the singles call itself
            failed (the caller re-raises that exception)."""

            if name in directions:
                return directions[name]
            if singles_failed is not None:
                return None
            raise KeyError(
                f"factor {name} missing from the direction evidence"
            )

        for level in R03_COST_LEVELS:
            for cell in R03_CELLS:
                variant = cost_variant(
                    cell_variant(backtest_config, cell), level
                )
                for arm, fit in {
                    R03_EQ_ARM: equal_fit, **fits,
                }.items():
                    try:
                        if singles_failed is not None:
                            raise singles_failed  # type: ignore[misc]
                        members = []
                        for name in R03_MEMBERS:
                            direction = _direction_or_failed(name)
                            if direction is None:
                                raise singles_failed  # type: ignore[misc]
                            members.append(
                                (float(direction), member_signals[name])
                            )
                        weights = [
                            float(fit["weights"][name]) for name in R03_MEMBERS
                        ]
                        signal = weighted_combo_signal(members, mask, weights)
                        metrics = evaluate_signal(
                            signal, loader, fold, variant
                        )
                        rows.append(_ok(f"{cell}::{arm}", level, metrics, cell))
                    except Exception as exc:  # noqa: BLE001 — p42 §3
                        rows.append(
                            _failed_row(f"{cell}::{arm}", level, exc, cell)
                        )

        fold_evidence[str(fold.test_end)] = {
            "directions": direction_evidence,
            "singles_failed": (
                f"{type(singles_failed).__name__}: {singles_failed}"
                if singles_failed is not None
                else None
            ),
            "weights_by_cell": {
                cell: {
                    R03_EQ_ARM: equal_fit,
                    **fits,
                }
                for cell in R03_CELLS
            },
        }
    return rows, fold_evidence


# --- the run driver (p42 §3/§6; mirrors run_r02_matrix) --------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_r03_matrix(
    config_path: str | Path,
    contract_sha256: str,
    out_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    require_clean_tree: bool = True,
    wall_clock_budget_s: float = 4 * 3600,
    min_eligible: int | None = None,
    adjudication_rules: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The p42 R03 run.  Returns the run summary; a fail-closed refusal
    raises :class:`R03Refusal` with the run left at its honest terminal
    state.  ``adjudication_rules`` is the test-only seam running the
    REAL adjudicator with relaxed canary thresholds (the p34 rev2 T8 /
    p37 pattern); production callers keep the frozen defaults."""

    from ashare_data.gates import (
        ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
        ProductionGateRunner,
    )

    from .data_loader import AshareDataLoader
    from .identity import content_hash
    from .lifecycle import (
        TERMINAL_STATUSES,
        DataQualificationError,
        LifecycleWriter,
        build_data_qualification_report,
        replay_run,
    )
    from .r01_baseline import (
        _expected_and_missing_sessions,
        load_config_file,
        resolve_folds_for,
    )
    from .run_store import RunStore
    from .runspec import resolve_runtime_runspec

    started = time.monotonic()

    def _budget_check(where: str) -> None:
        elapsed = time.monotonic() - started
        if elapsed > wall_clock_budget_s:
            raise R03Refusal(
                f"R03 wall-clock budget exceeded at {where}: "
                f"{elapsed:.0f}s > {wall_clock_budget_s:.0f}s (p42 §3)"
            )

    root = Path(repo_root) if repo_root is not None else (
        Path(__file__).resolve().parents[1]
    )
    config_path = Path(config_path)
    out_root = Path(out_root) if out_root is not None else (
        root / "data" / "r1" / "r03_combos"
    )

    # -- the supplied contract sha256 must match the committed contract
    # file (self-verified provenance; refuses before anything expensive).
    contract_path = root / "docs" / "p42_r03_combo_weights_draft.md"
    actual_contract_sha = _sha256_file(contract_path)
    if str(contract_sha256).lower() != actual_contract_sha:
        raise R03Refusal(
            "contract sha256 mismatch: supplied "
            f"{contract_sha256!r} != committed contract file "
            f"{actual_contract_sha} ({contract_path.name}); a formal run "
            "never stamps unverified provenance"
        )

    from ashare_data.config import (
        make_backtest_config,
        make_data_config,
        make_model_config,
        make_protocol_config,
        make_reward_config,
    )

    raw = load_config_file(config_path)
    data_config = make_data_config(raw, root)
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    reward_config = make_reward_config(raw)
    proto_cfg = make_protocol_config(raw)

    # -- A02/A03: identity confirm → frozen spec → open run.
    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    if not loader.dataset_id:
        raise R03Refusal(
            "the composite snapshot identity could not be confirmed; a "
            "formal R03 run refuses to proceed without a dataset identity"
        )
    spec = resolve_runtime_runspec(
        dataset_id=loader.dataset_id,
        data_cutoff=loader.frozen_data_cutoff(),
        purpose="historical_research",
        data_config=data_config,
        backtest_config=backtest_config,
        requested_budget=max(
            1, proto_cfg.screening.steps * proto_cfg.screening.batch_size
        ),
        n_folds=len(proto_cfg.folds),
        research_domain=proto_cfg.domain,
        seeds=tuple(proto_cfg.seeds),
        searcher=model_config.searcher,
        max_formula_len=int(model_config.max_formula_len),
        repo_root=root,
        require_clean_tree=require_clean_tree,
    )

    # -- formal gates on the frozen snapshot's own timeline (P36).
    gate_result = ProductionGateRunner(
        data_config, min_eligible=min_eligible
    ).run(
        mode="formal",
        purpose=ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
        snapshot_cutoff=spec.data_cutoff,
    )
    if not gate_result.ok:
        failed = [
            f"{check.name} ({check.detail})"
            for check in gate_result.checks
            if not check.ok
        ]
        raise R03Refusal(
            "production gate check failed (p42 §7): " + "; ".join(failed)
        )
    store = RunStore(data_config.data_dir)
    handle = store.open_run(spec)
    run_id = handle.run_id
    writer = LifecycleWriter(handle)

    def _mark_failed(reason: str) -> None:
        if writer.current_state not in TERMINAL_STATUSES:
            writer.fail(reason=reason)

    def _latest_artifact(artifact_type: str):
        index = handle.load_index()
        for entry in reversed(index.get("artifacts", [])):
            if entry.get("artifact_type") == artifact_type:
                return entry
        return None

    try:
        idea = {
            "idea_record_schema_version": 1,
            "hypothesis": (
                "R03 (p42, WP 6): on the p37 n10 stable region "
                "(every_10_days/horizon 10, 100k CNY), does a "
                "train-window-fitted shrinkage weight combination of the "
                "three frozen members beat the equal-weight rank combo by "
                "a stable >=50bp net-excess increment across the region, "
                "does the equal-weight net space replicate, or is there "
                "no reliable combination information?"
            ),
            "scope": (
                f"configs/{config_path.name} (sha256 "
                f"{_sha256_file(config_path).upper()}), folds "
                f"{[str(f['test_end']) for f in raw['protocol']['folds']]}, "
                f"cells {list(R03_CELLS)} with the p37 coupled params "
                "verbatim, members MOMENTUM_20/RSQ_60/REVERSAL_60 with "
                "per-fold train-window directions, weights "
                "(1-lam)/K + lam*max(val_icir,0)/Sigma fitted once per "
                f"fold on {R03_REFERENCE_CELL} and frozen across cells "
                f"(lam 0.5 primary, 0.25/0.75 descriptive), 0x/1x costs; "
                f"preregistration contract sha256 {contract_sha256}."
            ),
            "economic_mechanism": (
                "The p37 finding: diversifying cross-family rank signals "
                "concentrated in the top ten creates net space no single "
                "member has; fitted weights test whether the members' "
                "train-window validation ICIR carries exploitable "
                "relative weight information beyond equal weighting."
            ),
            "non_goals": (
                "No search; no new factors or vocabulary; no reward "
                "change (R04 not triggered by evidence); no period axis "
                "(p39 closed the reference cell; n10 x period not "
                "tested here); no capital/2x-cost scaling; no holdout or "
                "promotion claims; no default-config changes; no Web or "
                "cache work."
            ),
        }
        writer.record_idea(idea)
        writer.lock_spec()

        loader.load_data()
        if loader.dataset_id != spec.dataset_id:
            _mark_failed(
                "the database content drifted between identity freeze "
                "and load (A02 I2/p27 I6); the frozen spec no longer "
                "matches the confirmed composite snapshot"
            )
            raise R03Refusal(
                "dataset identity drifted between spec freeze and load"
            )
        _budget_check("load")
        frame_status = loader.loader_frame_status()
        expected_days, actual_days, missing_days = (
            _expected_and_missing_sessions(
                loader, data_config, cutoff=spec.data_cutoff
            )
        )
        data_report = build_data_qualification_report(
            spec,
            gate_result,
            expected_trading_days=expected_days,
            actual_trading_days=actual_days,
            missing_trading_days=missing_days,
            coverage=float(np.mean(loader.universe_mask)),
            loader_frames=frame_status,
        )
        try:
            writer.qualify_data(data_report)
        except DataQualificationError as exc:
            entry = _latest_artifact("data_qualification_report")
            completed = (
                bool(
                    handle.read_artifact(entry["artifact_id"]).get("completed")
                )
                if entry is not None
                else False
            )
            if entry is not None and completed:
                writer.reject(
                    evidence={
                        "artifact_id": entry["artifact_id"],
                        "artifact_type": "data_qualification_report",
                    }
                )
            else:
                _mark_failed(
                    f"data qualification gate did not complete: {exc}"
                )
            raise R03Refusal(f"data qualification refused: {exc}") from exc

        folds = resolve_folds_for(proto_cfg, loader)
        rows, fold_evidence = run_r03_panel(
            loader,
            proto_cfg,
            folds,
            backtest_config,
            model_config,
            reward_config,
            on_fold=lambda i, total, fold: _budget_check(
                f"fold {fold.test_end} ({i + 1}/{total})"
            ),
        )
        _budget_check("panel")

        cell_summaries = _cell_arm_summaries(rows)
        adjudication = decide_r03_adjudication(
            cell_summaries, adjudication_rules
        )

        benchmark_status = {
            str(row["fold_test_end"]): {
                "status": "failed" if row.get("failed") else "ok",
                "failure_reason": row.get("failure_reason"),
            }
            for row in rows
            if str(row["candidate"]) == "benchmark:equal_weight"
        }
        completeness_reasons = list(frame_status["reasons"])
        if missing_days:
            completeness_reasons.append(
                f"missing trading days: {missing_days}"
            )
        data_completeness = {
            "status": (
                "complete"
                if frame_status["status"] == "clean" and not missing_days
                else "incomplete"
            ),
            "loader_frames": frame_status,
            "expected_trading_days": int(expected_days),
            "actual_trading_days": int(actual_days),
            "missing_trading_days": [str(day) for day in missing_days],
            "reasons": completeness_reasons,
        }
        engine_version_attribution = {
            "execution_spec_version": int(EXECUTION_SPEC_VERSION),
            "portfolio_constructor_version": int(
                PORTFOLIO_CONSTRUCTOR_VERSION
            ),
            "protocol_version_note": (
                "PROTOCOL_VERSION 29 (p40/p41 statistical semantics); the "
                "R03 panel consumes the same evaluate_signal path as "
                "p37/p39 — combination weights are a research arm, not a "
                "protocol change"
            ),
        }
        reporting_notes = {
            "branch_adjudication": (
                "R03 selected by the frozen decision table (p42 §1): the "
                "equal-weight cross-family combo is the p37 n10 region's "
                "only net-positive arm while all three singles are "
                "net-negative; R04 not triggered (no reward-ranking "
                "evidence exists in the closed panels)"
            ),
            "reference_alive_requirement": (
                "the increment verdict requires the equal-weight "
                "reference to be viable in every cell (erratum 1): an "
                "increment over a dead baseline is meaningless"
            ),
            "weights_frozen_across_cells": (
                "directions and weights are learned once per fold on "
                f"{R03_REFERENCE_CELL} (train-window validation rank-IC "
                "sign and val_icir through the same baseline_candidates "
                "call) and reused frozen across all cells — weight "
                "learning never varies with the cell"
            ),
            "descriptive_lambdas": (
                "lam 0.25/0.75 arms are descriptive sensitivity "
                "disclosures only; they never enter the adjudication"
            ),
            "no_parameter_pick": (
                "the verdict identifies the increment or its absence "
                "only; it never justifies a default-config or promotion "
                "choice, and the 50bp margin was frozen before any R03 "
                "result existed"
            ),
        }
        report = {
            "r03_report_schema_version": R03_REPORT_SCHEMA_VERSION,
            "provenance": {
                "contract_sha256": str(contract_sha256),
                "contract_path": "docs/p42_r03_combo_weights_draft.md",
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path).upper(),
                "dataset_id": spec.dataset_id,
                "data_cutoff": spec.data_cutoff,
                "git_commit": spec.git_commit,
                "spec_id": spec.spec_id,
                "run_id": run_id,
                "research_domain": spec.research_domain,
                "frequency": proto_cfg.frequency,
                "horizon": proto_cfg.horizon,
                "initial_capital": float(backtest_config.initial_capital),
                "requested_budget": spec.requested_budget,
                "consumed_semantic_evaluations": 0,
                "cells": list(R03_CELLS),
                "reference_cell": R03_REFERENCE_CELL,
                "lambdas": dict(R03_LAMBDAS),
                "members": list(R03_MEMBERS),
            },
            "adjudication": adjudication,
            "adjudicated_arms": list(R03_ADJUDICATED_ARMS),
            "data_completeness": data_completeness,
            "benchmark_status": benchmark_status,
            "cell_summaries": cell_summaries,
            "arm_summaries": summarize_arms(rows),
            "engine_version_attribution": engine_version_attribution,
            "reporting_notes": reporting_notes,
            "fold_evidence": rows_to_jsonable(fold_evidence),
            "rows": rows_to_jsonable(rows),
            "completed": True,
        }
        report_record = handle.write_artifact("r03_combo_report", report)

        # -- verdict 1/2 park non-terminal at DATA_QUALIFIED; 3/4 retire
        # (the p37 administrative walk).
        verdict = adjudication["verdict"]
        if verdict in ("fitted_increment_region", "equal_weight_stands"):
            final_state = writer.current_state  # parked, non-terminal
        else:
            writer.retire(
                reason=(
                    f"R03 verdict {verdict}: {adjudication['note']}; "
                    f"report artifact {report_record.artifact_id}"
                )
            )
            final_state = writer.current_state
        summary = {
            "spec_id": spec.spec_id,
            "run_id": run_id,
            "state": replay_run(store.root, spec.spec_id, run_id)
            or final_state,
            "dataset_id": spec.dataset_id,
            "data_cutoff": spec.data_cutoff,
            "verdict": verdict,
            "adjudication": adjudication,
            "report_artifact_id": report_record.artifact_id,
            "row_count": len(rows),
            "report_content_hash": content_hash("r03_combo", report),
            "wall_seconds": round(time.monotonic() - started, 1),
        }
    except R03Refusal as exc:
        _mark_failed(str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 — the honest-terminal catch-all
        _mark_failed(f"R03 runtime failure: {type(exc).__name__}: {exc}")
        raise R03Refusal(f"runtime failure: {exc}") from exc
    finally:
        handle.close()

    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "r03_combo_report.json"
    report_path.write_text(
        json.dumps(rows_to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_path = out_root / "r03_combo_run_meta.json"
    meta_path.write_text(
        json.dumps(rows_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
