"""P34 R01 round-1 fixed-baseline diagnostic driver.

Executable projection of ``docs/p34_r01_fixed_baseline_diagnostic_draft.md``
(APPROVED via user direct instruction 2026-09-16 「完成R01」, p30–p33
precedent).  The plan's R01 work package: **baselines complete and
recorded before any search**, on a frozen medium Tier-A configuration.

What this module does (and deliberately does not do):

* walks one run through the real lifecycle chain — genesis (typed
  IdeaRecord) → SPEC_LOCKED (frozen RunSpec over the A02-confirmed
  composite identity) → DATA_QUALIFIED (real G1–G8 gate result + the
  typed loader-frame status, p8 §5.1) → FACTOR_SET_QUALIFIED (real
  §5.2 report with the baseline/ablation evidence produced here);
* produces the frozen baseline arms **through the single semantic
  paths** — ``benchmark_row`` / ``baseline_candidates`` / ``evaluate_signal``
  (the same engine the protocol scores every candidate with), plus the
  frozen rule combos and the cash control built from the same fold
  slices — at the 0x/1x/2x cost ladder;
* registers the report inside the run store (content-addressed
  artifact), writes the measurement-directory copy, and applies the
  frozen adjudication: a positive verdict parks the run at
  ``FACTOR_SET_QUALIFIED`` (round 2 needs its own explicit
  authorization — this module never admits a search plan), a negative
  verdict is a legal ``RETIRED`` result;
* it does NOT run any searcher, does NOT consume the unique-semantic
  budget, does NOT touch production defaults, and does NOT claim
  holdout/promotion semantics for the 2021–2025 development windows.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ashare_data.gates import ProductionGateRunner

#: Report artifact schema (measurement artifact; not an
#: ``artifact_schemas`` consumer — see p34 §4 version impact table).
R01_REPORT_SCHEMA_VERSION = 1

#: p34 §3.7 frozen adjudication thresholds (never tuned after seeing data).
R01_ADJUDICATION_RULES: dict[str, float] = {
    "min_positive_folds": 3,
    "min_avg_holdings": 5.0,
    "min_invested_ratio": 0.5,
}

#: p34 §3.2 frozen combo construction (members ⊆ the config's preregistered
#: baseline signals; directions come from the same train-window rule the
#: single-factor rows use — frozen at preregistration time, not by OOS).
R01_COMBO_ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("combo:all3", ("MOMENTUM_20", "RSQ_60", "REVERSAL_60")),
    ("combo:momentum_reversal", ("MOMENTUM_20", "REVERSAL_60")),
)

COST_LEVELS = ("0x", "1x", "2x")

_ARM_PROBE_KEYS = (
    "excess_return",
    "active_ir",
    "holdings_mean",
    "invested_ratio_mean",
    "cost_fraction_sum",
    "rebalance_due_count",
    "rebalance_count",
)


class R01Refusal(RuntimeError):
    """A fail-closed refusal of the R01 round-1 run (p34 §7)."""


# --- frozen cost ladder -------------------------------------------------------


def cost_variant(backtest_config, level: str):
    """The p34 §3.3 cost ladder: ``0x`` gross (every cost parameter zero),
    ``1x`` the configuration verbatim, ``2x`` doubled cost rates with the
    commission floor unchanged.  Unknown levels are a construction error;
    the caller's config is never mutated."""

    if level == "1x":
        return backtest_config
    if level == "0x":
        return dataclasses.replace(
            backtest_config,
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            transfer_fee_rate=0.0,
            slippage_rate=0.0,
        )
    if level == "2x":
        return dataclasses.replace(
            backtest_config,
            commission_rate=2.0 * backtest_config.commission_rate,
            stamp_tax_rate=2.0 * backtest_config.stamp_tax_rate,
            transfer_fee_rate=2.0 * backtest_config.transfer_fee_rate,
            slippage_rate=2.0 * backtest_config.slippage_rate,
        )
    raise ValueError(f"unknown cost level {level!r}: expected 0x/1x/2x")


# --- frozen combo construction ------------------------------------------------


def rank_normalize_signal(
    signal: np.ndarray, universe_mask: np.ndarray
) -> np.ndarray:
    """Per-date cross-sectional rank normalization over the eligible
    finite cells (p34 §3.2).  Ranks come from ``reward._rankdata`` — the
    single rank authority (IP-14); cells outside the universe mask or
    not finite stay NaN, and a fully empty cross-section stays NaN."""

    from .reward import _rankdata

    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(universe_mask, dtype=bool)
    if signal.shape != mask.shape:
        raise ValueError(
            f"signal shape {signal.shape} does not match universe mask "
            f"{mask.shape}"
        )
    out = np.full(signal.shape, np.nan)
    for j in range(signal.shape[1]):
        col = signal[:, j]
        sel = mask[:, j] & np.isfinite(col)
        if sel.any():
            out[sel, j] = _rankdata(col[sel])
    return out


def combo_signal(
    members: list[tuple[float, np.ndarray]], universe_mask: np.ndarray
) -> np.ndarray:
    """Equal-weight mean of the rank-normalized signed member signals
    (p34 §3.2).  A cell averages the members that have a finite rank
    there; a cell with no available member stays NaN."""

    if not members:
        raise ValueError("combo_signal needs at least one member")
    normalized = [
        rank_normalize_signal(
            float(direction) * np.asarray(signal, dtype=np.float64),
            universe_mask,
        )
        for direction, signal in members
    ]
    stack = np.stack(normalized)  # [members, stocks, dates]
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    totals = np.where(finite, stack, 0.0).sum(axis=0)
    combo = np.full(counts.shape, np.nan)
    has = counts > 0
    combo[has] = totals[has] / counts[has]
    return combo


def cash_signal(n_stocks: int, n_dates: int) -> np.ndarray:
    """The p34 §3.2 cash control: an all-NaN signal the engine cannot
    select from — the book that holds nothing, trades nothing and pays
    nothing, exposing any 'low-cost advantage' that is really just an
    empty book."""

    if n_stocks <= 0 or n_dates <= 0:
        raise ValueError("cash_signal needs a positive shape")
    return np.full((n_stocks, n_dates), np.nan)


# --- row hygiene ----------------------------------------------------------------


def rows_to_jsonable(obj: Any) -> Any:
    """Recursively map non-finite floats to ``None`` so every report row
    serializes as strict JSON (artifact writes reject NaN/Infinity)."""

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {key: rows_to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rows_to_jsonable(value) for value in obj]
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return rows_to_jsonable(obj.tolist())
    return obj


# --- baseline arms --------------------------------------------------------------


def baseline_directions_and_rows(
    loader,
    proto_cfg,
    fold,
    backtest_config,
    model_config,
    reward_config,
) -> list[dict]:
    """The 1x single-factor rows of one fold via the existing
    ``baseline_candidates`` path; their ``direction`` values are the
    frozen train-window rule every cost level reuses."""

    from .eval_search import baseline_candidates

    rows = baseline_candidates(
        loader, proto_cfg, fold, backtest_config, model_config, reward_config
    )
    for row in rows:
        row["cost_level"] = "1x"
    return rows


def run_baseline_arms(
    loader,
    proto_cfg,
    folds: list,
    backtest_config,
    model_config,
    reward_config,
    *,
    on_fold=None,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    """All frozen arms × folds × cost levels through the shared engine.

    Returns ``(rows, directions_by_fold)`` where directions map
    ``fold_test_end -> {factor_name: direction}`` (the train-window rule
    outputs, decided once at 1x and reused verbatim).  ``on_fold`` is a
    progress callback receiving ``(index, total, fold)``.
    """

    from .eval_folds import epoch_slice
    from .eval_metrics import benchmark_row, evaluate_signal
    from .vocab import FEATURE_NAMES

    rows: list[dict] = []
    directions_by_fold: dict[str, dict[str, int]] = {}
    total = len(folds)
    for index, fold in enumerate(folds):
        if on_fold is not None:
            on_fold(index, total, fold)
        # The cost-free equal-weight reference (once — it is the common
        # benchmark every cost level is measured against).
        row = benchmark_row(loader, fold)
        row["cost_level"] = "1x"
        rows.append(row)
        singles = baseline_directions_and_rows(
            loader, proto_cfg, fold, backtest_config, model_config,
            reward_config,
        )
        directions = {
            str(r["candidate"]).split(":", 1)[1]: int(r["direction"])
            for r in singles
        }
        directions_by_fold[str(fold.test_end)] = directions
        rows.extend(singles)
        fold_data = epoch_slice(loader, fold)
        factors = fold_data.factors
        mask = fold_data.universe_mask

        def _tag(payload: dict, candidate: str, level: str) -> dict:
            payload["candidate"] = candidate
            payload["formula_text"] = candidate
            payload["fold_train_end"] = fold.train_end
            payload["fold_test_end"] = fold.test_end
            payload["seed"] = None
            payload["cost_level"] = level
            payload["failed"] = False
            return payload

        for level in COST_LEVELS:
            if level == "1x":
                continue  # singles already measured on the 1x path above
            variant = cost_variant(backtest_config, level)
            for name, direction in directions.items():
                idx = FEATURE_NAMES.index(name)
                signal = float(direction) * np.asarray(
                    factors[idx], dtype=np.float64
                )
                metrics = evaluate_signal(signal, loader, fold, variant)
                rows.append(_tag(metrics, f"baseline:{name}", level))
        # Combos: the frozen construction, evaluated at every cost level.
        for level in COST_LEVELS:
            variant = cost_variant(backtest_config, level)
            for combo_name, member_names in R01_COMBO_ARMS:
                members = [
                    (directions[name], factors[FEATURE_NAMES.index(name)])
                    for name in member_names
                ]
                signal = combo_signal(members, mask)
                metrics = evaluate_signal(signal, loader, fold, variant)
                rows.append(_tag(metrics, combo_name, level))
        # The cash control at every level (definitionally identical; run
        # mechanically instead of special-casing it).
        cash = cash_signal(len(loader.ts_codes), len(fold_data.dates))
        for level in COST_LEVELS:
            variant = cost_variant(backtest_config, level)
            metrics = evaluate_signal(cash, loader, fold, variant)
            rows.append(_tag(metrics, "control:cash", level))
    return rows, directions_by_fold


def summarize_arms(rows: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-arm round-1 summary (p34 §3.4/§3.7): every arm, every fold —
    medians of the field distributions are the report's job; the
    adjudication consumes these means/counters verbatim."""

    from .baseline_harness import oos_active_ir

    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(str(row["candidate"]), []).append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for candidate, arm_rows in arms.items():
        by_fold: dict[str, dict[str, dict]] = {}
        for row in arm_rows:
            by_fold.setdefault(str(row["fold_test_end"]), {})[
                str(row["cost_level"])
            ] = row
        failed_folds = sum(
            1
            for per_level in by_fold.values()
            if per_level.get("1x", {}).get("failed")
        )
        excess_1x = [
            per_level["1x"]["excess_return"]
            for per_level in by_fold.values()
            if "1x" in per_level
            and per_level["1x"].get("excess_return") is not None
        ]
        active_irs = []
        for per_level in by_fold.values():
            row_1x = per_level.get("1x")
            if row_1x is None:
                continue
            active_irs.append(
                oos_active_ir(
                    np.asarray(row_1x["daily_returns"], dtype=np.float64),
                    np.asarray(
                        row_1x["benchmark_daily_returns"], dtype=np.float64
                    ),
                )
            )
        summaries[candidate] = {
            "n_folds": len(by_fold),
            "failed_folds": failed_folds,
            "unreliable": failed_folds >= 2,
            "mean_net_excess_1x": (
                float(np.mean(excess_1x)) if excess_1x else None
            ),
            "active_ir_1x": (
                float(np.mean(active_irs)) if active_irs else None
            ),
            "positive_folds_1x": int(
                sum(1 for value in excess_1x if value > 0)
            ),
            "mean_holdings_1x": (
                float(
                    np.mean(
                        [
                            per_level["1x"]["holdings_mean"]
                            for per_level in by_fold.values()
                            if "1x" in per_level
                            and per_level["1x"].get("holdings_mean")
                            is not None
                        ]
                    )
                )
                if any(
                    "1x" in per_level
                    and per_level["1x"].get("holdings_mean") is not None
                    for per_level in by_fold.values()
                )
                else None
            ),
            "invested_ratio_mean_1x": (
                float(
                    np.mean(
                        [
                            per_level["1x"]["invested_ratio_mean"]
                            for per_level in by_fold.values()
                            if "1x" in per_level
                            and per_level["1x"].get("invested_ratio_mean")
                            is not None
                        ]
                    )
                )
                if any(
                    "1x" in per_level
                    and per_level["1x"].get("invested_ratio_mean") is not None
                    for per_level in by_fold.values()
                )
                else None
            ),
            "mean_net_excess_2x": (
                float(
                    np.mean(
                        [
                            per_level["2x"]["excess_return"]
                            for per_level in by_fold.values()
                            if "2x" in per_level
                            and per_level["2x"].get("excess_return")
                            is not None
                        ]
                    )
                )
                if any(
                    "2x" in per_level
                    and per_level["2x"].get("excess_return") is not None
                    for per_level in by_fold.values()
                )
                else None
            ),
        }
    return summaries


# --- frozen adjudication --------------------------------------------------------


def decide_adjudication(
    arm_summaries: dict[str, dict[str, Any]],
    rules: dict[str, float] = R01_ADJUDICATION_RULES,
) -> dict[str, Any]:
    """The p34 §3.7 frozen rules (pure function; never sees prices, only
    the summarized rows).  Cost-bearing arms only — the benchmark and
    the cash control are references/controls, not value evidence."""

    flags: dict[str, list[str]] = {}
    qualified: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    for candidate, summary in arm_summaries.items():
        if candidate in ("benchmark:equal_weight", "control:cash"):
            continue
        arm_flags: list[str] = []
        shows_net_space = (
            summary["mean_net_excess_1x"] is not None
            and summary["active_ir_1x"] is not None
            and summary["mean_net_excess_1x"] > 0
            and summary["active_ir_1x"] > 0
        )
        if summary["unreliable"] or summary["failed_folds"]:
            arm_flags.append("unreliable")
        invested = summary["invested_ratio_mean_1x"]
        if invested is None or invested < rules["min_invested_ratio"]:
            arm_flags.append("low_activity")
        holdings = summary["mean_holdings_1x"]
        if holdings is None or holdings < rules["min_avg_holdings"]:
            arm_flags.append("thin_book")
        if shows_net_space:
            if summary["positive_folds_1x"] < rules["min_positive_folds"]:
                arm_flags.append("single_year_dependence")
            if (summary["mean_net_excess_2x"] or 0.0) <= 0:
                arm_flags.append("cost_fragile")
            if not any(
                flag in arm_flags
                for flag in (
                    "low_activity",
                    "thin_book",
                    "unreliable",
                    "single_year_dependence",
                )
            ):
                qualified.append(candidate)
        flags[candidate] = arm_flags
        details[candidate] = {
            "shows_net_space": bool(shows_net_space),
            "flags": list(arm_flags),
        }
    worth_search = bool(qualified)
    bottleneck = (
        "qualified_arms_present"
        if worth_search
        else (
            "no_qualified_arm: no cost-bearing arm shows durable "
            "net-of-cost space under the frozen rules (low_activity / "
            "thin_book / unreliable / single_year_dependence / "
            "non-positive net excess)"
        )
    )
    return {
        "worth_search": worth_search,
        "qualified_arms": qualified,
        "flags": flags,
        "arm_details": details,
        "rules": dict(rules),
        "bottleneck": bottleneck,
    }


# --- lifecycle-driving run ------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_and_missing_sessions(loader, data_config) -> tuple[int, int, list[str]]:
    """Expected open sessions inside the configured window vs the loaded
    axis (the §5.1 calendar reconciliation, computed from the single
    calendar authority the loader already resolved)."""

    start = str(data_config.start_date).replace("-", "")
    end = str(data_config.end_date).replace("-", "")
    sessions = [
        str(session)
        for session in loader._contract().sessions
        if start <= str(session) <= end
    ]
    loaded = {str(date).replace("-", "") for date in loader.dates}
    missing = [session for session in sessions if session not in loaded]
    return len(sessions), len(loaded), missing


def _feature_coverage(loader, feature_names: list[str]) -> dict[str, float]:
    """Per-feature finite coverage over the universe-masked cells of the
    loaded factor tensor (the §5.2 check-5 input)."""

    from .vocab import FEATURE_NAMES

    mask = loader.universe_mask
    tensor = loader.factor_tensor.numpy()
    coverage: dict[str, float] = {}
    for name in feature_names:
        idx = FEATURE_NAMES.index(name)
        # The factor tensor is [feature, stock, date]; row idx is the
        # feature's [stock, date] matrix.
        block = tensor[idx]
        finite = np.isfinite(block) & mask
        coverage[name] = (
            float(finite.sum() / mask.sum()) if mask.sum() else 0.0
        )
    return coverage


def run_r01_baseline(
    config_path: str | Path,
    contract_sha256: str,
    out_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    require_clean_tree: bool = True,
    wall_clock_budget_s: float = 6 * 3600,
    min_eligible: int | None = None,
) -> dict[str, Any]:
    """The p34 round-1 run (see the module docstring).  Returns the run
    summary; a fail-closed refusal raises :class:`R01Refusal` with the
    run left at its honest terminal state.  ``min_eligible`` is the
    ProductionGateRunner deployment tuning knob (the same seam the
    formal entries expose); production callers keep the default."""

    from .data_loader import AshareDataLoader
    from .identity import content_hash
    from .lifecycle import (
        LifecycleWriter,
        build_data_qualification_report,
        build_factor_qualification_report,
        replay_run,
    )
    from .research_domain import tier_restricted_features
    from .run_store import RunStore
    from .runspec import resolve_runtime_runspec

    started = time.monotonic()

    def _budget_check(where: str) -> None:
        elapsed = time.monotonic() - started
        if elapsed > wall_clock_budget_s:
            raise R01Refusal(
                f"R01 wall-clock budget exceeded at {where}: "
                f"{elapsed:.0f}s > {wall_clock_budget_s:.0f}s (p34 §3.6)"
            )

    root = Path(repo_root) if repo_root is not None else (
        Path(__file__).resolve().parents[1]
    )
    config_path = Path(config_path)
    out_root = Path(out_root) if out_root is not None else (
        root / "data" / "r1" / "r01_baseline"
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

    # -- formal gates BEFORE anything expensive (evaluation-entry order).
    gate_result = ProductionGateRunner(
        data_config, min_eligible=min_eligible
    ).run(mode="formal")
    if not gate_result.ok:
        failed = [
            f"{check.name} ({check.detail})"
            for check in gate_result.checks
            if not check.ok
        ]
        raise R01Refusal(
            "production gate check failed (p34 §7): " + "; ".join(failed)
        )

    # -- A02/A03: identity confirm → frozen spec → open run.
    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    if not loader.dataset_id:
        raise R01Refusal(
            "the composite snapshot identity could not be confirmed; a "
            "formal R01 run refuses to proceed without a dataset identity"
        )
    spec = resolve_runtime_runspec(
        dataset_id=loader.dataset_id,
        data_cutoff=loader.frozen_data_cutoff(),
        data_config=data_config,
        backtest_config=backtest_config,
        requested_budget=max(1, proto_cfg.screening.steps * proto_cfg.screening.batch_size),
        n_folds=len(proto_cfg.folds),
        research_domain=proto_cfg.domain,
        seeds=tuple(proto_cfg.seeds),
        searcher=model_config.searcher,
        max_formula_len=int(model_config.max_formula_len),
        repo_root=root,
        require_clean_tree=require_clean_tree,
    )
    store = RunStore(data_config.data_dir)
    handle = store.open_run(spec)
    run_id = handle.run_id
    writer = LifecycleWriter(handle)
    try:
        idea = {
            "idea_record_schema_version": 1,
            "hypothesis": (
                "R01 round-1 (p34): fixed simple baselines on the medium "
                "Tier-A configuration reveal whether data-qualified, "
                "cost-bearing simple signals leave net-of-cost active "
                "space worth a small search round."
            ),
            "scope": (
                f"configs/{config_path.name} (sha256 "
                f"{_sha256_file(config_path).upper()}), folds "
                f"{[str(f['test_end']) for f in raw['protocol']['folds']]}, "
                "arms: equal-weight benchmark / preregistered single "
                "factors / frozen-rule rank combos / cash control at "
                "0x/1x/2x costs; preregistration contract sha256 "
                f"{contract_sha256}."
            ),
            "economic_mechanism": (
                "Cross-sectional price momentum, short-horizon risk fit "
                "and reversal captured at the domain's every_10_days / "
                "horizon-10 execution point, net of A-share commission, "
                "stamp tax, transfer fee and slippage at 1x and 2x."
            ),
            "non_goals": (
                "No search arms; no slow_ab/short_a/unified rounds; no "
                "holdout or promotion claims (2021-2025 are development "
                "windows); no production default changes."
            ),
        }
        writer.record_idea(idea)
        writer.lock_spec()

        # -- load, drift guard, typed §5.1 report.
        loader.load_data()
        if loader.dataset_id != spec.dataset_id:
            writer.fail(
                reason=(
                    "the database content drifted between identity freeze "
                    "and load (A02 I2/p27 I6); the frozen spec no longer "
                    "matches the confirmed composite snapshot"
                )
            )
            raise R01Refusal(
                "dataset identity drifted between spec freeze and load"
            )
        _budget_check("load")
        frame_status = loader.loader_frame_status()
        expected_days, actual_days, missing_days = (
            _expected_and_missing_sessions(loader, data_config)
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
        except Exception as exc:  # noqa: BLE001 — evidence preserved, honest terminal
            writer.fail(
                reason=f"§5.1 data qualification gate refused the report: {exc}"
            )
            raise R01Refusal(f"data qualification refused: {exc}") from exc

        # -- the frozen baseline arms (p34 §3.2/§3.3).
        folds = resolve_folds_for(proto_cfg, loader)
        rows, _directions = run_baseline_arms(
            loader,
            proto_cfg,
            folds,
            backtest_config,
            model_config,
            reward_config,
        )
        _budget_check("baseline arms")

        # -- the report artifact (strict JSON inside the run store).
        safe_rows = rows_to_jsonable(rows)
        adjudication = decide_adjudication(summarize_arms(rows))
        report = {
            "r01_report_schema_version": R01_REPORT_SCHEMA_VERSION,
            "provenance": {
                "contract_sha256": str(contract_sha256),
                "contract_path": "docs/p34_r01_fixed_baseline_diagnostic_draft.md",
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path).upper(),
                "dataset_id": spec.dataset_id,
                "data_cutoff": spec.data_cutoff,
                "git_commit": spec.git_commit,
                "spec_id": spec.spec_id,
                "run_id": run_id,
                "research_domain": spec.research_domain,
                "tier_set": proto_cfg.tier_set,
                "seeds_recorded": list(spec.seeds),
                "seeds_note": (
                    "no stochastic arm runs in round 1; seeds are recorded "
                    "for the potential round-2 search and do not affect "
                    "these deterministic baselines"
                ),
                "requested_budget": spec.requested_budget,
                "consumed_semantic_evaluations": 0,
                "cost_ladder_definition": {
                    "0x": "all cost parameters zero (gross)",
                    "1x": "config verbatim (net)",
                    "2x": "cost rates doubled, commission floor unchanged",
                },
            },
            "adjudication": adjudication,
            "rows": safe_rows,
            "completed": True,
        }
        report_record = handle.write_artifact("r01_baseline_report", report)

        # -- §5.2 factor qualification with the real evidence blocks.
        feature_names = tier_restricted_features(
            proto_cfg.domain, proto_cfg.tier_set
        )
        coverage = _feature_coverage(loader, feature_names)
        baseline_evidence = {
            "completed": True,
            "result_hash": content_hash("r01_baseline", report),
            "report_artifact_id": report_record.artifact_id,
            "arm_count": len({r["candidate"] for r in rows}),
            "cost_levels": list(COST_LEVELS),
            "note": (
                "p34 §3.2 arms evaluated through evaluate_signal/"
                "baseline_candidates/benchmark_row (the single engine "
                "path); the frozen-rule combos and the cash control are "
                "part of this evidence"
            ),
        }
        ablation_rows = _leave_family_out_rows(
            loader, proto_cfg, folds, backtest_config, model_config,
            reward_config,
        )
        ablation_evidence = {
            "completed": True,
            "result_hash": content_hash(
                "r01_ablation", rows_to_jsonable(ablation_rows)
            ),
            "leave_family_out": rows_to_jsonable(ablation_rows),
            "note": (
                "leave-one-family-out subsets of the preregistered "
                "baseline signals scored by the same CandidateScorer "
                "(baseline_candidates) on the training windows only"
            ),
        }
        factor_report = build_factor_qualification_report(
            spec,
            feature_names=feature_names,
            universe_codes=list(loader.ts_codes),
            config=data_config,
            coverage_by_feature=coverage,
            baseline_evidence=baseline_evidence,
            ablation_evidence=ablation_evidence,
        )
        try:
            writer.qualify_factor_set(factor_report)
        except Exception as exc:  # noqa: BLE001 — evidence preserved, honest terminal
            writer.fail(
                reason=(
                    f"§5.2 factor qualification gate refused the report: {exc}"
                )
            )
            raise R01Refusal(f"factor qualification refused: {exc}") from exc

        # -- frozen adjudication (p34 §3.7): park or retire.
        if adjudication["worth_search"]:
            final_state = writer.current_state  # parked, non-terminal
        else:
            writer.retire(
                reason=(
                    "R01 round-1 adjudication: simple baselines show no "
                    "durable net-of-cost space; report artifact "
                    f"{report_record.artifact_id}; bottleneck: "
                    f"{adjudication['bottleneck']}"
                )
            )
            final_state = writer.current_state
        summary = {
            "spec_id": spec.spec_id,
            "run_id": run_id,
            "state": replay_run(store.root, spec.spec_id, run_id) or final_state,
            "dataset_id": spec.dataset_id,
            "data_cutoff": spec.data_cutoff,
            "adjudication": adjudication,
            "report_artifact_id": report_record.artifact_id,
            "row_count": len(rows),
            "wall_seconds": round(time.monotonic() - started, 1),
        }
    finally:
        handle.close()

    # -- measurement-directory copy (data/r1/r01_baseline/).
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "r01_baseline_report.json"
    report_path.write_text(
        json.dumps(rows_to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_path = out_root / "r01_baseline_run_meta.json"
    meta_path.write_text(
        json.dumps(rows_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["report_sha256"] = _sha256_file(report_path)
    summary["report_path"] = str(report_path)
    summary["meta_path"] = str(meta_path)
    return summary


# --- small helpers kept import-light --------------------------------------------


def load_config_file(config_path: Path) -> dict:
    """YAML parse of the run configuration (the production makers do all
    validation; this helper exists so tests can monkeypatch the file
    boundary)."""

    import yaml

    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def resolve_folds_for(proto_cfg, loader) -> list:
    """Fold resolution through the single authority (eval_folds)."""

    from .eval_folds import resolve_folds

    return resolve_folds(
        proto_cfg.folds,
        loader.dates,
        frequency=proto_cfg.frequency,
        horizon=proto_cfg.horizon,
    )


def _leave_family_out_rows(
    loader, proto_cfg, folds, backtest_config, model_config, reward_config
) -> list[dict]:
    """Leave-one-family-out ablation rows (p8 §5.2 check 6): each subset
    of the preregistered baseline signals minus one family, scored by
    the same ``baseline_candidates`` path on the training windows."""

    from ashare_data.config import ProtocolConfig

    from .eval_search import baseline_candidates
    from .feature_registry import family_of

    names = list(proto_cfg.baseline_signals)
    families = sorted({family_of(name) for name in names})
    rows: list[dict] = []
    for family in families:
        subset = [name for name in names if family_of(name) != family]
        if not subset:
            continue
        subset_cfg = dataclasses.replace(
            proto_cfg, baseline_signals=list(subset)
        )
        assert isinstance(subset_cfg, ProtocolConfig)
        for fold in folds:
            for row in baseline_candidates(
                loader, subset_cfg, fold, backtest_config, model_config,
                reward_config,
            ):
                rows.append(
                    {
                        "left_out_family": family,
                        "subset": list(subset),
                        "candidate": row["candidate"],
                        "fold_test_end": row["fold_test_end"],
                        "direction": row["direction"],
                        "val_reward": row["val_reward"],
                        "val_icir": row["val_icir"],
                        "train_icir": row["train_icir"],
                    }
                )
    return rows
