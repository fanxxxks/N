"""P37 R02 round-1 six-cell portfolio/cost matrix driver.

Executable projection of ``docs/p37_r02_portfolio_cost_matrix_draft.md``
(APPROVED via user direct instruction 2026-09-19 work-package 4 「把 R02
首轮缩成六格，先固定调仓周期」).  Round 1 answers ONLY the order-threshold
and holdings-structure question: the rebalance period is frozen at
``every_10_days``/horizon 10, capital at 100k CNY, and the panel is the
frozen 2x3 grid (holdings 10/20 x min-trade 1000/2500/5000) with the
coupled per-cell params preregistered verbatim (:data:`R02_GRID`).

What this module does (and deliberately does not do):

* walks one run through the real lifecycle chain — genesis (typed
  IdeaRecord) → SPEC_LOCKED → DATA_QUALIFIED (real G1–G8 + typed loader
  frames) → the six-cell panel evidence → the frozen four-way verdict;
* evaluates the p34 frozen candidate arms (three Tier-A single factors +
  two frozen-rule combos) on every cell / fold / cost level through
  ``evaluate_signal`` — the single engine path — with directions learned
  once per fold on the reference cell (train-window rank-IC sign,
  portfolio-config independent) and reused FROZEN across cells;
* adds the continuous-vs-lot observability through
  ``eval_metrics.lot_gap_fields`` (engine untouched) so the 整手 account's
  deviation from the ideal weights is measured, not asserted;
* applies the frozen adjudication (§7): ``stable_region_found`` /
  ``isolated_single_winner`` (recorded unstable, never a parameter pick)
  park the run at ``DATA_QUALIFIED``; ``gross_info_execution_blocked`` /
  ``panel_no_reliable_info`` retire it — round 2 (periods) is a separate
  authorization and run identity either way;
* it does NOT compare rebalance periods, optimizer methods, capital
  scales or 2x costs; does NOT run any searcher; does NOT touch
  production defaults; does NOT claim holdout/promotion semantics.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ashare_data.gates import (
    ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
    ProductionGateRunner,
)
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION

#: Report artifact schema (measurement artifact; not an
#: ``artifact_schemas`` consumer — p34 §4 version-table precedent).
R02_REPORT_SCHEMA_VERSION = 1

#: p37 §3.2 frozen adjudication thresholds (never tuned after seeing
#: data).  The per-cell holdings floor (⌈top_n/2⌉) is structural, not a
#: tunable — a book that never fills half its slots cannot answer the
#: order-threshold question.
R02_ADJUDICATION_RULES: dict[str, float] = {
    "min_invested_ratio": 0.5,
    "min_positive_folds": 3,
    "max_failed_folds": 1,
}


@dataclasses.dataclass(frozen=True)
class R02CellSpec:
    """One frozen grid cell: the five coupled portfolio parameters."""

    name: str
    top_n: int
    buy_rank: int
    sell_rank: int
    single_weight_cap: float
    min_trade_amount: float


#: p37 §3.2 — the six-cell panel.  Coupling rules frozen verbatim:
#: ``buy_rank = top_n`` (equal-weight unit = min(1/buy_rank, cap)),
#: ``single_weight_cap = 1/top_n`` (full investment representable),
#: ``sell_rank = ⌈1.5·top_n⌉`` (the r1_medium_a buffer ratio preserved).
R02_GRID: tuple[R02CellSpec, ...] = (
    R02CellSpec("n10_m1000", 10, 10, 15, 0.10, 1000.0),
    R02CellSpec("n10_m2500", 10, 10, 15, 0.10, 2500.0),
    R02CellSpec("n10_m5000", 10, 10, 15, 0.10, 5000.0),
    R02CellSpec("n20_m1000", 20, 20, 30, 0.05, 1000.0),
    R02CellSpec("n20_m2500", 20, 20, 30, 0.05, 2500.0),
    R02CellSpec("n20_m5000", 20, 20, 30, 0.05, 5000.0),
)

#: The reference cell = the r1_medium_a backtest block verbatim: where
#: per-fold directions are learned and where the cash control runs.
R02_REFERENCE_CELL = "n20_m5000"

#: p37 §5: 0x (gross, attribution) + 1x (the configuration's real costs).
R02_COST_LEVELS = ("0x", "1x")

_GRID_BY_NAME: dict[str, R02CellSpec] = {cell.name: cell for cell in R02_GRID}
_CELL_FIELDS = (
    "top_n",
    "buy_rank",
    "sell_rank",
    "single_weight_cap",
    "min_trade_amount",
)
#: Grid-axis orders (the adjacency authority): one step = one index move
#: on exactly one axis.
_TOP_N_ORDER = (10, 20)
_MIN_TRADE_ORDER = (1000.0, 2500.0, 5000.0)


class R02Refusal(RuntimeError):
    """A fail-closed refusal of the R02 round-1 run (p37 §9)."""


def cell_variant(backtest_config, cell_name: str):
    """The frozen grid cell as a :class:`BacktestConfig` variant: exactly
    the five coupled portfolio fields replaced, everything else (costs,
    dates, cadence, pool, capital) verbatim from the caller's config."""

    cell = _GRID_BY_NAME.get(str(cell_name))
    if cell is None:
        raise ValueError(
            f"unknown cell {cell_name!r}: expected one of "
            f"{sorted(_GRID_BY_NAME)}"
        )
    return dataclasses.replace(
        backtest_config,
        **{field: getattr(cell, field) for field in _CELL_FIELDS},
    )


def cell_adjacent(a: str, b: str) -> bool:
    """p37 §7 adjacency: cells differ by exactly one grid step on
    exactly one axis (self-adjacency is not adjacency)."""

    if a == b:
        return False
    ca = _GRID_BY_NAME.get(str(a))
    cb = _GRID_BY_NAME.get(str(b))
    if ca is None or cb is None:
        raise ValueError(
            f"unknown cell in adjacency check: {a!r}, {b!r}"
        )
    dn = abs(
        _TOP_N_ORDER.index(ca.top_n) - _TOP_N_ORDER.index(cb.top_n)
    )
    dm = abs(
        _MIN_TRADE_ORDER.index(ca.min_trade_amount)
        - _MIN_TRADE_ORDER.index(cb.min_trade_amount)
    )
    return dn + dm == 1


# --- the panel -----------------------------------------------------------------


def run_r02_panel(
    loader,
    proto_cfg,
    folds: list,
    backtest_config,
    model_config,
    reward_config,
    *,
    on_fold=None,
) -> tuple[list[dict], dict[str, dict[str, dict[str, Any]]]]:
    """All cells x arms x folds x cost levels through the shared engine.

    Returns ``(rows, directions_by_fold)``.  Row candidates carry the
    ``"<cell>::<arm>"`` composite so the single ``summarize_arms``
    authority (r01_baseline) computes the per-(cell, arm) summaries; the
    benchmark is the cost-free per-fold reference and the cash control
    runs once at the reference cell (an empty book is cell-independent
    by construction — disclosed, not assumed).  ``on_fold`` is the
    budget callback; an exception from it propagates (the budget guard
    must fail the run, not one fold).
    """

    from .eval_folds import epoch_slice
    from .eval_metrics import benchmark_row, evaluate_signal
    from .eval_search import baseline_candidates
    from .r01_baseline import R01_COMBO_ARMS, cash_signal, combo_signal, cost_variant
    from .vocab import FEATURE_NAMES

    rows: list[dict] = []
    directions_by_fold: dict[str, dict[str, dict[str, Any]]] = {}
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
        except Exception as exc:  # noqa: BLE001 — recorded per p37 §8
            rows.append(_failed_row("benchmark:equal_weight", "ref", exc))

        # Directions: learned once per fold ON THE REFERENCE CELL through
        # the frozen train-window rule (p37 §4).  The resulting reference
        # 1x single-factor rows are real panel rows — no duplicate engine
        # run for that (cell, level).
        reference_config = cell_variant(backtest_config, R02_REFERENCE_CELL)
        singles: list[dict] = []
        singles_failed: Exception | None = None
        try:
            singles = baseline_candidates(
                loader, proto_cfg, fold, reference_config, model_config,
                reward_config,
            )
        except Exception as exc:  # noqa: BLE001 — recorded per p37 §8
            singles_failed = exc
        directions: dict[str, int] = {}
        direction_evidence: dict[str, dict[str, Any]] = {}
        for r in singles:
            arm = str(r["candidate"]).split(":", 1)[1]
            directions[arm] = int(r["direction"])
            direction_evidence[arm] = {
                "direction": int(r["direction"]),
                "val_reward": r.get("val_reward"),
                "val_icir": r.get("val_icir"),
            }
            rows.append(
                _ok(
                    f"{R02_REFERENCE_CELL}::{r['candidate']}",
                    "1x",
                    r,
                    R02_REFERENCE_CELL,
                )
            )
        directions_by_fold[str(fold.test_end)] = direction_evidence

        fold_data = epoch_slice(loader, fold)
        factors = fold_data.factors
        mask = fold_data.universe_mask

        def _direction_or_failed(name: str) -> int | None:
            if name in directions:
                return directions[name]
            if singles_failed is not None:
                return None
            raise KeyError(
                f"factor {name} missing from the direction evidence"
            )

        for level in R02_COST_LEVELS:
            for cell in R02_GRID:
                variant = cost_variant(
                    cell_variant(backtest_config, cell.name), level
                )
                for name in proto_cfg.baseline_signals:
                    candidate = f"{cell.name}::baseline:{name}"
                    if cell.name == R02_REFERENCE_CELL and level == "1x":
                        continue  # measured on the direction path above
                    try:
                        direction = _direction_or_failed(name)
                        if direction is None:
                            raise singles_failed  # type: ignore[misc]
                        signal = float(direction) * np.asarray(
                            factors[FEATURE_NAMES.index(name)],
                            dtype=np.float64,
                        )
                        metrics = evaluate_signal(signal, loader, fold, variant)
                        rows.append(
                            _ok(candidate, level, metrics, cell.name)
                        )
                    except Exception as exc:  # noqa: BLE001 — per p37 §8
                        rows.append(
                            _failed_row(candidate, level, exc, cell.name)
                        )
                for combo_name, member_names in R01_COMBO_ARMS:
                    candidate = f"{cell.name}::{combo_name}"
                    try:
                        members = []
                        for member in member_names:
                            direction = _direction_or_failed(member)
                            if direction is None:
                                raise singles_failed  # type: ignore[misc]
                            members.append(
                                (
                                    direction,
                                    factors[FEATURE_NAMES.index(member)],
                                )
                            )
                        signal = combo_signal(members, mask)
                        metrics = evaluate_signal(signal, loader, fold, variant)
                        rows.append(
                            _ok(candidate, level, metrics, cell.name)
                        )
                    except Exception as exc:  # noqa: BLE001 — per p37 §8
                        rows.append(
                            _failed_row(candidate, level, exc, cell.name)
                        )
        # The cash control: an all-NaN signal holds nothing regardless
        # of the cell — measured once at the reference cell (p37 §4).
        cash = cash_signal(len(loader.ts_codes), len(fold_data.dates))
        try:
            metrics = evaluate_signal(cash, loader, fold, reference_config)
            rows.append(
                _ok("control:cash", "1x", metrics, R02_REFERENCE_CELL)
            )
        except Exception as exc:  # noqa: BLE001 — per p37 §8
            rows.append(_failed_row("control:cash", "1x", exc, R02_REFERENCE_CELL))
    return rows, directions_by_fold


# --- cell assembly and the frozen adjudication ---------------------------------


def assemble_cell_summaries(arm_summaries: dict) -> dict[str, dict]:
    """Group the per-(cell::arm) summaries (from the single
    ``summarize_arms`` authority) into per-cell entries carrying the
    frozen cell params."""

    cells: dict[str, dict[str, Any]] = {}
    for key, summary in arm_summaries.items():
        key = str(key)
        if "::" not in key:
            continue
        cell_name, arm = key.split("::", 1)
        spec = _GRID_BY_NAME[cell_name]
        entry = cells.setdefault(
            cell_name,
            {
                "top_n": spec.top_n,
                "params": {
                    "top_n": spec.top_n,
                    "buy_rank": spec.buy_rank,
                    "sell_rank": spec.sell_rank,
                    "single_weight_cap": spec.single_weight_cap,
                    "min_trade_amount": spec.min_trade_amount,
                },
                "arms": {},
            },
        )
        entry["arms"][arm] = summary
    return cells


def decide_r02_adjudication(
    cell_summaries: dict[str, dict[str, Any]],
    rules: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The p37 §7 frozen rules (pure function; never sees prices, only
    the summarized rows).  Arm viability per cell, cell viability, the
    adjacency stability region, and the four-way verdict.  The
    ``single_year_dependence`` flag is informational (round 1's
    robustness criterion is the adjacency region, not the fold count);
    it never changes viability."""

    merged = dict(R02_ADJUDICATION_RULES)
    merged.update(rules or {})
    cells_out: dict[str, dict[str, Any]] = {}
    for cell_name in sorted(_GRID_BY_NAME):
        spec = _GRID_BY_NAME[cell_name]
        entry = cell_summaries.get(cell_name) or {}
        arms_out: dict[str, dict[str, Any]] = {}
        viable_arms: list[str] = []
        for arm, summary in (entry.get("arms") or {}).items():
            flags: list[str] = []
            excess = summary.get("mean_net_excess_1x")
            ir = summary.get("active_ir_1x")
            invested = summary.get("invested_ratio_mean_1x")
            holdings = summary.get("mean_holdings_1x")
            failed = int(summary.get("failed_folds") or 0)
            net_space = bool(
                excess is not None
                and ir is not None
                and excess > 0
                and ir > 0
            )
            if not net_space:
                flags.append("no_net_space")
            if failed > int(merged["max_failed_folds"]):
                flags.append("unreliable")
            elif failed:
                flags.append("fold_failure")
            if invested is None or invested < float(
                merged["min_invested_ratio"]
            ):
                flags.append("low_activity")
            holdings_floor = math.ceil(spec.top_n / 2)
            if holdings is None or holdings < holdings_floor:
                flags.append("thin_book")
            if net_space and int(
                summary.get("positive_folds_1x") or 0
            ) < int(merged["min_positive_folds"]):
                flags.append("single_year_dependence")
            blocking = [
                flag
                for flag in ("low_activity", "thin_book", "unreliable")
                if flag in flags
            ]
            viable = net_space and not blocking
            if viable:
                viable_arms.append(arm)
            arms_out[arm] = {
                "viable": viable,
                "flags": flags,
                "evidence": {
                    "mean_net_excess_1x": excess,
                    "active_ir_1x": ir,
                    "invested_ratio_mean_1x": invested,
                    "mean_holdings_1x": holdings,
                    "failed_folds": failed,
                    "holdings_floor": holdings_floor,
                },
            }
        cells_out[cell_name] = {
            "top_n": spec.top_n,
            "arms": arms_out,
            "viable": bool(viable_arms),
            "viable_arms": viable_arms,
        }
    viable_cells = [name for name in cells_out if cells_out[name]["viable"]]
    stable_pairs = sorted(
        [a, b]
        for i, a in enumerate(viable_cells)
        for b in viable_cells[i + 1:]
        if cell_adjacent(a, b)
    )
    gross_info_arms: list[str] = []
    for cell_name, entry in cell_summaries.items():
        for arm, summary in (entry.get("arms") or {}).items():
            stitched = summary.get("stitched") or {}
            s0 = stitched.get("0x") or {}
            gross_excess = s0.get("excess_return")
            gross_ir = s0.get("oos_active_ir")
            if (
                gross_excess is not None
                and gross_excess > 0
                and gross_ir is not None
                and gross_ir > 0
            ):
                gross_info_arms.append(f"{cell_name}::{arm}")

    isolated: str | None = None
    if viable_cells and stable_pairs:
        verdict = "stable_region_found"
        note = (
            "adjacent viable cells exist — the net-of-cost space holds "
            "over a region, not a lone grid point; round 2 (period "
            "comparison) is a separate authorization"
        )
    elif viable_cells:
        verdict = "isolated_single_winner"
        isolated = viable_cells[0]
        note = (
            "exactly one viable cell with no viable neighbor — recorded "
            "as an UNSTABLE result（不稳定结果）: never a parameter-choice "
            "basis and never a default-config change justification"
        )
    elif gross_info_arms:
        verdict = "gross_info_execution_blocked"
        note = (
            "the gross signal carries information (0x stitched space) but "
            "no cell is viable — continue execution diagnostics; the "
            "suppression/unfilled/invested/initial-funding/lotgap "
            "evidence is carried in this report"
        )
    else:
        verdict = "panel_no_reliable_info"
        note = (
            "no viable cell and no gross information anywhere on the "
            "frozen panel — pause extended search (round-2 period "
            "comparison and matrix expansion both pause); the bottleneck "
            "is the signal, not the execution structure"
        )
    return {
        "verdict": verdict,
        "note": note,
        "viable_cells": viable_cells,
        "isolated_cell": isolated,
        "stable_pairs": stable_pairs,
        "gross_info_arms": sorted(gross_info_arms),
        "cells": cells_out,
        "rules": dict(merged),
    }


# --- lifecycle-driving run ------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_and_missing_sessions(loader, data_config, *, cutoff: str):
    """The §5.1 calendar reconciliation, capped at the confirmed data
    cutoff (the a6689a1 repair shared with the R01 driver — future
    pre-listed sessions are never missing trading days)."""

    from .r01_baseline import _expected_and_missing_sessions as _r01_impl

    return _r01_impl(loader, data_config, cutoff=cutoff)


def run_r02_matrix(
    config_path: str | Path,
    contract_sha256: str,
    out_root: str | Path | None = None,
    *,
    repo_root: str | Path | None = None,
    require_clean_tree: bool = True,
    wall_clock_budget_s: float = 6 * 3600,
    min_eligible: int | None = None,
    adjudication_rules: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The p37 round-1 run (see the module docstring).  Returns the run
    summary; a fail-closed refusal raises :class:`R02Refusal` with the
    run left at its honest terminal state.  ``adjudication_rules`` is
    the test-only seam that runs the REAL adjudicator with relaxed
    canary thresholds (the p34 rev2 T8 pattern); production callers keep
    the frozen defaults."""

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
        load_config_file,
        resolve_folds_for,
        rows_to_jsonable,
        summarize_arms,
    )
    from .run_store import RunStore
    from .runspec import resolve_runtime_runspec

    started = time.monotonic()

    def _budget_check(where: str) -> None:
        elapsed = time.monotonic() - started
        if elapsed > wall_clock_budget_s:
            raise R02Refusal(
                f"R02 wall-clock budget exceeded at {where}: "
                f"{elapsed:.0f}s > {wall_clock_budget_s:.0f}s (p37 §8)"
            )

    root = Path(repo_root) if repo_root is not None else (
        Path(__file__).resolve().parents[1]
    )
    config_path = Path(config_path)
    out_root = Path(out_root) if out_root is not None else (
        root / "data" / "r1" / "r02_matrix"
    )

    # -- C5 hardening: the supplied contract sha256 must match the
    # committed contract file (self-verified provenance; refuses before
    # anything expensive runs).
    contract_path = root / "docs" / "p37_r02_portfolio_cost_matrix_draft.md"
    actual_contract_sha = _sha256_file(contract_path)
    if str(contract_sha256).lower() != actual_contract_sha:
        raise R02Refusal(
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
        raise R02Refusal(
            "the composite snapshot identity could not be confirmed; a "
            "formal R02 run refuses to proceed without a dataset identity"
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
        raise R02Refusal(
            "production gate check failed (p37 §9): " + "; ".join(failed)
        )
    store = RunStore(data_config.data_dir)
    handle = store.open_run(spec)
    run_id = handle.run_id
    writer = LifecycleWriter(handle)

    def _mark_failed(reason: str) -> None:
        """Honest-terminal guard: mark FAILED unless the run already
        sits in a terminal state (a double transition is itself a
        machine error)."""

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
                "R02 round-1 (p37): on the frozen every_10_days/horizon-10 "
                "execution point with 100k CNY capital, the 2x3 grid of "
                "holdings 10/20 x min-trade 1000/2500/5000 measures "
                "whether the preregistered simple signals leave a stable "
                "net-of-cost region, an isolated unstable cell, an "
                "execution-blocked panel, or no reliable information."
            ),
            "scope": (
                f"configs/{config_path.name} (sha256 "
                f"{_sha256_file(config_path).upper()}), folds "
                f"{[str(f['test_end']) for f in raw['protocol']['folds']]}, "
                "six-cell grid with frozen coupled params (buy_rank=top_n, "
                "single_weight_cap=1/top_n, sell_rank=ceil(1.5*top_n)), "
                "arms: the p34 frozen baseline factors / rank combos / "
                "cash control at 0x/1x costs; directions learned once per "
                f"fold on the reference cell {R02_REFERENCE_CELL} and "
                "reused frozen across cells; preregistration contract "
                f"sha256 {contract_sha256}."
            ),
            "economic_mechanism": (
                "Cross-sectional price momentum, short-horizon risk fit "
                "and reversal captured at the domain's every_10_days / "
                "horizon-10 execution point, net of A-share commission, "
                "stamp tax, transfer fee and slippage at 1x, with the "
                "order-threshold and holdings-structure friction "
                "measured through the lot gap and the execution funnel."
            ),
            "non_goals": (
                "No rebalance-period comparison (round 2); no optimizer/"
                "capital/2x-cost arms; no search; no holdout or promotion "
                "claims (2021-2025 are development windows); no "
                "production default changes."
            ),
        }
        writer.record_idea(idea)
        writer.lock_spec()

        # -- load, drift guard, typed §5.1 report.
        loader.load_data()
        if loader.dataset_id != spec.dataset_id:
            _mark_failed(
                "the database content drifted between identity freeze "
                "and load (A02 I2/p27 I6); the frozen spec no longer "
                "matches the confirmed composite snapshot"
            )
            raise R02Refusal(
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
            # Machine discriminator (p8 §4): a COMPLETED gate verdict
            # failing validation is REJECTED with its registered report;
            # an incomplete evaluation is FAILED.
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
                    f"§5.1 data qualification gate did not complete: {exc}"
                )
            raise R02Refusal(f"data qualification refused: {exc}") from exc

        # -- the frozen six-cell panel (p37 §3-§5).
        folds = resolve_folds_for(proto_cfg, loader)
        rows, directions_by_fold = run_r02_panel(
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

        # -- summaries, cell assembly, frozen adjudication.
        arm_summaries = summarize_arms(rows)
        cell_summaries = assemble_cell_summaries(arm_summaries)
        adjudication = decide_r02_adjudication(
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
            "prior_version_numbers_exist": False,
            "prior_version_note": (
                "no pre-p35 (frozen-weight accounting) engine numbers "
                "exist (p34 measurement log §6.3) and R01 never executed "
                "an arm, so no old-vs-new numeric attribution exists for "
                "this matrix — recorded, not fabricated"
            ),
            "semantic_delta_reference": (
                "p35 share-level accounting (EXECUTION_SPEC_VERSION 3->4): "
                "non-due sessions drift weights by the adjusted o2o return; "
                "due sessions execute at share level on the raw entry open "
                "with lot floors and full exits"
            ),
        }
        reporting_notes = {
            "period_frozen": (
                "round 1 freezes every_10_days/horizon 10 — the period "
                "comparison is round 2 by separate authorization; nothing "
                "in this report supports a period choice"
            ),
            "direction_reuse": (
                "directions are learned once per fold on the reference "
                "cell n20_m5000 (train-window validation rank-IC sign, "
                "portfolio-config independent by construction) and reused "
                "frozen across all six cells — direction learning never "
                "varies with the cell"
            ),
            "adjacency": (
                "cells are adjacent iff exactly one axis (top_n or "
                "min_trade) moves exactly one grid step; stability means "
                "a viable cell has a viable neighbor"
            ),
            "no_parameter_pick": (
                "the verdict identifies stable regions only; an isolated "
                "winning cell is recorded as an unstable result and never "
                "justifies a parameter choice or a default-config change"
            ),
            "combo_rank_scale": (
                "combos average raw per-date cross-sectional ranks "
                "(reward._rankdata) — the p34 §3.2 frozen construction, "
                "disclosed; percentile standardization is a separate "
                "definition for a separate package"
            ),
        }
        report = {
            "r02_report_schema_version": R02_REPORT_SCHEMA_VERSION,
            "provenance": {
                "contract_sha256": str(contract_sha256),
                "contract_path": "docs/p37_r02_portfolio_cost_matrix_draft.md",
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path).upper(),
                "dataset_id": spec.dataset_id,
                "data_cutoff": spec.data_cutoff,
                "git_commit": spec.git_commit,
                "spec_id": spec.spec_id,
                "run_id": run_id,
                "research_domain": spec.research_domain,
                "tier_set": proto_cfg.tier_set,
                "frequency": proto_cfg.frequency,
                "horizon": proto_cfg.horizon,
                "initial_capital": float(backtest_config.initial_capital),
                "seeds_recorded": list(spec.seeds),
                "seeds_note": (
                    "no stochastic arm runs in round 1; the seeds are "
                    "recorded for provenance and do not affect these "
                    "deterministic panels"
                ),
                "requested_budget": spec.requested_budget,
                "consumed_semantic_evaluations": 0,
                "cost_ladder_definition": {
                    "0x": "all cost parameters zero (gross)",
                    "1x": "config verbatim (net, the real costs)",
                },
                "grid": [
                    {
                        "name": cell.name,
                        "top_n": cell.top_n,
                        "buy_rank": cell.buy_rank,
                        "sell_rank": cell.sell_rank,
                        "single_weight_cap": cell.single_weight_cap,
                        "min_trade_amount": cell.min_trade_amount,
                    }
                    for cell in R02_GRID
                ],
                "reference_cell": R02_REFERENCE_CELL,
                "cell_coupling_rules": (
                    "buy_rank=top_n; single_weight_cap=1/top_n (full "
                    "investment representable); sell_rank=ceil(1.5*top_n) "
                    "(the r1_medium_a buffer ratio preserved); every "
                    "other field verbatim from the frozen base config"
                ),
                "directions_by_fold": rows_to_jsonable(directions_by_fold),
            },
            "grid_adjudication": adjudication,
            "data_completeness": data_completeness,
            "benchmark_status": benchmark_status,
            "cell_summaries": cell_summaries,
            "arm_summaries": arm_summaries,
            "engine_version_attribution": engine_version_attribution,
            "reporting_notes": reporting_notes,
            "rows": rows_to_jsonable(rows),
            "completed": True,
        }
        report_record = handle.write_artifact("r02_matrix_report", report)

        # -- the frozen verdict decides the administrative walk (p37 §7):
        # a region (stable or isolated) parks the run at DATA_QUALIFIED —
        # round 2 is a separate authorization; blocked/no-info retires.
        verdict = adjudication["verdict"]
        if verdict in ("stable_region_found", "isolated_single_winner"):
            final_state = writer.current_state  # parked, non-terminal
        else:
            writer.retire(
                reason=(
                    f"R02 round-1 verdict {verdict}: {adjudication['note']}; "
                    f"report artifact {report_record.artifact_id}"
                )
            )
            final_state = writer.current_state
        summary = {
            "spec_id": spec.spec_id,
            "run_id": run_id,
            "state": replay_run(store.root, spec.spec_id, run_id) or final_state,
            "dataset_id": spec.dataset_id,
            "data_cutoff": spec.data_cutoff,
            "verdict": verdict,
            "grid_adjudication": adjudication,
            "report_artifact_id": report_record.artifact_id,
            "row_count": len(rows),
            "report_content_hash": content_hash("r02_matrix", report),
            "wall_seconds": round(time.monotonic() - started, 1),
        }
    except R02Refusal as exc:
        # p37 §8/§9: any post-open refusal leaves the run at its honest
        # terminal — never dangling non-terminal.
        _mark_failed(str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 — the honest-terminal catch-all
        _mark_failed(
            f"R02 round-1 runtime failure: {type(exc).__name__}: {exc}"
        )
        raise R02Refusal(f"runtime failure: {exc}") from exc
    finally:
        handle.close()

    # -- measurement-directory copy (data/r1/r02_matrix/).
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "r02_matrix_report.json"
    report_path.write_text(
        json.dumps(rows_to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_path = out_root / "r02_matrix_run_meta.json"
    meta_path.write_text(
        json.dumps(rows_to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["report_sha256"] = _sha256_file(report_path)
    summary["report_path"] = str(report_path)
    summary["meta_path"] = str(meta_path)
    return summary
