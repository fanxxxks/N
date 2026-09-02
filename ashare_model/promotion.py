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
record ``promotion_rule_version``.  The promotion-gate numbering
(G1-G7) is independent of the data-quality gate numbering in
``ashare_data/gates.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ashare_data.io_utils import atomic_write_json, read_json_safe

from .identity import formula_hash

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
# feature_registry_status gate.  Recorded in every verdict.
PROMOTION_RULE_VERSION = "2"


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
    failures: list[str] = []
    for cell in stress.get("grid", []):
        label = (
            f"cost x{cell.get('cost_multiplier')} "
            f"capital x{cell.get('capital_multiplier')}"
        )
        if (cell.get("n_days") or 0) < 2:
            failures.append(f"{label}: no stitched OOS series")
            continue
        sharpe = cell.get("sharpe")
        if sharpe is None or sharpe < thresholds.min_sharpe:
            failures.append(
                f"{label}: sharpe {sharpe} < {thresholds.min_sharpe}"
            )
        mdd = cell.get("max_drawdown")
        if mdd is not None and mdd > thresholds.max_drawdown:
            failures.append(
                f"{label}: max_drawdown {mdd:.3f} > {thresholds.max_drawdown}"
            )
        excess_ann = _annualized_excess(cell.get("excess_return"), cell.get("n_days"))
        if excess_ann is None or excess_ann < thresholds.min_excess_annualized:
            failures.append(
                f"{label}: annualized excess {excess_ann} < "
                f"{thresholds.min_excess_annualized}"
            )
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
) -> dict:
    """Apply the seven promotion gates to a current protocol artifact.

    Every gate returns ``{"passed": bool, "reasons": [...]}``; the
    challenger is promoted only when all seven pass.  ``allowed_data_tiers``
    is the P2 data-tier policy (default Tier A only; Tier B requires an
    explicit separate comparison; Tier C raises).  ``today`` is injectable
    for tests (paper-window completion check).
    """

    thresholds = thresholds or PromotionThresholds()
    allowed = validate_allowed_tiers(allowed_data_tiers)
    gates: dict[str, dict] = {}
    top = artifact.get("top_trial")

    # G1 -- data & formula P0 gate.  Version equality uses the single
    # comparison primitive from ashare_model.artifact_versions; a missing
    # field never matches and renders as ``None`` in the message (pinned
    # by the F2 characterization tests).  G1 checks protocol / reward /
    # execution / portfolio_constructor only; the wider field set
    # (model_version, searcher, imitation) belongs to classify_strategy.
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
    top_rows = _top_rows(artifact) if top else []
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
    gates["data_formula_p0"] = _gate(not reasons, reasons)

    # G2 -- statistical significance (stitched matrix).
    reasons = []
    dsr = artifact.get("dsr") or {}
    if dsr.get("dsr") is None:
        reasons.append("no Deflated Sharpe in the artifact")
    elif dsr["dsr"] < thresholds.dsr_min:
        reasons.append(f"dsr {dsr['dsr']:.3f} < {thresholds.dsr_min}")
    mt = artifact.get("max_t") or {}
    if mt.get("significant_95") is None:
        reasons.append("no max-t in the artifact")
    elif not mt["significant_95"]:
        reasons.append(
            f"max-t not significant (p={mt.get('p_value')} > "
            f"{thresholds.max_t_p_max})"
        )
    gates["significance"] = _gate(not reasons, reasons)

    # G3 -- excess return & risk constraints on the stitched top trial.
    reasons = []
    if top:
        if top.get("sharpe") is None or top["sharpe"] < thresholds.min_sharpe:
            reasons.append(
                f"stitched sharpe {top.get('sharpe')} < {thresholds.min_sharpe}"
            )
        excess_ann = _annualized_excess(top.get("excess_return"), top.get("n_days"))
        if excess_ann is None or excess_ann < thresholds.min_excess_annualized:
            reasons.append(
                f"annualized excess return {excess_ann} < "
                f"{thresholds.min_excess_annualized}"
            )
        mdd = top.get("max_drawdown")
        if mdd is not None and mdd > thresholds.max_drawdown:
            reasons.append(f"max drawdown {mdd:.3f} > {thresholds.max_drawdown}")
        turnover = top.get("average_turnover_mean")
        if turnover is not None and turnover > thresholds.max_average_turnover:
            reasons.append(
                f"average turnover {turnover:.4f} > "
                f"{thresholds.max_average_turnover}"
            )
        capacity = top.get("capacity_utilization_max")
        if (
            capacity is not None
            and capacity > thresholds.max_capacity_utilization
        ):
            reasons.append(
                f"capacity utilization {capacity:.3f} > "
                f"{thresholds.max_capacity_utilization}"
            )
    else:
        reasons.append("no top trial to constrain")
    gates["excess_and_risk"] = _gate(not reasons, reasons)

    # G4 -- cost & capacity stress.
    reasons = []
    if stress is None:
        reasons.append("no cost/capacity stress result provided")
    else:
        reasons = _stress_failures(stress, thresholds)
    gates["cost_capacity_stress"] = _gate(not reasons, reasons)

    # G5 -- at least one complete future paper-trading observation window.
    reasons = []
    if top_rows:
        last = top_rows[-1]
        fh = formula_hash(last.get("formula"), last.get("formula_text"))
        if paper_registry is None:
            reasons.append("no paper-window registry provided")
        else:
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
    else:
        reasons.append("no top trial to promote")
    gates["paper_window"] = _gate(not reasons, reasons)

    # G6 -- data credibility tier (P2-03/P2-04): the challenger formula is
    # traced back to the tiers of its features; every feature must belong
    # to the allowed tiers (default: Tier A only).  Tier C can never be
    # allowed here (validate_allowed_tiers rejects it up front).
    reasons = []
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
    gates["data_tier"] = _gate(not reasons, reasons)

    # G7 -- feature registry status (P12): the challenger formula's
    # features must all be registry-promotable (promotion_allowed, not
    # deprecated).  Same top-trial source as G6, so the artifact loading
    # path is covered by construction (newly sampled and historical
    # formulas both arrive as artifact top rows).  Fail-closed: an
    # untraceable formula or a registry gap is a rejection.  The
    # promotion-gate numbering (G1-G7) is independent of the data-quality
    # gate numbering in ashare_data/gates.py.
    reasons = []
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
    gates["feature_registry_status"] = _gate(not reasons, reasons)

    return {
        "promoted": all(g["passed"] for g in gates.values()),
        "challenger": {
            "candidate": top.get("candidate") if top else None,
            "seed": top.get("seed") if top else None,
            "formula_hash": (
                formula_hash(
                    top_rows[-1].get("formula"),
                    top_rows[-1].get("formula_text"),
                )
                if top_rows
                else None
            ),
        },
        "data_tier_policy": {
            "data_tier_version": DATA_TIER_VERSION,
            "allowed_tiers": list(allowed),
            "formula_data_tier": tier_report,
            "time_rules": dict(TIER_TIME_RULES),
        },
        "promotion_rule_version": PROMOTION_RULE_VERSION,
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--artifact", default=None, help="current protocol result JSON"
    )
    parser.add_argument("--config", default=None, help="project config (for stress)")
    parser.add_argument("--output", default="data/promotion_verdict.json")
    parser.add_argument("--regime", default="data/holdout_registry.json")
    parser.add_argument("--paper", default="data/paper_windows.json")
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

    regime = RegimeRegistry(root / args.regime)
    paper = PaperWindowRegistry(root / args.paper)

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
        )
    else:
        verdict = evaluate_challenger(
            artifact,
            regime=regime,
            paper_registry=paper,
            stress=stress,
            allowed_data_tiers=("A", "B") if args.allow_tier_b else ("A",),
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
