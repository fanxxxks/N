"""Protocol-artifact assembly for the evaluation protocol.

Extracted from ``evaluation.py`` (P7 Phase A5) by reason-to-change: this
module owns the *artifact schema* — JSON sanitization, provenance payload
blocks (universe policy, data regime, data tier), the top-level
``build_result`` assembly and the ledger-recording seam.  It changes when
the artifact *schema* changes — not when fold contracts, metric
definitions, statistical corrections or search backends change.

Monkeypatch compatibility: ``build_result`` reads ``PROTOCOL_VERSION`` and
``REWARD_VERSION`` **through the facade at call time** — tests pin versions
via ``monkeypatch.setattr(evaluation, "REWARD_VERSION", ...)`` and the
facade remains the protocol version's single home.
"""

from __future__ import annotations

import math

from ashare_data.config import BacktestConfig, ProtocolConfig
from ashare_portfolio.execution_spec import execution_provenance

from .data_loader import AshareDataLoader
from .data_tier import DATA_TIER_VERSION, formula_data_tier_report
from .eval_corrections import dsr_from_rows, max_t_from_rows
from .eval_metrics import (
    aggregate_results,
    stitch_oos_series,
    stitched_metrics,
    top_trial,
)
from .research_domain import RESEARCH_DOMAIN_VERSION


def _sanitize(value):
    """Replace non-finite floats with ``None`` so results stay valid JSON."""

    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def universe_policy_payload(loader: AshareDataLoader) -> dict | None:
    """The universe policy actually applied to a run, for artifact
    provenance: the configured index codes, the listing-age rule and the
    membership-boundary convention, plus the degraded flag.  No data hash
    and no lineage are recorded."""

    policy = getattr(loader, "universe_policy", None)
    if policy is None:
        return None
    return {
        "index_codes": [str(code) for code in policy.index_codes],
        "min_listed_sessions": int(policy.min_listed_sessions),
        "membership_end_inclusive": bool(policy.membership_end_inclusive),
        "degraded": (
            bool(loader.universe_status.degraded)
            if loader.universe_status is not None
            else None
        ),
    }


def _regime_payload(regime, proto_cfg: ProtocolConfig) -> dict | None:
    """The data regime in force for the artifact (record-only): the dev
    cutoff, the policy, the locked slice (if any) and each fold's window
    classification."""

    if regime is None or regime.regime is None:
        return None
    r = regime.regime
    locked = r.locked_slice
    return {
        "declared_at": r.declared_at,
        "dev_cutoff": r.dev_cutoff,
        "policy": r.policy,
        "locked_slice": (
            {
                "start": locked.start,
                "end": locked.end,
                "dataset_id": locked.dataset_id,
                "locked_at": locked.locked_at,
                "note": locked.note,
            }
            if locked is not None
            else None
        ),
        "folds": [
            {
                "train_end": f.train_end,
                "test_end": f.test_end,
                "kind": regime.classify_window(f.train_end, f.test_end),
            }
            for f in proto_cfg.folds
        ],
    }


def _data_tier_block(formula, formula_text: str | None) -> dict | None:
    """Compact credibility-tier block for one formula (v21, P2-02).

    Resolves the formula's features to their A/B/C tiers via
    :func:`ashare_model.data_tier.formula_data_tier_report` (``formula``
    tokens first; bare baseline rows fall back to ``formula_text``).
    ``None`` when there is no traceable formula (e.g. ``equal_weight``).
    """

    report = formula_data_tier_report(tokens=formula, feature_name=formula_text)
    if report is None:
        return None
    return {
        "data_tier_version": report["data_tier_version"],
        "max_tier": report["max_tier"],
        "tiers_used": report["tiers_used"],
    }


def build_result(
    proto_cfg: ProtocolConfig,
    tier_name: str,
    tier,
    rows: list[dict],
    data_end_date: str | None = None,
    extra_trial_rows: list[dict] | None = None,
    max_t_perms: int = 5000,
    universe_policy: dict | None = None,
    dataset_id: str | None = None,
    random_budget_matched: bool | None = None,
    random_budget: int | None = None,
    ledger: dict | None = None,
    regime=None,
    backtest_config: BacktestConfig | None = None,
) -> dict:
    """Assemble the protocol artifact (schema contract, see module docstring).

    v20: the artifact's adjudication (``top_trial`` / ``dsr`` / ``max_t``)
    consumes the **stitched** trial matrix (one trial = one (candidate,
    seed) series); the raw per-fold rows stay in ``rows`` for drill-down,
    the stitched trials live in the ``stitched`` block, and ``ledger`` /
    ``data_regime`` record the trial-ledger and data-regime provenance of
    the run (``None`` when absent).

    ``extra_trial_rows`` are trial rows from earlier protocol artifacts
    (e.g. screening runs whose OOS trials must count towards the DS/max-t
    multiplicity correction); they are stitched separately — a prior run's
    trial is never merged into this run's series — and join the correction
    pool.  ``universe_policy`` records the PIT universe policy fields that
    produced the rows.  ``dataset_id`` binds the artifact to the immutable
    dataset manifest the rows were measured on (``None`` for legacy
    databases, recorded as ``null``).  ``random_budget_matched`` /
    ``random_budget`` record the T1-05 baseline budget actually used.
    v21 (P2): every row / stitched trial / top trial records its
    free-data credibility tier (``data_tier`` block, resolved from the
    formula's features); ``data_tier_version`` pins the mapping at the
    artifact level.
    """

    # Late binding through the facade: tests pin versions via
    # ``monkeypatch.setattr(evaluation, "REWARD_VERSION", ...)`` and the
    # facade is PROTOCOL_VERSION's single home.
    from ashare_model import evaluation as _facade  # noqa: PLC0415

    if backtest_config is None:
        backtest_config = BacktestConfig(
            rebalance_frequency=proto_cfg.frequency,
            target_horizon=proto_cfg.horizon,
        )
    artifact_provenance = execution_provenance(backtest_config)
    if (
        artifact_provenance["portfolio_config"]["rebalance_frequency"]
        != proto_cfg.frequency
        or artifact_provenance["portfolio_config"]["target_horizon"]
        != proto_cfg.horizon
    ):
        raise ValueError(
            "protocol frequency/horizon must match BacktestConfig execution "
            "provenance"
        )

    # P2-02: annotate rows that were produced before this schema (or by
    # callers outside the protocol) with their credibility tier, derived
    # from the recorded formula tokens / bare feature name.
    for row in rows:
        if row.get("data_tier") is None:
            row["data_tier"] = _data_tier_block(
                row.get("formula"), row.get("formula_text")
            )

    stitched = stitch_oos_series(rows)
    for trial in stitched:
        trial.update(stitched_metrics(trial))
        trial["data_tier"] = _data_tier_block(None, trial.get("formula_text"))
    top = top_trial(rows)
    if top is not None:
        top["data_tier"] = _data_tier_block(None, top.get("formula_text"))
    return _sanitize(
        {
            "protocol_version": _facade.PROTOCOL_VERSION,
            "data_tier_version": DATA_TIER_VERSION,
            "reward_version": _facade.REWARD_VERSION,
            # P6 §4.4: the research domain this campaign ran in and the
            # registry generation its defaults resolve from.
            "research_domain": proto_cfg.domain,
            "research_domain_version": RESEARCH_DOMAIN_VERSION,
            **artifact_provenance,
            "dataset_id": dataset_id,
            "frequency": proto_cfg.frequency,
            "horizon": proto_cfg.horizon,
            "tier": tier_name,
            "steps": tier.steps,
            "batch_size": tier.batch_size,
            "seeds": list(proto_cfg.seeds),
            "random_samples": proto_cfg.random_samples,
            "random_seed": proto_cfg.random_seed,
            "random_budget_matched": random_budget_matched,
            "random_budget": random_budget,
            "folds": [
                {"train_end": f.train_end, "test_end": f.test_end}
                for f in proto_cfg.folds
            ],
            "baseline_signals": list(proto_cfg.baseline_signals),
            "data_end_date": data_end_date,
            "universe_policy": universe_policy,
            "data_regime": _regime_payload(regime, proto_cfg),
            "ledger": ledger,
            "n_candidates": len(rows),
            "rows": rows,
            "aggregates": aggregate_results(rows),
            "stitched": {
                "n_trials": len(stitched),
                "trials": stitched,
            },
            "top_trial": top,
            "dsr": dsr_from_rows(rows, extra_trial_rows=extra_trial_rows),
            "max_t": max_t_from_rows(
                rows, n_perms=max_t_perms, extra_trial_rows=extra_trial_rows
            ),
            "dsr_extra_trials": len(stitch_oos_series(extra_trial_rows or [])),
        }
    )


def _run_recorded(ledger, fn, *, algorithm: str, candidate: str, seed=None,
                  fold_train_end=None, fold_test_end=None):
    """Run ``fn`` inside one ledger trial (T4-01).

    The trial opens as ``running`` before the work and closes as
    ``succeeded`` or ``failed`` on both paths, so a crashed trial is
    recorded, never silently dropped.  The returned row(s) carry the
    ``trial_id`` back into the artifact.  Without a ledger the call runs
    unrecorded (legacy callers).
    """

    if ledger is None:
        return fn()
    with ledger.trial(
        algorithm=algorithm,
        candidate=candidate,
        seed=seed,
        fold_train_end=fold_train_end,
        fold_test_end=fold_test_end,
    ) as trial_id:
        result = fn()
    rows = result if isinstance(result, list) else [result]
    for row in rows:
        if isinstance(row, dict):
            row["trial_id"] = trial_id
    return result
