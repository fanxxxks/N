"""Walk-forward evaluation protocol: the measurement system for formula quality.

The protocol exists so that "is this formula good" is answered by a fixed,
versioned measurement instead of by whichever number a run happened to print.
Contract:

* Folds are anchored to absolute dates (never proportions) so they stay
  stable as the database grows; each fold trains up to ``train_end`` and is
  scored out-of-sample on (``train_end``, ``test_end``].
* Multiple seeds train independent formulas per fold; every per-fold,
  per-seed row is kept raw in the result so later analysis can drill down.
* Adjudication metrics are **reward-independent**: they come from the full
  backtest engine (net returns, Sharpe/Sortino, drawdown, turnover) and the
  shared rank-IC code path.  ``val_reward`` is archived for provenance only
  and never used to rank candidates, so results remain comparable across
  ``ashare_model.reward.REWARD_VERSION`` generations.
* Signals are traded in their **learned direction**: a negative-IC signal is
  flipped (direction decided on the training window for baselines, on the
  validation tail for trained formulas) before the long-only top-n engine
  consumes it, so negative-IC factors are never mechanically traded
  backwards.
* A **random-search baseline** (uniform sampling over structurally valid
  formulas, scored with the same reward path) joins every fold: it
  separates "the RL search is ineffective" from "the reward is
  uninformative".
* Candidate multiplicity is corrected with Deflated Sharpe and a
  studentized max-t block bootstrap (single trial matrix for both).

:data:`PROTOCOL_VERSION` is bumped on every semantic change to this module's
measurement behavior and is recorded in protocol artifacts and experiment
archives, exactly like :data:`ashare_model.reward.REWARD_VERSION`.

v5 changes the scored signal for formula candidates: the VM now returns
every formula signal cross-sectionally z-scored per date (terminal
standardization), and the training-side reward that selects candidates
consumes the backtest engine's tradability masks.  Baselines (raw factors,
already winsorized + robustly standardized) and the benchmark are
unchanged.

v6 grows the formula search space: four cross-sectional operators
(CS_RANK / CS_ZSCORE / CS_DEMEAN / CS_NEUTRALIZE) and enumerated windows
(5/10/60 for MA, STD, TS_RANK, CORR, DOWNVOL; DELTA10/20) join the
vocabulary.  Scoring semantics are unchanged, but the candidate pool is
drawn from a strictly larger operator set, so artifacts record the new
version for comparability.

v7 makes candidate scoring/selection common to RL, random search and bare
factors, and resolves every fold through the inclusive-anchor t+2 contract.

v8 applies the PIT universe mask to every measurement: the backtest
engine selects only signal-date AND entry-date eligible stocks, the
default equal-weight benchmark averages only those cells (requiring a
valid target), the benchmark row reuses the engine's benchmark path, and
the protocol row IC is computed over signal-date eligible cells with the
unified rank-IC implementation.  Artifacts record the applied universe
policy fields (no data hash, no lineage).

v9 makes the JUMP operator causal (a trailing 60-session baseline instead
of the full-timeline mean/std look-ahead; see
``ashare_model.reward.REWARD_VERSION`` v8).  The scoring machinery is
unchanged, but every formula containing JUMP now produces a different
signal, so prior protocol results are not comparable for those candidates.

v10 aligns the rolling CAPM factors' market window to each stock's own
valid sessions (see ``ashare_model.reward.REWARD_VERSION`` v9): stocks
with suspension gaps no longer regress on market returns from sessions
they did not trade, so BETA_60/IVOL_60/RSQ_60 and every candidate built
on them changes value.

v11 isolates the training signal from the selection data (see
``ashare_model.reward.REWARD_VERSION`` v10): every candidate-scoring
caller (trainer, random search, baselines) scores its primary window on
the in-sample head that stops where the validation tail begins, and the
rows rename ``full_window_*`` to ``train_*`` accordingly.

v12 adds the tradable-IC diagnostic to every scored row:
``ic_mean_tradable`` / ``icir_tradable`` recompute the rank IC excluding
stocks the engine could not buy on the entry day (suspension / one-word
limit-up opens); the benchmark row carries neutral placeholders.  The
primary ``ic_mean`` / ``icir`` semantics are unchanged.

v13 binds every artifact to the immutable dataset manifest (T1-01):
``dataset_id`` (the content-addressed Merkle root over partition hashes,
see :mod:`ashare_data.manifest`) is recorded in the artifact, and trial
rows loaded from prior artifacts whose dataset_id differs from the current
database are rejected explicitly (:class:`DatasetIdMismatch`) instead of
being silently mixed into the DS/max-t correction pool.  Legacy artifacts
without a dataset_id are accepted as pre-T1-01 (recorded as ``null``).

v14 consumes the T1-02 no-signal weight contract from the shared engine:
under-filled days keep cash (no upward renormalization, so
``single_weight_cap`` is a hard ceiling on every OOS measurement),
dispersion-less cross-sections are never rebalanced, and selection ties
resolve by stable stock identifiers — so every protocol measurement is
invariant under stock-row permutation.

v15 consumes the T1-03 robust measurement contract: candidate scoring
applies the hard signal-quality gates (valid-IC days, effective stock
count vs ``ic_min_stocks``, coverage, activity, sign stability,
validation-window lower quartile), the reward's IC term is the
effective-n shrunk (HAC) ICIR, and complexity is billed from the AST with
a hard ``max_complexity`` ceiling — degenerate and pathological formulas
are rejected before any OOS row is produced.

v16 consumes the T1-04 alignment and selection contract: the reward
basket selects on signal-date AND entry-date PIT eligibility exactly like
the engine; every scored row records the portfolio objectives (active IR,
risk exposure, average turnover, capacity utilization) and the capacity
gate (``capacity_above_maximum``) rejects illiquid positions; best-
formula selection uses constrained/Pareto ranking on those objectives
with IC as the auxiliary tie-break, instead of the fragile scalar.

v17 makes the random-search baseline **budget-matched** (T1-05): by
default it receives the trained candidate's exact evaluation budget
(``tier.steps x tier.batch_size`` unique formula evaluations) per fold,
and artifacts record ``random_budget_matched`` / ``random_budget``, so
RL-vs-baseline comparisons are budget-fair.

v18 (T2-01) makes the **unique semantic formula evaluation** the budget
unit of every searcher: formulas are canonicalized (ADD/MUL child
ordering, double-NEG/identity/invalid-op elimination, structural hashing),
degenerate constant-producing formulas are rejected pre-evaluation, and
numerically equivalent formulas (same rank fingerprint on the fixed
calibration slice) are scored once.  Duplicates never consume budget,
so the trainer, the random baseline and every later searcher are billed
identically.  Artifacts record ``unique_semantic_evals`` and the semantic
cache version.

v19 (T2-03) completes the baseline ladder: strongly-typed GP (DEAP) and
TPE (Optuna) join the random-search baseline with the same matched
budget per fold (``gp_enabled`` / ``tpe_enabled`` config, both on by
default), all three rows run on the shared semantic-budget evaluator,
and every trained row records its actual ``unique_semantic_evals``.
P4's paired admission experiment uses these budget semantics but gives
every arm the same independent pair seed and requested budget; fixed
baseline seeds from the historical T2 artifact are not admissible.

v20 (T4-01) re-establishes the valid-experiment protocol.  The 2021-2026
history has been viewed repeatedly, so it is **development/validation
data** (see :mod:`ashare_model.regime`); the protocol is now a **nested
walk-forward**: the inner loop tunes formula / reward / hyperparameters
inside each fold's training window, the outer loop performs exactly one
algorithm evaluation per (fold, seed), and adjudication consumes the
**stitched** outer OOS returns — one trial = one (candidate, seed)
concatenated series (``stitch_oos_series``), and Sharpe / Deflated
Sharpe / max-t / top-trial all operate on that stitched matrix
(``stitched`` artifact block).  Every trial is automatically recorded in
the append-only experiment ledger (:mod:`ashare_model.ledger`) — failed
and crashed trials included; a run with never-closed trials is tainted,
never silently clean.  Protocol runs refuse folds that touch a strictly
locked holdout slice (:class:`ashare_model.regime.HoldoutViolation`),
and the artifact records the data regime in force.  Prior artifacts
remain readable: their raw rows stitch under the same rule when loaded
with ``--trials``, and the per-fold row-level ``aggregates`` are kept
for drill-down.

v21 (P2) records the free-data credibility tier of every measured
formula: each row, stitched trial and the top trial carry a ``data_tier``
block (``data_tier_version`` / ``max_tier`` / ``tiers_used``) resolved
from the formula's features via :mod:`ashare_model.data_tier`
(``docs/p2_data_tier_contract.md``).  Formula semantics are unchanged;
the artifact schema gains provenance fields, and the promotion gates use
them (Tier A default, Tier B separate comparison, Tier C never).

v22 (P3) makes ``frequency`` / ``horizon`` executable protocol fields.
Every fold and validation subwindow retains its slice of the global
rebalance calendar; research IC/quality consumes the sparse causal target
``open[t+1+horizon] / open[t+1] - 1``, while the portfolio curve separately
consumes adjacent-open daily returns.  Protocol artifacts record execution
spec v2, the portfolio-constructor version and the complete resolved
portfolio configuration; pre-v22 artifacts remain readable history but are
not current promotion evidence.

v23 (P4) makes GP/TPE/Random/RL return one versioned ``SearchResult`` with
truthful requested/consumed budgets, termination/stagnation and
per-evaluation best-so-far curves.  This changes comparison and artifact
semantics, not the candidate reward itself.

v24 (P6) adds the research-domain dimension
(``docs/p6_research_domain_contract.md``): a run declares one of
``short_price_volume`` / ``medium_cross_section`` / ``slow_fundamental``
(or the reserved compatible semantic ``unified``).  In domain mode the
out-of-domain factor rows are neutral (zeroed), every searcher samples
only the domain's feature tokens, baselines must lie inside the domain,
the (frequency, horizon) execution point must be a legal point of the
domain, and the window id carries the domain.  Artifacts record
``research_domain`` and ``research_domain_version``; rewards from
different domains are never comparable.

v25 (P7-E) narrows the sampled candidate pool with semantic-type
legality (``docs/p7_semantic_types_contract.md``): the action mask and
the typed GP primitive set only admit formulas whose operator
applications satisfy the registered semantic signatures.  Measurement
semantics (engine, reward, corrections, gates) are unchanged; results
from pre-v25 candidate pools are not matched comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    ModelConfig,
    ProtocolConfig,
    RewardConfig,
    TierConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_reward_config,
    validate_baseline_signals,
)
from ashare_data.gates import ProductionGateRunner
from ashare_data.manifest import DatasetIdMismatch, check_dataset_id
from ashare_logging import export_log_txt, setup_run_logging

from .artifact_schemas import ProtocolResultArtifact, apply_schema_matrix
# CandidateScorer is re-exported (not used by facade code): the pre-split
# monkeypatch surface patches ``evaluation.CandidateScorer`` (see
# tests/test_eval_module_split.py FACADE_SURFACE).
from .candidates import CandidateScorer
from .data_loader import AshareDataLoader
from .data_tier import DATA_TIER_VERSION
from .eval_artifacts import (
    _run_recorded,
    build_result,
    universe_policy_payload,
)
from .eval_corrections import (
    deflated_sharpe,
    dsr_from_rows,
    expected_max_sr,
    max_t_from_rows,
    norm_cdf,
    norm_ppf,
    psr,
    selfcheck_rows,
)
from .eval_folds import (
    Fold,
    FoldData,
    epoch_slice,
    resolve_folds,
    search_window_id,
)
from .eval_metrics import (
    METRIC_KEYS,
    _tradable_ic_mask,
    aggregate_results,
    benchmark_row,
    evaluate_formula,
    evaluate_signal,
    stitch_oos_series,
    stitched_metrics,
    top_trial,
)
from .eval_search import (
    _build_trainer,
    baseline_candidates,
    run_fold,
    run_gp_search,
    run_random_search,
    run_tpe_search,
)
from .ledger import ExperimentLedger
from .regime import RegimeRegistry
from .research_domain import (
    RESEARCH_DOMAIN_VERSION,
    UNIFIED_DOMAIN_ID,
    feature_token_ids,
    restrict_tensor,
    resolve_domain,
)
from .reward import REWARD_VERSION
from .search_contract import SearchResult
from .vocab import FEATURE_NAMES

PROTOCOL_VERSION = "25"


def run_protocol(
    loader: AshareDataLoader,
    data_config: DataConfig,
    model_config: ModelConfig,
    backtest_config: BacktestConfig,
    reward_config: RewardConfig | None,
    proto_cfg: ProtocolConfig,
    tier_name: str,
    fold_indices: list[int] | None = None,
    seeds: list[int] | None = None,
    extra_trial_rows: list[dict] | None = None,
    max_t_perms: int = 5000,
    ledger=None,
    regime=None,
) -> dict:
    """Run the full protocol: baselines + one trained candidate per
    (fold, seed), all scored by the shared engine path.

    T4-01 wiring: ``regime`` is checked before the first trial (a fold
    touching a strictly locked holdout slice raises
    :class:`~ashare_model.regime.HoldoutViolation`), and ``ledger``
    records every trial automatically.  The benchmark row is a reference
    curve, not a trial, and is never recorded.
    """

    if tier_name not in ("screening", "confirmation"):
        raise ValueError(f"unknown tier: {tier_name}")
    tier = getattr(proto_cfg, tier_name)
    validate_baseline_signals(proto_cfg.baseline_signals, FEATURE_NAMES)

    # P6: research-domain semantics.  In domain mode the execution point
    # must be a legal point of the domain, baselines must stay inside the
    # domain, the out-of-domain factor rows become neutral, and the
    # searchers sample only the domain's feature tokens.  ``unified``
    # (the default) is byte-identical to pre-P6 behavior.
    feature_ids: list[int] | None = None
    domain = None
    if proto_cfg.domain != UNIFIED_DOMAIN_ID:
        domain = resolve_domain(proto_cfg.domain)
        if not domain.is_legal_execution(
            proto_cfg.frequency, proto_cfg.horizon
        ):
            raise ValueError(
                f"protocol (frequency={proto_cfg.frequency!r}, "
                f"horizon={proto_cfg.horizon}) is not a legal execution "
                f"point of domain {domain.id!r} "
                "(docs/p6_research_domain_contract.md §1.2)"
            )
        out_of_domain = [
            name
            for name in proto_cfg.baseline_signals
            if name not in domain.features
        ]
        if out_of_domain:
            raise ValueError(
                f"baseline signals {out_of_domain} are outside domain "
                f"{domain.id!r} (docs/p6_research_domain_contract.md §3.2)"
            )
        feature_ids = feature_token_ids(proto_cfg.domain)
        if loader.factor_tensor is None:
            raise ValueError(
                "the loader must be loaded before a domain protocol run"
            )
        loader.factor_tensor = torch.tensor(
            restrict_tensor(loader.factor_tensor.numpy(), proto_cfg.domain),
            dtype=torch.float32,
        )
        logger.info(
            "protocol.domain domain={} features={} frequency={} horizon={} "
            "turnover_budget={} cost_weight={}",
            domain.id,
            len(domain.features),
            proto_cfg.frequency,
            proto_cfg.horizon,
            backtest_config.turnover_budget,
            reward_config.cost_weight if reward_config is not None else None,
        )
    if fold_indices is not None:
        unknown = [i for i in fold_indices if not 0 <= i < len(proto_cfg.folds)]
        if unknown:
            raise ValueError(
                f"fold indices out of range: {unknown} "
                f"(config has {len(proto_cfg.folds)} folds)"
            )
        fold_cfgs = [proto_cfg.folds[i] for i in fold_indices]
    else:
        fold_cfgs = proto_cfg.folds
    seeds = proto_cfg.seeds if seeds is None else list(seeds)
    if regime is not None:
        regime.assert_folds_clear(fold_cfgs, dataset_id=loader.dataset_id)

    folds = resolve_folds(
        fold_cfgs,
        loader.dates,
        frequency=proto_cfg.frequency,
        horizon=proto_cfg.horizon,
    )
    rows: list[dict] = []
    for fold in folds:
        rows.append(benchmark_row(loader, fold))
        rows.extend(
            _run_recorded(
                ledger,
                lambda: baseline_candidates(
                    loader,
                    proto_cfg,
                    fold,
                    backtest_config,
                    model_config,
                    reward_config,
                ),
                algorithm="baseline",
                candidate="baseline",
                fold_train_end=fold.train_end,
                fold_test_end=fold.test_end,
            )
        )
        if proto_cfg.random_samples > 0:
            # T1-05: the baseline is budget-matched to the trained
            # candidate (steps x batch_size unique evaluations) unless
            # explicitly disabled.
            random_budget = (
                tier.steps * tier.batch_size
                if proto_cfg.random_match_budget
                else None
            )
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} random-search "
                f"baseline budget={random_budget or proto_cfg.random_samples} "
                f"seed={proto_cfg.random_seed}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_random_search(
                        loader,
                        model_config,
                        backtest_config,
                        reward_config,
                        fold,
                        proto_cfg.random_samples,
                        proto_cfg.random_seed,
                        budget=random_budget,
                        feature_ids=feature_ids,
                        domain_id=proto_cfg.domain,
                    ),
                    algorithm="random_search",
                    candidate="random_search",
                    seed=proto_cfg.random_seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
        # T2-03 baseline ladder: GP (DEAP) and TPE (Optuna) join the
        # random baseline with the same matched budget per fold.
        matched_budget = (
            tier.steps * tier.batch_size
            if proto_cfg.random_match_budget
            else proto_cfg.random_samples
        )
        if proto_cfg.gp_enabled and matched_budget > 0:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} gp baseline "
                f"budget={matched_budget} seed={proto_cfg.gp_seed}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_gp_search(
                        loader,
                        model_config,
                        backtest_config,
                        reward_config,
                        fold,
                        seed=proto_cfg.gp_seed,
                        budget=matched_budget,
                        feature_ids=feature_ids,
                        domain_id=proto_cfg.domain,
                    ),
                    algorithm="gp_search",
                    candidate="gp_search",
                    seed=proto_cfg.gp_seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
        if proto_cfg.tpe_enabled and matched_budget > 0:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} tpe baseline "
                f"budget={matched_budget} seed={proto_cfg.tpe_seed}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_tpe_search(
                        loader,
                        model_config,
                        backtest_config,
                        reward_config,
                        fold,
                        seed=proto_cfg.tpe_seed,
                        budget=matched_budget,
                        feature_ids=feature_ids,
                        domain_id=proto_cfg.domain,
                    ),
                    algorithm="tpe_search",
                    candidate="tpe_search",
                    seed=proto_cfg.tpe_seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
        for seed in seeds:
            logger.info(
                f"fold {fold.train_end} -> {fold.test_end} seed={seed} "
                f"tier={tier_name} steps={tier.steps} batch={tier.batch_size}"
            )
            rows.append(
                _run_recorded(
                    ledger,
                    lambda: run_fold(
                        loader,
                        data_config,
                        model_config,
                        backtest_config,
                        reward_config,
                        tier,
                        fold,
                        seed,
                        domain_id=proto_cfg.domain,
                        feature_ids=feature_ids,
                    ),
                    algorithm="trained",
                    candidate="trained",
                    seed=seed,
                    fold_train_end=fold.train_end,
                    fold_test_end=fold.test_end,
                )
            )
    ledger_payload = None
    if ledger is not None:
        open_trials = ledger.finalize()
        ledger_payload = {
            "path": str(ledger.path),
            "run_id": ledger.run_id,
            "tainted": bool(open_trials),
            "n_trials": len(list(ledger.trials_for())),
        }
    return build_result(
        proto_cfg,
        tier_name,
        tier,
        rows,
        data_end_date=loader.dates[-1],
        extra_trial_rows=extra_trial_rows,
        max_t_perms=max_t_perms,
        universe_policy=universe_policy_payload(loader),
        dataset_id=loader.dataset_id,
        random_budget_matched=(
            proto_cfg.random_match_budget and proto_cfg.random_samples > 0
        ),
        random_budget=(
            tier.steps * tier.batch_size
            if proto_cfg.random_match_budget and proto_cfg.random_samples > 0
            else proto_cfg.random_samples
        ),
        ledger=ledger_payload,
        regime=regime,
        backtest_config=backtest_config,
    )


def load_trial_rows(
    paths: str | None, expected_dataset_id: str | None = None
) -> list[dict]:
    """Trial rows from prior protocol artifacts (comma-separated paths).

    ``expected_dataset_id`` binds the loaded rows to the current database:
    an artifact whose recorded dataset_id differs is rejected with
    :class:`DatasetIdMismatch` (explicit rejection instead of silently
    mixing measurements from different data).  Artifacts without a
    dataset_id are pre-T1-01 legacy and pass as-is.
    """

    rows: list[dict] = []
    if not paths:
        return rows
    for path in paths.split(","):
        path = path.strip()
        if not path:
            continue
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # P7-C §4: unknown/future schema versions are hard-rejected;
        # current payloads validate; legacy flows to the pre-contract path.
        apply_schema_matrix(
            payload, artifact="protocol", model=ProtocolResultArtifact
        )
        check_dataset_id(payload.get("dataset_id"), expected_dataset_id)
        rows.extend(payload.get("rows", []))
    return rows


def main(argv=None) -> int:
    setup_run_logging(run_name="evaluation")
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation protocol"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--tier", choices=["screening", "confirmation"], default="screening"
    )
    parser.add_argument("--output", default="data/protocol_result.json")
    parser.add_argument(
        "--folds", type=int, default=None, help="run only the first N folds"
    )
    parser.add_argument(
        "--seeds", default=None, help="comma-separated seed override"
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="score pure-noise placeholder candidates and report whether "
        "the DS/max-t corrections stay insignificant (dry-run acceptance)",
    )
    parser.add_argument(
        "--trials",
        default=None,
        help="comma-separated prior protocol_result.json paths whose trial "
        "rows join the DS/max-t multiplicity correction",
    )
    parser.add_argument(
        "--ledger",
        default="data/experiment_ledger.jsonl",
        help="append-only trial ledger path (T4-01; every trial is recorded "
        "automatically, failed trials included)",
    )
    parser.add_argument(
        "--regime",
        default="data/holdout_registry.json",
        help="data-regime registry path (T4-01; folds touching a locked "
        "holdout slice are refused before the first trial)",
    )
    parser.add_argument(
        "--no-random-search",
        action="store_true",
        help="skip the random-search baseline (protocol.random_samples=0)",
    )
    parser.add_argument(
        "--max-t-perms", type=int, default=5000, help="max-t permutation count"
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
    )
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(data_config, min_eligible=args.min_eligible).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        proto_cfg = make_protocol_config(raw)
        if args.no_random_search:
            proto_cfg.random_samples = 0

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        extra_trials = load_trial_rows(args.trials, loader.dataset_id)
        ledger = ExperimentLedger(root / args.ledger)
        regime = RegimeRegistry(root / args.regime)

        if args.selfcheck:
            # Noise measurement is a measurement too: it must not touch a
            # locked holdout slice.
            regime.assert_folds_clear(proto_cfg.folds, dataset_id=loader.dataset_id)
            rows: list[dict] = []
            for fold_cfg in proto_cfg.folds:
                rows.append(
                    _run_recorded(
                        ledger,
                        lambda fold_cfg=fold_cfg: selfcheck_rows(
                            loader,
                            ProtocolConfig(folds=[fold_cfg]),
                            backtest_config,
                        )[0],
                        algorithm="selfcheck",
                        candidate="selfcheck:noise",
                        fold_train_end=fold_cfg.train_end,
                        fold_test_end=fold_cfg.test_end,
                    )
                )
            open_trials = ledger.finalize()
            ledger_payload = {
                "path": str(ledger.path),
                "run_id": ledger.run_id,
                "tainted": bool(open_trials),
                "n_trials": len(list(ledger.trials_for())),
            }
            result = build_result(
                proto_cfg,
                "selfcheck",
                TierConfig(0, 0),
                rows,
                data_end_date=loader.dates[-1],
                extra_trial_rows=extra_trials,
                max_t_perms=args.max_t_perms,
                universe_policy=universe_policy_payload(loader),
                dataset_id=loader.dataset_id,
                ledger=ledger_payload,
                regime=regime,
                backtest_config=backtest_config,
            )
        else:
            fold_indices = list(range(args.folds)) if args.folds else None
            seeds = (
                [int(s) for s in args.seeds.split(",")] if args.seeds else None
            )
            result = run_protocol(
                loader,
                data_config,
                model_config,
                backtest_config,
                reward_config,
                proto_cfg,
                args.tier,
                fold_indices=fold_indices,
                seeds=seeds,
                extra_trial_rows=extra_trials,
                max_t_perms=args.max_t_perms,
                ledger=ledger,
                regime=regime,
            )
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Protocol result written to {out_path}")
        print(json.dumps(result["aggregates"], ensure_ascii=False, indent=2))
        stitched = result.get("stitched") or {}
        print(
            f"stitched trials: {stitched.get('n_trials', 0)} "
            f"(one per (candidate, seed), across all folds)"
        )
        if result["top_trial"]:
            top = result["top_trial"]
            print(
                f"top trial: sharpe={top['sharpe']:.3f} "
                f"candidate={top['candidate']} fold_test_end={top['fold_test_end']} "
                f"seed={top['seed']}"
            )
        if result["dsr"]:
            dsr = result["dsr"]
            print(
                f"dsr: {dsr['dsr']:.3f} (n_trials={dsr['n_trials']}, "
                f"best={dsr['best_candidate']} sr={dsr['sr_best']:.3f})"
            )
        if result["max_t"]:
            mt = result["max_t"]
            print(
                f"max_t: observed={mt['observed_max_t']:.3f} "
                f"p={mt['p_value']:.4f} significant_95={mt['significant_95']}"
            )
        if args.selfcheck and result["dsr"] and result["max_t"]:
            passed = (
                result["dsr"]["dsr"] < 0.95
                and not result["max_t"]["significant_95"]
            )
            print(f"selfcheck passed={passed}")
    finally:
        export_log_txt(run_name="evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
