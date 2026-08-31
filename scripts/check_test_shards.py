"""CI test-shard authority (PR4/D1 of the 2026-08-31 test-runtime plan).

Single source of truth for how the CI test job splits ``tests/test_*.py``
across matrix legs.  Two properties are fail-closed:

* **union == full set**: every test file is assigned to exactly one shard;
  a newly added file that is not assigned breaks ``--check`` (and CI)
  instead of silently never running;
* **no overlap**: a file moved between shards must be removed from the
  old shard list, otherwise ``--check`` fails.

``--check`` validates both properties and exits 1 on any violation.
``--emit <shard>`` prints the shard's ``tests/...`` paths so the CI job
can pass them to pytest verbatim.  The CI matrix shard names and both
invocations are contract-checked by ``tests/test_ci_sharding.py``.

Shard membership is an explicit list (not globs): rebalancing is a
reviewed, diffable change, and the guard turns any drift into a red CI
run rather than a silent coverage hole.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SHARDS: dict[str, tuple[str, ...]] = {
    # ashare_data domain + shared io/logging/meta + doc-sync guards.
    "shard-data": (
        "test_akshare_client.py",
        "test_capital_flow.py",
        "test_ci_sharding.py",
        "test_config.py",
        "test_data_loader.py",
        "test_data_tier.py",
        "test_db.py",
        "test_fundamental_scope.py",
        "test_fundamentals.py",
        "test_io_utils.py",
        "test_lock_files.py",
        "test_logging.py",
        "test_logging_parallel.py",
        "test_manifest.py",
        "test_p8_lifecycle_contract_doc.py",
        "test_pit_import.py",
        "test_processor.py",
        "test_registry_docs.py",
        "test_sync.py",
        "test_tier_reports.py",
        "test_time_contract.py",
        "test_universe.py",
        "test_validation_docs.py",
    ),
    # Heavy training/evaluation/search core (holds the two largest
    # protocol runs; rebalance together with shard-model-2).
    "shard-model-1": (
        "test_alphagpt.py",
        "test_baseline_harness.py",
        "test_core.py",
        "test_eval_module_split.py",
        "test_evaluation.py",
        "test_factors.py",
        "test_grammar.py",
        "test_gp_search.py",
        "test_no_signal_semantics.py",
        "test_research_domain.py",
        "test_searcher_bench.py",
        "test_semantic_cache.py",
        "test_semantic_sampling.py",
        "test_signal_quality.py",
        "test_tpe_search.py",
        "test_train_module_split.py",
        "test_vocab.py",
    ),
    # Artifacts/protocol/admission/identity/registries + training split.
    "shard-model-2": (
        "test_admission.py",
        "test_artifact_schemas.py",
        "test_artifact_versions.py",
        "test_artifact_writer.py",
        "test_candidates.py",
        "test_completion_gates.py",
        "test_cost_matrix.py",
        "test_diagnostics.py",
        "test_experiment_tracking.py",
        "test_feature_metadata.py",
        "test_feature_registry.py",
        "test_gates.py",
        "test_identity.py",
        "test_ir.py",
        "test_ir_canonicalization.py",
        "test_ledger.py",
        "test_operator_registry.py",
        "test_p4_admission_diagnostics.py",
        "test_p4_elite_imitation.py",
        "test_p4_search_contract.py",
        "test_promotion.py",
        "test_regime.py",
        "test_research_doctor.py",
        "test_reward.py",
        "test_run_store.py",
        "test_runspec.py",
        "test_schemas.py",
        "test_stitched_oos.py",
        "test_train.py",
    ),
    # Portfolio/execution/trading/sim/web/archive/golden parity.
    "shard-portfolio-trading": (
        "test_analyze_sim.py",
        "test_archive_run.py",
        "test_backtest.py",
        "test_bare_factor_backtest.py",
        "test_dashboard.py",
        "test_execution.py",
        "test_golden_parity.py",
        "test_jobmanager.py",
        "test_ops.py",
        "test_p3_measurement.py",
        "test_p3_portfolio_parity.py",
        "test_portfolio_constructor.py",
        "test_portfolio_objectives.py",
        "test_portfolio_optimizer.py",
        "test_rebalance_policy.py",
        "test_run_sim.py",
        "test_trading.py",
        "test_vm.py",
        "test_webapi.py",
    ),
}


def discover_test_files() -> set[str]:
    """File names of every ``tests/test_*.py`` in the repository."""

    tests_dir = ROOT / "tests"
    return {p.name for p in tests_dir.glob("test_*.py")}


def validate(shards: dict[str, tuple[str, ...]], files: set[str]) -> list[str]:
    """Return human-readable errors for omission/overlap/unknown entries."""

    errors: list[str] = []
    seen: dict[str, str] = {}
    for shard, members in shards.items():
        for name in members:
            if name not in files:
                errors.append(
                    f"shard '{shard}' lists '{name}' but no such tests/test_*.py exists"
                )
                continue
            if name in seen:
                errors.append(
                    f"'{name}' is assigned to both '{seen[name]}' and '{shard}'"
                )
            seen[name] = shard
    for name in sorted(files - set(seen)):
        errors.append(f"'{name}' is not assigned to any shard (would never run in CI)")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail closed on any shard drift")
    parser.add_argument("--emit", metavar="SHARD", help="print the shard's tests/... paths")
    args = parser.parse_args(argv)

    if args.emit:
        members = SHARDS.get(args.emit)
        if members is None:
            print(
                f"unknown shard '{args.emit}'; known shards: {', '.join(sorted(SHARDS))}",
                file=sys.stderr,
            )
            return 1
        print(" ".join(f"tests/{name}" for name in members))
        return 0

    files = discover_test_files()
    errors = validate(SHARDS, files)
    if errors:
        for error in errors:
            print(f"shard drift: {error}", file=sys.stderr)
        return 1
    print(
        f"shard union == full set: {sum(len(v) for v in SHARDS.values())} files "
        f"across {len(SHARDS)} shards cover all {len(files)} test files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
