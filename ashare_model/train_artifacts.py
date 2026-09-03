"""Training-artifact persistence for the trainer (P7 Phase B2).

Extracted from ``train.py`` by reason-to-change (IP-07b, mirroring the
evaluation P7 split): this module owns the *artifact boundary* — the
strategy payload assembly, lifecycle RunSpec resolution, RunStore-bound
formal writing, the elite-archive mirror and the RL policy checkpoint.
It changes when the artifact schema or persistence contract changes —
not when the RL loop, search orchestration or window semantics change.

The former lazy imports become top-level here: none of the artifact-chain
modules (``artifact_writer`` / ``run_store`` / ``runspec`` /
``artifact_schemas`` / ``identity`` / ``elite_archive``) imports
``train``, so the historical ``evaluation``-cycle laziness has no reason
to survive the move (IP-07a removed the cycle's root).

The trainer keeps a thin ``_write_artifact`` delegation so the method
stays overridable/patchable on the facade class.
"""

from __future__ import annotations

import torch
from loguru import logger

from ashare_portfolio.execution_spec import execution_provenance

from .alphagpt import MODEL_VERSION
from .artifact_schemas import ArtifactSchemaError, StrategyArtifact
from .artifact_writer import write_boundary_artifact
from .data_tier import formula_data_tier_report
from .elite_archive import write_elite_archive
from .identity import candidate_id
from .research_domain import RESEARCH_DOMAIN_VERSION
from .reward import REWARD_VERSION
from .run_store import RunStore
from .runspec import resolve_runtime_runspec
from .semantic_cache import SEMANTIC_CACHE_VERSION
from .time_contract import TrainingTimeContract
from .versions import PROTOCOL_VERSION
from .vocab import GRAMMAR_VERSION


def write_trainer_artifact(
    trainer,
    *,
    contract: TrainingTimeContract,
    vm_device: torch.device,
    searcher: str = "rl",
    seed: int,
    requested_budget: int,
) -> list[int] | None:
    """Write the standard training artifact (selection + strategy JSON +
    the policy checkpoint for RL runs) for the trainer's current selection.

    P8-05: the strategy artifact is a lifecycle-bound boundary artifact —
    the run resolves its frozen RunSpec, opens a RunStore run and
    persists through :func:`write_boundary_artifact` (content-addressed,
    fail-closed identity, atomic convenience mirror).  The strategy
    JSON stays at its historical path as the display mirror.
    """

    selected = trainer.selection_result.selected
    if selected is None:
        return None

    output = _artifact_output(
        trainer, searcher=searcher, vm_device=vm_device
    )
    return _persist_strategy_artifact(
        trainer,
        output,
        searcher=searcher,
        seed=seed,
        requested_budget=requested_budget,
    )


def _artifact_output(
    trainer, *, searcher: str, vm_device: torch.device
) -> dict:
    """Assemble the strategy payload dict (verbatim from the historical
    ``_write_artifact`` body)."""

    selected = trainer.selection_result.selected
    score_payload = selected.to_dict()
    score_payload.pop("tokens", None)
    # P8-05: the searcher-internal candidate label is a diagnostic only;
    # the lifecycle candidate identity (identity.candidate_id over
    # spec_id + tokens + direction) is stamped by the formal writer.
    searcher_candidate_label = score_payload.pop("candidate_id", None)
    output = {
        "formula": trainer.best_tokens,
        **score_payload,
        "searcher_candidate_label": searcher_candidate_label,
        "history": trainer.history,
        "searcher": searcher,
        # Reproducibility provenance: the policy stays on CPU, so
        # (init_seed, seed) reproduce the same sampled formulas on any
        # machine; ``device`` records where the VM executed.
        "init_seed": trainer.init_seed,
        "device": str(vm_device),
        "model_version": MODEL_VERSION,
        # Vocabulary provenance: the formula is always remapped by name
        # on load, so later vocabulary additions cannot silently
        # reinterpret these token ids.
        "feature_names": list(trainer.vocab.feature_names),
        "operator_names": list(trainer.vocab.operator_names),
        "feature_version": trainer.vocab.feature_version,
        "grammar_version": GRAMMAR_VERSION,
        # Reward provenance: reward values are only comparable within
        # the same scoring implementation generation.
        "reward_version": REWARD_VERSION,
        # T2-01 provenance: the evaluation-budget ledger generation and
        # the unique semantic evaluations this run actually performed.
        "protocol_version": PROTOCOL_VERSION,
        # P6 provenance: the research domain this strategy was searched
        # in (reserved compatible semantic "unified" by default) and
        # the registry generation its defaults resolve from.
        "research_domain": trainer.domain_id,
        "research_domain_version": RESEARCH_DOMAIN_VERSION,
        **execution_provenance(trainer.backtest_config),
        "semantic_cache_version": SEMANTIC_CACHE_VERSION,
        "unique_semantic_evals": trainer.semantic_cache.budget_used,
        "semantic_cache_stats": trainer.semantic_cache.stats(),
        "search_contract_version": (
            trainer.search_result.contract_version
            if trainer.search_result
            else None
        ),
        "search_result": (
            trainer.search_result.to_dict() if trainer.search_result else None
        ),
        # P2-02: the strategy formula traces back to the credibility
        # tiers of its features (``None`` when nothing is traceable).
        "data_tier": formula_data_tier_report(tokens=trainer.best_tokens),
        # Data provenance: the immutable dataset manifest this formula
        # was selected on (None for pre-T1-01 databases).
        "dataset_id": trainer.loader.dataset_id,
    }
    if searcher == "rl":
        output.update(
            {
                "rl_initialization": trainer.rl_initialization,
                "imitation": (
                    trainer.imitation_result.to_dict()
                    if trainer.imitation_result is not None
                    else None
                ),
                "experimental": trainer.rl_initialization == "random",
            }
        )
    return output


def _persist_strategy_artifact(
    trainer,
    output: dict,
    *,
    searcher: str,
    seed: int,
    requested_budget: int,
) -> list[int] | None:
    """Resolve the frozen RunSpec, persist through the RunStore-bound
    formal writer and mirror the RL policy checkpoint (verbatim from the
    historical ``_write_artifact`` body)."""

    out_path = trainer.data_config.data_dir / "best_ashare_strategy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # P8-05: resolve the frozen RunSpec for this run and persist through
    # the RunStore-bound formal writer.  A formal strategy artifact
    # requires a resolved dataset identity (T1-01 manifest); a legacy
    # database without one is refused here, fail-closed.
    if not trainer.loader.dataset_id:
        raise ArtifactSchemaError(
            "formal strategy artifacts require a resolved dataset_id "
            "(dataset manifest); migrate the database with "
            "`python -m ashare_data.manifest` before training"
        )
    spec = resolve_runtime_runspec(
        dataset_id=trainer.loader.dataset_id,
        data_cutoff=trainer.loader.dates[-1],
        data_config=trainer.data_config,
        backtest_config=trainer.backtest_config,
        requested_budget=int(requested_budget),
        # Training evaluates a single validation tail; the campaign's
        # fold plan is carried by the protocol run's own spec.
        n_folds=1,
        research_domain=trainer.domain_id,
        seeds=tuple(sorted({int(trainer.init_seed), int(seed)})),
        searcher=searcher,
        max_formula_len=int(trainer.model_config.max_formula_len),
        # The clean-tree SPEC_LOCKED evidence gate activates with the
        # lifecycle stages (P8-06+); the spec still records the exact
        # git commit and dependency-lock hash.
        require_clean_tree=False,
    )
    cand = candidate_id(spec.spec_id, trainer.best_tokens, output["direction"])
    if (
        trainer.search_result is not None
        and trainer.search_result.elite_archive is not None
    ):
        archive_path = write_elite_archive(
            trainer.data_config.data_dir / "search_elite_archive.json",
            trainer.search_result.elite_archive,
        )
        logger.info("search.elite_archive path={}", archive_path)
    store = RunStore(trainer.data_config.data_dir)
    with store.open_run(spec) as handle:
        write_boundary_artifact(
            handle,
            artifact_type="strategy",
            model_cls=StrategyArtifact,
            payload=output,
            candidate_id=cand,
            convenience_path=out_path,
        )
    if searcher == "rl":
        # The checkpoint is the RL policy only: the non-RL searchers
        # are not models, so no .pt is written for them.
        torch.save(
            trainer.model.state_dict(),
            trainer.data_config.data_dir / "ashare_model.pt",
        )
    logger.success(
        f"Search complete (searcher={searcher}); "
        f"best formula saved to {out_path}"
    )
    return trainer.best_tokens


class ArtifactPersistenceMixin:
    """Artifact-boundary seam of ``AshareTrainer`` (B2/B5, IP-07b): the
    facade method delegates to :func:`write_trainer_artifact` so callers
    and class-attribute patches keep one stable surface."""

    def _write_artifact(
        self,
        *,
        contract: TrainingTimeContract,
        vm_device: torch.device,
        searcher: str = "rl",
        seed: int,
        requested_budget: int,
    ) -> list[int] | None:
        """Delegate to :func:`write_trainer_artifact`."""

        return write_trainer_artifact(
            self,
            contract=contract,
            vm_device=vm_device,
            searcher=searcher,
            seed=seed,
            requested_budget=requested_budget,
        )
