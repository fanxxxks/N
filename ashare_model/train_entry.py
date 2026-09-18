"""CLI entry for the RL trainer (A03, p27 §四 A03 训练子提交).

Extracted from the ``train.py`` facade by reason-to-change: the facade
line budget (``tests/test_train_module_split.py``, IP-07b) cannot absorb
the entry-level lifecycle wiring, and the entry identity establishment is
its own responsibility — freeze the plan, attach the preregistered run,
enforce the lifecycle gate, and only then pay for the expensive load
(p27 I6).  The facade keeps ``main`` on its import surface by delegation.

The trainer class is injected (no import of ``ashare_model.train`` — the
train chain stays acyclic); its registered monkeypatch surface
(``batched_basket_rewards`` / ``score_chunk_size`` / ``logger``) keeps
working through the facade's late binding because it is read from the
facade namespace at construction/loop time, not at entry time.
"""

from __future__ import annotations

import argparse

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_reward_config,
)
from ashare_data.gates import (
    ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
    ProductionGateRunner,
)
from ashare_logging import export_log_txt, setup_run_logging

from .data_loader import AshareDataLoader
from .lifecycle import LifecycleError, RunContext
from .run_store import RunStore, RunStoreError
from .runspec import resolve_runtime_runspec


def build_parser() -> argparse.ArgumentParser:
    """The train CLI surface (A03: --spec-id/--run-id/--lifecycle-mode)."""

    parser = argparse.ArgumentParser(description="Train A-share AlphaGPT")
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="VM compute device; policy and sampling always run on CPU "
        "(default: CUDA when available, else CPU)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
    )
    # A03 (p27 I2/I5): the formal entry requires the preregistered run
    # identity; the explicit non-formal modes are the only identity-free
    # paths (工程探针 / 历史审计).
    parser.add_argument(
        "--spec-id",
        default=None,
        help="preregistered RunSpec id of the run to attach (formal mode)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id of the preregistered run to attach (formal mode)",
    )
    parser.add_argument(
        "--lifecycle-mode",
        choices=["formal", "engineering", "audit"],
        default="formal",
        help="formal (default) attaches the preregistered run and enforces "
        "the lifecycle gate; engineering/audit are explicit non-formal "
        "declarations (p27 §3/I3)",
    )
    return parser


def establish_formal_run_context(
    *,
    data_config,
    model_config,
    backtest_config,
    spec_id: str,
    run_id: str,
    steps: int,
    batch_size: int,
    min_eligible: int | None = None,
):
    """Freeze the plan and attach the preregistered run BEFORE the heavy
    load (p27 I6): composite identity confirm → RunSpec freeze → run
    attach → entry gate → search-action gate → loader load → drift check.

    P36: the production gate fires between the spec freeze and the run
    attach, on the frozen snapshot's own timeline
    (``snapshot_cutoff=spec.data_cutoff``) — the same snapshot earns the
    same historical-research eligibility on any run day; missing bars at
    the snapshot window end still refuse.

    Returns ``(loader, run_context)`` with the loader already loaded.
    """

    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    # The search budget and seed pair are resolved exactly the way
    # trainer.search() resolves them (train_search_run: steps/batch fall
    # back to the model config; seed defaults to 42), so the frozen spec
    # describes the run this entry will execute.
    spec = resolve_runtime_runspec(
        dataset_id=loader.dataset_id,
        data_cutoff=loader.frozen_data_cutoff(),
        purpose="historical_research",
        data_config=data_config,
        backtest_config=backtest_config,
        requested_budget=max(1, steps * batch_size),
        n_folds=1,
        research_domain="unified",
        # The (init_seed, seed) pair the search will actually run:
        # AshareTrainer.__init__ default init_seed=42 and search()'s
        # default seed=42 (the CLI exposes neither), so the frozen
        # paired-seed set is {42} — same derivation as the artifact
        # writer's tuple(sorted({init_seed, seed})).
        seeds=(42,),
        searcher=model_config.searcher,
        max_formula_len=int(model_config.max_formula_len),
        require_clean_tree=False,
    )
    # P36: gate on the snapshot timeline — after the identity freeze the
    # cutoff is bound to, before any run write or the expensive load.
    ProductionGateRunner(
        data_config, min_eligible=min_eligible
    ).require_production(
        purpose=ELIGIBILITY_PURPOSE_HISTORICAL_RESEARCH,
        snapshot_cutoff=spec.data_cutoff,
    )
    run_context = RunContext.attach(
        RunStore(data_config.data_dir),
        spec_id=spec_id,
        run_id=run_id,
        entry_type="train",
        action_mode="research",
        spec=spec,
    )
    # I5: search ≠ loading — the run must have consumed the full
    # qualification chain before any search compute (this refuses before
    # the expensive load when the chain is incomplete).
    run_context.assert_search_ready()
    loader.load_data()
    if loader.dataset_id != spec.dataset_id:
        raise SystemExit(
            "train: the database content drifted between identity freeze "
            "and load; the frozen spec no longer matches the confirmed "
            "composite snapshot — republish and re-preregister "
            "(A02 I2, p27 I6)"
        )
    return loader, run_context


def _refuse_ambiguous_mode_declaration(args) -> None:
    """A03 (p27 I2): boundary action-mode declaration — refuse before any
    expensive work, never silently downgrade (R1/R9).  Formal mode needs
    the preregistered identity; the identity arguments may never ride
    along an explicitly declared non-formal mode."""

    if args.lifecycle_mode != "formal" and (args.spec_id or args.run_id):
        raise SystemExit(
            "train: --spec-id/--run-id are formal-mode identity "
            "arguments and cannot be combined with an explicit "
            "non-formal --lifecycle-mode (p27 I2)"
        )
    if args.lifecycle_mode == "formal" and not (args.spec_id and args.run_id):
        raise SystemExit(
            "train: formal mode requires --spec-id and --run-id of a "
            "preregistered run (create one with `python -m "
            "ashare_model.lifecycle start`); declare --lifecycle-mode "
            "engineering for an explicit non-formal probe (p27 I2/I5)"
        )


def main(trainer_cls) -> None:
    setup_run_logging(run_name="train")
    args = build_parser().parse_args()

    # A03 (p27 I4): the audit mode is READ-ONLY over artifacts and event
    # chains — the trainer produces business artifacts and is therefore
    # refused in audit mode; engineering is the explicit non-formal probe
    # declaration and is machine-marked in the log.
    if args.lifecycle_mode == "audit":
        raise SystemExit(
            "train: audit mode is read-only (p27 §3/I4); declare "
            "--lifecycle-mode engineering for an explicit non-formal probe"
        )
    if args.lifecycle_mode == "engineering":
        from loguru import logger

        logger.warning(
            "lifecycle=unverified (engineering run): no lifecycle identity "
            "— the artifacts of this entry are not formal evidence "
            "(p27 I2/§5.3)"
        )

    run_context = None
    try:
        from .train_windows import _project_root

        root = _project_root()
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)

        _refuse_ambiguous_mode_declaration(args)

        # (run_context is hoisted above the entry's try for the finally.)
        if args.lifecycle_mode == "formal":
            steps = int(args.steps or model_config.train_steps)
            batch_size = int(args.batch_size or model_config.batch_size)
            try:
                loader, run_context = establish_formal_run_context(
                    data_config=data_config,
                    model_config=model_config,
                    backtest_config=backtest_config,
                    spec_id=args.spec_id,
                    run_id=args.run_id,
                    steps=steps,
                    batch_size=batch_size,
                    min_eligible=args.min_eligible,
                )
            except (LifecycleError, RunStoreError) as exc:
                # CLI boundary: a lifecycle/store refusal exits with its
                # reason (p27 I11 — nothing has been loaded or written yet).
                raise SystemExit(f"train: {exc}")
        else:
            # P36: the non-formal paths keep the online (wall-clock) gate —
            # they run against the live database and carry no frozen
            # snapshot identity to derive a historical basis from.
            ProductionGateRunner(
                data_config, min_eligible=args.min_eligible
            ).require_production()
            loader = AshareDataLoader(data_config, model_config)

        trainer = trainer_cls(
            data_config,
            model_config,
            backtest_config,
            loader,
            reward_config,
            run_context=run_context,
            lifecycle_mode=args.lifecycle_mode,
        )
        result = trainer.search(
            searcher=model_config.searcher,
            steps=args.steps,
            batch_size=args.batch_size,
            device=args.device,
        )
        if result.selected is None:
            # No formula met the validation-quality floor: fail loudly so
            # scripts and CI never mistag a no-artifact run as success.
            raise SystemExit(2)
    finally:
        # A03: the entry owns the RunContext — the single-writer lock is
        # released on every exit path (review F1).
        if run_context is not None:
            run_context.close()
        export_log_txt(run_name="train")
