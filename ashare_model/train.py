"""REINFORCE trainer facade for A-share factor formulas (P7 Phase B).

Compatibility facade (IP-07b): ``AshareTrainer`` composes the
reason-to-change mixins — the RL loop (:mod:`ashare_model.train_loop`),
backend orchestration (:mod:`ashare_model.train_search_run`), artifact
persistence (:mod:`ashare_model.train_artifacts`) and window binding
(:mod:`ashare_model.train_windows`).  Every moved name remains an
attribute of this module (import-surface contract:
``tests/test_train_module_split.py``), and the registered monkeypatch
surface (``batched_basket_rewards`` / ``score_chunk_size`` /
``logger`` on the module namespace, ``AshareTrainer.train`` /
``train_search`` class attributes, ``trainer.vm.execute`` instances)
keeps working through facade late binding.

Usage:
    python -m ashare_model.train [--config config/ashare_config.yaml]
                                 [--steps N] [--batch-size N]
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch
from loguru import logger  # noqa: F401  (monkeypatch surface: re-export)

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    ModelConfig,
    RewardConfig,
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_reward_config,
)
from ashare_data.gates import ProductionGateRunner

from .alphagpt import AlphaGPTModel
from .lifecycle import RunContext, assert_formal_run_lifecycle
from .lifecycle import LifecycleError
from .candidates import (
    CandidateScore,
    CandidateScorer,
    CandidateSelector,
    SelectionResult,
    score_chunk_size,
)
from .data_loader import AshareDataLoader
from .reward import batched_basket_rewards
from .train_artifacts import ArtifactPersistenceMixin, write_trainer_artifact
from .train_loop import RLTrainingLoopMixin
from .train_search_run import SearchRunnerMixin
from .train_windows import (
    WindowPreparationMixin,
    _TrainWindow,
    _project_root,
    resolve_device,
    sample_random_formulas,
    validation_start,
    validation_windows,
)
from .vm import StackVM
from .vocab import FORMULA_VOCAB
from ashare_logging import export_log_txt, setup_run_logging


class AshareTrainer(
    RLTrainingLoopMixin,
    SearchRunnerMixin,
    WindowPreparationMixin,
    ArtifactPersistenceMixin,
):
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        backtest_config: BacktestConfig,
        loader: AshareDataLoader | None = None,
        reward_config: RewardConfig | None = None,
        init_seed: int = 42,
        # P6: research-domain identity and search-space restriction.
        # ``domain_id`` defaults to the reserved compatible semantic
        # "unified"; ``feature_ids`` (global vocab token ids, None = all)
        # restricts every sampling mask (docs/p6_research_domain_contract
        # .md 鎼?.2).
        domain_id: str = "unified",
        feature_ids: list[int] | None = None,
        # A03 (p27 I1/I2): the formal entry passes the RunContext attached
        # to the preregistered run.  Library/constructor use (tests,
        # probes) carries no lifecycle identity and must declare the
        # explicit non-formal mode (p27 §3/I5: 工程探针必须显式声明).
        run_context: RunContext | None = None,
        *,
        lifecycle_mode: str = "engineering",
    ):
        self.data_config = data_config
        # A03 (p27 I2): the formal entry gate consumes the RunContext's
        # real identity; without a context, only the explicitly declared
        # non-formal modes proceed — a missing identity is never a pass.
        if run_context is not None:
            if not isinstance(run_context, RunContext):
                raise LifecycleError(
                    "run_context must be the lifecycle RunContext (p27 I1: "
                    "formal entries receive the single identity type)"
                )
            if run_context.entry_type != "train":
                raise LifecycleError(
                    f"RunContext entry type {run_context.entry_type!r} "
                    "does not match the train entry (p27 I1)"
                )
        else:
            if lifecycle_mode not in ("engineering", "audit"):
                raise LifecycleError(
                    "formal train requires a RunContext (attach a "
                    "preregistered run via --spec-id/--run-id); "
                    "constructor use must declare "
                    "lifecycle_mode='engineering'|'audit' explicitly "
                    "(p27 I2/I5)"
                )
            assert_formal_run_lifecycle(
                "train",
                data_config.data_dir,
                action_mode=lifecycle_mode,
            )
        self.run_context = run_context
        self.model_config = model_config
        self.backtest_config = backtest_config
        self.reward_config = reward_config or RewardConfig()
        self.init_seed = init_seed
        self.domain_id = str(domain_id)
        self.feature_ids = feature_ids
        self.loader = loader or AshareDataLoader(data_config, model_config)
        if self.loader.factor_tensor is None:
            self.loader.load_data()
        self.vocab = FORMULA_VOCAB
        # The PIT eligibility mask is wired at construction like every
        # formal path; train() re-assigns the sliced device copy when the
        # factor tensor moves to the compute device.
        self.vm = StackVM(
            self.vocab,
            universe_mask=torch.tensor(
                self.loader.universe_mask, dtype=torch.bool
            ),
        )
        # Pin the weight initialization so the same (init_seed, seed)
        # pair reproduces the same training on any machine.
        torch.manual_seed(init_seed)
        self.model = AlphaGPTModel(model_config, self.vocab)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=model_config.learning_rate
        )
        self.best_val_reward = -float("inf")
        self.best_val_icir = -float("inf")
        self.best_train_reward = -float("inf")
        self.best_train_icir = -float("inf")
        self.best_direction = 1
        self.best_tokens: list[int] | None = None
        self.best_formula = ""
        self.history: list[dict[str, object]] = []
        self._collapse_streak = 0
        self.candidate_scorer = CandidateScorer(
            self.backtest_config,
            self.reward_config,
            # Resolve through the train module at call time so test/adaptor
            # injection and every generation path still share one scorer.
            reward_function=lambda *args, **kwargs: batched_basket_rewards(
                *args, **kwargs
            ),
        )
        self.candidate_selector = CandidateSelector()
        self.selection_result = SelectionResult(None, None, ())
        self._candidate_scores: OrderedDict[tuple[int, ...], CandidateScore] = (
            OrderedDict()
        )
        # Invalid/degenerate formulas (no canonical form) are rejected
        # pre-evaluation: they never touch the VM and never consume
        # evaluation budget.  Cached by token sequence, LRU-bounded, so a
        # converged policy that keeps sampling them does not re-score.
        self._invalid_cache: OrderedDict[tuple[int, ...], CandidateScore] = (
            OrderedDict()
        )
        # Operators observed across the whole run's executed formulas; the
        # run hard-fails when this stays empty (bare-factor screening).
        self._run_operator_coverage: set[str] = set()
        self.search_result: SearchResult | None = None
        self.rl_initialization = "random"
        self.imitation_result = None

    def _reward_chunk_size(self, signal_bytes: int) -> int:
        """Resolve the streamed-reward chunk size through the train module
        namespace: ``tests/test_train.py`` pins the streamed-vs-single-pass
        chunk semantics via ``monkeypatch.setattr(train_module,
        "score_chunk_size", ...)``, so the call must stay late-bound on
        this facade after the window code moved to ``train_windows``
        (B3, IP-07b 閳?registered monkeypatch surface)."""

        return score_chunk_size(signal_bytes)

    @property
    def best_reward(self) -> float:
        """Deprecated read-only alias for the explicit validation reward."""

        return self.best_val_reward

def main() -> None:
    setup_run_logging(run_name="train")
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
    args = parser.parse_args()

    try:
        root = _project_root()
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(data_config, min_eligible=args.min_eligible).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)

        # A03 (p27 I2): boundary action-mode declaration — refuse before
        # any expensive work, never silently downgrade (R1/R9).
        if args.lifecycle_mode != "formal" and (args.spec_id or args.run_id):
            raise SystemExit(
                "train: --spec-id/--run-id are formal-mode identity "
                "arguments and cannot be combined with an explicit "
                "non-formal --lifecycle-mode (p27 I2)"
            )
        if args.lifecycle_mode == "formal" and not (
            args.spec_id and args.run_id
        ):
            raise SystemExit(
                "train: formal mode requires --spec-id and --run-id of a "
                "preregistered run (create one with `python -m "
                "ashare_model.lifecycle start`); declare "
                "--lifecycle-mode engineering for an explicit non-formal "
                "probe (p27 I2/I5)"
            )

        run_context = None
        if args.lifecycle_mode == "formal":
            try:
                from .run_store import RunStore

                # A03 (p27 I6): freeze the plan and establish the run
                # identity BEFORE the expensive load — the composite
                # snapshot confirms cheaply, the spec freezes from it, the
                # run attaches, the gate fires, and only then does the
                # loader build frames.
                loader = AshareDataLoader(data_config, model_config)
                loader.confirm_input_snapshot()
                # The search budget and seed pair are resolved exactly the
                # way trainer.search() resolves them (train_search_run:
                # steps/batch fall back to the model config; seed defaults
                # to 42), so the frozen spec describes the run this entry
                # will execute.
                steps = int(args.steps or model_config.train_steps)
                batch_size = int(args.batch_size or model_config.batch_size)
                from .runspec import resolve_runtime_runspec

                spec = resolve_runtime_runspec(
                    dataset_id=loader.dataset_id,
                    data_cutoff=loader.frozen_data_cutoff(),
                    data_config=data_config,
                    backtest_config=backtest_config,
                    requested_budget=max(1, steps * batch_size),
                    n_folds=1,
                    research_domain="unified",
                    # The (init_seed, seed) pair the search will actually
                    # run: AshareTrainer.__init__ default init_seed=42 and
                    # search()'s default seed=42 (the CLI exposes neither),
                    # so the frozen paired-seed set is {42} — same
                    # derivation as the artifact writer's
                    # tuple(sorted({init_seed, seed})).
                    seeds=(42,),
                    searcher=model_config.searcher,
                    max_formula_len=int(model_config.max_formula_len),
                    require_clean_tree=False,
                )
                run_context = RunContext.attach(
                    RunStore(data_config.data_dir),
                    spec_id=args.spec_id,
                    run_id=args.run_id,
                    entry_type="train",
                    action_mode="research",
                    spec=spec,
                )
                # I5: search ≠ loading — the run must have consumed the
                # full qualification chain before any search compute (this
                # refuses before the expensive load when incomplete).
                run_context.assert_search_ready()
                loader.load_data()
                if loader.dataset_id != spec.dataset_id:
                    raise SystemExit(
                        "train: the database content drifted between "
                        "identity freeze and load; the frozen spec no "
                        "longer matches the confirmed composite snapshot "
                        "— republish and re-preregister (A02 I2, p27 I6)"
                    )
            except LifecycleError as exc:
                # CLI boundary: a lifecycle refusal exits with its reason
                # (p27 I11 — nothing has been loaded or written yet).
                raise SystemExit(f"train: {exc}")
        else:
            loader = AshareDataLoader(data_config, model_config)

        trainer = AshareTrainer(
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
        export_log_txt(run_name="train")


if __name__ == "__main__":
    main()
