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

A03 (p27): the CLI entry lives in :mod:`ashare_model.train_entry` (the
facade line budget is a registered contract); ``main`` remains on this
facade's import surface by delegation, and the formal lifecycle gate
consumes the RunContext attached there.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from loguru import logger  # noqa: F401  (monkeypatch surface: re-export)

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    ModelConfig,
    RewardConfig,
)

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
from .train_entry import main as _entry_main
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


def _trainer_entry_gate(
    run_context: RunContext | None,
    lifecycle_mode: str,
    data_config: DataConfig,
) -> RunContext | None:
    """A03 (p27 I1/I2): the trainer's entry-identity check.

    The formal entry passes the RunContext attached to the preregistered
    run (validated for type and entry match); library/constructor use
    (tests, probes) carries no lifecycle identity and must declare the
    explicit non-formal mode (p27 §3/I5: 工程探针必须显式声明) — a missing
    identity is never a formal pass.
    """

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
        return run_context
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
    return None


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
        self.run_context = _trainer_entry_gate(
            run_context, lifecycle_mode, data_config
        )
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
    """Delegates to :mod:`ashare_model.train_entry` (A03 facade budget)."""

    _entry_main(AshareTrainer)


if __name__ == "__main__":
    main()
