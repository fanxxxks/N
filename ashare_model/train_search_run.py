"""Backend search orchestration for the trainer (P7 Phase B5).

Extracted from ``train.py`` by reason-to-change (IP-07b, mirroring the
evaluation P7 split): this module owns the *backend boundary* — the
single ``search`` entry point over every registered backend, the RL
imitation/random preflight, the non-RL (gp/tpe/random) evaluator wiring
and the imitation pretraining.  It changes when backend orchestration or
budget-matching changes — not when the RL update rule
(:mod:`ashare_model.train_loop`), artifact schemas
(:mod:`ashare_model.train_artifacts`) or window arithmetic
(:mod:`ashare_model.train_windows`) change.

``SearchRunnerMixin`` is composed into ``AshareTrainer``; ``search`` /
``train_search`` remain class attributes of the facade, so the registered
monkeypatch surface (class-attribute patches of ``AshareTrainer.train``
and ``AshareTrainer.train_search``) keeps working.  Observable logging
reads the logger late-bound through the train facade
(:func:`~ashare_model.train_loop._facade_logger`) because
``tests/test_train.py`` pins ``monkeypatch.setattr(train_module.logger,
...)``.  The ``baseline_harness`` import stays lazy: baseline_harness
imports the train facade at module level, so eager import here would
recreate the cycle IP-07b removes elsewhere.

The module is import-leaf-ward of the ``train`` facade: it never imports
``ashare_model.train`` at module level.
"""

from __future__ import annotations

import numpy as np
import torch

from .baseline_harness import SemanticBudgetEvaluator
from .elite_archive import load_elite_archive
from .imitation import pretrain_on_elites
from .search_backends import (
    get_search_backend,
    log_search_start,
    log_search_stop,
)
from .search_contract import SearchRequest, SearchResult
from .train_loop import _facade_logger
from .versions import PROTOCOL_VERSION
from .train_windows import resolve_device
from ashare_portfolio.rebalance import RebalancePolicy
from .candidates import SelectionResult


class SearchRunnerMixin:
    """Backend-boundary lifecycle of ``AshareTrainer`` (B5, IP-07b)."""

    def search(
        self,
        *,
        searcher: str,
        steps: int | None = None,
        batch_size: int | None = None,
        seed: int = 42,
        train_end_date: str | None = None,
        save_artifacts: bool = True,
        device: str | None = None,
        window_cap: tuple[int, int] | None = None,
        rl_initialization: str | None = None,
        elite_archive=None,
    ) -> SearchResult:
        """Run any registered backend and return the common result schema.

        ``train`` and ``train_search`` remain compatibility entry points for
        existing callers, while production and experiment orchestration use
        this method as the single backend boundary.
        """

        resolved_steps = int(steps or self.model_config.train_steps)
        resolved_batch = int(batch_size or self.model_config.batch_size)
        request = SearchRequest(
            seed=int(seed),
            budget=resolved_steps * resolved_batch,
            max_formula_len=int(self.model_config.max_formula_len),
            steps=resolved_steps,
            batch_size=resolved_batch,
        )
        backend = get_search_backend(searcher)
        if searcher == "rl":
            initialization = str(
                rl_initialization or self.model_config.rl_initialization
            )
            if initialization not in {"imitation", "random"}:
                raise ValueError(
                    "rl_initialization must be 'imitation' or 'random'"
                )
            if initialization == "imitation":
                if self.imitation_result is not None:
                    raise RuntimeError(
                        "this trainer was already imitation-pretrained; use a fresh "
                        "trainer for an independent run"
                    )
                if elite_archive is None:
                    elite_archive = load_elite_archive(
                        self.data_config.data_dir / "search_elite_archive.json"
                    )
                self.pretrain_from_archive(elite_archive, seed=request.seed)
            else:
                if self.imitation_result is not None:
                    raise RuntimeError(
                        "random-initialized RL requires a fresh unpretrained trainer"
                    )
                self.rl_initialization = "random"
            log_search_start(searcher, request)

            def runner(req: SearchRequest, _evaluator) -> SearchResult:
                self.train(
                    steps=req.steps,
                    batch_size=req.batch_size,
                    seed=req.seed,
                    save_artifacts=save_artifacts,
                    train_end_date=train_end_date,
                    device=device,
                    window_cap=window_cap,
                )
                if self.search_result is None:
                    raise RuntimeError("RL run completed without SearchResult")
                return self.search_result

            result = backend.search(request, None, runner=runner)
            log_search_stop(result)
            return result

        self.train_search(
            searcher=searcher,
            steps=resolved_steps,
            batch_size=resolved_batch,
            seed=seed,
            train_end_date=train_end_date,
            save_artifacts=save_artifacts,
            device=device,
            window_cap=window_cap,
        )
        if self.search_result is None:
            raise RuntimeError(f"{searcher} run completed without SearchResult")
        return self.search_result

    def pretrain_from_archive(self, archive, *, seed: int = 42):
        """Imitate baseline elites, then reset the optimizer for RL."""

        result = pretrain_on_elites(
            self.model,
            archive,
            max_formula_len=int(self.model_config.max_formula_len),
            epochs=int(self.model_config.imitation_epochs),
            batch_size=int(self.model_config.imitation_batch_size),
            learning_rate=float(self.model_config.imitation_learning_rate),
            seed=int(seed),
        )
        # The supervised optimizer state is not part of the RL comparison.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.model_config.learning_rate
        )
        self.rl_initialization = "imitation"
        self.imitation_result = result
        _facade_logger().info(
            "rl.imitation samples={} tokens={} initial_loss={} final_loss={} "
            "initial_accuracy={} final_accuracy={}",
            result.sample_count,
            result.token_count,
            result.initial_loss,
            result.final_loss,
            result.initial_token_accuracy,
            result.final_token_accuracy,
        )
        return result

    def train_search(
        self,
        *,
        searcher: str,
        steps: int | None = None,
        batch_size: int | None = None,
        seed: int = 42,
        train_end_date: str | None = None,
        save_artifacts: bool = True,
        device: str | None = None,
        window_cap: tuple[int, int] | None = None,
    ) -> list[int] | None:
        """Run a non-RL searcher (``gp``, ``tpe`` or ``random``) over the
        training window with the matched unique-semantic-evaluation budget
        and produce the standard training artifact.

        The searcher is billed through the same semantic cache as RL
        (``steps x batch_size`` unique semantic evaluations), so the
        production default can switch backends without changing the
        budget semantics or the artifact contract.
        """

        if searcher not in ("gp", "tpe", "random"):
            raise ValueError(
                f"train_search supports 'gp', 'tpe' or 'random', got {searcher!r}"
            )
        steps = steps or self.model_config.train_steps
        batch_size = batch_size or self.model_config.batch_size
        vm_device = resolve_device(device)
        window = self.prepare_window(train_end_date, vm_device, window_cap)
        budget = steps * batch_size
        evaluator = self._build_search_evaluator(
            searcher=searcher, window=window, budget=budget
        )
        request = SearchRequest(
            seed=int(seed),
            budget=int(budget),
            max_formula_len=int(self.model_config.max_formula_len),
            steps=int(steps),
            batch_size=int(batch_size),
        )
        backend = get_search_backend(searcher)
        log_search_start(searcher, request)
        result = backend.search(
            request, evaluator, vocab=self.vocab, feature_ids=self.feature_ids
        )
        self.search_result = result
        log_search_stop(result)
        return self._finish_search(
            result=result,
            searcher=searcher,
            window_contract=window.contract,
            vm_device=vm_device,
            seed=seed,
            budget=budget,
            save_artifacts=save_artifacts,
        )

    def _build_search_evaluator(self, *, searcher: str, window, budget: int):
        """Wire the shared semantic-budget evaluator for one non-RL run:
        the VM executor, the trainer's own calibration fingerprint
        executor and the trainer's semantic cache as the budget ledger.

        The evaluator import is top-level (t21): baseline_harness now
        imports ``sample_random_formulas`` from ``train_windows`` — its
        last module-level train-facade edge is gone, so the historical
        call-time cycle break has no surviving reason."""

        def execute(tokens) -> np.ndarray | None:
            signal = self.vm.execute(tokens, window.factor_tensor)
            if signal is None:
                return None
            return signal.detach().cpu().numpy()

        return SemanticBudgetEvaluator(
            target=window.target_ret,
            realized_ret=window.realized_ret,
            rebalance_mask=window.rebalance_mask,
            universe_mask=window.train_universe_mask,
            backtest_config=self.backtest_config,
            reward_config=self.reward_config,
            val_windows=window.val_windows,
            train_signal_range=window.train_signal_range,
            budget=budget,
            execute=execute,
            fingerprint_execute=self._calibration_execute,
            dataset_id=self.loader.dataset_id,
            protocol_version=PROTOCOL_VERSION,
            window_id=self._window_id(
                window.contract,
                window.val_windows,
                RebalancePolicy.from_config(self.backtest_config),
                domain_id=self.domain_id,
            ),
            # Selection tie-break keys and the capacity-audit dollar volume
            # are sliced to the measured window (a capped admission window
            # is a stock slice of the loader's universe) — the same slice
            # train() applies, so every searcher sees the same shapes.
            tie_break_keys=np.asarray(self.loader.ts_codes)[
                : window.factor_tensor.shape[1]
            ],
            adv=np.asarray(self.loader.dollar_volume())[
                : window.factor_tensor.shape[1], : window.train_end_idx
            ],
            blocked_buy=window.blocked_buy,
            blocked_sell=window.blocked_sell,
            source=searcher,
            candidate_prefix=searcher,
            chunk=window.reward_chunk,
            # The evaluator bills the trainer's own semantic cache, so
            # ``trainer.semantic_cache.budget_used`` is the true unique-
            # semantic-evaluation ledger for every searcher backend (the
            # protocol's trained rows record exactly this number).
            cache=self.semantic_cache,
        )

    def _finish_search(
        self,
        *,
        result: SearchResult,
        searcher: str,
        window_contract,
        vm_device: torch.device,
        seed: int,
        budget: int,
        save_artifacts: bool,
    ) -> list[int] | None:
        """Post-run bookkeeping shared by every non-RL backend: candidate
        and history ingestion from the result, the no-eligible warning,
        and the artifact write."""

        selected = result.selected
        if selected is None:
            _facade_logger().warning(
                "No eligible formula met every validation gate "
                f"(searcher={searcher}); no strategy artifact is written."
            )
            self._sync_best_from_selection()
            return None
        self._candidate_scores = {
            tuple(score.tokens): score
            for score in result.scores
            if score.tokens is not None
        }
        self.selection_result = SelectionResult(selected, None, result.scores)
        self._sync_best_from_selection()
        self.history = [
            {
                "step": float(step),
                "avg_reward": float(reward),
                "best_val_reward": float(reward),
                "unique_semantic_evals": float(budget_used),
                "searcher": searcher,
            }
            for step, (budget_used, reward) in enumerate(result.best_so_far)
        ]
        if not save_artifacts:
            _facade_logger().success(
                f"Search complete (artifacts skipped); searcher={searcher} "
                f"val_reward={selected.val_reward:.3f} "
                f"val_icir={selected.val_icir:.3f} "
                f"formula={selected.formula_text}"
            )
            return self.best_tokens
        return self._write_artifact(
            contract=window_contract,
            vm_device=vm_device,
            searcher=searcher,
            seed=seed,
            requested_budget=int(budget),
        )
