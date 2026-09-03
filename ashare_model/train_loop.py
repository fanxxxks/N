"""REINFORCE training loop for the trainer (P7 Phase B4).

Extracted from ``train.py`` by reason-to-change (IP-07b, mirroring the
evaluation P7 split): this module owns the *RL loop lifecycle* — batch
sampling, semantic-deduplicated scoring, the actor-critic update, per-step
grammar/diagnostics recording, selection bookkeeping and the shaping of
the run into the shared ``SearchResult`` schema.  It changes when the RL
update rule or budget semantics change — not when artifact schemas
(:mod:`ashare_model.train_artifacts`), window arithmetic
(:mod:`ashare_model.train_windows`) or backend orchestration
(:mod:`ashare_model.train_search_run`) change.

``RLTrainingLoopMixin`` is composed into ``AshareTrainer``; every method
stays a class attribute of the facade, so the registered monkeypatch
surface (``AshareTrainer.train`` class-attribute patches) keeps working.
Observable logging reads the logger late-bound through the train facade
(:func:`_facade_logger`) because ``tests/test_train.py`` pins
``monkeypatch.setattr(train_module.logger, ...)`` — the same facade-bound
pattern the evaluation split uses for version constants.

The module is import-leaf-ward of the ``train`` facade: it never imports
``ashare_model.train`` at module level.  Statement grouping is a verbatim
regrouping of the historical loop: ``_StepState`` carries the per-step
mutable locals, ``_LoopContext`` the window-derived read-only context, and
``_RLRunState`` the run-level ledger — no behavioral change.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm import tqdm

from .alphagpt import build_action_mask
from .candidates import (
    PARETO_OBJECTIVES,
    CandidateScore,
    CandidateSpec,
)
from .complexity import complexity_bill
from .reward import REWARD_VERSION
from .rl_diagnostics import aggregate_rl_run, gradient_l2_norm, summarize_rl_step
from .search_contract import SearchResult
from .semantic_cache import SEMANTIC_CACHE_VERSION, SemanticCacheKey
from .semantic_sampling import advance_stack_state
from .train_windows import resolve_device
from .versions import PROTOCOL_VERSION
from . import ir as ir_module
from .vm import formula_decode
from .vocab import FORMULA_VOCAB
from ashare_portfolio.execution_spec import execution_provenance


def _facade_logger():
    """The train facade's logger, resolved at call time.

    ``tests/test_train.py`` pins the collapse warning via
    ``monkeypatch.setattr(train_module.logger, "warning", ...)``; every
    trainer-path log emitted from this module therefore reads the logger
    through the train module so the patch keeps late-binding (IP-07b
    registered monkeypatch surface, same pattern as eval_artifacts).
    """

    from ashare_model import train as _facade  # noqa: PLC0415

    return _facade.logger


@dataclass
class _LoopContext:
    """Window-derived read-only context shared by every step of one
    ``train()`` call (created once, after ``prepare_window``)."""

    batch_size: int
    max_len: int
    factor_tensor: torch.Tensor
    target_ret: np.ndarray
    realized_ret: np.ndarray
    rebalance_mask: np.ndarray
    val_windows: list[tuple[int, int]]
    train_signal_range: tuple[int, int]
    train_universe_mask: np.ndarray
    blocked_buy: np.ndarray
    blocked_sell: np.ndarray
    reward_chunk: int
    tie_break_keys: np.ndarray
    adv: np.ndarray


@dataclass
class _StepState:
    """Per-step mutable locals of the historical loop body (verbatim
    regrouping: the stage methods mutate these exactly like the original
    loop locals)."""

    step_results: dict[tuple[int, ...], CandidateScore] = field(
        default_factory=dict
    )
    pending: list[tuple[CandidateSpec, np.ndarray]] = field(
        default_factory=list
    )
    # evaluated: list of (token key, canonical cache key, fingerprint,
    # complexity bill).
    evaluated: list[
        tuple[tuple[int, ...], SemanticCacheKey, str | None, float | None]
    ] = field(default_factory=list)
    # Grammar stats are accumulated at execution time, so only formulas
    # the VM actually executed and produced a signal for count toward
    # length/operator coverage (a formula the VM rejected is structurally
    # invalid — never an operator user).
    executed_stats: list[tuple[tuple[int, ...], int, set[str]]] = field(
        default_factory=list
    )
    # canonical_share maps each canonical form to the first token key
    # that handled it this step: identical sequences and canonical
    # duplicates (e.g. ADD(a, b) vs ADD(b, a)) share that row's
    # evaluation — no repeated work, no repeated billing.
    canonical_share: dict[SemanticCacheKey, tuple[int, ...]] = field(
        default_factory=dict
    )
    cache_hits: int = 0
    semantic_dedups: int = 0
    semantic_duplicate_proposals: int = 0


@dataclass
class _RLRunState:
    """Run-level ledger accumulated across the RL loop (the aggregates the
    ``SearchResult`` diagnostics and best-so-far curve are shaped from)."""

    reward_values: list[float] = field(default_factory=list)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    formula_lengths: list[int] = field(default_factory=list)
    step_summaries: list[dict[str, object]] = field(default_factory=list)
    semantic_duplicates: int = 0
    best_so_far: list[tuple[int, float]] = field(default_factory=list)
    best_reward: float = -float("inf")


class RLTrainingLoopMixin:
    """REINFORCE loop lifecycle of ``AshareTrainer`` (B4, IP-07b)."""

    # Bounded LRU of evaluated formulas: rewards are a deterministic function
    # of the token sequence, so repeats (frequent once the policy
    # concentrates) are scored once and reused across steps.
    _REWARD_CACHE_CAP = 16384

    def train(
        self,
        steps: int | None = None,
        batch_size: int | None = None,
        seed: int = 42,
        save_artifacts: bool = True,
        train_end_date: str | None = None,
        device: str | None = None,
        window_cap: tuple[int, int] | None = None,
    ) -> list[int] | None:
        """Run REINFORCE training.

        ``device`` follows :func:`resolve_device` (auto/cpu/cuda) and
        places **only the factor tensor / VM** on the compute device.  The
        policy model, the sampling loop and the gradient update always
        stay on CPU: torch's CPU RNG stream (MT19937) and dropout are
        device-independent, so the same (init_seed, seed) samples the same
        formula sequence on every machine — a tested invariant.  Only the
        VM's float32 arithmetic differs between devices (~1e-7), which can
        matter solely when a signal lands exactly on a scoring threshold.
        The artifact records ``init_seed`` and ``device``.
        """

        torch.manual_seed(seed)
        np.random.seed(seed)
        steps = steps or self.model_config.train_steps
        batch_size = batch_size or self.model_config.batch_size
        vm_device = resolve_device(device)

        window = self.prepare_window(train_end_date, vm_device, window_cap)
        # Selection tie-break keys and the capacity-audit dollar volume are
        # sliced to the measured window (a capped admission window is a
        # stock slice of the loader's universe).
        ctx = _LoopContext(
            batch_size=batch_size,
            max_len=self.model_config.max_formula_len,
            factor_tensor=window.factor_tensor,
            target_ret=window.target_ret,
            realized_ret=window.realized_ret,
            rebalance_mask=window.rebalance_mask,
            val_windows=window.val_windows,
            train_signal_range=window.train_signal_range,
            train_universe_mask=window.train_universe_mask,
            blocked_buy=window.blocked_buy,
            blocked_sell=window.blocked_sell,
            reward_chunk=window.reward_chunk,
            tie_break_keys=np.asarray(self.loader.ts_codes)[
                : window.factor_tensor.shape[1]
            ],
            adv=np.asarray(self.loader.dollar_volume())[
                : window.factor_tensor.shape[1], : window.target_ret.shape[1]
            ],
        )

        run = _RLRunState()
        pbar = tqdm(range(steps))
        for step in pbar:
            self._train_step(step, pbar, ctx, run)

        return self._finish_training(
            contract=window.contract,
            run=run,
            seed=seed,
            steps=steps,
            batch_size=batch_size,
            vm_device=vm_device,
            save_artifacts=save_artifacts,
        )

    def _train_step(
        self,
        step: int,
        pbar,
        ctx: _LoopContext,
        run: _RLRunState,
    ) -> None:
        """One REINFORCE step, in the historical statement order: sample,
        route every sampled formula through the semantic-dedup gates,
        flush the trailing chunk, commit the budget ledger, resolve the
        batch rewards, select, monitor collapse, record grammar stats,
        update the policy and record the step."""

        policy_device = next(self.model.parameters()).device
        sequences, log_probs, values, entropies = self._sample_batch(
            ctx.batch_size, ctx.max_len, policy_device
        )
        rewards = torch.zeros(ctx.batch_size, device=policy_device)

        # Batched reward evaluation with semantic deduplication (T2-01):
        # the budget unit is the **unique semantic formula evaluation**.
        # Structurally invalid / degenerate formulas are rejected
        # pre-evaluation (never executed, never billed); canonical
        # duplicates hit the semantic cache; numerically equivalent
        # formulas (same calibration fingerprint) reuse the first
        # evaluation's score without billing.  Valid signals are scored
        # in chunks so the vectorized basket simulation stays
        # memory-bounded.
        state = _StepState()
        for i in range(ctx.batch_size):
            key = tuple(sequences[i].tolist())
            ckey = self.semantic_cache.key_for(key)
            if not self._route_one(key, ckey, state, ctx):
                continue
            # Score as soon as one full chunk is ready: buffering every
            # unique signal of a step at once costs batch_size x
            # [stocks, dates] x 4 bytes and exhausts RAM on 16 GB
            # machines for large batches before any chunk is scored.
            if len(state.pending) >= ctx.reward_chunk:
                self._score_pending_chunk(
                    state.pending,
                    ctx.target_ret,
                    ctx.realized_ret,
                    ctx.rebalance_mask,
                    ctx.val_windows,
                    state.step_results,
                    ctx.train_universe_mask,
                    ctx.blocked_buy,
                    ctx.blocked_sell,
                    ctx.train_signal_range,
                    ctx.tie_break_keys,
                    ctx.adv,
                )
                state.pending = []
        self._score_pending_chunk(
            state.pending,
            ctx.target_ret,
            ctx.realized_ret,
            ctx.rebalance_mask,
            ctx.val_windows,
            state.step_results,
            ctx.train_universe_mask,
            ctx.blocked_buy,
            ctx.blocked_sell,
            ctx.train_signal_range,
            ctx.tie_break_keys,
            ctx.adv,
        )
        self._commit_step_evaluations(state, run)
        batch_scores = self._resolve_batch_rewards(
            sequences, ctx.batch_size, state, rewards
        )
        unique_frac = self._update_selection_and_track(
            state, ctx.batch_size, pbar
        )
        stats = self._grammar_step_stats(state, ctx.batch_size, unique_frac)
        step_diagnostics, loss, value_loss, entropy = self._policy_update(
            log_probs,
            rewards,
            values,
            entropies,
            batch_scores,
            stats["lengths"],
            stats["step_op_coverage"],
            state.semantic_duplicate_proposals,
        )
        self._record_step(
            step,
            pbar,
            rewards,
            loss,
            value_loss,
            entropy,
            step_diagnostics,
            stats,
            state.semantic_duplicate_proposals,
            run,
        )

    def _sample_batch(
        self, batch_size: int, max_len: int, policy_device: torch.device
    ):
        """Sample one batch of postfix formulas under the typed action
        mask (the policy and sampling stay on CPU — the model's device:
        its RNG stream and dropout are device-independent by
        construction)."""

        inp = torch.zeros((batch_size, 1), dtype=torch.long, device=policy_device)
        stack_sizes = torch.zeros(batch_size, dtype=torch.long, device=policy_device)
        # P7-E: one semantic-type id per stack slot (0 = empty), so the
        # action mask can enforce operator signature legality.
        stack_types = torch.zeros(
            batch_size, max_len, dtype=torch.long, device=policy_device
        )
        done = torch.zeros(batch_size, dtype=torch.bool, device=policy_device)
        log_probs: list[torch.Tensor] = []
        sampled_tokens: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []

        for pos in range(max_len):
            logits, value = self.model(inp)
            values.append(value.squeeze(-1))
            mask = build_action_mask(
                stack_sizes, done, pos, max_len, self.vocab,
                feature_ids=self.feature_ids,
                stack_types=stack_types,
            )
            dist = Categorical(logits=logits + mask)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            sampled_tokens.append(action)
            inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            self._update_stack_state(action, stack_sizes, stack_types, done)

        sequences = torch.stack(sampled_tokens, dim=1)
        return sequences, log_probs, values, entropies

    def _route_one(
        self,
        key: tuple[int, ...],
        ckey: SemanticCacheKey | None,
        state: _StepState,
        ctx: _LoopContext,
    ) -> bool:
        """Route one sampled formula through the dedup gates.  Returns
        ``True`` when the formula was newly executed and its signal joined
        ``state.pending`` (the caller owns the chunk-size flush); every
        counter mutation lands on ``state`` exactly like the historical
        loop locals."""

        if ckey is None:
            self._resolve_invalid(key, state, ctx)
            return False
        first = state.canonical_share.get(ckey)
        if first is not None:
            # Same canonical formula already handled this step
            # (identical tokens or a commuted form): share its
            # evaluation; no new work, no budget.
            state.semantic_duplicate_proposals += 1
            return False
        state.canonical_share[ckey] = key
        score = self.semantic_cache.get(ckey)
        if score is not None:
            state.cache_hits += 1
            state.semantic_duplicate_proposals += 1
            state.step_results[key] = score
            return False
        return self._evaluate_fresh(key, ckey, state, ctx)

    def _resolve_invalid(
        self,
        key: tuple[int, ...],
        state: _StepState,
        ctx: _LoopContext,
    ) -> None:
        """Structurally invalid or degenerate (constant-producing)
        formulas: rejected pre-evaluation, cached by token sequence so
        repeats skip the rejection path."""

        score = self._invalid_cache.get(key)
        if score is None:
            spec = CandidateSpec(
                candidate_id="rl:" + ",".join(str(t) for t in key),
                formula_text=formula_decode(list(key), self.vocab),
                source="rl",
                tokens=key,
            )
            score = self.candidate_scorer.score(
                spec,
                None,
                ctx.target_ret,
                ctx.val_windows,
                blocked_buy=ctx.blocked_buy,
                blocked_sell=ctx.blocked_sell,
                formula_valid=False,
                train_signal_range=ctx.train_signal_range,
                universe_mask=ctx.train_universe_mask,
            )
            self._invalid_cache[key] = score
            while len(self._invalid_cache) > self._REWARD_CACHE_CAP:
                self._invalid_cache.popitem(last=False)
        state.step_results[key] = score

    def _evaluate_fresh(
        self,
        key: tuple[int, ...],
        ckey: SemanticCacheKey,
        state: _StepState,
        ctx: _LoopContext,
    ) -> bool:
        """A new canonical formula: check semantic equivalence on the
        calibration slice before paying for a full evaluation.  The
        semantic class is (fingerprint, complexity bill): two numerically
        equivalent formulas with different bills carry different
        penalties, so their scores are not interchangeable.  Returns
        ``True`` when the formula executed and its signal joined
        ``state.pending``."""

        canonical = ir_module.canonical_ast(key, self.vocab)
        bill = complexity_bill(canonical) if canonical is not None else None
        fingerprint = self.semantic_cache.fingerprint(
            key, self._calibration_execute, self.vocab
        )
        score = (
            self.semantic_cache.score_by_fingerprint(fingerprint, bill)
            if fingerprint is not None
            else None
        )
        if score is not None:
            state.semantic_dedups += 1
            state.semantic_duplicate_proposals += 1
            self.semantic_cache.put(ckey, score, fingerprint, bill)
            state.step_results[key] = score
            return False
        spec = CandidateSpec(
            candidate_id="rl:" + ",".join(str(token) for token in key),
            formula_text=formula_decode(list(key), self.vocab),
            source="rl",
            tokens=key,
        )
        signal = self.vm.execute(key, ctx.factor_tensor)
        if signal is None:
            state.step_results[key] = self.candidate_scorer.score(
                spec,
                None,
                ctx.target_ret,
                ctx.val_windows,
                blocked_buy=ctx.blocked_buy,
                blocked_sell=ctx.blocked_sell,
                formula_valid=False,
                train_signal_range=ctx.train_signal_range,
                universe_mask=ctx.train_universe_mask,
            )
            return False
        ast = ir_module.decode(key, self.vocab)
        state.executed_stats.append(
            (
                key,
                ir_module.formula_length(key, self.vocab),
                ir_module.operator_names(ast),
            )
        )
        signal_np = signal.detach().cpu().numpy()
        state.pending.append((spec, signal_np))
        state.evaluated.append((key, ckey, fingerprint, bill))
        return True

    def _score_pending_chunk(
        self,
        pending: list[tuple[CandidateSpec, np.ndarray]],
        target_ret: np.ndarray,
        realized_ret: np.ndarray,
        rebalance_mask: np.ndarray,
        val_windows: list[tuple[int, int]],
        step_results: dict[tuple[int, ...], CandidateScore],
        universe_mask: np.ndarray,
        blocked_buy: np.ndarray | None = None,
        blocked_sell: np.ndarray | None = None,
        train_signal_range: tuple[int, int] | None = None,
        tie_break_keys: np.ndarray | None = None,
        adv: np.ndarray | None = None,
    ) -> None:
        """Score one chunk of pending formulas and merge the outcomes."""

        if not pending:
            return
        scores = self.candidate_scorer.score_many(
            [spec for spec, _ in pending],
            [signal_np for _, signal_np in pending],
            target_ret,
            val_windows,
            blocked_buy=blocked_buy,
            blocked_sell=blocked_sell,
            train_signal_range=train_signal_range,
            universe_mask=universe_mask,
            # Deterministic selection ties: the loader's canonical sorted
            # code order is the stable key (T1-02); sliced to the measured
            # window (a capped admission window is a stock slice).
            tie_break_keys=tie_break_keys,
            adv=adv,
            realized_ret=realized_ret,
            rebalance_mask=rebalance_mask,
        )
        for score in scores:
            assert score.tokens is not None
            step_results[score.tokens] = score

    def _commit_step_evaluations(
        self, state: _StepState, run: _RLRunState
    ) -> None:
        """Commit every newly evaluated formula to the semantic cache and
        extend the best-so-far ledger (a NaN validation reward never
        poisons the curve — the canonical identity layer rejects
        non-finite bests fail-closed)."""

        for key, ckey, fingerprint, bill in state.evaluated:
            budget_before_put = int(self.semantic_cache.budget_used)
            self.semantic_cache.put(ckey, state.step_results[key], fingerprint, bill)
            budget_after_put = int(self.semantic_cache.budget_used)
            if budget_after_put > budget_before_put:
                # P9 defect fix (exposed by the v4 sampling layout): a
                # NaN validation reward must not poison the best-so-far
                # curve — max(-inf, nan) keeps -inf, which the canonical
                # identity layer rejects fail-closed.  The search-result
                # invariant still requires the curve to start at consumed
                # budget 1 and stay non-decreasing, so a non-finite best
                # is recorded as the reward floor (bad_reward): finite,
                # and every scored reward clips to [clip_low, clip_high]
                # above it.
                reward_value = float(state.step_results[key].val_reward)
                if math.isfinite(reward_value):
                    run.best_reward = max(run.best_reward, reward_value)
                if not math.isfinite(run.best_reward):
                    run.best_reward = float(self.reward_config.bad_reward)
                run.best_so_far.append((budget_after_put, run.best_reward))
            else:
                # A same-batch numerical equivalent is only discovered
                # when the first result is committed to the ledger.  It
                # consumed compute but not semantic-evaluation budget.
                state.semantic_duplicate_proposals += 1

    def _resolve_batch_rewards(
        self,
        sequences: torch.Tensor,
        batch_size: int,
        state: _StepState,
        rewards: torch.Tensor,
    ) -> list[CandidateScore]:
        """Fill the per-row in-sample rewards (the gradient reads the
        in-sample reward only: the validation tail ranks formulas but
        never teaches the policy) and collect the batch's scores."""

        batch_scores: list[CandidateScore] = []
        for i in range(batch_size):
            key = tuple(sequences[i].tolist())
            score = state.step_results.get(key)
            if score is None:
                # A row that shared its canonical form's evaluation this
                # step resolves through the first key of that form.
                ckey = self.semantic_cache.key_for(key)
                if ckey is not None:
                    first = state.canonical_share.get(ckey)
                    if first is not None:
                        score = state.step_results.get(first)
            if score is None:
                raise RuntimeError(
                    f"no score recorded for sampled formula {key}"
                )
            rewards[i] = score.train_reward
            batch_scores.append(score)
            self._candidate_scores[key] = score
        return batch_scores

    def _update_selection_and_track(
        self, state: _StepState, batch_size: int, pbar
    ) -> float:
        """Refresh the pareto selection from the cumulative scores, log a
        new eligible best, and advance the policy-collapse monitor.
        Returns the step's unique-formula fraction."""

        previous_key = (
            self.selection_result.selected.deterministic_key
            if self.selection_result.selected
            else None
        )
        self.selection_result = self.candidate_selector.select(
            self._candidate_scores.values(),
            pareto_objectives=PARETO_OBJECTIVES,
        )
        self._sync_best_from_selection()
        selected = self.selection_result.selected
        if selected is not None and selected.deterministic_key != previous_key:
            pbar.write(
                "[+] New eligible best: "
                f"val_reward={selected.val_reward:.3f} "
                f"val_icir={selected.val_icir:.3f} "
                f"formula={selected.formula_text}"
            )

        # Policy-collapse monitoring: when the batch keeps re-sampling
        # the same few formulas the REINFORCE search has collapsed.
        unique_frac = len(state.step_results) / max(batch_size, 1)
        collapse_fraction = self.model_config.collapse_warn_fraction
        if unique_frac < collapse_fraction:
            self._collapse_streak += 1
            if self._collapse_streak == self.model_config.collapse_warn_steps:
                _facade_logger().warning(
                    "policy sampling collapsed: "
                    f"{self._collapse_streak} consecutive steps with "
                    f"unique-formula fraction below {collapse_fraction} "
                    f"(latest {unique_frac:.3f})"
                )
        else:
            self._collapse_streak = 0
        return unique_frac

    def _grammar_step_stats(
        self, state: _StepState, batch_size: int, unique_frac: float
    ) -> dict[str, object]:
        """Grammar observability: formula length, operator coverage and
        the syntax/semantic unique rates are derived from the AST (the
        single source of truth); the cache-hit rate measures how much of
        the batch reused cross-step evaluations."""

        lengths = [length for _, length, _ in state.executed_stats]
        step_op_coverage: set[str] = set()
        for _, _, ops in state.executed_stats:
            step_op_coverage |= ops
        mean_formula_len = (
            float(sum(lengths)) / len(lengths) if lengths else 0.0
        )
        semantic_forms = {
            state.step_results[key].formula_text
            for key, _, _ in state.executed_stats
        }
        semantic_unique_frac = len(semantic_forms) / max(batch_size, 1)
        cache_hit_frac = state.cache_hits / max(batch_size, 1)
        semantic_dedup_frac = state.semantic_dedups / max(batch_size, 1)
        unique_semantic_evals = self.semantic_cache.budget_used
        self._run_operator_coverage |= step_op_coverage
        return {
            "unique_frac": unique_frac,
            "lengths": lengths,
            "step_op_coverage": step_op_coverage,
            "mean_formula_len": mean_formula_len,
            "semantic_unique_frac": semantic_unique_frac,
            "cache_hit_frac": cache_hit_frac,
            "semantic_dedup_frac": semantic_dedup_frac,
            "unique_semantic_evals": unique_semantic_evals,
        }

    def _policy_update(
        self,
        log_probs: list[torch.Tensor],
        rewards: torch.Tensor,
        values: list[torch.Tensor],
        entropies: list[torch.Tensor],
        batch_scores: list[CandidateScore],
        lengths: list[int],
        step_op_coverage: set[str],
        semantic_duplicate_proposals: int,
    ):
        """Actor-critic update: REINFORCE with the learned value as
        baseline, advantage clipping, and an entropy bonus; diagnostics
        are summarized before ``optimizer.step()``."""

        advantages = self._normalized_advantages(
            rewards,
            values[-1],
            float(self.model_config.advantage_clip),
        )
        loss, _, value_loss, entropy = self._policy_update_loss(
            log_probs,
            rewards,
            values[-1],
            entropies,
            value_loss_weight=self.model_config.value_loss_weight,
            advantage_clip=float(self.model_config.advantage_clip),
            entropy_coef=float(self.model_config.entropy_coef),
        )
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = gradient_l2_norm(self.model.parameters())
        step_diagnostics = summarize_rl_step(
            rewards=rewards,
            advantages=advantages,
            entropy=float(entropy.detach()),
            gradient_norm=gradient_norm,
            scores=batch_scores,
            formula_lengths=lengths,
            operator_names=step_op_coverage,
            semantic_duplicates=semantic_duplicate_proposals,
            proposal_count=len(batch_scores),
        )
        self.optimizer.step()
        return step_diagnostics, loss, value_loss, entropy

    def _record_step(
        self,
        step: int,
        pbar,
        rewards: torch.Tensor,
        loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy: torch.Tensor,
        step_diagnostics: dict[str, object],
        stats: dict[str, object],
        semantic_duplicate_proposals: int,
        run: _RLRunState,
    ) -> None:
        """Extend the run ledger and the per-step history, and emit the
        step's metric log and progress-bar postfix."""

        batch_size = len(rewards)
        run.reward_values.extend(
            float(value) for value in rewards.detach().cpu()
        )
        for reason, count in step_diagnostics["rejection_reasons"].items():
            run.rejection_reasons[reason] = (
                run.rejection_reasons.get(reason, 0) + int(count)
            )
        run.formula_lengths.extend(stats["lengths"])
        run.step_summaries.append(step_diagnostics)
        run.semantic_duplicates += semantic_duplicate_proposals

        self.history.append(
            {
                "step": float(step),
                "avg_reward": float(rewards.mean()),
                "best_val_reward": float(self.best_val_reward),
                "loss": float(loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                # Syntax unique rate: distinct token sequences / batch.
                "unique_frac": float(stats["unique_frac"]),
                # Grammar stats (T0-02): formula length, operator
                # coverage, semantic unique rate, cache hit rate.
                "mean_formula_len": stats["mean_formula_len"],
                "op_coverage": float(len(stats["step_op_coverage"])),
                "semantic_unique_frac": stats["semantic_unique_frac"],
                "cache_hit_frac": stats["cache_hit_frac"],
                # T2-01 budget ledger: cumulative unique semantic
                # evaluations and this step's semantic-dedup rate.
                "unique_semantic_evals": float(
                    stats["unique_semantic_evals"]
                ),
                "semantic_dedup_frac": stats["semantic_dedup_frac"],
                "semantic_duplicate_count": semantic_duplicate_proposals,
                "semantic_duplicate_rate": float(
                    semantic_duplicate_proposals / max(batch_size, 1)
                ),
                "rl_diagnostics": step_diagnostics,
            }
        )
        _facade_logger().info(
            "rl.metrics step={} reward_mean={} reward_std={} entropy={} "
            "advantage_variance={} gradient_norm={} semantic_duplicates={} "
            "operator_coverage={}",
            step,
            step_diagnostics["reward_distribution"]["mean"],
            step_diagnostics["reward_distribution"]["std"],
            step_diagnostics["entropy"],
            step_diagnostics["advantage_variance"],
            step_diagnostics["gradient_norm"],
            step_diagnostics["semantic_duplicates"],
            len(step_diagnostics["operator_coverage"]),
        )
        pbar.set_postfix(
            {
                "avg_reward": f"{rewards.mean().item():.3f}",
                "best": f"{self.best_val_reward:.3f}",
                "len": f"{stats['mean_formula_len']:.1f}",
                "ops": len(stats["step_op_coverage"]),
                "sem": f"{stats['semantic_unique_frac']:.2f}",
                "cache": f"{stats['cache_hit_frac']:.2f}",
                "budget": stats["unique_semantic_evals"],
            }
        )

    def _finish_training(
        self,
        *,
        contract,
        run: _RLRunState,
        seed: int,
        steps: int,
        batch_size: int,
        vm_device: torch.device,
        save_artifacts: bool,
    ) -> list[int] | None:
        """Post-loop tail of the historical ``train``: the bare-factor
        hard-fail, the run grammar log, the ``SearchResult`` shaping, the
        selection JSON and the artifact write."""

        # Hard fail on zero operator coverage: a run whose executed
        # formulas never used an operator is bare-factor screening, not
        # formula search — no artifact may be produced from it.
        if not self._run_operator_coverage:
            raise RuntimeError(
                "operator coverage is zero across the whole run: no executed "
                "formula used any operator (bare-factor screening only); "
                "training is invalid"
            )
        _facade_logger().info(
            "grammar stats (run): mean_formula_len={:.2f}, "
            "operator_coverage={}/39, unique_steps={}",
            float(np.mean([h["mean_formula_len"] for h in self.history])),
            len(self._run_operator_coverage),
            len(self.history),
        )

        self.search_result = self._build_rl_search_result(
            seed=seed,
            requested_budget=int(steps) * int(batch_size),
            proposal_count=int(steps) * int(batch_size),
            reward_values=run.reward_values,
            step_summaries=run.step_summaries,
            run_rejection_reasons=run.rejection_reasons,
            formula_lengths=run.formula_lengths,
            semantic_duplicates=run.semantic_duplicates,
            best_so_far=run.best_so_far,
        )

        selection_path = self.data_config.data_dir / "training_selection.json"
        if save_artifacts:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_payload = {
                "reward_version": REWARD_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                **execution_provenance(self.backtest_config),
                "semantic_cache_version": SEMANTIC_CACHE_VERSION,
                "dataset_id": self.loader.dataset_id,
                "search_result": (
                    self.search_result.to_dict() if self.search_result else None
                ),
                "train_end": contract.train_end,
                "train_anchor_end_exclusive": contract.train_anchor_end_exclusive,
                "train_signal_start": contract.train_signal_start,
                "train_signal_end": contract.train_signal_end,
                "train_label_end": contract.train_label_end,
                **self.selection_result.to_dict(compact=True),
            }
            selection_path.write_text(
                json.dumps(selection_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        selected = self.selection_result.selected
        if selected is None:
            rejected = self.selection_result.best_rejected
            rejected_detail = (
                f" best rejected={rejected.formula_text!r} "
                f"reasons={list(rejected.rejection_reasons)}"
                if rejected is not None
                else " no candidates were scored"
            )
            _facade_logger().warning(
                "No eligible formula met every validation gate;"
                f"{rejected_detail}. No strategy artifact is written."
            )
            return None

        if not save_artifacts:
            _facade_logger().success(
                f"Training complete (artifacts skipped); "
                f"val_reward={selected.val_reward:.3f} "
                f"val_icir={selected.val_icir:.3f} "
                f"direction={self.best_direction} "
                f"formula={self.best_formula}"
            )
            return self.best_tokens

        return self._write_artifact(
            contract=contract,
            vm_device=vm_device,
            searcher="rl",
            seed=seed,
            requested_budget=int(steps) * int(batch_size),
        )

    def _build_rl_search_result(
        self,
        *,
        seed: int,
        requested_budget: int,
        proposal_count: int,
        reward_values: list[float],
        step_summaries: list[dict[str, object]],
        run_rejection_reasons: dict[str, int],
        formula_lengths: list[int],
        semantic_duplicates: int,
        best_so_far: list[tuple[int, float]],
    ) -> SearchResult:
        """Shape the legacy REINFORCE loop into the shared result schema.

        The curve is sampled at RL step boundaries but its x-axis is the
        same cumulative unique-semantic budget used by every other backend.
        Repeated x coordinates (a fully duplicate step) collapse without
        inventing budget consumption.
        """

        scores = tuple(self._candidate_scores.values())
        rejections: dict[str, int] = {}
        for score in scores:
            for reason in score.rejection_reasons:
                rejections[reason] = rejections.get(reason, 0) + 1
        consumed_budget = int(self.semantic_cache.budget_used)
        if self.selection_result.selected is None:
            termination_reason = "no_eligible_candidate"
        elif consumed_budget >= requested_budget:
            termination_reason = "budget_exhausted"
        else:
            termination_reason = "steps_exhausted"
        run_diagnostics = aggregate_rl_run(
            reward_values=reward_values,
            step_summaries=step_summaries,
            rejection_reasons=run_rejection_reasons,
            formula_lengths=formula_lengths,
            operator_names={
                name
                for summary in step_summaries
                for name in summary["operator_coverage"]
            },
            semantic_duplicates=semantic_duplicates,
            proposal_count=proposal_count,
        )
        run_diagnostics["initialization"] = self.rl_initialization
        run_diagnostics["imitation"] = (
            self.imitation_result.to_dict()
            if self.imitation_result is not None
            else None
        )
        return SearchResult(
            backend="rl",
            seed=int(seed),
            requested_budget=int(requested_budget),
            consumed_budget=consumed_budget,
            termination_reason=termination_reason,
            best_so_far=tuple(best_so_far),
            scores=scores,
            selected=self.selection_result.selected,
            rejection_reasons=rejections,
            proposal_count=int(proposal_count),
            invalid_proposals=len(self._invalid_cache),
            semantic_duplicates=semantic_duplicates,
            diagnostics=run_diagnostics,
        )

    @staticmethod
    def _normalized_advantages(
        rewards: torch.Tensor,
        baseline: torch.Tensor,
        advantage_clip: float,
    ) -> torch.Tensor:
        """The single advantage definition used by both loss and metrics."""

        advantages = (rewards - baseline.detach()) / (
            rewards.std(unbiased=False) + 1e-6
        )
        return advantages.clamp(-float(advantage_clip), float(advantage_clip))

    @staticmethod
    def _policy_update_loss(
        log_probs: list[torch.Tensor],
        rewards: torch.Tensor,
        baseline: torch.Tensor,
        entropies: list[torch.Tensor],
        *,
        value_loss_weight: float,
        advantage_clip: float,
        entropy_coef: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Actor-critic loss: REINFORCE + value baseline + entropy bonus.

        Advantages are batch-normalized and clipped to
        ``[-advantage_clip, advantage_clip]`` so a degenerate reward spread
        (e.g. a fully converged batch) cannot explode the policy gradient;
        the entropy bonus counteracts mode collapse.  Pure function of its
        inputs so the numerics are unit-testable.
        """

        adv = RLTrainingLoopMixin._normalized_advantages(
            rewards,
            baseline,
            advantage_clip,
        )
        policy_loss = -(
            torch.stack(log_probs, dim=1).sum(dim=1) * adv
        ).mean()
        entropy = torch.stack(entropies, dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(baseline, rewards)
        loss = policy_loss + value_loss_weight * value_loss - entropy_coef * entropy
        return loss, policy_loss, value_loss, entropy

    @staticmethod
    def _update_stack_state(
        action: torch.Tensor,
        stack_sizes: torch.Tensor,
        stack_types: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Advance the sampling state by one sampled action.

        Postfix rules (see :mod:`ashare_model.ir`): a feature pushes one
        value, an operator of arity ``a`` pops ``a`` and pushes one
        (``stack - a + 1``), EOS terminates at ``stack == 1``, and PAD is
        only ever sampled after EOS.  ``done`` latches once EOS (or a
        legacy padding termination) is sampled.

        P7-E: ``stack_types`` ([batch, capacity] long tensor, 0 = empty)
        advances in lockstep — features push their semantic-type id,
        operators pop their arguments and push the resolved output id
        (rules from :mod:`ashare_model.semantic_sampling`, single source).
        The caller's mask guarantees the operator's signature was legal,
        so the output resolution here never sees an illegal application.
        """

        advance_stack_state(
            action,
            stack_sizes,
            stack_types,
            done,
            vocab=FORMULA_VOCAB,
        )

    def _sync_best_from_selection(self) -> None:
        selected = self.selection_result.selected
        if selected is None:
            self.best_val_reward = -float("inf")
            self.best_val_icir = -float("inf")
            self.best_train_reward = -float("inf")
            self.best_train_icir = -float("inf")
            self.best_direction = 1
            self.best_tokens = None
            self.best_formula = ""
            return
        self.best_val_reward = selected.val_reward
        self.best_val_icir = selected.val_icir
        self.best_train_reward = selected.train_reward
        self.best_train_icir = selected.train_icir
        self.best_direction = selected.direction
        self.best_tokens = list(selected.tokens) if selected.tokens is not None else None
        self.best_formula = selected.formula_text
