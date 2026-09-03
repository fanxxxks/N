"""REINFORCE trainer for A-share factor formulas.

Usage:
    python -m ashare_model.train [--config config/ashare_config.yaml]
                                 [--steps N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.distributions import Categorical
from tqdm import tqdm

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
from ashare_data.processor import open_to_open_returns

from .alphagpt import MODEL_VERSION, AlphaGPTModel, build_action_mask
from .artifact_schemas import StrategyArtifact
from .candidates import (
    PARETO_OBJECTIVES,
    CandidateScore,
    CandidateScorer,
    CandidateSelector,
    CandidateSpec,
    SelectionResult,
    score_chunk_size,
)
from .complexity import complexity_bill
from .data_loader import AshareDataLoader
from .data_tier import formula_data_tier_report
from .reward import (
    REWARD_VERSION,
    batched_basket_rewards,
)
from .research_domain import RESEARCH_DOMAIN_VERSION
from .semantic_sampling import advance_stack_state
from .rl_diagnostics import (
    aggregate_rl_run,
    gradient_l2_norm,
    summarize_rl_step,
)
from .targets import causal_target_returns
from .semantic_cache import (
    SEMANTIC_CACHE_VERSION,
    CalibrationSlice,
    SemanticCache,
    SemanticCacheKey,
    make_calibration_execute,
)
from .search_backends import (
    get_search_backend,
    log_search_start,
    log_search_stop,
)
from .search_contract import SearchRequest, SearchResult
from .time_contract import TrainingTimeContract
from .train_windows import (
    _TrainWindow,
    _project_root,
    resolve_device,
    sample_random_formulas,
    validation_start,
    validation_windows,
)
from .vm import StackVM, formula_decode
from .vocab import FORMULA_VOCAB, GRAMMAR_VERSION
from .versions import PROTOCOL_VERSION
from ashare_portfolio.rebalance import RebalancePolicy
from ashare_portfolio.execution_spec import execution_provenance
from . import ir as ir_module
from ashare_logging import export_log_txt, setup_run_logging

class AshareTrainer:
    # Bounded LRU of evaluated formulas: rewards are a deterministic function
    # of the token sequence, so repeats (frequent once the policy
    # concentrates) are scored once and reused across steps.
    _REWARD_CACHE_CAP = 16384

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
        # .md §4.2).
        domain_id: str = "unified",
        feature_ids: list[int] | None = None,
    ):
        self.data_config = data_config
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

        adv = AshareTrainer._normalized_advantages(
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
        max_len = self.model_config.max_formula_len
        vm_device = resolve_device(device)

        window = self.prepare_window(train_end_date, vm_device, window_cap)
        contract = window.contract
        train_end_idx = window.train_end_idx
        val_windows = window.val_windows
        val_start = window.val_start
        train_signal_range = window.train_signal_range
        factor_tensor = window.factor_tensor
        train_universe_mask = window.train_universe_mask
        target_ret = window.target_ret
        realized_ret = window.realized_ret
        rebalance_mask = window.rebalance_mask
        blocked_buy = window.blocked_buy
        blocked_sell = window.blocked_sell
        reward_chunk = window.reward_chunk
        # Selection tie-break keys and the capacity-audit dollar volume are
        # sliced to the measured window (a capped admission window is a
        # stock slice of the loader's universe).
        tie_break_keys = np.asarray(self.loader.ts_codes)[
            : factor_tensor.shape[1]
        ]
        adv = np.asarray(self.loader.dollar_volume())[
            : factor_tensor.shape[1], : target_ret.shape[1]
        ]

        run_reward_values: list[float] = []
        run_rejection_reasons: dict[str, int] = {}
        run_formula_lengths: list[int] = []
        run_step_summaries: list[dict[str, object]] = []
        run_semantic_duplicates = 0
        run_best_so_far: list[tuple[int, float]] = []
        run_best_reward = -float("inf")

        pbar = tqdm(range(steps))
        for step in pbar:
            # The policy and sampling stay on CPU (the model's device): its
            # RNG stream and dropout are device-independent by construction.
            policy_device = next(self.model.parameters()).device
            inp = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=policy_device
            )
            stack_sizes = torch.zeros(
                batch_size, dtype=torch.long, device=policy_device
            )
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
                self._update_stack_state(
                    action, stack_sizes, stack_types, done
                )

            sequences = torch.stack(sampled_tokens, dim=1)
            rewards = torch.zeros(batch_size, device=policy_device)

            # Batched reward evaluation with semantic deduplication (T2-01):
            # the budget unit is the **unique semantic formula evaluation**.
            # Structurally invalid / degenerate formulas are rejected
            # pre-evaluation (never executed, never billed); canonical
            # duplicates hit the semantic cache; numerically equivalent
            # formulas (same calibration fingerprint) reuse the first
            # evaluation's score without billing.  Valid signals are scored
            # in chunks so the vectorized basket simulation stays
            # memory-bounded.
            step_results: dict[tuple[int, ...], CandidateScore] = {}
            pending: list[tuple[CandidateSpec, np.ndarray]] = []
            # evaluated: list of (token key, canonical cache key,
            # fingerprint, complexity bill).
            evaluated: list[
                tuple[tuple[int, ...], SemanticCacheKey, str | None, float | None]
            ] = []
            # canonical_share maps each canonical form to the first token
            # key that handled it this step: identical sequences and
            # canonical duplicates (e.g. ADD(a, b) vs ADD(b, a)) share that
            # row's evaluation — no repeated work, no repeated billing.
            canonical_share: dict[SemanticCacheKey, tuple[int, ...]] = {}
            cache_hits = 0
            semantic_dedups = 0
            semantic_duplicate_proposals = 0
            # Grammar stats are accumulated at execution time, so only
            # formulas the VM actually executed and produced a signal for
            # count toward length/operator coverage (a formula the VM
            # rejected is structurally invalid — never an operator user).
            executed_stats: list[tuple[tuple[int, ...], int, set[str]]] = []
            for i in range(batch_size):
                key = tuple(sequences[i].tolist())
                ckey = self.semantic_cache.key_for(key)
                if ckey is None:
                    # Structurally invalid or degenerate (constant-
                    # producing): rejected pre-evaluation, cached by token
                    # sequence so repeats skip the rejection path.
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
                            target_ret,
                            val_windows,
                            blocked_buy=blocked_buy,
                            blocked_sell=blocked_sell,
                            formula_valid=False,
                            train_signal_range=train_signal_range,
                            universe_mask=train_universe_mask,
                        )
                        self._invalid_cache[key] = score
                        while len(self._invalid_cache) > self._REWARD_CACHE_CAP:
                            self._invalid_cache.popitem(last=False)
                    step_results[key] = score
                    continue
                first = canonical_share.get(ckey)
                if first is not None:
                    # Same canonical formula already handled this step
                    # (identical tokens or a commuted form): share its
                    # evaluation; no new work, no budget.
                    semantic_duplicate_proposals += 1
                    continue
                canonical_share[ckey] = key
                score = self.semantic_cache.get(ckey)
                if score is not None:
                    cache_hits += 1
                    semantic_duplicate_proposals += 1
                    step_results[key] = score
                    continue
                # A new canonical formula: check semantic equivalence on the
                # calibration slice before paying for a full evaluation.
                # The semantic class is (fingerprint, complexity bill): two
                # numerically equivalent formulas with different bills carry
                # different penalties, so their scores are not
                # interchangeable.
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
                    semantic_dedups += 1
                    semantic_duplicate_proposals += 1
                    self.semantic_cache.put(ckey, score, fingerprint, bill)
                    step_results[key] = score
                    continue
                spec = CandidateSpec(
                    candidate_id="rl:" + ",".join(str(token) for token in key),
                    formula_text=formula_decode(list(key), self.vocab),
                    source="rl",
                    tokens=key,
                )
                signal = self.vm.execute(key, factor_tensor)
                if signal is None:
                    step_results[key] = self.candidate_scorer.score(
                        spec,
                        None,
                        target_ret,
                        val_windows,
                        blocked_buy=blocked_buy,
                        blocked_sell=blocked_sell,
                        formula_valid=False,
                        train_signal_range=train_signal_range,
                        universe_mask=train_universe_mask,
                    )
                    continue
                ast = ir_module.decode(key, self.vocab)
                executed_stats.append(
                    (
                        key,
                        ir_module.formula_length(key, self.vocab),
                        ir_module.operator_names(ast),
                    )
                )
                signal_np = signal.detach().cpu().numpy()
                pending.append((spec, signal_np))
                evaluated.append((key, ckey, fingerprint, bill))
                # Score as soon as one full chunk is ready: buffering every
                # unique signal of a step at once costs batch_size x
                # [stocks, dates] x 4 bytes and exhausts RAM on 16 GB
                # machines for large batches before any chunk is scored.
                if len(pending) >= reward_chunk:
                    self._score_pending_chunk(
                        pending,
                        target_ret,
                        realized_ret,
                        rebalance_mask,
                        val_windows,
                        step_results,
                        train_universe_mask,
                        blocked_buy,
                        blocked_sell,
                        train_signal_range,
                        tie_break_keys,
                        adv,
                    )
                    pending = []
            self._score_pending_chunk(
                pending,
                target_ret,
                realized_ret,
                rebalance_mask,
                val_windows,
                step_results,
                train_universe_mask,
                blocked_buy,
                blocked_sell,
                train_signal_range,
                tie_break_keys,
                adv,
            )
            for key, ckey, fingerprint, bill in evaluated:
                budget_before_put = int(self.semantic_cache.budget_used)
                self.semantic_cache.put(ckey, step_results[key], fingerprint, bill)
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
                    reward_value = float(step_results[key].val_reward)
                    if math.isfinite(reward_value):
                        run_best_reward = max(run_best_reward, reward_value)
                    if not math.isfinite(run_best_reward):
                        run_best_reward = float(self.reward_config.bad_reward)
                    run_best_so_far.append((budget_after_put, run_best_reward))
                else:
                    # A same-batch numerical equivalent is only discovered
                    # when the first result is committed to the ledger.  It
                    # consumed compute but not semantic-evaluation budget.
                    semantic_duplicate_proposals += 1

            batch_scores: list[CandidateScore] = []
            for i in range(batch_size):
                key = tuple(sequences[i].tolist())
                score = step_results.get(key)
                if score is None:
                    # A row that shared its canonical form's evaluation this
                    # step resolves through the first key of that form.
                    ckey = self.semantic_cache.key_for(key)
                    if ckey is not None:
                        first = canonical_share.get(ckey)
                        if first is not None:
                            score = step_results.get(first)
                if score is None:
                    raise RuntimeError(
                        f"no score recorded for sampled formula {key}"
                    )
                # The gradient reads the in-sample reward only: the
                # validation tail ranks formulas but never teaches the
                # policy (test_policy_gradient_reward_window_excludes_
                # validation_tail pins this contract).
                rewards[i] = score.train_reward
                batch_scores.append(score)
                self._candidate_scores[key] = score

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
            unique_frac = len(step_results) / max(batch_size, 1)
            collapse_fraction = self.model_config.collapse_warn_fraction
            if unique_frac < collapse_fraction:
                self._collapse_streak += 1
                if self._collapse_streak == self.model_config.collapse_warn_steps:
                    logger.warning(
                        "policy sampling collapsed: "
                        f"{self._collapse_streak} consecutive steps with "
                        f"unique-formula fraction below {collapse_fraction} "
                        f"(latest {unique_frac:.3f})"
                    )
            else:
                self._collapse_streak = 0

            # Grammar observability: formula length, operator coverage and
            # the syntax/semantic unique rates are derived from the AST
            # (the single source of truth); the cache-hit rate measures how
            # much of the batch reused cross-step evaluations.
            lengths = [length for _, length, _ in executed_stats]
            step_op_coverage: set[str] = set()
            for _, _, ops in executed_stats:
                step_op_coverage |= ops
            mean_formula_len = (
                float(sum(lengths)) / len(lengths) if lengths else 0.0
            )
            semantic_forms = {
                step_results[key].formula_text for key, _, _ in executed_stats
            }
            semantic_unique_frac = len(semantic_forms) / max(batch_size, 1)
            cache_hit_frac = cache_hits / max(batch_size, 1)
            semantic_dedup_frac = semantic_dedups / max(batch_size, 1)
            unique_semantic_evals = self.semantic_cache.budget_used
            self._run_operator_coverage |= step_op_coverage

            # Actor-critic update: REINFORCE with the learned value as
            # baseline, advantage clipping, and an entropy bonus.
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
                proposal_count=batch_size,
            )
            self.optimizer.step()

            run_reward_values.extend(float(value) for value in rewards.detach().cpu())
            for reason, count in step_diagnostics["rejection_reasons"].items():
                run_rejection_reasons[reason] = (
                    run_rejection_reasons.get(reason, 0) + int(count)
                )
            run_formula_lengths.extend(lengths)
            run_step_summaries.append(step_diagnostics)
            run_semantic_duplicates += semantic_duplicate_proposals

            self.history.append(
                {
                    "step": float(step),
                    "avg_reward": float(rewards.mean()),
                    "best_val_reward": float(self.best_val_reward),
                    "loss": float(loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    # Syntax unique rate: distinct token sequences / batch.
                    "unique_frac": float(unique_frac),
                    # Grammar stats (T0-02): formula length, operator
                    # coverage, semantic unique rate, cache hit rate.
                    "mean_formula_len": mean_formula_len,
                    "op_coverage": float(len(step_op_coverage)),
                    "semantic_unique_frac": semantic_unique_frac,
                    "cache_hit_frac": cache_hit_frac,
                    # T2-01 budget ledger: cumulative unique semantic
                    # evaluations and this step's semantic-dedup rate.
                    "unique_semantic_evals": float(unique_semantic_evals),
                    "semantic_dedup_frac": semantic_dedup_frac,
                    "semantic_duplicate_count": semantic_duplicate_proposals,
                    "semantic_duplicate_rate": float(
                        semantic_duplicate_proposals / max(batch_size, 1)
                    ),
                    "rl_diagnostics": step_diagnostics,
                }
            )
            logger.info(
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
                    "len": f"{mean_formula_len:.1f}",
                    "ops": len(step_op_coverage),
                    "sem": f"{semantic_unique_frac:.2f}",
                    "cache": f"{cache_hit_frac:.2f}",
                    "budget": unique_semantic_evals,
                }
            )

        # Hard fail on zero operator coverage: a run whose executed
        # formulas never used an operator is bare-factor screening, not
        # formula search — no artifact may be produced from it.
        if not self._run_operator_coverage:
            raise RuntimeError(
                "operator coverage is zero across the whole run: no executed "
                "formula used any operator (bare-factor screening only); "
                "training is invalid"
            )
        logger.info(
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
            reward_values=run_reward_values,
            step_summaries=run_step_summaries,
            run_rejection_reasons=run_rejection_reasons,
            formula_lengths=run_formula_lengths,
            semantic_duplicates=run_semantic_duplicates,
            best_so_far=run_best_so_far,
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
            logger.warning(
                "No eligible formula met every validation gate;"
                f"{rejected_detail}. No strategy artifact is written."
            )
            return None

        if not save_artifacts:
            logger.success(
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

    def _write_artifact(
        self,
        *,
        contract: TrainingTimeContract,
        vm_device: torch.device,
        searcher: str = "rl",
        seed: int,
        requested_budget: int,
    ) -> list[int] | None:
        """Write the standard training artifact (selection + strategy JSON +
        the policy checkpoint for RL runs) for the current selection.

        P8-05: the strategy artifact is a lifecycle-bound boundary artifact —
        the run resolves its frozen RunSpec, opens a RunStore run and
        persists through :func:`write_boundary_artifact` (content-addressed,
        fail-closed identity, atomic convenience mirror).  The strategy
        JSON stays at its historical path as the display mirror.
        """

        selected = self.selection_result.selected
        if selected is None:
            return None
        # ``evaluation`` imports this module at module level; resolve lazily.
        from .artifact_writer import write_boundary_artifact  # noqa: PLC0415
        from .identity import candidate_id  # noqa: PLC0415
        from .run_store import RunStore  # noqa: PLC0415
        from .runspec import resolve_runtime_runspec  # noqa: PLC0415

        score_payload = selected.to_dict()
        score_payload.pop("tokens", None)
        # P8-05: the searcher-internal candidate label is a diagnostic only;
        # the lifecycle candidate identity (identity.candidate_id over
        # spec_id + tokens + direction) is stamped by the formal writer.
        searcher_candidate_label = score_payload.pop("candidate_id", None)
        output = {
            "formula": self.best_tokens,
            **score_payload,
            "searcher_candidate_label": searcher_candidate_label,
            "history": self.history,
            "searcher": searcher,
            # Reproducibility provenance: the policy stays on CPU, so
            # (init_seed, seed) reproduce the same sampled formulas on any
            # machine; ``device`` records where the VM executed.
            "init_seed": self.init_seed,
            "device": str(vm_device),
            "model_version": MODEL_VERSION,
            # Vocabulary provenance: the formula is always remapped by name
            # on load, so later vocabulary additions cannot silently
            # reinterpret these token ids.
            "feature_names": list(self.vocab.feature_names),
            "operator_names": list(self.vocab.operator_names),
            "feature_version": self.vocab.feature_version,
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
            "research_domain": self.domain_id,
            "research_domain_version": RESEARCH_DOMAIN_VERSION,
            **execution_provenance(self.backtest_config),
            "semantic_cache_version": SEMANTIC_CACHE_VERSION,
            "unique_semantic_evals": self.semantic_cache.budget_used,
            "semantic_cache_stats": self.semantic_cache.stats(),
            "search_contract_version": (
                self.search_result.contract_version if self.search_result else None
            ),
            "search_result": (
                self.search_result.to_dict() if self.search_result else None
            ),
            # P2-02: the strategy formula traces back to the credibility
            # tiers of its features (``None`` when nothing is traceable).
            "data_tier": formula_data_tier_report(tokens=self.best_tokens),
            # Data provenance: the immutable dataset manifest this formula
            # was selected on (None for pre-T1-01 databases).
            "dataset_id": self.loader.dataset_id,
        }
        if searcher == "rl":
            output.update(
                {
                    "rl_initialization": self.rl_initialization,
                    "imitation": (
                        self.imitation_result.to_dict()
                        if self.imitation_result is not None
                        else None
                    ),
                    "experimental": self.rl_initialization == "random",
                }
            )
        out_path = self.data_config.data_dir / "best_ashare_strategy.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # P8-05: resolve the frozen RunSpec for this run and persist through
        # the RunStore-bound formal writer.  A formal strategy artifact
        # requires a resolved dataset identity (T1-01 manifest); a legacy
        # database without one is refused here, fail-closed.
        if not self.loader.dataset_id:
            from .artifact_schemas import ArtifactSchemaError  # noqa: PLC0415

            raise ArtifactSchemaError(
                "formal strategy artifacts require a resolved dataset_id "
                "(dataset manifest); migrate the database with "
                "`python -m ashare_data.manifest` before training"
            )
        spec = resolve_runtime_runspec(
            dataset_id=self.loader.dataset_id,
            data_cutoff=self.loader.dates[-1],
            data_config=self.data_config,
            backtest_config=self.backtest_config,
            requested_budget=int(requested_budget),
            # Training evaluates a single validation tail; the campaign's
            # fold plan is carried by the protocol run's own spec.
            n_folds=1,
            research_domain=self.domain_id,
            seeds=tuple(sorted({int(self.init_seed), int(seed)})),
            searcher=searcher,
            max_formula_len=int(self.model_config.max_formula_len),
            # The clean-tree SPEC_LOCKED evidence gate activates with the
            # lifecycle stages (P8-06+); the spec still records the exact
            # git commit and dependency-lock hash.
            require_clean_tree=False,
        )
        cand = candidate_id(spec.spec_id, self.best_tokens, output["direction"])
        if self.search_result is not None and self.search_result.elite_archive is not None:
            from .elite_archive import write_elite_archive  # noqa: PLC0415

            archive_path = write_elite_archive(
                self.data_config.data_dir / "search_elite_archive.json",
                self.search_result.elite_archive,
            )
            logger.info("search.elite_archive path={}", archive_path)
        store = RunStore(self.data_config.data_dir)
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
                self.model.state_dict(),
                self.data_config.data_dir / "ashare_model.pt",
            )
        logger.success(
            f"Search complete (searcher={searcher}); "
            f"best formula saved to {out_path}"
        )
        return self.best_tokens

    def _training_contract(
        self, train_end_date: str | None = None
    ) -> TrainingTimeContract:
        return TrainingTimeContract.resolve(
            self.loader.dates,
            train_end_date or self.backtest_config.train_end_date,
            horizon=self.backtest_config.target_horizon,
        )

    def prepare_window(
        self,
        train_end_date: str | None,
        vm_device: torch.device,
        window_cap: tuple[int, int] | None = None,
    ) -> "_TrainWindow":
        """Bind everything the searchers (RL, random, GP) need about the
        training window: the factor tensor, VM masks, target, windows, the
        semantic-budget cache and the calibration fingerprint executor.

        Shared by :meth:`train` (RL) and :meth:`train_search` (the
        non-RL backends) so every searcher measures the identical window.

        ``window_cap`` is a measurement override ``(stocks, dates)`` that
        slices the window head for tractable experiments (the admission
        tier): every searcher then measures the identical capped window,
        and the validation windows / signal range are re-derived inside
        the cap.  The cap is recorded by callers that persist results.
        """

        contract = self._training_contract(train_end_date)
        train_end_idx = contract.train_label_end
        stock_slice = slice(None)
        if window_cap is not None:
            stocks, dates = int(window_cap[0]), int(window_cap[1])
            if stocks <= 0 or dates <= 0:
                raise ValueError("window_cap must be positive (stocks, dates)")
            stock_slice = slice(0, stocks)
            train_end_idx = min(train_end_idx, dates)
        # The capped window's signal end reserves the exact entry+holding
        # context declared by the target contract.
        train_signal_end = min(
            contract.train_signal_end,
            max(0, train_end_idx - contract.exit_offset),
        )
        policy = RebalancePolicy.from_config(self.backtest_config)
        full_rebalance_mask = policy.rebalance_mask(self.loader.dates)
        rebalance_mask = full_rebalance_mask[:train_end_idx]
        # Hold out the tail of the training window for out-of-sample best
        # formula selection, split into independent sub-windows; the best
        # formula is decided on the *median* validation reward so a single
        # lucky tail stretch cannot win the selection.
        val_windows = self._validation_windows(
            train_signal_end,
            rebalance_mask=rebalance_mask,
        )
        # The *learning* window is the in-sample head only: the policy
        # gradient scores candidates on columns strictly before the first
        # validation window, so the selection data never doubles as the
        # training signal (val rewards rank formulas; IS rewards teach the
        # policy — two jobs, two windows).
        val_start = validation_start(train_signal_end, self.model_config)
        train_signal_range = (contract.train_signal_start, val_start)
        factor_tensor = self.loader.factor_tensor[
            :, stock_slice, :train_end_idx
        ].to(vm_device)
        # The VM executes on the compute device; the industry-group tensor
        # for CS_NEUTRALIZE and the PIT eligibility mask for the
        # cross-sectional operators move with the factor stack (the loader
        # always builds the mask in load_data, so there is no fallback).
        industry_codes = getattr(self.loader, "industry_codes", None)
        self.vm.industry_codes = (
            industry_codes[stock_slice, :train_end_idx].to(vm_device)
            if industry_codes is not None
            else None
        )
        universe_mask = self.loader.universe_mask
        self.vm.universe_mask = torch.tensor(
            universe_mask[stock_slice, :train_end_idx],
            dtype=torch.bool,
            device=vm_device,
        )
        # The candidate scorer needs the same PIT mask on the numpy side:
        # every quality statistic (rank IC, basket selection, near-constant
        # rejection, direction) is gated to signal-date eligible cells.
        train_universe_mask = universe_mask[stock_slice, :train_end_idx]
        # Recompute labels only from prices inside the inclusive training
        # anchor. Precomputed global targets at the tail may reference fold-
        # external t+1/t+2 prices and are intentionally never sliced here.
        train_open = self.loader.raw_data_cache["open"][
            stock_slice, :train_end_idx
        ].numpy()
        target_ret = causal_target_returns(
            train_open,
            self.loader.dates[:train_end_idx],
            policy,
            rebalance_mask=rebalance_mask,
        )
        realized_ret = open_to_open_returns(train_open)
        # Same semantics as loader.mask_by_universe, applied to the capped
        # window (the loader's mask is the full stock axis): values outside
        # the PIT eligibility mask become NaN.
        target_ret = np.asarray(target_ret, dtype=np.float64).copy()
        target_ret[~np.asarray(train_universe_mask, dtype=bool)] = np.nan
        # Tradability masks (buy/sell blocked per stock and date) align the
        # training basket with the backtest engine's execution rules; both
        # matrices are shared by every formula scored this run.
        blocked_buy, blocked_sell = self.loader.tradability_masks()
        blocked_buy = blocked_buy[stock_slice, :train_end_idx]
        blocked_sell = blocked_sell[stock_slice, :train_end_idx]

        # T2-01: the semantic cache is the evaluation-budget ledger.  Its
        # key carries the full evaluation context (dataset_id, reward and
        # protocol versions, the training window), so scores are never
        # reused across datasets or measurement generations, and its budget
        # counts **unique semantic evaluations** — structurally identical
        # (canonical AST hash) and numerically equivalent (calibration
        # fingerprint) formulas never bill twice.
        self.semantic_cache = SemanticCache(
            dataset_id=self.loader.dataset_id,
            reward_version=REWARD_VERSION,
            protocol_version=PROTOCOL_VERSION,
            window_id=self._window_id(
                contract, val_windows, policy, domain_id=self.domain_id
            ),
            cap=self._REWARD_CACHE_CAP,
        )
        calibration_slice = CalibrationSlice.of(factor_tensor.shape[2])
        self._calibration_execute = make_calibration_execute(
            self.vm,
            factor_tensor,
            universe_mask[stock_slice, :train_end_idx],
            self.vm.industry_codes,
            calibration_slice,
        )
        # Chunk the batched reward evaluation so the stacked signal matrix
        # (original stack + batched copy + both-direction copy) stays within
        # a fixed memory budget (~512 MB of float64 signals).
        signal_bytes = factor_tensor.shape[1] * train_end_idx * 8
        reward_chunk = score_chunk_size(signal_bytes)
        return _TrainWindow(
            contract=contract,
            train_end_idx=train_end_idx,
            val_windows=val_windows,
            val_start=val_start,
            train_signal_range=train_signal_range,
            factor_tensor=factor_tensor,
            vm_device=vm_device,
            train_universe_mask=train_universe_mask,
            target_ret=target_ret,
            realized_ret=realized_ret,
            rebalance_mask=rebalance_mask,
            blocked_buy=blocked_buy,
            blocked_sell=blocked_sell,
            reward_chunk=reward_chunk,
        )

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
                    from .elite_archive import load_elite_archive  # noqa: PLC0415

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

        from .imitation import pretrain_on_elites  # noqa: PLC0415

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
        logger.info(
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

        # Lazily imported: baseline_harness imports this module for uniform
        # random formula sampling, so the cycle is broken at call time.
        from .baseline_harness import SemanticBudgetEvaluator  # noqa: PLC0415

        def execute(tokens) -> np.ndarray | None:
            signal = self.vm.execute(tokens, window.factor_tensor)
            if signal is None:
                return None
            return signal.detach().cpu().numpy()

        evaluator = SemanticBudgetEvaluator(
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

        selected = result.selected
        if selected is None:
            logger.warning(
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
            logger.success(
                f"Search complete (artifacts skipped); searcher={searcher} "
                f"val_reward={selected.val_reward:.3f} "
                f"val_icir={selected.val_icir:.3f} "
                f"formula={selected.formula_text}"
            )
            return self.best_tokens
        return self._write_artifact(
            contract=window.contract,
            vm_device=vm_device,
            searcher=searcher,
            seed=seed,
            requested_budget=int(budget),
        )

    @staticmethod
    def _window_id(
        contract,
        val_windows: list[tuple[int, int]],
        policy: RebalancePolicy,
        domain_id: str = "unified",
    ) -> str:
        """Deterministic id of the training window: the exact columns the
        semantic-cache key binds scores to.  A non-unified research domain
        (P6 §4.3) appends its id so domain scores never mix with unified
        or other-domain scores."""

        val = "|".join(f"{a}:{b}" for a, b in val_windows)
        domain = (
            ""
            if str(domain_id) == "unified"
            else f":domain:{str(domain_id)}"
        )
        return (
            f"train:{contract.train_signal_start}:{contract.train_signal_end}:"
            f"label:{contract.train_label_end}:frequency:{policy.frequency}:"
            f"horizon:{policy.horizon}:val:{val}{domain}"
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

    @property
    def best_reward(self) -> float:
        """Deprecated read-only alias for the explicit validation reward."""

        return self.best_val_reward

    def _validation_start(self, train_end_idx: int) -> int:
        """First index of the validation tail inside the training window.

        Delegates to the module-level :func:`validation_start` so the
        protocol's random-search baseline can share the exact selection
        windows without instantiating a trainer.
        """
        return validation_start(train_end_idx, self.model_config)

    def _validation_windows(
        self,
        train_end_idx: int,
        *,
        rebalance_mask: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        """Independent validation sub-windows covering the training tail.

        Delegates to the module-level :func:`validation_windows`.
        """
        return validation_windows(
            train_end_idx,
            self.model_config,
            rebalance_mask=rebalance_mask,
        )

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
    args = parser.parse_args()

    try:
        root = _project_root()
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(data_config, min_eligible=args.min_eligible).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        loader = AshareDataLoader(data_config, model_config)
        trainer = AshareTrainer(
            data_config, model_config, backtest_config, loader, reward_config
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
