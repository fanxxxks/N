"""REINFORCE trainer for A-share factor formulas.

Usage:
    python -m ashare_model.train [--config config/ashare_config.yaml]
                                 [--steps N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
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
from .ops import OPS_CONFIG
from .reward import (
    REWARD_VERSION,
    batched_basket_rewards,
)
from .semantic_cache import (
    SEMANTIC_CACHE_VERSION,
    CalibrationSlice,
    SemanticCache,
    SemanticCacheKey,
    make_calibration_execute,
)
from .time_contract import TrainingTimeContract
from .vm import StackVM, formula_decode
from .vocab import FORMULA_VOCAB, GRAMMAR_VERSION
from . import ir as ir_module
from ashare_logging import export_log_txt, setup_run_logging

# Sampling-state tables, built once at import: the arity of every operator
# token and the feature-token id range, so the per-position stack update
# never rebuilds tensors inside the sampling loop (the policy and sampling
# always run on CPU, matching these tensors).
_OPERATOR_ARITY = torch.zeros(FORMULA_VOCAB.size)
for _i, (_, _, _arity) in enumerate(OPS_CONFIG):
    _OPERATOR_ARITY[FORMULA_VOCAB.operator_offset + _i] = _arity
_FEATURE_IDS = torch.arange(FORMULA_VOCAB.feature_offset, FORMULA_VOCAB.operator_offset)


@dataclass
class _TrainWindow:
    """One training window bound for every searcher (RL, random, GP): the
    tensors, masks and windows the candidate scorer consumes, plus the
    semantic-budget cache and the calibration fingerprint executor."""

    contract: "TrainingTimeContract"
    train_end_idx: int
    val_windows: list[tuple[int, int]]
    val_start: int
    train_signal_range: tuple[int, int]
    factor_tensor: torch.Tensor
    vm_device: torch.device
    train_universe_mask: np.ndarray
    target_ret: np.ndarray
    blocked_buy: np.ndarray
    blocked_sell: np.ndarray
    reward_chunk: int


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validation_start(train_signal_end: int, model_config) -> int:
    """First index of the validation tail inside the training window.

    The tail keeps at least two dates (a reward needs one daily return);
    windows too small to hold anything out validate on the full window.
    """

    if train_signal_end <= 2:
        return 0
    val_frac = float(np.clip(model_config.validation_fraction, 0.0, 0.5))
    val_start = int(round(train_signal_end * (1.0 - val_frac)))
    return max(1, min(val_start, train_signal_end - 2))


def validation_windows(
    train_signal_end: int, model_config
) -> list[tuple[int, int]]:
    """Independent validation sub-windows covering the training tail.

    The tail is split into ``model_config.validation_splits`` disjoint
    ``(start, end)`` column intervals of at least 3 columns each
    (a rank-ICIR needs a minimum number of cross-sections to be
    meaningful).  Tails too short to split degrade to a single window,
    preserving the historical single-holdout behavior.  Shared by the
    trainer and the protocol's random-search baseline so both score
    against the identical selection windows.
    """

    val_start = validation_start(train_signal_end, model_config)
    val_len = train_signal_end - val_start
    splits = max(1, int(model_config.validation_splits))
    min_len = 3
    if val_len < splits * min_len:
        return [(val_start, train_signal_end)]
    base = val_len // splits
    windows: list[tuple[int, int]] = []
    for k in range(splits):
        start = val_start + k * base
        end = val_start + (k + 1) * base if k < splits - 1 else train_signal_end
        windows.append((start, end))
    return windows


def sample_random_formulas(
    seed: int, vocab, max_len: int, n: int
) -> list[tuple[int, ...]]:
    """Sample ``n`` structurally valid postfix formulas under a uniform
    prior over the legal action mask (the exact legality rules the policy
    samples under, so the random-search baseline and the RL policy share
    one search space).  Deterministic in ``seed``; pins torch's CPU RNG
    exactly like the trainer does.  Every sequence is EOS-terminated, so
    its effective (non-padded) length is always >= 2.
    """

    torch.manual_seed(seed)
    stack_sizes = torch.zeros(n, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    seqs: list[torch.Tensor] = []
    for pos in range(max_len):
        mask = build_action_mask(stack_sizes, done, pos, max_len, vocab)
        allowed = (mask == 0.0).float()
        totals = allowed.sum(dim=1, keepdim=True)
        # The legal mask guarantees at least one allowed token per row; the
        # uniform fallback keeps sampling total even if that invariant is
        # ever violated by a vocabulary change.
        safe = torch.where(
            totals > 0, allowed, torch.full_like(allowed, 1.0 / vocab.size)
        )
        probs = safe / safe.sum(dim=1, keepdim=True)
        action = torch.multinomial(probs, 1).squeeze(1)
        seqs.append(action)
        AshareTrainer._update_stack_state(action, stack_sizes, done)
    return [tuple(int(t) for t in row) for row in torch.stack(seqs, dim=1).tolist()]


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve a ``--device`` value to a concrete torch device.

    ``auto`` (the default) uses CUDA when available and falls back to CPU,
    so the same entry point runs accelerated on a GPU machine and unchanged
    on GPU-less CI.  Unknown values fail loudly.
    """

    if requested in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in ("cpu", "cuda"):
        raise ValueError(
            f"unknown device {requested!r}; expected 'auto', 'cpu' or 'cuda'"
        )
    return torch.device(requested)


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
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.backtest_config = backtest_config
        self.reward_config = reward_config or RewardConfig()
        self.init_seed = init_seed
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
        self.history: list[dict[str, float]] = []
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

        adv = (rewards - baseline.detach()) / (
            rewards.std(unbiased=False) + 1e-6
        )
        adv = adv.clamp(-advantage_clip, advantage_clip)
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
            done = torch.zeros(batch_size, dtype=torch.bool, device=policy_device)
            log_probs: list[torch.Tensor] = []
            sampled_tokens: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []

            for pos in range(max_len):
                logits, value = self.model(inp)
                values.append(value.squeeze(-1))
                mask = build_action_mask(
                    stack_sizes, done, pos, max_len, self.vocab
                )
                dist = Categorical(logits=logits + mask)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                sampled_tokens.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                self._update_stack_state(action, stack_sizes, done)

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
                    continue
                canonical_share[ckey] = key
                score = self.semantic_cache.get(ckey)
                if score is not None:
                    cache_hits += 1
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
                self.semantic_cache.put(ckey, step_results[key], fingerprint, bill)

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
            self.optimizer.step()

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
                }
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

        selection_path = self.data_config.data_dir / "training_selection.json"
        if save_artifacts:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            from .evaluation import PROTOCOL_VERSION  # noqa: PLC0415

            selection_payload = {
                "reward_version": REWARD_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "semantic_cache_version": SEMANTIC_CACHE_VERSION,
                "dataset_id": self.loader.dataset_id,
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
            contract=contract, vm_device=vm_device, searcher="rl"
        )

    def _write_artifact(
        self,
        *,
        contract: TrainingTimeContract,
        vm_device: torch.device,
        searcher: str = "rl",
    ) -> list[int] | None:
        """Write the standard training artifact (selection + strategy JSON +
        the policy checkpoint for RL runs) for the current selection."""

        selected = self.selection_result.selected
        if selected is None:
            return None
        # ``evaluation`` imports this module at module level; resolve lazily.
        from .evaluation import PROTOCOL_VERSION  # noqa: PLC0415

        score_payload = selected.to_dict()
        score_payload.pop("tokens", None)
        output = {
            "formula": self.best_tokens,
            **score_payload,
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
            "semantic_cache_version": SEMANTIC_CACHE_VERSION,
            "unique_semantic_evals": self.semantic_cache.budget_used,
            "semantic_cache_stats": self.semantic_cache.stats(),
            # Data provenance: the immutable dataset manifest this formula
            # was selected on (None for pre-T1-01 databases).
            "dataset_id": self.loader.dataset_id,
        }
        out_path = self.data_config.data_dir / "best_ashare_strategy.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
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
        # The capped window's signal end: labels need t+1/t+2, so the last
        # two capped columns are labels only.
        train_signal_end = min(
            contract.train_signal_end, max(0, train_end_idx - 2)
        )
        # Hold out the tail of the training window for out-of-sample best
        # formula selection, split into independent sub-windows; the best
        # formula is decided on the *median* validation reward so a single
        # lucky tail stretch cannot win the selection.
        val_windows = self._validation_windows(train_signal_end)
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
        target_ret = open_to_open_returns(train_open)
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
        # fingerprint) formulas never bill twice.  ``evaluation`` imports
        # this module at module level, so its constant is resolved lazily.
        from .evaluation import PROTOCOL_VERSION  # noqa: PLC0415

        self.semantic_cache = SemanticCache(
            dataset_id=self.loader.dataset_id,
            reward_version=REWARD_VERSION,
            protocol_version=PROTOCOL_VERSION,
            window_id=self._window_id(contract, val_windows),
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
            blocked_buy=blocked_buy,
            blocked_sell=blocked_sell,
            reward_chunk=reward_chunk,
        )

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
        """Run a non-RL searcher (``gp`` or ``random``) over the training
        window with the matched unique-semantic-evaluation budget and
        produce the standard training artifact.

        The searcher is billed through the same semantic cache as RL
        (``steps x batch_size`` unique semantic evaluations), so the
        production default can switch backends without changing the
        budget semantics or the artifact contract.
        """

        if searcher not in ("gp", "random"):
            raise ValueError(f"train_search supports 'gp' or 'random', got {searcher!r}")
        steps = steps or self.model_config.train_steps
        batch_size = batch_size or self.model_config.batch_size
        vm_device = resolve_device(device)
        window = self.prepare_window(train_end_date, vm_device, window_cap)
        budget = steps * batch_size

        # Lazily imported: baseline_harness/gp_search import this module at
        # module level, so the cycle is broken at call time.
        from .baseline_harness import (  # noqa: PLC0415
            SemanticBudgetEvaluator,
            canonical_form_pool,
        )
        from .evaluation import PROTOCOL_VERSION  # noqa: PLC0415
        from .gp_search import run_gp_baseline  # noqa: PLC0415

        def execute(tokens) -> np.ndarray | None:
            signal = self.vm.execute(tokens, window.factor_tensor)
            if signal is None:
                return None
            return signal.detach().cpu().numpy()

        evaluator = SemanticBudgetEvaluator(
            target=window.target_ret,
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
            window_id=self._window_id(window.contract, window.val_windows),
            tie_break_keys=np.asarray(self.loader.ts_codes),
            adv=np.asarray(self.loader.dollar_volume())[:, :window.train_end_idx],
            blocked_buy=window.blocked_buy,
            blocked_sell=window.blocked_sell,
            source=searcher,
            candidate_prefix=searcher,
            chunk=window.reward_chunk,
        )
        if searcher == "gp":
            result = run_gp_baseline(
                seed=seed,
                evaluator=evaluator,
                max_formula_len=self.model_config.max_formula_len,
            )
        else:
            for key in canonical_form_pool(
                seed,
                self.vocab,
                self.model_config.max_formula_len,
                budget,
            ):
                evaluator.propose(key)
                if evaluator.budget_used >= budget:
                    break
            result = evaluator.finish()

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
            contract=window.contract, vm_device=vm_device, searcher=searcher
        )

    @staticmethod
    def _window_id(contract, val_windows: list[tuple[int, int]]) -> str:
        """Deterministic id of the training window: the exact columns the
        semantic-cache key binds scores to."""

        val = "|".join(f"{a}:{b}" for a, b in val_windows)
        return (
            f"train:{contract.train_signal_start}:{contract.train_signal_end}:"
            f"label:{contract.train_label_end}:val:{val}"
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

    def _validation_windows(self, train_end_idx: int) -> list[tuple[int, int]]:
        """Independent validation sub-windows covering the training tail.

        Delegates to the module-level :func:`validation_windows`.
        """
        return validation_windows(train_end_idx, self.model_config)

    @staticmethod
    def _update_stack_state(
        action: torch.Tensor,
        stack_sizes: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Advance the stack-only sampling state by one sampled action.

        Postfix rules (see :mod:`ashare_model.ir`): a feature pushes one
        value, an operator of arity ``a`` pops ``a`` and pushes one
        (``stack - a + 1``), EOS terminates at ``stack == 1``, and PAD is
        only ever sampled after EOS.  ``done`` latches once EOS (or a
        legacy padding termination) is sampled.
        """

        is_pad = action == FORMULA_VOCAB.pad_token_id
        eos_id = FORMULA_VOCAB.eos_token_id
        is_eos = (
            action == eos_id
            if eos_id is not None
            else torch.zeros_like(action, dtype=torch.bool)
        )
        is_feature = (action.unsqueeze(1) == _FEATURE_IDS).any(dim=1)

        # Precomputed per-token arity table (module level, built once).
        arity = _OPERATOR_ARITY[action]

        new_stack = stack_sizes.clone()
        new_stack = torch.where(is_feature, stack_sizes + 1, new_stack)
        new_stack = torch.where(
            ~is_feature & ~is_pad & ~is_eos, stack_sizes - arity + 1, new_stack
        )
        new_stack = torch.clamp(new_stack, min=0)
        stack_sizes.copy_(new_stack)
        done.copy_(done | is_eos | is_pad)


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
        if model_config.searcher == "rl":
            tokens = trainer.train(
                steps=args.steps, batch_size=args.batch_size, device=args.device
            )
        else:
            # T2-03 searcher backend: the configured default searcher (gp /
            # random) replaces RL; the budget is the same steps x batch_size
            # unique semantic evaluations and the artifact contract is
            # unchanged (RL stays available via model.searcher: rl).
            tokens = trainer.train_search(
                searcher=model_config.searcher,
                steps=args.steps,
                batch_size=args.batch_size,
                device=args.device,
            )
        if tokens is None:
            # No formula met the validation-quality floor: fail loudly so
            # scripts and CI never mistag a no-artifact run as success.
            raise SystemExit(2)
    finally:
        export_log_txt(run_name="train")


if __name__ == "__main__":
    main()
