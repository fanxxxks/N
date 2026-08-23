"""REINFORCE trainer for A-share factor formulas.

Usage:
    python -m ashare_model.train [--config config/ashare_config.yaml]
                                 [--steps N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
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
from ashare_data.processor import open_to_open_returns
from ashare_data.universe import require_production_universe

from .alphagpt import AlphaGPTModel, build_action_mask
from .candidates import (
    CandidateScore,
    CandidateScorer,
    CandidateSelector,
    CandidateSpec,
    SelectionResult,
    score_chunk_size,
)
from .data_loader import AshareDataLoader
from .ops import OPS_CONFIG
from .reward import (
    REWARD_VERSION,
    batched_basket_rewards,
)
from .time_contract import TrainingTimeContract
from .vm import StackVM, formula_decode
from .vocab import FORMULA_VOCAB
from ashare_logging import export_log_txt, setup_run_logging

# Sampling-state tables, built once at import: the arity of every operator
# token and the feature-token id range, so the per-position stack update
# never rebuilds tensors inside the sampling loop (the policy and sampling
# always run on CPU, matching these tensors).
_OPERATOR_ARITY = torch.zeros(FORMULA_VOCAB.size)
for _i, (_, _, _arity) in enumerate(OPS_CONFIG):
    _OPERATOR_ARITY[FORMULA_VOCAB.operator_offset + _i] = _arity
_FEATURE_IDS = torch.arange(1, FORMULA_VOCAB.operator_offset)


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
    exactly like the trainer does.
    """

    torch.manual_seed(seed)
    open_slots = torch.ones(n, dtype=torch.long)
    stack_sizes = torch.zeros(n, dtype=torch.long)
    seqs: list[torch.Tensor] = []
    for pos in range(max_len):
        mask = build_action_mask(open_slots, stack_sizes, pos, max_len, vocab)
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
        AshareTrainer._update_stack_state(action, open_slots, stack_sizes)
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
            operator_offset=self.vocab.operator_offset,
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
        self._reward_cache: OrderedDict[tuple[int, ...], CandidateScore] = OrderedDict()

    def _cache_put(
        self,
        key: tuple[int, ...],
        value: CandidateScore,
    ) -> None:
        self._reward_cache[key] = value
        self._reward_cache.move_to_end(key)
        while len(self._reward_cache) > self._REWARD_CACHE_CAP:
            self._reward_cache.popitem(last=False)

    def _cache_touch(self, key: tuple[int, ...]) -> None:
        self._reward_cache.move_to_end(key)

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

        contract = self._training_contract(train_end_date)
        train_end_idx = contract.train_label_end
        # Hold out the tail of the training window for out-of-sample best
        # formula selection, split into independent sub-windows; the best
        # formula is decided on the *median* validation reward so a single
        # lucky tail stretch cannot win the selection.
        val_windows = self._validation_windows(contract.train_signal_end)
        # The *learning* window is the in-sample head only: the policy
        # gradient scores candidates on columns strictly before the first
        # validation window, so the selection data never doubles as the
        # training signal (val rewards rank formulas; IS rewards teach the
        # policy — two jobs, two windows).
        val_start = validation_start(contract.train_signal_end, self.model_config)
        train_signal_range = (contract.train_signal_start, val_start)
        factor_tensor = self.loader.factor_tensor[:, :, :train_end_idx].to(
            vm_device
        )
        # The VM executes on the compute device; the industry-group tensor
        # for CS_NEUTRALIZE and the PIT eligibility mask for the
        # cross-sectional operators move with the factor stack (the loader
        # always builds the mask in load_data, so there is no fallback).
        industry_codes = getattr(self.loader, "industry_codes", None)
        self.vm.industry_codes = (
            industry_codes[:, :train_end_idx].to(vm_device)
            if industry_codes is not None
            else None
        )
        universe_mask = self.loader.universe_mask
        self.vm.universe_mask = torch.tensor(
            universe_mask[:, :train_end_idx],
            dtype=torch.bool,
            device=vm_device,
        )
        # The candidate scorer needs the same PIT mask on the numpy side:
        # every quality statistic (rank IC, basket selection, near-constant
        # rejection, direction) is gated to signal-date eligible cells.
        train_universe_mask = universe_mask[:, :train_end_idx]
        # Recompute labels only from prices inside the inclusive training
        # anchor. Precomputed global targets at the tail may reference fold-
        # external t+1/t+2 prices and are intentionally never sliced here.
        train_open = self.loader.raw_data_cache["open"][:, :train_end_idx].numpy()
        target_ret = open_to_open_returns(train_open)
        target_ret = self.loader.mask_by_universe(target_ret)
        # Tradability masks (buy/sell blocked per stock and date) align the
        # training basket with the backtest engine's execution rules; both
        # matrices are shared by every formula scored this run.
        blocked_buy, blocked_sell = self.loader.tradability_masks()
        blocked_buy = blocked_buy[:, :train_end_idx]
        blocked_sell = blocked_sell[:, :train_end_idx]

        # Chunk the batched reward evaluation so the stacked signal matrix
        # (original stack + batched copy + both-direction copy) stays within
        # a fixed memory budget (~512 MB of float64 signals).
        signal_bytes = factor_tensor.shape[1] * train_end_idx * 8
        reward_chunk = score_chunk_size(signal_bytes)

        pbar = tqdm(range(steps))
        for step in pbar:
            # The policy and sampling stay on CPU (the model's device): its
            # RNG stream and dropout are device-independent by construction.
            policy_device = next(self.model.parameters()).device
            inp = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=policy_device
            )
            open_slots = torch.ones(
                batch_size, dtype=torch.long, device=policy_device
            )
            stack_sizes = torch.zeros(
                batch_size, dtype=torch.long, device=policy_device
            )
            log_probs: list[torch.Tensor] = []
            sampled_tokens: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []

            for pos in range(max_len):
                logits, value, _ = self.model(inp)
                values.append(value.squeeze(-1))
                mask = build_action_mask(
                    open_slots, stack_sizes, pos, max_len, self.vocab
                )
                dist = Categorical(logits=logits + mask)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                sampled_tokens.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
                self._update_stack_state(
                    action, open_slots, stack_sizes
                )

            sequences = torch.stack(sampled_tokens, dim=1)
            rewards = torch.zeros(batch_size, device=policy_device)

            # Batched reward evaluation with formula deduplication: a formula
            # is executed once per step (and reused across steps through the
            # bounded cache); valid signals are scored in chunks so the
            # vectorized basket simulation stays memory-bounded.
            step_results: dict[tuple[int, ...], CandidateScore] = {}
            pending: list[tuple[CandidateSpec, np.ndarray]] = []
            seen: set[tuple[int, ...]] = set()
            for i in range(batch_size):
                key = tuple(sequences[i].tolist())
                if key in self._reward_cache:
                    self._cache_touch(key)
                    step_results[key] = self._reward_cache[key]
                    continue
                if key in seen:
                    continue
                seen.add(key)
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
                signal_np = signal.detach().cpu().numpy()
                pending.append((spec, signal_np))
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
            )

            for i in range(batch_size):
                key = tuple(sequences[i].tolist())
                score = step_results[key]
                # The gradient reads the in-sample reward only: the
                # validation tail ranks formulas but never teaches the
                # policy (test_policy_gradient_reward_window_excludes_
                # validation_tail pins this contract).
                rewards[i] = score.train_reward
                # Cache every outcome (invalid and constant formulas too):
                # the token sequence fully determines it, so repeats can
                # skip the VM execution as well.
                self._cache_put(key, score)
                self._candidate_scores[key] = score

            previous_key = (
                self.selection_result.selected.deterministic_key
                if self.selection_result.selected
                else None
            )
            self.selection_result = self.candidate_selector.select(
                self._candidate_scores.values()
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
                    "unique_frac": float(unique_frac),
                }
            )
            pbar.set_postfix(
                {
                    "avg_reward": f"{rewards.mean().item():.3f}",
                    "best": f"{self.best_val_reward:.3f}",
                }
            )

        selection_path = self.data_config.data_dir / "training_selection.json"
        if save_artifacts:
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_payload = {
                "reward_version": REWARD_VERSION,
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

        score_payload = selected.to_dict()
        score_payload.pop("tokens", None)
        output = {
            "formula": self.best_tokens,
            **score_payload,
            "history": self.history,
            # Reproducibility provenance: the policy stays on CPU, so
            # (init_seed, seed) reproduce the same sampled formulas on any
            # machine; ``device`` records where the VM executed.
            "init_seed": self.init_seed,
            "device": str(vm_device),
            # Vocabulary provenance: the formula is always remapped by name
            # on load, so later vocabulary additions cannot silently
            # reinterpret these token ids.
            "feature_names": list(self.vocab.feature_names),
            "operator_names": list(self.vocab.operator_names),
            "feature_version": self.vocab.feature_version,
            # Reward provenance: reward values are only comparable within
            # the same scoring implementation generation.
            "reward_version": REWARD_VERSION,
        }
        out_path = self.data_config.data_dir / "best_ashare_strategy.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        torch.save(self.model.state_dict(), self.data_config.data_dir / "ashare_model.pt")
        logger.success(f"Training complete; best formula saved to {out_path}")
        return self.best_tokens

    def _training_contract(
        self, train_end_date: str | None = None
    ) -> TrainingTimeContract:
        return TrainingTimeContract.resolve(
            self.loader.dates,
            train_end_date or self.backtest_config.train_end_date,
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
    @staticmethod
    def _update_stack_state(
        action: torch.Tensor,
        open_slots: torch.Tensor,
        stack_sizes: torch.Tensor,
    ) -> None:
        is_pad = action == FORMULA_VOCAB.pad_token_id
        is_feature = (action.unsqueeze(1) == _FEATURE_IDS).any(dim=1)

        feature = is_feature
        pad = is_pad
        # Precomputed per-token arity table (module level, built once).
        arity = _OPERATOR_ARITY[action]

        old_stack = stack_sizes.clone()
        new_stack = old_stack.clone()
        new_stack = torch.where(feature, old_stack + 1, new_stack)
        new_stack = torch.where(~feature & ~pad, old_stack - arity + 1, new_stack)
        new_stack = torch.clamp(new_stack, min=0)

        open_slots_new = open_slots.clone()
        open_slots_new = torch.where(feature, open_slots - 1, open_slots_new)
        open_slots_new = torch.where(
            ~feature & ~pad, open_slots + arity - 1, open_slots_new
        )
        open_slots.copy_(torch.clamp(open_slots_new, min=0))
        stack_sizes.copy_(new_stack)


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
    args = parser.parse_args()

    try:
        root = _project_root()
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        require_production_universe(data_config)
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        loader = AshareDataLoader(data_config, model_config)
        trainer = AshareTrainer(
            data_config, model_config, backtest_config, loader, reward_config
        )
        if (
            trainer.train(
                steps=args.steps, batch_size=args.batch_size, device=args.device
            )
            is None
        ):
            # No formula met the validation-quality floor: fail loudly so
            # scripts and CI never mistag a no-artifact run as success.
            raise SystemExit(2)
    finally:
        export_log_txt(run_name="train")


if __name__ == "__main__":
    main()
