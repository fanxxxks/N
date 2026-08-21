"""REINFORCE trainer for A-share factor formulas.

Usage:
    python -m ashare_model.train [--config config/ashare_config.yaml]
                                [--offline] [--steps N] [--batch-size N]
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

from .alphagpt import AlphaGPTModel, build_action_mask
from .data_loader import AshareDataLoader, date_index
from .ops import OPS_CONFIG
from .reward import (
    REWARD_VERSION,
    batched_basket_rewards,
    signal_direction,
)
from .vm import StackVM, formula_decode
from .vocab import FORMULA_VOCAB
from ashare_logging import export_log_txt, setup_run_logging


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validation_start(train_end_idx: int, model_config) -> int:
    """First index of the validation tail inside the training window.

    The tail keeps at least two dates (a reward needs one daily return);
    windows too small to hold anything out validate on the full window.
    """

    if train_end_idx <= 2:
        return 0
    val_frac = float(np.clip(model_config.validation_fraction, 0.0, 0.5))
    val_start = int(round(train_end_idx * (1.0 - val_frac)))
    return max(1, min(val_start, train_end_idx - 2))


def validation_windows(
    train_end_idx: int, model_config
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

    val_start = validation_start(train_end_idx, model_config)
    val_len = train_end_idx - val_start
    splits = max(1, int(model_config.validation_splits))
    min_len = 3
    if val_len < splits * min_len:
        return [(val_start, train_end_idx)]
    base = val_len // splits
    windows: list[tuple[int, int]] = []
    for k in range(splits):
        start = val_start + k * base
        end = val_start + (k + 1) * base if k < splits - 1 else train_end_idx
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
        self.vm = StackVM(self.vocab)
        # Pin the weight initialization so the same (init_seed, seed)
        # pair reproduces the same training on any machine.
        torch.manual_seed(init_seed)
        self.model = AlphaGPTModel(model_config, self.vocab)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=model_config.learning_rate
        )
        self.best_reward = -float("inf")
        self.best_icir = -float("inf")
        self.best_direction = 1
        self.best_tokens: list[int] | None = None
        self.best_formula = ""
        self.history: list[dict[str, float]] = []
        self._collapse_streak = 0
        self._reward_cache: OrderedDict[
            tuple[int, ...], tuple[float, float | None, float, float | None]
        ] = OrderedDict()

    def _cache_put(
        self,
        key: tuple[int, ...],
        value: tuple[float, float | None, float, float | None],
    ) -> None:
        self._reward_cache[key] = value
        self._reward_cache.move_to_end(key)
        while len(self._reward_cache) > self._REWARD_CACHE_CAP:
            self._reward_cache.popitem(last=False)

    def _cache_touch(self, key: tuple[int, ...]) -> None:
        self._reward_cache.move_to_end(key)

    @staticmethod
    def _reward_chunk_size(signal_bytes: int) -> int:
        """Formulas per reward-scoring chunk under the ~512 MB budget.

        One chunk stacks its signals as float64, so ``chunk x signal_bytes``
        bytes stay below the budget; the chunk is capped at 64 (small
        windows would otherwise build huge numpy stacks) and floored at 1
        so a single oversized signal still makes progress.
        """

        return max(1, min(64, (512 * (1 << 20)) // max(signal_bytes, 1)))

    def _complexity_penalty(self, key: tuple[int, ...]) -> float:
        """Reward penalty for operator-free (bare single-factor) formulas.

        A formula whose tokens contain no operator is a copy of one raw
        feature; it is penalized (not banned) so the policy still has a
        gradient towards combinations while degenerate copies lose their
        edge over noise.
        """

        if any(t >= self.vocab.operator_offset for t in key):
            return 0.0
        return float(self.reward_config.complexity_penalty)

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
        pending: list[tuple[tuple[int, ...], np.ndarray]],
        target_ret: np.ndarray,
        val_windows: list[tuple[int, int]],
        step_results: dict[
            tuple[int, ...], tuple[float, float | None, float, float | None]
        ],
        blocked_buy: np.ndarray | None = None,
        blocked_sell: np.ndarray | None = None,
    ) -> None:
        """Score one chunk of pending formulas and merge the outcomes."""

        if not pending:
            return
        chunk_rewards, chunk_val, chunk_icir, chunk_val_icir = (
            batched_basket_rewards(
                np.stack([signal_np for _, signal_np in pending]),
                target_ret,
                self.backtest_config,
                self.reward_config,
                val_windows,
                blocked_buy=blocked_buy,
                blocked_sell=blocked_sell,
            )
        )
        for (key, _), reward, val_reward, icir, val_icir in zip(
            pending, chunk_rewards, chunk_val, chunk_icir, chunk_val_icir
        ):
            step_results[key] = (
                float(reward),
                float(val_reward) if val_reward is not None else None,
                float(icir),
                float(val_icir) if val_icir is not None else None,
            )

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

        train_end_idx = self._train_end_index(train_end_date)
        if train_end_idx <= 2:
            logger.warning(
                f"Training window is degenerate ({train_end_idx} dates); "
                "check backtest.train_end_date against the loaded data range "
                f"({self.loader.dates[0]} .. {self.loader.dates[-1]})."
            )
        factor_tensor = self.loader.factor_tensor[:, :, :train_end_idx].to(
            vm_device
        )
        # The VM executes on the compute device; the industry-group tensor
        # for CS_NEUTRALIZE moves with the factor stack (None when the
        # loader carries no industry data).
        industry_codes = getattr(self.loader, "industry_codes", None)
        self.vm.industry_codes = (
            industry_codes[:, :train_end_idx].to(vm_device)
            if industry_codes is not None
            else None
        )
        target_ret = self.loader.target_ret[:, :train_end_idx].numpy()
        # Tradability masks (buy/sell blocked per stock and date) align the
        # training basket with the backtest engine's execution rules; both
        # matrices are shared by every formula scored this run.
        blocked_buy, blocked_sell = self.loader.tradability_masks()
        blocked_buy = blocked_buy[:, :train_end_idx]
        blocked_sell = blocked_sell[:, :train_end_idx]

        # Hold out the tail of the training window for out-of-sample best
        # formula selection, split into independent sub-windows; the best
        # formula is decided on the *median* validation reward so a single
        # lucky tail stretch cannot win the selection.
        val_windows = self._validation_windows(train_end_idx)

        # Chunk the batched reward evaluation so the stacked signal matrix
        # stays within a fixed memory budget (~512 MB of float64 signals).
        signal_bytes = factor_tensor.shape[1] * train_end_idx * 8
        reward_chunk = self._reward_chunk_size(signal_bytes)

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
            step_results: dict[
                tuple[int, ...], tuple[float, float | None, float, float | None]
            ] = {}
            pending: list[tuple[tuple[int, ...], np.ndarray]] = []
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
                signal = self.vm.execute(key, factor_tensor)
                if signal is None:
                    step_results[key] = (
                        float(self.reward_config.reward_clip_low),
                        None,
                        0.0,
                        None,
                    )
                    continue
                signal_np = signal.detach().cpu().numpy()
                if np.nanstd(signal_np) < 1e-4:
                    step_results[key] = (
                        float(self.reward_config.bad_reward),
                        None,
                        0.0,
                        None,
                    )
                    continue
                pending.append((key, signal_np))
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
                        blocked_buy,
                        blocked_sell,
                    )
                    pending = []
            self._score_pending_chunk(
                pending, target_ret, val_windows, step_results, blocked_buy, blocked_sell
            )

            for i in range(batch_size):
                key = tuple(sequences[i].tolist())
                reward, val_reward, icir, val_icir = step_results[key]
                if val_reward is not None:
                    # Valid, non-constant formulas pay the bare-copy penalty
                    # on both the gradient reward and the selection value.
                    penalty = self._complexity_penalty(key)
                    reward -= penalty
                    val_reward -= penalty
                rewards[i] = reward
                # Cache every outcome (invalid and constant formulas too):
                # the token sequence fully determines it, so repeats can
                # skip the VM execution as well.
                self._cache_put(key, (reward, val_reward, icir, val_icir))
                if val_reward is not None:
                    # Out-of-sample selection: the best formula is decided on
                    # the median validation reward, while the gradient still
                    # sees the full training window.  The full-window ICIR is
                    # tracked alongside for the signal-quality gate.
                    if val_reward > self.best_reward:
                        self.best_reward = val_reward
                        self.best_icir = icir
                        self.best_tokens = sequences[i].tolist()
                        self.best_formula = formula_decode(
                            self.best_tokens, self.vocab
                        )
                        pbar.write(
                            f"[+] New best (validation): reward={val_reward:.3f} "
                            f"icir={icir:.3f} formula={self.best_formula}"
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
                    "best_reward": float(self.best_reward),
                    "loss": float(loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "unique_frac": float(unique_frac),
                }
            )
            pbar.set_postfix(
                {
                    "avg_reward": f"{rewards.mean().item():.3f}",
                    "best": f"{self.best_reward:.3f}",
                }
            )

        # Learned trade direction of the best formula, decided on the
        # validation tail only (out-of-sample-safe for deployment): a
        # negative-IC signal is flipped so the top-n long-only engines never
        # mechanically trade it backwards.
        if self.best_tokens is not None:
            signal = self.vm.execute(self.best_tokens, factor_tensor)
            if signal is not None:
                sig_np = signal.detach().cpu().numpy()
                val_start = self._validation_start(train_end_idx)
                self.best_direction = signal_direction(
                    sig_np[:, val_start:train_end_idx],
                    target_ret[:, val_start:train_end_idx],
                    self.reward_config.ic_min_stocks,
                )

        if self.best_tokens is None or self.best_reward < float(
            self.reward_config.min_val_reward
        ):
            logger.warning(
                f"No formula met the validation-quality floor "
                f"(best={self.best_reward:.3f}, "
                f"min_val_reward={self.reward_config.min_val_reward:.3f}); "
                "no formula is saved"
            )
            return None
        if self.best_icir < float(self.reward_config.min_val_icir):
            logger.warning(
                f"No formula met the signal-quality gate "
                f"(best_icir={self.best_icir:.3f}, "
                f"min_val_icir={self.reward_config.min_val_icir:.3f}); "
                "no formula is saved"
            )
            return None

        if not save_artifacts:
            logger.success(
                f"Training complete (artifacts skipped); "
                f"best_reward={self.best_reward:.3f} "
                f"best_icir={self.best_icir:.3f} "
                f"direction={self.best_direction} "
                f"formula={self.best_formula}"
            )
            return self.best_tokens

        output = {
            "formula": self.best_tokens,
            "formula_text": self.best_formula,
            "best_reward": self.best_reward,
            "best_icir": self.best_icir,
            "direction": self.best_direction,
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
            # Reward provenance: best_reward values are only comparable
            # within the same reward implementation generation.
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

    def _train_end_index(self, train_end_date: str | None = None) -> int:
        """First column index at or after the training-window end date.

        ``train_end_date`` overrides ``backtest_config.train_end_date`` so
        the evaluation protocol can train each walk-forward fold against its
        own absolute cutoff without touching the shared config.
        """

        train_end = (train_end_date or self.backtest_config.train_end_date).replace(
            "-", ""
        )
        return date_index(self.loader.dates, train_end)

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
        open_slots: torch.Tensor,
        stack_sizes: torch.Tensor,
    ) -> None:
        feature_ids = torch.arange(1, FORMULA_VOCAB.operator_offset, device=action.device)
        is_pad = action == FORMULA_VOCAB.pad_token_id
        is_feature = (action.unsqueeze(1) == feature_ids).any(dim=1)

        feature = is_feature
        pad = is_pad
        # For every operator token, compute delta = arity - 1.
        arity = torch.zeros_like(action)
        for i, (_, _, a) in enumerate(OPS_CONFIG):
            token = FORMULA_VOCAB.operator_offset + i
            arity = torch.where(action == token, torch.tensor(a, device=action.device), arity)

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
    parser.add_argument("--offline", action="store_true")
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
