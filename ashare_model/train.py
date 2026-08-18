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
from .reward import REWARD_VERSION, batched_basket_rewards
from .vm import StackVM, formula_decode
from .vocab import FORMULA_VOCAB
from ashare_logging import export_log_txt, setup_run_logging


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
        self.best_tokens: list[int] | None = None
        self.best_formula = ""
        self.history: list[dict[str, float]] = []
        self._reward_cache: OrderedDict[
            tuple[int, ...], tuple[float, float | None]
        ] = OrderedDict()

    def _cache_put(self, key: tuple[int, ...], value: tuple[float, float | None]) -> None:
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

    def _score_pending_chunk(
        self,
        pending: list[tuple[tuple[int, ...], np.ndarray]],
        target_ret: np.ndarray,
        val_start: int,
        step_results: dict[tuple[int, ...], tuple[float, float | None]],
    ) -> None:
        """Score one chunk of pending formulas and merge the outcomes."""

        if not pending:
            return
        chunk_rewards, chunk_val = batched_basket_rewards(
            np.stack([signal_np for _, signal_np in pending]),
            target_ret,
            self.backtest_config,
            self.reward_config,
            val_start,
        )
        for (key, _), reward, val_reward in zip(
            pending, chunk_rewards, chunk_val
        ):
            step_results[key] = (float(reward), float(val_reward))

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
        target_ret = self.loader.target_ret[:, :train_end_idx].numpy()

        # Hold out the tail of the training window for out-of-sample best
        # formula selection, so the saved formula is not chosen purely on
        # in-sample reward.
        val_start = self._validation_start(train_end_idx)

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

            for pos in range(max_len):
                logits, value, _ = self.model(inp)
                values.append(value.squeeze(-1))
                mask = build_action_mask(
                    open_slots, stack_sizes, pos, max_len, self.vocab
                )
                dist = Categorical(logits=logits + mask)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
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
                tuple[int, ...], tuple[float, float | None]
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
                    )
                    continue
                signal_np = signal.detach().cpu().numpy()
                if np.nanstd(signal_np) < 1e-4:
                    step_results[key] = (
                        float(self.reward_config.bad_reward),
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
                        pending, target_ret, val_start, step_results
                    )
                    pending = []
            self._score_pending_chunk(pending, target_ret, val_start, step_results)

            for i in range(batch_size):
                key = tuple(sequences[i].tolist())
                reward, val_reward = step_results[key]
                rewards[i] = reward
                # Cache every outcome (invalid and constant formulas too):
                # the token sequence fully determines it, so repeats can
                # skip the VM execution as well.
                self._cache_put(key, (reward, val_reward))
                if val_reward is not None:
                    # Out-of-sample selection: the best formula is decided on
                    # the validation slice, while the gradient still sees the
                    # full training window.
                    if val_reward > self.best_reward:
                        self.best_reward = val_reward
                        self.best_tokens = sequences[i].tolist()
                        self.best_formula = formula_decode(
                            self.best_tokens, self.vocab
                        )
                        pbar.write(
                            f"[+] New best (validation): reward={val_reward:.3f} "
                            f"formula={self.best_formula}"
                        )

            # Actor-critic update: REINFORCE with the learned value as
            # baseline, plus a value regression loss.
            baseline = values[-1]
            adv = (rewards - baseline.detach()) / (
                rewards.std(unbiased=False) + 1e-6
            )
            policy_loss = -(
                torch.stack(log_probs, dim=1).sum(dim=1) * adv
            ).mean()
            value_loss = torch.nn.functional.mse_loss(baseline, rewards)
            loss = policy_loss + self.model_config.value_loss_weight * value_loss
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
                }
            )
            pbar.set_postfix(
                {
                    "avg_reward": f"{rewards.mean().item():.3f}",
                    "best": f"{self.best_reward:.3f}",
                }
            )

        if self.best_tokens is None:
            logger.warning("No valid formula found")
            return None

        if not save_artifacts:
            logger.success(
                f"Training complete (artifacts skipped); "
                f"best_reward={self.best_reward:.3f} formula={self.best_formula}"
            )
            return self.best_tokens

        output = {
            "formula": self.best_tokens,
            "formula_text": self.best_formula,
            "best_reward": self.best_reward,
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

        The tail keeps at least two dates (a reward needs one daily return);
        windows too small to hold anything out validate on the full window.
        """
        if train_end_idx <= 2:
            return 0
        val_frac = float(np.clip(self.model_config.validation_fraction, 0.0, 0.5))
        val_start = int(round(train_end_idx * (1.0 - val_frac)))
        return max(1, min(val_start, train_end_idx - 2))

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
        trainer.train(
            steps=args.steps, batch_size=args.batch_size, device=args.device
        )
    finally:
        export_log_txt(run_name="train")


if __name__ == "__main__":
    main()
