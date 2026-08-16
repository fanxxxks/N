"""REINFORCE trainer for A-share factor formulas.

Usage:
    python -m ashare_model.train [--config config/ashare_config.yaml]
                                [--offline] [--steps N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
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
from .data_loader import AshareDataLoader
from .ops import OPS_CONFIG
from .reward import REWARD_VERSION, fast_basket_reward
from .vm import StackVM, formula_decode
from .vocab import FORMULA_VOCAB
from ashare_logging import export_log_txt, setup_run_logging


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class AshareTrainer:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        backtest_config: BacktestConfig,
        loader: AshareDataLoader | None = None,
        reward_config: RewardConfig | None = None,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.backtest_config = backtest_config
        self.reward_config = reward_config or RewardConfig()
        self.loader = loader or AshareDataLoader(data_config, model_config)
        if self.loader.factor_tensor is None:
            self.loader.load_data()
        self.vocab = FORMULA_VOCAB
        self.vm = StackVM(self.vocab)
        self.model = AlphaGPTModel(model_config, self.vocab)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=model_config.learning_rate
        )
        self.best_reward = -float("inf")
        self.best_tokens: list[int] | None = None
        self.best_formula = ""
        self.history: list[dict[str, float]] = []

    def train(
        self,
        steps: int | None = None,
        batch_size: int | None = None,
        seed: int = 42,
        save_artifacts: bool = True,
    ) -> list[int] | None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        steps = steps or self.model_config.train_steps
        batch_size = batch_size or self.model_config.batch_size
        max_len = self.model_config.max_formula_len
        device = next(self.model.parameters()).device

        train_end_idx = self._train_end_index()
        if train_end_idx <= 2:
            logger.warning(
                f"Training window is degenerate ({train_end_idx} dates); "
                "check backtest.train_end_date against the loaded data range "
                f"({self.loader.dates[0]} .. {self.loader.dates[-1]})."
            )
        factor_tensor = self.loader.factor_tensor[:, :, :train_end_idx]
        target_ret = self.loader.target_ret[:, :train_end_idx].numpy()

        # Hold out the tail of the training window for out-of-sample best
        # formula selection, so the saved formula is not chosen purely on
        # in-sample reward.
        val_start = self._validation_start(train_end_idx)

        pbar = tqdm(range(steps))
        for step in pbar:
            inp = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            open_slots = torch.ones(batch_size, dtype=torch.long, device=device)
            stack_sizes = torch.zeros(batch_size, dtype=torch.long, device=device)
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
            rewards = torch.zeros(batch_size, device=device)

            for i in range(batch_size):
                tokens = sequences[i].tolist()
                signal = self.vm.execute(tokens, factor_tensor)
                if signal is None:
                    rewards[i] = self.reward_config.reward_clip_low
                    continue
                signal_np = signal.detach().cpu().numpy()
                if np.nanstd(signal_np) < 1e-4:
                    rewards[i] = self.reward_config.bad_reward
                    continue
                reward, _ = fast_basket_reward(
                    signal_np,
                    target_ret,
                    self.backtest_config,
                    self.reward_config,
                )
                rewards[i] = reward

                # Out-of-sample selection: the best formula is decided on the
                # validation slice, while the gradient still sees the full
                # training window.
                val_reward, _ = fast_basket_reward(
                    signal_np[:, val_start:train_end_idx],
                    target_ret[:, val_start:train_end_idx],
                    self.backtest_config,
                    self.reward_config,
                )
                if val_reward > self.best_reward:
                    self.best_reward = val_reward
                    self.best_tokens = tokens
                    self.best_formula = formula_decode(tokens, self.vocab)
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

    def _train_end_index(self) -> int:
        train_end = self.backtest_config.train_end_date.replace("-", "")
        for idx, date in enumerate(self.loader.dates):
            if date >= train_end:
                return max(idx, 1)
        return len(self.loader.dates)

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
        trainer.train(steps=args.steps, batch_size=args.batch_size)
    finally:
        export_log_txt(run_name="train")


if __name__ == "__main__":
    main()
