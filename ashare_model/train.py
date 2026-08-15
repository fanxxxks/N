"""REINFORCE trainer for A-share factor formulas.

Usage:
    python -m ashare_model.train [--config config/ashare_config.yaml]
                                [--offline] [--steps N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import json
import math
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
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
)

from .alphagpt import AlphaGPTModel, build_action_mask
from .data_loader import AshareDataLoader
from .ops import OPS_CONFIG
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
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.backtest_config = backtest_config
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
        dates = self.loader.dates[:train_end_idx]

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
                    rewards[i] = self.model_config.reward_clip_low
                    continue
                signal_np = signal.detach().cpu().numpy()
                if np.nanstd(signal_np) < 1e-4:
                    rewards[i] = -2.0
                    continue
                reward, _ = self._fast_reward(
                    signal_np,
                    target_ret,
                    self.loader.ts_codes,
                    dates,
                )
                rewards[i] = reward

                # Out-of-sample selection: the best formula is decided on the
                # validation slice, while the gradient still sees the full
                # training window.
                val_reward, _ = self._fast_reward(
                    signal_np[:, val_start:train_end_idx],
                    target_ret[:, val_start:train_end_idx],
                    self.loader.ts_codes,
                    dates[val_start:train_end_idx],
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

        output = {
            "formula": self.best_tokens,
            "formula_text": self.best_formula,
            "best_reward": self.best_reward,
            "history": self.history,
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

    def _fast_reward(
        self,
        signal: np.ndarray,
        target_ret: np.ndarray,
        ts_codes: list[str],
        dates: list[str],
    ) -> tuple[float, float]:
        """Cheap long-only Sortino reward used inside the RL loop.

        Trading costs are charged per unit of daily turnover using the same
        fee model as the backtest, so the training reward cannot be gamed by
        hyperactive strategies that the backtest would bleed dry.
        """

        cfg = self.backtest_config
        # Round-trip cost per unit turnover: both sides pay commission,
        # slippage and transfer fee; sells additionally pay stamp tax.
        cost_per_turnover = (
            2.0 * cfg.commission_rate
            + 2.0 * cfg.slippage_rate
            + 2.0 * cfg.transfer_fee_rate
            + cfg.stamp_tax_rate
        )

        n_stocks, n_dates = signal.shape
        if n_dates < 2 or n_stocks == 0:
            # Not enough data for a single daily return: neutral low reward.
            return float(self.model_config.reward_clip_low), 0.0
        top_n = min(self.backtest_config.top_n, n_stocks)
        daily = np.zeros(n_dates - 1, dtype=np.float64)
        previous: set[int] = set()
        turnover_total = 0.0

        for t in range(n_dates - 1):
            if np.isfinite(signal[:, t]).sum() < top_n:
                daily[t] = 0.0
                continue
            top_idx = np.argpartition(signal[:, t], -top_n)[-top_n:]
            top_idx = top_idx[np.argsort(signal[:, t][top_idx])[::-1]]
            rets = target_ret[top_idx, t]
            rets = rets[np.isfinite(rets)]
            daily[t] = float(np.mean(rets)) if rets.size else 0.0
            current = set(top_idx.tolist())
            turnover = (len(current ^ previous) / top_n) if previous else 0.0
            daily[t] -= turnover * cost_per_turnover
            turnover_total += turnover
            previous = current

        mean_daily = float(np.mean(daily))
        downside = daily[daily < 0]
        if downside.size >= 3:
            downside_std = float(downside.std(ddof=1) * math.sqrt(252))
        else:
            daily_std = float(daily.std(ddof=1)) if daily.size > 1 else 0.0
            downside_std = daily_std * math.sqrt(252) + 1e-6
        ann_mean = mean_daily * 252
        sortino = ann_mean / (downside_std + 1e-9)
        reward = sortino
        avg_turnover = turnover_total / max(n_dates - 1, 1)
        if avg_turnover > 0.5:
            reward -= 1.0
        if mean_daily < 0:
            reward = -2.0
        if top_n == 0 or (np.nanstd(daily) < 1e-6 and mean_daily <= 0):
            reward = -2.0
        return float(
            np.clip(
                reward,
                self.model_config.reward_clip_low,
                self.model_config.reward_clip_high,
            )
        ), mean_daily


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
        loader = AshareDataLoader(data_config, model_config)
        trainer = AshareTrainer(data_config, model_config, backtest_config, loader)
        trainer.train(steps=args.steps, batch_size=args.batch_size)
    finally:
        export_log_txt(run_name="train")


if __name__ == "__main__":
    main()
