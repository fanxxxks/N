"""Matched-budget baseline harness CLI (T1-05).

Runs the budget-matched random-search baseline on one fold and reports
the phase-1 research-validity statistics:

* the baseline's selection and per-candidate rejection counts;
* the reward<->OOS-active-IR correlation over a deterministic spread of
  the scored candidates (Spearman rho + permutation p-value) — the
  completion gate "the training reward is a stable positive proxy of
  out-of-sample portfolio quality".

Usage:
    python scripts/baseline_harness.py --config config/ashare_config.yaml \
        --fold 0 --budget 200 --seed 42 --output data/baseline_harness.json

``--budget`` defaults to the screening tier's matched budget
(steps x batch_size).  ``--evaluate`` bounds how many candidates get the
full OOS engine measurement for the correlation (a deterministic,
reward-spread-covering sample; default 200).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_reward_config,
    make_protocol_config,
)
from ashare_data.gates import ProductionGateRunner
from ashare_logging import export_log_txt, setup_run_logging

from ashare_model.baseline_harness import (
    oos_active_ir,
    reward_oos_correlation,
    run_matched_baseline,
)
from ashare_model.data_loader import AshareDataLoader
from ashare_model.evaluation import evaluate_signal, resolve_folds
from ashare_model.train import validation_start, validation_windows
from ashare_model.vm import StackVM
from ashare_model.vocab import FORMULA_VOCAB


def main(argv=None) -> int:
    setup_run_logging(run_name="baseline_harness")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--fold", type=int, default=0, help="fold index to run (default 0)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="unique formula evaluations (default: screening steps x batch)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate",
        type=int,
        default=200,
        help="max candidates measured OOS for the correlation (default 200)",
    )
    parser.add_argument("--output", default="data/baseline_harness.json")
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window",
    )
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(
            data_config, min_eligible=args.min_eligible
        ).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        proto_cfg = make_protocol_config(raw)
        fold_cfg = proto_cfg.folds[args.fold]

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        fold = resolve_folds([fold_cfg], loader.dates)[0]
        contract = fold.contract
        train_price_end = contract.train_label_end
        train_signal_end = contract.train_signal_end
        budget = args.budget or (
            proto_cfg.screening.steps * proto_cfg.screening.batch_size
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        factors = loader.factor_tensor[:, :, :train_price_end].to(device)
        industry_codes = getattr(loader, "industry_codes", None)
        vm = StackVM(
            FORMULA_VOCAB,
            industry_codes=(
                industry_codes[:, :train_price_end].to(device)
                if industry_codes is not None
                else None
            ),
            universe_mask=torch.tensor(
                loader.universe_mask[:, :train_price_end],
                dtype=torch.bool,
                device=device,
            ),
        )
        universe_mask = loader.universe_mask[:, :train_price_end]
        target = loader.mask_by_universe(
            loader.target_ret.numpy()[:, :train_price_end]
        )
        val_windows = validation_windows(train_signal_end, model_config)
        train_signal_range = (
            contract.train_signal_start,
            validation_start(train_signal_end, model_config),
        )

        def execute(tokens):
            signal = vm.execute(list(tokens), factors)
            if signal is None:
                return None
            return signal.detach().cpu().numpy()

        result = run_matched_baseline(
            target=target,
            universe_mask=universe_mask,
            backtest_config=backtest_config,
            reward_config=reward_config,
            val_windows=val_windows,
            train_signal_range=train_signal_range,
            budget=budget,
            seed=args.seed,
            execute=execute,
            max_formula_len=model_config.max_formula_len,
            tie_break_keys=np.asarray(loader.ts_codes),
            adv=np.asarray(loader.dollar_volume())[:, :train_price_end],
        )

        # OOS active IR for a deterministic, reward-spread-covering sample
        # of the scored candidates (the engine measurement per candidate).
        scored = sorted(
            (s for s in result.scores if s.eligible),
            key=lambda s: s.val_reward,
        )
        sample = scored[: args.evaluate // 2] + scored[
            max(len(scored) // 2, args.evaluate // 2) : len(scored)
        ][: args.evaluate - args.evaluate // 2]
        pairs = []
        for score in sample:
            metrics = evaluate_signal(
                float(score.direction)
                * vm.execute(list(score.tokens), loader.factor_tensor.to(device))
                .detach()
                .cpu()
                .numpy(),
                loader,
                fold,
                backtest_config,
            )
            if metrics.get("failed"):
                continue
            pairs.append(
                (
                    score.val_reward,
                    oos_active_ir(
                        metrics["daily_returns"],
                        metrics["benchmark_daily_returns"],
                    ),
                )
            )
        correlation = (
            reward_oos_correlation(pairs, seed=args.seed)
            if pairs
            else None
        )

        payload = {
            "fold": args.fold,
            "fold_train_end": fold.train_end,
            "fold_test_end": fold.test_end,
            "budget": budget,
            "seed": args.seed,
            "dataset_id": loader.dataset_id,
            **result.to_dict(),
            "reward_oos": correlation,
        }
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if correlation:
            print(
                f"reward<->OOS active IR: rho={correlation['rho']:.3f} "
                f"p={correlation['p_value']:.4f} n={correlation['n']} "
                f"stable_positive={correlation['positive_stable']}"
            )
        return 0
    finally:
        export_log_txt(run_name="baseline_harness")


if __name__ == "__main__":
    raise SystemExit(main())
