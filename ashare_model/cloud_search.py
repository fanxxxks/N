"""Cloud worker: existing search, validation selection, frozen account test."""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def snapshot_probe(request, *, full=False):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/cloud_validate_data.py"), "--request", str(request)]
    if not full:
        command.append("--snapshot-only")
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
        env={**__import__("os").environ, "PYTHONUTF8": "1"})
    if process.returncode != 0:
        raise ValueError("read-only data validation refused: " + process.stdout[-4000:] + process.stderr[-2000:])
    return json.loads(process.stdout)


def account_config(settings, capital, window):
    from ashare_data.config import BacktestConfig
    from ashare_portfolio.constructor import validate_portfolio_config
    profile = settings["accounts"][str(int(capital))]
    cfg = BacktestConfig(**{**settings.get("backtest", {}), **profile,
        "initial_capital": capital, "start_date": window["start"], "end_date": window["test_end"],
        "train_end_date": window["train_end"], "target_horizon": settings["horizon"],
        "rebalance_frequency": "daily" if settings["horizon"] == 1 else f"every_{settings['horizon']}_days"})
    validate_portfolio_config(cfg)
    minimum = cfg.min_trade_amount or 0
    if minimum > capital * min(cfg.single_weight_cap, 1.0 / cfg.top_n):
        raise ValueError(f"configuration_infeasible: capital={capital}, holdings/cap/minimum")
    return cfg


def account_report(loader, signal, config, start_after, end):
    import numpy as np
    from ashare_model.backtest import AshareBacktestEngine
    from ashare_model.continuous_account import summarize_account
    from ashare_portfolio.rebalance import RebalancePolicy
    dates = list(loader.dates)
    lo = bisect_right(dates, start_after.replace("-", ""))
    hi = bisect_right(dates, end.replace("-", ""))
    if hi - lo < 4:
        raise ValueError("account evaluation interval too short")
    # Account starts from cash at this boundary; state then persists across
    # every daily period and year inside the interval. No yearly injection.
    raw = {k: v.numpy()[:, lo:hi] for k, v in loader.raw_data_cache.items()}
    mask = np.asarray(loader.universe_mask)[:, lo:hi]
    due = RebalancePolicy.from_config(config).rebalance_mask(dates[lo:hi])
    engine = AshareBacktestEngine(config)
    result = engine.run(signal[:, lo:hi], raw, list(loader.ts_codes), dates[lo:hi], mask, rebalance_mask=due)
    report = summarize_account(result, config.initial_capital, raw, list(loader.ts_codes))
    report["benchmark_return"] = report["theoretical_equal_weight_return"]
    report["net_excess_return"] = (1 + report["net_return"]) / (1 + report["benchmark_return"]) - 1
    report["benchmark_kind"] = "theoretical PIT universe equal-weight (not a replicable small account)"
    report["no_trade_reason"] = None
    if not result.trades:
        if not np.any(np.isfinite(signal[:, lo:hi]) & mask):
            report["no_trade_reason"] = "signal_missing"
        elif report["execution_rejections"] or any(report["constructor_suppression"].values()):
            report["no_trade_reason"] = "execution_or_price_constraints"
        else:
            report["no_trade_reason"] = "normal_no_trade"
    return report


def run_worker(request_path):
    # A full uncached identity re-hash occurs in a data-only process before
    # torch is imported. The loader repeats that SAME authority at load time.
    request = json.loads(request_path.read_text(encoding="utf-8"))
    settings = request["settings"]
    identity = request["identity"]
    window = identity["window"]
    output = Path(request["attempt_output"])
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from run_cloud_search import code_identity, file_sha, write_json
    started = time.monotonic()
    if code_identity() != identity["code"]:
        raise ValueError("code changed after task planning")
    for key in ("primary", "v2"):
        if file_sha(settings[key]) != identity["data"][key]["sha256"]:
            raise ValueError("input file changed after plan: " + settings[key])
    validation = snapshot_probe(request_path, full=True)
    write_json(output / "data_validation.json", validation)

    import numpy as np
    import torch
    from ashare_data.config import DataConfig, ModelConfig, RewardConfig
    from ashare_data.manifest import DatasetManifest
    from ashare_model.data_loader import AshareDataLoader
    from ashare_model.runspec import CompositeSnapshot, _capture_versions
    from ashare_model.train import AshareTrainer
    from ashare_model.search_backends import get_search_backend
    from ashare_model.search_contract import SearchRequest
    from ashare_model.elite_archive import build_elite_archive
    from ashare_model.ir import decode, operator_names
    from ashare_model.vm import StackVM
    from ashare_model.vocab import FORMULA_VOCAB
    from ashare_portfolio.execution_spec import execution_provenance

    torch.set_num_threads(settings["threads"])
    torch.set_num_interop_threads(1)
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested GPU is unavailable")
    main_config = account_config(settings, 50000, window)
    # Check all capital profiles before spending search compute.
    account_configs = {capital: account_config(settings, capital, window) for capital in (10000, 50000, 100000)}
    model_config = ModelConfig(**{**settings.get("model", {}), "rl_initialization": "random"})
    reward_config = RewardConfig(**settings.get("reward", {}))
    data_config = DataConfig(data_dir=output / "cache", duckdb_path=Path(settings["primary"]),
        v2_duckdb_path=Path(settings["v2"]), parquet_dir=output / "parquet",
        start_date=window["start"], end_date=window["test_end"], index_codes=settings["index_codes"],
        min_listed_sessions=settings.get("min_listed_sessions", 60))

    class IsolatedSnapshotLoader(AshareDataLoader):
        def confirm_input_snapshot(self):
            confirmed = snapshot_probe(request_path)
            self.dataset_id = confirmed["dataset_id"]
            self.dataset_identity_status = "confirmed"
            self._confirmed_snapshot = CompositeSnapshot(self.dataset_id, confirmed["publication_id"],
                DatasetManifest.from_dict(confirmed["primary_manifest"]),
                DatasetManifest.from_dict(confirmed["v2_manifest"]))

    loader = IsolatedSnapshotLoader(data_config, model_config)
    loader.load_data()
    if loader.dataset_id != settings["dataset_id"]:
        raise ValueError("loader dataset identity drift")
    if request.get("test_from"):
        source = Path(request["test_from"]["path"])
        if file_sha(source) != request["test_from"]["sha256"]:
            raise ValueError("frozen validation result changed")
        origin = json.loads(source.read_text(encoding="utf-8"))
        if origin["outcome"] != "frozen_validation_completed" or origin["test_accounts"]:
            raise ValueError("frequency selection source must be validation only")
        frozen = {**origin["selected"], "frequency_selection": identity["frequency_selection"], "identity": identity}
        write_json(output / "frozen_selection.json", frozen)
        write_json(output / "strategy.json", frozen)
        # Preserve the selected source artifacts separately; weights still are
        # observations and are never loaded to resume training.
        import shutil
        for name in ("model.pt", "candidates.json", "elite_archive.json", "search.json", "resolved_config.json"):
            shutil.copyfile(source.parent / name, output / name)
        vm = StackVM(FORMULA_VOCAB, universe_mask=torch.tensor(loader.universe_mask, dtype=torch.bool))
        vm.industry_codes = loader.industry_codes
        signal = vm.execute(frozen["candidate"]["tokens"], loader.factor_tensor)
        if signal is None:
            raise ValueError("frozen formula could not execute")
        signal = signal.numpy() * frozen["candidate"]["direction"]
        tests = [{"capital": capital, "account": account_report(loader, signal, config,
            window["validation_end"], window["test_end"]), "config": asdict(config)}
            for capital, config in account_configs.items()]
        write_json(output / "test_accounts.json", tests)
        write_json(output / "result.json", {**origin, "task_id": request["task_id"], "identity": identity,
            "outcome": "frozen_frequency_test_completed", "selected": frozen, "test_accounts": tests,
            "frequency_search_cost": identity["frequency_selection"]["search_cost"],
            "elapsed_seconds": time.monotonic() - started})
        return 0
    missing = set(settings["feature_names"]) - set(FORMULA_VOCAB.feature_names)
    if missing:
        raise ValueError(f"unknown feature names: {sorted(missing)}")
    feature_ids = [FORMULA_VOCAB.feature_names.index(n) + FORMULA_VOCAB.feature_offset for n in settings["feature_names"]]
    trainer = AshareTrainer(data_config, model_config, main_config, loader, reward_config,
        init_seed=identity["seed"], feature_ids=feature_ids, lifecycle_mode="engineering")
    budget = identity["budget"]
    attempts = int(budget * settings.get("attempt_multiplier", 20))
    # Keep every encountered canonical form for this bounded run. The default
    # LRU would otherwise forget old classes and bill a duplicate after eviction.
    trainer._REWARD_CACHE_CAP = max(trainer._REWARD_CACHE_CAP, attempts + 1)
    search_started = time.monotonic()
    curve = []
    def persist_progress(result, points):
        write_json(output / "search_progress.json", {"search": result, "curve": points,
            "search_seconds": time.monotonic() - search_started})
    if identity["method"] == "rl":
        batch = model_config.batch_size
        def progress(run, active_trainer):
            now = time.monotonic() - search_started
            known = len(curve)
            for x, score in run.best_so_far[known:]:
                curve.append({"effective_evaluations": x, "best_validation_score": score if math.isfinite(score) else None,
                    "elapsed_seconds": now})
            persist_progress({"requested_budget": budget, "consumed_budget": active_trainer.semantic_cache.budget_used,
                "proposal_count": run.proposal_count, "invalid_proposals": run.invalid_proposals,
                "semantic_duplicates": run.semantic_duplicates}, curve)
        trainer.train(steps=attempts, batch_size=batch, seed=identity["seed"],
            train_end_date=window["train_end"], device=settings["device"], save_artifacts=False,
            effective_budget=budget, proposal_limit=attempts,
            time_limit_s=settings["time_limit"] * 0.8, progress_callback=progress)
        result = trainer.search_result
    else:
        prepared = trainer.prepare_window(window["train_end"], torch.device(settings["device"]), None)
        evaluator = trainer._build_search_evaluator(searcher=identity["method"], window=prepared, budget=budget)
        # The cloud evaluator is eager: record time and budget per full evaluation.
        evaluator._chunk = 1
        original_propose = evaluator.propose
        class ProposalLimit(Exception):
            pass
        def propose(tokens):
            if evaluator._proposal_count >= attempts or time.monotonic() - search_started > settings["time_limit"] * 0.8:
                raise ProposalLimit()
            score, consumed = original_propose(tokens)
            for x, best in evaluator._best_so_far[len(curve):]:
                curve.append({"effective_evaluations": int(x), "best_validation_score": best if math.isfinite(best) else None,
                    "elapsed_seconds": time.monotonic() - search_started})
            persist_progress({"requested_budget": budget, "consumed_budget": evaluator.budget_used,
                "proposal_count": evaluator._proposal_count, "invalid_proposals": evaluator._n_invalid,
                "semantic_duplicates": evaluator._proposal_count - evaluator.budget_used - evaluator._n_invalid}, curve)
            return score, consumed
        evaluator.propose = propose
        try:
            if identity["method"] == "random":
                from ashare_model.train_windows import sample_random_formulas
                block = 0
                while evaluator.budget_used < budget:
                    # Raw proposals preserve the sampler's duplicate/invalid cost.
                    for tokens in sample_random_formulas(identity["seed"] + block, trainer.vocab,
                            model_config.max_formula_len, min(256, attempts), feature_ids=feature_ids):
                        evaluator.propose(tokens)
                        if evaluator.budget_used >= budget:
                            break
                    block += 1
                result = evaluator.finish(backend="random", seed=identity["seed"], termination_reason="budget_exhausted")
            else:
                result = get_search_backend(identity["method"]).search(SearchRequest(identity["seed"], budget,
                    model_config.max_formula_len), evaluator, vocab=trainer.vocab, feature_ids=feature_ids)
        except ProposalLimit:
            result = evaluator.finish(backend=identity["method"], seed=identity["seed"], termination_reason="steps_exhausted")
    search_seconds = time.monotonic() - search_started
    if result is None:
        raise ValueError("search returned no ledger")
    search = result.to_dict()
    search["best_so_far"] = [[x, s if math.isfinite(s) else None] for x, s in result.best_so_far]
    search["semantic_duplicates"] = max(0, result.proposal_count - result.consumed_budget - result.invalid_proposals)
    search["stop_reason"] = ("budget_exhausted" if result.consumed_budget >= budget else
        "time_limit" if search_seconds >= settings["time_limit"] * 0.8 else result.termination_reason)
    write_json(output / "search.json", search)
    write_json(output / "candidates.json", [s.to_dict() for s in result.scores])
    archive = build_elite_archive(identity["method"], result.scores, max_size=settings.get("candidates", 15))
    write_json(output / "elite_archive.json", archive.to_dict())
    # Model weights are an observation artifact, never an automatic resume point.
    torch.save({"state_dict": trainer.model.state_dict(), "initialization": "random",
        "complete_resume_checkpoint": False}, output / "model.pt")
    versions = _capture_versions()
    write_json(output / "resolved_config.json", {"settings": settings, "versions": versions,
        "model": asdict(model_config), "reward": asdict(reward_config),
        "accounts": {str(k): asdict(v) for k, v in account_configs.items()},
        **execution_provenance(main_config), "lifecycle_mode": "engineering",
        "research_claim": "historical development comparison; no promotion or significance claim",
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda, "torch_threads": torch.get_num_threads(),
            "duckdb_threads": __import__("os").environ.get("ASHARE_DUCKDB_THREADS"),
            "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "policy_device": str(next(trainer.model.parameters()).device), "factor_device": settings["device"]}})
    eligible = sorted((s for s in result.scores if s.eligible and s.tokens and operator_names(decode(s.tokens, trainer.vocab))),
        key=lambda s: (-s.val_reward, s.complexity_cost, s.deterministic_key))[:settings.get("candidates", 15)]
    validation_rows = []
    vm = StackVM(trainer.vocab, universe_mask=torch.tensor(loader.universe_mask, dtype=torch.bool))
    vm.industry_codes = loader.industry_codes
    # Validation sees only a prefix ending at validation_end. Operators cannot
    # see a test column even if an operator implementation later changes.
    val_hi = bisect_right(loader.dates, window["validation_end"].replace("-", ""))
    for candidate in eligible:
        val_vm = StackVM(trainer.vocab, universe_mask=torch.tensor(loader.universe_mask[:, :val_hi], dtype=torch.bool))
        val_vm.industry_codes = loader.industry_codes[:, :val_hi] if loader.industry_codes is not None else None
        signal = val_vm.execute(candidate.tokens, loader.factor_tensor[:, :, :val_hi])
        if signal is None:
            continue
        padded = np.full(loader.universe_mask.shape, np.nan)
        padded[:, :val_hi] = signal.numpy() * candidate.direction
        report = account_report(loader, padded, main_config, window["train_end"], window["validation_end"])
        validation_rows.append({"candidate": candidate.to_dict(), "account": report})
    write_json(output / "validation_accounts.json", validation_rows)
    viable = [r for r in validation_rows if r["account"]["actual_order_count"] > 0 and r["account"]["valuation_complete"]
              and r["account"]["net_excess_return"] > settings.get("minimum_validation_excess", 0.0)]
    outcome = "no_qualified_candidate"
    tests = []
    if viable:
        winner = sorted(viable, key=lambda r: (-r["account"]["net_excess_return"],
            r["candidate"]["complexity_cost"], r["candidate"]["candidate_id"]))[0]
        frozen = {"candidate": winner["candidate"], "selected_by": "validation_net_excess_50000",
            "validation_account": winner["account"], "frequency": main_config.rebalance_frequency,
            "target_horizon": main_config.target_horizon, "identity": identity, "frozen_before_test": True}
        write_json(output / "frozen_selection.json", frozen)
        write_json(output / "strategy.json", frozen)
        if settings.get("evaluation_mode") == "validation_only":
            outcome = "frozen_validation_completed"
        else:
            candidate = winner["candidate"]
            signal = vm.execute(candidate["tokens"], loader.factor_tensor).numpy() * candidate["direction"]
            for capital, config in account_configs.items():
                tests.append({"capital": capital, "account": account_report(loader, signal, config,
                    window["validation_end"], window["test_end"]), "config": asdict(config)})
            outcome = "frozen_test_completed"
    write_json(output / "test_accounts.json", tests)
    payload = {"task_id": request["task_id"], "identity": identity, "method": identity["method"],
        "seed": identity["seed"], "window": window, "historical_research": True,
        "outcome": outcome, "no_candidate_reason": None if viable else
        {"search_rejections": result.rejection_reasons, "eligible_shortlist": len(eligible),
         "validation_tradable_and_profitable": len(viable)},
        "search": search, "curve": curve, "search_seconds": search_seconds,
        "elapsed_seconds": time.monotonic() - started, "test_accounts": tests,
        "selected": viable and json.loads((output / "frozen_selection.json").read_text(encoding="utf-8")) or None}
    write_json(output / "result.json", payload)
    return 0
