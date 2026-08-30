"""Tests for the searcher cost benchmark (P1-04/P1-05).

Contract (docs/p4_search_transformer_contract.md §3,
SEARCHER_BENCH_VERSION 2):

* Four searchers (gp / tpe / random / rl) run on the SAME capped window
  (``prepare_window`` window_cap), same nominal budget, same seed.
* Budget unit = unique semantic evaluation (T2-01 ledger).  Non-RL
  searchers get (steps=budget, batch_size=1); RL gets
  (steps=4, batch=budget/4) — budget >= 16 and budget % 4 == 0.
* Each row records requested/consumed budget, termination/stagnation,
  best-so-far, unique_semantic_evals, wall_seconds,
  wall_per_1000_evals (= wall / evals * 1000), peak_rss_mb (stdlib RSS
  polling; None only on unsupported platforms), completed and the
  selected validation reward.  The smoke acceptance (P1-05): all four
  searchers complete under the same small budget.
"""
from __future__ import annotations

import json
import time

import pytest

from ashare_data.config import BacktestConfig, DataConfig, ModelConfig, RewardConfig
from ashare_model.data_loader import AshareDataLoader
from ashare_model.searcher_bench import (
    SEARCHER_BENCH_VERSION,
    benchmark_searchers,
    current_rss_mb,
    measure_peak_rss,
    rl_split,
)
from ashare_model.train import AshareTrainer

SEARCHERS = ("gp", "tpe", "random", "rl")


def _reward_cfg(**kwargs):
    defaults = dict(
        ic_min_stocks=3,
        min_val_reward=-1e9,
        min_val_icir=-1e9,
        min_valid_ic_days=2,
        min_effective_stocks=2,
        min_coverage=0.0,
        min_activity=0.0,
        min_sign_stability=0.0,
        min_val_window_q25=-1e9,
    )
    defaults.update(kwargs)
    return RewardConfig(**defaults)


def _bench(populated_db: DataConfig, budget: int = 16, cap=(3, 30)):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    bt = BacktestConfig(top_n=2, train_end_date="2024-02-01")
    return benchmark_searchers(
        populated_db,
        model_config,
        bt,
        _reward_cfg(),
        loader,
        searchers=SEARCHERS,
        budget=budget,
        seed=42,
        train_end_date="2024-02-01",
        window_cap=cap,
        device="cpu",
    )


def test_rl_split_contract():
    assert rl_split(128) == (4, 32)
    assert rl_split(1000) == (4, 250)
    assert rl_split(16) == (4, 4)
    for bad in (0, 4, 15, 17, 1002):
        with pytest.raises(ValueError):
            rl_split(bad)


def test_current_rss_mb_positive_on_this_platform():
    rss = current_rss_mb()
    if rss is None:
        pytest.skip("RSS measurement unsupported on this platform")
    assert rss > 0.0


def test_measure_peak_rss_tracks_the_call():
    peak = measure_peak_rss(lambda: time.sleep(0.05))[1]
    if peak is None:
        pytest.skip("RSS measurement unsupported on this platform")
    assert peak > 0.0


def test_all_four_searchers_complete_under_small_budget(populated_db: DataConfig):
    """P1-05 smoke acceptance: every searcher completes under the same
    small budget on the capped window, one fold, one seed."""
    payload = _bench(populated_db, budget=16)
    assert payload["version"] == SEARCHER_BENCH_VERSION
    assert payload["provenance"]["budget"] == 16
    assert payload["provenance"]["seed"] == 42
    assert payload["provenance"]["window_cap"] == [3, 30]
    assert set(payload["rows"]) == set(SEARCHERS)
    for searcher in SEARCHERS:
        row = payload["rows"][searcher]
        assert row["completed"] is True, searcher
        assert row["unique_semantic_evals"] >= 1, searcher
        assert row["requested_budget"] == 16, searcher
        assert row["consumed_budget"] == row["unique_semantic_evals"], searcher
        assert row["consumed_budget"] <= row["requested_budget"], searcher
        assert row["termination_reason"] in {
            "budget_exhausted",
            "steps_exhausted",
            "proposal_stagnation",
            "candidate_pool_exhausted",
            "no_eligible_candidate",
        }, searcher
        assert isinstance(row["best_so_far"], list), searcher
        assert row["search_result"]["backend"] == searcher
        assert row["wall_seconds"] >= 0.0, searcher
        per_1000 = row["wall_per_1000_evals"]
        assert per_1000 == pytest.approx(
            row["wall_seconds"] / row["unique_semantic_evals"] * 1000.0,
            rel=1e-9,
        )
        assert per_1000 >= 0.0, searcher
        assert row["error"] is None, searcher
        assert row["steps"] >= 1 and row["batch_size"] >= 1, searcher


def test_benchmark_records_same_nominal_budget_for_every_searcher(
    populated_db: DataConfig,
):
    payload = _bench(populated_db, budget=16)
    for searcher in SEARCHERS:
        assert payload["rows"][searcher]["budget"] == 16


def test_benchmark_rejects_invalid_budget(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    with pytest.raises(ValueError):
        benchmark_searchers(
            populated_db,
            model_config,
            BacktestConfig(top_n=2, train_end_date="2024-02-01"),
            _reward_cfg(),
            loader,
            searchers=SEARCHERS,
            budget=15,
            seed=42,
            train_end_date="2024-02-01",
            window_cap=(3, 30),
        )


def test_benchmark_payload_is_json_serializable(populated_db: DataConfig):
    payload = _bench(populated_db, budget=16)
    json.dumps(payload)  # must not raise


def test_benchmark_records_provenance(populated_db: DataConfig):
    payload = _bench(populated_db, budget=16)
    prov = payload["provenance"]
    assert prov["dataset_id"] is None  # synthetic DB has no manifest
    assert prov["train_end_date"] == "2024-02-01"
    assert prov["device"] == "cpu"
    assert prov["searchers"] == list(SEARCHERS)
    assert prov["max_formula_len"] == 4


def test_benchmark_failed_row_is_recorded_not_dropped(
    populated_db: DataConfig, monkeypatch
):
    """A crashing searcher yields a row (completed=False, error text,
    real wall time, None reward and per-1000) — never a raise and never
    a silent gap."""
    from ashare_model.train import AshareTrainer

    def boom(self, *args, **kwargs):
        raise RuntimeError("bench boom")

    monkeypatch.setattr(AshareTrainer, "train", boom)
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    payload = benchmark_searchers(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        _reward_cfg(),
        loader,
        searchers=("rl",),
        budget=16,
        seed=42,
        train_end_date="2024-02-01",
        window_cap=(3, 30),
        device="cpu",
    )
    row = payload["rows"]["rl"]
    assert row["completed"] is False
    assert "bench boom" in row["error"]
    assert row["wall_seconds"] > 0.0  # the failed attempt is still timed
    assert row["wall_per_1000_evals"] is None  # only meaningful on success
    assert row["selected_val_reward"] is None  # never a non-finite JSON value
    json.dumps(payload)  # must stay JSON-serializable


def test_train_search_bills_the_trainers_semantic_cache(populated_db: DataConfig):
    """P1-04 measurement fix: the evaluator shares the trainer's semantic
    cache, so ``trainer.semantic_cache.budget_used`` is the real unique-
    semantic-evaluation ledger for every non-RL backend (it used to stay
    0 for gp/tpe/random, which made the protocol's trained-row evals 0)."""
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    for searcher in ("gp", "tpe", "random"):
        trainer = AshareTrainer(
            populated_db,
            model_config,
            BacktestConfig(top_n=2, train_end_date="2024-02-01"),
            loader,
            reward_config=_reward_cfg(),
        )
        trainer.train_search(
            searcher=searcher,
            steps=1,
            batch_size=2,
            save_artifacts=False,
        )
        assert 0 < trainer.semantic_cache.budget_used <= 2, searcher
