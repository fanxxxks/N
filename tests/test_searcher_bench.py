"""Tests for the searcher cost benchmark (P1-04/P1-05).

Contract (docs/p4_search_transformer_contract.md §3,
SEARCHER_BENCH_VERSION 4):

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
from ashare_data.db import AshareDB
from ashare_data.manifest import resolve_dataset_id
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
    with AshareDB(populated_db.duckdb_path) as db:
        expected_dataset_id = resolve_dataset_id(db, populated_db)
    assert prov["dataset_id"] == expected_dataset_id
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


# --- P14 §5.3/§5.4: length prior (random path) + research/promotion tracks --
# Contract: docs/p14_search_digest_preregistration.md — the per-step EOS
# prior (profile p14-uniform-2-11-v1) applies to the random sampler's
# proposal distribution (cap-content fraction ≤ 25%, short-content coverage
# ≥ 40%), and the v4 campaign separates the research (full vocabulary) and
# promotion_tier_a (tier-A samplable vocabulary) tracks with per-track
# budgets and a fail-closed tier-purity validation.
# (p14 symbols are imported lazily inside each test so the RED run of this
#  file contains only the expected new-test failures.)


def test_p14_length_prior_random_proposal_distribution():
    from ashare_model.search_length_prior import LENGTH_PRIOR_PROFILE
    from ashare_model.train_windows import sample_random_formulas
    from ashare_model.vocab import FORMULA_VOCAB

    assert LENGTH_PRIOR_PROFILE == "p14-uniform-2-11-v1"
    seqs = sample_random_formulas(
        seed=7, vocab=FORMULA_VOCAB, max_len=12, n=2000
    )
    assert len(seqs) == 2000
    eos = FORMULA_VOCAB.eos_token_id
    assert all(eos in seq for seq in seqs)  # still EOS-terminated
    assert all(len(seq) <= 12 for seq in seqs)  # cap unchanged
    contents = [seq.index(eos) for seq in seqs]
    cap = sum(1 for c in contents if c == 11) / len(contents)
    short = sum(1 for c in contents if c <= 8) / len(contents)
    assert cap <= 0.25, f"cap stacking unchanged: {cap:.3f}"
    assert short >= 0.40, f"no short-formula coverage: {short:.3f}"


def test_p14_tier_a_feature_ids_come_from_the_single_tier_authority():
    from ashare_model.data_tier import feature_tier
    from ashare_model.searcher_bench import tier_a_feature_ids
    from ashare_model.vocab import FORMULA_VOCAB

    ids = tier_a_feature_ids(FORMULA_VOCAB)
    names = {
        FORMULA_VOCAB.feature_names[token - FORMULA_VOCAB.feature_offset]
        for token in ids
    }
    assert len(ids) == 41  # p14 appendix A: samplable tier-A features
    assert all(feature_tier(name).value == "A" for name in names)
    assert not (names & set(FORMULA_VOCAB.deprecated_names))


def test_p14_campaign_runs_research_and_promotion_tracks(
    populated_db: DataConfig, tmp_path
):
    from ashare_model.search_length_prior import LENGTH_PRIOR_PROFILE
    from ashare_model.searcher_bench import (
        P14_TRACKS,
        SEARCHERS as BENCH_SEARCHERS,
        benchmark_campaign,
        tier_a_feature_ids,
    )
    from ashare_model.vocab import FORMULA_VOCAB

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    payload = benchmark_campaign(
        populated_db,
        model_config,
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        _reward_cfg(),
        loader,
        seeds=(42, 7),
        budget=16,
        research_budget=10,
        promotion_budget=6,
        rl_steps=2,
        train_end_date="2024-02-01",
        window_cap=(3, 30),
        fold_index=0,
        device="cpu",
        run_dir=tmp_path / "run",
    )
    assert payload["campaign_status"] == "completed"
    rows = payload["rows"]
    assert [
        (row["seed"], row["track"], row["searcher"]) for row in rows
    ] == [
        (seed, track, searcher)
        for seed in (42, 7)
        for track in P14_TRACKS
        for searcher in BENCH_SEARCHERS
    ]
    for row in rows:
        assert row["length_prior_profile"] == LENGTH_PRIOR_PROFILE
        if row["track"] == "research":
            assert row["requested_budget"] == 10
        else:
            assert row["requested_budget"] == 6
            assert row["tier_restriction"]["tier"] == "A"
    # The A-track restriction is real: promotion rows carry the tier-A
    # feature count from the single tier authority.
    promotion_rows = [row for row in rows if row["track"] == "promotion_tier_a"]
    assert {row["tier_restriction"]["feature_count"] for row in promotion_rows} == {
        len(tier_a_feature_ids(FORMULA_VOCAB))
    }


def test_p14_campaign_rejects_mismatched_budget_split(
    populated_db: DataConfig, tmp_path
):
    from ashare_model.searcher_bench import benchmark_campaign

    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    model_config = ModelConfig(batch_size=2, train_steps=1, max_formula_len=4)
    with pytest.raises(ValueError, match="budget"):
        benchmark_campaign(
            populated_db,
            model_config,
            BacktestConfig(top_n=2, train_end_date="2024-02-01"),
            _reward_cfg(),
            loader,
            seeds=(42,),
            budget=16,
            research_budget=9,
            promotion_budget=6,
            rl_steps=8,
            train_end_date="2024-02-01",
            window_cap=(3, 30),
            fold_index=0,
            device="cpu",
            run_dir=tmp_path / "run",
        )


def test_p14_promotion_tier_purity_validation_is_fail_closed():
    """Any billed candidate reaching beyond tier A fails the promotion row
    (p14 §5.4.4): the validator rejects a C-tier formula and accepts an
    A-tier one."""
    import types

    from ashare_model import searcher_bench as sb
    from ashare_model.vocab import FORMULA_VOCAB

    vocab = FORMULA_VOCAB
    ind_rel = vocab.feature_offset + vocab.feature_names.index("IND_REL_RET_5")
    ret1 = vocab.feature_offset + vocab.feature_names.index("RET_1")
    eos = vocab.eos_token_id
    c_row = types.SimpleNamespace(
        scores=[
            types.SimpleNamespace(tokens=(ind_rel, eos)),
            types.SimpleNamespace(tokens=(ret1, eos)),
        ]
    )
    with pytest.raises(ValueError, match="tier"):
        sb._validate_promotion_tier_purity(c_row, vocab)
    a_row = types.SimpleNamespace(scores=[types.SimpleNamespace(tokens=(ret1, eos))])
    sb._validate_promotion_tier_purity(a_row, vocab)  # must not raise


def test_p14_constants_match_the_preregistered_split():
    from ashare_model.search_length_prior import LENGTH_PRIOR_PROFILE
    from ashare_model.searcher_bench import (
        P14_PROMOTION_BUDGET,
        P14_RESEARCH_BUDGET,
        P14_TRACKS,
    )

    assert P14_RESEARCH_BUDGET == 1200
    assert P14_PROMOTION_BUDGET == 800
    assert P14_RESEARCH_BUDGET + P14_PROMOTION_BUDGET == 2000
    assert P14_TRACKS == ("research", "promotion_tier_a")
    assert LENGTH_PRIOR_PROFILE == "p14-uniform-2-11-v1"
