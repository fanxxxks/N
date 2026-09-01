"""P10 four-searcher matched comparison campaign tests.

Contract source: docs/p10_searcher_fairness_contract.md —

* §4.3 / §10-1: GP node cap aligned to the shared EOS-inclusive token
  budget (``_max_nodes == max_formula_len - 1``); legacy-cap trees keep
  their exact token parity (§10-7).
* §10-2: every backend's evaluated formulas fit the same EOS-inclusive
  cap (``len(tokens) + 1 <= max_formula_len``).
* §4.1 / §4.4 / §6 / §7: the campaign runs seeds × (gp, tpe, random, rl)
  with a per-row requested unique-semantic-evaluation budget, records
  requested/consumed/proposal/invalid/semantic-duplicate/termination/
  stagnation/length-histogram per row, keeps every row in the
  append-only ledger, supports resume, and fail-closes on wall-cap,
  calibration-deviation and identity-drift triggers.
* §9: SEARCH_CONTRACT_VERSION 3, SEARCHER_BENCH_VERSION 3 (P10 bumps;
  pre/post results are never matched comparisons).
"""
from __future__ import annotations

import json

import pytest

from ashare_data.config import BacktestConfig, ModelConfig, RewardConfig
from ashare_model.data_loader import AshareDataLoader
from ashare_model.gp_search import _max_nodes, tokens_to_tree, tree_to_tokens
from ashare_model.ledger import ExperimentLedger
from ashare_model.search_contract import SEARCH_CONTRACT_VERSION
from ashare_model.searcher_bench import (
    P10_ROW_ORDER,
    SEARCHER_BENCH_VERSION,
    benchmark_campaign,
    benchmark_searchers,
)
from ashare_model.train import AshareTrainer
from ashare_model.vocab import FORMULA_VOCAB

SEARCHERS = ("gp", "tpe", "random", "rl")


def _reward_cfg() -> RewardConfig:
    return RewardConfig(
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


def _model_cfg(max_len: int) -> ModelConfig:
    return ModelConfig(batch_size=2, train_steps=1, max_formula_len=max_len)


def _campaign(populated_db, run_dir, *, seeds=(42, 7), budget: int = 16,
              rl_steps: int = 8, max_len: int = 4, **kwargs):
    loader = AshareDataLoader(populated_db, _model_cfg(max_len))
    loader.load_data()
    return benchmark_campaign(
        populated_db,
        _model_cfg(max_len),
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        _reward_cfg(),
        loader,
        seeds=tuple(seeds),
        budget=budget,
        rl_steps=rl_steps,
        train_end_date="2024-02-01",
        window_cap=(3, 30),
        fold_index=0,
        device="cpu",
        run_dir=run_dir,
        **kwargs,
    )


def _unary_chain(n_ops: int) -> list[int]:
    """Postfix token chain ``MA5(MA5(...(feature)))`` with ``n_ops``
    unary operators over the first vocabulary feature (n_ops + 1 nodes,
    each node exactly one token — the invariant pinned by the semantic
    sampling property tests)."""
    vocab = FORMULA_VOCAB
    feature_token = vocab.feature_offset + 0
    ma5 = vocab.operator_offset + vocab.operator_names.index("MA5")
    return [feature_token] + [ma5] * n_ops


# ---------------------------------------------------------------------------
# §9 version pins
# ---------------------------------------------------------------------------


def test_p10_bumps_search_contract_and_bench_versions():
    # P10 §9: GP node-cap alignment changes the matched effective space
    # (SEARCH_CONTRACT 2->3); the campaign seed-list runner changes the
    # bench schema (SEARCHER_BENCH 2->3). Pre/post results are never
    # matched comparisons.
    assert SEARCH_CONTRACT_VERSION == 3
    assert SEARCHER_BENCH_VERSION == 3


# ---------------------------------------------------------------------------
# §4.3 / §10-1, §10-7: GP node cap alignment + legacy parity
# ---------------------------------------------------------------------------


def test_p10_gp_node_cap_equals_max_len_minus_one():
    for max_len in (2, 3, 4, 8, 12, 16):
        assert _max_nodes(max_len) == max_len - 1, max_len
    # The production cap: 11 nodes -> 11 content tokens + EOS = 12.
    assert _max_nodes(12) == 11


def test_p10_gp_can_express_formulas_at_the_shared_token_budget():
    """An 11-node formula (12 tokens incl. EOS) must be GP-legal at the
    production max_len of 12 — impossible under the pre-P10 cap of 6
    nodes, which under-restricted GP relative to the other three
    backends (contract §1.2 / §4.3)."""
    chain = _unary_chain(10)  # 11 nodes -> 12 tokens incl. EOS
    tree = tokens_to_tree(tuple(chain), FORMULA_VOCAB)
    assert tree is not None
    assert list(tree_to_tokens(tree, FORMULA_VOCAB)) == chain
    assert len(chain) <= _max_nodes(12)
    assert len(chain) + 1 <= 12


def test_p10_gp_roundtrip_parity_for_legacy_cap_trees():
    """Trees inside the old 6-node cap keep their exact token parity —
    historical GP formulas stay parseable (contract §10-7)."""
    chain = _unary_chain(3)  # 4 nodes, legal under the old cap too
    tree = tokens_to_tree(tuple(chain), FORMULA_VOCAB)
    assert tree is not None
    assert list(tree_to_tokens(tree, FORMULA_VOCAB)) == chain


# ---------------------------------------------------------------------------
# §10-2: unified EOS-inclusive cap over real proposals of all four backends
# ---------------------------------------------------------------------------


def test_p10_all_backends_evaluated_formulas_fit_shared_cap(populated_db):
    loader = AshareDataLoader(populated_db, _model_cfg(8))
    loader.load_data()
    payload = benchmark_searchers(
        populated_db,
        _model_cfg(8),
        BacktestConfig(top_n=2, train_end_date="2024-02-01"),
        _reward_cfg(),
        loader,
        searchers=SEARCHERS,
        budget=16,
        seed=42,
        train_end_date="2024-02-01",
        window_cap=(3, 30),
        device="cpu",
    )
    for searcher in SEARCHERS:
        row = payload["rows"][searcher]
        hist = row["formula_len_histogram"]
        assert hist, searcher  # evaluated formulas recorded
        # Every billed evaluation has a scored proposal (so the distinct
        # canonical count is >= the billed count), and every histogram
        # entry came from a proposal (so it is <= the proposal count).
        assert sum(hist.values()) >= row["unique_semantic_evals"], searcher
        assert sum(hist.values()) <= row["proposal_count"], searcher
        # Keys are canonical total lengths (content + EOS): never above
        # max_len — the shared EOS-inclusive cap (contract §4.3).
        assert max(int(k) for k in hist) <= 8, (searcher, hist)
        assert min(int(k) for k in hist) >= 2, (searcher, hist)


# ---------------------------------------------------------------------------
# §4.1 / §4.4 / §6 / §7: campaign structure, ledger, resume, circuit breakers
# ---------------------------------------------------------------------------


def test_p10_campaign_rows_ledger_and_identity(populated_db, tmp_path):
    payload = _campaign(populated_db, tmp_path / "run", seeds=(42, 7))
    assert payload["version"] == SEARCHER_BENCH_VERSION
    assert payload["campaign_status"] == "completed"
    assert payload["not_run"] == []
    rows = payload["rows"]
    assert [(r["seed"], r["searcher"]) for r in rows] == [
        (seed, name) for seed in (42, 7) for name in P10_ROW_ORDER
    ]
    for row in rows:
        assert row["completed"] is True
        assert row["requested_budget"] == 16
        assert 0 < row["consumed_budget"] <= row["requested_budget"]
        assert row["consumed_budget"] == row["unique_semantic_evals"]
        assert row["proposal_count"] >= row["consumed_budget"]
        assert row["semantic_duplicates"] >= 0
        assert row["invalid_proposals"] >= 0
        assert row["termination_reason"] in {
            "budget_exhausted",
            "steps_exhausted",
            "proposal_stagnation",
            "candidate_pool_exhausted",
            "no_eligible_candidate",
        }
        assert row["error"] is None
        assert row["wall_seconds"] >= 0.0
        assert isinstance(row["formula_len_histogram"], dict)
        assert row["formula_len_histogram"]
        assert row["stagnation_reason"] is None or isinstance(
            row["stagnation_reason"], str
        )
    # Identity: everything except the seed is identical across rows.
    identity_keys = (
        "requested_budget",
        "dataset_id",
        "window_cap",
        "train_end_date",
        "device",
        "max_formula_len",
    )
    first = rows[0]
    for row in rows[1:]:
        for key in identity_keys:
            assert row[key] == first[key], key
    assert {row["seed"] for row in rows} == {42, 7}

    # Append-only ledger: one closed trial per row, chain re-verified on load.
    ledger = ExperimentLedger(
        tmp_path / "run" / "ledger.jsonl", run_id=payload["run_id"]
    )
    trials = list(ledger.trials_for())
    assert len(trials) == len(rows)
    assert all(entry.status == "succeeded" for entry in trials)

    # The campaign JSON is persisted atomically and is JSON-serializable.
    on_disk = json.loads(
        (tmp_path / "run" / "campaign.json").read_text(encoding="utf-8")
    )
    assert on_disk["campaign_status"] == "completed"
    assert len(on_disk["rows"]) == len(rows)
    json.dumps(payload)


def test_p10_campaign_resume_retries_failed_rows(populated_db, tmp_path,
                                                 monkeypatch):
    real_train_search = AshareTrainer.train_search

    def failing_tpe(self, *, searcher, **kwargs):
        if searcher == "tpe":
            raise RuntimeError("p10 resume boom")
        return real_train_search(self, searcher=searcher, **kwargs)

    run_dir = tmp_path / "run"
    monkeypatch.setattr(AshareTrainer, "train_search", failing_tpe)
    first = _campaign(populated_db, run_dir, seeds=(42,))
    assert first["campaign_status"] == "failed"
    failed_rows = [r for r in first["rows"] if not r["completed"]]
    assert [r["searcher"] for r in failed_rows] == ["tpe"]

    monkeypatch.setattr(AshareTrainer, "train_search", real_train_search)
    resumed = _campaign(populated_db, run_dir, seeds=(42,))
    assert resumed["campaign_status"] == "completed"
    assert all(row["completed"] for row in resumed["rows"])
    assert resumed["run_id"] == first["run_id"]

    # The failed attempt stays in the append-only ledger; the retry is a
    # new trial for the same row; completed rows are never re-run.
    ledger = ExperimentLedger(run_dir / "ledger.jsonl", run_id=resumed["run_id"])
    tpe_entries = [
        e for e in ledger.iter_entries() if e.algorithm == "searcher:tpe"
    ]
    # Two trials (attempt + retry), each contributing a running entry
    # plus its terminal entry — nothing is rewritten in place.
    assert sorted(e.status for e in tpe_entries) == [
        "failed",
        "running",
        "running",
        "succeeded",
    ]
    tpe_trials = list(ledger.trials_for(algorithm="searcher:tpe"))
    assert sorted(e.status for e in tpe_trials) == ["failed", "succeeded"]
    gp_success = [
        e
        for e in ledger.iter_entries()
        if e.algorithm == "searcher:gp" and e.status == "succeeded"
    ]
    assert len(gp_success) == 1


def test_p10_campaign_wall_cap_stops_before_any_row(populated_db, tmp_path):
    payload = _campaign(
        populated_db, tmp_path / "run", seeds=(42,), wall_cap_s=0.0
    )
    assert payload["campaign_status"] == "stopped_wall_cap"
    assert payload["rows"] == []
    assert [(r["seed"], r["searcher"]) for r in payload["not_run"]] == [
        (42, name) for name in P10_ROW_ORDER
    ]
    ledger = ExperimentLedger(
        tmp_path / "run" / "ledger.jsonl", run_id=payload["run_id"]
    )
    assert list(ledger.trials_for()) == []


def test_p10_campaign_calibration_deviation_stops_after_seed_block(
    populated_db, tmp_path
):
    tiny_rates = {name: 1e-4 for name in SEARCHERS}
    payload = _campaign(
        populated_db,
        tmp_path / "run",
        seeds=(42, 7),
        calibration_s_per_eval=tiny_rates,
    )
    assert payload["campaign_status"] == "stopped_calibration_deviation"
    assert [(r["seed"], r["searcher"]) for r in payload["rows"]] == [
        (42, name) for name in P10_ROW_ORDER
    ]
    assert [(r["seed"], r["searcher"]) for r in payload["not_run"]] == [
        (7, name) for name in P10_ROW_ORDER
    ]


def test_p10_campaign_resume_rejects_identity_drift(populated_db, tmp_path):
    _campaign(populated_db, tmp_path / "run", seeds=(42,))
    with pytest.raises(RuntimeError, match="identity"):
        _campaign(populated_db, tmp_path / "run", seeds=(42,), budget=32)


def test_p10_campaign_payload_is_json_serializable(populated_db, tmp_path):
    payload = _campaign(populated_db, tmp_path / "run", seeds=(42,))
    json.dumps(payload)
