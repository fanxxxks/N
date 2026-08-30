"""P4 elite archive and supervised-imitation contracts.

Assertion source: ``docs/p4_search_transformer_contract.md`` section 4.
"""

from __future__ import annotations

import json

import pytest
import torch

from ashare_data.config import ModelConfig
from ashare_model.alphagpt import AlphaGPTModel
from ashare_model.candidates import CandidateScore
from ashare_model.elite_archive import (
    ELITE_ARCHIVE_VERSION,
    EliteArchive,
    build_elite_archive,
    load_elite_archive,
    merge_elite_archives,
    write_elite_archive,
)
from ashare_model.vocab import FORMULA_VOCAB


def _score(
    token: int,
    *,
    source: str,
    val_reward: float,
    train_reward: float,
    eligible: bool = True,
    complexity_cost: float = 1.0,
) -> CandidateScore:
    tokens = (
        token,
        FORMULA_VOCAB.eos_token_id,
        *([FORMULA_VOCAB.pad_token_id] * 10),
    )
    return CandidateScore(
        tokens=tokens,
        candidate_id=f"{source}:{token}",
        formula_text=f"F{token}",
        source=source,
        direction=1,
        val_reward=val_reward,
        val_icir=1.0,
        train_reward=train_reward,
        train_icir=1.0,
        complexity_penalty=0.0,
        complexity_cost=complexity_cost,
        eligible=eligible,
        rejection_reasons=(),
    )


def test_elite_archive_filters_ranks_deduplicates_and_round_trips():
    high = _score(2, source="gp", val_reward=0.8, train_reward=0.6)
    duplicate_low = _score(2, source="gp", val_reward=0.7, train_reward=0.9)
    low = _score(3, source="gp", val_reward=0.5, train_reward=0.4)
    rejected = _score(
        4, source="gp", val_reward=9.0, train_reward=9.0, eligible=False
    )
    archive = build_elite_archive(
        "gp", [low, rejected, duplicate_low, high], max_size=2
    )
    assert archive.version == ELITE_ARCHIVE_VERSION == 1
    assert archive.source_backends == ("gp",)
    assert [entry.val_reward for entry in archive.entries] == [0.8, 0.5]
    assert archive.entries[0].tokens == high.tokens

    restored = EliteArchive.from_dict(json.loads(json.dumps(archive.to_dict())))
    assert restored == archive
    payload = archive.to_dict()
    payload["version"] = 999
    with pytest.raises(ValueError, match="unsupported elite archive version"):
        EliteArchive.from_dict(payload)


def test_merge_archive_keeps_baseline_sources_and_deterministic_top_k():
    gp = build_elite_archive(
        "gp", [_score(2, source="gp", val_reward=0.5, train_reward=0.4)]
    )
    tpe = build_elite_archive(
        "tpe", [_score(3, source="tpe", val_reward=0.7, train_reward=0.4)]
    )
    random = build_elite_archive(
        "random", [_score(4, source="random", val_reward=0.6, train_reward=0.4)]
    )
    merged = merge_elite_archives([gp, tpe, random], max_size=2)
    assert merged.source_backends == ("gp", "tpe", "random")
    assert [entry.val_reward for entry in merged.entries] == [0.7, 0.6]


def test_elite_archive_file_round_trip_and_unknown_version_rejection(tmp_path):
    archive = build_elite_archive(
        "random",
        [_score(2, source="random", val_reward=0.5, train_reward=0.4)],
    )
    path = write_elite_archive(tmp_path / "elite.json", archive)
    assert load_elite_archive(path) == archive
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported elite archive version"):
        load_elite_archive(path)


def test_imitation_teacher_forcing_reduces_loss_on_baseline_elites():
    from ashare_model.imitation import IMITATION_VERSION, pretrain_on_elites

    torch.manual_seed(5)
    config = ModelConfig(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        num_loops=1,
        dropout=0.0,
        max_formula_len=12,
    )
    model = AlphaGPTModel(config)
    archive = build_elite_archive(
        "gp",
        [
            _score(2, source="gp", val_reward=0.8, train_reward=0.7),
            _score(3, source="gp", val_reward=0.7, train_reward=0.6),
        ],
    )
    result = pretrain_on_elites(
        model,
        archive,
        max_formula_len=12,
        epochs=12,
        batch_size=2,
        learning_rate=0.01,
        seed=5,
    )
    assert result.version == IMITATION_VERSION == 1
    assert result.sample_count == 2
    assert result.token_count == 24
    assert result.final_loss < result.initial_loss
    assert result.final_token_accuracy > result.initial_token_accuracy


def test_imitation_refuses_empty_or_nonbaseline_archive():
    from ashare_model.imitation import pretrain_on_elites

    config = ModelConfig(
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        num_loops=1,
        dropout=0.0,
        max_formula_len=12,
    )
    model = AlphaGPTModel(config)
    empty = EliteArchive(source_backends=("gp",), entries=())
    with pytest.raises(ValueError, match="non-empty"):
        pretrain_on_elites(model, empty, max_formula_len=12)
    rl_archive = build_elite_archive(
        "rl", [_score(2, source="rl", val_reward=0.8, train_reward=0.7)]
    )
    with pytest.raises(ValueError, match="gp.*tpe.*random"):
        pretrain_on_elites(model, rl_archive, max_formula_len=12)
