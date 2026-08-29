"""P4 paired admission and RL diagnostic contracts.

Assertion source: ``docs/p4_search_transformer_contract.md`` sections 3 and 5.
"""

from __future__ import annotations

import math

import pytest
import torch

from ashare_model.admission import (
    ADMISSION_RULE_VERSION,
    PAIR_SEEDS,
    decide_p4_admission,
    paired_seed_plan,
)
from ashare_model.candidates import CandidateScore
from ashare_model.rl_diagnostics import (
    RL_DIAGNOSTICS_VERSION,
    gradient_l2_norm,
    rejection_reason_counts,
    reward_distribution,
    summarize_rl_step,
)


def test_pair_seed_plan_uses_same_seed_within_pair_and_unique_seeds_across_pairs():
    assert len(PAIR_SEEDS) >= 5
    assert len(set(PAIR_SEEDS)) == len(PAIR_SEEDS)
    plan = paired_seed_plan()
    assert len(plan) == len(PAIR_SEEDS)
    for row, seed in zip(plan, PAIR_SEEDS):
        assert row["pair_seed"] == seed
        assert set(row["backend_seeds"]) == {
            "gp",
            "tpe",
            "random",
            "rl_random",
            "rl_imitation",
        }
        assert set(row["backend_seeds"].values()) == {seed}


def test_p4_admission_rule_requires_imitation_to_beat_random_rl_and_gp():
    verdict = decide_p4_admission(
        imitation_areas=[5, 5, 5, 5, 1],
        imitation_oos_irs=[5, 5, 5, 5, 1],
        random_rl_areas=[1, 1, 1, 1, 2],
        random_rl_oos_irs=[1, 1, 1, 1, 2],
        gp_areas=[2, 2, 2, 2, 3],
        gp_oos_irs=[2, 2, 2, 2, 3],
    )
    assert verdict["rule_version"] == ADMISSION_RULE_VERSION == 2
    assert verdict["rl_admitted"] is True
    assert verdict["advanced_rl_allowed"] is True
    assert verdict["default_searcher"] == "rl"


def test_p4_admission_failure_keeps_gp_and_forbids_advanced_rl():
    verdict = decide_p4_admission(
        imitation_areas=[5, 5, 5, 5, 1],
        imitation_oos_irs=[1, 1, 1, 1, 1],
        random_rl_areas=[1, 1, 1, 1, 2],
        random_rl_oos_irs=[2, 2, 2, 2, 2],
        gp_areas=[2, 2, 2, 2, 3],
        gp_oos_irs=[3, 3, 3, 3, 3],
    )
    assert verdict["rl_admitted"] is False
    assert verdict["advanced_rl_allowed"] is False
    assert verdict["default_searcher"] == "gp"


def test_p4_admission_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="aligned"):
        decide_p4_admission(
            imitation_areas=[1, 2],
            imitation_oos_irs=[1],
            random_rl_areas=[0, 0],
            random_rl_oos_irs=[0, 0],
            gp_areas=[0, 0],
            gp_oos_irs=[0, 0],
        )


def _rejected(reason: str) -> CandidateScore:
    return CandidateScore(
        tokens=(1,),
        candidate_id=reason,
        formula_text=reason,
        source="rl",
        direction=1,
        val_reward=0.0,
        val_icir=0.0,
        train_reward=0.0,
        train_icir=0.0,
        complexity_penalty=0.0,
        eligible=False,
        rejection_reasons=(reason,),
    )


def test_rl_diagnostic_schema_records_required_exact_metrics():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    distribution = reward_distribution(rewards)
    assert distribution == {
        "count": 4,
        "min": 1.0,
        "q25": 1.75,
        "median": 2.5,
        "q75": 3.25,
        "max": 4.0,
        "mean": 2.5,
        "std": pytest.approx(math.sqrt(1.25)),
    }
    reasons = rejection_reason_counts(
        [_rejected("low_icir"), _rejected("low_icir"), _rejected("constant")]
    )
    assert reasons == {"constant": 1, "low_icir": 2}
    summary = summarize_rl_step(
        rewards=rewards,
        advantages=torch.tensor([-1.0, -0.5, 0.5, 1.0]),
        entropy=0.7,
        gradient_norm=1.2,
        scores=[_rejected("low_icir")],
        formula_lengths=[2, 4],
        operator_names={"ADD", "RANK"},
        semantic_duplicates=1,
        proposal_count=4,
    )
    assert summary["version"] == RL_DIAGNOSTICS_VERSION == 1
    assert summary["reward_distribution"]["count"] == 4
    assert summary["rejection_reasons"] == {"low_icir": 1}
    assert summary["entropy"] == 0.7
    assert summary["semantic_duplicates"] == 1
    assert summary["semantic_duplicate_rate"] == 0.25
    assert summary["advantage_variance"] == pytest.approx(0.625)
    assert summary["gradient_norm"] == 1.2
    assert summary["formula_length"] == {
        "count": 2,
        "min": 2,
        "mean": 3.0,
        "max": 4,
    }
    assert summary["operator_coverage"] == ["ADD", "RANK"]


def test_gradient_norm_is_global_l2_without_clipping_or_mutation():
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parameter.grad = torch.tensor([6.0, 8.0])
    before = parameter.grad.clone()
    assert gradient_l2_norm([parameter]) == 10.0
    assert torch.equal(parameter.grad, before)
