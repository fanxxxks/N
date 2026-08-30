"""P0-04 contract tests: legacy artifact classification, stamping, API.

Old artifacts (reward-v10 strategy, protocol-v12 result) must be clearly
marked legacy and never presented as the current champion: the pure
classification rules, the idempotent stamping migration, the web API's
computed flag and the research doctor's classification surface.
"""

from __future__ import annotations

from dataclasses import replace
import json

from ashare_data.config import BacktestConfig
from ashare_model.alphagpt import MODEL_VERSION
from ashare_model.imitation import IMITATION_VERSION
from ashare_model.bare_factor_backtest import BARE_FACTOR_BACKTEST_VERSION
from ashare_model.evaluation import PROTOCOL_VERSION
from ashare_model.reward import REWARD_VERSION
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import execution_provenance
from ashare_portfolio.golden import EXECUTION_SPEC_VERSION
from ashare_model.artifact_versions import (
    classify_artifact,
    classify_bare_factor,
    classify_protocol,
    classify_strategy,
    stamp_legacy_artifacts,
)
from ashare_model.research_domain import RESEARCH_DOMAIN_VERSION
from ashare_model import research_doctor as doctor


def _current_strategy() -> dict:
    return {
        "formula": [1, 2],
        "searcher": "gp",
        "reward_version": REWARD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_version": MODEL_VERSION,
        "dataset_id": "d1",
        "direction": 1,
        **execution_provenance(BacktestConfig()),
    }


def _v10_strategy() -> dict:
    return {
        "formula": [1, 2],
        "direction": -1,
        "reward_version": "10",
        # no searcher / protocol_version / model_version / dataset_id
    }


def _current_protocol() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "reward_version": REWARD_VERSION,
        "dataset_id": "d1",
        "stitched": {"x": 1},
        "ledger": {"path": "data/experiment_ledger.jsonl"},
        # P6: current-generation protocol artifacts record the research
        # domain (docs/p6_research_domain_contract.md §4.4).
        "research_domain": "unified",
        "research_domain_version": RESEARCH_DOMAIN_VERSION,
        **execution_provenance(BacktestConfig()),
    }


def _current_bare_factor() -> dict:
    base = BacktestConfig()
    quadrants = []
    for frequency, method in (
        ("daily", "equal_weight"),
        ("daily", "optimizer"),
        ("weekly", "equal_weight"),
        ("weekly", "optimizer"),
    ):
        config = replace(
            base,
            rebalance_frequency=frequency,
            target_horizon=1,
            portfolio_method=method,
        )
        quadrants.append(
            {
                "frequency": frequency,
                "horizon": 1,
                "method": method,
                **execution_provenance(config),
                "factors": [
                    {
                        "name": "TURNOVER",
                        "direction": 1,
                        "frequency": frequency,
                        "horizon": 1,
                        "method": method,
                        "total_return": 0.0,
                        "annual_return": 0.0,
                        "sharpe": 0.0,
                        "sortino": 0.0,
                        "max_drawdown": 0.0,
                        "turnover": 0.0,
                        "order_count": 0,
                        "cost": 0.0,
                    }
                ],
            }
        )
    return {
        "version": BARE_FACTOR_BACKTEST_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "reward_version": REWARD_VERSION,
        "dataset_id": "d1",
        **execution_provenance(base),
        "quadrants": quadrants,
    }


def _v12_protocol() -> dict:
    return {
        "protocol_version": "12",
        "reward_version": "10",
        # no dataset_id / stitched / ledger
    }


def test_current_strategy_is_not_legacy():
    verdict = classify_strategy(_current_strategy())
    assert verdict == {"legacy": False, "reasons": []}


def test_current_semantic_versions_are_pinned():
    # P6 contract §5 bumped PROTOCOL_VERSION 23 -> 24 (research-domain
    # dimension); P7-E contract §4 bumps 24 -> 25 (semantic-type sampling
    # legality narrows the candidate pool; measurement semantics
    # unchanged).  Reward and execution semantics stay at their P3
    # generations, and the bare-factor schema stays at P3 v3.
    assert REWARD_VERSION == "14"
    assert PROTOCOL_VERSION == "25"
    assert EXECUTION_SPEC_VERSION == 2
    assert BARE_FACTOR_BACKTEST_VERSION == 3


def test_pre_p7e_strategy_is_legacy_by_protocol_generation():
    """P7-E migration policy: old formulas remain readable/executable,
    but a v24 candidate-pool artifact cannot claim current status."""
    payload = _current_strategy()
    payload["protocol_version"] = "24"

    verdict = classify_strategy(payload)

    assert verdict["legacy"] is True
    assert (
        f"protocol_version 24 != current {PROTOCOL_VERSION}"
        in verdict["reasons"]
    )


def test_strategy_without_execution_provenance_is_legacy():
    payload = _current_strategy()
    payload.pop("execution_version")
    payload.pop("portfolio_config")
    verdict = classify_strategy(payload)
    assert verdict["legacy"] is True
    reasons = "; ".join(verdict["reasons"])
    assert "no execution_version (pre-P3)" in reasons
    assert "no portfolio_config (pre-P3)" in reasons


def test_partial_portfolio_config_cannot_claim_current():
    payload = _current_strategy()
    payload["portfolio_config"] = {"portfolio_method": "equal_weight"}
    verdict = classify_strategy(payload)
    assert verdict["legacy"] is True
    assert any("portfolio_config" in reason for reason in verdict["reasons"])


def test_strategy_without_reward_version_is_legacy():
    payload = _current_strategy()
    payload.pop("reward_version")
    verdict = classify_strategy(payload)
    assert verdict["legacy"] is True
    assert "no reward_version" in "; ".join(verdict["reasons"])


def test_v2_checkpoint_and_random_rl_artifacts_cannot_claim_current():
    v2 = _current_strategy()
    v2["model_version"] = 2
    verdict = classify_strategy(v2)
    assert verdict["legacy"] is True
    assert f"model_version 2 != current {MODEL_VERSION}" in verdict["reasons"]

    random_rl = _current_strategy()
    random_rl.update(
        {
            "searcher": "rl",
            "rl_initialization": "random",
            "imitation": None,
            "experimental": True,
        }
    )
    verdict = classify_strategy(random_rl)
    assert verdict["legacy"] is True
    assert "RL strategy is not elite-imitation initialized" in verdict["reasons"]
    assert "RL strategy has no imitation provenance" in verdict["reasons"]


def test_v3_imitation_rl_artifact_has_current_provenance():
    imitation_rl = _current_strategy()
    imitation_rl.update(
        {
            "searcher": "rl",
            "rl_initialization": "imitation",
            "imitation": {"version": IMITATION_VERSION},
            "experimental": False,
        }
    )
    assert classify_strategy(imitation_rl) == {"legacy": False, "reasons": []}


def test_v10_strategy_is_legacy_with_reasons():
    verdict = classify_strategy(_v10_strategy())
    assert verdict["legacy"] is True
    reasons = "; ".join(verdict["reasons"])
    assert f"reward_version 10 != current {REWARD_VERSION}" in reasons
    assert "no searcher field (pre-T2-03)" in reasons
    assert "no protocol_version (pre-T2-01)" in reasons
    assert "no model_version" in reasons
    assert "no dataset_id (pre-T1-01)" in reasons


def test_current_protocol_is_not_legacy():
    assert classify_protocol(_current_protocol()) == {
        "legacy": False,
        "reasons": [],
    }


def test_protocol_without_execution_provenance_is_legacy():
    payload = _current_protocol()
    payload.pop("execution_version")
    payload.pop("portfolio_config")
    verdict = classify_protocol(payload)
    assert verdict["legacy"] is True
    reasons = "; ".join(verdict["reasons"])
    assert "no execution_version (pre-P3)" in reasons
    assert "no portfolio_config (pre-P3)" in reasons


def test_protocol_without_semantic_versions_is_legacy():
    payload = _current_protocol()
    payload.pop("protocol_version")
    payload.pop("reward_version")
    verdict = classify_protocol(payload)
    assert verdict["legacy"] is True
    reasons = "; ".join(verdict["reasons"])
    assert "no protocol_version" in reasons
    assert "no reward_version" in reasons


def test_protocol_missing_domain_is_legacy():
    # P6 contract §4.4: a current-generation protocol artifact without the
    # research_domain field predates the domain dimension and is legacy.
    payload = _current_protocol()
    assert classify_protocol(payload)["legacy"] is False
    payload.pop("research_domain")
    verdict = classify_protocol(payload)
    assert verdict["legacy"] is True
    assert any("research_domain" in reason for reason in verdict["reasons"])


def test_current_bare_factor_with_full_provenance_is_not_legacy():
    assert classify_bare_factor(_current_bare_factor()) == {
        "legacy": False,
        "reasons": [],
    }


def test_current_bare_factor_without_four_quadrants_is_legacy():
    payload = _current_bare_factor()
    payload.pop("quadrants")
    verdict = classify_bare_factor(payload)
    assert verdict["legacy"] is True
    assert any("quadrants" in reason for reason in verdict["reasons"])


def test_v2_single_config_bare_factor_is_legacy_after_quadrant_schema():
    payload = {
        "version": 2,
        "protocol_version": PROTOCOL_VERSION,
        "reward_version": REWARD_VERSION,
        "dataset_id": "d1",
        **execution_provenance(BacktestConfig()),
    }
    verdict = classify_bare_factor(payload)
    assert verdict["legacy"] is True
    assert any("version 2 != current 3" in reason for reason in verdict["reasons"])


def test_bare_factor_without_execution_provenance_is_legacy():
    verdict = classify_bare_factor(
        {
            "version": 1,
            "protocol_version": "21",
            "reward_version": "13",
            "dataset_id": "d1",
        }
    )
    assert verdict["legacy"] is True
    reasons = "; ".join(verdict["reasons"])
    assert "execution_version" in reasons
    assert "portfolio_config" in reasons


def test_v12_protocol_is_legacy_with_reasons():
    verdict = classify_protocol(_v12_protocol())
    assert verdict["legacy"] is True
    reasons = "; ".join(verdict["reasons"])
    assert f"protocol_version 12 != current {PROTOCOL_VERSION}" in reasons
    assert f"reward_version 10 != current {REWARD_VERSION}" in reasons
    assert "no dataset_id (pre-T1-01)" in reasons
    assert "no stitched OOS block (pre-T4-01)" in reasons
    assert "no ledger (pre-T4-01)" in reasons


def test_classify_unknown_kind_never_legacy():
    assert classify_artifact("mystery", _v10_strategy()) == {
        "legacy": False,
        "reasons": [],
    }


def test_stamp_marks_v10_v12_and_is_idempotent(tmp_path):
    (tmp_path / "best_ashare_strategy.json").write_text(
        json.dumps(_v10_strategy()), encoding="utf-8"
    )
    (tmp_path / "protocol_result.json").write_text(
        json.dumps(_v12_protocol()), encoding="utf-8"
    )

    outcomes = stamp_legacy_artifacts(tmp_path)
    by_name = {o["name"]: o for o in outcomes}
    assert by_name["strategy"]["stamped"] is True
    assert by_name["protocol"]["stamped"] is True

    strategy = json.loads(
        (tmp_path / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
    assert strategy["legacy"] is True
    assert any("reward_version 10" in r for r in strategy["legacy_reason"])
    assert "legacy_stamped_at" in strategy
    protocol = json.loads(
        (tmp_path / "protocol_result.json").read_text(encoding="utf-8")
    )
    assert protocol["legacy"] is True

    # Idempotent: a second run touches nothing and reports already legacy.
    outcomes = stamp_legacy_artifacts(tmp_path)
    assert all(o["stamped"] is False for o in outcomes)
    by_name = {o["name"]: o for o in outcomes}
    assert by_name["strategy"]["legacy"] is True
    assert by_name["protocol"]["legacy"] is True
    assert by_name["bare_factor"]["legacy"] is False
    stamped_at = strategy["legacy_stamped_at"]
    again = json.loads(
        (tmp_path / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
    assert again["legacy_stamped_at"] == stamped_at


def test_stamp_leaves_current_artifacts_untouched(tmp_path):
    (tmp_path / "best_ashare_strategy.json").write_text(
        json.dumps(_current_strategy()), encoding="utf-8"
    )
    outcomes = stamp_legacy_artifacts(tmp_path)
    assert outcomes[0]["stamped"] is False
    assert outcomes[0]["legacy"] is False
    payload = json.loads(
        (tmp_path / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
    assert "legacy" not in payload


def test_stamp_tolerates_missing_artifacts(tmp_path):
    outcomes = stamp_legacy_artifacts(tmp_path)
    assert len(outcomes) == 3
    assert all(o["stamped"] is False for o in outcomes)


class _StubDataConfig:
    def __init__(self, data_dir):
        self.data_dir = data_dir


def test_webapi_never_presents_old_strategy_without_legacy_flag(
    tmp_path, monkeypatch
):
    import webapi.service as service

    (tmp_path / "best_ashare_strategy.json").write_text(
        json.dumps(_v10_strategy()), encoding="utf-8"
    )
    monkeypatch.setattr(
        service,
        "_get_configs",
        lambda: (_StubDataConfig(tmp_path), None, None, None),
    )
    payload = service.get_strategy()
    assert payload["legacy"] is True
    assert any("reward_version 10" in r for r in payload["legacy_reason"])


def test_webapi_passes_current_strategy_through(tmp_path, monkeypatch):
    import webapi.service as service

    (tmp_path / "best_ashare_strategy.json").write_text(
        json.dumps(_current_strategy()), encoding="utf-8"
    )
    monkeypatch.setattr(
        service,
        "_get_configs",
        lambda: (_StubDataConfig(tmp_path), None, None, None),
    )
    payload = service.get_strategy()
    assert "legacy" not in payload
    assert payload["searcher"] == "gp"


def test_doctor_reports_classification_for_unstamped_old_artifact():
    """An old artifact without a stamp stays an error for the doctor, but
    its classification is visible in the report (P0-04 enforcement)."""

    inputs = dict(
        code={"commit": "c", "branch": "b", "dirty": False},
        data={"dataset_id": "d1"},
        gates={"mode": "formal", "ok": True, "degraded": False, "checks": []},
        artifacts=[
            {
                "name": "strategy",
                "path": "/s",
                "exists": True,
                "legacy": False,  # not stamped
                "legacy_reasons": [],
                "classification": classify_strategy(_v10_strategy()),
                "fields": {"reward_version": "10"},
            }
        ],
        dependencies={"deap": "1.4.4"},
        model_searcher="gp",
        runtime_estimates_={},
    )
    report = doctor.build_report(**inputs)
    assert report["healthy"] is False
    assert report["artifacts"][0]["classification"]["legacy"] is True
    assert any(
        "not marked legacy" in f["message"]
        for f in report["findings"]
        if f["severity"] == "error"
    )
