"""P0-04 contract tests: legacy artifact classification, stamping, API.

Old artifacts (reward-v10 strategy, protocol-v12 result) must be clearly
marked legacy and never presented as the current champion: the pure
classification rules, the idempotent stamping migration, the web API's
computed flag and the research doctor's classification surface.
"""

from __future__ import annotations

import json

from ashare_data.config import BacktestConfig
from ashare_model.alphagpt import MODEL_VERSION
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
        **execution_provenance(BacktestConfig()),
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


def test_p3_semantic_versions_are_bumped_together():
    assert REWARD_VERSION == "14"
    assert PROTOCOL_VERSION == "22"
    assert EXECUTION_SPEC_VERSION == 2
    assert BARE_FACTOR_BACKTEST_VERSION == 2


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


def test_current_bare_factor_with_full_provenance_is_not_legacy():
    payload = {
        "version": BARE_FACTOR_BACKTEST_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "reward_version": REWARD_VERSION,
        "dataset_id": "d1",
        **execution_provenance(BacktestConfig()),
    }
    assert classify_bare_factor(payload) == {"legacy": False, "reasons": []}


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
