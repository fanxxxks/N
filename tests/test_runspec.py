"""Immutable RunSpec, spec_id and the resolving factory (P8-03).

Lifecycle contract §2/§10: the frozen RunSpec is the pre-registration
artifact; ``spec_id`` is its strict content identity
(``content_hash("runspec", payload)``), ``run_id`` is a per-execution
UUID. Non-semantic fields (created_at, hostname, machine paths) must not
exist in the payload; unresolved versions, a dirty tracked tree, an empty
dataset id and unknown searcher/domain values must be rejected
fail-closed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from ashare_data.config import BacktestConfig, DataConfig
from ashare_model import runspec
from ashare_model.admission import PAIR_SEEDS
from ashare_model.identity import canonical_json_strict
from ashare_model.runspec import (
    RUNSPEC_SCHEMA_VERSION,
    FoldSpec,
    RunSpec,
    UniversePolicySpec,
    WindowSpec,
    build_runspec,
    new_run_id,
)
from ashare_portfolio.execution_spec import portfolio_config_provenance

CANONICAL_PORTFOLIO = portfolio_config_provenance(BacktestConfig())

BASE_KWARGS = dict(
    dataset_id="ds-test-0001",
    data_cutoff="20241231",
    research_domain="unified",
    frequency="daily",
    target="forward_return",
    horizon=1,
    requested_budget=2000,
    n_folds=5,
    dev_window=WindowSpec(start="20150101", end="20231231"),
    holdout_window=WindowSpec(start="20240101", end="20241231"),
    preregistered_contract_hash="a" * 64,
    resolved_thresholds={"factor_min_coverage": 0.9, "min_eligible": 100.0},
    portfolio_config=CANONICAL_PORTFOLIO,
)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", cwd=root)
    _git("config", "user.email", "test@example.com", cwd=root)
    _git("config", "user.name", "test", cwd=root)
    (root / "requirements.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
    _git("add", ".", cwd=root)
    _git("commit", "-m", "init", cwd=root)
    return root


def _build(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    return build_runspec(**kwargs)


# -- schema version and identity --------------------------------------------


def test_schema_version_is_pinned():
    assert RUNSPEC_SCHEMA_VERSION == 1
    assert _build().runspec_schema_version == 1


def test_spec_id_is_deterministic_and_independent_of_kwarg_order():
    spec_a = _build()
    reordered = dict(reversed(list(BASE_KWARGS.items())))
    spec_b = build_runspec(**reordered)
    assert spec_a.spec_id == spec_b.spec_id
    assert spec_a.to_payload() == spec_b.to_payload()


def test_spec_id_matches_independent_canonical_oracle():
    spec = _build()
    payload = spec.to_payload()
    expected = hashlib.sha256(
        json.dumps(
            {"kind": "runspec", "value": payload},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert spec.spec_id == expected
    assert canonical_json_strict(payload) == json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"dataset_id": "ds-test-0002"},
        {"data_cutoff": "20241130"},
        {
            "frequency": "every_5_days",
            "horizon": 5,
            "portfolio_config": portfolio_config_provenance(
                dataclasses.replace(
                    BacktestConfig(), rebalance_frequency="every_5_days", target_horizon=5
                )
            ),
        },
        {"seeds": (1, 2, 3, 4, 5)},
        {"max_formula_len": 16},
        {"requested_budget": 3000},
        {"resolved_thresholds": {"factor_min_coverage": 0.95, "min_eligible": 100.0}},
        {
            "universe_policy": UniversePolicySpec(
                index_codes=("000300.SH",),
                min_listed_sessions=DataConfig().min_listed_sessions,
                membership_end_inclusive=False,
            )
        },
        {"searcher": "tpe"},
        {"n_folds": 3},
        {"dev_window": WindowSpec(start="20150101", end="20230630")},
        {"holdout_window": None},
        {
            "portfolio_config": portfolio_config_provenance(
                dataclasses.replace(BacktestConfig(), initial_capital=200000.0)
            )
        },
        {"preregistered_contract_hash": "b" * 64},
        {"target": "forward_return_5d"},
        {"research_domain": "short_price_volume"},
    ],
)
def test_any_semantic_change_changes_spec_id(mutation):
    assert _build(**mutation).spec_id != _build().spec_id


def test_machine_paths_and_non_semantic_fields_are_absent():
    payload_text = json.dumps(_build().to_payload())
    assert "\\" not in payload_text
    assert "created_at" not in payload_text
    assert "hostname" not in payload_text
    assert "run_id" not in payload_text


def test_defaults_are_resolved_explicitly():
    from ashare_model.evaluation import PROTOCOL_VERSION

    spec = _build()
    assert spec.searcher == "gp"
    assert spec.max_formula_len == 12
    assert spec.seeds == PAIR_SEEDS
    assert spec.budget_accounting == "unique_semantic_evaluations"
    assert spec.universe_policy.index_codes == tuple(DataConfig().index_codes)
    assert spec.protocol_version == PROTOCOL_VERSION


# -- fail-closed factory -----------------------------------------------------


def test_empty_dataset_id_rejected():
    with pytest.raises(ValueError):
        _build(dataset_id="")
    with pytest.raises(ValueError):
        _build(dataset_id="   ")


def test_unresolved_version_rejected(monkeypatch):
    broken = runspec._capture_versions()
    broken["protocol_version"] = None
    monkeypatch.setattr(runspec, "_capture_versions", lambda: broken)
    with pytest.raises(ValueError, match="unresolved version"):
        _build()


def test_unknown_searcher_rejected():
    with pytest.raises(ValueError, match="searcher"):
        _build(searcher="gpt-5")


def test_unknown_research_domain_rejected():
    with pytest.raises(ValueError):
        _build(research_domain="no_such_domain")


def test_frequency_horizon_must_match_portfolio_block():
    with pytest.raises(ValueError, match="portfolio_config"):
        _build(
            frequency="every_5_days",
            horizon=5,
        )


def test_empty_resolved_thresholds_rejected():
    with pytest.raises(ValueError, match="resolved_thresholds"):
        _build(resolved_thresholds={})


def test_malformed_contract_hash_rejected():
    with pytest.raises(ValueError, match="preregistered_contract_hash"):
        _build(preregistered_contract_hash="xyz")


def test_incomplete_portfolio_provenance_rejected():
    broken = dict(CANONICAL_PORTFOLIO)
    del broken["initial_capital"]
    with pytest.raises(ValueError):
        _build(portfolio_config=broken)


# -- git and dependency-lock capture -----------------------------------------


def test_clean_tracked_tree_builds_and_captures_provenance(git_repo):
    spec = build_runspec(repo_root=git_repo, **BASE_KWARGS)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(git_repo), check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    assert spec.git_commit == head
    expected_lock = hashlib.sha256(
        b"requirements.txt:numpy==1.26.4\n"
    ).hexdigest()
    assert spec.dependency_lock_hash == expected_lock


def test_dirty_tracked_tree_rejected(git_repo):
    (git_repo / "requirements.txt").write_text(
        "numpy==1.26.4\npydantic==9.9.9\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="dirty"):
        build_runspec(repo_root=git_repo, **BASE_KWARGS)


def test_untracked_only_tree_is_allowed(git_repo):
    (git_repo / "notes.md").write_text("scratch", encoding="utf-8")
    spec = build_runspec(repo_root=git_repo, **BASE_KWARGS)
    assert spec.dependency_lock_hash


def test_missing_dependency_lock_rejected(git_repo):
    (git_repo / "requirements.txt").unlink()
    _git("add", "-A", cwd=git_repo)
    _git("commit", "-m", "drop lock", cwd=git_repo)
    with pytest.raises(ValueError, match="dependency lock"):
        build_runspec(repo_root=git_repo, **BASE_KWARGS)


# -- run identity and immutability -------------------------------------------


def test_run_ids_differ_for_the_same_spec():
    assert new_run_id() != new_run_id()
    assert re.fullmatch(r"[0-9a-f]{32}", new_run_id())


def test_run_spec_is_frozen_and_extra_forbid():
    spec = _build()
    with pytest.raises(ValidationError):
        spec.dataset_id = "mutated"
    payload = spec.to_payload()
    with pytest.raises(ValidationError):
        RunSpec(**{**payload, "unknown_field": 1})
