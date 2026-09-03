"""T4-01 contract tests: Champion/Challenger promotion gates.

A challenger is promoted only when all seven gates pass at once: data &
formula P0, statistical significance, excess-return & risk constraints,
cost/capacity stress, at least one complete future paper-trading
observation window, data-credibility tier (P2-03/P2-04), and feature
registry status (P12: every feature registry-promotable).  These tests
pin the contract before the implementation exists — every gate failure
is a *reason*, and the verdict records the thresholds used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ashare_data.config import (
    BacktestConfig,
    FoldConfig,
    ProtocolConfig,
    TierConfig,
)
from ashare_model import evaluation
from ashare_model.evaluation import PROTOCOL_VERSION, build_result
from ashare_model.feature_registry import FEATURE_REGISTRY_VERSION
from ashare_model.reward import REWARD_VERSION
from ashare_portfolio.constructor import PORTFOLIO_CONSTRUCTOR_VERSION
from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION
from ashare_model.promotion import (
    PaperWindowRegistry,
    evaluate_challenger,
    formula_hash,
    stress_champion,
)
from ashare_model.regime import RegimeRegistry, declare_dev_regime

from test_evaluation import _loader  # noqa: F401
from test_stitched_oos import _row


def _strong_rows():
    """A clearly strong stitched candidate: 2 folds x 250 sessions."""
    rng = np.random.default_rng(7)
    rows = []
    for train_end, test_end in [
        ("2026-01-01", "2026-06-30"),
        ("2026-06-30", "2026-12-31"),
    ]:
        daily = [0.004 + x for x in rng.normal(0.0, 0.001, 250)]
        rows.append(
            _row(
                "challenger",
                42,
                train_end,
                test_end,
                daily=daily,
                formula_text="RET_1",
                eligible=True,
                rejection_reasons=[],
                direction=1,
                formula=[1],
                average_turnover=0.05,
                capacity_utilization=0.05,
            )
        )
    return rows


def _strong_artifact(regime=None, **kwargs):
    rows = _strong_rows()
    proto = ProtocolConfig(
        folds=[
            FoldConfig("2026-01-01", "2026-06-30"),
            FoldConfig("2026-06-30", "2026-12-31"),
        ],
        seeds=[42],
        baseline_signals=[],
        random_samples=0,
    )
    return build_result(
        proto, "confirmation", TierConfig(50, 256), rows,
        spec_id="a" * 64,
        run_id="b" * 32,
        data_end_date="2026-12-31",
        dataset_id="ds-current",
        regime=regime,
        **kwargs,
    )


def _future_regime(tmp_path):
    # Everything through 2026-01-01 is dev; the challenger's folds and the
    # paper window run after it (future data) — a valid final evaluation.
    return RegimeRegistry(
        tmp_path / "registry.json", declare_dev_regime("2026-01-01")
    )


def _passing_stress():
    return {
        "grid": [
            {
                "cost_multiplier": c,
                "capital_multiplier": k,
                "n_days": 500,
                "sharpe": 2.0,
                "max_drawdown": 0.05,
                "excess_return": 0.20,
                "annual_return": 0.30,
            }
            for c in (0.5, 1.0, 2.0)
            for k in (0.1, 1.0, 10.0)
        ],
        "n_cells": 9,
    }


def _paper_registry(tmp_path, formula="RET_1", sessions=120, start="2026-02-01",
                    end="2026-06-30", token=None):
    registry = PaperWindowRegistry(tmp_path / "paper.json")
    registry.register(
        formula_hash=formula_hash([token] if token is not None else [1], formula),
        start=start,
        end=end,
        sessions=sessions,
        config_sha256="cfg-1",
        equity_path="data/sim_trades/",
    )
    return registry


def _verdict(artifact, **kwargs):
    return evaluate_challenger(artifact, **kwargs)


def test_all_gates_pass_promotes(tmp_path):
    artifact = _strong_artifact()
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    assert verdict["promoted"] is True
    assert all(g["passed"] for g in verdict["gates"].values())
    assert verdict["challenger"]["candidate"] == "challenger"
    assert verdict["challenger"]["formula_hash"] == formula_hash([1], "RET_1")
    assert verdict["thresholds"]["dsr_min"] == 0.95


def test_g1_rejects_old_protocol_version(tmp_path):
    artifact = _strong_artifact()
    artifact["protocol_version"] = "19"
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["data_formula_p0"]
    assert not gate["passed"]
    assert any("protocol_version 19" in r for r in gate["reasons"])


def test_g1_rejects_missing_execution_version(tmp_path):
    artifact = _strong_artifact()
    artifact.pop("execution_version", None)
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["data_formula_p0"]
    assert not gate["passed"]
    assert any("execution_version" in reason for reason in gate["reasons"])


def test_g1_rejects_runtime_portfolio_config_mismatch(tmp_path):
    recorded = BacktestConfig(
        portfolio_method="optimizer",
        top_n=20,
        buy_rank=20,
        sell_rank=30,
    )
    artifact = _strong_artifact(backtest_config=recorded)
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
        backtest_config=BacktestConfig(),
    )
    gate = verdict["gates"]["data_formula_p0"]
    assert not gate["passed"]
    assert any("portfolio_config" in reason for reason in gate["reasons"])


def test_g1_rejects_dataset_mismatch(tmp_path):
    artifact = _strong_artifact()
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-other",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["data_formula_p0"]["passed"]


def test_g1_rejects_missing_regime(tmp_path):
    verdict = _verdict(
        _strong_artifact(),
        regime=RegimeRegistry(tmp_path / "missing.json"),  # nothing declared
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["data_formula_p0"]["passed"]


def test_g1_rejects_dev_data_final_evaluation(tmp_path):
    # The regime declares dev through 2030: the challenger's folds are dev.
    regime = RegimeRegistry(
        tmp_path / "registry.json", declare_dev_regime("2030-12-31")
    )
    verdict = _verdict(
        _strong_artifact(),
        regime=regime,
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["data_formula_p0"]["passed"]
    assert any("dev/validation" in r for r in
               verdict["gates"]["data_formula_p0"]["reasons"])


def test_g1_rejects_ineligible_formula(tmp_path):
    artifact = _strong_artifact()
    for row in artifact["rows"]:
        row["eligible"] = False
        row["rejection_reasons"] = ["insufficient valid IC days"]
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["data_formula_p0"]["passed"]
    assert any("eligibility" in r for r in
               verdict["gates"]["data_formula_p0"]["reasons"])


def test_g2_rejects_insignificant_dsr(tmp_path):
    artifact = _strong_artifact()
    artifact["dsr"]["dsr"] = 0.50
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["significance"]
    assert not gate["passed"]
    assert any("dsr" in r for r in gate["reasons"])


def test_g2_rejects_insignificant_max_t(tmp_path):
    artifact = _strong_artifact()
    artifact["max_t"]["significant_95"] = False
    artifact["max_t"]["p_value"] = 0.42
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["significance"]
    assert not gate["passed"]
    assert any("max-t" in r for r in gate["reasons"])


def test_g3_rejects_drawdown_breach(tmp_path):
    artifact = _strong_artifact()
    artifact["top_trial"]["max_drawdown"] = 0.60
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["excess_and_risk"]
    assert not gate["passed"]
    assert any("drawdown" in r for r in gate["reasons"])


def test_g3_rejects_negative_excess(tmp_path):
    artifact = _strong_artifact()
    artifact["top_trial"]["excess_return"] = -0.05
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["excess_and_risk"]["passed"]


def test_g4_requires_stress_result(tmp_path):
    verdict = _verdict(
        _strong_artifact(),
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=None,
    )
    gate = verdict["gates"]["cost_capacity_stress"]
    assert not gate["passed"]
    assert any("stress" in r for r in gate["reasons"])


def test_g4_rejects_stressed_cell_breach(tmp_path):
    stress = _passing_stress()
    stress["grid"][6]["max_drawdown"] = 0.75  # the 2x-cost cell blows up
    verdict = _verdict(
        _strong_artifact(),
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=stress,
    )
    gate = verdict["gates"]["cost_capacity_stress"]
    assert not gate["passed"]
    assert any("cost x2.0" in r for r in gate["reasons"])


def test_g5_requires_completed_paper_window(tmp_path):
    verdict = _verdict(
        _strong_artifact(),
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=PaperWindowRegistry(tmp_path / "empty.json"),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["paper_window"]
    assert not gate["passed"]
    assert any("paper window" in r for r in gate["reasons"])


def test_g5_rejects_short_window(tmp_path):
    verdict = _verdict(
        _strong_artifact(),
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path, sessions=10),
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["paper_window"]["passed"]


def test_g5_rejects_unfinished_window(tmp_path):
    registry = PaperWindowRegistry(tmp_path / "paper.json")
    registry.register(
        formula_hash=formula_hash([1], "RET_1"),
        start="2026-02-01",
        end="2999-01-01",  # still running
        sessions=120,
    )
    verdict = _verdict(
        _strong_artifact(),
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=registry,
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["paper_window"]["passed"]


def test_g5_rejects_window_on_dev_data(tmp_path):
    # Window inside the dev period (before the declared cutoff): it is
    # not a *future* observation, so it cannot be the final evidence.
    verdict = _verdict(
        _strong_artifact(),
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(
            tmp_path, start="2025-01-01", end="2025-06-30"
        ),
        stress=_passing_stress(),
    )
    assert not verdict["gates"]["paper_window"]["passed"]


def test_paper_registry_roundtrip(tmp_path):
    registry = _paper_registry(tmp_path)
    reloaded = PaperWindowRegistry(tmp_path / "paper.json")
    windows = reloaded.windows()
    assert len(windows) == 1
    assert windows[0].sessions == 120
    assert reloaded.completed(formula_hash([1], "RET_1"), 60) is not None
    assert reloaded.completed(formula_hash([2], "OTHER"), 60) is None


def test_stress_champion_grid_on_synthetic_data(populated_db):
    loader = _loader(populated_db)
    fold_cfgs = [FoldConfig("2024-01-10", "2024-01-25")]
    stress = stress_champion(
        loader,
        fold_cfgs,
        tokens=[1],
        direction=1,
        bt_cfg=BacktestConfig(),
        cost_multipliers=(0.5, 1.0, 2.0),
        capital_multipliers=(0.1, 1.0, 10.0),
    )
    assert stress["n_cells"] == 9
    for cell in stress["grid"]:
        assert cell["n_days"] >= 1
        assert cell["sharpe"] is not None
    # The 1x1 cell reproduces the plain evaluation path.
    fold = evaluation.resolve_folds(fold_cfgs, loader.dates)[0]
    plain = evaluation.evaluate_formula([1], loader, fold, BacktestConfig())
    one_one = next(
        c for c in stress["grid"]
        if c["cost_multiplier"] == 1.0 and c["capital_multiplier"] == 1.0
    )
    assert one_one["total_return"] == pytest.approx(plain["total_return"], abs=1e-9)


def test_cli_register_paper_window(tmp_path):
    import subprocess
    import sys

    paper_path = tmp_path / "paper.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ashare_model.promotion",
            "--register-paper",
            "2026-02-01",
            "2026-06-30",
            "120",
            "data/sim_trades/",
            "--formula-hash",
            "abc123",
            "--paper",
            str(paper_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    registry = PaperWindowRegistry(paper_path)
    assert registry.completed("abc123", 60) is not None


@pytest.mark.parametrize(
    "payload",
    [
        {"artifact_schema_version": 1, "rows": []},
        {"artifact_schema_version": 3, "rows": []},
    ],
    ids=["legacy-v1", "unknown-v3"],
)
def test_cli_promotion_rejects_noncurrent_protocol_artifact(tmp_path, payload):
    import json

    from ashare_model.artifact_schemas import ArtifactSchemaError
    from ashare_model.promotion import main

    artifact_path = tmp_path / "protocol.json"
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "verdict.json"
    with pytest.raises(ArtifactSchemaError):
        main(
            [
                "--artifact",
                str(artifact_path),
                "--output",
                str(output_path),
                "--regime",
                str(tmp_path / "regime.json"),
                "--paper",
                str(tmp_path / "paper.json"),
            ]
        )
    assert not output_path.exists()


def test_cli_promotion_refuses_legacy_artifact_without_dataset(
    tmp_path, fresh_populated_db
):
    """End-to-end promotion CLI: an artifact without a dataset_id is
    refused at the P0 gate (legacy measurements cannot be promoted).

    t23: the promotion CLI runs in a SUBPROCESS, so the G8 freshness gate
    (p16, t13) cannot be monkeypatched — the fixture axis is
    today-relative instead (fresh_populated_db) and the fold dates derive
    from the same axis positions the historical 2024-01-01 config used.
    """
    import json as _json
    import subprocess
    import sys

    import yaml

    populated_db, dates = fresh_populated_db
    train_end, test_end = dates[6], dates[17]

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(populated_db.data_dir),
                "duckdb_path": str(populated_db.duckdb_path),
                "parquet_dir": str(populated_db.parquet_dir),
                "index_codes": ["000300.SH"],
                "min_listed_sessions": 1,
                "model": {"max_formula_len": 6},
                "protocol": {
                    "folds": [
                        {"train_end": train_end, "test_end": test_end}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    loader = _loader(populated_db)
    rows = _strong_rows()[:1]
    rows[0]["fold_train_end"] = "2024-01-10"
    rows[0]["fold_test_end"] = "2024-01-25"
    artifact = build_result(
        ProtocolConfig(
            folds=[FoldConfig("2024-01-10", "2024-01-25")],
            seeds=[42],
            baseline_signals=[],
            random_samples=0,
        ),
        "confirmation",
        TierConfig(50, 256),
        rows,
        spec_id="a" * 64,
        run_id="b" * 32,
        data_end_date="2024-01-25",
        # dataset_id deliberately absent: legacy measurement.
    )
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        _json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
    )
    verdict_path = tmp_path / "verdict.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ashare_model.promotion",
            "--artifact",
            str(artifact_path),
            "--config",
            str(cfg_path),
            "--output",
            str(verdict_path),
            "--regime",
            str(tmp_path / "holdout_registry.json"),
            "--paper",
            str(tmp_path / "paper.json"),
            "--min-eligible",
            "3",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 1  # refused, not promoted
    verdict = _json.loads(verdict_path.read_text(encoding="utf-8"))
    assert verdict["promoted"] is False
    assert any(
        "dataset_id" in r
        for r in verdict["gates"]["data_formula_p0"]["reasons"]
    )


# --- P2-03 / P2-04: data-credibility-tier gate --------------------------------


def _tok(name: str) -> int:
    from ashare_model.vocab import FORMULA_VOCAB

    return FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name)


def _tier_artifact(feature_name: str):
    """A strong artifact whose champion formula is the bare given factor."""
    artifact = _strong_artifact()
    for row in artifact["rows"]:
        row["formula_text"] = feature_name
        row["formula"] = [_tok(feature_name)]
    artifact["top_trial"]["formula_text"] = feature_name
    return artifact


def test_data_tier_gate_rejects_tier_b_by_default(tmp_path):
    """P2-03: Champion candidates default to Tier A only; a Tier B formula
    (e.g. ROE) is refused and the reason names the violating feature."""
    artifact = _tier_artifact("ROE")
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path, formula="ROE", token=_tok("ROE")),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["data_tier"]
    assert not gate["passed"]
    assert any("ROE" in r and "B" in r for r in gate["reasons"])
    assert verdict["data_tier_policy"]["allowed_tiers"] == ["A"]
    assert verdict["data_tier_policy"]["formula_data_tier"]["max_tier"] == "B"


def test_data_tier_gate_passes_tier_a_formula(tmp_path):
    verdict = _verdict(
        _strong_artifact(),  # champion formula = RET_1 (Tier A)
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    assert verdict["gates"]["data_tier"]["passed"] is True
    assert verdict["promoted"] is True


def test_data_tier_gate_tier_b_requires_separate_comparison(tmp_path):
    """P2-03: Tier B is promotable only through an explicit separate
    comparison (allowed_data_tiers=(A, B)); the policy is recorded."""
    artifact = _tier_artifact("ROE")
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path, formula="ROE", token=_tok("ROE")),
        stress=_passing_stress(),
        allowed_data_tiers=("A", "B"),
    )
    assert verdict["gates"]["data_tier"]["passed"] is True
    assert verdict["data_tier_policy"]["allowed_tiers"] == ["A", "B"]
    assert verdict["promoted"] is True


def test_data_tier_gate_never_accepts_tier_c(tmp_path):
    """P2-04: Tier C (industry snapshot / placeholder) never enters
    promotion results, even under the explicit Tier B comparison."""
    artifact = _tier_artifact("INDUSTRY_MOMENTUM")
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(
            tmp_path, formula="INDUSTRY_MOMENTUM", token=_tok("INDUSTRY_MOMENTUM")
        ),
        stress=_passing_stress(),
        allowed_data_tiers=("A", "B"),
    )
    gate = verdict["gates"]["data_tier"]
    assert not gate["passed"]
    assert any("tier C" in r for r in gate["reasons"])
    assert verdict["promoted"] is False


def test_data_tier_gate_rejects_untraceable_formula(tmp_path):
    artifact = _strong_artifact()
    for row in artifact["rows"]:
        row["formula_text"] = "equal_weight"
        row["formula"] = None
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path, formula="equal_weight"),
        stress=_passing_stress(),
    )
    gate = verdict["gates"]["data_tier"]
    assert not gate["passed"]
    assert any("no traceable formula" in r for r in gate["reasons"])


def test_data_tier_policy_rejects_c_and_unknown_tiers():
    artifact = _strong_artifact()
    with pytest.raises(ValueError, match="research-display only"):
        _verdict(artifact, allowed_data_tiers=("C",))
    with pytest.raises(ValueError, match="unknown"):
        _verdict(artifact, allowed_data_tiers=("X",))


# ---------------------------------------------------------------------------
# F2 characterization pins (parity guards for the version-judgement
# refactor, t3-M1): G1 compares exactly four recorded versions against the
# current code generation — protocol / reward / execution /
# portfolio_constructor — and treats a *missing* field as a mismatch
# (``None`` is rendered into the message).  G1 deliberately does NOT
# inspect ``model_version`` / ``searcher``; that wider field set belongs to
# ``ashare_model.artifact_versions.classify_strategy``.  The asymmetry is
# pinned here as current behavior, not endorsed as ideal — aligning the
# coverage would tighten the promotion gate and is a research-semantics
# change, out of F2 scope.
# ---------------------------------------------------------------------------


def _g1_gate(artifact, tmp_path):
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    return verdict["gates"]["data_formula_p0"]


def test_g1_version_mismatch_messages_exact(tmp_path):
    artifact = _strong_artifact()
    artifact["protocol_version"] = "19"
    artifact["reward_version"] = "10"
    artifact["execution_version"] = 1
    artifact["portfolio_constructor_version"] = 0
    gate = _g1_gate(artifact, tmp_path)
    assert not gate["passed"]
    reasons = gate["reasons"]
    assert (
        f"artifact protocol_version 19 != {PROTOCOL_VERSION}; "
        "only current stitched artifacts qualify"
    ) in reasons
    assert f"artifact reward_version 10 != {REWARD_VERSION}" in reasons
    assert (
        f"artifact execution_version 1 != {EXECUTION_SPEC_VERSION}; "
        "legacy execution semantics cannot be promoted"
    ) in reasons
    assert (
        "artifact portfolio_constructor_version 0 != "
        f"{PORTFOLIO_CONSTRUCTOR_VERSION}"
    ) in reasons


def test_g1_treats_missing_versions_as_mismatch(tmp_path):
    artifact = _strong_artifact()
    for field in (
        "protocol_version",
        "reward_version",
        "portfolio_constructor_version",
    ):
        artifact.pop(field, None)
    gate = _g1_gate(artifact, tmp_path)
    assert not gate["passed"]
    reasons = gate["reasons"]
    assert (
        f"artifact protocol_version None != {PROTOCOL_VERSION}; "
        "only current stitched artifacts qualify"
    ) in reasons
    assert f"artifact reward_version None != {REWARD_VERSION}" in reasons
    assert (
        "artifact portfolio_constructor_version None != "
        f"{PORTFOLIO_CONSTRUCTOR_VERSION}"
    ) in reasons


def test_g1_version_comparison_is_string_coerced(tmp_path):
    # The current generation mixes int and str version constants
    # (EXECUTION_SPEC_VERSION=2 vs PROTOCOL_VERSION="25"); a recorded value
    # matches iff str(recorded) == str(current).
    artifact = _strong_artifact()
    artifact["protocol_version"] = int(PROTOCOL_VERSION)
    artifact["reward_version"] = int(REWARD_VERSION)
    artifact["execution_version"] = str(EXECUTION_SPEC_VERSION)
    artifact["portfolio_constructor_version"] = str(PORTFOLIO_CONSTRUCTOR_VERSION)
    gate = _g1_gate(artifact, tmp_path)
    assert gate["passed"], gate["reasons"]


def test_g1_does_not_check_model_version_or_searcher(tmp_path):
    # Field-coverage asymmetry pin: a stale model_version / unknown
    # searcher still passes G1.
    artifact = _strong_artifact()
    artifact["model_version"] = 0
    artifact["searcher"] = "deliberately-unknown"
    gate = _g1_gate(artifact, tmp_path)
    assert gate["passed"], gate["reasons"]


# --- P12: promotion gate G7 (feature_registry_status) -------------------------
#
# Preregistered by docs/p12_promotion_enforcement_contract.md (merged to
# main @ 97a86fe): the challenger formula's features must all be
# registry-promotable (``promotion_allowed`` and not ``deprecated``);
# fail-closed on untraceable formulas.  The rejection reason strings are
# the contract §5.2 preregistered literals.


def _op_tok(name: str) -> int:
    from ashare_model.vocab import FORMULA_VOCAB

    return FORMULA_VOCAB.operator_offset + FORMULA_VOCAB.operator_names.index(name)


def _registry_gate_evidence(verdict) -> str:
    return "; ".join(
        f"{name}={'PASS' if gate['passed'] else 'FAIL'}"
        for name, gate in verdict["gates"].items()
    )


def _require_g7(verdict):
    gate = verdict["gates"].get("feature_registry_status")
    assert gate is not None, (
        "P12 §1.1 structural gap: no feature_registry_status gate in the "
        f"promotion verdict ({_registry_gate_evidence(verdict)}) — "
        "feature registry status has zero consumers on the promotion path"
    )
    return gate


def _combo_artifact(feature_name: str):
    """A strong artifact whose champion formula is ``ADD(RET_1, <feature>)``.

    Both features are Tier A (PIT_DAILY), so the data-tier gate passes
    and any rejection is attributable to the feature-registry gate.
    Returns ``(tokens, formula_text, artifact)``.
    """
    artifact = _strong_artifact()
    tokens = [_tok("RET_1"), _tok(feature_name), _op_tok("ADD")]
    text = f"RET_1 {feature_name} ADD"
    for row in artifact["rows"]:
        row["formula_text"] = text
        row["formula"] = list(tokens)
    artifact["top_trial"]["formula_text"] = text
    return tokens, text, artifact


def _paper_registry_for(tmp_path, tokens, text):
    registry = PaperWindowRegistry(tmp_path / "paper.json")
    registry.register(
        formula_hash=formula_hash(tokens, text),
        start="2026-02-01",
        end="2026-06-30",
        sessions=120,
        config_sha256="cfg-1",
        equity_path="data/sim_trades/",
    )
    return registry


def test_g7_rejects_family3_not_promotable_formula(tmp_path):
    """P12 §7 RED-1: a Tier-A combination formula embedding LIMIT_UP_CNT_5
    (family ③, P9 adjudication: promotion_allowed=False) passes every
    pre-existing gate but must be refused by the feature-registry gate."""
    tokens, text, artifact = _combo_artifact("LIMIT_UP_CNT_5")
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry_for(tmp_path, tokens, text),
        stress=_passing_stress(),
    )
    gate = _require_g7(verdict)
    assert not gate["passed"]
    assert (
        "feature LIMIT_UP_CNT_5 is not promotion_allowed "
        "(feature registry status)"
    ) in gate["reasons"]
    # Family ③ is not deprecated: no attached deprecation reason.
    assert not any("deprecated" in r for r in gate["reasons"])
    assert verdict["promoted"] is False
    # The rejection must be attributable to G7 alone (Tier A formula).
    assert verdict["gates"]["data_tier"]["passed"] is True


def test_g7_rejects_deprecated_formula_loaded_from_artifact(tmp_path):
    """P12 §7 RED-2: sampling no longer emits LIMIT_STREAK (deprecated,
    P9 §7.3 conditional deprecation TRIGGERED), but the artifact loading
    path can still carry it — exactly why the promotion gate must
    enforce the registry status itself."""
    tokens, text, artifact = _combo_artifact("LIMIT_STREAK")
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry_for(tmp_path, tokens, text),
        stress=_passing_stress(),
    )
    gate = _require_g7(verdict)
    assert not gate["passed"]
    assert (
        "feature LIMIT_STREAK is not promotion_allowed "
        "(feature registry status)"
    ) in gate["reasons"]
    assert (
        "feature LIMIT_STREAK is deprecated (feature registry status)"
    ) in gate["reasons"]
    assert verdict["promoted"] is False
    assert verdict["gates"]["data_tier"]["passed"] is True


def test_g7_rejects_bare_factor_baseline_row(tmp_path):
    """P12 §7 RED-3: the baseline-row routing (``formula=None``,
    ``formula_text=NAME``) must hit the registry gate symmetrically to
    the G6 bare-factor path."""
    artifact = _strong_artifact()
    for row in artifact["rows"]:
        row["formula_text"] = "LIMIT_UP_CNT_5"
        row["formula"] = None
    artifact["top_trial"]["formula_text"] = "LIMIT_UP_CNT_5"
    registry = PaperWindowRegistry(tmp_path / "paper.json")
    registry.register(
        formula_hash=formula_hash(None, "LIMIT_UP_CNT_5"),
        start="2026-02-01",
        end="2026-06-30",
        sessions=120,
        config_sha256="cfg-1",
        equity_path="data/sim_trades/",
    )
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=registry,
        stress=_passing_stress(),
    )
    gate = _require_g7(verdict)
    assert not gate["passed"]
    assert (
        "feature LIMIT_UP_CNT_5 is not promotion_allowed "
        "(feature registry status)"
    ) in gate["reasons"]
    assert verdict["promoted"] is False
    # G6's feature_name routing still passes (LIMIT_UP_CNT_5 is Tier A).
    assert verdict["gates"]["data_tier"]["passed"] is True


def test_g7_fail_closed_on_undecodable_tokens(tmp_path):
    """P12 §7 RED-4a: undecodable tokens fail closed without weakening
    G6's existing untraceable-formula behavior."""
    artifact = _strong_artifact()
    for row in artifact["rows"]:
        row["formula"] = [10**9]  # out-of-vocabulary token
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = _require_g7(verdict)
    assert not gate["passed"]
    assert any(
        "no traceable formula for registry status" in r
        for r in gate["reasons"]
    )
    policy = verdict["registry_status_policy"]
    assert policy["report"]["traceable"] is False
    assert policy["report"]["per_feature"] == {}
    # G6's existing untraceable behavior is unchanged (no weakening).
    assert not verdict["gates"]["data_tier"]["passed"]
    assert any(
        "no traceable formula" in r
        for r in verdict["gates"]["data_tier"]["reasons"]
    )


def test_g7_fail_closed_on_unknown_feature_name(tmp_path):
    """P12 §7 RED-4b: a feature name outside the registry vocabulary is
    rejected, never silently skipped."""
    artifact = _strong_artifact()
    for row in artifact["rows"]:
        row["formula_text"] = "NOT_IN_REGISTRY"
        row["formula"] = None
    artifact["top_trial"]["formula_text"] = "NOT_IN_REGISTRY"
    registry = PaperWindowRegistry(tmp_path / "paper.json")
    registry.register(
        formula_hash=formula_hash(None, "NOT_IN_REGISTRY"),
        start="2026-02-01",
        end="2026-06-30",
        sessions=120,
        config_sha256="cfg-1",
        equity_path="data/sim_trades/",
    )
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=registry,
        stress=_passing_stress(),
    )
    gate = _require_g7(verdict)
    assert not gate["passed"]
    assert any(
        "no traceable formula for registry status" in r
        for r in gate["reasons"]
    )
    assert not verdict["gates"]["data_tier"]["passed"]  # G6 refuses equally


def test_g7_passes_registry_promotable_formula(tmp_path):
    """P12 §7 RED-5: a promotable-feature formula passes G7 with the
    per-feature report recorded; the pre-existing gates stay untouched."""
    artifact = _strong_artifact()  # champion formula = bare RET_1
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry(tmp_path),
        stress=_passing_stress(),
    )
    gate = _require_g7(verdict)
    assert gate["passed"] is True
    policy = verdict["registry_status_policy"]
    assert policy["feature_registry_version"] == FEATURE_REGISTRY_VERSION
    assert policy["report"]["traceable"] is True
    assert policy["report"]["per_feature"] == {
        "RET_1": {"promotion_allowed": True, "deprecated": False}
    }
    assert verdict["promotion_rule_version"] == "2"
    assert verdict["promoted"] is True


def test_g7_rejects_p10_seed7_live_formula(tmp_path):
    """P12 §1.3/§7 RED-6: the only P10 Stage-B/C formula that passed all
    four OOS hard gates (random seed-7) embeds LIMIT_UP_CNT_5; G7 must
    refuse it.  The formula also carries MARGIN_CROWD_60 (Tier B), so the
    test runs the P2-03 explicit Tier A+B comparison (--allow-tier-b
    path) where G6 passes: the rejection is then attributable to the
    registry gate alone.  Tokens are read from the git-tracked P10
    summary (provenance pin; the file is part of the repo tree)."""
    import json as _json

    summary_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "p10_searcher_comparison_20260901"
        / "summary.json"
    )
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    row = next(
        r
        for r in summary["rows"]
        if r["seed"] == 7 and r["searcher"] == "random"
    )
    tokens = list(row["selected"]["tokens"])
    text = row["selected"]["formula_text"]
    assert "LIMIT_UP_CNT_5" in text
    artifact = _strong_artifact()
    for r in artifact["rows"]:
        r["formula_text"] = text
        r["formula"] = list(tokens)
        # t46 Plan A: the P10 campaign ran under grammar 5; the row-declared
        # generation routes the bare tokens through the frozen grammar-5
        # decode layout instead of the shifted grammar-6 feature block.
        r["grammar_version"] = 5
    artifact["top_trial"]["formula_text"] = text
    verdict = _verdict(
        artifact,
        regime=_future_regime(tmp_path),
        dataset_id="ds-current",
        paper_registry=_paper_registry_for(tmp_path, tokens, text),
        stress=_passing_stress(),
        allowed_data_tiers=("A", "B"),
    )
    gate = _require_g7(verdict)
    assert not gate["passed"]
    assert (
        "feature LIMIT_UP_CNT_5 is not promotion_allowed "
        "(feature registry status)"
    ) in gate["reasons"]
    assert verdict["promoted"] is False
    # Tier A+B policy passes G6 — the gap G7 closes, exactly as §1.3.
    assert verdict["gates"]["data_tier"]["passed"] is True


def test_promotion_rule_version_pinned():
    """P12 §7: the promotion gate set is versioned from v2 (G1-G6 were
    the unversioned v1 era)."""
    from ashare_model.promotion import PROMOTION_RULE_VERSION

    assert PROMOTION_RULE_VERSION == "2"
