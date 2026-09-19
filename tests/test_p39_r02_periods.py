"""P39 R02 round-2 legal-period comparison tests.

Executable projection of ``docs/p39_r02_periods_draft.md`` (APPROVED via
the 2026-09-19 「全部授权」).  Round 2 varies ONLY the rebalance period
over the medium domain's complete legal execution set (every_5_days/5 and
every_10_days/10) with the portfolio config preregistered at the p37
reference cell — directions learned per (period, fold), the period-level
four-way adjudication mirroring p37, and no cross-period IC/reward
ranking.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import yaml

from ashare_data.config import BacktestConfig

from test_p34_r01_baseline import (
    _contract_sha as _p34_contract_sha,  # noqa: F401  (fixture lineage)
    _mini_dates,
    gate_run_asof,  # noqa: F401  (pytest fixture, re-exported)
    mini_env,  # noqa: F401  (pytest fixture, re-exported)
)

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _contract_sha() -> str:
    import hashlib

    digest = hashlib.sha256()
    with (_ROOT / "docs" / "p39_r02_periods_draft.md").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- P1: the legal period set is frozen and matches the domain authority -----


def test_p1_legal_period_set_frozen():
    from ashare_domain import RESEARCH_DOMAINS

    from ashare_model.r02_matrix import R02_PERIODS

    assert R02_PERIODS == (("every_5_days", 5), ("every_10_days", 10))
    domain = RESEARCH_DOMAINS["medium_cross_section"]
    for frequency, horizon in R02_PERIODS:
        assert domain.is_legal_execution(frequency, horizon)
    # The domain's frequency tuple IS the round-2 axis (no invented points).
    assert {f for f, _ in R02_PERIODS} == set(domain.frequencies)


# --- P2: the period variants change exactly the two cadence fields ----------


def test_p2_period_backtest_variant_two_fields_only():
    from ashare_model.r02_matrix import period_backtest_variant

    base = BacktestConfig(
        rebalance_frequency="every_10_days",
        target_horizon=10,
        top_n=20,
        buy_rank=20,
        sell_rank=30,
        single_weight_cap=0.05,
        min_trade_amount=5000.0,
    )
    variant = period_backtest_variant(base, "every_5_days", 5)
    for field in dataclasses.fields(BacktestConfig):
        if field.name in ("rebalance_frequency", "target_horizon"):
            continue
        assert getattr(variant, field.name) == getattr(base, field.name)
    assert variant.rebalance_frequency == "every_5_days"
    assert variant.target_horizon == 5
    # The fold-policy agreement the engine checks stays constructible.
    from ashare_portfolio.rebalance import RebalancePolicy

    assert RebalancePolicy.from_config(variant) == RebalancePolicy(
        "every_5_days", 5
    )


# --- P3: the period-level four-way adjudication ------------------------------


def _arm_summary(**over):
    payload = {
        "mean_net_excess_1x": 0.05,
        "active_ir_1x": 0.2,
        "invested_ratio_mean_1x": 0.9,
        "mean_holdings_1x": 12.0,
        "failed_folds": 0,
        "positive_folds_1x": 4,
        "stitched": {
            "0x": {"excess_return": 0.08, "oos_active_ir": 0.3},
        },
    }
    payload.update(over)
    return payload


def _period_panel(fast="dead", slow="dead"):
    dead = _arm_summary(
        mean_net_excess_1x=-0.01,
        active_ir_1x=-0.05,
        invested_ratio_mean_1x=0.2,
        mean_holdings_1x=12.0,
        failed_folds=0,
        positive_folds_1x=1,
        stitched={"0x": {"excess_return": -0.02, "oos_active_ir": -0.1}},
    )
    live = _arm_summary()
    return {
        "every_5_days": {
            "frequency": "every_5_days",
            "horizon": 5,
            "top_n": 20,
            "arms": {"baseline:X": live if fast == "live" else dead},
        },
        "every_10_days": {
            "frequency": "every_10_days",
            "horizon": 10,
            "top_n": 20,
            "arms": {"baseline:X": live if slow == "live" else dead},
        },
    }


def test_p3_stable_period_region():
    from ashare_model.r02_matrix import decide_r02_periods_adjudication

    verdict = decide_r02_periods_adjudication(
        _period_panel(fast="live", slow="live")
    )
    assert verdict["verdict"] == "stable_period_region"
    assert set(verdict["viable_periods"]) == {"every_5_days", "every_10_days"}


def test_p3_isolated_period_winner_unstable():
    from ashare_model.r02_matrix import decide_r02_periods_adjudication

    verdict = decide_r02_periods_adjudication(
        _period_panel(fast="live", slow="dead")
    )
    assert verdict["verdict"] == "isolated_period_winner"
    assert verdict["isolated_period"] == "every_5_days"
    assert "不稳定" in verdict["note"] or "unstable" in verdict["note"].lower()


def test_p3_gross_info_execution_blocked():
    from ashare_model.r02_matrix import decide_r02_periods_adjudication

    panel = _period_panel(fast="dead", slow="dead")
    panel["every_5_days"]["arms"]["baseline:X"] = _arm_summary(
        mean_net_excess_1x=-0.02,
        active_ir_1x=-0.1,
        invested_ratio_mean_1x=0.2,
        stitched={"0x": {"excess_return": 0.15, "oos_active_ir": 0.4}},
    )
    verdict = decide_r02_periods_adjudication(panel)
    assert verdict["verdict"] == "gross_info_execution_blocked_period_panel"
    assert "every_5_days::baseline:X" in verdict["gross_info_arms"]


def test_p3_period_panel_no_reliable_info():
    from ashare_model.r02_matrix import decide_r02_periods_adjudication

    verdict = decide_r02_periods_adjudication(_period_panel())
    assert verdict["verdict"] == "period_panel_no_reliable_info"
    assert verdict["gross_info_arms"] == []


# --- P4: end-to-end through the real lifecycle chain --------------------------


def test_p4_end_to_end_periods_chain(mini_env, gate_run_asof):
    """Both legal periods through the real chain on the shared mini
    fixture (REAL adjudicator, relaxed canary thresholds via the explicit
    seam): every period x arm x fold x cost level present, the benchmark
    once per fold PER PERIOD (no cross-period fold collision), the cash
    control once per period, directions recorded per (period, fold), the
    verdict from the four-way set, and the administrative state mapped
    consistently (verdict 1/2 park, 3/4 retire)."""

    import ashare_model.r02_matrix as r02
    from ashare_data.config import make_data_config
    from ashare_model.lifecycle import replay_run
    from ashare_model.run_store import RunStore

    tmp_path, _data_config, cfg_path = mini_env
    gate_run_asof(_mini_dates()[-1])
    summary = r02.run_r02_periods(
        cfg_path,
        contract_sha256=_contract_sha(),
        out_root=tmp_path / "r02_periods_out",
        require_clean_tree=False,
        min_eligible=3,
        adjudication_rules={
            "min_positive_folds": 1,
            "min_invested_ratio": 0.3,
            "max_failed_folds": 1,
        },
    )
    verdict = summary["period_adjudication"]["verdict"]
    assert verdict in {
        "stable_period_region",
        "isolated_period_winner",
        "gross_info_execution_blocked_period_panel",
        "period_panel_no_reliable_info",
    }
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_config = make_data_config(raw, tmp_path)
    store = RunStore(data_config.data_dir)
    state = replay_run(store.root, summary["spec_id"], summary["run_id"])
    if verdict in ("stable_period_region", "isolated_period_winner"):
        assert state == "DATA_QUALIFIED"
    else:
        assert state == "RETIRED"

    report_path = (
        tmp_path / "r02_periods_out" / "r02_periods_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["r02_periods_report_schema_version"] == 1
    assert payload["provenance"]["contract_sha256"] == _contract_sha()
    assert payload["provenance"]["consumed_semantic_evaluations"] == 0
    assert payload["period_adjudication"]["verdict"] == verdict
    rows = payload["rows"]
    # Every (period, arm, fold, level) cell present — both periods in one
    # report, never merged across periods.
    periods_in_rows = {row["period"] for row in rows}
    assert periods_in_rows == {"every_5_days", "every_10_days"}
    arms = {
        "baseline:MOMENTUM_20", "baseline:RSQ_60", "baseline:REVERSAL_60",
        "combo:all3", "combo:momentum_reversal",
    }
    for period in ("every_5_days", "every_10_days"):
        # The preregistered round-2 scope is the REFERENCE CELL ONLY —
        # the cell axis must not silently widen back to the six-cell grid
        # (the strengthened assertion after the first formal run executed
        # a superset panel through an un-wired cells parameter).
        cells_in_period = {
            row["cell"]
            for row in rows
            if row["period"] == period and "::" in row["candidate"]
        }
        assert cells_in_period == {"n20_m5000"}
        keys = {
            (row["candidate"].split("::", 1)[1], row["cost_level"])
            for row in rows
            if row["period"] == period and "::" in row["candidate"]
        }
        assert len(keys) == len(arms) * 2
        bench = [
            row for row in rows
            if row["period"] == period
            and row["candidate"].endswith("benchmark:equal_weight")
        ]
        assert len(bench) == 1
        assert bench[0]["cost_level"] == "ref"
        cash = [
            row for row in rows
            if row["period"] == period
            and row["candidate"].endswith("control:cash")
        ]
        assert len(cash) == 1
        assert cash[0]["cell"] == "n20_m5000"
    # Directions recorded per (period, fold).
    directions = payload["provenance"]["directions_by_period"]
    assert set(directions) == {"every_5_days", "every_10_days"}


def test_p4_contract_sha_mismatch_refused(mini_env, gate_run_asof):
    import pytest as _pytest

    import ashare_model.r02_matrix as r02

    tmp_path, _data_config, cfg_path = mini_env
    gate_run_asof(_mini_dates()[-1])
    with _pytest.raises(r02.R02Refusal, match="contract sha256 mismatch"):
        r02.run_r02_periods(
            cfg_path,
            contract_sha256="0" * 64,
            out_root=tmp_path / "r02_periods_out_sha",
            require_clean_tree=False,
            min_eligible=3,
        )
