"""P34 R01 WP3 report-completion tests (contract rev 3, 2026-09-18).

Executable projection of the p34 §4 Rev 3 additions (user direct
instruction 「完成工作包 3」): the round-1 report must separate

* per-(arm, fold, cost level) test-window net excess AND active IR,
* cross-fold **stitched** statistics computed on the concatenated daily
  series (never the mean of per-fold IRs standing in for the stitched
  IR) alongside the per-fold annual distribution,
* the execution funnel (signal change / target change / planned vs
  filled orders / suppression reasons / holdings state) so that a
  normally quiet book (target ≈ current, no order needed) is
  distinguishable from a blocked one (orders wanted, suppressed),
* explicit statuses for failed folds, a missing benchmark reference and
  incomplete data,
* a descriptive four-way arm diagnostic (weak signal / execution
  blocked / cost swallowed / measurement incomplete) that is REPORTED
  alongside — never substituted for — the frozen §3.7 adjudication,
* the engine-version attribution stating honestly that no prior
  (pre-p35) execution-version baseline numbers exist.

The frozen arms, directions, cost ladder and adjudication thresholds
are pinned by ``test_p34_r01_baseline.py``; this file only pins the
Rev 3 additions.  Fixture reuse: the mini dual-DB environment is
imported from the sibling test module (single fixture source).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from ashare_data.config import (
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
)

from test_p34_r01_baseline import (
    _loaded_mini_loader,
    gate_run_asof,  # noqa: F401 — re-exported fixture for the e2e tests
    mini_env,  # noqa: F401 — single mini-environment fixture source
)

# The closed headline set of the descriptive four-way diagnostic.
HEADLINES = {
    "measurement_incomplete",
    "weak_signal",
    "execution_blocked",
    "cost_swallowed",
    "net_space_shown",
}

# The frozen descriptive thresholds (rev 3) — the report must carry
# them so the classification is self-describing, never tuned in secret.
FULLY_SUPPRESSED_SHARE_MAX = 0.10
UNFILLED_ORDER_RATE_MAX = 0.10
MIN_INVESTED_RATIO = 0.5


# --- W1: per-row active IR + the execution funnel ------------------------------


def test_w1_row_active_ir_and_funnel_partition(mini_env):
    """``evaluate_signal`` rows carry the test-window active IR (recomputed
    here from the row's own daily series — the single authority formula)
    and a funnel whose due-session buckets partition exactly."""

    from ashare_model.baseline_harness import oos_active_ir
    from ashare_model.eval_folds import epoch_slice, resolve_folds
    from ashare_model.eval_metrics import evaluate_signal
    from ashare_model.vocab import FEATURE_NAMES

    _, data_config, cfg_path = mini_env
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_config = make_data_config(raw, mini_env[0])
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    proto_cfg = make_protocol_config(raw)
    loader = _loaded_mini_loader(data_config, model_config)
    fold = resolve_folds(
        proto_cfg.folds, loader.dates,
        frequency=proto_cfg.frequency, horizon=proto_cfg.horizon,
    )[0]
    factors, _, _, _ = epoch_slice(loader, fold)
    signal = np.asarray(
        factors[FEATURE_NAMES.index("MOMENTUM_20")], dtype=np.float64
    )
    row = evaluate_signal(signal, loader, fold, backtest_config)

    # Test-window active IR on the row, equal to the authority formula
    # applied to the row's own series (not the scorer's train-window IR).
    expected_ir = oos_active_ir(
        np.asarray(row["daily_returns"]),
        np.asarray(row["benchmark_daily_returns"]),
    )
    assert math.isclose(row["oos_active_ir"], expected_ir, rel_tol=0, abs_tol=1e-12)

    # The funnel buckets partition the due sessions exactly.
    buckets = (
        row["funnel_due_no_signal_sessions"]
        + row["funnel_due_target_stable_sessions"]
        + row["funnel_due_fully_suppressed_sessions"]
        + row["funnel_due_partial_suppressed_sessions"]
        + row["funnel_due_executed_clean_sessions"]
    )
    assert buckets == row["funnel_due_sessions"] == row["rebalance_due_count"]
    assert row["funnel_not_due_sessions"] + row["funnel_due_sessions"] == (
        row["funnel_sessions_total"]
    )
    # Suppression reasons are disjoint by construction (the threshold pass
    # runs before the min-trade pass on the updated target).
    assert (
        row["funnel_suppressed_threshold_count"]
        + row["funnel_suppressed_min_trade_count"]
        == row["suppressed_trade_count"]
    )
    # The engine can only fill orders the constructor planned; the gap is
    # the cash-capped / missing-bar unfilled share.
    assert row["funnel_filled_order_count"] <= row["funnel_planned_order_count"]
    assert row["funnel_planned_turnover_sum"] + 1e-9 >= (
        row["funnel_executed_turnover_sum"]
    )
    # Holdings state: the minima exist next to the means.
    assert 0 <= row["funnel_holdings_min"] <= row["holdings_mean"]
    assert 0.0 <= row["funnel_invested_ratio_min"] <= row["invested_ratio_mean"]
    assert 0 <= row["funnel_sessions_zero_holdings"] <= row["funnel_sessions_total"]


def test_w1_cash_control_is_no_signal_not_suppression(mini_env):
    """The cash control's due sessions are *no_signal* (degenerate signal),
    never *target stable* and never *suppressed* — the funnel keeps an
    empty book from masquerading as either quiet or blocked."""

    from ashare_model.eval_folds import epoch_slice, resolve_folds
    from ashare_model.eval_metrics import evaluate_signal

    _, data_config, cfg_path = mini_env
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_config = make_data_config(raw, mini_env[0])
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    proto_cfg = make_protocol_config(raw)
    loader = _loaded_mini_loader(data_config, model_config)
    fold = resolve_folds(
        proto_cfg.folds, loader.dates,
        frequency=proto_cfg.frequency, horizon=proto_cfg.horizon,
    )[0]
    _, _, _, dates = epoch_slice(loader, fold)
    cash = np.full((len(loader.ts_codes), len(dates)), np.nan)
    row = evaluate_signal(cash, loader, fold, backtest_config)
    assert row["funnel_due_sessions"] >= 1  # sessions came due
    assert (
        row["funnel_due_no_signal_sessions"] == row["funnel_due_sessions"]
    )
    assert row["funnel_due_target_stable_sessions"] == 0
    assert row["funnel_due_fully_suppressed_sessions"] == 0
    assert row["funnel_due_partial_suppressed_sessions"] == 0
    assert row["suppressed_trade_count"] == 0
    assert row["funnel_sessions_zero_holdings"] == row["funnel_sessions_total"]
    # The cash book earns exactly zero while the benchmark does not: its
    # active IR is the authority formula on its own series (strongly
    # negative against a rising benchmark — that is the control's point,
    # so we pin consistency, not a magic constant).
    from ashare_model.baseline_harness import oos_active_ir as _ir

    assert row["oos_active_ir"] == _ir(
        np.asarray(row["daily_returns"]),
        np.asarray(row["benchmark_daily_returns"]),
    )


# --- W2: stitched cross-fold statistics + the annual distribution --------------


def _fold_row(
    candidate: str,
    test_end: str,
    level: str,
    daily: list[float],
    bench: list[float],
    *,
    failed: bool = False,
    excess: float | None = None,
    holdings_mean: float = 10.0,
    invested_ratio_mean: float = 0.95,
    turnover: float = 0.2,
) -> dict:
    """A minimal but complete summary-feed row (fields the summarizer and
    the stitching authority consume)."""

    bench_total = float(np.prod([1.0 + x for x in bench]) - 1.0)
    total = float(np.prod([1.0 + x for x in daily]) - 1.0)
    if excess is None:
        excess = (1.0 + total) / (1.0 + bench_total) - 1.0
    return {
        "candidate": candidate,
        "fold_train_end": "train",
        "fold_test_end": test_end,
        "seed": None,
        "cost_level": level,
        "failed": failed,
        "failure_reason": "boom" if failed else None,
        "n_dates": len(daily),
        "total_return": total,
        "benchmark_return": bench_total,
        "excess_return": excess,
        "average_turnover": turnover,
        "holdings_mean": holdings_mean,
        "holdings_max": int(holdings_mean) + 1,
        "invested_ratio_mean": invested_ratio_mean,
        "rebalance_due_count": 6,
        "rebalance_count": 6,
        "daily_returns": list(daily),
        "benchmark_daily_returns": list(bench),
        # p40 R2: mark dates derived from the fold so two folds of one
        # arm never collide (stitching requires the real date axis).
        "daily_dates": [f"{test_end}#{i:04d}" for i in range(len(daily))],
        "oos_active_ir": None,  # filled by the caller when hand-known
    }


def _three_level_rows(candidate: str, test_end: str, scale: float):
    """0x/1x/2x rows for one fold built from one daily shape."""

    daily = [0.01 * scale, 0.02 * scale, -0.01 * scale]
    bench = [0.005, 0.0, 0.0]
    rows = []
    for level, factor in (("0x", 1.0), ("1x", 0.5), ("2x", 0.25)):
        row = _fold_row(
            candidate, test_end, level,
            [x * factor for x in daily], bench,
        )
        rows.append(row)
    return rows


def test_w2_stitched_statistics_are_concatenated_not_averaged():
    """The summary's stitched block computes on the cross-fold concatenated
    daily series: its total return and active IR equal the hand recomputed
    concatenated values, and with these fold shapes the mean of per-fold
    IRs is a *different* number — substituting it is detectable."""

    from ashare_model.baseline_harness import oos_active_ir
    from ashare_model.r01_baseline import summarize_arms

    rows = _three_level_rows("baseline:X", "20231231", 1.0)
    rows += _three_level_rows("baseline:X", "20241231", 1.4)
    for row in rows:
        # Give every row a real per-fold active IR (the summarizer consumes
        # the row field; the stitched block must NOT average these).
        row["oos_active_ir"] = oos_active_ir(
            np.asarray(row["daily_returns"]),
            np.asarray(row["benchmark_daily_returns"]),
        )
    summary = summarize_arms(rows)["baseline:X"]

    stitched = summary["stitched"]["1x"]
    concat_daily = (
        rows[1]["daily_returns"] + rows[4]["daily_returns"]
    )
    concat_bench = (
        rows[1]["benchmark_daily_returns"] + rows[4]["benchmark_daily_returns"]
    )
    assert stitched["n_days"] == len(concat_daily) == 6
    assert stitched["n_segments"] == 2
    assert math.isclose(
        stitched["total_return"],
        float(np.prod([1.0 + x for x in concat_daily]) - 1.0),
        rel_tol=1e-12,
    )
    assert math.isclose(
        stitched["oos_active_ir"],
        oos_active_ir(
            np.asarray(concat_daily), np.asarray(concat_bench)
        ),
        rel_tol=1e-12,
    )
    mean_of_fold_irs = float(np.mean([rows[1]["oos_active_ir"], rows[4]["oos_active_ir"]]))
    assert not math.isclose(
        stitched["oos_active_ir"], mean_of_fold_irs, rel_tol=1e-9
    ), "the stitched IR must be a different statistic than the fold-IR mean"

    # The frozen adjudication statistics stay fold means under their own
    # names (never silently switched to stitched values).
    assert math.isclose(
        summary["mean_net_excess_1x"],
        float(np.mean([rows[1]["excess_return"], rows[4]["excess_return"]])),
        rel_tol=1e-12,
    )

    # Annual distribution: every fold, every level, net excess + IR.
    annual = summary["annual"]
    assert set(annual) == {"20231231", "20241231"}
    for fold_end in annual:
        levels = annual[fold_end]["levels"]
        assert set(levels) == {"0x", "1x", "2x"}
        for level, payload in levels.items():
            assert "excess_return" in payload and "oos_active_ir" in payload
        assert annual[fold_end]["status"] == "ok"

    # Cross-fold median/min/max of the annual values (contract §3.4).
    stats = summary["fold_stats_1x"]
    fold_excess = [rows[1]["excess_return"], rows[4]["excess_return"]]
    assert math.isclose(stats["excess_return"]["median"], float(np.median(fold_excess)), rel_tol=1e-12)
    assert math.isclose(stats["excess_return"]["min"], float(np.min(fold_excess)), rel_tol=1e-12)
    assert math.isclose(stats["excess_return"]["max"], float(np.max(fold_excess)), rel_tol=1e-12)
    assert stats["excess_return"]["n"] == 2


def test_w2_failed_fold_excluded_and_recorded():
    """A failed (arm, fold) contributes no returns to the stitched series
    and is recorded in failed_folds; its annual status says failed."""

    from ashare_model.r01_baseline import summarize_arms

    rows = _three_level_rows("baseline:X", "20231231", 1.0)
    rows += _three_level_rows("baseline:X", "20241231", 1.4)
    ok_daily = (
        rows[1]["daily_returns"] + rows[4]["daily_returns"]
    )  # both succeeded 1x folds concatenate
    rows.append(
        _fold_row("baseline:X", "20251231", "1x", [], [], failed=True)
    )
    summary = summarize_arms(rows)["baseline:X"]
    stitched = summary["stitched"]["1x"]
    assert stitched["n_days"] == len(ok_daily)
    failed_folds = {entry["fold_test_end"] for entry in stitched["failed_folds"]}
    assert failed_folds == {"20251231"}
    assert summary["annual"]["20251231"]["status"] == "failed"
    assert "boom" in (summary["annual"]["20251231"].get("failure_reason") or "")
    assert summary["failed_folds"] == 1
    assert summary["unreliable"] is False  # one failed fold: informational


# --- W3: the descriptive four-way diagnostic ------------------------------------


def _full_summary(
    *,
    failed_folds: int = 0,
    benchmark_failed_folds: int = 0,
    excess_0x: float = 0.05,
    ir_0x: float = 1.0,
    excess_1x: float = 0.01,
    excess_2x: float = 0.0,
    fully_suppressed_sessions: int = 0,
    due_sessions: int = 20,
    planned_orders: int = 60,
    filled_orders: int = 60,
    invested_ratio_mean_1x: float = 0.9,
    positive_folds_1x: int = 5,
) -> dict:
    return {
        "n_folds": 5,
        "failed_folds": failed_folds,
        "unreliable": failed_folds >= 2,
        "mean_net_excess_1x": excess_1x,
        "active_ir_1x": 0.5,
        "positive_folds_1x": positive_folds_1x,
        "mean_holdings_1x": 12.0,
        "invested_ratio_mean_1x": invested_ratio_mean_1x,
        "mean_net_excess_2x": excess_2x,
        "annual": {},
        "fold_stats_1x": {},
        "benchmark_missing_folds": [
            f"fold{i}" for i in range(benchmark_failed_folds)
        ],
        "stitched": {
            "0x": {"excess_return": excess_0x, "oos_active_ir": ir_0x, "n_days": 100},
            "1x": {"excess_return": excess_1x, "oos_active_ir": 0.4, "n_days": 100},
            "2x": {"excess_return": excess_2x, "oos_active_ir": 0.2, "n_days": 100},
        },
        "funnel_totals_1x": {
            "funnel_due_sessions": due_sessions,
            "funnel_due_fully_suppressed_sessions": fully_suppressed_sessions,
            "funnel_planned_order_count": planned_orders,
            "funnel_filled_order_count": filled_orders,
        },
    }


def test_w3_classification_four_cases():
    from ashare_model.r01_baseline import classify_arm_diagnostics

    summaries = {
        "benchmark:equal_weight": _full_summary(),
        "control:cash": _full_summary(),
        # 1. measurement incomplete dominates: a failed fold means the
        #    arm's numbers cannot support any attribution.
        "arm:incomplete": _full_summary(failed_folds=1, excess_1x=0.05),
        # 2. weak signal: no GROSS alpha at 0x — costs are not the story.
        "arm:weak": _full_summary(excess_0x=-0.02, ir_0x=-0.5, excess_1x=-0.03),
        # 3. execution blocked: healthy gross alpha but >10% of due
        #    sessions produced zero orders *because trades were dropped*.
        "arm:blocked": _full_summary(fully_suppressed_sessions=5, due_sessions=20),
        # 3b. normal low turnover is NOT blocked: zero suppressions and
        #    every planned order filled, whatever the turnover level.
        "arm:quiet": _full_summary(fully_suppressed_sessions=0),
        # 4. cost swallowed: positive at 0x, dead at 1x, clean funnel.
        "arm:eaten": _full_summary(excess_1x=-0.01, excess_2x=-0.02),
        # 5. healthy: positive net space at both levels.
        "arm:healthy": _full_summary(excess_1x=0.02, excess_2x=0.01),
    }
    diagnostics = classify_arm_diagnostics(summaries)
    # References are never classified (mirrors the adjudication scope).
    assert "benchmark:equal_weight" not in diagnostics
    assert "control:cash" not in diagnostics
    assert diagnostics["arm:incomplete"]["headline"] == "measurement_incomplete"
    assert diagnostics["arm:weak"]["headline"] == "weak_signal"
    assert diagnostics["arm:blocked"]["headline"] == "execution_blocked"
    assert diagnostics["arm:quiet"]["headline"] != "execution_blocked"
    assert diagnostics["arm:eaten"]["headline"] == "cost_swallowed"
    assert diagnostics["arm:healthy"]["headline"] == "net_space_shown"
    for arm, diag in diagnostics.items():
        assert diag["headline"] in HEADLINES
        assert isinstance(diag["flags"], list)
        assert isinstance(diag["evidence"], dict)


def test_w3_unfilled_orders_and_uninvested_block_evidence():
    from ashare_model.r01_baseline import classify_arm_diagnostics

    # Planned orders largely unfilled (cash-capped / missing bars) is
    # execution friction even when nothing was "suppressed" by threshold.
    blocked_by_unfill = classify_arm_diagnostics(
        {"arm:x": _full_summary(planned_orders=100, filled_orders=50)}
    )
    assert blocked_by_unfill["arm:x"]["headline"] == "execution_blocked"
    # Wants alpha (gross positive) but never deploys capital.
    blocked_by_uninvested = classify_arm_diagnostics(
        {"arm:x": _full_summary(invested_ratio_mean_1x=0.2)}
    )
    assert blocked_by_uninvested["arm:x"]["headline"] == "execution_blocked"
    # A missing benchmark reference makes the fold's excess unmeasurable.
    missing_bench = classify_arm_diagnostics(
        {"arm:x": _full_summary(benchmark_failed_folds=1)}
    )
    assert missing_bench["arm:x"]["headline"] == "measurement_incomplete"
    assert any("benchmark" in flag for flag in missing_bench["arm:x"]["flags"])


# --- W4/W5: the report artifact end to end --------------------------------------


def test_w5_report_v2_blocks_and_rows(mini_env, gate_run_asof, tmp_path):
    """The driven report artifact carries schema v2 with the Rev 3 blocks:
    data completeness, benchmark status, the descriptive diagnostics with
    self-describing rules, the engine-version attribution (with the honest
    no-prior-numbers statement), and oos_active_ir on every row."""

    import ashare_model.r01_baseline as r01
    from ashare_model.eval_folds import resolve_folds  # noqa: F401
    from ashare_portfolio.execution_spec import EXECUTION_SPEC_VERSION

    tmp_root, data_config, cfg_path = mini_env
    gate_run_asof(_loaded_gate_era())
    summary = r01.run_r01_baseline(
        cfg_path,
        contract_sha256=_live_contract_sha(),
        out_root=tmp_path / "r01_out_wp3",
        require_clean_tree=False,
        min_eligible=3,
        adjudication_rules={
            "min_positive_folds": 1,
            "min_avg_holdings": 2.0,
            "min_invested_ratio": 0.5,
        },
    )
    report_path = tmp_path / "r01_out_wp3" / "r01_baseline_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["r01_report_schema_version"] == 2

    # Data completeness: typed loader status + session reconciliation +
    # an explicit complete/incomplete verdict.
    completeness = payload["data_completeness"]
    assert completeness["status"] in ("complete", "incomplete")
    assert completeness["loader_frames"]["status"] == "clean"
    assert completeness["missing_trading_days"] == []
    assert completeness["expected_trading_days"] >= 1
    assert completeness["actual_trading_days"] >= 1

    # Benchmark reference status per fold, explicit.
    assert set(payload["benchmark_status"]) == {str(summary_fold_end(payload))}
    assert payload["benchmark_status"][str(summary_fold_end(payload))]["status"] == "ok"

    # Descriptive diagnostics: every cost-bearing arm, closed headline
    # set, frozen thresholds carried in the report itself.
    diagnostics = payload["arm_diagnostics"]
    assert {
        "baseline:MOMENTUM_20", "baseline:RSQ_60", "baseline:REVERSAL_60",
        "combo:all3", "combo:momentum_reversal",
    } <= set(diagnostics)
    for diag in diagnostics.values():
        assert diag["headline"] in HEADLINES
    rules = payload["diagnostic_rules"]
    assert rules["fully_suppressed_share_max"] == FULLY_SUPPRESSED_SHARE_MAX
    assert rules["unfilled_order_rate_max"] == UNFILLED_ORDER_RATE_MAX
    assert rules["min_invested_ratio"] == MIN_INVESTED_RATIO

    # Engine-version attribution: the live execution version plus the
    # honest statement that no prior-version baseline numbers exist.
    attribution = payload["engine_version_attribution"]
    assert attribution["execution_spec_version"] == EXECUTION_SPEC_VERSION
    assert attribution["portfolio_constructor_version"] >= 1
    assert attribution["prior_version_numbers_exist"] is False
    assert "p35" in attribution["prior_version_note"]

    # Reporting notes disclose the frozen raw-rank combo scale and the
    # stitched-vs-fold-mean distinction.
    assert "raw" in payload["reporting_notes"]["combo_rank_scale"].lower()
    assert "stitch" in payload["reporting_notes"]["stitched_vs_fold_mean"].lower()

    # Every row (all arms, all levels, benchmark ref included) carries
    # the test-window active IR.
    for row in payload["rows"]:
        assert "oos_active_ir" in row
        assert row["oos_active_ir"] is not None

    # The frozen adjudication is untouched and still present verbatim.
    assert payload["adjudication"] == summary["adjudication"]

    # Summaries carry stitched + annual + statuses for every arm (the
    # benchmark reference arm stitches its own "ref" level).
    for arm, arm_summary in payload["arm_summaries"].items():
        assert isinstance(arm_summary["annual"], dict)
        if arm == "benchmark:equal_weight":
            assert set(arm_summary["stitched"]) == {"ref"}
        else:
            assert set(arm_summary["stitched"]) >= {"0x", "1x", "2x"}


def _loaded_gate_era() -> str:
    from test_p34_r01_baseline import _mini_dates

    return _mini_dates()[-1]


def _live_contract_sha() -> str:
    import hashlib

    repo = Path(__file__).resolve().parents[1]
    return hashlib.sha256(
        (
            repo / "docs" / "p34_r01_fixed_baseline_diagnostic_draft.md"
        ).read_bytes()
    ).hexdigest()


def summary_fold_end(payload: dict) -> str:
    return payload["rows"][0]["fold_test_end"]
