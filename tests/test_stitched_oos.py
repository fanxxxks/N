"""T4-01 contract tests: stitched out-of-sample protocol (v20).

Phase-4 rule: nested walk-forward — the inner loop tunes formula,
reward and hyperparameters inside each fold's training window, the outer
loop performs exactly one algorithm evaluation per (fold, seed), and
the **outer OOS returns are stitched per (algorithm, seed)** before any
adjudication statistic (Sharpe / DSR / max-t) is computed.  One trial =
one stitched (candidate, seed) series, never one fold row.

These tests pin the contract before the implementation exists:

* stitching concatenates fold OOS series in chronological fold order;
* failed folds are recorded, never silently dropped, and contribute no
  returns; a (candidate, seed) with no succeeded fold produces no trial;
* the same fold measured twice for one (candidate, seed) is an integrity
  error;
* DSR / max-t / top-trial adjudicate on the stitched matrix;
* protocol runs record every trial in the ledger (failures included) and
  refuse folds that touch a locked holdout slice;
* the artifact carries the stitched block, ledger and data-regime info.
"""

from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    FoldConfig,
    ModelConfig,
    ProtocolConfig,
    TierConfig,
)
from ashare_model import evaluation
from ashare_model.backtest import AshareBacktestEngine
from ashare_model.evaluation import (
    PROTOCOL_VERSION,
    build_result,
    dsr_from_rows,
    max_t_from_rows,
    run_protocol,
    stitch_oos_series,
    stitched_metrics,
    top_trial,
)
from ashare_model.ledger import ExperimentLedger
from ashare_model.regime import (
    RegimeRegistry,
    declare_dev_regime,
    lock_slice,
)

from test_evaluation import _FakeTrainer, _loader  # noqa: F401


def _row(
    candidate="trained",
    seed=42,
    fold_train_end="2023-12-31",
    fold_test_end="2024-12-31",
    daily=None,
    bench=None,
    failed=False,
    **extra,
):
    t = len(daily) if daily is not None else 3
    daily = [float(x) for x in daily] if daily is not None else [0.001] * t
    bench = [float(x) for x in bench] if bench is not None else [0.0] * t
    row = {
        "candidate": candidate,
        "seed": seed,
        "fold_train_end": fold_train_end,
        "fold_test_end": fold_test_end,
        "daily_returns": daily,
        "benchmark_daily_returns": bench,
        "failed": failed,
        "formula_text": "X",
        "average_turnover": 5.0,
        "capacity_utilization": 0.1,
    }
    row.update(extra)
    return row


def _two_fold_rows(candidate="trained", seed=42):
    return [
        _row(candidate, seed, "2023-12-31", "2024-12-31", [0.01, 0.02]),
        _row(candidate, seed, "2024-12-31", "2025-12-31", [0.03, 0.04]),
    ]


# --- stitching --------------------------------------------------------------


def test_stitch_concatenates_folds_in_chronological_order():
    rows = _two_fold_rows()
    trials = stitch_oos_series(rows)
    assert len(trials) == 1
    trial = trials[0]
    assert trial["candidate"] == "trained" and trial["seed"] == 42
    assert trial["daily_returns"] == [0.01, 0.02, 0.03, 0.04]
    assert trial["n_days"] == 4
    assert [s["fold_test_end"] for s in trial["segments"]] == [
        "2024-12-31",
        "2025-12-31",
    ]
    assert trial["failed_folds"] == []
    assert trial["algorithm"] == "trained"


def test_stitch_orders_segments_by_fold_dates():
    rows = [
        _row("A", 1, "2024-12-31", "2025-12-31", [0.03]),
        _row("A", 1, "2023-12-31", "2024-12-31", [0.01]),
    ]
    trial = stitch_oos_series(rows)[0]
    assert trial["daily_returns"] == [0.01, 0.03]


def test_stitch_excludes_failed_folds_but_records_them():
    rows = [
        _row("trained", 42, "2023-12-31", "2024-12-31", [0.01], failed=True),
        _row("trained", 42, "2024-12-31", "2025-12-31", [0.03]),
    ]
    trial = stitch_oos_series(rows)[0]
    assert trial["daily_returns"] == [0.03]
    assert trial["failed_folds"] == [
        {"fold_train_end": "2023-12-31", "fold_test_end": "2024-12-31"}
    ]
    # A (candidate, seed) with no succeeded fold produces no trial.
    assert stitch_oos_series([_row("gone", 1, failed=True)]) == []


def test_stitch_excess_is_daily_minus_benchmark():
    rows = [
        _row("A", 1, daily=[0.05, -0.01], bench=[0.01, 0.01]),
        _row("A", 1, "2024-12-31", "2025-12-31", daily=[0.02], bench=[0.0]),
    ]
    trial = stitch_oos_series(rows)[0]
    assert trial["excess_returns"] == pytest.approx([0.04, -0.02, 0.02])


def test_stitch_missing_benchmark_uses_zeros():
    row = _row("A", 1, daily=[0.05, 0.01])
    row.pop("benchmark_daily_returns")
    trial = stitch_oos_series([row])[0]
    assert trial["excess_returns"] == pytest.approx([0.05, 0.01])


def test_stitch_duplicate_fold_row_is_integrity_error():
    rows = _two_fold_rows() + [_row("trained", 42, "2024-12-31", "2025-12-31", [0.99])]
    with pytest.raises(ValueError):
        stitch_oos_series(rows)


def test_stitch_algorithm_and_provenance_fields():
    rows = [
        _row("baseline:RSQ_60", None, daily=[0.01], formula_text="RSQ_60"),
        _row("baseline:RSQ_60", None, "2024-12-31", "2025-12-31",
             daily=[0.02], formula_text="RSQ_60"),
        _row("benchmark:equal_weight", None, daily=[0.005]),
    ]
    trials = {t["candidate"]: t for t in stitch_oos_series(rows)}
    assert trials["baseline:RSQ_60"]["algorithm"] == "baseline"
    assert trials["benchmark:equal_weight"]["algorithm"] == "benchmark"
    # Provenance fields for drill-down: last-segment fold anchor, formula
    # text, and fold-level portfolio statistics aggregated over segments.
    assert trials["baseline:RSQ_60"]["fold_test_end"] == "2025-12-31"
    assert trials["baseline:RSQ_60"]["formula_text"] == "RSQ_60"
    assert trials["baseline:RSQ_60"]["average_turnover_mean"] == pytest.approx(5.0)
    assert trials["baseline:RSQ_60"]["capacity_utilization_max"] == pytest.approx(0.1)


def test_stitched_metrics_match_engine_on_concatenated_series():
    daily = [0.001, -0.002, 0.003, 0.004, -0.001]
    trial = stitch_oos_series([_row("A", 1, daily=daily)])[0]
    metrics = stitched_metrics(trial)
    equity = AshareBacktestEngine._equity_curve(daily)
    engine = AshareBacktestEngine._metrics(daily, equity)
    assert metrics["sharpe"] == pytest.approx(engine["sharpe"])
    assert metrics["max_drawdown"] == pytest.approx(engine["max_drawdown"])
    assert metrics["total_return"] == pytest.approx(engine["total_return"])
    assert metrics["annual_return"] == pytest.approx(engine["annual_return"])
    assert metrics["n_days"] == 5


# --- stitched adjudication --------------------------------------------------


def test_dsr_uses_stitched_trial_matrix():
    # 3 candidates x 2 folds: the trial matrix is 3 stitched series, not
    # 6 fold rows.
    rows = []
    for i in range(3):
        rng = np.random.default_rng(i)
        rows.append(_row(f"c{i}", i, daily=rng.normal(0.0, 0.01, 50)))
        rows.append(_row(f"c{i}", i, "2024-12-31", "2025-12-31",
                         daily=rng.normal(0.0, 0.01, 50)))
    dsr = dsr_from_rows(rows)
    assert dsr is not None and dsr["n_trials"] == 3
    assert dsr["t_best"] == 100


def test_dsr_extra_rows_join_the_stitched_pool():
    rows = [_row("a", 1, daily=[0.001] * 20)]
    extra = [_row("x", 9, daily=[0.002] * 20)]
    dsr = dsr_from_rows(rows, extra_trial_rows=extra)
    assert dsr["n_trials"] == 2


def test_max_t_uses_stitched_trial_matrix():
    rows = []
    for i in range(3):
        rng = np.random.default_rng(i)
        rows.append(_row(f"c{i}", i, daily=rng.normal(0.0, 0.01, 50)))
        rows.append(_row(f"c{i}", i, "2024-12-31", "2025-12-31",
                         daily=rng.normal(0.0, 0.01, 50)))
    mt = max_t_from_rows(rows, n_perms=200)
    assert mt is not None and mt["n_trials"] == 3


def test_top_trial_is_a_stitched_trial():
    rows = _two_fold_rows("good", 1) + _two_fold_rows("bad", 2)
    # Make "good" clearly stronger in both folds.
    for row in rows:
        if row["candidate"] == "good":
            row["daily_returns"] = [0.004, 0.004]
        else:
            row["daily_returns"] = [-0.004, -0.004]
    top = top_trial(rows)
    assert top["candidate"] == "good"
    assert top["n_days"] == 4
    assert top["seed"] == 1


def test_build_result_v20_schema_with_stitched_block():
    rows = _two_fold_rows("trained", 42)
    result = build_result(
        ProtocolConfig(), "screening", TierConfig(150, 256), rows,
        data_end_date="2026-01-01",
    )
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert "stitched" in result
    assert result["stitched"]["n_trials"] == 1
    assert result["stitched"]["trials"][0]["candidate"] == "trained"
    assert result["top_trial"]["candidate"] == "trained"
    assert result["dsr"] is not None and result["max_t"] is not None
    assert result["dsr"]["n_trials"] == 1


def test_build_result_v21_records_data_tiers():
    """P2-02: every row / stitched trial / top trial traces back to the
    credibility tier of its features; the artifact pins the tier version."""
    from ashare_model.data_tier import DATA_TIER_VERSION
    from ashare_model.vocab import FORMULA_VOCAB

    def _tok(name: str) -> int:
        return FORMULA_VOCAB.feature_offset + FORMULA_VOCAB.feature_names.index(name)

    rows = _two_fold_rows("trained", 42)
    for row in rows:
        row["formula_text"] = "ROE"
        row["formula"] = [_tok("ROE")]
    rows[0]["formula_text"] = "RET_1"
    rows[0]["formula"] = [_tok("RET_1")]
    result = build_result(
        ProtocolConfig(), "screening", TierConfig(150, 256), rows,
        data_end_date="2026-01-01",
    )
    assert result["data_tier_version"] == DATA_TIER_VERSION
    assert result["rows"][0]["data_tier"]["max_tier"] == "A"
    assert result["rows"][1]["data_tier"]["max_tier"] == "B"
    assert result["rows"][1]["data_tier"]["tiers_used"] == ["B"]
    # The stitched trial and the top trial carry the weak tier of the
    # recorded formula text (the top trial is the ROE row here).
    assert result["stitched"]["trials"][0]["data_tier"]["max_tier"] == "B"
    assert result["top_trial"]["data_tier"]["max_tier"] == "B"


def test_build_result_untraceable_rows_record_null_tier():
    rows = _two_fold_rows("trained", 42)
    for row in rows:
        row.pop("formula_text", None)
        row.pop("formula", None)
    result = build_result(
        ProtocolConfig(), "screening", TierConfig(150, 256), rows,
        data_end_date="2026-01-01",
    )
    assert result["rows"][0]["data_tier"] is None
    assert result["stitched"]["trials"][0]["data_tier"] is None
    assert result["top_trial"]["data_tier"] is None


def test_build_result_records_ledger_and_regime():
    rows = _two_fold_rows()
    regime = RegimeRegistry(
        "n/a", declare_dev_regime("2026-12-31")
    )
    result = build_result(
        ProtocolConfig(), "screening", TierConfig(150, 256), rows,
        data_end_date="2026-01-01",
        ledger={"path": "data/ledger.jsonl", "run_id": "run-1", "tainted": False,
                "n_trials": 4},
        regime=regime,
    )
    assert result["ledger"] == {
        "path": "data/ledger.jsonl", "run_id": "run-1", "tainted": False,
        "n_trials": 4,
    }
    assert result["data_regime"]["dev_cutoff"] == "2026-12-31"
    # Default folds (test windows 2021..2025) classify as dev data.
    kinds = [f["kind"] for f in result["data_regime"]["folds"]]
    assert kinds == ["dev"] * 5


# --- ledger + regime wiring through run_protocol -----------------------------


def _small_proto_config() -> ProtocolConfig:
    return ProtocolConfig(
        seeds=[42],
        folds=[
            FoldConfig("2024-01-10", "2024-01-25"),
        ],
        baseline_signals=["MOMENTUM_20"],
        random_samples=0,
        gp_enabled=False,
        tpe_enabled=False,
        screening=TierConfig(1, 256),
    )


def test_run_protocol_records_every_trial_in_ledger(
    populated_db: DataConfig, monkeypatch, tmp_path
):
    loader = _loader(populated_db)
    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([1])
    )
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl", run_id="run-t4")
    result = run_protocol(
        loader,
        populated_db,
        ModelConfig(max_formula_len=6),
        BacktestConfig(),
        None,
        _small_proto_config(),
        "screening",
        ledger=ledger,
    )
    # 1 benchmark (not a trial) + 1 baseline batch + 1 trained trial.
    trials = list(ledger.trials_for())
    assert len(trials) == 2
    assert all(t.status == "succeeded" for t in trials)
    assert ledger.finalize() == []
    # Rows carry their trial ids back into the artifact.
    assert result["rows"][0]["candidate"] == "benchmark:equal_weight"
    trained = next(r for r in result["rows"] if r["candidate"] == "trained")
    assert trained["trial_id"].startswith("run-t4-")
    assert result["ledger"]["run_id"] == "run-t4"
    assert result["ledger"]["tainted"] is False


def test_run_protocol_ledger_records_crash_as_failed(
    populated_db: DataConfig, monkeypatch, tmp_path
):
    loader = _loader(populated_db)

    def _boom(*args, **kwargs):
        raise RuntimeError("crash mid-trial")

    monkeypatch.setattr(evaluation, "_build_trainer", _boom)
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl", run_id="run-crash")
    with pytest.raises(RuntimeError, match="crash mid-trial"):
        run_protocol(
            loader,
            populated_db,
            ModelConfig(max_formula_len=6),
            BacktestConfig(),
            None,
            _small_proto_config(),
            "screening",
            ledger=ledger,
        )
    # The crashed trial is recorded as failed — never silently dropped —
    # and no trial is left open.
    failed = [t for t in ledger.trials_for() if t.status == "failed"]
    assert len(failed) == 1
    assert "crash mid-trial" in (failed[0].error or "")
    assert ledger.finalize() == []


def test_run_protocol_rejects_locked_holdout_before_any_trial(
    populated_db: DataConfig, monkeypatch, tmp_path
):
    from ashare_model.regime import HoldoutViolation

    loader = _loader(populated_db)
    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([1])
    )
    regime = RegimeRegistry(
        tmp_path / "registry.json",
        lock_slice(
            declare_dev_regime("2024-01-15"),
            start="2024-01-20",
            end="2024-02-29",
            dataset_id=loader.dataset_id,
        ),
    )
    ledger = ExperimentLedger(tmp_path / "ledger.jsonl", run_id="run-lock")
    with pytest.raises(HoldoutViolation):
        run_protocol(
            loader,
            populated_db,
            ModelConfig(max_formula_len=6),
            BacktestConfig(),
            None,
            _small_proto_config(),
            "screening",
            ledger=ledger,
            regime=regime,
        )
    # Zero trials ran: the gate fired before the first trial.
    assert list(ledger.trials_for()) == []
    assert ledger.finalize() == []


def test_selfcheck_rows_stitch_into_one_trial(populated_db: DataConfig):
    loader = _loader(populated_db)
    proto = ProtocolConfig(
        folds=[
            FoldConfig("2024-01-10", "2024-01-25"),
            FoldConfig("2024-01-25", "2024-02-10"),
        ]
    )
    rows = evaluation.selfcheck_rows(loader, proto, BacktestConfig())
    trials = stitch_oos_series(rows)
    assert len(trials) == 1
    assert trials[0]["algorithm"] == "selfcheck"
    assert trials[0]["n_days"] == sum(r["n_dates"] for r in rows)
