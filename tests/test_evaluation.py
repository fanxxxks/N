from __future__ import annotations

import json

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
from ashare_model.data_loader import AshareDataLoader
from ashare_model.evaluation import (
    PROTOCOL_VERSION,
    aggregate_results,
    baseline_candidates,
    benchmark_row,
    build_result,
    epoch_slice,
    evaluate_formula,
    evaluate_signal,
    resolve_folds,
    run_fold,
    run_protocol,
    top_trial,
)
from ashare_model.reward import REWARD_VERSION
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB


def _loader(populated_db: DataConfig) -> AshareDataLoader:
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    return loader


def _fold(dates, train_end="2024-01-10", test_end="2024-01-25") -> evaluation.Fold:
    return resolve_folds([FoldConfig(train_end, test_end)], dates)[0]


class _FakeTrainer:
    """Trainer stand-in injected through evaluation._build_trainer."""

    def __init__(self, tokens, best_reward=1.5, best_formula="RET_1", history=None):
        self._tokens = tokens
        self.best_reward = best_reward
        self.best_formula = best_formula
        self.history = history if history is not None else [{"avg_reward": 0.1}]

    def train(self, **kwargs):
        return self._tokens


# --- fold resolution and slicing -------------------------------------------


def test_resolve_folds_maps_dates_to_columns(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    assert loader.dates[fold.train_end_idx] >= "20240110"
    assert loader.dates[fold.train_end_idx - 1] < "20240110"
    assert loader.dates[fold.test_end_idx - 1] < "20240125"
    assert fold.test_end_idx - fold.train_end_idx >= 3


def test_resolve_folds_rejects_insufficient_data(populated_db: DataConfig):
    loader = _loader(populated_db)
    dates = loader.dates
    # train_end past the data range: the test window collapses.
    with pytest.raises(ValueError, match="test window is empty"):
        resolve_folds([FoldConfig("2024-03-01", "2024-03-15")], dates)
    # train window clamps to one column: too small to train.
    with pytest.raises(ValueError, match="train window has"):
        resolve_folds([FoldConfig("2023-12-01", "2024-01-25")], dates)
    # test window of two columns is degenerate.
    with pytest.raises(ValueError, match="test window has"):
        resolve_folds([FoldConfig("2024-01-03", "2024-01-05")], dates)


def test_epoch_slice_aligns_to_fold_indices(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    factors, raw, target, dates = epoch_slice(loader, fold)
    assert dates == loader.dates[fold.train_end_idx : fold.test_end_idx]
    assert factors.shape == (FORMULA_VOCAB.feature_count, 3, len(dates))
    assert target.shape == (3, len(dates))
    for key in ("open", "high", "low", "close", "pre_close", "volume"):
        assert raw[key].shape == (3, len(dates))


# --- scoring ----------------------------------------------------------------


def test_evaluate_signal_matches_engine_run(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    factors, raw, _, dates = epoch_slice(loader, fold)
    signal = factors[0]
    metrics = evaluate_signal(signal, loader, fold, BacktestConfig())
    result = AshareBacktestEngine(BacktestConfig()).run(
        signal, raw, loader.ts_codes, dates, stock_names=loader.stock_names
    )
    assert metrics["total_return"] == pytest.approx(
        result.metrics["total_return"], abs=1e-12
    )
    assert metrics["sharpe"] == pytest.approx(result.metrics["sharpe"], abs=1e-12)
    assert metrics["average_turnover"] == pytest.approx(
        result.metrics["average_turnover"], abs=1e-12
    )
    assert metrics["benchmark_return"] == pytest.approx(
        result.benchmark_equity[-1] - 1.0, abs=1e-12
    )
    # Three stocks are below the IC minimum cross-section, so IC stats
    # degrade to the documented zeros instead of NaN.
    assert metrics["ic_mean"] == 0.0 and metrics["n_ic_dates"] == 0


def test_evaluate_formula_equals_sliced_feature_signal(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    # Token 1 is the first vocabulary feature (RET_1).
    m1 = evaluate_formula([1], loader, fold, BacktestConfig())
    factors, _, _, _ = epoch_slice(loader, fold)
    m2 = evaluate_signal(factors[0], loader, fold, BacktestConfig())
    assert m1 is not None
    assert m1 == m2


def test_evaluate_formula_returns_none_for_invalid_tokens(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    assert evaluate_formula([999], loader, fold, BacktestConfig()) is None
    assert evaluate_formula([], loader, fold, BacktestConfig()) is None


def test_evaluate_signal_rejects_misaligned_signal(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_signal(
            loader.factor_tensor[0].numpy()[:, : fold.test_end_idx],
            loader,
            fold,
            BacktestConfig(),
        )


def test_benchmark_row_matches_engine_reference_curve(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    factors, raw, _, dates = epoch_slice(loader, fold)
    result = AshareBacktestEngine(BacktestConfig()).run(
        factors[0], raw, loader.ts_codes, dates, stock_names=loader.stock_names
    )
    row = benchmark_row(loader, fold)
    assert row["candidate"] == "benchmark:equal_weight"
    assert row["total_return"] == pytest.approx(
        result.benchmark_equity[-1] - 1.0, abs=1e-12
    )
    assert row["average_turnover"] is None  # cost-free reference, no turnover


def test_baseline_candidates_match_factor_rows(populated_db: DataConfig):
    loader = _loader(populated_db)
    proto = ProtocolConfig(baseline_signals=["MOMENTUM_20", "ROE"])
    fold = _fold(loader.dates)
    rows = baseline_candidates(loader, proto, fold, BacktestConfig())
    assert [r["candidate"] for r in rows] == [
        "baseline:MOMENTUM_20",
        "baseline:ROE",
    ]
    factors, _, _, _ = epoch_slice(loader, fold)
    for row, name in zip(rows, ["MOMENTUM_20", "ROE"]):
        ref = evaluate_signal(
            factors[FEATURE_NAMES.index(name)], loader, fold, BacktestConfig()
        )
        assert row["total_return"] == ref["total_return"]
        assert row["seed"] is None and row["val_reward"] is None


# --- aggregation and selection ---------------------------------------------


def test_aggregate_results_median_iqr_and_failed_skipping():
    rows = [
        {"candidate": "trained", "sharpe": 1.0, "failed": False},
        {"candidate": "trained", "sharpe": 2.0, "failed": False},
        {"candidate": "trained", "sharpe": 4.0, "failed": False},
        {"candidate": "trained", "sharpe": 100.0, "failed": False},
        {"candidate": "trained", "failed": True},
    ]
    agg = aggregate_results(rows)
    assert agg["trained"]["n_rows"] == 5
    s = agg["trained"]["metrics"]["sharpe"]
    assert s["median"] == pytest.approx(3.0)
    assert s["q25"] == pytest.approx(1.75)
    assert s["q75"] == pytest.approx(28.0)
    assert s["iqr"] == pytest.approx(26.25)
    assert s["n"] == 4
    assert "formula_text" not in agg["trained"]


def test_top_trial_ignores_val_reward(monkeypatch):
    """Adjudication must be reward-version-independent: val_reward is
    archived provenance, never a ranking input."""

    rows = [
        {
            "candidate": "A",
            "fold_test_end": "2024-01-25",
            "seed": 1,
            "sharpe": 1.0,
            "val_reward": -2.0,
            "failed": False,
        },
        {
            "candidate": "B",
            "fold_test_end": "2024-01-25",
            "seed": 1,
            "sharpe": 0.5,
            "val_reward": 4.9,
            "failed": False,
        },
    ]
    assert top_trial(rows)["candidate"] == "A"
    agg_before = aggregate_results(rows)
    monkeypatch.setattr(evaluation, "REWARD_VERSION", "999")
    assert aggregate_results(rows) == agg_before
    assert top_trial(rows)["candidate"] == "A"


def test_build_result_schema_and_sanitization():
    proto = ProtocolConfig()
    tier = proto.screening
    rows = [
        {
            "candidate": "trained",
            "sharpe": float("nan"),
            "failed": False,
            "formula_text": "X",
        }
    ]
    result = build_result(proto, "screening", tier, rows, data_end_date="20240201")
    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["reward_version"] == REWARD_VERSION
    assert result["frequency"] == "daily"
    assert result["horizon"] == 1
    assert result["tier"] == "screening"
    assert result["steps"] == tier.steps
    assert result["batch_size"] == tier.batch_size
    assert result["folds"] == [
        {"train_end": f.train_end, "test_end": f.test_end} for f in proto.folds
    ]
    assert result["seeds"] == proto.seeds
    assert result["data_end_date"] == "20240201"
    assert result["n_candidates"] == 1
    # Non-finite floats are sanitized so the artifact stays valid JSON.
    assert result["rows"][0]["sharpe"] is None
    assert result["top_trial"] is None
    assert result["dsr"] is None and result["max_t"] is None


# --- fold-level runs --------------------------------------------------------


def test_run_fold_success_with_fake_trainer(populated_db: DataConfig, monkeypatch):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    tier = TierConfig(1, 256)
    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([1])
    )
    row = run_fold(
        loader, populated_db, ModelConfig(), BacktestConfig(), None, tier, fold, 42
    )
    assert not row["failed"]
    assert row["formula"] == [1] and row["formula_text"] == "RET_1"
    assert row["val_reward"] == 1.5
    assert row["seed"] == 42
    assert row["fold_test_end"] == fold.test_end
    assert row["total_return"] == pytest.approx(
        evaluate_formula([1], loader, fold, BacktestConfig())["total_return"]
    )


def test_run_fold_failure_paths(populated_db: DataConfig, monkeypatch):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    tier = TierConfig(1, 256)
    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer(None)
    )
    row = run_fold(
        loader, populated_db, ModelConfig(), BacktestConfig(), None, tier, fold, 42
    )
    assert row["failed"] and row["reason"] == "no valid formula found"

    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([999])
    )
    row = run_fold(
        loader, populated_db, ModelConfig(), BacktestConfig(), None, tier, fold, 42
    )
    assert row["failed"] and row["reason"] == "formula invalid at eval time"


# --- protocol orchestration -------------------------------------------------


def test_run_protocol_rows_and_determinism(populated_db: DataConfig, monkeypatch):
    loader = _loader(populated_db)
    proto = ProtocolConfig(
        folds=[FoldConfig("2024-01-10", "2024-01-25")], seeds=[42, 7]
    )
    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([1])
    )
    result = run_protocol(
        loader,
        populated_db,
        ModelConfig(),
        BacktestConfig(),
        None,
        proto,
        "screening",
    )
    rows = result["rows"]
    assert len(rows) == 6  # 1 benchmark + 3 baselines + 2 seeds
    trained = [r for r in rows if r["candidate"] == "trained"]
    assert len(trained) == 2
    assert all(r["formula"] == [1] and r["val_reward"] == 1.5 for r in trained)
    assert result["aggregates"]["trained"]["n_rows"] == 2
    assert result["top_trial"] is not None
    assert result["data_end_date"] == loader.dates[-1]
    assert result["tier"] == "screening"
    assert result["steps"] == proto.screening.steps

    again = run_protocol(
        loader,
        populated_db,
        ModelConfig(),
        BacktestConfig(),
        None,
        proto,
        "screening",
    )
    assert again == result


def test_run_protocol_validates_tier_and_fold_indices(populated_db: DataConfig):
    loader = _loader(populated_db)
    proto = ProtocolConfig()
    with pytest.raises(ValueError, match="unknown tier"):
        run_protocol(
            loader, populated_db, ModelConfig(), BacktestConfig(), None, proto, "nope"
        )
    with pytest.raises(ValueError, match="out of range"):
        run_protocol(
            loader,
            populated_db,
            ModelConfig(),
            BacktestConfig(),
            None,
            proto,
            "screening",
            fold_indices=[99],
        )


def test_run_protocol_rejects_unknown_baseline(populated_db: DataConfig):
    loader = _loader(populated_db)
    proto = ProtocolConfig(baseline_signals=["NOT_A_FACTOR"])
    with pytest.raises(ValueError, match="NOT_A_FACTOR"):
        run_protocol(
            loader, populated_db, ModelConfig(), BacktestConfig(), None, proto, "screening"
        )


# --- CLI smoke --------------------------------------------------------------


def test_cli_smoke(tmp_path, populated_db: DataConfig):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "protocol_result.json"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(populated_db.data_dir),
                "duckdb_path": str(populated_db.duckdb_path),
                "parquet_dir": str(populated_db.parquet_dir),
                "model": {"max_formula_len": 6},
                "protocol": {
                    "folds": [
                        {"train_end": "2024-01-10", "test_end": "2024-01-25"}
                    ],
                    "seeds": [42],
                    "screening": {"steps": 1, "batch_size": 256},
                },
            }
        ),
        encoding="utf-8",
    )
    rc = evaluation.main(
        [
            "--config",
            str(cfg_path),
            "--tier",
            "screening",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["tier"] == "screening"
    assert payload["steps"] == 1 and payload["batch_size"] == 256
    assert payload["seeds"] == [42]
    assert any(r["candidate"] == "baseline:MOMENTUM_20" for r in payload["rows"])
    assert any(r["candidate"] == "benchmark:equal_weight" for r in payload["rows"])
    # The protocol must never clobber the working strategy artifacts.
    assert not (populated_db.data_dir / "best_ashare_strategy.json").exists()
