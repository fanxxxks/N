from __future__ import annotations

import json
import math

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
from ashare_model.data_loader import AshareDataLoader
from ashare_model.evaluation import (
    PROTOCOL_VERSION,
    aggregate_results,
    baseline_candidates,
    benchmark_row,
    build_result,
    deflated_sharpe,
    dsr_from_rows,
    epoch_slice,
    evaluate_formula,
    evaluate_signal,
    expected_max_sr,
    load_trial_rows,
    max_t_from_rows,
    norm_cdf,
    norm_ppf,
    psr,
    resolve_folds,
    run_fold,
    run_protocol,
    selfcheck_rows,
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

    def __init__(
        self,
        tokens,
        best_reward=1.5,
        best_formula="RET_1",
        history=None,
        best_direction=1,
    ):
        self._tokens = tokens
        self.best_reward = best_reward
        self.best_formula = best_formula
        self.best_direction = best_direction
        self.history = history if history is not None else [{"avg_reward": 0.1}]

    def train(self, **kwargs):
        return self._tokens


# --- fold resolution and slicing -------------------------------------------


def test_resolve_folds_maps_dates_to_columns(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    contract = fold.contract
    # Configured anchors are inclusive, while every stored boundary is
    # half-open and keeps the two price columns required by t+2 exits.
    assert loader.dates[contract.train_anchor_end_exclusive - 1] == "20240110"
    assert loader.dates[contract.test_price_end - 1] == "20240125"
    assert contract.test_signal_start == contract.train_anchor_end_exclusive
    assert contract.test_signal_end == contract.test_price_end - 2
    assert contract.test_signal_count >= 1


def test_resolve_folds_rejects_insufficient_data(populated_db: DataConfig):
    loader = _loader(populated_db)
    dates = loader.dates
    # train_end past the data range: the test window collapses.
    with pytest.raises(ValueError, match=r"test window has no complete t\+2 returns"):
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
        assert row["seed"] is None
        assert "val_reward" in row and "val_icir" in row
        assert row["direction"] in (-1, 1)


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
    assert row["failed"] and row["reason"] == "no eligible formula found"

    monkeypatch.setattr(
        evaluation, "_build_trainer", lambda *a, **k: _FakeTrainer([999])
    )
    row = run_fold(
        loader, populated_db, ModelConfig(), BacktestConfig(), None, tier, fold, 42
    )
    assert row["failed"] and row["reason"] == "formula invalid at eval time"


def test_run_fold_applies_learned_direction(populated_db: DataConfig, monkeypatch):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    tier = TierConfig(1, 256)
    monkeypatch.setattr(
        evaluation,
        "_build_trainer",
        lambda *a, **k: _FakeTrainer([1], best_direction=-1),
    )
    row = run_fold(
        loader, populated_db, ModelConfig(), BacktestConfig(), None, tier, fold, 42
    )
    assert not row["failed"]
    assert row["direction"] == -1
    flipped = evaluate_formula([1], loader, fold, BacktestConfig(), direction=-1)
    assert row["total_return"] == pytest.approx(flipped["total_return"])
    assert row["ic_mean"] == pytest.approx(flipped["ic_mean"])


def test_baseline_candidates_trade_learned_direction(
    populated_db: DataConfig, monkeypatch
):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    proto = ProtocolConfig(baseline_signals=["TURNOVER"])
    rows = baseline_candidates(loader, proto, fold, BacktestConfig())
    assert len(rows) == 1
    assert rows[0]["direction"] in (-1, 1)
    factors, _, _, _ = epoch_slice(loader, fold)
    idx = FEATURE_NAMES.index("TURNOVER")
    ref = evaluate_signal(
        rows[0]["direction"] * factors[idx], loader, fold, BacktestConfig()
    )
    assert rows[0]["total_return"] == pytest.approx(ref["total_return"])
    assert rows[0]["ic_mean"] == pytest.approx(ref["ic_mean"])


def test_random_search_row_shape(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    row = evaluation.run_random_search(
        loader,
        ModelConfig(max_formula_len=6),
        BacktestConfig(),
        None,
        fold,
        n_samples=32,
        seed=7,
    )
    assert row["candidate"] == "random_search"
    assert row["fold_train_end"] == fold.train_end
    assert row["fold_test_end"] == fold.test_end
    assert row["seed"] == 7
    assert row["n_samples"] == 32
    assert row["direction"] in (-1, 1)
    if row["failed"]:
        assert row["reason"]
    else:
        assert row["formula"] and row["formula_text"]
        assert math.isfinite(row["val_reward"])


def test_random_search_is_deterministic(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    kwargs = dict(
        model_config=ModelConfig(max_formula_len=6),
        backtest_config=BacktestConfig(),
        reward_config=None,
        fold=fold,
        n_samples=32,
        seed=7,
    )
    first = evaluation.run_random_search(loader, **kwargs)
    second = evaluation.run_random_search(loader, **kwargs)
    assert first == second


def test_random_search_disabled_by_zero_budget(populated_db: DataConfig):
    loader = _loader(populated_db)
    fold = _fold(loader.dates)
    row = evaluation.run_random_search(
        loader,
        ModelConfig(),
        BacktestConfig(),
        None,
        fold,
        n_samples=0,
        seed=7,
    )
    assert row["failed"] and row["reason"]


# --- protocol orchestration -------------------------------------------------


def test_run_protocol_rows_and_determinism(populated_db: DataConfig, monkeypatch):
    loader = _loader(populated_db)
    proto = ProtocolConfig(
        folds=[FoldConfig("2024-01-10", "2024-01-25")],
        seeds=[42, 7],
        random_samples=64,
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
    assert len(rows) == 11  # 1 benchmark + 7 baselines + 1 random search + 2 seeds
    trained = [r for r in rows if r["candidate"] == "trained"]
    assert len(trained) == 2
    assert all(r["formula"] == [1] and r["val_reward"] == 1.5 for r in trained)
    assert all(r["direction"] == 1 for r in trained)
    random_rows = [r for r in rows if r["candidate"] == "random_search"]
    assert len(random_rows) == 1
    assert random_rows[0]["n_samples"] == proto.random_samples
    assert result["aggregates"]["trained"]["n_rows"] == 2
    assert result["aggregates"]["random_search"]["n_rows"] == 1
    assert result["top_trial"] is not None
    assert result["data_end_date"] == loader.dates[-1]
    assert result["tier"] == "screening"
    assert result["steps"] == proto.screening.steps
    assert result["random_samples"] == proto.random_samples
    assert result["random_seed"] == proto.random_seed

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
                    "index_codes": ["000300.SH"],
                "model": {"max_formula_len": 6},
                "protocol": {
                    "folds": [
                        {"train_end": "2024-01-10", "test_end": "2024-01-25"}
                    ],
                    "seeds": [42],
                    "screening": {"steps": 1, "batch_size": 256},
                    "random_samples": 64,
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
    assert any(r["candidate"] == "random_search" for r in payload["rows"])
    # The protocol must never clobber the working strategy artifacts.
    assert not (populated_db.data_dir / "best_ashare_strategy.json").exists()


# --- P1b: Deflated Sharpe and max-t -----------------------------------------


def _noise_rows(n=30, t=100, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        daily = rng.normal(0.0, 0.01, size=t)
        rows.append(
            {
                "candidate": f"noise{i}",
                "failed": False,
                "seed": i,
                "fold_test_end": "2024-01-25",
                "val_reward": float(i),
                "daily_returns": [float(x) for x in daily],
                "benchmark_daily_returns": [0.0] * t,
            }
        )
    return rows


def _strong_row(t=100, seed=1):
    rng = np.random.default_rng(seed)
    daily = 0.004 + rng.normal(0.0, 0.001, size=t)
    return {
        "candidate": "strong",
        "failed": False,
        "seed": None,
        "fold_test_end": "2024-01-25",
        "daily_returns": [float(x) for x in daily],
        "benchmark_daily_returns": [0.0] * t,
    }


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.96) == pytest.approx(0.9750021, abs=1e-6)
    assert norm_cdf(-1.96) == pytest.approx(0.0249979, abs=1e-6)


def test_norm_ppf_known_values_and_roundtrip():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert norm_ppf(0.01) == pytest.approx(-2.326348, abs=1e-5)
    for x in (-3.0, -1.0, 0.0, 0.5, 2.0):
        assert norm_ppf(norm_cdf(x)) == pytest.approx(x, abs=1e-7)


def test_norm_ppf_rejects_out_of_range():
    with pytest.raises(ValueError):
        norm_ppf(0.0)
    with pytest.raises(ValueError):
        norm_ppf(1.0)


def test_psr_hand_computed():
    # sr=1.0, sr0=0.5, t=250, skew=0, kurt=3:
    # num = 0.5*sqrt(249), den = sqrt(1.5) -> cdf ~ 1.0
    assert psr(1.0, 0.5, 250, 0.0, 3.0) == pytest.approx(1.0, abs=1e-9)
    # Equal to the benchmark: PSR collapses to 0.5.
    assert psr(1.0, 1.0, 250, 0.0, 3.0) == pytest.approx(0.5, abs=1e-12)


def test_expected_max_sr_increases_with_trials():
    e10 = expected_max_sr(10, 250, 0.0, 3.0)
    e100 = expected_max_sr(100, 250, 0.0, 3.0)
    assert e10 == pytest.approx(0.0998, abs=1e-3)
    assert e100 == pytest.approx(0.1603, abs=1e-3)
    assert e10 < e100


def test_deflated_sharpe_decreases_with_more_trials():
    dsr10 = deflated_sharpe(0.3, 10, 250, 0.0, 3.0)
    dsr100 = deflated_sharpe(0.3, 100, 250, 0.0, 3.0)
    assert 1.0 > dsr10 > dsr100


def test_dsr_noise_rows_insignificant_and_decoupled(monkeypatch):
    rows = _noise_rows()
    dsr = dsr_from_rows(rows)
    assert dsr is not None and dsr["n_trials"] == 30
    assert dsr["dsr"] < 0.95
    before = {k: v for k, v in dsr.items() if k != "best_candidate"}
    # val_reward is archived provenance: it must not move the correction.
    for row in rows:
        row["val_reward"] = 999.0 - row["val_reward"]
    after = {
        k: v for k, v in dsr_from_rows(rows).items() if k != "best_candidate"
    }
    assert after == before
    monkeypatch.setattr(evaluation, "REWARD_VERSION", "999")
    assert (
        {k: v for k, v in dsr_from_rows(rows).items() if k != "best_candidate"}
        == before
    )


def test_dsr_strong_trial_stands_out():
    rows = _noise_rows(n=30) + [_strong_row()]
    dsr = dsr_from_rows(rows)
    assert dsr["n_trials"] == 31
    assert dsr["best_candidate"] == "strong"
    assert dsr["sr_best"] > 3.0
    assert dsr["dsr"] > 0.9


def test_dsr_single_trial_uses_zero_benchmark():
    rows = [_strong_row()]
    dsr = dsr_from_rows(rows)
    assert dsr is not None and dsr["n_trials"] == 1
    assert dsr["dsr"] == pytest.approx(
        psr(dsr["sr_best"], 0.0, dsr["t_best"], dsr["skew_best"], dsr["kurt_best"]),
        rel=1e-12,
    )


def test_max_t_noise_rows_not_significant():
    mt = max_t_from_rows(_noise_rows(), n_perms=1000, seed=0)
    assert mt is not None
    assert mt["p_value"] > 0.05
    assert not mt["significant_95"]


def test_max_t_strong_trial_significant():
    mt = max_t_from_rows(
        _noise_rows(n=30) + [_strong_row()], n_perms=2000, seed=0
    )
    assert mt is not None
    assert mt["observed_max_t"] > 3.0
    assert mt["p_value"] < 0.05
    assert mt["significant_95"]


def test_selfcheck_rows_insignificant_on_synthetic(populated_db: DataConfig):
    loader = _loader(populated_db)
    proto = ProtocolConfig(
        folds=[
            FoldConfig("2024-01-10", "2024-01-25"),
            FoldConfig("2024-01-25", "2024-02-10"),
        ]
    )
    rows = selfcheck_rows(loader, proto, BacktestConfig())
    assert [r["candidate"] for r in rows] == ["selfcheck:noise"] * 2
    result = build_result(proto, "selfcheck", TierConfig(0, 0), rows)
    assert result["tier"] == "selfcheck" and result["steps"] == 0
    assert result["dsr"] is not None and result["dsr"]["dsr"] < 0.95
    assert result["max_t"] is not None and result["max_t"]["p_value"] > 0.05
    # Note: annualized Sharpe on this 3-stock / 11-day fixture explodes by
    # construction ((1+r)^(252/11)); the "OOS median ~ 0" acceptance belongs
    # to the real-data selfcheck, where test windows span full years.


def test_build_result_includes_extra_trial_rows():
    rows = _noise_rows(n=3)
    extra = _noise_rows(n=2, seed=7)
    result = build_result(
        ProtocolConfig(), "screening", TierConfig(50, 256), rows,
        extra_trial_rows=extra,
    )
    assert result["dsr"]["n_trials"] == 5
    assert result["dsr_extra_trials"] == 2
    assert result["n_candidates"] == 3


def test_load_trial_rows_reads_protocol_artifacts(tmp_path):
    p1 = tmp_path / "a.json"
    p1.write_text(
        json.dumps({"rows": [{"candidate": "trained", "seed": 42}]}),
        encoding="utf-8",
    )
    p2 = tmp_path / "b.json"
    p2.write_text(
        json.dumps({"rows": [{"candidate": "baseline:X"}]}), encoding="utf-8"
    )
    rows = load_trial_rows(f"{p1},{p2}")
    assert [r["candidate"] for r in rows] == ["trained", "baseline:X"]
    assert load_trial_rows(None) == []


def test_cli_confirmation_smoke(tmp_path, populated_db: DataConfig):
    """The confirmation tier records its own steps/batch in the artifact."""

    import yaml

    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "confirmation_result.json"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                    "data_dir": str(populated_db.data_dir),
                    "duckdb_path": str(populated_db.duckdb_path),
                    "parquet_dir": str(populated_db.parquet_dir),
                    "index_codes": ["000300.SH"],
                "model": {"max_formula_len": 6},
                "protocol": {
                    "folds": [
                        {"train_end": "2024-01-10", "test_end": "2024-01-25"}
                    ],
                    "seeds": [42],
                    "confirmation": {"steps": 1, "batch_size": 256},
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
            "confirmation",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["tier"] == "confirmation"
    assert payload["steps"] == 1 and payload["batch_size"] == 256


def test_cli_selfcheck_smoke(tmp_path, populated_db: DataConfig):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    out_path = tmp_path / "selfcheck_result.json"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                    "data_dir": str(populated_db.data_dir),
                    "duckdb_path": str(populated_db.duckdb_path),
                    "parquet_dir": str(populated_db.parquet_dir),
                    "index_codes": ["000300.SH"],
                "protocol": {
                    "folds": [
                        {"train_end": "2024-01-10", "test_end": "2024-01-25"},
                        {"train_end": "2024-01-25", "test_end": "2024-02-10"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    rc = evaluation.main(
        [
            "--config",
            str(cfg_path),
            "--selfcheck",
            "--max-t-perms",
            "200",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["tier"] == "selfcheck"
    assert payload["steps"] == 0 and payload["batch_size"] == 0
    assert all(r["candidate"] == "selfcheck:noise" for r in payload["rows"])
    assert payload["dsr"] is not None and payload["max_t"] is not None
