"""P34 R01 fixed-baseline diagnostic round 1 (executable projection).

Authority chain: ``docs/p34_r01_fixed_baseline_diagnostic_draft.md``
(APPROVED via user direct instruction 2026-09-16 「完成R01」, p30–p33
precedent).  The tests pin the contract's behavior surface:

* R1/R2 — the p8 §5.1 loader-frame degradation typing (the t16 staged gap
  closes: the data report consumes the loader's typed degradation sources
  and only a genuinely clean loader can Data Qualify);
* R3–R6 — the frozen baseline-arm machinery (additive engine
  observability, rank combos with the frozen construction rule, the cash
  control, the 0x/1x/2x cost ladder);
* R7 — the frozen adjudication rules (pure function, no OOS peeking);
* R8 — the mini-database end-to-end driver: real lifecycle chain
  (genesis → spec lock → data qualification → factor qualification) with
  REAL report builders on a tiny deterministic dual database, the report
  artifact registered inside the run store, and the negative-adjudication
  retirement path.

The end-to-end test passes ``require_clean_tree=False`` explicitly: it
runs in a tmp git-less environment (RunSpec clean-tree enforcement is
covered by the p8-03 tests); the production driver keeps the contract's
``True`` default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    ModelConfig,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_reward_config,
)
from ashare_data.db import AshareDB
from ashare_data.gates import GateCheck, GateResult

MINI_SESSIONS = 130
MINI_CODES = [f"{i:06d}.SZ" for i in range(1, 14)]
MINI_START = "2024-01-01"
MINI_TRAIN_END = "2024-04-30"
MINI_TEST_END = "2024-06-28"


def _mini_dates() -> list[str]:
    return (
        pd.bdate_range(MINI_START, periods=MINI_SESSIONS)
        .strftime("%Y%m%d")
        .tolist()
    )


def _mini_close(code_index: int, session_index: int) -> float:
    """Deterministic price ladder: 0 rises fast, 1 slow, 2 slower, the
    rest stay flat — a momentum signal ranks 0 > 1 > 2 > rest by
    construction alone."""

    if code_index == 0:
        return 10.0 + 0.5 * session_index
    if code_index == 1:
        return 20.0 + 0.1 * session_index
    if code_index == 2:
        return 15.0 + 0.03 * session_index
    return 10.0 + float(code_index)


def _mini_bars() -> pd.DataFrame:
    dates = _mini_dates()
    rows = []
    for i, code in enumerate(MINI_CODES):
        for t, date in enumerate(dates):
            close = _mini_close(i, t)
            pre_close = _mini_close(i, t - 1) if t else close * 0.99
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "pre_close": pre_close,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * close,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                    "close_raw": close,
                    "high_raw": close * 1.01,
                    "low_raw": close * 0.99,
                    "up_limit": round(pre_close * 1.10, 2),
                    "down_limit": round(pre_close * 0.90, 2),
                }
            )
    return pd.DataFrame(rows)


def build_mini_db(tmp_path: Path) -> DataConfig:
    """Populate the deterministic dual database with conftest helpers
    only (no second DDL, no second publication implementation)."""

    from conftest import (
        populate_v2_industry_tables,
        populate_v2_margin_table,
        populate_v2_price_tables,
        publish_manifest,
    )

    dates = _mini_dates()
    data_config = DataConfig(
        data_dir=tmp_path / "data",
        duckdb_path=tmp_path / "data" / "ashare.duckdb",
        parquet_dir=tmp_path / "data" / "parquet",
        start_date=MINI_START,
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )
    bars = _mini_bars()
    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    data_config.parquet_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {
                    "ts_code": code,
                    "name": code,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for code in MINI_CODES
            ],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in dates], data_config
        )
        db.upsert_constituents(
            [
                {
                    "index_code": "000300.SH",
                    "ts_code": code,
                    "in_date": f"2020010{1 + (i % 2)}",
                    "out_date": "99991231",
                }
                for i, code in enumerate(MINI_CODES)
            ],
            data_config,
        )
    v2_path = Path(data_config.v2_duckdb_path)
    if not v2_path.is_absolute():
        v2_path = data_config.duckdb_path.parent / v2_path
    populate_v2_price_tables(v2_path, dates, MINI_CODES, bars)
    populate_v2_industry_tables(
        v2_path,
        membership_rows=[
            {
                "ts_code": code,
                "index_code": "000300.SH",
                "level": 1,
                "in_date": f"2020010{1 + (i % 2)}",
                "out_date": None,
                "is_new": "1",
                "industry_name": f"IND{i % 3}",
                "available_date": f"2020010{1 + (i % 2)}",
            }
            for i, code in enumerate(MINI_CODES)
        ],
    )
    populate_v2_margin_table(
        v2_path,
        [
            {
                "ts_code": code,
                "trade_date": date,
                "rzye": 1_000_000.0,
                "available_date": date,
            }
            for code in MINI_CODES
            for date in dates
        ],
    )
    publish_manifest(data_config)
    return data_config


def write_mini_yaml(root: Path, data_config: DataConfig) -> Path:
    """The real mini YAML consumed through the production config makers."""

    cfg = {
        "data_dir": str(data_config.data_dir),
        "duckdb_path": str(data_config.duckdb_path),
        "parquet_dir": str(data_config.parquet_dir),
        "start_date": MINI_START,
        "end_date": "2024-12-31",
        "index_codes": ["000300.SH"],
        "min_listed_sessions": 1,
        "model": {"searcher": "random", "max_formula_len": 4},
        "reward": {
            "min_valid_ic_days": 4,
            "min_coverage": 0.2,
            "min_activity": 0.05,
        },
        "backtest": {
            "train_end_date": MINI_TRAIN_END,
            "end_date": MINI_TEST_END,
            "top_n": 3,
            "single_weight_cap": 1.0,
            "initial_capital": 100000.0,
            "rebalance_frequency": "every_10_days",
            "target_horizon": 10,
            "portfolio_method": "equal_weight",
            "buy_rank": 3,
            "sell_rank": 4,
            "min_trade_amount": 1000.0,
            "target_weight_change_threshold": 0.01,
            "commission_rate": 0.00025,
            "min_commission": 5.0,
            "stamp_tax_rate": 0.0005,
            "transfer_fee_rate": 0.00001,
            "slippage_rate": 0.0005,
        },
        "protocol": {
            "domain": "medium_cross_section",
            "tier_set": "A",
            "frequency": "every_10_days",
            "horizon": 10,
            "seeds": [42],
            "folds": [
                {"train_end": MINI_TRAIN_END, "test_end": MINI_TEST_END}
            ],
            "baseline_signals": ["MOMENTUM_20", "RSQ_60", "REVERSAL_60"],
        },
    }
    cfg_path = root / "mini_r01.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _loaded_mini_loader(data_config: DataConfig, model_config: ModelConfig):
    from ashare_model.data_loader import AshareDataLoader

    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    loader.load_data()
    return loader


@pytest.fixture()
def mini_env(tmp_path):
    data_config = build_mini_db(tmp_path)
    cfg_path = write_mini_yaml(tmp_path, data_config)
    return tmp_path, data_config, cfg_path


@pytest.fixture()
def gate_run_asof(monkeypatch):
    """Inject the G8 evaluation day through the p16-reserved ``today=``
    seam (the same pattern as the conftest ``production_gate_asof``
    fixture, applied to ``run`` — the driver is a production caller and
    never passes ``today`` itself)."""

    def _install(era: str):
        from ashare_data.gates import ProductionGateRunner

        real = ProductionGateRunner.run

        def patched(self, *, mode="dev", today=None):
            return real(self, mode=mode, today=today or era)

        monkeypatch.setattr(ProductionGateRunner, "run", patched)

    return _install


def _mini_spec(mini_env, require_clean_tree=False):
    from ashare_model.data_loader import AshareDataLoader
    from ashare_model.runspec import resolve_runtime_runspec

    tmp_path, data_config, cfg_path = mini_env
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_config = make_data_config(raw, tmp_path)
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    spec = resolve_runtime_runspec(
        dataset_id=loader.dataset_id,
        data_cutoff=loader.frozen_data_cutoff(),
        data_config=data_config,
        backtest_config=backtest_config,
        requested_budget=512,
        n_folds=1,
        research_domain="medium_cross_section",
        seeds=(42,),
        searcher=model_config.searcher,
        max_formula_len=int(model_config.max_formula_len),
        require_clean_tree=require_clean_tree,
    )
    return spec, data_config, model_config, backtest_config, raw


# --- R1: typed loader-frame status -------------------------------------------


def test_r1_loader_frame_status_clean_and_degraded(mini_env):
    from ashare_model.data_loader import AshareDataLoader

    _, data_config, _ = mini_env
    model_config = ModelConfig()
    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    loader.load_data()
    status = loader.loader_frame_status()
    assert status["status"] == "clean"
    assert status["reasons"] == []
    # The typed degradation sources the status derives from (unit-level
    # logic check: the method reports what the loader's own fields say).
    loader.price_degraded["000001.SZ"] = "missing adjusted basis"
    status = loader.loader_frame_status()
    assert status["status"] == "degraded"
    assert any("price" in reason for reason in status["reasons"])
    loader.price_degraded.clear()
    loader.industry_source_degraded = "stale membership"
    status = loader.loader_frame_status()
    assert status["status"] == "degraded"
    assert any("industry" in reason for reason in status["reasons"])


# --- R2: typed DataQualificationReport ---------------------------------------


def _formal_gate_result() -> GateResult:
    checks = tuple(
        GateCheck(f"G{i} mini check {i}", True, f"ok {i}") for i in range(1, 9)
    )
    return GateResult(
        mode="formal", ok=True, degraded=False, checks=checks,
        status=None, warnings=(),
    )


def test_r2_typed_data_report_clean_passes(mini_env):
    from ashare_model import lifecycle

    spec, *_ = _mini_spec(mini_env)
    report = lifecycle.build_data_qualification_report(
        spec,
        _formal_gate_result(),
        expected_trading_days=47,
        actual_trading_days=47,
        missing_trading_days=[],
        coverage=0.95,
        loader_frames={"status": "clean", "reasons": []},
    )
    lifecycle._validate_data_report(report, spec)  # must not raise


def test_r2_typed_data_report_degraded_refused(mini_env):
    from ashare_model import lifecycle

    spec, *_ = _mini_spec(mini_env)
    report = lifecycle.build_data_qualification_report(
        spec,
        _formal_gate_result(),
        expected_trading_days=47,
        actual_trading_days=47,
        missing_trading_days=[],
        coverage=0.95,
        loader_frames={
            "status": "degraded",
            "reasons": ["price_semantics_degraded=['000001.SZ']"],
        },
    )
    with pytest.raises(lifecycle.DataQualificationError, match="loader-frame"):
        lifecycle._validate_data_report(report, spec)


# --- R3: additive engine observability + cash control ------------------------


def test_r3_evaluate_signal_additive_fields_and_cash_control(mini_env):
    from ashare_model.eval_folds import epoch_slice, resolve_folds
    from ashare_model.eval_metrics import evaluate_signal
    from ashare_model.vocab import FEATURE_NAMES

    _, data_config, cfg_path = mini_env
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_config = make_data_config(raw, mini_env[0])
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    loader = _loaded_mini_loader(data_config, model_config)
    fold = resolve_folds(
        raw["protocol"]["folds"],
        loader.dates,
        frequency=raw["protocol"]["frequency"],
        horizon=raw["protocol"]["horizon"],
    )[0]
    factors, _, _, dates = epoch_slice(loader, fold)
    idx = FEATURE_NAMES.index("MOMENTUM_20")
    row = evaluate_signal(
        np.asarray(factors[idx], dtype=np.float64), loader, fold,
        backtest_config,
    )
    assert row["rebalance_due_count"] >= 1
    assert row["rebalance_count"] >= 1
    assert row["holdings_mean"] > 0
    assert row["holdings_max"] >= row["holdings_mean"]
    assert 0.0 < row["invested_ratio_mean"] <= 1.0
    assert row["cost_fraction_sum"] > 0
    # Cash control: an all-NaN signal holds nothing, trades nothing,
    # pays nothing — the frozen control that exposes fake low-cost wins.
    cash = np.full((len(loader.ts_codes), len(dates)), np.nan)
    cash_row = evaluate_signal(cash, loader, fold, backtest_config)
    assert cash_row["holdings_mean"] == 0.0
    assert cash_row["holdings_max"] == 0
    assert cash_row["invested_ratio_mean"] == 0.0
    assert cash_row["cost_fraction_sum"] == 0.0
    assert cash_row["total_return"] == 0.0
    assert cash_row["rebalance_count"] == 0
    assert cash_row["n_ic_dates"] == 0


# --- R4: frozen combo construction -------------------------------------------


def test_r4_rank_normalize_and_combo_hand_computed():
    from ashare_model.r01_baseline import combo_signal, rank_normalize_signal

    mask = np.array(
        [[True, True, True], [True, True, True],
         [True, True, False], [True, True, True]],
        dtype=bool,
    )
    a = np.array(
        [[1.0, np.nan, 5.0], [3.0, 2.0, 1.0],
         [2.0, 1.0, np.nan], [np.nan, 3.0, 4.0]]
    )
    # Date 0: finite eligible [1, 3, 2] -> ranks [1, 3, 2]; NaN stays NaN.
    # Date 1: finite eligible [2, 1, 3] -> ranks [2, 1, 3]; row-0 NaN.
    # Date 2: finite AND eligible rows 0/1/3 -> [5, 1, 4] -> [3, 1, NaN, 2]
    # (row 2 is masked out, not ranked).
    ranks = rank_normalize_signal(a, mask)
    assert np.isclose(ranks[0, 0], 1.0)
    assert np.isclose(ranks[1, 0], 3.0)
    assert np.isclose(ranks[2, 0], 2.0)
    assert np.isnan(ranks[3, 0])
    assert np.isnan(ranks[0, 1])
    assert np.isclose(ranks[1, 1], 2.0)
    assert np.isclose(ranks[2, 1], 1.0)
    assert np.isclose(ranks[3, 1], 3.0)
    assert np.isclose(ranks[0, 2], 3.0)
    assert np.isclose(ranks[1, 2], 1.0)
    assert np.isnan(ranks[2, 2])
    assert np.isclose(ranks[3, 2], 2.0)
    # All-missing eligible column -> NaN everywhere.
    empty = np.full((4, 1), np.nan)
    out = rank_normalize_signal(empty, mask[:, :1])
    assert np.isnan(out).all()
    # Combo: signed rank means; missing members average the available ones.
    b = np.array(
        [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0],
         [4.0, 3.0, np.nan], [3.0, 4.0, 3.0]]
    )
    combo = combo_signal([(1.0, a), (-1.0, b)], mask)
    # Date 0: member ranks a=[1,3,2,NaN], -b=[3,4,1,2] ->
    # means [2.0, 3.5, 1.5, 2.0] (row 3 averages the one available member).
    assert np.isclose(combo[0, 0], 2.0)
    assert np.isclose(combo[1, 0], 3.5)
    assert np.isclose(combo[2, 0], 1.5)
    assert np.isclose(combo[3, 0], 2.0)
    # Date 2: row 2 has no available member -> NaN.
    assert np.isnan(combo[2, 2])


# --- R5: json-safe rows -------------------------------------------------------


def test_r5_rows_to_jsonable_nan_becomes_null():
    from ashare_model.r01_baseline import rows_to_jsonable

    rows = [
        {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("nan")}}
    ]
    payload = rows_to_jsonable(rows)
    text = json.dumps(payload, allow_nan=False)  # raises if NaN survives
    assert "NaN" not in text and "Infinity" not in text
    assert payload[0]["a"] is None and payload[0]["b"][1] is None


# --- R6: the 0x/1x/2x cost ladder ---------------------------------------------


def test_r6_cost_variants_definitions_and_ladder(mini_env):
    from dataclasses import replace as dc_replace

    from ashare_model.eval_folds import epoch_slice, resolve_folds
    from ashare_model.eval_metrics import evaluate_signal
    from ashare_model.r01_baseline import cost_variant
    from ashare_model.vocab import FEATURE_NAMES

    bt = BacktestConfig(
        commission_rate=0.00025, min_commission=5.0,
        stamp_tax_rate=0.0005, transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    zero = cost_variant(bt, "0x")
    assert (
        zero.commission_rate == 0.0 and zero.min_commission == 0.0
        and zero.stamp_tax_rate == 0.0 and zero.transfer_fee_rate == 0.0
        and zero.slippage_rate == 0.0
    )
    one = cost_variant(bt, "1x")
    assert one == bt
    two = cost_variant(bt, "2x")
    assert two.commission_rate == 2 * bt.commission_rate
    assert two.stamp_tax_rate == 2 * bt.stamp_tax_rate
    assert two.transfer_fee_rate == 2 * bt.transfer_fee_rate
    assert two.slippage_rate == 2 * bt.slippage_rate
    assert two.min_commission == bt.min_commission
    with pytest.raises(ValueError):
        cost_variant(bt, "3x")
    assert cost_variant(bt, "1x") is not None
    # The frozen definitions never mutate the caller's config.
    assert bt == BacktestConfig(
        commission_rate=0.00025, min_commission=5.0,
        stamp_tax_rate=0.0005, transfer_fee_rate=0.00001,
        slippage_rate=0.0005,
    )
    assert dc_replace(bt) == bt

    _, data_config, cfg_path = mini_env
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_config = make_data_config(raw, mini_env[0])
    model_config = make_model_config(raw)
    backtest_config = make_backtest_config(raw)
    loader = _loaded_mini_loader(data_config, model_config)
    fold = resolve_folds(
        raw["protocol"]["folds"], loader.dates,
        frequency=raw["protocol"]["frequency"],
        horizon=raw["protocol"]["horizon"],
    )[0]
    factors, _, _, _ = epoch_slice(loader, fold)
    signal = np.asarray(
        factors[FEATURE_NAMES.index("MOMENTUM_20")], dtype=np.float64
    )
    sums = {}
    for level in ("0x", "1x", "2x"):
        row = evaluate_signal(signal, loader, fold, cost_variant(backtest_config, level))
        sums[level] = row["cost_fraction_sum"]
    assert sums["0x"] == 0.0
    assert sums["2x"] > sums["1x"] > 0.0


# --- R7: frozen adjudication rules --------------------------------------------


def _arm(**over):
    base = {
        "candidate": "baseline:X",
        "failed_folds": 0,
        "unreliable": False,
        "mean_net_excess_1x": 0.02,
        "active_ir_1x": 0.8,
        "positive_folds_1x": 4,
        "mean_holdings_1x": 18.0,
        "invested_ratio_mean_1x": 0.95,
        "mean_net_excess_2x": 0.015,
    }
    base.update(over)
    return base


def test_r7_adjudication_paths():
    from ashare_model.r01_baseline import R01_ADJUDICATION_RULES, decide_adjudication

    # Positive path: a qualifying arm parks the run for round 2.
    verdict = decide_adjudication([_arm()], R01_ADJUDICATION_RULES)
    assert verdict["worth_search"] is True
    assert verdict["qualified_arms"] == ["baseline:X"]
    assert verdict["flags"]["baseline:X"] == []

    # Negative path: every arm below water — legal negative result.
    verdict = decide_adjudication(
        [_arm(mean_net_excess_1x=-0.01, active_ir_1x=-0.3,
              positive_folds_1x=1, mean_net_excess_2x=-0.02)],
        R01_ADJUDICATION_RULES,
    )
    assert verdict["worth_search"] is False
    assert verdict["qualified_arms"] == []
    assert "no_qualified_arm" in verdict["bottleneck"]

    # Low-activity arms are construction artifacts, not value evidence.
    verdict = decide_adjudication(
        [_arm(invested_ratio_mean_1x=0.2)], R01_ADJUDICATION_RULES
    )
    assert verdict["worth_search"] is False
    assert "low_activity" in verdict["flags"]["baseline:X"]

    # Single-year dependence and cost fragility are recorded as flags.
    verdict = decide_adjudication(
        [_arm(positive_folds_1x=2, mean_net_excess_2x=-0.01)],
        R01_ADJUDICATION_RULES,
    )
    assert verdict["worth_search"] is False
    flags = verdict["flags"]["baseline:X"]
    assert "single_year_dependence" in flags
    assert "cost_fragile" in flags

    # Thin books cannot vote: fewer than min_avg_holdings disqualifies.
    verdict = decide_adjudication(
        [_arm(mean_holdings_1x=3.0)], R01_ADJUDICATION_RULES
    )
    assert verdict["worth_search"] is False
    assert "thin_book" in verdict["flags"]["baseline:X"]


# --- R8: the mini end-to-end driver -------------------------------------------


def test_r8_end_to_end_driver_positive_parks_at_factor_qualified(
    mini_env, gate_run_asof, monkeypatch
):
    import ashare_model.r01_baseline as r01

    tmp_path, data_config, cfg_path = mini_env
    dates = _mini_dates()
    gate_run_asof(dates[-1])
    summary = r01.run_r01_baseline(
        cfg_path,
        contract_sha256="a" * 64,
        out_root=tmp_path / "r01_out",
        require_clean_tree=False,
    )
    # The momentum ladder is the oracle: the momentum-family arms beat the
    # equal-weight benchmark net of costs in this scenario, so the frozen
    # rules keep the run parked (non-terminal) at FACTOR_SET_QUALIFIED.
    from ashare_model.lifecycle import replay_run
    from ashare_model.run_store import RunStore

    store = RunStore(data_config.data_dir)
    state = replay_run(store.root, summary["spec_id"], summary["run_id"])
    assert summary["adjudication"]["worth_search"] is True
    assert state == "FACTOR_SET_QUALIFIED"
    # The report artifact lives in the run store and the measurement
    # directory, json-safe, with full provenance.
    report_path = tmp_path / "r01_out" / "r01_baseline_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["r01_report_schema_version"] == 1
    assert payload["provenance"]["contract_sha256"] == "a" * 64
    assert payload["provenance"]["dataset_id"] == summary["dataset_id"]
    assert payload["adjudication"] == summary["adjudication"]
    arms = {row["candidate"] for row in payload["rows"]}
    assert {
        "benchmark:equal_weight",
        "baseline:MOMENTUM_20", "baseline:RSQ_60", "baseline:REVERSAL_60",
        "combo:all3", "combo:momentum_reversal", "control:cash",
    } <= arms
    levels = {row["cost_level"] for row in payload["rows"]}
    assert levels == {"0x", "1x", "2x"}
    # Every cost-bearing arm discloses every fold (full-arm reporting).
    for arm in ("baseline:MOMENTUM_20", "combo:all3", "control:cash"):
        n_rows = sum(
            1 for row in payload["rows"]
            if row["candidate"] == arm and row["cost_level"] == "1x"
        )
        assert n_rows == 1


def test_r8_end_to_end_driver_negative_retires(
    mini_env, gate_run_asof, monkeypatch
):
    """The negative path: with the adjudication forced negative the run
    terminates RETIRED with the report hash in the reason (the machine
    discriminator: administrative stop, not a gate failure)."""

    import ashare_model.r01_baseline as r01
    from ashare_model.lifecycle import replay_run
    from ashare_model.run_store import RunStore

    tmp_path, data_config, cfg_path = mini_env
    dates = _mini_dates()
    gate_run_asof(dates[-1])
    real_decide = r01.decide_adjudication

    def forced_negative(arm_summaries, rules):
        verdict = real_decide(arm_summaries, rules)
        verdict["worth_search"] = False
        verdict["qualified_arms"] = []
        verdict["bottleneck"] = "forced_negative_for_test"
        return verdict

    monkeypatch.setattr(r01, "decide_adjudication", forced_negative)
    summary = r01.run_r01_baseline(
        cfg_path,
        contract_sha256="b" * 64,
        out_root=tmp_path / "r01_out_neg",
        require_clean_tree=False,
    )
    store = RunStore(data_config.data_dir)
    state = replay_run(store.root, summary["spec_id"], summary["run_id"])
    assert summary["adjudication"]["worth_search"] is False
    assert state == "RETIRED"
