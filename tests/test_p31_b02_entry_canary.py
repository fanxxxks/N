"""P31 B02: the real-entry canary on a small deterministic database.

Executable projection of ``docs/p31_b02_entry_canary_ci_draft.md`` (§3
behavior contract).  Authority chain: user direct instruction 2026-09-15
「完成B02」+ captain countersign (p29/p30 precedent).

The canary walks the FULL formal loop on a tiny deterministic dual
database — real YAML config parse → composite identity confirm → RunSpec
freeze → lifecycle qualification chain → formal train entry (minimal
random search) → strategy artifact → formal backtest entry → backtest
artifact → webapi read of the same run — and adds the two entry-level
behaviors no existing test pins end to end:

* identity invalidation blocks a formal entry BEFORE the loader builds
  frames and before the run gains any business artifact (A02 p26
  consumed at the entry boundary, not only at the protocol layer);
* a future-only perturbation of the bar table leaves every
  past-session feature value bit-identical (the strongest
  time-causality form, asserted on the real loader factor tensor).

The hand-computed execution check (C7) derives its expected dates,
weights and money amounts from the SCENARIO construction and the
documented fee formula (``ashare_execution.ExecutionCostModel``:
buy = max(rate*notional, min) + transfer + slippage; sell adds stamp
tax; rebalance fraction = total / capital) — never from the engine's
own output.

Guard-type tests (AGENTS §2.2.3): the canary pins existing behavior at
the entry boundary; its RED evidence is the first real run recorded in
the measurement log, not an expected failure.
"""

from __future__ import annotations

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
    make_sim_config,
)
from ashare_data.db import AshareDB

CANARY_SESSIONS = 40
# Real-shaped SZ main-board codes: the universe contract's stock-code
# validator (``is_valid_a_share_code``) rejects non-stock symbols, so a
# synthetic universe must use board-plausible prefixes.
CANARY_CODES = [f"{i:06d}.SZ" for i in range(1, 13)]
# Session axis shared with the ``production_gate_asof`` fixture era:
# both are ``pd.bdate_range("2024-01-01", periods=40)``, so the gate's
# injected evaluation day equals the canary's last session by
# construction (no clock, no network).
CANARY_START = "2024-01-01"


def _canary_dates() -> list[str]:
    return (
        pd.bdate_range(CANARY_START, periods=CANARY_SESSIONS)
        .strftime("%Y%m%d")
        .tolist()
    )


def _canary_close(code_index: int, session_index: int) -> float:
    """The deterministic scenario prices (the hand-check oracle).

    Stock 0 rises 0.1/session from 10, stock 1 rises 0.05/session from
    20, the rest stay flat at 10 + i.  On every session the 1-session
    return of stock 0 strictly exceeds stock 1's, which strictly exceeds
    every flat stock's 0.0 — so a momentum signal ranks 0 > 1 > rest
    deterministically on the constructed values alone.
    """

    if code_index == 0:
        return 10.0 + 0.1 * session_index
    if code_index == 1:
        return 20.0 + 0.05 * session_index
    return 10.0 + float(code_index)


def _canary_bars() -> pd.DataFrame:
    dates = _canary_dates()
    rows = []
    for i, code in enumerate(CANARY_CODES):
        for t, date in enumerate(dates):
            close = _canary_close(i, t)
            pre_close = _canary_close(i, t - 1) if t else close * 0.99
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


def build_canary_db(tmp_path: Path) -> DataConfig:
    """Populate the deterministic dual database (conftest helpers only —
    no second DDL, no second publication implementation) and publish the
    composite generation (C0).

    Scenario features beyond the price ladder: stock 5 carries a
    constant adj_factor of 2.0 (adjusted basis ≠ raw); stock 7 is
    suspended on session 20; stock 8 is limit-up blocked for buys on
    session 22; the v2 income statement row for stock 2 becomes
    available mid-window (announcement delay: end_date 2023-12-31 →
    available 2024-01-15).  None of these touch stocks 0/1, so the
    hand-checked top-2 book is unaffected by construction.
    """

    from conftest import (
        populate_v2_industry_tables,
        populate_v2_price_tables,
        populate_v2_statement_tables,
        publish_manifest,
    )

    dates = _canary_dates()
    data_config = DataConfig(
        data_dir=tmp_path / "data",
        duckdb_path=tmp_path / "data" / "ashare.duckdb",
        parquet_dir=tmp_path / "data" / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )
    bars = _canary_bars()
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
                for code in CANARY_CODES
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
                    # Staggered in_dates keep the interval table PIT-shaped
                    # (one identical in_date across all rows is exactly the
                    # "current snapshot stretched over history" shape the
                    # universe contract refuses).
                    "in_date": f"2020010{1 + (i % 2)}",
                    "out_date": "99991231",
                }
                for i, code in enumerate(CANARY_CODES)
            ],
            data_config,
        )
    v2_path = Path(data_config.v2_duckdb_path)
    if not v2_path.is_absolute():
        v2_path = data_config.duckdb_path.parent / v2_path
    populate_v2_price_tables(
        v2_path,
        dates,
        CANARY_CODES,
        bars,
        adj_factor={(CANARY_CODES[5], date): 2.0 for date in dates},
        limit_overrides={
            # Buy-block cell: open == close >= up_limit on session 22.
            (CANARY_CODES[8], dates[22]): (
                _canary_close(8, 22) * 1.000001,
                _canary_close(8, 22) * 0.5,
            ),
        },
        suspensions=[
            {
                "ts_code": CANARY_CODES[7],
                "trade_date": dates[20],
                "suspend_type": "S",
            }
        ],
    )
    populate_v2_industry_tables(v2_path)
    populate_v2_statement_tables(
        v2_path,
        income_rows=[
            {
                "ts_code": CANARY_CODES[2],
                "end_date": "20231231",
                "ann_date": "20240115",
                "f_ann_date": "20240115",
                "available_date": "20240115",
                "report_type": "1",
                "comp_type": "1",
                "update_flag": "0",
                "end_type": "1231",
                "total_revenue": 100.0,
                "revenue": 100.0,
                "oper_cost": 60.0,
                "total_profit": 40.0,
                "n_income": 30.0,
                "n_income_attr_p": 30.0,
                "ingest_id": "canary_announcement_delay",
            }
        ],
    )
    publish_manifest(data_config)
    return data_config


def write_canary_yaml(root: Path, data_config: DataConfig) -> Path:
    """The real canary YAML: one file for the formal entries AND the
    webapi config baseline (``<root>/config/ashare_config.yaml``), so
    the API reads the same resolved paths the entries consume (C1/C5).

    ``reward`` overrides are canary-scale floors for a 12-stock universe
    (make_reward_config passes every RewardConfig field through YAML);
    they tune THIS run only — no production default changes.  The sim
    block mirrors the backtest deployment sizes
    (``validate_execution_config`` refuses drift).
    """

    cfg = {
        "data_dir": str(data_config.data_dir),
        "duckdb_path": str(data_config.duckdb_path),
        "parquet_dir": str(data_config.parquet_dir),
        "start_date": "2024-01-01",
        "end_date": "2026-12-31",
        "index_codes": ["000300.SH"],
        "min_listed_sessions": 1,
        "model": {
            "searcher": "random",
            "max_formula_len": 4,
            "train_steps": 2,
            "batch_size": 8,
        },
        "reward": {
            "min_valid_ic_days": 4,
            "min_coverage": 0.2,
            "min_activity": 0.05,
        },
        "backtest": {
            "train_end_date": "2024-02-06",
            "end_date": "2024-12-31",
            "top_n": 2,
            "single_weight_cap": 1.0,
            "initial_capital": 100000.0,
            "rebalance_frequency": "daily",
            "target_horizon": 1,
        },
        "sim": {
            "initial_capital": 100000.0,
            "max_positions": 2,
            "single_weight_cap": 1.0,
            "state_path": str(
                data_config.data_dir / "sim_portfolio_state.json"
            ),
            "orders_dir": str(data_config.data_dir / "sim_orders"),
            "trades_dir": str(data_config.data_dir / "sim_trades"),
        },
        "protocol": {
            "folds": [{"train_end": "2024-02-06", "test_end": "2024-02-20"}],
            "seeds": [42],
            "random_samples": 0,
        },
    }
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / "ashare_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _canary_configs(cfg_path: Path, root: Path):
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return (
        make_data_config(raw, root),
        make_model_config(raw),
        make_backtest_config(raw),
        make_sim_config(raw, root),
    )


def _freeze_spec(data_config, model_config, backtest_config):
    """Freeze the canary RunSpec EXACTLY the way
    ``train_entry.establish_formal_run_context`` does (same derivation,
    same constants) — the formal entry re-derives it and
    ``RunContext.attach`` refuses any mismatch."""

    from ashare_model.data_loader import AshareDataLoader
    from ashare_model.runspec import resolve_runtime_runspec

    loader = AshareDataLoader(data_config, model_config)
    loader.confirm_input_snapshot()
    spec = resolve_runtime_runspec(
        dataset_id=loader.dataset_id,
        data_cutoff=loader.frozen_data_cutoff(),
        data_config=data_config,
        backtest_config=backtest_config,
        requested_budget=max(1, 2 * 8),
        n_folds=1,
        research_domain="unified",
        seeds=(42,),
        searcher="random",
        max_formula_len=int(model_config.max_formula_len),
        require_clean_tree=False,
    )
    return spec


def _qualify_to_admission(store, spec):
    """Drive the preregistered run through the real lifecycle chain to
    SEARCH_PLAN_ADMITTED (C2): genesis → spec lock → data qualification →
    factor qualification → search-plan admission, with the writer's own
    evidence validation on every transition."""

    from ashare_model import lifecycle
    from ashare_model.identity import content_hash
    from test_run_store import LIFECYCLE_IDEA, _clean_data_report

    handle = store.open_run(spec)
    writer = lifecycle.LifecycleWriter(handle)
    writer.record_idea(LIFECYCLE_IDEA)
    writer.lock_spec()
    writer.qualify_data(_clean_data_report(spec))
    factor_report = {
        "report_schema_version": 1,
        "spec_id": spec.spec_id,
        "research_domain": spec.research_domain,
        "frequency": spec.frequency,
        "horizon": int(spec.horizon),
        "feature_count": 1,
        "features": [
            {
                "name": "RET_1",
                "family": "price",
                "pit_level": "EVENT",
                "data_tier": "A",
                "promotion_allowed": True,
                "domain_id": "unified",
                "domain_consistent": True,
                "in_domain_execution": True,
                "coverage": 0.95,
            }
        ],
        "domain_execution_legal": True,
        "weakest_data_tier": "A",
        "promotion_input_eligible": True,
        "data_eligibility": {"ok": True, "reasons": [], "checks": {}},
        "baseline_evidence": {
            "completed": True,
            "result_hash": content_hash("baseline", {"seed": 7}),
        },
        "ablation_evidence": {
            "completed": True,
            "result_hash": content_hash("ablation", {"seed": 7}),
        },
        "completed": True,
    }
    writer.qualify_factor_set(factor_report)
    writer.admit_search_plan(lifecycle.build_search_admission_evidence(spec))
    run_id = handle.run_id
    handle.close()
    return run_id


def _business_artifacts(store, spec, run_id) -> list[str]:
    handle = store.attach_run(spec, run_id)
    try:
        index = handle.load_index()
    finally:
        handle.close()
    return [
        a["artifact_type"]
        for a in index["artifacts"]
        if a["artifact_type"]
        in {"strategy", "protocol", "backtest", "paper_state"}
    ]


def _train_argv(cfg_path: Path, spec, run_id: str) -> list[str]:
    return [
        "train",
        "--config", str(cfg_path),
        "--min-eligible", "3",
        "--steps", "2",
        "--batch-size", "8",
        "--device", "cpu",
        "--spec-id", spec.spec_id,
        "--run-id", run_id,
    ]


# --- C1–C5: the full closed loop through the real formal entries -----------


def test_canary_loop_config_spec_train_backtest_api(
    tmp_path, canary_env, production_gate_asof, monkeypatch
):
    """p31 §3.2: real YAML parse → composite identity → RunSpec freeze →
    lifecycle chain → formal train entry (minimal random search) →
    strategy artifact → formal backtest entry → backtest artifact →
    webapi list/detail read the same run (C1–C5)."""

    from ashare_model import backtest as backtest_module
    from ashare_model import train as train_module
    from ashare_model.run_store import RunStore
    from webapi import service

    tmp, data_config, cfg_path = canary_env
    _, model_config, backtest_config, _ = _canary_configs(cfg_path, tmp)
    spec = _freeze_spec(data_config, model_config, backtest_config)
    store = RunStore(data_config.data_dir)
    run_id = _qualify_to_admission(store, spec)
    assert _business_artifacts(store, spec, run_id) == []

    # C3: the formal train entry re-derives the spec, attaches the run,
    # passes assert_search_ready and runs the minimal random search on
    # the canary data (CPU, seeded → deterministic).
    monkeypatch.setattr(sys, "argv", _train_argv(cfg_path, spec, run_id))
    train_module.main()

    business = _business_artifacts(store, spec, run_id)
    assert "strategy" in business, (
        "the formal train entry produced no strategy artifact"
    )
    strategy_count = business.count("strategy")

    # C4: the formal backtest entry consumes the registered strategy
    # (registered-artifact check + RunContext + dataset consistency) and
    # registers the backtest result into the SAME run.
    out_path = tmp / "backtest_result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest",
            "--config", str(cfg_path),
            "--min-eligible", "3",
            "--output", str(out_path),
        ],
    )
    backtest_module.main()
    assert out_path.exists()

    business = _business_artifacts(store, spec, run_id)
    assert business.count("backtest") == 1
    assert business.count("strategy") == strategy_count, (
        "the backtest entry must not mint additional strategies"
    )

    # C5: the webapi reads the SAME run root the entries wrote to.  The
    # service resolves the RunStore root from the config baseline
    # (F05/A01: one root for readers and writers).
    monkeypatch.setattr(service, "ROOT", tmp)
    listed = service.list_runs()
    assert listed["runs_dir"] == str(data_config.data_dir / "runs")
    entry = next((r for r in listed["runs"] if r["run_id"] == run_id), None)
    assert entry is not None
    assert entry["spec_id"] == spec.spec_id
    assert entry["state"] == "SEARCH_PLAN_ADMITTED"

    detail = service.get_run_detail(None, spec.spec_id, run_id)
    artifact_types = {a["artifact_type"] for a in detail["artifacts"]}
    assert {"strategy", "backtest"} <= artifact_types
    states = [e["to_state"] for e in detail["events"]]
    assert states[-1] == "SEARCH_PLAN_ADMITTED"


@pytest.fixture
def canary_env(tmp_path):
    """The canary environment: deterministic dual DB + the real YAML
    config at the webapi baseline location.  Only tests that do NOT
    mutate the database share this fixture shape; the identity and
    perturbation tests build their own instance (they mutate it)."""

    data_config = build_canary_db(tmp_path)
    cfg_path = write_canary_yaml(tmp_path, data_config)
    return tmp_path, data_config, cfg_path


# --- D1: the no-best-yet search history is artifact-writable ---------------


def test_search_history_maps_no_best_yet_to_none():
    """D1 (p31 defect fix): the search contract's best-so-far curve is
    legal with ``-inf`` while no candidate is eligible yet, but the
    typed strategy artifact refuses non-finite floats — before the fix,
    every formal search whose first evaluation was ineligible could not
    persist its strategy artifact.  The trainer history is telemetry:
    "no best yet" must be ``None``; finite values pass through."""

    from ashare_model.train_loop import history_reward_value

    assert history_reward_value(float("-inf")) is None
    assert history_reward_value(float("inf")) is None
    assert history_reward_value(float("nan")) is None
    assert history_reward_value(-20.0) == -20.0
    assert history_reward_value(10.0) == 10.0
    assert history_reward_value(0) == 0.0

    from ashare_model.train_artifacts import _json_telemetry_value

    curve = {"best_so_far": [[1, float("-inf")], [2, 10.0]], "x": 1}
    safe = _json_telemetry_value(curve)
    assert safe["best_so_far"][0][1] is None
    assert safe["best_so_far"][1][1] == 10.0
    assert safe["x"] == 1
    # Identity-ish leaves are untouched; the canonical writer stays the
    # fail-closed backstop for non-JSON types.
    payload = {
        "spec_id": "s", "direction": 1, "flag": True, "none": None,
        "nested": {"inf": float("inf"), "list": [1.0, float("nan")]},
    }
    assert _json_telemetry_value(payload) == {
        "spec_id": "s", "direction": 1, "flag": True, "none": None,
        "nested": {"inf": None, "list": [1.0, None]},
    }


# --- C6: identity invalidation blocks the formal entry ---------------------


def test_canary_identity_invalidation_blocks_entry_before_business_write(
    tmp_path, production_gate_asof, monkeypatch
):
    """C6/p31 §3.3: after the canary DB is published, mutating v2 content
    (a value change preserving row counts) refuses the formal train
    entry at the identity confirm — before ``load_data`` builds any
    frame and before the run gains any business artifact (A02 p26 I2/I7
    consumed at the entry boundary).  The refusal surfaces as
    ``DatasetIdMismatch`` (the loader's own contract type)."""

    from ashare_data.manifest import DatasetIdMismatch
    from ashare_model import train as train_module
    from ashare_model.data_loader import AshareDataLoader
    from ashare_model.run_store import RunStore

    data_config = build_canary_db(tmp_path)
    cfg_path = write_canary_yaml(tmp_path, data_config)
    _, model_config, backtest_config, _ = _canary_configs(cfg_path, tmp_path)
    spec = _freeze_spec(data_config, model_config, backtest_config)
    store = RunStore(data_config.data_dir)
    run_id = _qualify_to_admission(store, spec)
    before = _business_artifacts(store, spec, run_id)
    assert before == []

    dates = _canary_dates()
    from ashare_data.config import resolve_v2_duckdb_path

    with AshareDB(resolve_v2_duckdb_path(data_config)) as db:
        db.execute(
            "UPDATE daily_bar_raw SET close = close * 3, amount = amount * 3 "
            f"WHERE trade_date = '{dates[-1]}'"
        )

    def _forbidden(self, *a, **k):
        raise AssertionError("load_data ran after the identity refusal")

    monkeypatch.setattr(AshareDataLoader, "load_data", _forbidden)
    monkeypatch.setattr(sys, "argv", _train_argv(cfg_path, spec, run_id))
    with pytest.raises(DatasetIdMismatch):
        train_module.main()

    assert _business_artifacts(store, spec, run_id) == [], (
        "a refused entry must not add business artifacts"
    )


# --- C7: hand-computed execution on the real loader cache ------------------


def test_canary_hand_computed_execution_on_real_loader(tmp_path):
    """C7/p31 §3.3: the engine runs on the REAL loader cache of the
    canary DB with a hand-built signal; expected dates, weights and the
    first two equity points come from the scenario construction and the
    documented fee formula — never from the engine output.

    Fee math (BacktestConfig defaults, capital 100000): first period
    buys 0.5+0.5 → two 50000-yuan buy orders; per order commission
    max(0.00025*50000, 5) = 12.5, transfer 0.00001*50000 = 0.5,
    slippage 0.0005*50000 = 25 → 38.0 per order, 76.0 total → cost
    fraction 76/100000 = 0.00076.  Second period has identical targets
    (no weight drift in the weight-compounding book) → zero buy/sell
    weights → zero cost, net = gross.
    """

    from ashare_model.backtest import AshareBacktestEngine
    from ashare_model.data_loader import AshareDataLoader

    data_config = build_canary_db(tmp_path)
    dates = _canary_dates()
    loader = AshareDataLoader(data_config, ModelConfig(max_formula_len=4))
    loader.confirm_input_snapshot()
    loader.load_data()

    backtest_config = BacktestConfig(
        train_end_date="2024-02-06",
        end_date="2024-12-31",
        top_n=2,
        single_weight_cap=1.0,
        rebalance_frequency="daily",
        target_horizon=1,
    )
    signal = np.zeros((len(CANARY_CODES), len(dates)), dtype=np.float64)
    signal[0, :] = 1.0
    signal[1, :] = 0.9
    result = AshareBacktestEngine(backtest_config).run(
        signal,
        {k: v.numpy() for k, v in loader.raw_data_cache.items()},
        loader.ts_codes,
        loader.dates,
        universe_mask=loader.universe_mask,
    )

    # Dates: signal t → entry t+1 → exit t+2 on the SESSION axis; t=4 is
    # Friday 20240105, its entry is the next session Monday 20240108 —
    # consecutive sessions across the weekend gap.
    assert result.positions[0]["signal_date"] == dates[0]
    assert result.positions[0]["entry_date"] == dates[1]
    assert result.positions[0]["exit_date"] == dates[2]
    assert result.positions[4]["entry_date"] == dates[5]
    assert dates[4] == "20240105" and dates[5] == "20240108"

    # Weights: equal-weight top-2 of the hand-ranked signal; the book is
    # stable, so every period holds the same two codes at 0.5/0.5.
    assert len(result.positions) == CANARY_SESSIONS - 2
    for position in result.positions:
        assert set(position["ts_codes"]) == {
            CANARY_CODES[0],
            CANARY_CODES[1],
        }
        assert position["weights"] == [0.5, 0.5]
    assert result.turnover[0] == pytest.approx(1.0)
    assert result.turnover[1] == pytest.approx(0.0)

    # Money: gross = mean of the two constructed open-to-open returns;
    # net = gross − documented buy-fee fraction (76/100000).
    def open_of(i: int, t: int) -> float:
        return _canary_close(i, t)

    gross1 = 0.5 * (open_of(0, 2) / open_of(0, 1) - 1) + 0.5 * (
        open_of(1, 2) / open_of(1, 1) - 1
    )
    net1 = gross1 - (2 * 38.0) / 100000.0
    assert result.equity_curve[0] == pytest.approx(1.0)
    assert result.equity_curve[1] == pytest.approx(1.0 + net1, rel=1e-9)
    gross2 = 0.5 * (open_of(0, 3) / open_of(0, 2) - 1) + 0.5 * (
        open_of(1, 3) / open_of(1, 2) - 1
    )
    assert result.equity_curve[2] == pytest.approx(
        (1.0 + net1) * (1.0 + gross2), rel=1e-9
    )
    assert len(result.equity_curve) == len(result.positions) + 1


# --- C8: future perturbation leaves the past bit-identical -----------------


def test_canary_future_perturbation_leaves_past_features_unchanged(tmp_path):
    """C8/p31 §3.3: tripling the OHLC of the LAST ten sessions changes no
    feature value at any earlier session (real loader → factor tensor).
    The post-perturbation load declares the explicit legacy audit mode
    (``dataset_identity_mode='query'``): the perturbed database no
    longer matches the published composite generation, and this probe
    is an engineering read, not a formal run — the formal refusal
    itself is C6's subject."""

    from ashare_data.config import resolve_v2_duckdb_path
    from ashare_model.data_loader import AshareDataLoader

    data_config = build_canary_db(tmp_path)
    loader = AshareDataLoader(data_config, ModelConfig(max_formula_len=4))
    loader.confirm_input_snapshot()
    loader.load_data()
    before = loader.factor_tensor.detach().cpu().numpy().copy()
    n_dates = len(loader.dates)
    # Every session strictly BEFORE the perturbation boundary (dates[30],
    # review R1 #1: the invariance window must cover ALL of the past, not
    # a prefix of it) — a 1..5-session look-ahead leak cannot escape.
    past = loader.dates.index(_canary_dates()[30])

    dates = _canary_dates()
    with AshareDB(resolve_v2_duckdb_path(data_config)) as db:
        db.execute(
            "UPDATE daily_bar_raw SET close = close * 3, open = open * 3, "
            "high = high * 3, low = low * 3, pre_close = pre_close * 3, "
            "amount = amount * 3 "
            f"WHERE trade_date >= '{dates[30]}'"
        )

    loader2 = AshareDataLoader(
        data_config,
        ModelConfig(max_formula_len=4),
        dataset_identity_mode="query",
    )
    loader2.load_data()
    after = loader2.factor_tensor.detach().cpu().numpy()

    def with_dates_last(array: np.ndarray) -> np.ndarray:
        axis = array.shape.index(n_dates)
        return np.moveaxis(array, axis, -1)

    before_d, after_d = with_dates_last(before), with_dates_last(after)
    assert before_d.shape == after_d.shape
    assert np.array_equal(before_d[..., :past], after_d[..., :past]), (
        "a future-only bar perturbation changed past-session features"
    )
    assert not np.array_equal(before_d, after_d), (
        "the perturbation must change the future it targets"
    )
