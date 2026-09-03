"""Shared pytest fixtures and session-level run logging."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_data.config import BacktestConfig, DataConfig
from ashare_data.db import AshareDB
from ashare_data.manifest import build_dataset_manifest, save_manifest
from ashare_logging import (
    export_log_txt,
    guard_process_exit,
    setup_run_logging,
    worker_log_suffix,
)


DEFAULT_CODES = ["000001.SZ", "600000.SH", "300001.SZ"]


def make_bars(
    n_dates: int = 40,
    ts_codes: list[str] | None = None,
) -> tuple[list[str], list[str], pd.DataFrame]:
    ts_codes = ts_codes or DEFAULT_CODES
    dates = pd.bdate_range("2024-01-01", periods=n_dates).strftime("%Y%m%d").tolist()
    rows = []
    for code in ts_codes:
        for i, date in enumerate(dates):
            close = 10.0 + i * 0.1 + (0.5 if code == "000001.SZ" else 0.0)
            open_ = close * 0.99
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "open": open_,
                    "high": close * 1.02,
                    "low": open_ * 0.98,
                    "close": close,
                    "pre_close": close * 0.99,
                    "volume": 1_000_000.0,
                    "amount": 1_000_000.0 * close,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
            )
    return dates, ts_codes, pd.DataFrame(rows)


@pytest.fixture(scope="session", autouse=True)
def _run_logging():
    setup_run_logging(run_name="pytest")
    yield
    # F-08 companion coverage (IP-11): the 2026-09-03 pytest exit lag and
    # the sync tail hang are one root-cause family.  Name any surviving
    # non-daemon thread (with its stack) in this session log — exported
    # right below — before interpreter teardown.  Detect-only: a forced
    # exit here would swallow the terminal report / CI result.
    guard_process_exit(timeout=5.0, force_exit=False)
    project_root = Path(__file__).resolve().parents[1]
    # Per-worker export target (PR3/B2+C2): each xdist worker exports its
    # own buffer to its own file; serial runs keep logs/pytest.txt.
    export_log_txt(path=project_root / "logs" / f"pytest{worker_log_suffix()}.txt")


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


@pytest.fixture
def bars_data():
    return make_bars()


def _populate_db(
    data_config: DataConfig,
    dates: list[str],
    ts_codes: list[str],
    bars,
    *,
    persist_manifest: bool,
) -> DataConfig:
    """Populate the shared synthetic database (single implementation).

    ``persist_manifest`` controls whether the content-addressed dataset
    manifest is saved: formal artifact flows (P8-05) require a resolved
    dataset identity, while manifest-machinery tests need a database
    without one.
    """

    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    data_config.parquet_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {
                    "ts_code": c,
                    "name": c,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for c in ts_codes
            ],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in dates],
            data_config,
        )
        db.upsert_constituents(
            [
                {
                    "index_code": "000300.SH",
                    "ts_code": c,
                    "in_date": f"2020010{i + 1}",
                    "out_date": "99991231",
                }
                for i, c in enumerate(ts_codes)
            ],
            data_config,
        )
        if persist_manifest:
            # P8-05: formal artifacts require a resolved dataset identity,
            # so the shared fixture persists the content-addressed manifest.
            save_manifest(
                db, data_config, build_dataset_manifest(db, data_config)
            )
    return data_config


@pytest.fixture
def populated_db(data_config: DataConfig, bars_data) -> DataConfig:
    dates, ts_codes, bars = bars_data
    return _populate_db(data_config, dates, ts_codes, bars, persist_manifest=True)


@pytest.fixture
def populated_db_without_manifest(data_config: DataConfig, bars_data) -> DataConfig:
    """The same synthetic database without a persisted dataset manifest."""
    dates, ts_codes, bars = bars_data
    return _populate_db(data_config, dates, ts_codes, bars, persist_manifest=False)


# --- P8-05: formal-artifact fixture helpers (single implementation) ----------


def make_test_spec(**overrides):
    """A valid frozen RunSpec for fixture writers.

    Provenance (git commit, dependency-lock hash) is captured from the real
    repository with the clean-tree gate off — test worktrees are routinely
    dirty; the SPEC_LOCKED clean-tree evidence gate is a lifecycle-stage
    concern (P8-06+), not exercised here.
    """

    from ashare_model.runspec import WindowSpec, build_runspec
    from ashare_portfolio.execution_spec import portfolio_config_provenance

    kwargs = dict(
        dataset_id="ds-test",
        data_cutoff="20241231",
        frequency="daily",
        target="forward_return",
        horizon=1,
        requested_budget=100,
        n_folds=1,
        dev_window=WindowSpec(start="20240101", end="20240201"),
        holdout_window=WindowSpec(start="20240201", end="20241231"),
        preregistered_contract_hash="a" * 64,
        resolved_thresholds={"min_eligible": 1.0},
        portfolio_config=portfolio_config_provenance(BacktestConfig()),
        seeds=(42,),
        searcher="random",
        repo_root=None,
        require_clean_tree=False,
    )
    kwargs.update(overrides)
    return build_runspec(**kwargs)


def open_test_handle(store_root, spec=None, **spec_overrides):
    """Open a fresh run in a test RunStore. The caller closes the handle."""

    from ashare_model.run_store import RunStore

    return RunStore(store_root).open_run(spec or make_test_spec(**spec_overrides))


def write_current_strategy(
    data_dir: Path,
    *,
    formula=(1,),
    formula_text: str = "RET_1",
    direction: int = 1,
    dataset_id: str,
    backtest_config: BacktestConfig | None = None,
    spec=None,
) -> dict:
    """Write a schema-v2 strategy artifact through the formal P8-05 path
    (RunStore-registered + convenience mirror) and return the persisted
    payload. This is the fixture replacement for hand-written legacy
    ``best_ashare_strategy.json`` files.
    """

    from ashare_model.alphagpt import MODEL_VERSION
    from ashare_model.artifact_schemas import StrategyArtifact
    from ashare_model.artifact_writer import write_boundary_artifact
    from ashare_model.identity import candidate_id
    from ashare_model.research_domain import RESEARCH_DOMAIN_VERSION
    from ashare_model.reward import REWARD_VERSION
    from ashare_model.search_contract import SEARCH_CONTRACT_VERSION
    from ashare_model.semantic_cache import SEMANTIC_CACHE_VERSION
    from ashare_model.vocab import FORMULA_VOCAB, GRAMMAR_VERSION
    from ashare_portfolio.execution_spec import execution_provenance

    from ashare_model.evaluation import PROTOCOL_VERSION
    from ashare_portfolio.execution_spec import portfolio_config_provenance

    backtest_config = backtest_config or BacktestConfig()
    if spec is None:
        spec = make_test_spec(
            dataset_id=dataset_id,
            frequency=backtest_config.rebalance_frequency,
            horizon=backtest_config.target_horizon,
            portfolio_config=portfolio_config_provenance(backtest_config),
        )
    cand = candidate_id(spec.spec_id, list(formula), direction)
    payload = {
        "formula": list(formula),
        "formula_text": formula_text,
        "source": "random",
        "direction": direction,
        "data_tier": None,
        "val_reward": 0.1,
        "val_icir": 0.2,
        "train_reward": 0.1,
        "train_icir": 0.2,
        "complexity_penalty": 0.0,
        "complexity_cost": 1.0,
        "active_ir": None,
        "risk_exposure": None,
        "average_turnover": None,
        "capacity_utilization": None,
        "eligible": True,
        "rejection_reasons": [],
        "history": [],
        "searcher": "random",
        "init_seed": 42,
        "device": "cpu",
        "model_version": MODEL_VERSION,
        # Real vocab identity: resolve_formula_tokens remaps by name, so the
        # fixture must carry the true vocabulary to round-trip losslessly.
        "feature_names": list(FORMULA_VOCAB.feature_names),
        "operator_names": list(FORMULA_VOCAB.operator_names),
        "feature_version": FORMULA_VOCAB.feature_version,
        "grammar_version": GRAMMAR_VERSION,
        "reward_version": REWARD_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "research_domain": "unified",
        "research_domain_version": RESEARCH_DOMAIN_VERSION,
        **execution_provenance(backtest_config),
        "semantic_cache_version": SEMANTIC_CACHE_VERSION,
        "unique_semantic_evals": 1,
        "semantic_cache_stats": None,
        "search_contract_version": SEARCH_CONTRACT_VERSION,
        "search_result": None,
        "dataset_id": dataset_id,
        "searcher_candidate_label": f"random:{','.join(str(t) for t in formula)}",
    }
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open_test_handle(data_dir, spec=spec) as handle:
        write_boundary_artifact(
            handle,
            artifact_type="strategy",
            model_cls=StrategyArtifact,
            payload=payload,
            candidate_id=cand,
            convenience_path=data_dir / "best_ashare_strategy.json",
        )
    # Return the persisted mirror — exactly what formal consumers will read.
    return json.loads(
        (data_dir / "best_ashare_strategy.json").read_text(encoding="utf-8")
    )
