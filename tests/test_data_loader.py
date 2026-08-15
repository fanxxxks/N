from __future__ import annotations

import pytest

from ashare_data.config import DataConfig, ModelConfig
from ashare_data.db import AshareDB
from ashare_model.data_loader import AshareDataLoader, build_loader_from_config
from ashare_model.vocab import FORMULA_VOCAB


def test_load_universe_and_data(populated_db: DataConfig):
    loader = AshareDataLoader(populated_db, ModelConfig())
    assert loader.load_universe() == ["000001.SZ", "300001.SZ", "600000.SH"]
    loader.load_data()
    assert loader.factor_tensor.shape[0] == FORMULA_VOCAB.feature_count
    assert set(loader.raw_data_cache) == {
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "turnover_rate",
        "adj_factor",
    }
    assert loader.target_ret[:, -2:].abs().sum() == 0


def test_load_data_raises_when_universe_empty(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
    loader = AshareDataLoader(data_config, ModelConfig())
    with pytest.raises(ValueError, match="No universe symbols loaded"):
        loader.load_data()


def test_load_data_raises_when_no_bars(data_config: DataConfig):
    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_stocks(
            [{"ts_code": "000001.SZ", "name": "A", "industry": None, "list_date": "20200101", "is_st": False}],
            data_config,
        )
    loader = AshareDataLoader(data_config, ModelConfig())
    with pytest.raises(ValueError, match="No daily bars found"):
        loader.load_data()


def test_build_loader_from_config(tmp_path):
    loader = build_loader_from_config(tmp_path)
    assert isinstance(loader, AshareDataLoader)


def test_target_ret_masks_suspended_days(data_config: DataConfig):
    """A suspended (missing) day must not fabricate 1e10-scale targets."""
    from tests.conftest import make_bars

    dates, ts_codes, bars = make_bars(5)
    bars = bars[~((bars["ts_code"] == "600000.SH") & (bars["trade_date"] == dates[1]))]
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
    loader = AshareDataLoader(data_config, ModelConfig())
    loader.load_data(ts_codes=ts_codes, dates=dates)
    assert loader.target_ret is not None
    assert loader.target_ret.abs().max().item() < 1.0


def test_load_stock_meta_and_st_exclusion(data_config: DataConfig):
    from tests.conftest import make_bars

    dates, ts_codes, bars = make_bars(6)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {"ts_code": "000001.SZ", "name": "平安银行", "industry": None, "list_date": "20200101", "is_st": False},
                {"ts_code": "600000.SH", "name": "浦发银行", "industry": None, "list_date": "20200101", "is_st": False},
                {"ts_code": "300001.SZ", "name": "*ST 风险", "industry": None, "list_date": "20200101", "is_st": True},
            ],
            data_config,
        )
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": c, "in_date": "20240101", "out_date": "99991231"}
                for c in ts_codes
            ],
            data_config,
        )
    loader = AshareDataLoader(data_config, ModelConfig())
    loader.load_stock_meta()
    assert loader.stock_names["000001.SZ"] == "平安银行"
    assert "300001.SZ" in loader.st_stocks
    universe = loader.load_universe()
    assert "300001.SZ" not in universe
    assert "000001.SZ" in universe and "600000.SH" in universe


def test_load_universe_filters_index_codes(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "000300.SZ", "in_date": "20240101", "out_date": "99991231"},
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20240101", "out_date": "99991231"},
                {"index_code": "000300.SH", "ts_code": "900901.SH", "in_date": "20240101", "out_date": "99991231"},
            ],
            data_config,
        )
    loader = AshareDataLoader(data_config, ModelConfig())
    assert loader.load_universe() == ["000001.SZ"]
