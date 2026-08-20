from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import DataConfig, ModelConfig
from ashare_data.db import AshareDB
from ashare_model.data_loader import AshareDataLoader, build_loader_from_config, date_index
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB


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


def test_load_data_feeds_industry_relative_factors(populated_db: DataConfig):
    # With Shenwan membership present the loader injects the industry code
    # frame: the industry-relative rows leave the neutral state, and a
    # single-member industry demeans to exactly zero (stays neutral).
    with AshareDB(populated_db.duckdb_path) as db:
        db.replace_sw_members("801780", ["000001.SZ", "600000.SH"], populated_db)
        db.replace_sw_members("801880", ["300001.SZ"], populated_db)
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    tensor = loader.factor_tensor.numpy()
    pair_row = tensor[FEATURE_NAMES.index("IND_REL_RET_20")]
    assert np.count_nonzero(pair_row) > 0
    # The loader sorts ts_codes; locate the stocks by code.  The two-member
    # industry demeans to exact opposites, the single-member industry to
    # exactly zero.
    a = loader.ts_codes.index("000001.SZ")
    b = loader.ts_codes.index("600000.SH")
    c = loader.ts_codes.index("300001.SZ")
    assert np.allclose(pair_row[a], -pair_row[b])
    assert np.allclose(pair_row[c], 0.0)
    # Without any membership table the same factor stays fully neutral.
    with AshareDB(populated_db.duckdb_path) as db:
        db.execute(f"DROP TABLE IF EXISTS {populated_db.sw_member_table}")
    bare = AshareDataLoader(populated_db, ModelConfig())
    bare.load_data()
    assert np.allclose(
        bare.factor_tensor[FEATURE_NAMES.index("IND_REL_RET_20")].numpy(), 0.0
    )


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


# --- date_index -------------------------------------------------------------


def test_date_index_finds_first_column_at_or_after():
    dates = ["20240101", "20240102", "20240105", "20240108"]
    assert date_index(dates, "2024-01-02") == 1
    assert date_index(dates, "20240103") == 2  # falls between columns
    assert date_index(dates, "2024-01-08") == 3


def test_date_index_clamps_and_runs_past_end():
    dates = ["20240101", "20240102"]
    # A cutoff before the first column clamps to 1: the window never
    # collapses to index 0.
    assert date_index(dates, "2023-12-01") == 1
    assert date_index(dates, "2024-02-01") == 2
