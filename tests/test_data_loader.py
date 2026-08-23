from __future__ import annotations

import numpy as np
import pytest
import torch

from ashare_data.config import DataConfig, ModelConfig
from ashare_data.db import AshareDB
from ashare_data.universe import UniverseContractError, UniverseReason
from ashare_model.data_loader import AshareDataLoader, build_loader_from_config
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB
from tests.conftest import make_bars


def _seed_loader_db(
    config: DataConfig,
    dates: list[str],
    codes: list[str],
    bars,
    *,
    list_dates: dict[str, str | None] | None = None,
    current_st: set[str] | None = None,
    memberships: list[dict] | None = None,
    calendar: list[dict] | None = None,
) -> None:
    list_dates = list_dates or {code: "20200101" for code in codes}
    current_st = current_st or set()
    if memberships is None:
        memberships = [
            {
                "index_code": "000300.SH",
                "ts_code": code,
                "in_date": "20200101",
                "out_date": "99991231",
            }
            for code in codes
        ]
    if calendar is None:
        calendar = [{"trade_date": date, "is_open": True} for date in dates]
    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
        if bars is not None and not bars.empty:
            db.upsert_daily(bars.to_dict("records"), config)
        db.upsert_stocks(
            [
                {
                    "ts_code": code,
                    "name": f"*ST {code}" if code in current_st else code,
                    "industry": None,
                    "list_date": list_dates.get(code),
                    "is_st": code in current_st,
                }
                for code in codes
            ],
            config,
        )
        db.upsert_calendar(calendar, config)
        db.upsert_constituents(memberships, config)


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


def test_load_data_exposes_industry_codes_for_vm(populated_db: DataConfig):
    # The VM's CS_NEUTRALIZE consumes dense industry group ids aligned with
    # the factor stack: same industry -> same id, unmapped stock -> NaN.
    with AshareDB(populated_db.duckdb_path) as db:
        db.replace_sw_members("801780", ["000001.SZ", "600000.SH"], populated_db)
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    codes = loader.industry_codes
    assert codes.dtype == torch.float32
    assert codes.shape == (len(loader.ts_codes), len(loader.dates))
    a = loader.ts_codes.index("000001.SZ")
    b = loader.ts_codes.index("600000.SH")
    c = loader.ts_codes.index("300001.SZ")
    assert codes[a, 0] == codes[b, 0]
    assert torch.isnan(codes[c, 0])


def test_industry_codes_all_nan_without_membership(populated_db: DataConfig):
    # Without a Shenwan membership table the codes degrade to all-NaN: the
    # neutralization operator then falls back to the full-market demean.
    loader = AshareDataLoader(populated_db, ModelConfig())
    loader.load_data()
    assert torch.isnan(loader.industry_codes).all()
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
        db.upsert_calendar(
            [{"trade_date": "20240102", "is_open": True}], data_config
        )
    loader = AshareDataLoader(data_config, ModelConfig())
    with pytest.raises(ValueError, match="no historical membership intervals"):
        loader.load_data()


def test_load_data_raises_when_no_bars(data_config: DataConfig):
    data_config.data_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_stocks(
            [{"ts_code": "000001.SZ", "name": "A", "industry": None, "list_date": "20200101", "is_st": False}],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": "20240102", "is_open": True}], data_config
        )
        db.upsert_constituents(
            [{"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20200101", "out_date": "99991231"}],
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
    dates, ts_codes, bars = make_bars(5)
    bars = bars[~((bars["ts_code"] == "600000.SH") & (bars["trade_date"] == dates[1]))]
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_daily(bars.to_dict("records"), data_config)
        db.upsert_stocks(
            [
                {"ts_code": code, "name": code, "industry": None, "list_date": "20200101", "is_st": False}
                for code in ts_codes
            ],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": date, "is_open": True} for date in dates],
            data_config,
        )
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": code, "in_date": f"2020010{i + 1}", "out_date": "99991231"}
                for i, code in enumerate(ts_codes)
            ],
            data_config,
        )
    loader = AshareDataLoader(data_config, ModelConfig())
    loader.load_data(ts_codes=ts_codes, dates=dates)
    assert loader.target_ret is not None
    target = loader.target_ret.numpy()
    assert np.nanmax(np.abs(target)) < 1.0
    missing_row = loader.ts_codes.index("600000.SH")
    missing_col = loader.dates.index(dates[1])
    assert np.isnan(target[missing_row, missing_col])


def test_current_st_stock_is_retained_with_unknown_status_reason(
    data_config: DataConfig,
):
    dates, ts_codes, bars = make_bars(6)
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": f"2020010{i + 1}",
            "out_date": "99991231",
        }
        for i, code in enumerate(ts_codes)
    ]
    _seed_loader_db(
        data_config,
        dates,
        ts_codes,
        bars,
        current_st={"300001.SZ"},
        memberships=memberships,
    )
    loader = AshareDataLoader(data_config, ModelConfig())
    loader.load_data()

    assert "300001.SZ" in loader.ts_codes
    assert not hasattr(loader, "st_stocks")
    assert loader.stock_list_dates["300001.SZ"] == "20200101"
    # The current snapshot is exposed explicitly for same-day execution
    # only (run_sim consumes it); nothing else may read it as history.
    assert loader.current_st_codes == {"300001.SZ"}
    assert loader.universe_reason_codes is not None
    unknown = int(UniverseReason.STATUS_UNKNOWN)
    assert np.all(loader.universe_reason_codes & unknown)
    assert loader.universe_mask is not None
    assert loader.universe_mask.flags.writeable is False
    assert loader.universe_mask.shape == (
        len(loader.ts_codes),
        len(loader.dates),
    )


def test_loader_factor_reference_sets_exclude_pre_join_members(
    data_config: DataConfig,
):
    # The loader must hand its PIT mask to the factor engine: an extreme
    # pre-join value of a future member cannot shift the eligible stocks'
    # factor values on pre-join dates.
    dates, codes, bars = make_bars(8, ["000001.SZ", "600000.SH", "300001.SZ"])
    bars.loc[
        (bars["ts_code"] == "300001.SZ") & (bars["trade_date"] < dates[2]),
        "amount",
    ] = 1e12
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": "000001.SZ",
            "in_date": "20200101",
            "out_date": "99991231",
        },
        {
            "index_code": "000300.SH",
            "ts_code": "600000.SH",
            "in_date": "20200101",
            "out_date": "99991231",
        },
        {
            "index_code": "000300.SH",
            "ts_code": "300001.SZ",
            "in_date": dates[2],
            "out_date": "99991231",
        },
    ]
    _seed_loader_db(
        data_config,
        dates,
        codes,
        bars,
        memberships=memberships,
    )
    loader = AshareDataLoader(data_config, ModelConfig()).load_data()
    tensor = loader.factor_tensor.numpy()
    amt = FEATURE_NAMES.index("AMOUNT_SHARE")
    a = loader.ts_codes.index("000001.SZ")
    b = loader.ts_codes.index("600000.SH")
    # Pre-join, the amount-share denominator counts only the two eligible
    # stocks: their standardized shares are exact opposites at full scale.
    # Had the extreme ineligible amount leaked into the denominator, both
    # shares would collapse to the neutral 0.
    for day in (0, 1):
        assert abs(tensor[amt, a, day]) > 0.5
        assert tensor[amt, a, day] == pytest.approx(-tensor[amt, b, day])


def test_load_universe_filters_index_codes(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_stocks(
            [{"ts_code": "000001.SZ", "name": "A", "industry": None, "list_date": "20200101", "is_st": False}],
            data_config,
        )
        db.upsert_calendar(
            [{"trade_date": "20240102", "is_open": True}], data_config
        )
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "000300.SZ", "in_date": "20200101", "out_date": "99991231"},
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20200102", "out_date": "99991231"},
                {"index_code": "000300.SH", "ts_code": "900901.SH", "in_date": "20200103", "out_date": "99991231"},
                {"index_code": "000905.SH", "ts_code": "600000.SH", "in_date": "20200101", "out_date": "99991231"},
            ],
            data_config,
        )
    loader = AshareDataLoader(data_config, ModelConfig())
    assert loader.load_universe() == ["000001.SZ"]


def test_requested_axes_slice_universe_and_reload_clears_shape_caches(
    data_config: DataConfig,
):
    dates, codes, bars = make_bars(6, ["000001.SZ", "600000.SH"])
    bars = bars[
        ~(
            (bars["ts_code"] == "600000.SH")
            & (bars["trade_date"] == dates[2])
        )
    ]
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": "000001.SZ",
            "in_date": "20200101",
            "out_date": "99991231",
        },
        {
            "index_code": "000300.SH",
            "ts_code": "600000.SH",
            "in_date": dates[1],
            "out_date": "99991231",
        },
    ]
    _seed_loader_db(
        data_config,
        dates,
        codes,
        bars,
        memberships=memberships,
    )
    loader = AshareDataLoader(data_config, ModelConfig()).load_data()
    assert loader.universe_mask is not None
    full_codes = list(loader.ts_codes)
    full_dates = list(loader.dates)
    full_mask = loader.universe_mask.copy()
    full_reasons = loader.universe_reason_codes.copy()
    loader.tradability_masks()
    assert loader._tradability_cache is not None

    selected_codes = ["600000.SH"]
    selected_dates = [dates[1], dates[2], dates[4]]
    expected_rows = [full_codes.index(code) for code in selected_codes]
    expected_cols = [full_dates.index(date) for date in selected_dates]
    loader.load_data(ts_codes=selected_codes, dates=selected_dates)

    assert loader.ts_codes == selected_codes
    assert loader.dates == selected_dates
    assert loader._tradability_cache is None
    assert np.array_equal(
        loader.universe_mask,
        full_mask[np.ix_(expected_rows, expected_cols)],
    )
    assert np.array_equal(
        loader.universe_reason_codes,
        full_reasons[np.ix_(expected_rows, expected_cols)],
    )
    assert loader.factor_tensor.shape[1:] == (1, 3)


@pytest.mark.parametrize(
    ("missing", "match"),
    [
        ("membership", "no historical membership intervals"),
        ("list_date", r"stocks\.list_date"),
        ("calendar", "no rows with is_open=True"),
    ],
)
def test_loader_reports_missing_universe_sources(
    data_config: DataConfig,
    missing: str,
    match: str,
):
    dates, codes, bars = make_bars(2, ["000001.SZ"])
    kwargs = {
        "list_dates": {"000001.SZ": None if missing == "list_date" else "20200101"},
        "memberships": [] if missing == "membership" else None,
        "calendar": [] if missing == "calendar" else None,
    }
    _seed_loader_db(data_config, dates, codes, bars, **kwargs)

    with pytest.raises(UniverseContractError, match=match):
        AshareDataLoader(data_config, ModelConfig()).load_data()


def test_loader_rejects_current_only_constituent_snapshot(data_config: DataConfig):
    dates, codes, bars = make_bars(2, ["000001.SZ", "600000.SH"])
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": "20200101",
            "out_date": "99991231",
        }
        for code in codes
    ]
    _seed_loader_db(
        data_config,
        dates,
        codes,
        bars,
        memberships=memberships,
    )

    with pytest.raises(UniverseContractError, match="current snapshot stretched"):
        AshareDataLoader(data_config, ModelConfig()).load_data()
