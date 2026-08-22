from __future__ import annotations

import pytest

from ashare_data.config import DataConfig, ModelConfig
from ashare_data.db import AshareDB
from ashare_data.universe import (
    UniverseContractError,
    UniverseDevelopmentFallbackWarning,
    require_production_universe,
    resolve_universe_contract,
)
from ashare_model.data_loader import AshareDataLoader
from tests.conftest import make_bars


def _seed_contract(
    config: DataConfig,
    memberships: list[dict],
    *,
    list_dates: dict[str, str | None] | None = None,
    calendar: list[dict] | None = None,
) -> None:
    codes = sorted({str(row["ts_code"]) for row in memberships})
    list_dates = list_dates or {code: "20190101" for code in codes}
    calendar = calendar or [
        {"trade_date": "20240102", "is_open": True},
        {"trade_date": "20240103", "is_open": True},
    ]
    with AshareDB(config.duckdb_path) as db:
        db.create_schema(config)
        db.upsert_stocks(
            [
                {
                    "ts_code": code,
                    "name": code,
                    "industry": None,
                    "list_date": list_dates.get(code),
                    "is_st": False,
                }
                for code in codes
            ],
            config,
        )
        db.upsert_calendar(calendar, config)
        db.upsert_constituents(memberships, config)


def test_strict_contract_reports_missing_interval_columns(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.execute(f"DROP TABLE {data_config.constituents_table}")
        db.execute(
            f"CREATE TABLE {data_config.constituents_table} "
            "(index_code VARCHAR, ts_code VARCHAR)"
        )
        db.upsert_calendar(
            [{"trade_date": "20240102", "is_open": True}], data_config
        )
    with pytest.raises(
        UniverseContractError,
        match=r"constituents missing required columns: in_date, out_date",
    ):
        require_production_universe(data_config)


def test_strict_contract_reports_missing_list_date(data_config: DataConfig):
    rows = [
        {
            "index_code": "000300.SH",
            "ts_code": "000001.SZ",
            "in_date": "20200101",
            "out_date": "99991231",
        }
    ]
    _seed_contract(data_config, rows, list_dates={"000001.SZ": None})
    with pytest.raises(UniverseContractError, match=r"stocks\.list_date"):
        require_production_universe(data_config)


def test_explicit_development_fallback_warns_and_records_status(
    data_config: DataConfig,
):
    rows = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": "20240102",
            "out_date": "99991231",
        }
        for code in ("000001.SZ", "600000.SH")
    ]
    _seed_contract(
        data_config,
        rows,
        list_dates={"000001.SZ": None, "600000.SH": None},
    )
    with pytest.warns(
        UniverseDevelopmentFallbackWarning,
        match="development universe fallback enabled",
    ):
        resolved = resolve_universe_contract(
            data_config, allow_development_fallback=True
        )
    assert resolved.status.degraded is True
    assert resolved.status.strict is False
    assert resolved.status.mode == "development_fallback"
    assert resolved.status.membership_source == "development_all_period"
    assert resolved.status.session_source.endswith(".is_open=True")
    assert resolved.status.warnings
    assert resolved.membership_mask(
        ["000001.SZ", "600000.SH"], ["20240102", "20240103"]
    ).all()


def test_environment_variable_cannot_enable_development_fallback(
    data_config: DataConfig, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("ASHARE_ALLOW_DEVELOPMENT_UNIVERSE_FALLBACK", "1")
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_calendar(
            [{"trade_date": "20240102", "is_open": True}], data_config
        )
    with pytest.raises(UniverseContractError):
        require_production_universe(data_config)


def test_development_membership_fallback_does_not_fabricate_sessions(
    data_config: DataConfig,
):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
    with pytest.raises(
        UniverseContractError, match=r"no rows with is_open=True"
    ):
        resolve_universe_contract(
            data_config, allow_development_fallback=True
        )


def test_current_snapshot_shape_is_not_historical_validity(data_config: DataConfig):
    rows = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": "20200101",
            "out_date": "99991231",
        }
        for code in ("000001.SZ", "600000.SH")
    ]
    _seed_contract(data_config, rows)
    with pytest.raises(UniverseContractError, match="current snapshot stretched"):
        require_production_universe(data_config)


def test_membership_mask_supports_reentry_and_half_open_intervals(
    data_config: DataConfig,
):
    rows = [
        {
            "index_code": "000300.SH",
            "ts_code": "000001.SZ",
            "in_date": "20240101",
            "out_date": "20240103",
        },
        {
            "index_code": "000300.SH",
            "ts_code": "000001.SZ",
            "in_date": "20240104",
            "out_date": "99991231",
        },
    ]
    _seed_contract(
        data_config,
        rows,
        calendar=[
            {"trade_date": date, "is_open": True}
            for date in ("20240102", "20240103", "20240104")
        ],
    )
    resolved = resolve_universe_contract(data_config)
    mask = resolved.membership_mask(
        ["000001.SZ"], ["20240102", "20240103", "20240104"]
    )
    assert mask.tolist() == [[True, False, True]]


def test_calendar_open_rows_define_loader_session_axis(data_config: DataConfig):
    dates, codes, bars = make_bars(3, ["000001.SZ"])
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": "000001.SZ",
            "in_date": "20200101",
            "out_date": "99991231",
        }
    ]
    _seed_contract(
        data_config,
        memberships,
        calendar=[
            {"trade_date": dates[0], "is_open": True},
            {"trade_date": dates[1], "is_open": False},
            {"trade_date": dates[2], "is_open": True},
        ],
    )
    with AshareDB(data_config.duckdb_path) as db:
        db.upsert_daily(bars.to_dict("records"), data_config)
    loader = AshareDataLoader(data_config, ModelConfig()).load_data()
    assert loader.dates == [dates[0], dates[2]]
    assert loader.raw_data_cache["open"].shape[1] == 2


def test_constituent_persistence_allows_adjacent_and_cross_index_overlap(
    data_config: DataConfig,
):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20200101", "out_date": "20210101"},
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20210101", "out_date": "20220101"},
                {"index_code": "000905.SH", "ts_code": "000001.SZ", "in_date": "20200601", "out_date": "20210601"},
            ],
            data_config,
        )
        assert db.query(
            f"SELECT COUNT(*) AS n FROM {data_config.constituents_table}"
        ).iloc[0]["n"] == 3
        with pytest.raises(UniverseContractError, match="overlapping"):
            db.upsert_constituents(
                [
                    {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20201201", "out_date": "20211201"}
                ],
                data_config,
            )


def test_create_schema_migrates_legacy_constituent_primary_key(
    data_config: DataConfig,
):
    with AshareDB(data_config.duckdb_path) as db:
        db.execute(
            f"""
            CREATE TABLE {data_config.constituents_table} (
                index_code VARCHAR,
                ts_code VARCHAR,
                in_date VARCHAR,
                out_date VARCHAR,
                PRIMARY KEY (index_code, ts_code)
            )
            """
        )
        db.execute(
            f"INSERT INTO {data_config.constituents_table} VALUES "
            "('000300.SH', '000001.SZ', '20200101', '20210101')"
        )
        db.create_schema(data_config)
        pk = db.query(
            """
            SELECT constraint_column_names
            FROM duckdb_constraints()
            WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'
            """,
            [data_config.constituents_table],
        ).iloc[0]["constraint_column_names"]
        assert list(pk) == ["index_code", "ts_code", "in_date"]
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20210101", "out_date": "20220101"}
            ],
            data_config,
        )
        assert db.query(
            f"SELECT COUNT(*) AS n FROM {data_config.constituents_table}"
        ).iloc[0]["n"] == 2


def test_loader_is_strict_by_default(data_config: DataConfig):
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_calendar(
            [{"trade_date": "20240102", "is_open": True}], data_config
        )
    loader = AshareDataLoader(data_config, ModelConfig())
    assert loader.allow_development_universe_fallback is False
    with pytest.raises(UniverseContractError):
        loader.load_data(ts_codes=["000001.SZ"], dates=["20240102"])
