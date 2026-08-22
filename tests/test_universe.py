from __future__ import annotations

import numpy as np
import pytest

from ashare_data.config import DataConfig, ModelConfig
from ashare_data.db import AshareDB
from ashare_data.universe import (
    UniverseContractError,
    UniverseDevelopmentFallbackWarning,
    UniverseMask,
    UniversePolicy,
    UniverseReason,
    build_universe_mask,
    require_production_universe,
    resolve_universe_contract,
)
from ashare_model.data_loader import AshareDataLoader
from tests.conftest import make_bars


INDEX_300 = "000300.SH"
INDEX_500 = "000905.SH"
STOCK = "000001.SZ"


def _policy(
    *,
    index_codes: tuple[str, ...] = (INDEX_300,),
    min_listed_sessions: int = 1,
    membership_end_inclusive: bool = False,
) -> UniversePolicy:
    return UniversePolicy(
        index_codes=index_codes,
        min_listed_sessions=min_listed_sessions,
        membership_end_inclusive=membership_end_inclusive,
    )


def _membership(
    in_date: str,
    out_date: str,
    *,
    index_code: str = INDEX_300,
    ts_code: str = STOCK,
) -> dict[str, str]:
    return {
        "index_code": index_code,
        "ts_code": ts_code,
        "in_date": in_date,
        "out_date": out_date,
    }


def _has_reason(value: np.uint16, reason: UniverseReason) -> bool:
    return bool(int(value) & int(reason))


def test_build_universe_mask_uses_half_open_membership_boundaries():
    dates = ["20240102", "20240103", "20240104", "20240105"]
    result = build_universe_mask(
        [STOCK],
        dates,
        dates,
        [_membership("20240103", "20240105")],
        {STOCK: "20200101"},
        np.ones((1, 4), dtype=bool),
        _policy(),
    )

    assert result.eligible.tolist() == [[False, True, True, False]]
    assert _has_reason(result.reasons[0, 0], UniverseReason.NOT_MEMBER)
    assert not _has_reason(result.reasons[0, 1], UniverseReason.NOT_MEMBER)
    assert _has_reason(result.reasons[0, 3], UniverseReason.NOT_MEMBER)


@pytest.mark.parametrize(
    ("inclusive", "expected"),
    [
        (False, [[True, False]]),
        (True, [[True, True]]),
    ],
)
def test_membership_end_semantics_are_converted_once(
    inclusive: bool, expected: list[list[bool]]
):
    dates = ["20240102", "20240103"]
    result = build_universe_mask(
        [STOCK],
        dates,
        dates,
        [_membership("20240102", "20240103")],
        {STOCK: "20200101"},
        [[True, True]],
        _policy(membership_end_inclusive=inclusive),
    )

    assert result.eligible.tolist() == expected


def test_inclusive_open_ended_membership_does_not_overflow():
    result = build_universe_mask(
        [STOCK],
        ["20240102"],
        ["20240102"],
        [_membership("20240102", "99991231")],
        {STOCK: "20200101"},
        [[True]],
        _policy(membership_end_inclusive=True),
    )

    assert result.eligible.tolist() == [[True]]


def test_all_domain_dates_are_normalized_to_yyyymmdd():
    result = build_universe_mask(
        [STOCK],
        ["2024-01-02"],
        ["2024-01-02"],
        [_membership("2020-01-01", "9999-12-31")],
        {STOCK: "2020-01-01"},
        [[True]],
        _policy(),
    )

    assert result.eligible.tolist() == [[True]]


def test_multi_index_union_allows_cross_index_overlap():
    dates = ["20240102", "20240103", "20240104"]
    result = build_universe_mask(
        [STOCK],
        dates,
        dates,
        [
            _membership("20240102", "20240104", index_code=INDEX_300),
            _membership("20240103", "20240105", index_code=INDEX_500),
        ],
        {STOCK: "20200101"},
        [[True, True, True]],
        _policy(index_codes=(INDEX_300, INDEX_500)),
    )

    assert result.eligible.tolist() == [[True, True, True]]


@pytest.mark.parametrize(
    "intervals",
    [
        [
            _membership("20240102", "20240104"),
            _membership("20240102", "20240104"),
        ],
        [
            _membership("20240102", "20240105"),
            _membership("20240104", "20240106"),
        ],
    ],
)
def test_same_index_duplicate_or_overlapping_intervals_are_rejected(
    intervals: list[dict[str, str]],
):
    with pytest.raises(UniverseContractError, match="duplicate or overlapping"):
        build_universe_mask(
            [STOCK],
            ["20240102"],
            ["20240102"],
            intervals,
            {STOCK: "20200101"},
            [[True]],
            _policy(),
        )


def test_non_overlapping_reentry_intervals_are_preserved():
    dates = ["20240102", "20240103", "20240104"]
    result = build_universe_mask(
        [STOCK],
        dates,
        dates,
        [
            _membership("20240102", "20240103"),
            _membership("20240104", "20240105"),
        ],
        {STOCK: "20200101"},
        [[True, True, True]],
        _policy(),
    )

    assert result.eligible.tolist() == [[True, False, True]]


def test_listing_age_counts_open_sessions_and_nth_session_passes():
    sessions = ["20240105", "20240108", "20240109"]
    result = build_universe_mask(
        [STOCK],
        sessions,
        sessions,
        [_membership("20200101", "99991231")],
        {STOCK: "20240105"},
        [[True, True, True]],
        _policy(min_listed_sessions=2),
    )

    assert result.eligible.tolist() == [[False, True, True]]
    assert _has_reason(
        result.reasons[0, 0], UniverseReason.LISTING_AGE_INSUFFICIENT
    )
    assert not _has_reason(
        result.reasons[0, 1], UniverseReason.LISTING_AGE_INSUFFICIENT
    )


def test_not_yet_listed_and_listing_age_reasons_are_mutually_exclusive():
    dates = ["20240105", "20240108", "20240109"]
    result = build_universe_mask(
        [STOCK],
        dates,
        dates,
        [_membership("20200101", "99991231")],
        {STOCK: "20240108"},
        [[True, True, True]],
        _policy(min_listed_sessions=2),
    )

    assert _has_reason(result.reasons[0, 0], UniverseReason.NOT_YET_LISTED)
    assert not _has_reason(
        result.reasons[0, 0], UniverseReason.LISTING_AGE_INSUFFICIENT
    )
    assert not _has_reason(result.reasons[0, 1], UniverseReason.NOT_YET_LISTED)
    assert _has_reason(
        result.reasons[0, 1], UniverseReason.LISTING_AGE_INSUFFICIENT
    )
    assert result.eligible.tolist() == [[False, False, True]]


def test_missing_bar_reason_combines_with_membership_reason():
    dates = ["20240102", "20240103"]
    result = build_universe_mask(
        [STOCK],
        dates,
        dates,
        [_membership("20240103", "99991231")],
        {STOCK: "20200101"},
        [[False, False]],
        _policy(),
    )

    assert _has_reason(result.reasons[0, 0], UniverseReason.NOT_MEMBER)
    assert _has_reason(result.reasons[0, 0], UniverseReason.MISSING_BAR)
    assert not _has_reason(result.reasons[0, 1], UniverseReason.NOT_MEMBER)
    assert _has_reason(result.reasons[0, 1], UniverseReason.MISSING_BAR)
    assert result.eligible.tolist() == [[False, False]]


def test_unknown_listing_status_is_a_non_blocking_audit_reason():
    result = build_universe_mask(
        [STOCK],
        ["20240102"],
        ["20240102"],
        [_membership("20200101", "99991231")],
        {},
        [[True]],
        _policy(),
    )

    assert result.eligible.tolist() == [[True]]
    assert _has_reason(result.reasons[0, 0], UniverseReason.STATUS_UNKNOWN)


def test_nan_listing_status_is_a_non_blocking_audit_reason():
    result = build_universe_mask(
        [STOCK],
        ["20240102"],
        ["20240102"],
        [_membership("20200101", "99991231")],
        {STOCK: np.nan},
        [[True]],
        _policy(),
    )

    assert result.eligible.tolist() == [[True]]
    assert _has_reason(result.reasons[0, 0], UniverseReason.STATUS_UNKNOWN)


def test_universe_mask_arrays_have_fixed_dtypes_and_are_read_only():
    result = UniverseMask(
        eligible=np.array([[1]], dtype=np.int8),
        reasons=np.array([[UniverseReason.STATUS_UNKNOWN]], dtype=object),
    )

    assert result.eligible.dtype == np.bool_
    assert result.reasons.dtype == np.uint16
    assert result.eligible.shape == result.reasons.shape == (1, 1)
    assert result.eligible.flags.writeable is False
    assert result.reasons.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        result.eligible[0, 0] = False
    with pytest.raises(ValueError, match="read-only"):
        result.reasons[0, 0] = 0


@pytest.mark.parametrize(
    ("field", "signal_dates", "open_sessions", "list_dates", "match"),
    [
        ("signal", ["20240230"], ["20240102"], {STOCK: "20200101"}, "signal_dates"),
        ("session", ["20240102"], ["bad-date"], {STOCK: "20200101"}, "open_sessions"),
        ("listing", ["20240102"], ["20240102"], {STOCK: "20241301"}, "list_dates"),
    ],
)
def test_invalid_input_dates_are_rejected(
    field: str,
    signal_dates: list[str],
    open_sessions: list[str],
    list_dates: dict[str, str],
    match: str,
):
    del field
    with pytest.raises(UniverseContractError, match=match):
        build_universe_mask(
            [STOCK],
            signal_dates,
            open_sessions,
            [_membership("20200101", "99991231")],
            list_dates,
            [[True]],
            _policy(),
        )


def test_bar_presence_shape_is_validated():
    with pytest.raises(UniverseContractError, match=r"\[stock, date\] shape"):
        build_universe_mask(
            [STOCK],
            ["20240102", "20240103"],
            ["20240102", "20240103"],
            [_membership("20200101", "99991231")],
            {STOCK: "20200101"},
            [[True]],
            _policy(),
        )


@pytest.mark.parametrize(
    ("in_date", "out_date", "inclusive"),
    [
        ("20240103", "20240103", False),
        ("20240104", "20240103", True),
    ],
)
def test_invalid_membership_intervals_are_rejected(
    in_date: str, out_date: str, inclusive: bool
):
    with pytest.raises(UniverseContractError, match="invalid .* interval"):
        build_universe_mask(
            [STOCK],
            ["20240103"],
            ["20240103"],
            [_membership(in_date, out_date)],
            {STOCK: "20200101"},
            [[True]],
            _policy(membership_end_inclusive=inclusive),
        )


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
