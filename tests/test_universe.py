from __future__ import annotations

import numpy as np
import pytest
import torch

from ashare_data.config import (
    BacktestConfig,
    DataConfig,
    FoldConfig,
    ModelConfig,
    ProtocolConfig,
    RewardConfig,
)
from ashare_data.db import AshareDB
from ashare_data.processor import open_to_open_returns
from ashare_data.universe import (
    UniverseContractError,
    UniverseDevelopmentFallbackWarning,
    UniverseMask,
    UniversePolicy,
    UniverseReason,
    build_universe_mask,
    member_bar_coverage,
    require_production_universe,
    resolve_universe_contract,
)
from ashare_model.backtest import AshareBacktestEngine, equal_weight_benchmark_returns
from ashare_model.candidates import CandidateScorer, CandidateSelector, CandidateSpec
from ashare_model.data_loader import AshareDataLoader
from ashare_model.diagnostics import factor_report
from ashare_model.evaluation import (
    baseline_candidates,
    benchmark_row,
    resolve_folds,
)
from ashare_model.ops import OPS_CONFIG
from ashare_model.reward import (
    batched_basket_rewards,
    formula_reward,
    icir_from_series,
    rank_ic_series,
)
from ashare_model.train import sample_random_formulas
from ashare_model.vm import StackVM
from ashare_model.vocab import FEATURE_NAMES, FORMULA_VOCAB
from tests.conftest import make_bars
from tests.test_run_sim import _make_runner, _orders_for, _replace_stock_bars, _write_sim_db


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


def test_missing_dated_st_status_is_a_non_blocking_audit_reason():
    result = build_universe_mask(
        [STOCK],
        ["20240102"],
        ["20240102"],
        [_membership("20200101", "99991231")],
        {STOCK: "20200101"},
        [[True]],
        _policy(),
    )

    assert result.eligible.tolist() == [[True]]
    assert _has_reason(result.reasons[0, 0], UniverseReason.STATUS_UNKNOWN)


def test_member_bar_coverage_audits_zero_bar_intervals(data_config: DataConfig):
    """H3 audit: coverage rows per interval; an interval with open sessions
    but no bars is the signature of a never-synced historical member."""
    with AshareDB(data_config.duckdb_path) as db:
        db.create_schema(data_config)
        db.upsert_calendar(
            [{"trade_date": f"2024010{i}", "is_open": True} for i in range(1, 6)],
            data_config,
        )
        db.upsert_constituents(
            [
                {"index_code": "000300.SH", "ts_code": "000001.SZ", "in_date": "20240101", "out_date": "20240105"},
                {"index_code": "000300.SH", "ts_code": "600000.SH", "in_date": "20240101", "out_date": "20240105"},
                {"index_code": "000300.SH", "ts_code": "600999.SH", "in_date": "20250101", "out_date": "99991231"},
            ],
            data_config,
        )
        db.upsert_daily(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"2024010{i}",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "pre_close": 9.9,
                    "volume": 100.0,
                    "amount": 1000.0,
                    "turnover_rate": 1.0,
                    "adj_factor": 1.0,
                }
                for i in range(1, 6)
            ],
            data_config,
        )
        frame = member_bar_coverage(db, data_config)
    rows = frame.set_index("ts_code")
    # Half-open interval [20240101, 20240105): the bar on 20240105 must
    # not count, so five synced bars yield four audited bars.
    assert rows.loc["000001.SZ", "bars"] == 4
    assert rows.loc["000001.SZ", "coverage"] == 1.0
    # Synced member with a bar gap inside the interval: partial coverage.
    assert rows.loc["600000.SH", "bars"] == 0
    assert rows.loc["600000.SH", "coverage"] == 0.0
    # Future-only interval: no open sessions, coverage undefined (NaN).
    assert rows.loc["600999.SH", "sessions"] == 0
    assert np.isnan(rows.loc["600999.SH", "coverage"])


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
    assert resolved.codes == ["000001.SZ", "600000.SH"]


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


# --- centralized future-member sentinel contract -----------------------------
#
# One future member ``F`` with full pre-join bar history and extreme values is
# carried through the whole chain (factor preprocessing -> VM CS operators ->
# terminal z-score -> IC/ICIR -> CandidateScore -> reward -> random-search
# selection -> baseline -> diagnostics -> benchmark -> top-N -> backtest ->
# sim).  With ``F`` present vs removed, every eligible-stock result must be
# identical before the join day and only allowed to differ from the join day
# on.


_SENTINEL_BASE = ["000001.SZ", "600000.SH", "300001.SZ"]
_FUTURE = "300999.SZ"


def _sentinel_setup(
    tmp_path,
    *,
    join_day: int,
    n_dates: int,
    future_opens: list[float] | None = None,
) -> tuple[AshareDataLoader, AshareDataLoader, list[str], object, list[dict]]:
    """Two loaders over one DB: with the future member and without it.
    Returns ``(full, minus, dates, bars, memberships)``."""

    codes = [*_SENTINEL_BASE, _FUTURE]
    dates, _, default_bars = make_bars(n_dates, codes)
    # Extreme pre-join history: open/close scale 1e6 so any unmasked
    # cross-sectional statistic would be wrecked by F.
    bars = _replace_stock_bars(
        default_bars, _FUTURE, future_opens or [1e6] * n_dates
    )
    memberships = [
        {
            "index_code": "000300.SH",
            "ts_code": code,
            "in_date": "20200101",
            "out_date": "99991231",
        }
        for code in _SENTINEL_BASE
    ] + [
        {
            "index_code": "000300.SH",
            "ts_code": _FUTURE,
            "in_date": dates[join_day],
            "out_date": "99991231",
        }
    ]
    stocks = [
        {
            "ts_code": code,
            "name": code,
            "industry": None,
            "list_date": "20200101",
            "is_st": False,
        }
        for code in codes
    ]
    data_config = DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        min_listed_sessions=1,
        index_codes=["000300.SH"],
        index_names=["沪深300"],
    )
    _write_sim_db(
        data_config, dates, codes, bars,
        memberships=memberships, stocks=stocks,
    )
    full = AshareDataLoader(data_config, ModelConfig(max_formula_len=6))
    full.load_data(ts_codes=codes, dates=dates)
    minus = AshareDataLoader(data_config, ModelConfig(max_formula_len=6))
    minus.load_data(ts_codes=_SENTINEL_BASE, dates=dates)
    return full, minus, dates, bars, memberships, stocks


def _base_rows(loader: AshareDataLoader) -> list[int]:
    return [loader.ts_codes.index(code) for code in _SENTINEL_BASE]


def _future_row(loader: AshareDataLoader) -> int | None:
    try:
        return loader.ts_codes.index(_FUTURE)
    except ValueError:
        return None


def test_sentinel_future_member_changes_nothing_before_join(tmp_path):
    """The full-chain sentinel: F joins on the last signal date, so every
    computed quantity on the whole analysis window is pre-join and must be
    identical whether F is present or removed."""

    n_dates = 16
    join_day = n_dates - 2  # dates[14]: the last signal date's entry day
    full, minus, dates, bars, memberships, stocks = _sentinel_setup(
        tmp_path, join_day=join_day, n_dates=n_dates
    )
    f_row = _future_row(full)
    assert f_row is not None
    base_full = _base_rows(full)
    pre = slice(0, join_day + 1)  # dates[:join_day+1] are strictly pre-join
    # The mask itself: F ineligible before its join day.
    assert not full.universe_mask[f_row, :join_day].any()
    assert full.universe_mask[f_row, join_day:].all()

    # --- factor preprocessing -------------------------------------------
    f_full = full.factor_tensor.numpy()
    f_minus = minus.factor_tensor.numpy()
    assert np.allclose(
        f_full[:, base_full, : join_day], f_minus[:, :, : join_day]
    )
    # F's own pre-join history stays observable (never zeroed), only its
    # reference membership is excluded.
    assert np.isfinite(f_full[:, f_row, : join_day]).all()

    # --- VM CS operators + terminal z-score ------------------------------
    op_id = {
        name: FORMULA_VOCAB.operator_offset + i
        for i, (name, _, _) in enumerate(OPS_CONFIG)
    }
    cs_formula = [1, op_id["CS_RANK"]]
    vm_full = StackVM(
        FORMULA_VOCAB,
        universe_mask=torch.tensor(full.universe_mask, dtype=torch.bool),
    )
    vm_minus = StackVM(
        FORMULA_VOCAB,
        universe_mask=torch.tensor(minus.universe_mask, dtype=torch.bool),
    )
    sig_full = vm_full.execute(cs_formula, full.factor_tensor)
    sig_minus = vm_minus.execute(cs_formula, minus.factor_tensor)
    assert sig_full is not None and sig_minus is not None
    sig_full_np = sig_full.detach().numpy()
    sig_minus_np = sig_minus.detach().numpy()
    # Ineligible cells stay NaN (non-participating), never neutral zeros.
    assert np.isnan(sig_full_np[f_row, : join_day]).all()
    assert np.allclose(sig_full_np[base_full, : join_day], sig_minus_np[:, : join_day], equal_nan=True)

    # --- IC / ICIR --------------------------------------------------------
    target_full = full.mask_by_universe(
        open_to_open_returns(full.raw_data_cache["open"].numpy())
    )
    target_minus = minus.mask_by_universe(
        open_to_open_returns(minus.raw_data_cache["open"].numpy())
    )
    ic_full = rank_ic_series(
        sig_full_np[None], target_full, min_stocks=3,
        universe_mask=full.universe_mask,
    )
    ic_minus = rank_ic_series(
        sig_minus_np[None], target_minus, min_stocks=3,
        universe_mask=minus.universe_mask,
    )
    pre_ic = slice(0, join_day)
    assert np.allclose(ic_full[:, pre_ic], ic_minus[:, pre_ic], equal_nan=True)
    assert np.allclose(icir_from_series(ic_full[:, pre_ic]), icir_from_series(ic_minus[:, pre_ic]))

    # --- CandidateScore (bare-factor signal: F's extreme row is finite) --
    bt_cfg = BacktestConfig(top_n=2, train_end_date="2024-01-10")
    reward_cfg = RewardConfig()
    val_windows = [(4, 10)]
    scorer_full = CandidateScorer(bt_cfg, reward_cfg)
    scorer_minus = CandidateScorer(bt_cfg, reward_cfg)
    spec = CandidateSpec(candidate_id="sentinel", formula_text="RET_1", source="sentinel", tokens=(1,))
    raw_signal_full = f_full[FEATURE_NAMES.index("RET_1")]
    raw_signal_minus = f_minus[FEATURE_NAMES.index("RET_1")]
    score_full = scorer_full.score(
        spec, raw_signal_full, target_full, val_windows,
        universe_mask=full.universe_mask,
    )
    score_minus = scorer_minus.score(
        spec, raw_signal_minus, target_minus, val_windows,
        universe_mask=minus.universe_mask,
    )
    assert score_full.to_dict() == score_minus.to_dict()

    # --- reward -----------------------------------------------------------
    r_full = formula_reward(
        raw_signal_full, target_full, bt_cfg, reward_cfg,
        universe_mask=full.universe_mask,
    )
    r_minus = formula_reward(
        raw_signal_minus, target_minus, bt_cfg, reward_cfg,
        universe_mask=minus.universe_mask,
    )
    assert r_full == r_minus
    b_full = batched_basket_rewards(
        raw_signal_full[None], target_full, bt_cfg, reward_cfg, val_windows,
        universe_mask=full.universe_mask,
    )
    b_minus = batched_basket_rewards(
        raw_signal_minus[None], target_minus, bt_cfg, reward_cfg, val_windows,
        universe_mask=minus.universe_mask,
    )
    for left, right in zip(b_full, b_minus):
        if left is None:
            assert right is None
        else:
            assert np.allclose(left, right)

    # --- random-search best candidate -------------------------------------
    formulas = sample_random_formulas(seed=7, vocab=FORMULA_VOCAB, max_len=6, n=8)
    specs_full, signals_full = [], []
    specs_minus, signals_minus = [], []
    for key in formulas:
        spec = CandidateSpec(
            candidate_id="sentinel:" + ",".join(str(t) for t in key),
            formula_text="".join(str(t) for t in key),
            source="sentinel", tokens=key,
        )
        s_f = vm_full.execute(list(key), full.factor_tensor)
        s_m = vm_minus.execute(list(key), minus.factor_tensor)
        assert (s_f is None) == (s_m is None)
        specs_full.append(spec)
        specs_minus.append(spec)
        signals_full.append(None if s_f is None else s_f.detach().numpy())
        signals_minus.append(None if s_m is None else s_m.detach().numpy())
    selection_full = CandidateSelector().select(
        scorer_full.score_many(
            specs_full, signals_full, target_full, val_windows,
            universe_mask=full.universe_mask,
        )
    )
    selection_minus = CandidateSelector().select(
        scorer_minus.score_many(
            specs_minus, signals_minus, target_minus, val_windows,
            universe_mask=minus.universe_mask,
        )
    )
    assert selection_full.to_dict() == selection_minus.to_dict()

    # --- baseline + benchmark + diagnostics -------------------------------
    fold = resolve_folds(
        [FoldConfig("2024-01-10", "2024-01-18")], dates
    )[0]
    proto = ProtocolConfig(baseline_signals=["RET_1"])
    base_full_rows = baseline_candidates(full, proto, fold, bt_cfg)
    base_minus_rows = baseline_candidates(minus, proto, fold, bt_cfg)
    assert base_full_rows == base_minus_rows
    assert benchmark_row(full, fold) == benchmark_row(minus, fold)
    report_full = factor_report(full, "2024-01-10")
    report_minus = factor_report(minus, "2024-01-10")
    # The code-union count legitimately differs (F exists in one loader);
    # every eligible-derived statistic must be identical.
    assert report_full["stock_count"] == 4 and report_minus["stock_count"] == 3
    assert {
        k: v for k, v in report_full.items() if k != "stock_count"
    } == {k: v for k, v in report_minus.items() if k != "stock_count"}

    # --- benchmark helper + top-N + backtest ------------------------------
    sig_range = list(range(join_day - 1))
    bench_full = equal_weight_benchmark_returns(
        target_full, sig_range, full.universe_mask
    )
    bench_minus = equal_weight_benchmark_returns(
        target_minus, sig_range, minus.universe_mask
    )
    assert bench_full == bench_minus
    engine_full = AshareBacktestEngine(bt_cfg)
    engine_minus = AshareBacktestEngine(bt_cfg)
    t = join_day - 2
    sel_full = engine_full._select_top_n(
        raw_signal_full[:, t], t + 1,
        full.raw_data_cache["open"].numpy(),
        full.raw_data_cache["high"].numpy(),
        full.raw_data_cache["low"].numpy(),
        full.raw_data_cache["pre_close"].numpy(),
        full.raw_data_cache["volume"].numpy(),
        full.ts_codes, "buy",
        eligible=full.universe_mask[:, t] & full.universe_mask[:, t + 1],
    )
    sel_minus = engine_minus._select_top_n(
        raw_signal_minus[:, t], t + 1,
        minus.raw_data_cache["open"].numpy(),
        minus.raw_data_cache["high"].numpy(),
        minus.raw_data_cache["low"].numpy(),
        minus.raw_data_cache["pre_close"].numpy(),
        minus.raw_data_cache["volume"].numpy(),
        minus.ts_codes, "buy",
        eligible=minus.universe_mask[:, t] & minus.universe_mask[:, t + 1],
    )
    assert [full.ts_codes[i] for i in sel_full] == [minus.ts_codes[i] for i in sel_minus]
    result_full = engine_full.run(
        raw_signal_full,
        {k: v.numpy() for k, v in full.raw_data_cache.items()},
        full.ts_codes, dates,
        universe_mask=full.universe_mask,
    )
    result_minus = engine_minus.run(
        raw_signal_minus,
        {k: v.numpy() for k, v in minus.raw_data_cache.items()},
        minus.ts_codes, dates,
        universe_mask=minus.universe_mask,
    )
    assert result_full.daily_returns[: join_day - 1] == result_minus.daily_returns[: join_day - 1]
    assert [p["ts_codes"] for p in result_full.positions[: join_day - 1]] == [
        p["ts_codes"] for p in result_minus.positions[: join_day - 1]
    ]

    # --- sim ---------------------------------------------------------------
    runner_full, run_dates = _make_runner(
        tmp_path / "sim_full", n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=[*_SENTINEL_BASE, _FUTURE], bars=bars,
        memberships=memberships, stocks=stocks,
    )
    runner_minus, _ = _make_runner(
        tmp_path / "sim_minus", n_dates=n_dates, top_n=1, max_positions=1,
        ts_codes=_SENTINEL_BASE, bars=bars,
        memberships=memberships, stocks=stocks,
    )
    runner_full.run()
    runner_minus.run()
    for exec_date in run_dates[1 : join_day + 1]:
        assert _orders_for(runner_full, exec_date) == _orders_for(runner_minus, exec_date)


def test_sentinel_differences_allowed_from_join_day(tmp_path):
    """Mid-window join: identical pre-join, and the join day itself may
    change factors, the selected set and the backtest."""

    n_dates = 14
    join_day = 7
    # F's return jumps massively exactly on the join day, so its final
    # signal ranks top the moment it becomes eligible.
    full, minus, dates, _, _, _ = _sentinel_setup(
        tmp_path, join_day=join_day, n_dates=n_dates,
        future_opens=[1e6] * join_day + [2e6] * (n_dates - join_day),
    )
    f_row = _future_row(full)
    base_full = _base_rows(full)
    f_full = full.factor_tensor.numpy()
    f_minus = minus.factor_tensor.numpy()
    # Pre-join: identical factor cross-sections for every eligible stock.
    assert np.allclose(f_full[:, base_full, :join_day], f_minus[:, :, :join_day])
    # From the join day on F enters the reference set: eligible stocks'
    # factor values are allowed to differ.
    assert not np.allclose(f_full[:, base_full, join_day], f_minus[:, :, join_day])

    # The engine cannot hold F before the join day and can hold it from
    # the join day on (F's extreme signal ranks top once eligible).
    bt_cfg = BacktestConfig(top_n=1, single_weight_cap=1.0, train_end_date="2024-01-08")
    raw_signal_full = f_full[FEATURE_NAMES.index("RET_1")]
    raw_signal_minus = f_minus[FEATURE_NAMES.index("RET_1")]
    result_full = AshareBacktestEngine(bt_cfg).run(
        raw_signal_full,
        {k: v.numpy() for k, v in full.raw_data_cache.items()},
        full.ts_codes, dates,
        universe_mask=full.universe_mask,
    )
    result_minus = AshareBacktestEngine(bt_cfg).run(
        raw_signal_minus,
        {k: v.numpy() for k, v in minus.raw_data_cache.items()},
        minus.ts_codes, dates,
        universe_mask=minus.universe_mask,
    )
    assert result_full.daily_returns[:join_day] == result_minus.daily_returns[:join_day]
    assert all(
        _FUTURE not in p["ts_codes"] for p in result_full.positions[:join_day]
    )
    assert _FUTURE in result_full.positions[join_day]["ts_codes"]
    assert result_full.daily_returns[join_day:] != result_minus.daily_returns[join_day:]
