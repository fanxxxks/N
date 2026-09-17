"""P33 B04 (docs/p33_b04_load_cache_performance_draft.md §4): loader cache
hit-path skip, outcome/reason observability, and the capital-frame date
pushdown parity guards.

Assertion source: the APPROVED p33 contract (user direct instruction
2026-09-16 「完成B04」).  R1-R6 are the contract's RED set; R7 is the
pre-registered golden/parity guard for the behavior-invariant date
pushdown (AGENTS §2.2-3: no meaningful RED exists by design — the
contract §4 note records why a parity guard replaces it).
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd
import pytest

from ashare_data.capital_flow import (
    EXTERNAL_FACTOR_NAMES,
    _INDUSTRY_WINDOW,
    _MARGIN_CROWD_WINDOW,
    _MARGIN_WINDOW,
    build_capital_frames,
)
from ashare_data.config import DataConfig, ModelConfig, resolve_v2_duckdb_path
from ashare_data.db import AshareDB
from ashare_model import data_loader as data_loader_module
from ashare_model.data_loader import AshareDataLoader
from ashare_model.factor_cache import (
    factor_cache_key,
    factor_cache_payload,
)
from ashare_model.load_cache import (
    loader_factor_cache_identity,
    loader_factor_cache_identity_with_reason,
)
from tests.conftest import (
    populate_v2_industry_tables,
    populate_v2_margin_table,
    populate_v2_price_tables,
    publish_manifest,
)


DATES = ["20240102", "20240103", "20240104"]
CODES = ["000001.SZ", "600000.SH"]


# --- identity-source seeding (same production schema as test_load_cache) ----


def _insert_ingest_row(con, ingest_id: str) -> None:
    con.execute(
        "INSERT INTO source_ingest_runs (ingest_id, provider, endpoint, "
        "parameters_json, fields_json, started_at, status, row_count, "
        "raw_payload_sha256, raw_path, code_commit, sdk_version, "
        "hit_row_limit) VALUES (?, 'test', 'test', '{}', '{}', "
        "'2026-01-01 00:00:00', 'succeeded', 0, ?, 'test', 'test', "
        "'test', false)",
        [ingest_id, ingest_id],
    )


def _ensure_marker_table(con, table: str) -> None:
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema='main' AND table_name = ?",
        [table],
    ).fetchone()[0]
    if exists:
        return
    # SQL-injection hardening: DuckDB cannot bind DDL identifiers, so
    # each allowed marker table gets verbatim statements below.
    if table == "industry_membership_pit":
        con.execute(
            "CREATE TABLE industry_membership_pit (available_date VARCHAR)"
        )
        con.execute(
            "INSERT INTO industry_membership_pit VALUES ('20260101')"
        )
    elif table == "income_statement_pit":
        con.execute(
            "CREATE TABLE income_statement_pit (available_date VARCHAR)"
        )
        con.execute("INSERT INTO income_statement_pit VALUES ('20260101')")
    elif table == "balance_sheet_pit":
        con.execute(
            "CREATE TABLE balance_sheet_pit (available_date VARCHAR)"
        )
        con.execute("INSERT INTO balance_sheet_pit VALUES ('20260101')")
    elif table == "cashflow_statement_pit":
        con.execute(
            "CREATE TABLE cashflow_statement_pit (available_date VARCHAR)"
        )
        con.execute("INSERT INTO cashflow_statement_pit VALUES ('20260101')")
    else:
        raise ValueError(f"unknown marker table: {table!r}")


def _seed_identity_sources(cfg, ingest_id: str = "b04-ing-1") -> None:
    v2_path = str(resolve_v2_duckdb_path(cfg))
    con = duckdb.connect(v2_path)
    _insert_ingest_row(con, ingest_id)
    for table in (
        "industry_membership_pit",
        "income_statement_pit",
        "balance_sheet_pit",
        "cashflow_statement_pit",
    ):
        _ensure_marker_table(con, table)
    con.close()
    publish_manifest(cfg)


# --- R1/R2/R5 instrumentation -------------------------------------------------


@pytest.fixture
def frame_call_counts(monkeypatch):
    """Count the expensive frame builds the loader performs (the real
    functions still run — this observes, never mocks behavior)."""

    counts = {"pit": 0, "capital": 0, "factor": 0}
    real_pit = data_loader_module.build_pit_frames
    real_capital = data_loader_module.build_capital_frames
    real_factor = data_loader_module.compute_factor_tensor

    def _pit(*args, **kwargs):
        counts["pit"] += 1
        return real_pit(*args, **kwargs)

    def _capital(*args, **kwargs):
        counts["capital"] += 1
        return real_capital(*args, **kwargs)

    def _factor(*args, **kwargs):
        counts["factor"] += 1
        return real_factor(*args, **kwargs)

    monkeypatch.setattr(data_loader_module, "build_pit_frames", _pit)
    monkeypatch.setattr(data_loader_module, "build_capital_frames", _capital)
    monkeypatch.setattr(data_loader_module, "compute_factor_tensor", _factor)
    return counts


def _snapshot_loader_state(loader) -> dict:
    return {
        "raw": {
            key: value.numpy().copy()
            for key, value in loader.raw_data_cache.items()
        },
        "adjusted": {
            key: value.numpy().copy()
            for key, value in loader.adjusted_data_cache.items()
        },
        "target_ret": loader.target_ret.numpy().copy(),
        "industry_codes": loader.industry_codes.numpy().copy(),
        "universe_mask": np.array(loader.universe_mask, copy=True),
        "price_degraded": dict(loader.price_degraded),
        "dates": list(loader.dates),
        "ts_codes": list(loader.ts_codes),
        "industry_index_lag": loader.industry_index_lag,
        "industry_source_degraded": loader.industry_source_degraded,
    }


def _assert_state_equal(left: dict, right: dict) -> None:
    for key in ("raw", "adjusted"):
        assert set(left[key]) == set(right[key]), key
        for name in left[key]:
            assert np.array_equal(
                left[key][name], right[key][name], equal_nan=True
            ), f"{key}[{name}] differs"
    for key in ("target_ret", "industry_codes", "universe_mask"):
        assert np.array_equal(left[key], right[key], equal_nan=True), key
    for key in (
        "price_degraded",
        "dates",
        "ts_codes",
        "industry_index_lag",
        "industry_source_degraded",
    ):
        assert left[key] == right[key], key


# --- R1: hot load skips the expensive frame builds ----------------------------


def test_r1_hot_load_skips_pit_capital_and_factor_builds(
    populated_db, frame_call_counts
):
    cfg = populated_db
    _seed_identity_sources(cfg)

    cold = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert cold.factor_cache_outcome == "miss"
    assert frame_call_counts == {"pit": 1, "capital": 1, "factor": 1}
    cold_tensor = cold.factor_tensor.numpy().copy()

    hot = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert hot.factor_cache_outcome == "hit"
    # The hit path must not rebuild the expensive frames (F07 fix).
    assert frame_call_counts == {"pit": 1, "capital": 1, "factor": 1}
    assert np.array_equal(
        hot.factor_tensor.numpy(), cold_tensor, equal_nan=True
    )


# --- R2: hit-path artifacts equal the cold path -------------------------------


def test_r2_hit_path_artifacts_equal_cold(populated_db, frame_call_counts):
    cfg = populated_db
    _seed_identity_sources(cfg)

    cold = AshareDataLoader(cfg, ModelConfig()).load_data()
    cold_state = _snapshot_loader_state(cold)

    hot = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert hot.factor_cache_outcome == "hit"
    _assert_state_equal(_snapshot_loader_state(hot), cold_state)


# --- R3: invalidated outcome with reasons --------------------------------------


def test_r3a_corrupt_entry_is_invalidated_not_silent_miss(populated_db):
    cfg = populated_db
    _seed_identity_sources(cfg)

    cold = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert cold.factor_cache_outcome == "miss"
    cold_tensor = cold.factor_tensor.numpy().copy()

    cache_dir = cfg.data_dir / "factor_tensor_cache"
    entry = next(cache_dir.glob("*.npz"))
    entry.write_bytes(b"definitely not an npz payload")

    hot = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert hot.factor_cache_outcome == "invalidated"
    assert hot.factor_cache_reason == "invalidated:unreadable"
    assert np.array_equal(
        hot.factor_tensor.numpy(), cold_tensor, equal_nan=True
    )


def test_r3b_stored_identity_mismatch_is_invalidated(populated_db):
    cfg = populated_db
    _seed_identity_sources(cfg)

    cold = AshareDataLoader(cfg, ModelConfig()).load_data()
    cold_tensor = cold.factor_tensor.numpy().copy()

    identity = loader_factor_cache_identity(
        str(resolve_v2_duckdb_path(cfg)),
        cold.dataset_id,
        cold.dates,
        cold.ts_codes,
    )
    assert identity is not None
    key = factor_cache_key(identity)
    entry = cfg.data_dir / "factor_tensor_cache" / f"{key}.npz"
    assert entry.is_file()
    payload = dict(factor_cache_payload(identity))
    payload["dataset_id"] = "a-different-dataset"
    np.savez(
        entry,
        tensor=np.zeros_like(cold_tensor),
        identity=json.dumps(payload, sort_keys=True),
    )

    hot = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert hot.factor_cache_outcome == "invalidated"
    assert hot.factor_cache_reason == "invalidated:identity_mismatch"
    # The mismatched tensor is never reused: the recompute equals cold.
    assert np.array_equal(
        hot.factor_tensor.numpy(), cold_tensor, equal_nan=True
    )


# --- R4: a dependent-input change cannot false-hit ------------------------------


def test_r4_dependent_input_change_cannot_false_hit(populated_db):
    cfg = populated_db
    _seed_identity_sources(cfg, "b04-ing-1")

    cold = AshareDataLoader(cfg, ModelConfig()).load_data()
    cold_tensor = cold.factor_tensor.numpy().copy()
    cache_dir = cfg.data_dir / "factor_tensor_cache"
    assert len(list(cache_dir.glob("*.npz"))) == 1

    # The v2 store content changes (one more ingest row) and the composite
    # generation is republished — every identity element that the changed
    # input feeds must move, forcing a miss.
    v2_path = str(resolve_v2_duckdb_path(cfg))
    con = duckdb.connect(v2_path)
    _insert_ingest_row(con, "b04-ing-2")
    con.close()
    publish_manifest(cfg)

    fresh = AshareDataLoader(cfg, ModelConfig()).load_data()
    assert fresh.factor_cache_outcome == "miss"
    assert fresh.factor_cache_reason == "miss:no_entry"
    assert np.array_equal(
        fresh.factor_tensor.numpy(), cold_tensor, equal_nan=True
    )
    # The old entry is kept (append-only evidence), a new key was written.
    assert len(list(cache_dir.glob("*.npz"))) == 2


# --- R5: identity derivation precedes the expensive frames ---------------------


def test_r5_identity_derivation_precedes_expensive_frames(
    populated_db, monkeypatch
):
    cfg = populated_db
    _seed_identity_sources(cfg)

    order = []
    real_pit = data_loader_module.build_pit_frames

    def _pit(*args, **kwargs):
        order.append("pit")
        return real_pit(*args, **kwargs)

    monkeypatch.setattr(data_loader_module, "build_pit_frames", _pit)
    for name in (
        "loader_factor_cache_identity",
        "loader_factor_cache_identity_with_reason",
    ):
        real = getattr(data_loader_module, name, None)
        if real is None:
            continue

        def _wrap(*args, __real=real, **kwargs):
            order.append("identity")
            return __real(*args, **kwargs)

        monkeypatch.setattr(data_loader_module, name, _wrap)

    AshareDataLoader(cfg, ModelConfig()).load_data()
    assert "identity" in order and "pit" in order
    assert order.index("identity") < order.index("pit")


# --- R6: the identity reason vocabulary -----------------------------------------

@pytest.fixture
def identity_store(tmp_path):
    """A minimal v2 store carrying every identity-source table."""

    path = tmp_path / "v2.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE source_ingest_runs (ingest_id VARCHAR)")
    con.execute("INSERT INTO source_ingest_runs VALUES ('ing-1'), ('ing-2')")
    con.execute(
        "CREATE TABLE industry_membership_pit (available_date VARCHAR)"
    )
    con.execute("INSERT INTO industry_membership_pit VALUES ('20260821')")
    # SQL-injection hardening: verbatim statements (no dynamic table
    # name in SQL text).
    con.execute("CREATE TABLE income_statement_pit (available_date VARCHAR)")
    con.execute("INSERT INTO income_statement_pit VALUES ('20260630')")
    con.execute("CREATE TABLE balance_sheet_pit (available_date VARCHAR)")
    con.execute("INSERT INTO balance_sheet_pit VALUES ('20260630')")
    con.execute("CREATE TABLE cashflow_statement_pit (available_date VARCHAR)")
    con.execute("INSERT INTO cashflow_statement_pit VALUES ('20260630')")
    con.close()
    return path


def test_r6_identity_reason_vocabulary(tmp_path):
    # ingest source missing entirely
    path = tmp_path / "v2.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE daily_bar_raw (trade_date VARCHAR)")
    con.close()
    identity, reason = loader_factor_cache_identity_with_reason(
        path, "ds-1", DATES, CODES
    )
    assert identity is None
    assert reason == "ingest_source_missing"

    # ingest source present but empty
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE source_ingest_runs (ingest_id VARCHAR)")
    con.close()
    identity, reason = loader_factor_cache_identity_with_reason(
        path, "ds-1", DATES, CODES
    )
    assert identity is None
    assert reason == "ingest_source_empty"

    # marker tables missing
    con = duckdb.connect(str(path))
    con.execute("INSERT INTO source_ingest_runs VALUES ('i1')")
    con.close()
    identity, reason = loader_factor_cache_identity_with_reason(
        path, "ds-1", DATES, CODES
    )
    assert identity is None
    assert reason == "marker_tables_missing"

    # identity derivation failure (unreadable store)
    garbage = tmp_path / "garbage.duckdb"
    garbage.write_bytes(b"definitely not a duckdb file")
    identity, reason = loader_factor_cache_identity_with_reason(
        garbage, "ds-1", DATES, CODES
    )
    assert identity is None
    assert reason == "identity_derivation_failed"

    # mandatory identity inputs missing
    identity, reason = loader_factor_cache_identity_with_reason(
        path, None, DATES, CODES
    )
    assert identity is None
    assert reason == "identity_inputs_missing"

    identity, reason = loader_factor_cache_identity_with_reason(
        path, "ds-1", [], CODES
    )
    assert identity is None
    assert reason == "identity_inputs_missing"


def test_r6b_compat_projection_matches_reasoned_derivation(identity_store):
    identity = loader_factor_cache_identity(identity_store, "ds-1", DATES, CODES)
    reasoned, reason = loader_factor_cache_identity_with_reason(
        identity_store, "ds-1", DATES, CODES
    )
    assert identity is not None
    assert identity == reasoned
    assert reason is None


def test_r6c_loader_records_disabled_reason(populated_db_without_manifest):
    cfg = populated_db_without_manifest
    loader = AshareDataLoader(
        cfg, ModelConfig(), dataset_identity_mode="query"
    ).load_data()
    assert loader.dataset_id is None
    assert loader.factor_cache_outcome == "disabled"
    assert loader.factor_cache_reason == "disabled:identity_inputs_missing"


def test_r6d_loader_records_disabled_reasons_per_cause(populated_db):
    """Dual-review F2: the disabled causes reachable through the loader's
    own config surface are asserted on the loader's factor_cache_reason.
    (identity_derivation_failed stays derivation-function-level — contract
    rev 2: its loader-level trigger, a v2 store that opens for the bar
    read but then fails identity queries, is not constructible without
    mocking the DB layer, which AGENTS §10.1 forbids on core paths.)"""

    cfg = populated_db
    v2_path = str(resolve_v2_duckdb_path(cfg))

    # The published v2 store carries the publication schema with an empty
    # ingest ledger.
    loader = AshareDataLoader(
        cfg, ModelConfig(), dataset_identity_mode="query"
    ).load_data()
    assert loader.dataset_id is not None
    assert loader.factor_cache_reason == "disabled:ingest_source_empty"

    con = duckdb.connect(v2_path)
    con.execute("DROP TABLE source_ingest_runs")
    con.close()
    loader = AshareDataLoader(
        cfg, ModelConfig(), dataset_identity_mode="query"
    ).load_data()
    assert loader.factor_cache_reason == "disabled:ingest_source_missing"

    con = duckdb.connect(v2_path)
    con.execute("CREATE TABLE source_ingest_runs (ingest_id VARCHAR)")
    con.execute("INSERT INTO source_ingest_runs VALUES ('i1')")
    con.execute("DROP TABLE industry_membership_pit")
    con.close()
    loader = AshareDataLoader(
        cfg, ModelConfig(), dataset_identity_mode="query"
    ).load_data()
    assert loader.factor_cache_reason == "disabled:marker_tables_missing"


# --- R7: capital-frame date pushdown parity (behavior-invariant guard) ---------
#
# 24 window sessions: >= _INDUSTRY_WINDOW + 1 so INDUSTRY_MOMENTUM (and
# MARGIN_BALANCE_CHG) produce non-NaN hand-computed cells — the guard must
# have teeth on EVERY pushed-down leg (a 6-session window made the sw leg
# NaN under any outcome, per the dual-review F1 finding).


_WINDOW = [
    date.strftime("%Y%m%d")
    for date in pd.bdate_range("2024-01-02", periods=24)
]
_OUT_PRE = "20231229"
_OUT_POST = "20240207"
_CODES = ["000001.SZ", "600000.SH", "000002.SZ"]
_SW = "801010"


def _margin_rows() -> list[dict]:
    rows = []
    for position, code in enumerate(_CODES):
        for index, session in enumerate(_WINDOW):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": session,
                    "available_date": session,
                    "rzye": 1000.0 * (index + 1) + position,
                    "rzmre": 10.0 * (index + 1) + position,
                }
            )
        # Out-of-window rows: dropped by the window reindex, must never leak.
        rows.append(
            {
                "ts_code": code,
                "trade_date": _OUT_PRE,
                "available_date": _OUT_PRE,
                "rzye": 777.0,
                "rzmre": 7.0,
            }
        )
        rows.append(
            {
                "ts_code": code,
                "trade_date": _OUT_POST,
                "available_date": _OUT_POST,
                "rzye": 888.0,
                "rzmre": 8.0,
            }
        )
    return rows


def _sw_rows(window_only: bool) -> list[dict]:
    # Window-session closes depend on the WINDOW position — identical in
    # the full and the windowed store (the same fixture-asymmetry lesson
    # as _circ_mv); out-of-window rows are exclusion probes with distinct
    # arbitrary values.
    closes = {
        session: 50.0 + 0.1 * index
        for index, session in enumerate(_WINDOW)
    }
    if not window_only:
        closes[_OUT_PRE] = 11.0
        closes[_OUT_POST] = 12.0
    return [
        {
            "index_code": _SW,
            "industry_name": "probe",
            "trade_date": session,
            "close": close,
        }
        for session, close in closes.items()
    ]


def _membership_rows() -> list[dict]:
    return [
        {
            "ts_code": code,
            "index_code": f"{_SW}.SI",
            "level": 1,
            "in_date": "20200101",
            "out_date": "99991231",
            "is_new": "N",
            "industry_name": "probe",
            "available_date": "20200102",
        }
        for code in _CODES
    ]


def _bars_frame():
    # populate_v2_price_tables writes exactly the (_WINDOW x _CODES) cells;
    # out-of-window presence in the full store comes from the margin rows,
    # the direct daily_basic INSERT and the sw-index rows.
    rows = []
    for code in _CODES:
        for index, session in enumerate(_WINDOW):
            close = 10.0 + index
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": session,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "pre_close": close,
                    "volume": 1000.0,
                    "amount": 1000.0 * close,
                }
            )
    return pd.DataFrame(rows)


def _circ_mv() -> dict:
    # Window dates carry the SAME circ_mv in the full and the windowed
    # store — the parity guard compares the two stores, so every value
    # that is not an out-of-window exclusion probe must be identical.
    return {
        (code, session): 10000.0 + index
        for code in _CODES
        for index, session in enumerate(_WINDOW)
    }


def _make_config(tmp_path, name: str) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path / name,
        duckdb_path=tmp_path / name / "ashare.duckdb",
        parquet_dir=tmp_path / name / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


def _populate_capital_fixture(cfg: DataConfig, *, window_only: bool) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.parquet_dir.mkdir(parents=True, exist_ok=True)
    v2_path = resolve_v2_duckdb_path(cfg)
    populate_v2_price_tables(
        v2_path, _WINDOW, _CODES, _bars_frame(),
        circ_mv=_circ_mv(),
    )
    if not window_only:
        # Out-of-window daily_basic rows (the pushdown target): present in
        # the full store, physically absent from the windowed store.
        with AshareDB(v2_path) as db:
            for code in _CODES:
                for session in (_OUT_PRE, _OUT_POST):
                    db.execute(
                        "INSERT INTO daily_basic_pit (ts_code, trade_date, "
                        "turnover_rate, circ_mv) VALUES (?, ?, ?, ?)",
                        [code, session, 1.0, 555.0],
                    )
    populate_v2_margin_table(v2_path, _margin_rows())
    populate_v2_industry_tables(v2_path, _membership_rows())
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_sw_index(_sw_rows(window_only), cfg)


def test_r7_capital_frames_window_pushdown_parity(tmp_path):
    """Golden parity (p33 §4 R7, pre-registered): the frames built from the
    full store (window + out-of-window rows) must equal the frames built
    from a physically windowed store, and the hand-derived contract cells
    must hold.  Green before AND after the pushdown by design — it guards
    the behavior-invariant optimization, it does not drive it."""

    full_cfg = _make_config(tmp_path, "full")
    windowed_cfg = _make_config(tmp_path, "windowed")
    _populate_capital_fixture(full_cfg, window_only=False)
    _populate_capital_fixture(windowed_cfg, window_only=True)

    full = build_capital_frames(full_cfg, _CODES, _WINDOW)
    windowed = build_capital_frames(windowed_cfg, _CODES, _WINDOW)
    assert set(full) == set(EXTERNAL_FACTOR_NAMES)
    for name in EXTERNAL_FACTOR_NAMES:
        assert np.array_equal(
            full[name].to_numpy(dtype=float),
            windowed[name].to_numpy(dtype=float),
            equal_nan=True,
        ), name

    # Hand-derived cells (p24 §2 formulas on the frozen fixture numbers).
    balance = full["MARGIN_BALANCE_TO_CIRC_MV"]
    for position, code in enumerate(_CODES):
        for index, session in enumerate(_WINDOW):
            expected = (1000.0 * (index + 1) + position) / (10000.0 + index)
            actual = balance.loc[code, session]
            assert actual == expected, (code, session)

    pressure = full["MARGIN_BUY_PRESSURE_5"]
    first = _CODES[0]
    # min_periods=5: the first four window columns stay NaN.
    for index in range(4):
        assert np.isnan(pressure.loc[first, _WINDOW[index]])
    expected_p5 = (10.0 + 20.0 + 30.0 + 40.0 + 50.0) / (10000.0 + 4)
    assert pressure.loc[first, _WINDOW[4]] == expected_p5
    expected_p23 = (200.0 + 210.0 + 220.0 + 230.0 + 240.0) / (10000.0 + 23)
    assert pressure.loc[first, _WINDOW[23]] == expected_p23

    # 20-session change: NaN through column 19, hand values from column 20
    # (dual-review F1: the CHG leg carries non-NaN teeth on this window).
    change = full["MARGIN_BALANCE_CHG"]
    for index in range(_MARGIN_WINDOW):
        assert np.isnan(change.loc[first, _WINDOW[index]])
    assert change.loc[first, _WINDOW[20]] == 21000.0 / 1000.0 - 1.0
    assert change.loc[first, _WINDOW[23]] == 24000.0 / 4000.0 - 1.0

    # 60-session crowding stays NaN on a 24-session window.
    assert full["MARGIN_CROWD_60"].isna().to_numpy().all()
    assert _MARGIN_CROWD_WINDOW > len(_WINDOW)
    # The sw pushdown leg's teeth: the window is long enough for the
    # 20-session industry momentum to produce non-NaN hand values.
    assert len(_WINDOW) >= _INDUSTRY_WINDOW + 1

    # 20-session industry momentum from the sw close source: NaN through
    # column 19, hand values from column 20 (dual-review F1: a missing
    # WHERE or a wrong window on the sw pushdown must fail HERE — the
    # source would degrade to an all-NaN momentum frame).
    momentum = full["INDUSTRY_MOMENTUM"]
    for index in range(_INDUSTRY_WINDOW):
        assert np.isnan(momentum.loc[first, _WINDOW[index]])
    # All three codes belong to the single _SW industry in this fixture.
    for code in _CODES:
        expected_m20 = (50.0 + 0.1 * 20) / 50.0 - 1.0
        assert momentum.loc[code, _WINDOW[20]] == expected_m20, code
        expected_m23 = (50.0 + 0.1 * 23) / (50.0 + 0.1 * 3) - 1.0
        assert momentum.loc[code, _WINDOW[23]] == expected_m23, code
