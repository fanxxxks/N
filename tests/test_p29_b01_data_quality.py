"""P29 B01: PIT/tradable data quality — time-spec, as-of, coverage and
the per-family eligibility report (RED first).

Executable projection of ``docs/p29_b01_pit_tradable_data_quality_draft.md``
(§3 invariants I1–I14, §6 RED matrix R1–R16).  Approval: captain per the
user delegation (``reports/alphagpt-ab/00_authority_exception_record.md``
§11 rulings 5+7 + §4ter item 4) plus the user's direct instruction
2026-09-15 「完成B01」.  Every check composes an existing authority
(``build_universe_mask``, the gates constants, ``_financial_coverage``,
``industry_membership_gate``, ``_lag_sessions``) — the tests pin the
four-status semantics, the denominator rule and the gap lists with
HAND-COMPUTED numbers, never values derived from the implementation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import ashare_data.gates as gates_module
from ashare_data.config import DataConfig, resolve_v2_duckdb_path
from ashare_data.db import AshareDB
from ashare_data.pit_import import (
    DAILY_PRICE_FAMILY_TIME_SEMANTICS,
    INDEX_WEIGHT_TIME_SEMANTICS,
    DateAxisFormatError,
    SourceTimeSemantics,
    assert_comparable_date_axes,
    month_end_snapshot_intervals,
    normalize_import_date,
    validate_canonical_date_column,
    validate_time_semantics,
)
from ashare_data.tushare_store import (
    _INDEX_WEIGHT_PIT_DDL,
    _SOURCE_INGEST_RUNS_DDL,
)
from ashare_data.universe import UniversePolicy, expected_observable_units

from conftest import (
    populate_v2_industry_tables,
    populate_v2_margin_table,
    populate_v2_price_tables,
    populate_v2_statement_tables,
)

# -- fixture scenario (hand-computed) -------------------------------------------
#
# Sessions: 20240201..20240208 (6 open sessions).
# A 000001.SZ  listed 20200101, member [20200101, 99991231)  -> 6 expected
# B 600000.SH  listed 20200101, member [20200101, 99991231)  -> 6 expected
# C 688001.SH  listed 20240205, member [20240205, 99991231)  -> 4 expected
#   (STAR board; pre-listing sessions 0201/0202 are NOT in the denominator)
# D 300001.SZ  listed 20200101, member [20240201, 20240206)  -> 3 expected
#   (leaves the universe after 20240205; later sessions are NOT expected)
#
# Default bars: A missing 20240207 (one real gap); B full; C from 0205;
# D to 0205.  => expected units 19, covered 18, coverage 18/19, one gap.

SESSIONS = [
    "20240201", "20240202", "20240205", "20240206", "20240207", "20240208",
]
EVAL_DAY = "20240209"
INDEX = "000300.SH"
A, B, C, D = "000001.SZ", "600000.SH", "688001.SH", "300001.SZ"
ALL_CODES = [A, B, C, D]

STOCKS = [
    {"ts_code": A, "name": A, "industry": None, "list_date": "20200101", "is_st": False},
    {"ts_code": B, "name": B, "industry": None, "list_date": "20200101", "is_st": False},
    {"ts_code": C, "name": C, "industry": None, "list_date": "20240205", "is_st": False},
    {"ts_code": D, "name": D, "industry": None, "list_date": "20200101", "is_st": False},
]
MEMBERSHIP = [
    {"index_code": INDEX, "ts_code": A, "in_date": "20200101", "out_date": "99991231"},
    {"index_code": INDEX, "ts_code": B, "in_date": "20200101", "out_date": "99991231"},
    {"index_code": INDEX, "ts_code": C, "in_date": "20240205", "out_date": "99991231"},
    {"index_code": INDEX, "ts_code": D, "in_date": "20240201", "out_date": "20240206"},
]


def _scenario_bars(*, drop: set[tuple[str, str]] = ()) -> pd.DataFrame:
    """Synthetic bar grid on the fixed SESSIONS axis (hand-shaped)."""

    rows = []
    for code in ALL_CODES:
        for i, date in enumerate(SESSIONS):
            if code == C and date < "20240205":
                continue
            if code == D and date >= "20240206":
                continue
            if (code, date) in drop:
                continue
            close = 10.0 + i * 0.1
            rows.append({
                "ts_code": code, "trade_date": date,
                "open": close * 0.99, "high": close * 1.02,
                "low": close * 0.97, "close": close,
                "pre_close": close * 0.99,
                "volume": 1_000_000.0, "amount": 1_000_000.0 * close,
                "turnover_rate": 1.0, "adj_factor": 1.0,
            })
    return pd.DataFrame(rows)


def _seed_ingest_runs(v2_path: Path, rows: list[dict]) -> None:
    """Explicit ``source_ingest_runs`` rows (I8 evidence fixture)."""

    with AshareDB(v2_path) as db:
        db.execute(_SOURCE_INGEST_RUNS_DDL)
        for row in rows:
            db.execute(
                "INSERT INTO source_ingest_runs VALUES (?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    row["ingest_id"],
                    row.get("provider", "tushare"),
                    row["endpoint"],
                    row.get("parameters_json", "{}"),
                    row.get("fields_json", "[]"),
                    row.get("started_at", "2024-02-09 08:00:00"),
                    row.get("finished_at", "2024-02-09 08:00:01"),
                    row["status"],
                    row.get("row_count", 0),
                    row.get("raw_payload_sha256", "0" * 64),
                    row.get("raw_path", "raw/test"),
                    row.get("code_commit", "tests"),
                    row.get("sdk_version", "tests"),
                    row.get("hit_row_limit", False),
                    row.get("error_type"),
                    row.get("error_message"),
                ],
            )


def _quality_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path,
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=[INDEX],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


_DEFAULT_INDUSTRY = [
    {"ts_code": A, "index_code": "801010.SI", "level": 1,
     "in_date": "20240201", "out_date": "20240206", "is_new": "1",
     "industry_name": "nonglinmu", "available_date": "20240201"},
    {"ts_code": A, "index_code": "801780.SI", "level": 1,
     "in_date": "20240206", "out_date": None, "is_new": "1",
     "industry_name": "bank", "available_date": "20240206"},
    {"ts_code": B, "index_code": "801780.SI", "level": 1,
     "in_date": "20240201", "out_date": None, "is_new": "1",
     "industry_name": "bank", "available_date": "20240201"},
    {"ts_code": C, "index_code": "801780.SI", "level": 1,
     "in_date": "20240205", "out_date": None, "is_new": "1",
     "industry_name": "bank", "available_date": "20240205"},
    {"ts_code": D, "index_code": "801010.SI", "level": 1,
     "in_date": "20240201", "out_date": "20240206", "is_new": "1",
     "industry_name": "nonglinmu", "available_date": "20240201"},
]


def _build_scenario(
    tmp_path: Path,
    *,
    drop_limit_for: set[tuple[str, str]] = (),
    adj_factor: dict | None = None,
    bars_drop: set[tuple[str, str]] = (),
    ingest_rows: list[dict] | None = None,
    margin_rows: list[dict] | None = None,
    industry_membership: list[dict] | None = None,
) -> DataConfig:
    cfg = _quality_config(tmp_path)
    bars = _scenario_bars(drop=bars_drop)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.parquet_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_daily(bars.to_dict("records"), cfg)
        db.upsert_stocks(STOCKS, cfg)
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True} for d in SESSIONS], cfg
        )
        db.upsert_constituents(MEMBERSHIP, cfg)
    v2_path = resolve_v2_duckdb_path(cfg)
    populate_v2_price_tables(
        v2_path, SESSIONS, ALL_CODES, bars,
        drop_limit_for=drop_limit_for,
        adj_factor=adj_factor if adj_factor is not None else 1.0,
    )
    populate_v2_industry_tables(
        v2_path,
        membership_rows=industry_membership
        if industry_membership is not None
        else _DEFAULT_INDUSTRY,
    )
    if margin_rows is not None:
        populate_v2_margin_table(v2_path, margin_rows)
    if ingest_rows:
        _seed_ingest_runs(v2_path, ingest_rows)
    return cfg


def _financial_rows() -> tuple[list[dict], list[dict], list[dict]]:
    """12 fully-covered business keys (A/B/D x 4 quarter ends) per table.

    C has no expected reports: every quarter end precedes its 20240205
    listing (pre-listing exclusion, I6/R1)."""

    periods = ["20230331", "20230630", "20230930", "20231231"]
    income, balance, cashflow = [], [], []
    for code in (A, B, D):
        for period in periods:
            common = {
                "ts_code": code, "end_date": period,
                "ann_date": "20240115", "f_ann_date": "20240115",
                "available_date": "20240116", "report_type": "1",
                "comp_type": "1", "update_flag": "1",
                "end_type": "Q1" if period.endswith("0331") else "Q3",
            }
            income.append({
                **common, "total_revenue": 100.0, "revenue": 100.0,
                "oper_cost": 60.0, "total_profit": 20.0, "n_income": 18.0,
                "n_income_attr_p": 17.0, "ingest_id": "ing-fin",
            })
            balance.append({
                **common, "total_assets": 500.0, "total_liab": 300.0,
                "total_hldr_eqy_exc_min_int": 190.0,
                "total_hldr_eqy_inc_min_int": 200.0, "ingest_id": "ing-fin",
            })
            cashflow.append({
                **common, "n_cashflow_act": 30.0, "n_cashflow_inv_act": -10.0,
                "n_cash_flows_fnc_act": 5.0, "c_cash_equ_end_period": 80.0,
                "ingest_id": "ing-fin",
            })
    return income, balance, cashflow


def _seed_financial(v2_path: Path) -> None:
    income, balance, cashflow = _financial_rows()
    populate_v2_statement_tables(
        v2_path, income_rows=income, balance_rows=balance,
        cashflow_rows=cashflow,
    )


# -- R12 (unit level): import-boundary normalization + axis assertions (I1/I2) --


def test_normalize_import_date_canonicalizes_iso_and_rejects_invalid():
    assert normalize_import_date("2024-01-02") == "20240102"
    assert normalize_import_date("20240102") == "20240102"
    with pytest.raises(DateAxisFormatError):
        normalize_import_date("20240132")


def test_validate_canonical_date_column_lists_offenders_fail_closed():
    # Clean column: passes through unchanged (already canonical).
    values = validate_canonical_date_column(
        ["20240101", "20240102", "20240103"],
        table="daily_bar", field="trade_date",
    )
    assert values == ["20240101", "20240102", "20240103"]
    # ANY non-canonical value fails closed with the offender named — the
    # ISO→compact fix primitive is normalize_import_date, not a silent fix.
    with pytest.raises(DateAxisFormatError) as excinfo:
        validate_canonical_date_column(
            ["20240101", "2024-01-02", "2024-01-03"],
            table="daily_bar", field="trade_date",
        )
    message = str(excinfo.value)
    assert "daily_bar" in message and "trade_date" in message
    assert "2024-01-02" in message  # offenders are named, never silently fixed


def test_assert_comparable_date_axes_rejects_any_non_canonical_axis():
    assert_comparable_date_axes(["20240201", "20240202"], ["20240203"])
    with pytest.raises(DateAxisFormatError):
        assert_comparable_date_axes(
            ["20240201", "20240202"], ["2024-02-03"], field="mask_join"
        )


# -- R5 (I5): month-end snapshot -> daily visible intervals ---------------------


def test_month_end_snapshot_intervals_non_trading_month_end_deferred():
    sessions = [
        "20240102", "20240131", "20240201", "20240229", "20240301",
        "20240628", "20240701",
    ]
    intervals = month_end_snapshot_intervals(
        ["20240131", "20240630"], sessions
    )
    # 20240131 (Wed, open) -> visible the NEXT open session 20240201.
    # 20240630 (Sunday, non-trading month end) -> deferred to 20240701.
    # Half-open [visible_from, next_visible_from); trailing open-ended.
    assert intervals == [
        ("20240131", "20240201", "20240701"),
        ("20240630", "20240701", "99991231"),
    ]


def test_month_end_snapshot_intervals_drops_snapshot_without_following_session():
    sessions = ["20240102", "20240131"]
    assert month_end_snapshot_intervals(["20240131"], sessions) == []
    assert month_end_snapshot_intervals([], sessions) == []


def test_month_end_snapshot_intervals_fail_closed_on_bad_input():
    with pytest.raises(DateAxisFormatError):
        month_end_snapshot_intervals(["2024-01-31"], ["20240201"])
    with pytest.raises(ValueError):
        month_end_snapshot_intervals(["20240229", "20240131"], ["20240301"])


def test_month_end_snapshot_intervals_requires_sorted_unique_sessions():
    """F3 repair pin: open_sessions is the bisect precondition — unordered
    or duplicated sessions must fail closed, never silently mis-interval."""

    with pytest.raises(ValueError):
        month_end_snapshot_intervals(["20240131"], ["20240201", "20240102"])
    with pytest.raises(ValueError):
        month_end_snapshot_intervals(["20240131"], ["20240201", "20240201"])


# -- I3/I4: four-time semantics declarations ------------------------------------


def test_shipped_time_semantics_declarations_validate():
    from ashare_data.capital_flow import (
        INDUSTRY_MEMBERSHIP_TIME_SEMANTICS,
        MARGIN_DETAIL_TIME_SEMANTICS,
    )
    from ashare_data.fundamentals import FINANCIAL_VERSION_TIME_SEMANTICS
    from ashare_model.data_eligibility import EVENTS_AUDIT_TIME_SEMANTICS

    for decl in (
        DAILY_PRICE_FAMILY_TIME_SEMANTICS,
        INDEX_WEIGHT_TIME_SEMANTICS,
        FINANCIAL_VERSION_TIME_SEMANTICS,
        MARGIN_DETAIL_TIME_SEMANTICS,
        INDUSTRY_MEMBERSHIP_TIME_SEMANTICS,
        EVENTS_AUDIT_TIME_SEMANTICS,
    ):
        validate_time_semantics(decl)
        assert decl.collection_authority == "source_ingest_runs"


def test_time_semantics_validation_enforces_i4_tier_limit_note():
    base = SourceTimeSemantics(
        family="x", tables=("t",), economic_time="trade_date",
        availability_time="available_date",
        collection_authority="source_ingest_runs",
        revision_semantics="unrestorable", revision_history_complete=False,
        tier_limit_note="", visibility_rule="rule",
    )
    with pytest.raises(ValueError):
        validate_time_semantics(base)  # unrestorable without a Tier note (I4)
    with pytest.raises(ValueError):
        validate_time_semantics(
            SourceTimeSemantics(
                family="x", tables=("t",), economic_time="trade_date",
                availability_time=None,
                collection_authority="guess",
                revision_semantics="single_final",
                revision_history_complete=True, tier_limit_note="",
                visibility_rule="rule",
            )
        )


# -- R1/R2/R13 (I6): the observable-unit denominator ----------------------------


def _policy() -> UniversePolicy:
    return UniversePolicy(
        index_codes=(INDEX,), min_listed_sessions=1,
        membership_end_inclusive=False,
    )


def test_expected_units_exclude_pre_listing_and_post_membership_r1_r2_r13():
    list_dates = {row["ts_code"]: row["list_date"] for row in STOCKS}
    expected = expected_observable_units(
        ALL_CODES, SESSIONS, SESSIONS, MEMBERSHIP, list_dates, _policy(),
    )
    assert int(expected.sum()) == 19  # hand-computed; cartesian would be 24
    code_pos = {code: position for position, code in enumerate(ALL_CODES)}
    date_pos = {date: position for position, date in enumerate(SESSIONS)}
    # R1: C's pre-listing sessions (0201, 0202) are not observable units.
    assert not expected[code_pos[C], date_pos["20240201"]]
    assert not expected[code_pos[C], date_pos["20240202"]]
    # R2: D's post-out_date sessions (0206..) are not observable units.
    assert not expected[code_pos[D], date_pos["20240206"]]
    assert expected[code_pos[D], date_pos["20240205"]]


def test_expected_units_rejects_mixed_format_axes_i2():
    list_dates = {row["ts_code"]: row["list_date"] for row in STOCKS}
    with pytest.raises(DateAxisFormatError):
        expected_observable_units(
            ALL_CODES, ["20240201", "2024-02-02"], SESSIONS,
            MEMBERSHIP, list_dates, _policy(),
        )


# -- the report: version registration, spaces, thresholds ------------------------


def test_report_version_is_standalone_audit_product():
    """The B01 report is an audit product, not a run-input authority:
    its schema version is standalone (the approved READINESS_REPORT_VERSION
    precedent) and must NOT be spec-bound — binding it would change the
    RunSpec payload shape and force RUNSPEC/ARTIFACT_SCHEMA_VERSION bumps
    for a report no formal entry consumes (p29 §4.2; AGENTS §3.2)."""

    from ashare_model import readiness
    from ashare_model.data_eligibility import DATA_QUALITY_REPORT_VERSION
    from ashare_model.runspec import _VERSION_IMPORTS

    # v2 (F8): the repair round added payload structure (quality
    # suspension_endpoint/ingest_batches_seen, coverage.pre_family_sessions,
    # the events revision sub-item, denominator_rule wording) — the
    # additive-fields-still-bump convention applies to the standalone
    # version too, so v1 and v2 report payloads are distinguishable.
    assert DATA_QUALITY_REPORT_VERSION == 2
    assert "data_quality_report_version" not in _VERSION_IMPORTS
    # The pattern precedent: the A05 readiness report version is likewise
    # standalone (not in the owner index).
    assert "readiness_report_version" not in _VERSION_IMPORTS
    assert readiness.READINESS_REPORT_VERSION == "1"


def test_board_classification_for_gap_lists():
    from ashare_model.data_eligibility import board_of

    assert board_of("688001.SH") == "star"
    assert board_of("689009.SH") == "star"
    assert board_of("300001.SZ") == "chinext"
    assert board_of("600000.SH") == "sse_main"
    assert board_of("000001.SZ") == "szse_main"
    assert board_of("430047.BJ") == "bse"
    assert board_of("XYZ") == "unknown"


def test_report_four_state_vocabulary_and_space_labels(tmp_path):
    from ashare_model import data_eligibility as de
    from ashare_model.data_eligibility import (
        QUALITY_FAMILIES, SPACE_D7, SPACE_REPORT,
    )
    from ashare_model.readiness import (
        STATUS_FAIL, STATUS_NOT_APPLICABLE, STATUS_PASS, STATUS_UNKNOWN,
    )

    assert set(QUALITY_FAMILIES) == {
        de.SOURCE_DAILY, de.SOURCE_PRICE_LIMIT, de.SOURCE_MARGIN,
        de.SOURCE_INDUSTRY_MEMBERSHIP, de.SOURCE_INDUSTRY_INDEX,
        de.SOURCE_FINANCIAL, de.SOURCE_DATASET_MANIFEST,
        de.SOURCE_INDEX_WEIGHT, de.SOURCE_EVENTS_AUDIT,
    }
    cfg = _build_scenario(tmp_path)
    report = de.build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    assert report.report_version == 2  # F8: payload structure bumped at the repair round
    assert report.evaluation_day == EVAL_DAY
    for family, check in report.families.items():
        assert family in QUALITY_FAMILIES
        assert check.status in {
            STATUS_PASS, STATUS_FAIL, STATUS_UNKNOWN, STATUS_NOT_APPLICABLE,
        }
    # I11: p17 families carry the D7 space; informational families do not.
    assert report.spaces[de.SOURCE_DAILY] == SPACE_D7
    assert report.spaces[de.SOURCE_INDEX_WEIGHT] == SPACE_REPORT
    assert report.threshold_sources["daily_max_lag_sessions"] == (
        "ashare_data.gates.DAILY_ELIGIBILITY_MAX_LAG_SESSIONS"
    )


# -- audit mode: coverage denominator, gaps, formats -----------------------------


def test_report_audit_mode_daily_family_r1_r2_r13_r16(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_DAILY, STATUS_PASS, build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path, bars_drop={(A, "20240207")})
    before = (cfg.duckdb_path.stat().st_mtime_ns, cfg.duckdb_path.stat().st_size)
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    after = (cfg.duckdb_path.stat().st_mtime_ns, cfg.duckdb_path.stat().st_size)
    # R16: the report is read-only (the primary database untouched).
    assert before == after

    daily = report.families[SOURCE_DAILY]
    # A missing bar is a listed gap, not automatically a defect (suspensions
    # legitimately have no bar) — the family verdict rides lag/quality.
    assert daily.status == STATUS_PASS
    assert daily.required is False  # audit mode binds no RunSpec
    assert daily.coverage["expected_units"] == 19
    assert daily.coverage["covered_units"] == 18
    assert daily.coverage["ratio"] == pytest.approx(18 / 19)
    assert "listed" in daily.coverage["denominator_rule"]
    assert daily.coverage["gap_count"] == 1
    gap = daily.gaps[0]
    assert (gap.ts_code, gap.date, gap.kind) == (A, "20240207", "missing_bar")


def test_report_price_limit_family_r10_board_gap_list(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_PRICE_LIMIT, STATUS_FAIL, build_data_quality_report,
    )

    cfg = _build_scenario(
        tmp_path, drop_limit_for={(C, "20240206"), (B, "20240207")}
    )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_PRICE_LIMIT]
    assert family.status == STATUS_FAIL  # I7: missing records are defects
    kinds = {
        (gap.ts_code, gap.date, gap.kind, gap.board) for gap in family.gaps
        if gap.kind == "missing_price_limit"
    }
    assert (C, "20240206", "missing_price_limit", "star") in kinds
    assert (B, "20240207", "missing_price_limit", "sse_main") in kinds
    assert family.coverage["gap_count"] == 2


def test_report_daily_invalid_adj_factor_gap_r8(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_DAILY, STATUS_FAIL, build_data_quality_report,
    )

    cfg = _build_scenario(
        tmp_path, adj_factor={(A, "20240202"): 0.0},
        bars_drop={(A, "20240207")},
    )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    daily = report.families[SOURCE_DAILY]
    assert daily.status == STATUS_FAIL
    broken = [g for g in daily.gaps if g.kind == "invalid_adj_factor"]
    assert [(g.ts_code, g.date) for g in broken] == [(A, "20240202")]
    # F7 repair pin: the lag direction keys on the LAG, not the overall
    # status — a fresh family failing on adj-factor defects must not
    # print the reverse-false "lag 0 > 3".
    assert "lag 0 <= 3" in daily.detail


def test_report_mixed_format_audit_i1_r12_and_g_space_note(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_DAILY, STATUS_PASS, build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    # The historic pathology: one ISO-format row in the legacy main daily_bar
    # (same economic day as an existing compact row, distinct string PK).
    with AshareDB(cfg.duckdb_path) as db:
        db.execute(
            "INSERT INTO daily_bar (ts_code, trade_date, open, high, low, "
            "close, pre_close, volume, amount) VALUES (?, ?, ?, ?, ?, ?, ?, "
            "?, ?)",
            [B, "2024-02-08", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    findings = report.mixed_format_findings
    match = [
        f for f in findings
        if f["table"] == "daily_bar" and f["field"] == "trade_date"
    ]
    assert match, "the ISO row must be audited as a mixed-format finding (I1)"
    assert match[0]["count"] == 1
    assert "2024-02-08" in str(match[0]["examples"])
    assert "G1-G8" in str(match[0].get("note", ""))
    # The v2 D7 family axis is clean: the family verdict itself is unaffected,
    # and the finding carries the G-space ownership note (judgement zero-change,
    # contract §2.3 non-goal 2).
    assert report.families[SOURCE_DAILY].status == STATUS_PASS
    assert report.families[SOURCE_DAILY].quality["mixed_format_count"] >= 1


def test_report_fail_closed_on_mixed_calendar_axis_i2(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_DAILY, STATUS_UNKNOWN, build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.execute("INSERT INTO trade_calendar VALUES (?, ?)", ["2024-02-08", True])
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    daily = report.families[SOURCE_DAILY]
    # I2: the coverage comparison runs on the calendar axis — a mixed axis
    # must fail closed to unknown, never compare and never pass.
    assert daily.status == STATUS_UNKNOWN
    assert "format" in daily.detail.lower()


# -- I8: zero-event vs missed collection (event-style families) ------------------


def test_report_zero_event_vs_missed_collection_r3_r4_r9_r15(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_PRICE_LIMIT, STATUS_FAIL, build_data_quality_report,
    )

    cfg = _build_scenario(
        tmp_path,
        ingest_rows=[
            {
                "ingest_id": "ing-susp-0206",
                # The ledger convention: the endpoint column records the V2
                # TABLE name (backfill passes the table to the batch
                # recorder; the store validates it against V2_TABLES).
                "endpoint": "daily_suspension",
                "parameters_json": '{"trade_date": "20240206"}',
                "status": "succeeded",
                "row_count": 0,
            },
            {
                "ingest_id": "ing-susp-0207",
                "endpoint": "daily_suspension",
                "parameters_json": '{"trade_date": "20240207"}',
                "status": "failed",
                "error_type": "ConnectionError",
            },
        ],
    )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_PRICE_LIMIT]
    quality = family.quality
    # R3: a succeeded batch with zero rows is a genuine zero-event day.
    assert "20240206" in quality["zero_event_dates"]
    # R4: a failed batch is a missed collection, never a zero-event.
    assert "20240207" in quality["missed_collection_dates"]
    # R15: the evidence cites the ingest ledger explicitly.
    evidence = {
        item["date"]: (item["ingest_id"], item["status"])
        for item in quality["ingest_evidence"]
    }
    assert evidence["20240206"] == ("ing-susp-0206", "succeeded")
    assert evidence["20240207"] == ("ing-susp-0207", "failed")
    missed = [g for g in family.gaps if g.kind == "missed_collection"]
    assert (None, "20240207") in [(g.ts_code, g.date) for g in missed]
    assert family.status == STATUS_FAIL  # unprovable absence must not pass


def test_suspension_endpoint_naming_matches_the_ledger_convention():
    """F1 repair pin: the ledger's endpoint column records the V2 TABLE
    name (the writer passes the table to the batch recorder and the store
    validates the endpoint against V2_TABLES) — the Tushare API name
    (``suspend_d``) is NOT the key.  A mismatch makes the zero-event
    branch unreachable on the real ledger (the review finding)."""

    from ashare_data.pit_import import DAILY_PRICE_FAMILY_TIME_SEMANTICS
    from ashare_data.tushare_store import V2_TABLES
    from ashare_model.data_eligibility import SUSPENSION_INGEST_ENDPOINT

    assert SUSPENSION_INGEST_ENDPOINT == "daily_suspension"
    assert SUSPENSION_INGEST_ENDPOINT in V2_TABLES
    assert SUSPENSION_INGEST_ENDPOINT in (
        DAILY_PRICE_FAMILY_TIME_SEMANTICS.tables
    )


def test_zero_event_requires_the_ledger_endpoint_name(tmp_path):
    """A batch recorded under a WRONG endpoint name must NOT prove a zero
    event: absence of evidence stays a missed collection (I8, fail-closed
    default — the exact pre-review failure mode)."""

    from ashare_model.data_eligibility import (
        SOURCE_PRICE_LIMIT, build_data_quality_report,
    )

    cfg = _build_scenario(
        tmp_path,
        ingest_rows=[
            {
                "ingest_id": "ing-wrong-name",
                "endpoint": "suspend_d",  # Tushare API name, not the ledger key
                "parameters_json": '{"trade_date": "20240206"}',
                "status": "succeeded",
                "row_count": 0,
            },
        ],
    )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    quality = report.families[SOURCE_PRICE_LIMIT].quality
    assert "20240206" not in quality["zero_event_dates"]
    assert "20240206" in quality["missed_collection_dates"]
    assert quality["suspension_endpoint"] == "daily_suspension"
    assert quality["ingest_batches_seen"] == 0


def test_daily_family_fail_detail_states_the_lag_direction(tmp_path):
    """F2 repair pin: the fail path must print the real direction — a
    stale family must never read 'lag N <= T sessions'."""

    from ashare_model.data_eligibility import (
        SOURCE_DAILY, STATUS_FAIL, build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    with AshareDB(cfg.duckdb_path) as db:
        db.upsert_calendar(
            [{"trade_date": d, "is_open": True}
             for d in ("20240212", "20240213", "20240214", "20240215",
                       "20240216")],
            cfg,
        )
    report = build_data_quality_report(cfg, evaluation_day="20240217")
    daily = report.families[SOURCE_DAILY]
    assert daily.status == STATUS_FAIL
    assert daily.lag["worst_sessions"] == 4
    assert ">" in daily.detail
    assert "<=" not in daily.detail


def test_margin_missing_sessions_fail(tmp_path):
    """F4 repair pin: whole-market margin sessions without rows inside the
    family's active span are defects (no suspension-like excuse) — they
    fail the family and enter the gap list."""

    from ashare_model.data_eligibility import (
        SOURCE_MARGIN, STATUS_FAIL, build_data_quality_report,
    )

    active = [date for date in SESSIONS if date != "20240205"]
    margin_rows = [
        {
            "ts_code": code, "trade_date": date, "rzye": 1.0, "rzmre": 1.0,
            "available_date": date,
        }
        for code in (A,) for date in active
    ]
    cfg = _build_scenario(tmp_path, margin_rows=margin_rows)
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_MARGIN]
    assert family.status == STATUS_FAIL  # the missing session is a defect
    gap_dates = [
        g.date for g in family.gaps if g.kind == "missing_margin_date"
    ]
    assert gap_dates == ["20240205"]
    assert family.coverage["expected_units"] == 6
    assert family.coverage["gap_count"] == 1


def test_margin_pre_family_sessions_are_out_of_the_denominator(tmp_path):
    """F4 companion pin: sessions before the family's first collected
    session are the pre-family era — out of the denominator, never
    fabricated into a coverage failure (mirrors the daily family's
    honest-gap treatment without inventing a zero excuse)."""

    from ashare_model.data_eligibility import (
        SOURCE_MARGIN, STATUS_PASS, build_data_quality_report,
    )

    margin_rows = [
        {
            "ts_code": code, "trade_date": date, "rzye": 1.0, "rzmre": 1.0,
            "available_date": date,
        }
        for code in (A,) for date in ("20240206", "20240207", "20240208")
    ]
    cfg = _build_scenario(tmp_path, margin_rows=margin_rows)
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_MARGIN]
    assert family.status == STATUS_PASS
    assert family.coverage["expected_units"] == 3  # 0201/0202 out (pre-family)
    assert family.coverage["gap_count"] == 0
    assert family.coverage["pre_family_sessions"] == 3
    assert family.coverage["ratio"] == pytest.approx(1.0)


def test_price_limit_gap_examples_are_deterministic(tmp_path):
    """F5 repair pin: gap examples come out in (ts_code, trade_date) order
    regardless of the engine's scan order — reproducible evidence."""

    from ashare_model.data_eligibility import (
        SOURCE_PRICE_LIMIT, build_data_quality_report,
    )

    cfg = _build_scenario(
        tmp_path, drop_limit_for={(B, "20240202"), (A, "20240206")}
    )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    gaps = [
        (g.ts_code, g.date)
        for g in report.families[SOURCE_PRICE_LIMIT].gaps
        if g.kind == "missing_price_limit"
    ]
    assert gaps == [(A, "20240206"), (B, "20240202")]


def test_events_family_declares_four_time_semantics():
    """F6 repair pin: the informational events family carries its I3/I4
    declaration (unrestorable revision history + Tier note)."""

    from ashare_model.data_eligibility import EVENTS_AUDIT_TIME_SEMANTICS

    validate_time_semantics(EVENTS_AUDIT_TIME_SEMANTICS)
    assert EVENTS_AUDIT_TIME_SEMANTICS.revision_semantics == "unrestorable"
    assert not EVENTS_AUDIT_TIME_SEMANTICS.revision_history_complete
    assert EVENTS_AUDIT_TIME_SEMANTICS.tier_limit_note


# -- margin / industry / financial / index-weight / events families --------------


def test_report_margin_family_lag_and_coverage(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_MARGIN, STATUS_PASS, build_data_quality_report,
    )

    margin_rows = [
        {
            "ts_code": code, "trade_date": date,
            "rzye": 1000.0, "rzmre": 10.0,
            "available_date": date,  # writer-computed T+1 (consumed as-is)
        }
        for code in (A, B)
        for date in SESSIONS
    ]
    cfg = _build_scenario(tmp_path, margin_rows=margin_rows)
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_MARGIN]
    assert family.status == STATUS_PASS
    assert family.lag["worst_sessions"] == 0
    assert family.coverage["ratio"] == pytest.approx(1.0)
    assert family.revision["semantics"] == "single_final"


def test_report_margin_threshold_is_consumed_not_copied_i12(
    tmp_path, monkeypatch
):
    from ashare_model.data_eligibility import (
        SOURCE_MARGIN, build_data_quality_report,
    )

    margin_rows = [
        {
            "ts_code": code, "trade_date": date, "rzye": 1.0, "rzmre": 1.0,
            "available_date": date,
        }
        for code in (A,) for date in SESSIONS
    ]
    cfg = _build_scenario(tmp_path, margin_rows=margin_rows)
    monkeypatch.setattr(gates_module, "MARGIN_MAX_LAG_SESSIONS", 0)
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    assert report.families[SOURCE_MARGIN].lag["tolerance"] == 0


def test_report_industry_membership_switch_and_overlap_r7(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_INDUSTRY_MEMBERSHIP, STATUS_FAIL, STATUS_PASS,
        build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_INDUSTRY_MEMBERSHIP]
    # Adjacent half-open intervals: the switch day carries ONE industry.
    assert family.status == STATUS_PASS
    assert family.quality["overlap_violations"] == 0
    assert family.quality["availability_complete"] is True

    overlap_membership = [
        {"ts_code": A, "index_code": "801010.SI", "level": 1,
         "in_date": "20240201", "out_date": "20240206", "is_new": "1",
         "industry_name": "nonglinmu", "available_date": "20240201"},
        {"ts_code": A, "index_code": "801780.SI", "level": 1,
         "in_date": "20240205", "out_date": None, "is_new": "1",
         "industry_name": "bank", "available_date": "20240205"},
    ]
    cfg2 = _build_scenario(
        tmp_path / "overlap", industry_membership=overlap_membership
    )
    report2 = build_data_quality_report(cfg2, evaluation_day=EVAL_DAY)
    family2 = report2.families[SOURCE_INDUSTRY_MEMBERSHIP]
    assert family2.status == STATUS_FAIL
    overlap_gaps = [g for g in family2.gaps if g.kind == "industry_overlap"]
    assert (A, "20240205") in [(g.ts_code, g.date) for g in overlap_gaps]


def test_report_financial_coverage_revision_and_r11_non_quarter_end(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_FINANCIAL, STATUS_PASS, build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    v2_path = resolve_v2_duckdb_path(cfg)
    _seed_financial(v2_path)
    with AshareDB(v2_path) as db:
        # R6: one restated version (same business key, update_flag 2).
        db.execute(
            "INSERT INTO income_statement_pit (ts_code, end_date, ann_date, "
            "f_ann_date, available_date, report_type, comp_type, update_flag, "
            "end_type, total_revenue, revenue, oper_cost, total_profit, "
            "n_income, n_income_attr_p, ingest_id) VALUES (?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [A, "20231231", "20240206", "20240206", "20240207", "1", "1",
             "2", "Q3", 110.0, 110.0, 60.0, 22.0, 19.0, 18.0, "ing-fin-2"],
        )
        # R11: a real irregular non-quarter-end cumulative row.
        db.execute(
            "INSERT INTO income_statement_pit (ts_code, end_date, ann_date, "
            "f_ann_date, available_date, report_type, comp_type, update_flag, "
            "end_type, total_revenue, revenue, oper_cost, total_profit, "
            "n_income, n_income_attr_p, ingest_id) VALUES (?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [B, "20240215", "20240216", "20240216", "20240217", "1", "1",
             "1", "FY", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "ing-fin-3"],
        )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_FINANCIAL]
    assert family.status == STATUS_PASS
    assert family.coverage["ratio"] == pytest.approx(1.0)
    assert family.coverage["expected_units"] == 36  # 3 codes x 4 periods x 3 tables
    # R6: the restated key counts once, and the revision sub-item sees it.
    assert family.revision["versioned_keys"] >= 1
    assert family.revision["semantics"] == "versioned_rows"
    assert family.revision["revision_history_complete"] is False
    assert family.revision["tier_limit_note"]
    # R11: the non-quarter-end row is a quality finding, never a report period.
    assert family.quality["non_quarter_end_rows"] == 1
    assert all(gap.date != "20240215" for gap in family.gaps)


def test_report_index_weight_visibility_rule_violation_i5(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_INDEX_WEIGHT, STATUS_FAIL, build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    with AshareDB(resolve_v2_duckdb_path(cfg)) as db:
        db.execute(_INDEX_WEIGHT_PIT_DDL)
        rows = [
            # Correct I5 rule: month-end 20240131 -> visible 20240201.
            (INDEX, A, "20240131", 0.5, "20240201", "ing-w", "h1"),
            # Violation: visible two sessions late.
            (INDEX, B, "20240131", 0.5, "20240202", "ing-w", "h2"),
        ]
        for row in rows:
            db.execute(
                "INSERT INTO index_weight_pit VALUES (?, ?, ?, ?, ?, ?, ?)",
                list(row),
            )
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_INDEX_WEIGHT]
    assert family.status == STATUS_FAIL
    bad = [g for g in family.gaps if g.kind == "visibility_rule_violation"]
    assert [(g.ts_code, g.date) for g in bad] == [(B, "20240131")]


def test_report_events_audit_family_unknown_when_tables_missing_r14(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_EVENTS_AUDIT, STATUS_UNKNOWN, build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    report = build_data_quality_report(cfg, evaluation_day=EVAL_DAY)
    family = report.families[SOURCE_EVENTS_AUDIT]
    # A missing source is unknown (not computable) — never a pass.
    assert family.status == STATUS_UNKNOWN
    assert "not computable" in family.detail


# -- spec mode: on-demand principle (I10) ----------------------------------------


def test_report_spec_mode_unrequired_families_are_not_applicable_r14(tmp_path):
    from ashare_model.data_eligibility import (
        SOURCE_DAILY, SOURCE_FINANCIAL, SOURCE_INDUSTRY_INDEX,
        SOURCE_INDUSTRY_MEMBERSHIP, SOURCE_MARGIN, SOURCE_PRICE_LIMIT,
        STATUS_NOT_APPLICABLE, STATUS_PASS, DataEligibilityError,
        build_data_quality_report,
    )

    cfg = _build_scenario(tmp_path)
    report = build_data_quality_report(
        cfg, evaluation_day=EVAL_DAY, feature_names=["RET_1"],
    )
    # RET_1 is a PIT_DAILY price feature: only daily + price-limit + the
    # manifest identity are required; everything else is not_applicable
    # with an explicit reason (p17 §12 non-goal: no global freshness gate).
    assert report.families[SOURCE_DAILY].required is True
    assert report.families[SOURCE_DAILY].status == STATUS_PASS
    assert report.families[SOURCE_PRICE_LIMIT].required is True
    for family in (
        SOURCE_MARGIN, SOURCE_FINANCIAL,
        SOURCE_INDUSTRY_MEMBERSHIP, SOURCE_INDUSTRY_INDEX,
    ):
        check = report.families[family]
        assert check.status == STATUS_NOT_APPLICABLE
        assert check.required is False
        assert "not required" in check.detail
    with pytest.raises(DataEligibilityError):
        build_data_quality_report(
            cfg, evaluation_day=EVAL_DAY, feature_names=["NOT_A_FEATURE"],
        )
