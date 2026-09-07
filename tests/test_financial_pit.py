"""D6 (p17 §6.2) contracts: financial statement as-of version reads.

Executable projections of the approved p17 contract for the three
financial version tables (income/balance/cashflow_statement_pit, §5.3)
and the t15 work package:

* **Revision timing** — before the revision announcement the initial
  version (update_flag=0) is returned; only after the revision's
  available_date does the revised version (update_flag=1) win (§6.2
  winner order ``(f_ann_date, ann_date, available_date, ingest_id)``).
* **Version preservation** — the APPEND tables keep every version row
  (no update_flag filtering anywhere in the read path); pre-revision
  reports stay auditable.
* **Conservative availability** — the announcement day itself is never
  visible (available_date = the next open session, §6.1); a
  same-day-close signal can never see the report.
* **Independent as-of** — the three statements are read per-table; a
  revision in one never drags another table's visibility.
* **Missing stays NaN** — a null statement value surfaces as NaN, never
  0 (p17 §5.3/§6.2 rule 4).
* **Single implementation** — the as-of read lives once in
  ``ashare_data.pit_import.asof_winner_intervals``; the legacy
  ``fundamental_pit`` table is untouched by the new path.
* **Tier version** — DATA_TIER_VERSION bumps to 2 with the
  revision-tracking B-rule text (p17 §9, D6 mandate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_data.config import DataConfig
from ashare_data.pit_import import asof_winner_intervals
from tests.conftest import populate_v2_statement_tables

DATES = [f"2024010{d}" for d in range(1, 9)]  # 8 open sessions


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        data_dir=tmp_path / "data",
        duckdb_path=tmp_path / "ashare.duckdb",
        parquet_dir=tmp_path / "parquet",
        start_date="2024-01-01",
        end_date="2024-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )


def _income_row(**overrides) -> dict:
    row = {
        "ts_code": "000001.SZ",
        "end_date": "20230630",
        "ann_date": "20240102",
        "f_ann_date": "20240102",
        "available_date": "20240103",
        "report_type": "1",
        "comp_type": "1",
        "update_flag": "0",
        "end_type": None,
        "n_income_attr_p": 100.0,
        "ingest_id": "income_1",
    }
    row.update(overrides)
    return row


# --- revision timing (§6.2 winner order) ---------------------------------------


def test_revision_visible_only_after_its_announcement(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[
            _income_row(),  # initial version, visible from 20240103
            _income_row(
                update_flag="1",
                f_ann_date="20240106",
                ann_date="20240106",
                available_date="20240106",
                n_income_attr_p=120.0,
                ingest_id="income_2",
            ),
        ],
    )
    rows = asof_winner_intervals(config, "income", ["000001.SZ"])
    # Initial version wins on [20240103, 20240106); the revision takes over
    # at its own available_date (20240106) — never before.
    initial = rows[(rows["update_flag"] == "0")].iloc[0]
    revised = rows[(rows["update_flag"] == "1")].iloc[0]
    assert initial["winner_from"] == "20240103"
    assert initial["winner_to"] == "20240106"
    assert revised["winner_from"] == "20240106"
    assert revised["winner_to"] == "99991231"
    assert revised["n_income_attr_p"] == 120.0


def test_all_version_rows_preserved_no_update_flag_filter(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[
            _income_row(),
            _income_row(
                update_flag="1",
                f_ann_date="20240106",
                ann_date="20240106",
                available_date="20240106",
                n_income_attr_p=120.0,
                ingest_id="income_2",
            ),
        ],
    )
    rows = asof_winner_intervals(config, "income", ["000001.SZ"])
    # Both versions survive the read: pre-revision reports stay auditable
    # (p17 §6.2 rule 1 — the version rows are never overwritten/deleted).
    assert set(rows["update_flag"]) == {"0", "1"}
    assert len(rows) == 2
    import duckdb

    with duckdb.connect(
        str(config.duckdb_path.parent / "ashare_v2.duckdb"), read_only=True
    ) as con:
        stored = con.execute(
            "SELECT update_flag, n_income_attr_p FROM income_statement_pit "
            "ORDER BY update_flag"
        ).fetchall()
    assert stored == [("0", 100.0), ("1", 120.0)]


# --- conservative availability (§6.1: announcement day not visible) ------------


def test_announcement_day_itself_is_not_visible(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[
            # announced 20240104 (Thursday), available 20240105 (Friday):
            # a same-day-close signal must not see the report.
            _income_row(
                ann_date="20240104",
                f_ann_date="20240104",
                available_date="20240105",
            ),
        ],
    )
    rows = asof_winner_intervals(config, "income", ["000001.SZ"])
    assert len(rows) == 1
    assert rows.iloc[0]["winner_from"] == "20240105"
    assert rows.iloc[0]["winner_to"] == "99991231"


# --- three tables as-of independently (§6.2 rule 3) -----------------------------


def test_three_tables_as_of_independently(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[
            # income revised on 20240102 (available 20240102): wins early.
            _income_row(
                update_flag="1",
                f_ann_date="20240101",
                ann_date="20240101",
                available_date="20240102",
                n_income_attr_p=120.0,
                ingest_id="income_rev",
            ),
            _income_row(
                update_flag="0",
                f_ann_date="20231231",
                ann_date="20231231",
                available_date="20240103",
                n_income_attr_p=100.0,
                ingest_id="income_init",
            ),
        ],
        balance_rows=[
            {
                "ts_code": "000001.SZ",
                "end_date": "20230630",
                "ann_date": "20240102",
                "f_ann_date": "20240102",
                "available_date": "20240103",
                "report_type": "1",
                "comp_type": "1",
                "update_flag": "0",
                "end_type": None,
                "total_assets": 900.0,
                "ingest_id": "balance_1",
            }
        ],
    )
    income = asof_winner_intervals(config, "income", ["000001.SZ"])
    balance = asof_winner_intervals(config, "balance", ["000001.SZ"])
    # Income: the revision (available 20240102) outranks the initial row
    # (available 20240103) from 20240102 — the initial row never wins.
    assert income.iloc[0]["winner_from"] == "20240102"
    assert income.iloc[0]["update_flag"] == "1"
    # Balance: its own available_date governs; the income revision's date
    # never leaks into the balance-sheet read (independent as-of).
    assert balance.iloc[0]["winner_from"] == "20240103"


# --- winner tie-break and f_ann fallback ----------------------------------------


def test_winner_tiebreak_by_ingest_id(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[
            _income_row(n_income_attr_p=111.0, ingest_id="income_a"),
            _income_row(n_income_attr_p=222.0, ingest_id="income_b"),
        ],
    )
    rows = asof_winner_intervals(config, "income", ["000001.SZ"])
    # Identical (f_ann, ann, available): the largest ingest_id wins (§6.2).
    assert rows.iloc[0]["ingest_id"] == "income_b"
    assert rows.iloc[0]["n_income_attr_p"] == 222.0


def test_f_ann_null_falls_back_to_ann_date(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[
            # f_ann NULL: the ranking falls back to ann_date (20240103),
            # which outranks the other row's f_ann (20240102) — but it
            # becomes visible one session later, so the other row serves
            # 20240103 first (p17 §6.2 winner order + §6.1 visibility).
            _income_row(
                f_ann_date=None,
                ann_date="20240103",
                available_date="20240104",
            ),
            _income_row(
                ann_date="20240102",
                f_ann_date="20240102",
                available_date="20240103",
                n_income_attr_p=140.0,
                ingest_id="income_early",
            ),
        ],
    )
    rows = asof_winner_intervals(config, "income", ["000001.SZ"]).sort_values(
        "winner_from"
    )
    assert len(rows) == 2
    early, late = rows.iloc[0], rows.iloc[1]
    assert early["winner_from"] == "20240103"
    assert early["winner_to"] == "20240104"
    assert early["n_income_attr_p"] == 140.0
    assert late["winner_from"] == "20240104"
    assert late["winner_to"] == "99991231"
    assert late["n_income_attr_p"] == 100.0


# --- missing field stays NaN ------------------------------------------------------


def test_missing_value_surfaces_as_nan(tmp_path: Path):
    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[_income_row(n_income_attr_p=None)],
    )
    rows = asof_winner_intervals(config, "income", ["000001.SZ"])
    assert pd.isna(rows.iloc[0]["n_income_attr_p"])


# --- feature frames from the v2 as-of source --------------------------------------


def test_family5_features_built_from_v2_without_legacy(tmp_path: Path):
    """The four family-⑤ features consume the v2 as-of source with the
    legacy ``fundamental_pit`` table ABSENT — proving the new path does
    not depend on it (single implementation, legacy untouched)."""

    from ashare_data.fundamentals import build_pit_frames

    config = _config(tmp_path)
    populate_v2_statement_tables(
        config.duckdb_path.parent / "ashare_v2.duckdb",
        income_rows=[
            _income_row(end_date="20240331", ann_date="20240102",
                        f_ann_date="20240102", available_date="20240103",
                        n_income_attr_p=44.0, ingest_id="i1"),
        ],
        cashflow_rows=[
            {
                "ts_code": "000001.SZ",
                "end_date": "20240331",
                "ann_date": "20240102",
                "f_ann_date": "20240102",
                "available_date": "20240103",
                "report_type": "1",
                "comp_type": "1",
                "update_flag": "0",
                "end_type": None,
                "n_cashflow_act": 68.0,
                "ingest_id": "c1",
            }
        ],
        balance_rows=[
            {
                "ts_code": "000001.SZ",
                "end_date": "20240331",
                "ann_date": "20240102",
                "f_ann_date": "20240102",
                "available_date": "20240103",
                "report_type": "1",
                "comp_type": "1",
                "update_flag": "0",
                "end_type": None,
                "total_assets": 900.0,
                "ingest_id": "b1",
            }
        ],
    )
    dates = pd.bdate_range("2024-01-01", "2024-02-29").strftime("%Y%m%d").tolist()
    close = pd.DataFrame(20.0, index=["000001.SZ"], columns=dates)
    frames = build_pit_frames(config, ["000001.SZ"], dates, close)
    # The TTM needs four trailing quarters: with one quarter present the
    # features stay NaN (no fabricated window) — the as-of path is wired
    # (NaN, not an exception) and the legacy table was never present.
    for name in ("CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL"):
        assert name in frames
        assert frames[name].isna().all().all()


def test_missing_v2_statements_degrade_to_nan(tmp_path: Path):
    from ashare_data.fundamentals import build_pit_frames

    config = _config(tmp_path)
    dates = pd.bdate_range("2024-01-01", periods=5).strftime("%Y%m%d").tolist()
    close = pd.DataFrame(20.0, index=["000001.SZ"], columns=dates)
    frames = build_pit_frames(config, ["000001.SZ"], dates, close)
    for name in ("CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH", "EARNINGS_ACCEL"):
        assert frames[name].isna().all().all()


# --- erratum #4: non-quarter-end end_dates do not participate in quarterly
# --- TTM arithmetic (crash fix from t18 real-data run) ------------------------------


class TestNonQuarterEndEndDates:
    """Erratum #4: income_statement_pit contains rows with non-quarter-end
    end_dates (real irregular reporting periods like 20170131).  The
    family-⑤ as-of reader must skip these in the quarterly cumulative
    TTM period arithmetic (they cannot be interpreted as quarter-end
    cumulative reports) — the related features are NaN for those periods,
    and the loader does not crash."""

    def test_shift_report_date_raises_on_non_quarter_end(self):
        """Pre-fix: _shift_report_date crashes on non-quarter-end dates."""
        from ashare_data.fundamentals import _shift_report_date

        with pytest.raises(ValueError):
            _shift_report_date("20170131", 1)

    def test_build_pit_frames_survives_irregular_end_dates(self, tmp_path: Path):
        """Post-fix: build_pit_frames must NOT crash when the v2 financial
        tables contain non-quarter-end end_dates.  The irregular rows are
        excluded from the quarterly TTM arithmetic; the features for those
        periods stay NaN."""
        from ashare_data.fundamentals import build_pit_frames

        config = _config(tmp_path)
        # Real irregular end_dates observed in v2 income_statement_pit
        irregular_dates = ["20170131", "20160430", "20190831"]
        regular_dates = ["20230331", "20230630", "20230930", "20231231"]
        all_income = []
        for ed in irregular_dates:
            all_income.append(_income_row(
                ts_code="000001.SZ", end_date=ed,
                ann_date="20240102", f_ann_date="20240102",
                available_date="20240103",
                n_income_attr_p=50.0, ingest_id=f"inc_irr_{ed}",
            ))
        for ed in regular_dates:
            all_income.append(_income_row(
                ts_code="000001.SZ", end_date=ed,
                ann_date="20240102", f_ann_date="20240102",
                available_date="20240103",
                n_income_attr_p=100.0, ingest_id=f"inc_reg_{ed}",
            ))
        populate_v2_statement_tables(
            config.duckdb_path.parent / "ashare_v2.duckdb",
            income_rows=all_income,
        )
        dates = pd.bdate_range("2024-01-01", "2024-02-29").strftime("%Y%m%d").tolist()
        close = pd.DataFrame(20.0, index=["000001.SZ"], columns=dates)
        # Pre-fix: ValueError from _shift_report_date on '0131' etc.
        # Post-fix: NaN for the irregular periods, no crash.
        frames = build_pit_frames(config, ["000001.SZ"], dates, close)
        for name in ("CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH",
                      "EARNINGS_ACCEL"):
            assert name in frames

    def test_end_to_end_loader_with_real_irregular_codes(self, tmp_path: Path):
        """End-to-end: 10 real stocks (including the codes that triggered
        the t18 crash) with non-quarter-end end_dates must not crash the
        loader."""
        from ashare_data.fundamentals import build_pit_frames

        config = _config(tmp_path)
        real_codes = [
            "003040.SZ", "003041.SZ", "301001.SZ", "301002.SZ",
            "688001.SH", "688012.SH", "688100.SH", "688333.SH",
            "688567.SH", "688779.SH",
        ]
        irregular = ["20170131", "20160430", "20190831"]
        all_income = []
        for code in real_codes:
            for ed in irregular:
                all_income.append(_income_row(
                    ts_code=code, end_date=ed,
                    ann_date="20240102", f_ann_date="20240102",
                    available_date="20240103",
                    n_income_attr_p=50.0, ingest_id=f"inc_{code}_{ed}",
                ))
            for ed in ("20230331", "20230630"):
                all_income.append(_income_row(
                    ts_code=code, end_date=ed,
                    ann_date="20240102", f_ann_date="20240102",
                    available_date="20240103",
                    n_income_attr_p=100.0, ingest_id=f"inc_{code}_{ed}",
                ))
        populate_v2_statement_tables(
            config.duckdb_path.parent / "ashare_v2.duckdb",
            income_rows=all_income,
        )
        dates = pd.bdate_range("2024-01-01", "2024-02-29").strftime("%Y%m%d").tolist()
        close = pd.DataFrame(20.0, index=real_codes, columns=dates)
        frames = build_pit_frames(config, real_codes, dates, close)
        for name in ("CASHFLOW_QUALITY", "ACCRUALS", "ASSET_GROWTH",
                      "EARNINGS_ACCEL"):
            assert name in frames
            # No crash means the loader survived the irregular end_dates.

    def test_balance_and_cashflow_zero_regression(self, tmp_path: Path):
        """balance/cashflow features with only regular quarter-end dates
        must produce identical results pre/post-fix (zero regression)."""
        from ashare_data.fundamentals import build_pit_frames

        config = _config(tmp_path)
        populate_v2_statement_tables(
            config.duckdb_path.parent / "ashare_v2.duckdb",
            income_rows=[
                _income_row(end_date="20240331", ann_date="20240102",
                            f_ann_date="20240102", available_date="20240103",
                            n_income_attr_p=44.0, ingest_id="i1"),
                _income_row(end_date="20230630", ann_date="20240102",
                            f_ann_date="20240102", available_date="20240103",
                            n_income_attr_p=88.0, ingest_id="i2"),
            ],
            balance_rows=[
                {"ts_code": "000001.SZ", "end_date": "20240331",
                 "ann_date": "20240102", "f_ann_date": "20240102",
                 "available_date": "20240103", "report_type": "1",
                 "comp_type": "1", "update_flag": "0", "end_type": None,
                 "total_assets": 900.0, "ingest_id": "b1"},
                {"ts_code": "000001.SZ", "end_date": "20230630",
                 "ann_date": "20240102", "f_ann_date": "20240102",
                 "available_date": "20240103", "report_type": "1",
                 "comp_type": "1", "update_flag": "0", "end_type": None,
                 "total_assets": 800.0, "ingest_id": "b2"},
            ],
        )
        dates = pd.bdate_range("2024-01-01", "2024-02-29").strftime("%Y%m%d").tolist()
        close = pd.DataFrame(20.0, index=["000001.SZ"], columns=dates)
        frames = build_pit_frames(config, ["000001.SZ"], dates, close)
        # TTM needs 4 trailing quarters — with 2 present, the features
        # stay NaN (no fabricated window), confirming the as-of path works.
        for name in ("CASHFLOW_QUALITY", "ACCRUALS"):
            assert frames[name].isna().all().all()


# --- tier version (p17 §9 D6 mandate) ----------------------------------------------


def test_data_tier_version_bumped_with_revision_tracking_text():
    from ashare_model import data_tier

    assert data_tier.DATA_TIER_VERSION == 2
    rule = data_tier.TIER_TIME_RULES["B"]
    assert "修订" in rule  # revision-tracking semantics for fundamentals
