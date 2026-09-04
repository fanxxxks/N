"""D2 contract tests for the Tushare raw-snapshot store (p17 @ 5d05987).

Behavior contract (task t7 work package + p17 sections 4/5/6/13/16, as
corrected by erratum t44 / commit 5d05987):

* Schema creation is idempotent: all 19 v2 tables (source_ingest_runs +
  18 data tables) exist after repeated ``create_schema`` calls.
* PK / NOT NULL constraints are enforced by the database and asserted
  directly; the writer additionally enforces the ``YYYYMMDD`` date format
  and computes ``source_row_hash`` as SHA-256 over the canonical
  (key-sorted, UTF-8, NaN-free) JSON of the row's source content.
* L0 snapshots (``data/raw/tushare/<endpoint>/<ingest_id>/part-*.parquet``)
  are written BEFORE the database transaction commits; the order is
  asserted.  Snapshot failures leave the database untouched and record a
  failed ledger row.
* Single-batch atomicity: any failure inside the write transaction rolls
  the batch back completely (zero half-batch residue) and records the
  failure in ``source_ingest_runs`` (error_type / error_message).
* KEYED tables suppress conflicting-key rows with identical content
  (idempotent rerun) and fail the whole batch on a same-key/different-hash
  conflict (error_type='row_conflict'); APPEND tables keep every distinct
  content version (no business-key UPDATE/DELETE ever).
* Availability columns are computed by the writer per p17 section 6.1
  through an injected ``next_open_session`` resolver (the existing
  calendar authority is wired by D3; no second calendar lives here).
* Everything runs on a temporary DuckDB (tmp_path) — the production
  database is never opened.

The erratum-corrected contract details asserted here (all from
``5d05987``): daily_suspension restored definition (trade_date /
suspend_type / suspend_timing additive / available_at TIMESTAMP / source),
industry_taxonomy without is_new, margin_detail_pit ``rqchl``,
repurchase_event_pit ``proc``, daily_price_limit keeping ``pre_close``
(typed explicit request supplies it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_data.db import AshareDB
from ashare_data.manifest import build_v2_dataset_manifest
from ashare_data.schemas import TushareIngestResult, TushareSnapshotRecord
from ashare_data.tushare_store import (
    TUSHARE_STORE_SCHEMA_VERSION,
    V2_TABLES,
    TushareRowConflictError,
    TushareStore,
    TushareStoreError,
    new_ingest_id,
    row_source_hash,
)

FAKE_TOKEN = "TOK_SECRET_SHOULD_NEVER_APPEAR"


def _next_session(date: str) -> str:
    """Deterministic injected resolver for tests: the next calendar day,
    rolling over weekends.  Fixture only — the production resolver is the
    existing calendar authority, wired by D3 (no second calendar here)."""

    day = pd.Timestamp(date) + pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day = day + pd.Timedelta(days=1)
    return day.strftime("%Y%m%d")


@pytest.fixture
def store(tmp_path: Path):
    """A v2 store on a temporary DuckDB + temporary raw root."""

    db = AshareDB(tmp_path / "ashare_v2.duckdb")
    st = TushareStore(
        db,
        tmp_path / "raw",
        session_resolver=_next_session,
        code_commit="fc646a1test",
        sdk_version="test-sdk",
    )
    st.create_schema()
    yield st, db
    db.close()


def _frame(endpoint: str, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _daily_rows(n: int = 2) -> list[dict]:
    return [
        {
            "ts_code": f"00000{i}.SZ",
            "trade_date": "20240315",
            "open": 10.0 + i,
            "high": 11.0 + i,
            "low": 9.0 + i,
            "close": 10.5 + i,
            "pre_close": 10.2 + i,
            "change": 0.3 + i,
            "pct_chg": 3.0 + i,
            "vol": 1000.0 + i,
            "amount": 1_000_000.0 + i,
        }
        for i in range(n)
    ]


def _ledger_rows(db: AshareDB) -> pd.DataFrame:
    return db.query("SELECT * FROM source_ingest_runs ORDER BY started_at")


def _table_count(db: AshareDB, table: str) -> int:
    return int(db.query(f"SELECT COUNT(*) AS n FROM {table}").iloc[0]["n"])


class TestSchemaContract:
    def test_schema_version_is_pinned(self):
        assert TUSHARE_STORE_SCHEMA_VERSION == "1"

    def test_all_19_tables_registered(self):
        assert len(V2_TABLES) == 18
        names = {spec.name for spec in V2_TABLES.values()}
        assert "source_ingest_runs" not in names
        assert {
            "daily_bar_raw",
            "daily_adj_factor",
            "daily_basic_pit",
            "daily_price_limit",
            "daily_suspension",
            "industry_taxonomy",
            "industry_membership_pit",
            "income_statement_pit",
            "balance_sheet_pit",
            "cashflow_statement_pit",
            "financial_indicator_audit",
            "financial_disclosure_schedule",
            "earnings_forecast_pit",
            "earnings_express_pit",
            "repurchase_event_pit",
            "holder_number_pit",
            "margin_detail_pit",
            "index_weight_pit",
        } == names

    def test_schema_creation_is_idempotent(self, store):
        st, db = store
        expected = {spec.name for spec in V2_TABLES.values()} | {"source_ingest_runs"}
        st.create_schema()  # second call must be a no-op, not an error
        tables = set(
            db.query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            )["table_name"]
        )
        assert expected <= tables

    def test_pk_constraint_enforced_by_database(self, store):
        st, db = store
        st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", _daily_rows(1)),
                        params={"trade_date": "20240315"},
                        fields=["ts_code", "trade_date"], row_limit=6000)
        with pytest.raises(Exception):
            db.execute(
                "INSERT INTO daily_bar_raw (ts_code, trade_date, open) "
                "VALUES ('000000.SZ', '20240315', 1.0)"
            )

    def test_not_null_constraint_enforced_by_database(self, store):
        st, _ = store
        with pytest.raises(Exception):
            st.ingest_batch(
                "daily_bar_raw",
                _frame("daily_bar_raw", [
                    {**_daily_rows(1)[0], "ts_code": None},
                ]),
                params={"trade_date": "20240315"},
                fields=["ts_code", "trade_date"],
                row_limit=6000,
            )
        assert _table_count(store[1], "daily_bar_raw") == 0


class TestWriterValidation:
    def test_malformed_date_rejected_fail_closed(self, store):
        st, db = store
        rows = [{**_daily_rows(1)[0], "trade_date": "2024-3-15"}]
        with pytest.raises(TushareStoreError, match="YYYYMMDD"):
            st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", rows),
                            params={"trade_date": "2024-3-15"},
                            fields=["ts_code", "trade_date"], row_limit=6000)
        assert _table_count(db, "daily_bar_raw") == 0
        ledger = _ledger_rows(db)
        assert ledger.iloc[0]["status"] == "failed"

    def test_missing_not_null_source_column_rejected(self, store):
        st, db = store
        row = _daily_rows(1)[0]
        del row["ts_code"]
        with pytest.raises(TushareStoreError):
            st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", [row]),
                            params={"trade_date": "20240315"},
                            fields=["trade_date"], row_limit=6000)
        assert _table_count(db, "daily_bar_raw") == 0
        assert _ledger_rows(db).iloc[0]["status"] == "failed"

    def test_unknown_endpoint_rejected(self, store):
        st, _ = store
        with pytest.raises(TushareStoreError, match="unknown endpoint"):
            st.ingest_batch("not_a_table", _frame("x", [{}]),
                            params={}, fields=[], row_limit=None)


class TestContentIdentity:
    def test_source_row_hash_matches_independent_oracle(self):
        row = {"ts_code": "000001.SZ", "trade_date": "20240315",
               "close": 10.5, "vol": None}
        expected = hashlib.sha256(
            json.dumps(
                {"close": 10.5, "trade_date": "20240315", "ts_code": "000001.SZ",
                 "vol": None},
                sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        assert row_source_hash(row) == expected

    def test_row_hash_normalizes_nan_to_null(self):
        row_a = {"ts_code": "000001.SZ", "close": float("nan")}
        row_b = {"ts_code": "000001.SZ", "close": None}
        assert row_source_hash(row_a) == row_source_hash(row_b)

    def test_keyed_idempotent_rerun_is_noop(self, store):
        st, db = store
        frame = _frame("daily_bar_raw", _daily_rows(2))
        st.ingest_batch("daily_bar_raw", frame,
                        params={"trade_date": "20240315"},
                        fields=list(frame.columns), row_limit=6000)
        st.ingest_batch("daily_bar_raw", frame,
                        params={"trade_date": "20240315"},
                        fields=list(frame.columns), row_limit=6000)
        assert _table_count(db, "daily_bar_raw") == 2
        ledger = _ledger_rows(db)
        assert (ledger["status"] == "succeeded").all()
        assert len(ledger) == 2  # two batches, data deduplicated

    def test_append_versions_coexist_without_update(self, store):
        st, db = store
        base = {
            "ts_code": "000001.SZ", "end_date": "20231231",
            "report_type": "1", "comp_type": "1", "end_type": "1",
            "total_revenue": 100.0, "revenue": 90.0, "oper_cost": 60.0,
            "total_profit": 40.0, "n_income": 35.0, "n_income_attr_p": 34.0,
        }
        first = {**base, "ann_date": "20240315", "f_ann_date": "20240315",
                 "update_flag": "0"}
        revised = {**base, "ann_date": "20240420", "f_ann_date": "20240420",
                   "update_flag": "1", "n_income": 36.0}
        st.ingest_batch("income_statement_pit", _frame("income_statement_pit", [first]),
                        params={"ts_code": "000001.SZ"}, fields=list(first), row_limit=3000)
        st.ingest_batch("income_statement_pit", _frame("income_statement_pit", [revised]),
                        params={"ts_code": "000001.SZ"}, fields=list(revised), row_limit=3000)
        rows = db.query(
            "SELECT * FROM income_statement_pit ORDER BY ann_date"
        )
        assert len(rows) == 2  # both versions kept, no overwrite
        assert list(rows["update_flag"]) == ["0", "1"]


class TestAppendIdempotency:
    """Repair t47 (review finding): the APPEND branch must target the
    ``source_row_hash`` conflict explicitly and pre-deduplicate
    intra-batch hash duplicates (DuckDB cannot combine intra-statement
    duplicate conflict targets with ON CONFLICT).  Deduplication is
    deterministic (keep first) and preserves the idempotent semantics."""

    @staticmethod
    def _income_rows() -> list[dict]:
        base = {
            "ts_code": "000001.SZ", "end_date": "20231231",
            "report_type": "1", "comp_type": "1", "end_type": "1",
            "total_revenue": 100.0, "revenue": 90.0, "oper_cost": 60.0,
            "total_profit": 40.0, "n_income": 35.0, "n_income_attr_p": 34.0,
            "ann_date": "20240315", "f_ann_date": "20240315",
            "update_flag": "0",
        }
        second = {**base, "end_date": "20240331", "ann_date": "20240420",
                  "f_ann_date": "20240420", "n_income": 36.0}
        return [base, second]

    def test_append_same_content_batch_rerun_is_idempotent_noop(self, store):
        st, db = store
        frame = _frame("income_statement_pit", self._income_rows())
        st.ingest_batch("income_statement_pit", frame,
                        params={"ts_code": "000001.SZ"},
                        fields=list(frame.columns), row_limit=3000)
        st.ingest_batch("income_statement_pit", frame,
                        params={"ts_code": "000001.SZ"},
                        fields=list(frame.columns), row_limit=3000)
        assert _table_count(db, "income_statement_pit") == 2
        ledger = _ledger_rows(db)
        assert len(ledger) == 2
        assert (ledger["status"] == "succeeded").all()

    def test_append_intra_batch_duplicate_content_deduplicates(self, store):
        st, db = store
        row = self._income_rows()[0]
        frame = _frame("income_statement_pit", [row, dict(row)])
        result = st.ingest_batch("income_statement_pit", frame,
                                 params={"ts_code": "000001.SZ"},
                                 fields=list(frame.columns), row_limit=3000)
        assert result.status == "succeeded"
        assert int(result.row_count) == 2  # raw batch rows
        assert _table_count(db, "income_statement_pit") == 1  # deduplicated

    def test_keyed_same_key_different_hash_fails_whole_batch(self, store):
        st, db = store
        st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", _daily_rows(1)),
                        params={"trade_date": "20240315"},
                        fields=["ts_code", "trade_date"], row_limit=6000)
        conflicting = [{**_daily_rows(1)[0], "close": 999.0}]
        with pytest.raises(TushareRowConflictError):
            st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", conflicting),
                            params={"trade_date": "20240315"},
                            fields=["ts_code", "trade_date"], row_limit=6000)
        assert _table_count(db, "daily_bar_raw") == 1  # original survives
        ledger = _ledger_rows(db)
        assert ledger.iloc[-1]["status"] == "failed"
        assert ledger.iloc[-1]["error_type"] == "row_conflict"
        # The conflicting batch's snapshot is preserved as evidence.
        assert (Path(st.raw_root) / "daily_bar_raw").exists()


class TestAvailability:
    def test_margin_detail_available_next_session_after_trade_date(self, store):
        st, db = store
        row = {
            "ts_code": "000001.SZ", "trade_date": "20240315",
            "rzye": 1.0, "rqye": 2.0, "rzmre": 3.0, "rqyl": 4.0,
            "rzche": 5.0, "rqchl": 6.0, "rqmcl": 7.0, "rzrqye": 8.0,
        }
        st.ingest_batch("margin_detail_pit", _frame("margin_detail_pit", [row]),
                        params={"trade_date": "20240315"},
                        fields=list(row), row_limit=5000)
        stored = db.query("SELECT * FROM margin_detail_pit")
        assert stored.iloc[0]["available_date"] == _next_session("20240315")

    def test_income_available_after_max_of_ann_dates(self, store):
        st, db = store
        row = {
            "ts_code": "000001.SZ", "end_date": "20231231",
            "ann_date": "20240318", "f_ann_date": "20240315",
            "report_type": "1", "comp_type": "1", "update_flag": "0",
            "end_type": "1", "total_revenue": 1.0, "revenue": 1.0,
            "oper_cost": 1.0, "total_profit": 1.0, "n_income": 1.0,
            "n_income_attr_p": 1.0,
        }
        st.ingest_batch("income_statement_pit", _frame("income_statement_pit", [row]),
                        params={"ts_code": "000001.SZ"}, fields=list(row),
                        row_limit=3000)
        stored = db.query("SELECT * FROM income_statement_pit")
        # anchor = max(f_ann_date, ann_date) = 20240318 (§6.1)
        assert stored.iloc[0]["available_date"] == _next_session("20240318")

    def test_suspension_available_at_is_trade_date_preopen(self, store):
        st, db = store
        row = {"ts_code": "000001.SZ", "trade_date": "20240403",
               "suspend_type": "S", "suspend_timing": "盘中停牌"}
        st.ingest_batch("daily_suspension", _frame("daily_suspension", [row]),
                        params={"trade_date": "20240403"},
                        fields=list(row), row_limit=3000)
        stored = db.query("SELECT * FROM daily_suspension")
        record = stored.iloc[0]
        assert str(record["available_at"]).startswith("2024-04-03 09:00")
        assert record["source"] == "tushare"
        assert record["suspend_timing"] == "盘中停牌"

    def test_index_weight_available_after_month_end(self, store):
        st, db = store
        row = {"index_code": "000300.SH", "con_code": "000001.SZ",
               "trade_date": "20240329", "weight": 0.9}
        st.ingest_batch("index_weight_pit", _frame("index_weight_pit", [row]),
                        params={"index_code": "000300.SH"},
                        fields=list(row), row_limit=3000)
        stored = db.query("SELECT * FROM index_weight_pit")
        assert stored.iloc[0]["available_date"] == _next_session("20240331")

    def test_membership_available_after_in_date(self, store):
        st, db = store
        row = {"index_code": "801010.SI", "ts_code": "000001.SZ",
               "in_date": "20220729", "out_date": None, "is_new": "Y",
               "industry_name": "农林牧渔"}
        st.ingest_batch("industry_membership_pit",
                        _frame("industry_membership_pit", [row]),
                        params={"l1_code": "801010.SI"}, fields=list(row),
                        row_limit=2000)
        stored = db.query("SELECT * FROM industry_membership_pit")
        assert stored.iloc[0]["available_date"] == _next_session("20220729")


class TestSnapshotAndAtomicity:
    def test_snapshot_written_before_db_commit(self, store, monkeypatch):
        st, db = store
        order: list[str] = []
        original_snapshot = TushareStore._write_snapshot
        original_transact = TushareStore._transact_batch

        def spy_snapshot(self, *args, **kwargs):
            order.append("snapshot")
            return original_snapshot(self, *args, **kwargs)

        def spy_transact(self, *args, **kwargs):
            order.append("db")
            return original_transact(self, *args, **kwargs)

        monkeypatch.setattr(st, "_write_snapshot", spy_snapshot.__get__(st))
        monkeypatch.setattr(st, "_transact_batch", spy_transact.__get__(st))
        st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", _daily_rows(1)),
                        params={"trade_date": "20240315"},
                        fields=["ts_code", "trade_date"], row_limit=6000)
        assert order == ["snapshot", "db"]

    def test_snapshot_failure_leaves_db_clean_and_ledger_failed(
        self, store, monkeypatch
    ):
        st, db = store

        def boom(self, *args, **kwargs):
            raise RuntimeError("snapshot disk full")

        monkeypatch.setattr(st, "_write_snapshot", boom.__get__(st))
        with pytest.raises(RuntimeError, match="snapshot disk full"):
            st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", _daily_rows(1)),
                            params={"trade_date": "20240315"},
                            fields=["ts_code", "trade_date"], row_limit=6000)
        assert _table_count(db, "daily_bar_raw") == 0
        ledger = _ledger_rows(db)
        assert ledger.iloc[0]["status"] == "failed"
        assert ledger.iloc[0]["error_type"] == "RuntimeError"
        assert "snapshot disk full" in ledger.iloc[0]["error_message"]

    def test_midbatch_failure_rolls_back_zero_residue(self, store, monkeypatch):
        st, db = store
        original = TushareStore._assert_no_keyed_conflict

        def boom(self, *args, **kwargs):
            raise RuntimeError("injected mid-batch failure")

        monkeypatch.setattr(st, "_assert_no_keyed_conflict", boom.__get__(st))
        with pytest.raises(RuntimeError, match="injected"):
            st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", _daily_rows(3)),
                            params={"trade_date": "20240315"},
                            fields=["ts_code", "trade_date"], row_limit=6000)
        assert _table_count(db, "daily_bar_raw") == 0  # zero half-batch residue
        ledger = _ledger_rows(db)
        assert ledger.iloc[0]["status"] == "failed"
        assert "injected mid-batch failure" in ledger.iloc[0]["error_message"]

    def test_snapshot_content_addressed_and_reproducible(self, store):
        st, _ = store
        frame = _frame("daily_bar_raw", _daily_rows(2))
        result = st.ingest_batch("daily_bar_raw", frame,
                                 params={"trade_date": "20240315"},
                                 fields=list(frame.columns), row_limit=6000)
        raw_file = Path(st.raw_root) / result.raw_path
        assert raw_file.exists()
        assert result.raw_payload_sha256 == hashlib.sha256(
            raw_file.read_bytes()
        ).hexdigest()
        restored = pd.read_parquet(raw_file)
        assert len(restored) == 2  # full source fidelity at L0


class TestLedgerContract:
    def test_succeeded_ledger_record_is_complete(self, store):
        st, db = store
        frame = _frame("daily_bar_raw", _daily_rows(2))
        result = st.ingest_batch("daily_bar_raw", frame,
                                 params={"trade_date": "20240315"},
                                 fields=list(frame.columns), row_limit=6000)
        assert isinstance(result, TushareIngestResult)
        assert result.status == "succeeded"
        row = _ledger_rows(db).iloc[0]
        assert row["ingest_id"] == result.ingest_id
        assert row["provider"] == "tushare"
        assert row["endpoint"] == "daily_bar_raw"
        assert json.loads(row["parameters_json"]) == {"trade_date": "20240315"}
        assert set(json.loads(row["fields_json"])) == set(frame.columns)
        assert row["status"] == "succeeded"
        assert int(row["row_count"]) == 2
        assert row["raw_payload_sha256"] == result.raw_payload_sha256
        assert row["code_commit"] == "fc646a1test"
        assert row["sdk_version"] == "test-sdk"
        assert bool(row["hit_row_limit"]) is False
        assert row["error_type"] is None and row["error_message"] is None
        assert pd.notna(row["started_at"]) and pd.notna(row["finished_at"])

    def test_hit_row_limit_flag_recorded(self, store):
        st, db = store
        result = st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", _daily_rows(2)),
                                 params={"trade_date": "20240315"},
                                 fields=["ts_code", "trade_date"], row_limit=2)
        assert result.hit_row_limit is True
        assert bool(_ledger_rows(db).iloc[0]["hit_row_limit"]) is True

    def test_ingest_id_format_and_uniqueness(self, store):
        first = new_ingest_id("daily_bar_raw")
        second = new_ingest_id("daily_bar_raw")
        assert first != second
        assert first.startswith("daily_bar_raw_")
        suffix = first.rsplit("_", 1)[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_no_token_reaches_snapshot_or_ledger(self, store, monkeypatch):
        st, db = store
        monkeypatch.setenv("TUSHARE_TOKEN", FAKE_TOKEN)
        frame = _frame("daily_bar_raw", _daily_rows(1))
        result = st.ingest_batch("daily_bar_raw", frame,
                                 params={"trade_date": "20240315"},
                                 fields=["ts_code"], row_limit=6000)
        raw_bytes = (Path(st.raw_root) / result.raw_path).read_bytes()
        ledger_text = "\n".join(
            str(v) for v in _ledger_rows(db).iloc[0].tolist()
        )
        assert FAKE_TOKEN not in raw_bytes.decode("latin-1")
        assert FAKE_TOKEN not in ledger_text

    def test_empty_batch_succeeds_with_zero_rows(self, store):
        st, db = store
        result = st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", []),
                                 params={"trade_date": "20240315"},
                                 fields=["ts_code"], row_limit=6000)
        assert result.status == "succeeded"
        assert result.row_count == 0
        assert _table_count(db, "daily_bar_raw") == 0
        assert len(_ledger_rows(db)) == 1


class TestSnapshotOnly:
    """L0-only batches for endpoints without a formal v2 table
    (p17 §11.2: stock_basic / trade_cal are audit/metadata surfaces)."""

    def test_snapshot_only_writes_l0_and_ledger_without_formal_table(self, store):
        st, db = store
        frame = _frame("stock_basic", [
            {"ts_code": "000001.SZ", "name": "平安银行", "list_status": "L"},
            {"ts_code": "000002.SZ", "name": "万科A", "list_status": "L"},
        ])
        result = st.snapshot_only("stock_basic", frame,
                                  params={"list_status": "L"},
                                  fields=list(frame.columns), row_limit=None)
        assert isinstance(result, TushareIngestResult)
        assert result.status == "succeeded"
        assert result.row_count == 2
        raw_file = Path(st.raw_root) / result.raw_path
        assert raw_file.exists()
        assert len(pd.read_parquet(raw_file)) == 2  # full source fidelity
        ledger = _ledger_rows(db)
        assert len(ledger) == 1
        assert ledger.iloc[0]["endpoint"] == "stock_basic"
        assert ledger.iloc[0]["status"] == "succeeded"
        # No formal table touched: the batch endpoint has no v2 table.
        assert "stock_basic" not in V2_TABLES

    def test_snapshot_only_failure_records_failed_ledger(self, store, monkeypatch):
        st, db = store

        def boom(self, *args, **kwargs):
            raise RuntimeError("snapshot fault")

        monkeypatch.setattr(st, "_write_snapshot", boom.__get__(st))
        with pytest.raises(RuntimeError, match="snapshot fault"):
            st.snapshot_only("trade_cal", _frame("trade_cal", []),
                             params={"exchange": "SSE"}, fields=["cal_date"],
                             row_limit=None)
        ledger = _ledger_rows(db)
        assert ledger.iloc[0]["status"] == "failed"
        assert ledger.iloc[0]["error_type"] == "RuntimeError"


class TestAllTablesSmoke:
    """Every data table's spec must map a minimal valid source row end to
    end — this catches DDL/spec drift across all 18 tables at once."""

    def test_every_table_ingests_a_minimal_valid_row(self, store):
        st, db = store
        for spec in V2_TABLES.values():
            row = {}
            for column in spec.source_columns:
                if column.is_date:
                    row[column.name] = "20240315"
                elif column.sql_type == "VARCHAR":
                    row[column.name] = f"v_{column.name}"
                elif column.sql_type == "BIGINT":
                    row[column.name] = 7
                else:
                    row[column.name] = 1.5
            frame = _frame(spec.name, [row])
            result = st.ingest_batch(
                spec.name, frame, params={"smoke": spec.name},
                fields=list(row), row_limit=None,
            )
            assert result.status == "succeeded", spec.name
            assert _table_count(db, spec.name) == 1, spec.name

    def test_daily_suspension_uses_erratum_definition(self):
        spec = V2_TABLES["daily_suspension"]
        names = {c.name for c in spec.source_columns}
        assert {"ts_code", "trade_date", "suspend_type", "suspend_timing"} <= names
        assert "resume_date" not in names
        assert "ann_date" not in names
        assert "resume_date" not in spec.ddl
        assert "ann_date" not in spec.ddl
        assert "suspend_timing" in spec.ddl

    def test_margin_detail_uses_rqchl_and_repurchase_uses_proc(self):
        margin = {c.name for c in V2_TABLES["margin_detail_pit"].source_columns}
        assert "rqchl" in margin and "rqche" not in margin
        rep = {c.name for c in V2_TABLES["repurchase_event_pit"].source_columns}
        assert "proc" in rep and "proc_date" not in rep
        tax = {c.name for c in V2_TABLES["industry_taxonomy"].source_columns}
        assert "is_new" not in tax

    def test_price_limit_keeps_pre_close(self):
        limit = {c.name for c in V2_TABLES["daily_price_limit"].source_columns}
        assert "pre_close" in limit


class TestV2Manifest:
    def test_v2_manifest_is_deterministic_and_content_addressed(self, store):
        st, db = store
        st.ingest_batch("daily_bar_raw", _frame("daily_bar_raw", _daily_rows(2)),
                        params={"trade_date": "20240315"},
                        fields=["ts_code", "trade_date"], row_limit=6000)
        first = build_v2_dataset_manifest(
            db, source_versions={"tushare_store": TUSHARE_STORE_SCHEMA_VERSION}
        )
        second = build_v2_dataset_manifest(
            db, source_versions={"tushare_store": TUSHARE_STORE_SCHEMA_VERSION}
        )
        assert first.dataset_id == second.dataset_id
        st.ingest_batch("daily_adj_factor",
                        _frame("daily_adj_factor", [
                            {"ts_code": "000001.SZ", "trade_date": "20240315",
                             "adj_factor": 1.0}]),
                        params={"trade_date": "20240315"},
                        fields=["ts_code"], row_limit=6000)
        third = build_v2_dataset_manifest(db)
        assert third.dataset_id != first.dataset_id
        assert first.source_versions["tushare_store"] == "1"
