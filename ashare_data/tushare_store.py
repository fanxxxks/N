"""Tushare v2 raw-snapshot store (D2, p17 @ 5d05987).

The L0/L1 write path of the PIT four-layer architecture (p17 sections
4/5/6/13):

* **L0 snapshot** — every ingest batch lands
  ``data/raw/tushare/<endpoint>/<ingest_id>/part-00000.parquet`` FIRST,
  content-addressed by the SHA-256 of the file bytes, before any database
  write (the order is enforced by tests).  The snapshot preserves the full
  source frame (every source column) so nothing is lost upstream of the
  curated L1 tables.  Token, HTTP headers and account information are
  never persisted (p17 §15.10).
* **L1 ledger** — ``source_ingest_runs`` records every batch (succeeded or
  failed) with complete provenance: ingest_id, endpoint, canonical request
  parameters, source field list, start/finish timestamps (Asia/Shanghai
  wall clock), row count, snapshot SHA-256, snapshot path, writing-code
  commit, tushare SDK version, row-limit flag and failure type/message.
* **L1 tables** — 18 data tables with column definitions verbatim from the
  p17 section 5 DDL (erratum t44: daily_suspension restored to the user's
  original definition with ``suspend_timing`` as an additive source column
  and ``available_at``/``source``; industry_taxonomy without ``is_new``;
  ``margin_detail_pit.rqchl``; ``repurchase_event_pit.proc``).  Column
  names are the API's own names (Appendix B) — never renamed silently.
* **Write semantics** — KEYED tables (business key unique) use
  ``ON CONFLICT DO NOTHING`` against the business-key constraint and fail
  the whole batch when a suppressed row's content differs
  (``error_type='row_conflict'``; identical-content rerun is an idempotent
  no-op).  APPEND tables (version/event) key on ``source_row_hash`` with
  an explicit ``ON CONFLICT (source_row_hash) DO NOTHING``: same content
  rerun is idempotent, different content is always a new version row;
  business-key UPDATE/DELETE never happens (p17 §15.4).  Both branches
  pre-deduplicate the batch by ``source_row_hash`` (keep first) because
  DuckDB cannot combine intra-statement duplicate conflict targets with
  ON CONFLICT — the dedup is deterministic and changes no semantics (the
  duplicated row is the same content and adds zero information).
* **Atomicity** — one batch = one transaction: rows, the KEYED conflict
  assertion and the succeeded ledger row commit together.  Any failure
  rolls the batch back completely (zero half-batch residue) and a failed
  ledger row is written in its own transaction (p17 §16).
* **Availability** — tables with ``available_date`` get it computed per
  p17 §6.1 through the injected ``session_resolver`` (the existing
  calendar authority is wired by D3; this module deliberately contains no
  second calendar).  ``daily_suspension.available_at`` is the trade-date
  pre-open instant (09:00 Asia/Shanghai — the suspension decision is fixed
  before the open, so the same-day session is legal per §6.1).

All writes go through the single DuckDB wrapper (:class:`ashare_data.db.
AshareDB`); production ``data/ashare.duckdb`` is never opened by this
store — the v2 database lives at ``data/ashare_v2.duckdb`` and every test
runs on a temporary DuckDB (p17 §8).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .db import AshareDB
from .schemas import TushareIngestResult, TushareSnapshotRecord

#: Schema version of the v2 store (p17 §9: new artifact-facing version,
#: compatibility authority = the v2 dataset manifest which records this
#: value in ``source_versions``).
TUSHARE_STORE_SCHEMA_VERSION = "1"

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_PROVIDER = "tushare"
_DATE_RE = re.compile(r"^\d{8}$")


class TushareStoreError(Exception):
    """Base class of Tushare v2 store failures."""


class TushareRowConflictError(TushareStoreError):
    """A KEYED batch hit an existing business key with different content
    (p17 §5.2: whole batch fails, ``error_type='row_conflict'``)."""


def _error_type(exc: Exception) -> str:
    """Ledger ``error_type``: the contract-mandated token for keyed
    content conflicts, the exception class name otherwise."""

    if isinstance(exc, TushareRowConflictError):
        return "row_conflict"
    return type(exc).__name__


def _now_shanghai() -> datetime:
    """Asia/Shanghai wall clock, naive (the TIMESTAMP column convention)."""

    return datetime.now(_SHANGHAI).replace(tzinfo=None)


def new_ingest_id(
    endpoint: str,
    *,
    now: datetime | None = None,
    suffix: str | None = None,
) -> str:
    """``<endpoint>_<UTC yyyymmddTHHMMSSZ>_<8-hex>`` (p17 §4.1) — globally
    unique, filesystem-safe."""

    stamp = (now or datetime.now(timezone.utc)).astimezone(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    return f"{endpoint}_{stamp}_{suffix or secrets.token_hex(4)}"


def _normalize(value):
    """NaN/NaT/None -> None; everything else passes through unchanged."""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def row_source_hash(values: dict) -> str:
    """SHA-256 hex over the canonical JSON of one row's source content.

    Canonical form: keys sorted, UTF-8, ``ensure_ascii=False``,
    separators without whitespace, NaN/NaN-likes normalized to ``null``
    (``allow_nan=False`` fail-closed).  The hash covers the row's SOURCE
    content only — never ingest metadata (``ingest_id``/timestamps) — so
    an identical-content rerun hashes identically and stays idempotent
    regardless of which batch carried it (p17 §4.3).
    """

    payload = {
        str(key): (
            None if (value is None or (isinstance(value, float) and math.isnan(value)))
            else value
        )
        for key, value in values.items()
    }
    text = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V2Column:
    """One source-mapped column of a v2 table."""

    name: str
    sql_type: str = "DOUBLE"
    nullable: bool = True
    is_date: bool = False  # VARCHAR YYYYMMDD, validated by the writer


@dataclass(frozen=True)
class V2Table:
    """Write contract of one v2 data table (p17 §5 column authority)."""

    name: str
    mode: str  # 'KEYED' | 'APPEND'
    ddl: str
    source_columns: tuple[V2Column, ...]
    business_key: tuple[str, ...] = ()
    #: p17 §6.1 availability rule:
    #:   None                       — no availability column
    #:   ('same_day_preopen',)      — available_at TIMESTAMP + source
    #:   ('next_session', anchors)  — available_date = next open session
    #:                                after the max of the anchor columns
    #:   ('month_end', anchor)      — available_date = next open session
    #:                                after the anchor's month end
    availability: tuple | None = None
    #: When True, one source row is expanded into up to 3 DDL rows (one
    #: per SW level: index_code=l1/l2/l3_code, industry_name=l1/l2/l3_name,
    #: level=1/2/3); null level codes are skipped (erratum #2, t50).
    expand_levels: bool = False


_SOURCE_INGEST_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS source_ingest_runs (
    ingest_id         VARCHAR PRIMARY KEY,
    provider          VARCHAR NOT NULL,
    endpoint          VARCHAR NOT NULL,
    parameters_json   VARCHAR NOT NULL,
    fields_json       VARCHAR NOT NULL,
    started_at        TIMESTAMP NOT NULL,
    finished_at       TIMESTAMP,
    status            VARCHAR NOT NULL,
    row_count         BIGINT  NOT NULL,
    raw_payload_sha256 VARCHAR NOT NULL,
    raw_path          VARCHAR NOT NULL,
    code_commit       VARCHAR NOT NULL,
    sdk_version       VARCHAR NOT NULL,
    hit_row_limit     BOOLEAN NOT NULL,
    error_type        VARCHAR,
    error_message     VARCHAR
)
"""

# DDL verbatim from p17 §5.2-§5.4 (erratum t44 / commit 5d05987).
_DAILY_BAR_RAW_DDL = """
CREATE TABLE IF NOT EXISTS daily_bar_raw (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, pre_close DOUBLE,
    change DOUBLE, pct_chg DOUBLE, vol DOUBLE, amount DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
)
"""
_DAILY_ADJ_FACTOR_DDL = """
CREATE TABLE IF NOT EXISTS daily_adj_factor (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    adj_factor DOUBLE NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
)
"""
_DAILY_BASIC_PIT_DDL = """
CREATE TABLE IF NOT EXISTS daily_basic_pit (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    close DOUBLE, turnover_rate DOUBLE, turnover_rate_f DOUBLE,
    volume_ratio DOUBLE, pe DOUBLE, pe_ttm DOUBLE, pb DOUBLE,
    ps DOUBLE, ps_ttm DOUBLE, dv_ratio DOUBLE, dv_ttm DOUBLE,
    total_share DOUBLE, float_share DOUBLE, free_share DOUBLE,
    total_mv DOUBLE, circ_mv DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
)
"""
_DAILY_PRICE_LIMIT_DDL = """
CREATE TABLE IF NOT EXISTS daily_price_limit (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    pre_close DOUBLE,
    up_limit DOUBLE NOT NULL, down_limit DOUBLE NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
)
"""
_DAILY_SUSPENSION_DDL = """
CREATE TABLE IF NOT EXISTS daily_suspension (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    suspend_type VARCHAR NOT NULL,
    suspend_timing VARCHAR,
    available_at TIMESTAMP NOT NULL,
    source VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date, suspend_type)
)
"""
_INDUSTRY_TAXONOMY_DDL = """
CREATE TABLE IF NOT EXISTS industry_taxonomy (
    index_code VARCHAR PRIMARY KEY,
    industry_name VARCHAR NOT NULL, level VARCHAR NOT NULL,
    industry_code VARCHAR NOT NULL, src VARCHAR,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL
)
"""
_INDUSTRY_MEMBERSHIP_PIT_DDL = """
CREATE TABLE IF NOT EXISTS industry_membership_pit (
    index_code VARCHAR NOT NULL, ts_code VARCHAR NOT NULL,
    level INTEGER NOT NULL,
    in_date VARCHAR NOT NULL,
    out_date VARCHAR, is_new VARCHAR NOT NULL,
    industry_name VARCHAR NOT NULL, available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (index_code, ts_code, in_date)
)
"""
_INCOME_STATEMENT_PIT_DDL = """
CREATE TABLE IF NOT EXISTS income_statement_pit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR NOT NULL, f_ann_date VARCHAR,
    available_date VARCHAR NOT NULL,
    report_type VARCHAR NOT NULL, comp_type VARCHAR NOT NULL,
    update_flag VARCHAR NOT NULL, end_type VARCHAR,
    total_revenue DOUBLE, revenue DOUBLE, oper_cost DOUBLE,
    total_profit DOUBLE, n_income DOUBLE, n_income_attr_p DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_BALANCE_SHEET_PIT_DDL = """
CREATE TABLE IF NOT EXISTS balance_sheet_pit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR NOT NULL, f_ann_date VARCHAR,
    available_date VARCHAR NOT NULL,
    report_type VARCHAR NOT NULL, comp_type VARCHAR NOT NULL,
    update_flag VARCHAR NOT NULL, end_type VARCHAR,
    total_assets DOUBLE, total_liab DOUBLE,
    total_hldr_eqy_exc_min_int DOUBLE, total_hldr_eqy_inc_min_int DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_CASHFLOW_STATEMENT_PIT_DDL = """
CREATE TABLE IF NOT EXISTS cashflow_statement_pit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR NOT NULL, f_ann_date VARCHAR,
    available_date VARCHAR NOT NULL,
    report_type VARCHAR NOT NULL, comp_type VARCHAR NOT NULL,
    update_flag VARCHAR NOT NULL, end_type VARCHAR,
    n_cashflow_act DOUBLE, n_cashflow_inv_act DOUBLE,
    n_cash_flows_fnc_act DOUBLE, c_cash_equ_end_period DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_FINANCIAL_INDICATOR_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS financial_indicator_audit (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    ann_date VARCHAR, f_ann_date VARCHAR, update_flag VARCHAR,
    report_type VARCHAR, eps DOUBLE, roe DOUBLE, roe_waa DOUBLE,
    grossprofit_margin DOUBLE, netprofit_margin DOUBLE,
    debt_to_assets DOUBLE,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_FINANCIAL_DISCLOSURE_SCHEDULE_DDL = """
CREATE TABLE IF NOT EXISTS financial_disclosure_schedule (
    ts_code VARCHAR NOT NULL, end_date VARCHAR NOT NULL,
    pre_date VARCHAR, actual_date VARCHAR,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_EARNINGS_FORECAST_PIT_DDL = """
CREATE TABLE IF NOT EXISTS earnings_forecast_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR NOT NULL, type VARCHAR,
    p_change_min DOUBLE, p_change_max DOUBLE,
    net_profit_min DOUBLE, net_profit_max DOUBLE,
    last_parent_net DOUBLE, summary VARCHAR, change_reason VARCHAR,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_EARNINGS_EXPRESS_PIT_DDL = """
CREATE TABLE IF NOT EXISTS earnings_express_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR NOT NULL,
    revenue DOUBLE, operate_profit DOUBLE, total_profit DOUBLE,
    n_income DOUBLE, total_assets DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_REPURCHASE_EVENT_PIT_DDL = """
CREATE TABLE IF NOT EXISTS repurchase_event_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR, proc VARCHAR, exp_date VARCHAR,
    vol DOUBLE, amount DOUBLE, high_limit DOUBLE, low_limit DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_HOLDER_NUMBER_PIT_DDL = """
CREATE TABLE IF NOT EXISTS holder_number_pit (
    ts_code VARCHAR NOT NULL, ann_date VARCHAR NOT NULL,
    end_date VARCHAR,
    holder_num BIGINT,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (source_row_hash)
)
"""
_MARGIN_DETAIL_PIT_DDL = """
CREATE TABLE IF NOT EXISTS margin_detail_pit (
    ts_code VARCHAR NOT NULL, trade_date VARCHAR NOT NULL,
    rzye DOUBLE, rqye DOUBLE, rzmre DOUBLE, rqyl DOUBLE,
    rzche DOUBLE, rqchl DOUBLE, rqmcl DOUBLE, rzrqye DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (ts_code, trade_date)
)
"""
_INDEX_WEIGHT_PIT_DDL = """
CREATE TABLE IF NOT EXISTS index_weight_pit (
    index_code VARCHAR NOT NULL, con_code VARCHAR NOT NULL,
    trade_date VARCHAR NOT NULL, weight DOUBLE,
    available_date VARCHAR NOT NULL,
    ingest_id VARCHAR NOT NULL, source_row_hash VARCHAR NOT NULL,
    PRIMARY KEY (index_code, con_code, trade_date)
)
"""


def _c(name: str, sql_type: str = "DOUBLE", nullable: bool = True,
       is_date: bool = False) -> V2Column:
    return V2Column(name=name, sql_type=sql_type, nullable=nullable,
                    is_date=is_date)


_V2_TABLE_SPECS: tuple[V2Table, ...] = (
    V2Table(
        name="daily_bar_raw",
        mode="KEYED",
        ddl=_DAILY_BAR_RAW_DDL,
        business_key=("ts_code", "trade_date"),
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("trade_date", "VARCHAR", nullable=False, is_date=True),
            _c("open"), _c("high"), _c("low"), _c("close"), _c("pre_close"),
            _c("change"), _c("pct_chg"), _c("vol"), _c("amount"),
        ),
    ),
    V2Table(
        name="daily_adj_factor",
        mode="KEYED",
        ddl=_DAILY_ADJ_FACTOR_DDL,
        business_key=("ts_code", "trade_date"),
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("trade_date", "VARCHAR", nullable=False, is_date=True),
            _c("adj_factor", nullable=False),
        ),
    ),
    V2Table(
        name="daily_basic_pit",
        mode="KEYED",
        ddl=_DAILY_BASIC_PIT_DDL,
        business_key=("ts_code", "trade_date"),
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("trade_date", "VARCHAR", nullable=False, is_date=True),
            _c("close"), _c("turnover_rate"), _c("turnover_rate_f"),
            _c("volume_ratio"), _c("pe"), _c("pe_ttm"), _c("pb"), _c("ps"),
            _c("ps_ttm"), _c("dv_ratio"), _c("dv_ttm"), _c("total_share"),
            _c("float_share"), _c("free_share"), _c("total_mv"),
            _c("circ_mv"),
        ),
    ),
    V2Table(
        name="daily_price_limit",
        mode="KEYED",
        ddl=_DAILY_PRICE_LIMIT_DDL,
        business_key=("ts_code", "trade_date"),
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("trade_date", "VARCHAR", nullable=False, is_date=True),
            _c("pre_close"),
            _c("up_limit", nullable=False),
            _c("down_limit", nullable=False),
        ),
    ),
    V2Table(
        name="daily_suspension",
        mode="KEYED",
        ddl=_DAILY_SUSPENSION_DDL,
        business_key=("ts_code", "trade_date", "suspend_type"),
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("trade_date", "VARCHAR", nullable=False, is_date=True),
            _c("suspend_type", "VARCHAR", nullable=False),
            _c("suspend_timing", "VARCHAR"),  # additive source column
        ),
        availability=("same_day_preopen",),
    ),
    V2Table(
        name="industry_taxonomy",
        mode="KEYED",
        ddl=_INDUSTRY_TAXONOMY_DDL,
        business_key=("index_code",),
        source_columns=(
            _c("index_code", "VARCHAR", nullable=False),
            _c("industry_name", "VARCHAR", nullable=False),
            _c("level", "VARCHAR", nullable=False),
            _c("industry_code", "VARCHAR", nullable=False),
            _c("src", "VARCHAR"),
        ),
    ),
    V2Table(
        name="industry_membership_pit",
        mode="KEYED",
        ddl=_INDUSTRY_MEMBERSHIP_PIT_DDL,
        business_key=("index_code", "ts_code", "in_date"),
        source_columns=(
            _c("index_code", "VARCHAR", nullable=False),
            _c("ts_code", "VARCHAR", nullable=False),
            _c("level", "BIGINT", nullable=False),
            _c("in_date", "VARCHAR", nullable=False, is_date=True),
            _c("out_date", "VARCHAR", is_date=True),
            _c("is_new", "VARCHAR", nullable=False),
            _c("industry_name", "VARCHAR", nullable=False),
        ),
        availability=("next_session", ("in_date",)),
        expand_levels=True,
    ),
    V2Table(
        name="income_statement_pit",
        mode="APPEND",
        ddl=_INCOME_STATEMENT_PIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("end_date", "VARCHAR", nullable=False, is_date=True),
            _c("ann_date", "VARCHAR", nullable=False, is_date=True),
            _c("f_ann_date", "VARCHAR", is_date=True),
            _c("report_type", "VARCHAR", nullable=False),
            _c("comp_type", "VARCHAR", nullable=False),
            _c("update_flag", "VARCHAR", nullable=False),
            _c("end_type", "VARCHAR"),
            _c("total_revenue"), _c("revenue"), _c("oper_cost"),
            _c("total_profit"), _c("n_income"), _c("n_income_attr_p"),
        ),
        availability=("next_session", ("f_ann_date", "ann_date")),
    ),
    V2Table(
        name="balance_sheet_pit",
        mode="APPEND",
        ddl=_BALANCE_SHEET_PIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("end_date", "VARCHAR", nullable=False, is_date=True),
            _c("ann_date", "VARCHAR", nullable=False, is_date=True),
            _c("f_ann_date", "VARCHAR", is_date=True),
            _c("report_type", "VARCHAR", nullable=False),
            _c("comp_type", "VARCHAR", nullable=False),
            _c("update_flag", "VARCHAR", nullable=False),
            _c("end_type", "VARCHAR"),
            _c("total_assets"), _c("total_liab"),
            _c("total_hldr_eqy_exc_min_int"),
            _c("total_hldr_eqy_inc_min_int"),
        ),
        availability=("next_session", ("f_ann_date", "ann_date")),
    ),
    V2Table(
        name="cashflow_statement_pit",
        mode="APPEND",
        ddl=_CASHFLOW_STATEMENT_PIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("end_date", "VARCHAR", nullable=False, is_date=True),
            _c("ann_date", "VARCHAR", nullable=False, is_date=True),
            _c("f_ann_date", "VARCHAR", is_date=True),
            _c("report_type", "VARCHAR", nullable=False),
            _c("comp_type", "VARCHAR", nullable=False),
            _c("update_flag", "VARCHAR", nullable=False),
            _c("end_type", "VARCHAR"),
            _c("n_cashflow_act"), _c("n_cashflow_inv_act"),
            _c("n_cash_flows_fnc_act"), _c("c_cash_equ_end_period"),
        ),
        availability=("next_session", ("f_ann_date", "ann_date")),
    ),
    V2Table(
        name="financial_indicator_audit",
        mode="APPEND",
        ddl=_FINANCIAL_INDICATOR_AUDIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("end_date", "VARCHAR", nullable=False, is_date=True),
            _c("ann_date", "VARCHAR", is_date=True),
            _c("f_ann_date", "VARCHAR", is_date=True),
            _c("update_flag", "VARCHAR"),
            _c("report_type", "VARCHAR"),
            _c("eps"), _c("roe"), _c("roe_waa"), _c("grossprofit_margin"),
            _c("netprofit_margin"), _c("debt_to_assets"),
        ),
    ),
    V2Table(
        name="financial_disclosure_schedule",
        mode="APPEND",
        ddl=_FINANCIAL_DISCLOSURE_SCHEDULE_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("end_date", "VARCHAR", nullable=False, is_date=True),
            _c("pre_date", "VARCHAR", is_date=True),
            _c("actual_date", "VARCHAR", is_date=True),
        ),
    ),
    V2Table(
        name="earnings_forecast_pit",
        mode="APPEND",
        ddl=_EARNINGS_FORECAST_PIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("ann_date", "VARCHAR", nullable=False, is_date=True),
            _c("end_date", "VARCHAR", nullable=False, is_date=True),
            _c("type", "VARCHAR"),
            _c("p_change_min"), _c("p_change_max"), _c("net_profit_min"),
            _c("net_profit_max"), _c("last_parent_net"), _c("summary", "VARCHAR"),
            _c("change_reason", "VARCHAR"),
        ),
        availability=("next_session", ("ann_date",)),
    ),
    V2Table(
        name="earnings_express_pit",
        mode="APPEND",
        ddl=_EARNINGS_EXPRESS_PIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("ann_date", "VARCHAR", nullable=False, is_date=True),
            _c("end_date", "VARCHAR", nullable=False, is_date=True),
            _c("revenue"), _c("operate_profit"), _c("total_profit"),
            _c("n_income"), _c("total_assets"),
        ),
        availability=("next_session", ("ann_date",)),
    ),
    V2Table(
        name="repurchase_event_pit",
        mode="APPEND",
        ddl=_REPURCHASE_EVENT_PIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("ann_date", "VARCHAR", nullable=False, is_date=True),
            _c("end_date", "VARCHAR", is_date=True),
            _c("proc", "VARCHAR"),
            _c("exp_date", "VARCHAR", is_date=True),
            _c("vol"), _c("amount"), _c("high_limit"), _c("low_limit"),
        ),
        availability=("next_session", ("ann_date",)),
    ),
    V2Table(
        name="holder_number_pit",
        mode="APPEND",
        ddl=_HOLDER_NUMBER_PIT_DDL,
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("ann_date", "VARCHAR", nullable=False, is_date=True),
            _c("end_date", "VARCHAR", is_date=True),  # nullable (erratum #3)
            _c("holder_num", "BIGINT"),
        ),
        availability=("next_session", ("ann_date",)),
    ),
    V2Table(
        name="margin_detail_pit",
        mode="KEYED",
        ddl=_MARGIN_DETAIL_PIT_DDL,
        business_key=("ts_code", "trade_date"),
        source_columns=(
            _c("ts_code", "VARCHAR", nullable=False),
            _c("trade_date", "VARCHAR", nullable=False, is_date=True),
            _c("rzye"), _c("rqye"), _c("rzmre"), _c("rqyl"), _c("rzche"),
            _c("rqchl"), _c("rqmcl"), _c("rzrqye"),
        ),
        availability=("next_session", ("trade_date",)),
    ),
    V2Table(
        name="index_weight_pit",
        mode="KEYED",
        ddl=_INDEX_WEIGHT_PIT_DDL,
        business_key=("index_code", "con_code", "trade_date"),
        source_columns=(
            _c("index_code", "VARCHAR", nullable=False),
            _c("con_code", "VARCHAR", nullable=False),
            _c("trade_date", "VARCHAR", nullable=False, is_date=True),
            _c("weight"),
        ),
        availability=("month_end", ("trade_date",)),
    ),
)

V2_TABLES: dict[str, V2Table] = {spec.name: spec for spec in _V2_TABLE_SPECS}


class TushareStore:
    """Append-only L0 snapshot + L1 ledger/table writer for Tushare data.

    ``session_resolver(date_yyyymmdd) -> date_yyyymmdd`` returns the first
    open session strictly AFTER the given date; it must be wired to the
    existing calendar authority (D3).  ``code_commit``/``sdk_version`` are
    provenance values supplied by the caller.
    """

    def __init__(
        self,
        db: AshareDB,
        raw_root,
        *,
        session_resolver,
        code_commit: str,
        sdk_version: str,
        ingest_id_factory=None,
        clock=_now_shanghai,
    ):
        self.db = db
        self.raw_root = raw_root
        self.session_resolver = session_resolver
        self.code_commit = code_commit
        self.sdk_version = sdk_version
        self._ingest_id_factory = ingest_id_factory or new_ingest_id
        self._clock = clock

    # -- schema ------------------------------------------------------------

    def create_schema(self) -> None:
        """Create the ledger and all 18 data tables (idempotent)."""

        self.db.execute(_SOURCE_INGEST_RUNS_DDL)
        for spec in _V2_TABLE_SPECS:
            self.db.execute(spec.ddl)

    # -- ingest ------------------------------------------------------------

    def ingest_batch(
        self,
        endpoint: str,
        frame: pd.DataFrame,
        *,
        params: dict,
        fields: list[str] | None = None,
        row_limit: int | None = None,
    ) -> TushareIngestResult:
        """Ingest one batch: L0 snapshot first, then one atomic L1
        transaction (rows + conflict assertion + succeeded ledger row).

        Any failure rolls the batch back completely and records a failed
        ledger row before the original exception is re-raised.
        """

        spec = V2_TABLES.get(endpoint)
        if spec is None:
            raise TushareStoreError(
                f"unknown endpoint {endpoint!r}; known: {sorted(V2_TABLES)}"
            )
        ingest_id = self._ingest_id_factory(endpoint)
        started_at = self._clock()
        snapshot: TushareSnapshotRecord | None = None
        try:
            # L0 first (p17 §4.2): the snapshot precedes every database
            # write; a snapshot failure is itself a failed batch and must
            # reach the ledger.
            snapshot = self._write_snapshot(
                endpoint, ingest_id, frame, row_limit=row_limit
            )
            result = self._transact_batch(
                spec, ingest_id, frame, snapshot, params,
                fields if fields is not None else list(frame.columns),
                started_at, row_limit=row_limit,
            )
        except Exception as exc:
            self._record_failure(
                ingest_id, endpoint, exc, snapshot, frame, params,
                list(frame.columns), started_at, row_limit=row_limit,
            )
            raise
        return result

    def snapshot_only(
        self,
        endpoint: str,
        frame: pd.DataFrame,
        *,
        params: dict,
        fields: list[str] | None = None,
        row_limit: int | None = None,
    ) -> TushareIngestResult:
        """L0-only batch for endpoints without a formal v2 table
        (p17 §11.2: ``stock_basic`` / ``trade_cal`` are audit/metadata
        surfaces — full-market frames never enter a formal table, §15.5).

        Writes the raw snapshot and a succeeded ledger row; no L1 table is
        touched.  Ledger-based resume applies identically.
        """

        ingest_id = self._ingest_id_factory(endpoint)
        started_at = self._clock()
        fields_resolved = fields if fields is not None else list(frame.columns)
        snapshot: TushareSnapshotRecord | None = None
        try:
            snapshot = self._write_snapshot(
                endpoint, ingest_id, frame, row_limit=row_limit
            )
            self.db.execute("BEGIN TRANSACTION")
            try:
                self._insert_ledger_succeeded(
                    ingest_id, endpoint, params, fields_resolved,
                    started_at, frame, snapshot, row_limit=row_limit,
                )
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        except Exception as exc:
            self._record_failure(
                ingest_id, endpoint, exc, snapshot, frame, params,
                fields_resolved, started_at, row_limit=row_limit,
            )
            raise
        return TushareIngestResult(
            ingest_id=ingest_id,
            endpoint=endpoint,
            status="succeeded",
            row_count=int(len(frame)),
            raw_path=snapshot.raw_path,
            raw_payload_sha256=snapshot.raw_payload_sha256,
            hit_row_limit=bool(row_limit is not None and len(frame) >= row_limit),
        )

    # -- seams (overridable for order/fault-injection tests) -----------------

    def _write_snapshot(
        self, endpoint: str, ingest_id: str, frame: pd.DataFrame, *,
        row_limit: int | None,
    ) -> TushareSnapshotRecord:
        directory = Path(self.raw_root) / endpoint / ingest_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "part-00000.parquet"
        tmp = directory / "part-00000.tmp.parquet"
        frame.to_parquet(tmp, index=False)
        tmp.replace(target)
        return TushareSnapshotRecord(
            ingest_id=ingest_id,
            endpoint=endpoint,
            raw_path=f"{endpoint}/{ingest_id}/part-00000.parquet",
            raw_payload_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            row_count=int(len(frame)),
            hit_row_limit=bool(row_limit is not None and len(frame) >= row_limit),
        )

    def _transact_batch(
        self,
        spec: V2Table,
        ingest_id: str,
        frame: pd.DataFrame,
        snapshot: TushareSnapshotRecord,
        params: dict,
        fields: list[str],
        started_at: datetime,
        *,
        row_limit: int | None,
    ) -> TushareIngestResult:
        batch = self._map_rows(spec, frame, ingest_id)
        # Intra-batch content deduplication (repair t47): DuckDB cannot
        # combine duplicate conflict-target values within one INSERT ..
        # ON CONFLICT statement, and two rows with the same
        # source_row_hash are by definition the same content (same
        # business key).  Keep-first is deterministic and preserves the
        # idempotent semantics: the duplicated row adds zero information
        # and an identical-content rerun is already a no-op across
        # batches.  Same-key/DIFFERENT-hash rows never collapse — for
        # KEYED they still fail closed via the business-key constraint.
        batch = batch.drop_duplicates(subset=["source_row_hash"], keep="first")
        self.db.execute("BEGIN TRANSACTION")
        try:
            self.db.register("_tushare_batch", batch)
            columns = ", ".join(batch.columns)
            if spec.mode == "APPEND":
                # APPEND tables key on source_row_hash: an identical
                # content rerun (cross-batch or intra-batch) is an
                # idempotent no-op; different content is a new version.
                conflict = " ON CONFLICT (source_row_hash) DO NOTHING"
            else:
                # KEYED tables key on the business key.
                conflict = " ON CONFLICT DO NOTHING"
            self.db.execute(
                f"INSERT INTO {spec.name} ({columns}) "
                f"SELECT {columns} FROM _tushare_batch{conflict}"
            )
            if spec.mode == "KEYED":
                self._assert_no_keyed_conflict(spec, batch)
            self._insert_ledger_succeeded(
                ingest_id, spec.name, params, fields, started_at, frame,
                snapshot, row_limit=row_limit,
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return TushareIngestResult(
            ingest_id=ingest_id,
            endpoint=spec.name,
            status="succeeded",
            row_count=int(len(frame)),
            raw_path=snapshot.raw_path,
            raw_payload_sha256=snapshot.raw_payload_sha256,
            hit_row_limit=bool(row_limit is not None and len(frame) >= row_limit),
        )

    def _insert_ledger_succeeded(
        self,
        ingest_id: str,
        endpoint: str,
        params: dict,
        fields: list[str],
        started_at: datetime,
        frame: pd.DataFrame,
        snapshot: TushareSnapshotRecord,
        *,
        row_limit: int | None,
    ) -> None:
        self.db.execute(
            "INSERT INTO source_ingest_runs VALUES (?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ingest_id,
                _PROVIDER,
                endpoint,
                json.dumps(params, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")),
                json.dumps(sorted(str(c) for c in fields),
                           ensure_ascii=False),
                started_at,
                self._clock(),
                "succeeded",
                int(len(frame)),
                snapshot.raw_payload_sha256,
                snapshot.raw_path,
                self.code_commit,
                self.sdk_version,
                bool(row_limit is not None and len(frame) >= row_limit),
                None,
                None,
            ],
        )

    def _assert_no_keyed_conflict(self, spec: V2Table, batch: pd.DataFrame) -> None:
        """KEYED semantics (p17 §5.2): a suppressed row whose stored
        content hash differs from the batch hash fails the whole batch;
        identical content stays an idempotent no-op."""

        join = " AND ".join(f"t.{col} = b.{col}" for col in spec.business_key)
        mismatches = self.db.query(
            f"SELECT {', '.join(f't.{col}' for col in spec.business_key)} "
            f"FROM {spec.name} t JOIN _tushare_batch b ON {join} "
            "WHERE t.source_row_hash <> b.source_row_hash LIMIT 5"
        )
        if not mismatches.empty:
            keys = "; ".join(
                "/".join(str(v) for v in tuple(row)[1:])
                for row in mismatches.itertuples(index=False)
            )
            raise TushareRowConflictError(
                f"{spec.name}: keyed rows already stored with different "
                f"content ({keys}) — whole batch rejected, zero residue"
            )

    def _record_failure(
        self,
        ingest_id: str,
        endpoint: str,
        exc: Exception,
        snapshot: TushareSnapshotRecord | None,
        frame: pd.DataFrame,
        params: dict,
        fields: list[str],
        started_at: datetime,
        *,
        row_limit: int | None,
    ) -> None:
        """Record a failed batch in its own transaction (the batch itself
        rolled back); the original exception is re-raised by the caller."""

        self.db.execute("BEGIN TRANSACTION")
        try:
            self.db.execute(
                "INSERT INTO source_ingest_runs VALUES (?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    ingest_id,
                    _PROVIDER,
                    endpoint,
                    json.dumps(params, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")),
                    json.dumps(sorted(str(c) for c in fields),
                               ensure_ascii=False),
                    started_at,
                    self._clock(),
                    "failed",
                    int(len(frame)),
                    snapshot.raw_payload_sha256 if snapshot else "",
                    snapshot.raw_path if snapshot else "",
                    self.code_commit,
                    self.sdk_version,
                    bool(
                        snapshot is not None
                        and row_limit is not None
                        and len(frame) >= row_limit
                    ),
                    _error_type(exc),
                    str(exc),
                ],
            )
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    # -- row mapping ---------------------------------------------------------

    def _map_rows(self, spec: V2Table, frame: pd.DataFrame,
                  ingest_id: str) -> pd.DataFrame:
        if len(frame) == 0:
            return pd.DataFrame(
                columns=[c.name for c in spec.source_columns]
                + self._availability_columns(spec)
                + ["ingest_id", "source_row_hash"]
            )
        records: list[dict] = []
        for position, raw_row in enumerate(frame.to_dict("records")):
            if spec.expand_levels:
                records.extend(
                    self._map_expanded_row(spec, raw_row, ingest_id, position)
                )
            else:
                records.append(
                    self._map_row(spec, raw_row, ingest_id, position)
                )
        return pd.DataFrame(records)

    def _map_expanded_row(self, spec: V2Table, raw_row: dict,
                          ingest_id: str, position: int) -> list[dict]:
        """Expand one index_member_all source row into up to 3 DDL rows
        (one per non-null SW level: index_code = l1/l2/l3_code,
        industry_name = l1/l2/l3_name, level = 1/2/3).  Each expanded row
        gets its own source_row_hash because index_code/industry_name
        differ per level — the hash covers the mapped content, not the
        raw source row (erratum #2, t50)."""

        level_defs = [
            (1, "l1_code", "l1_name"),
            (2, "l2_code", "l2_name"),
            (3, "l3_code", "l3_name"),
        ]
        results = []
        for level_num, code_col, name_col in level_defs:
            index_code = _normalize(raw_row.get(code_col))
            industry_name = _normalize(raw_row.get(name_col))
            if not index_code or not industry_name:
                continue
            expanded = dict(raw_row)
            expanded["index_code"] = index_code
            expanded["industry_name"] = industry_name
            expanded["level"] = level_num
            expanded["position"] = position
            results.append(
                self._map_row(spec, expanded, ingest_id, position)
            )
        return results

    def _map_row(self, spec: V2Table, raw_row: dict, ingest_id: str,
                 position: int) -> dict:
        where = f"{spec.name} row {position}"
        mapped: dict[str, object] = {}
        content: dict[str, object] = {}
        for column in spec.source_columns:
            if column.name not in raw_row:
                value = None
            else:
                value = _normalize(raw_row[column.name])
            if value is None and not column.nullable:
                raise TushareStoreError(
                    f"{where}: NOT NULL column {column.name!r} is missing "
                    "or null (fail closed)"
                )
            if (
                value is not None
                and column.is_date
                and not _DATE_RE.match(str(value))
            ):
                raise TushareStoreError(
                    f"{where}: column {column.name!r} must be VARCHAR "
                    f"YYYYMMDD, got {value!r}"
                )
            mapped[column.name] = value
            content[column.name] = value
        mapped.update(self._availability_values(spec, raw_row, where))
        mapped["ingest_id"] = ingest_id
        mapped["source_row_hash"] = row_source_hash(content)
        return mapped

    def _availability_columns(self, spec: V2Table) -> list[str]:
        if spec.availability is None:
            return []
        if spec.availability[0] == "same_day_preopen":
            return ["available_at", "source"]
        return ["available_date"]

    def _availability_values(self, spec: V2Table, raw_row: dict,
                             where: str) -> dict:
        """Compute the availability columns per p17 §6.1 (injected
        resolver; no second calendar lives here)."""

        if spec.availability is None:
            return {}
        kind = spec.availability[0]
        if kind == "same_day_preopen":
            trade_date = _normalize(raw_row.get("trade_date"))
            if not trade_date:
                raise TushareStoreError(
                    f"{where}: trade_date required for available_at"
                )
            preopen = datetime.strptime(str(trade_date), "%Y%m%d").replace(
                hour=9, minute=0
            )
            return {"available_at": preopen, "source": _PROVIDER}
        if kind == "next_session":
            anchors = spec.availability[1]
            candidates = [
                str(_normalize(raw_row.get(anchor)))
                for anchor in anchors
                if _normalize(raw_row.get(anchor)) is not None
            ]
            if not candidates:
                raise TushareStoreError(
                    f"{where}: no availability anchor present "
                    f"({anchors}) — cannot compute available_date"
                )
            anchor = max(candidates)
            return {"available_date": self.session_resolver(anchor)}
        if kind == "month_end":
            (anchor_col,) = spec.availability[1]
            anchor = _normalize(raw_row.get(anchor_col))
            if not anchor:
                raise TushareStoreError(
                    f"{where}: {anchor_col!r} required for month-end "
                    "availability"
                )
            day = pd.Timestamp(str(anchor))
            month_end = (day + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")
            return {"available_date": self.session_resolver(month_end)}
        raise TushareStoreError(  # pragma: no cover - defensive
            f"{where}: unknown availability rule {spec.availability!r}"
        )
