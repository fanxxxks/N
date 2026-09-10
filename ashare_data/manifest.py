"""Immutable dataset manifests: content-addressed ``dataset_id`` values.

Every sync produces a manifest that fingerprints the data **content** — the
schema, per-partition row digests and partition hashes of every data table —
and derives an immutable ``dataset_id`` from it.  The id is content
addressed: identical content always produces the identical id (regardless
of insertion order or run timing), and any schema or row change produces a
different id, so experiments can bind to "which data did this run see"
without relying on prose.

Design contract:

* **Partitions** — tables with a date column are partitioned by calendar
  year (``substr(<key column>, 1, 4)``); tables without one form a single
  ``ALL`` partition.  Each partition contributes one hash.
* **Row digests** — each row is serialized canonically (typed columns cast
  to text with an injective escape, ``~``-joined) and hashed; the partition
  hash is the md5 of the sorted per-row digests, so the hash is independent
  of physical row order inside the table.
* **Caching** — partition hashes are stored in the
  ``dataset_manifest_cache`` table and reused while ``row_count`` and the
  partition's ``MAX(key column)`` are unchanged.  Fast verification
  therefore costs a few cheap aggregate queries instead of re-hashing the
  whole DuckDB; ``verify="full"`` recomputes every hash and catches
  value-only mutations (e.g. a restatement that keeps counts and max dates
  identical).  The cache is an optimization, never a correctness
  short-circuit: it is only consulted for exact (count, max) matches.
* **Persistence** — ``dataset_manifest`` keeps one row per distinct
  dataset_id; saving an id that already exists is a no-op (the same content
  cannot create a duplicate record).
* **Publication** (A02, p26 §3/I7) — ``publish_snapshot`` appends an
  immutable, monotonic *generation* to ``snapshot_publication`` and is the
  sanctioned publication entry: it re-hashes the content uncached (a
  published generation asserts the content it names, so a stale partition
  hash from ``dataset_manifest_cache`` must never be published) and
  records the member library plus the publication-protocol version.
  ``latest_manifest`` resolves through the newest publication generation
  — never through ``created_at`` ordering, whose second-granularity ties
  used to be broken by the content-arbitrary ``dataset_id`` hex (the OB-1
  defect: a same-second stale row could win permanently).  Databases with
  no publications keep the legacy "most recently created row" semantics
  for read-only audit.
* **Binding** — :func:`check_dataset_id` is the explicit migration policy
  for old artifacts: a payload without a dataset_id (pre-T1-01 artifacts)
  is accepted as legacy, while a payload whose id differs from the current
  database is rejected loudly instead of being silently mixed into a run.

The manifest records source/tool versions as *metadata* (``source_versions``)
but they never enter the content hash: the id identifies the data, and the
versions explain how it was produced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .db import AshareDB

# A02 (p26 §4.2, captain ruling 2): 1 -> 2. The record's confirmation
# semantics changed — a manifest may now assert a *content-confirmed*
# dual-database snapshot identity and carry its publication generation,
# where v1 only ever meant "the latest saved id of the primary DB". Old v1
# rows stay readable (``from_dict``) but are never presented as confirmed;
# the two meanings must not be mixed.
MANIFEST_VERSION = "2"
_MANIFEST_TABLE = "dataset_manifest"
_CACHE_TABLE = "dataset_manifest_cache"
_NULL_TOKEN = "@@NULL@@"

# Canonical text escape: '~' is the column separator and '@' escapes itself
# and the separator, making the serialization injective (different rows can
# never serialize to the same string).
_ESCAPE_SQL = (
    f"REPLACE(REPLACE(COALESCE(CAST({{col}} AS VARCHAR), '{_NULL_TOKEN}'), "
    "'~', '@~'), '@', '@@')"
)


class DatasetIdMismatch(ValueError):
    """An artifact records a dataset_id different from the current database."""


def partition_spec(config) -> dict[str, tuple[str | None, str | None]]:
    """``table -> (partition expression, cache-validity key column)``.

    The partition expression maps each row to a partition key string
    (``None`` means the table forms a single ``ALL`` partition).  The key
    column's MAX is tracked for cache invalidation (``None`` means the
    partition is validated by row count only).  The factor cache is a
    derived, disposable table and is deliberately not fingerprinted.

    Accepts the production data config or the Tushare v2 table-name shim
    (:data:`TUSHARE_V2_CONFIG`), in which case the v2 partition spec is
    returned — one manifest machinery for both databases.
    """

    if isinstance(config, _TushareV2TableNames):
        return dict(_V2_PARTITION_SPEC)
    return {
        config.daily_table: ("substr(trade_date, 1, 4)", "trade_date"),
        config.calendar_table: ("substr(trade_date, 1, 4)", "trade_date"),
        config.fundamentals_table: ("substr(report_date, 1, 4)", "report_date"),
        config.margin_table: ("substr(trade_date, 1, 4)", "trade_date"),
        config.sw_index_table: ("substr(trade_date, 1, 4)", "trade_date"),
        config.stocks_table: (None, None),
        config.constituents_table: (None, None),
        config.sw_member_table: (None, None),
    }


class _TushareV2TableNames:
    """Attribute shim exposing the Tushare v2 table names through the
    existing ``partition_spec(config)`` contract, so the v2 dataset
    manifest reuses the SAME fingerprint machinery as the production
    manifest (p17 §4.4: dataset_id recomputed from content by the
    existing manifest authority; p13 precedent — no second hashing
    implementation)."""

    source_ingest_runs = "source_ingest_runs"
    daily_bar_raw = "daily_bar_raw"
    daily_adj_factor = "daily_adj_factor"
    daily_basic_pit = "daily_basic_pit"
    daily_price_limit = "daily_price_limit"
    daily_suspension = "daily_suspension"
    industry_taxonomy = "industry_taxonomy"
    industry_membership_pit = "industry_membership_pit"
    income_statement_pit = "income_statement_pit"
    balance_sheet_pit = "balance_sheet_pit"
    cashflow_statement_pit = "cashflow_statement_pit"
    financial_indicator_audit = "financial_indicator_audit"
    financial_disclosure_schedule = "financial_disclosure_schedule"
    earnings_forecast_pit = "earnings_forecast_pit"
    earnings_express_pit = "earnings_express_pit"
    repurchase_event_pit = "repurchase_event_pit"
    holder_number_pit = "holder_number_pit"
    margin_detail_pit = "margin_detail_pit"
    index_weight_pit = "index_weight_pit"


TUSHARE_V2_CONFIG = _TushareV2TableNames()

#: Partition expressions of the 19 v2 tables (date-keyed tables partition
#: by year of their date axis; the ledger and the taxonomy dimension form
#: single ``ALL`` partitions).
_V2_PARTITION_SPEC: dict[str, tuple[str | None, str | None]] = {
    "source_ingest_runs": (None, None),
    "daily_bar_raw": ("substr(trade_date, 1, 4)", "trade_date"),
    "daily_adj_factor": ("substr(trade_date, 1, 4)", "trade_date"),
    "daily_basic_pit": ("substr(trade_date, 1, 4)", "trade_date"),
    "daily_price_limit": ("substr(trade_date, 1, 4)", "trade_date"),
    "daily_suspension": ("substr(trade_date, 1, 4)", "trade_date"),
    "margin_detail_pit": ("substr(trade_date, 1, 4)", "trade_date"),
    "index_weight_pit": ("substr(trade_date, 1, 4)", "trade_date"),
    "industry_membership_pit": ("substr(in_date, 1, 4)", "in_date"),
    "income_statement_pit": ("substr(end_date, 1, 4)", "end_date"),
    "balance_sheet_pit": ("substr(end_date, 1, 4)", "end_date"),
    "cashflow_statement_pit": ("substr(end_date, 1, 4)", "end_date"),
    "financial_indicator_audit": ("substr(end_date, 1, 4)", "end_date"),
    "financial_disclosure_schedule": ("substr(end_date, 1, 4)", "end_date"),
    "earnings_forecast_pit": ("substr(end_date, 1, 4)", "end_date"),
    "earnings_express_pit": ("substr(end_date, 1, 4)", "end_date"),
    "repurchase_event_pit": ("substr(ann_date, 1, 4)", "ann_date"),
    "holder_number_pit": ("substr(end_date, 1, 4)", "end_date"),
    "industry_taxonomy": (None, None),
}


def build_v2_dataset_manifest(
    db: AshareDB,
    *,
    source_versions: dict[str, str] | None = None,
    use_cache: bool = True,
) -> DatasetManifest:
    """Content-addressed manifest of the Tushare v2 database state.

    Deterministic for identical content (append-only rows, order-free
    hashing) and recomputed from table content per the p13 precedent:
    the dataset_id identifies the data, ``TUSHARE_STORE_SCHEMA_VERSION``
    rides in ``source_versions`` as provenance metadata.
    """

    return build_dataset_manifest(
        db, TUSHARE_V2_CONFIG, source_versions=source_versions,
        use_cache=use_cache,
    )


def table_columns(db: AshareDB, table: str) -> list[str]:
    """Sorted column names of ``table`` (the canonical schema axis)."""

    cols = db.query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY column_name",
        [table],
    )
    if cols.empty:
        raise ValueError(f"unknown table {table!r} (not in the manifest spec)")
    return [str(name) for name in cols["column_name"]]


def schema_digest(db: AshareDB, table: str) -> str:
    """sha256 over the sorted ``column:type`` lines of ``table``."""

    cols = db.query(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY column_name",
        [table],
    )
    if cols.empty:
        raise ValueError(f"unknown table {table!r} (not in the manifest spec)")
    payload = "\n".join(
        f"{row.column_name}:{row.data_type}" for row in cols.itertuples()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_digest_sql(columns: list[str]) -> str:
    inner = ", ".join(_ESCAPE_SQL.format(col=column) for column in columns)
    return f"md5(concat_ws('~', {inner}))"


def _partition_where(config, table: str, partition_key: str) -> str:
    expr, _ = partition_spec(config)[table]
    if expr is None:
        return ""
    return f" WHERE {expr} = '{partition_key}'"


def partition_keys(db: AshareDB, config, table: str) -> list[str]:
    """Partition keys of ``table`` (sorted; ``["ALL"]`` when unpartitioned)."""

    expr, _ = partition_spec(config)[table]
    if expr is None:
        return ["ALL"]
    keys = db.query(f"SELECT DISTINCT {expr} AS pk FROM {table} ORDER BY pk")
    return [str(key) for key in keys["pk"]]


def _partition_row_count_max(db: AshareDB, config, table: str, partition_key: str):
    where = _partition_where(config, table, partition_key)
    _, key_col = partition_spec(config)[table]
    if key_col is None:
        row = db.query(f"SELECT COUNT(*) AS n FROM {table}{where}").iloc[0]
        return int(row["n"]), ""
    row = db.query(
        f"SELECT COUNT(*) AS n, MAX({key_col}) AS mx FROM {table}{where}"
    ).iloc[0]
    max_key = "" if row["mx"] is None else str(row["mx"])
    return int(row["n"]), max_key


def _compute_partition_hash(
    db: AshareDB, config, table: str, partition_key: str
) -> str:
    """md5 of the sorted per-row digests of one partition (order-free)."""

    columns = table_columns(db, table)
    digest_sql = _row_digest_sql(columns)
    where = _partition_where(config, table, partition_key)
    row = db.query(
        "SELECT COALESCE(md5(string_agg(canon, '|' ORDER BY canon)), '') AS h "
        f"FROM (SELECT {digest_sql} AS canon FROM {table}{where})"
    ).iloc[0]
    return str(row["h"])


def _ensure_cache_table(db: AshareDB) -> None:
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_CACHE_TABLE} (
            table_name VARCHAR,
            partition_key VARCHAR,
            row_count BIGINT,
            max_key VARCHAR,
            hash VARCHAR,
            computed_at VARCHAR,
            PRIMARY KEY (table_name, partition_key)
        )
        """
    )


def _cache_lookup(
    db: AshareDB, table: str, partition_key: str, row_count: int, max_key: str
) -> str | None:
    """Cached partition hash iff (count, max key) still match exactly.

    Read-only connections read the cache when it exists and degrade to a
    cache miss (recompute) when the table is absent — manifest building
    must never require write access.
    """

    if not db.read_only:
        _ensure_cache_table(db)
    try:
        row = db.query(
            f"SELECT hash FROM {_CACHE_TABLE} "
            "WHERE table_name = ? AND partition_key = ? AND row_count = ? AND max_key = ?",
            [table, partition_key, row_count, max_key],
        )
    except duckdb.CatalogException:
        return None
    return str(row.iloc[0]["hash"]) if not row.empty else None


def _cache_store(
    db: AshareDB, table: str, partition_key: str, row_count: int, max_key: str,
    hash_value: str,
) -> None:
    if db.read_only:
        return
    _ensure_cache_table(db)
    db.execute(
        f"""
        INSERT INTO {_CACHE_TABLE} (table_name, partition_key, row_count,
                                    max_key, hash, computed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (table_name, partition_key) DO UPDATE SET
            row_count = EXCLUDED.row_count,
            max_key = EXCLUDED.max_key,
            hash = EXCLUDED.hash,
            computed_at = EXCLUDED.computed_at
        """,
        [
            table,
            partition_key,
            row_count,
            max_key,
            hash_value,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ],
    )


@dataclass(frozen=True)
class PartitionFingerprint:
    partition_key: str
    row_count: int
    max_key: str
    hash: str


@dataclass(frozen=True)
class TableFingerprint:
    table: str
    schema_digest: str
    row_count: int
    date_range: tuple[str, str] | None
    partitions: tuple[PartitionFingerprint, ...]

    @property
    def merkle(self) -> str:
        """One table's contribution to the manifest root."""

        lines = [
            f"schema:{self.schema_digest}",
            f"rows:{self.row_count}",
            f"range:{self.date_range[0]}..{self.date_range[1]}"
            if self.date_range
            else "range:",
        ]
        lines.extend(
            f"{partition.partition_key}:{partition.hash}"
            for partition in sorted(self.partitions, key=lambda p: p.partition_key)
        )
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    manifest_version: str
    dataset_id: str
    merkle_root: str
    created_at: str
    source_versions: dict[str, str]
    tables: tuple[TableFingerprint, ...]

    @property
    def total_rows(self) -> int:
        return sum(table.row_count for table in self.tables)

    @property
    def date_range(self) -> tuple[str, str] | None:
        for table in self.tables:
            if table.date_range is not None:
                return table.date_range
        return None

    def table(self, name: str) -> TableFingerprint:
        for table in self.tables:
            if table.table == name:
                return table
        raise KeyError(f"table {name!r} is not part of the manifest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "dataset_id": self.dataset_id,
            "merkle_root": self.merkle_root,
            "created_at": self.created_at,
            "source_versions": dict(self.source_versions),
            "tables": [
                {
                    "table": table.table,
                    "schema_digest": table.schema_digest,
                    "row_count": table.row_count,
                    "date_range": (
                        list(table.date_range) if table.date_range else None
                    ),
                    "partitions": [
                        {
                            "partition_key": partition.partition_key,
                            "row_count": partition.row_count,
                            "max_key": partition.max_key,
                            "hash": partition.hash,
                        }
                        for partition in table.partitions
                    ],
                }
                for table in self.tables
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(
            manifest_version=str(payload["manifest_version"]),
            dataset_id=str(payload["dataset_id"]),
            merkle_root=str(payload["merkle_root"]),
            created_at=str(payload["created_at"]),
            source_versions={
                str(key): str(value) for key, value in payload["source_versions"].items()
            },
            tables=tuple(
                TableFingerprint(
                    table=str(entry["table"]),
                    schema_digest=str(entry["schema_digest"]),
                    row_count=int(entry["row_count"]),
                    date_range=(
                        tuple(str(value) for value in entry["date_range"])
                        if entry.get("date_range")
                        else None
                    ),
                    partitions=tuple(
                        PartitionFingerprint(
                            partition_key=str(part["partition_key"]),
                            row_count=int(part["row_count"]),
                            max_key=str(part["max_key"]),
                            hash=str(part["hash"]),
                        )
                        for part in entry["partitions"]
                    ),
                )
                for entry in payload["tables"]
            ),
        )


def compute_partition_fingerprint(
    db: AshareDB,
    config,
    table: str,
    partition_key: str,
    *,
    use_cache: bool = True,
) -> PartitionFingerprint:
    """Fingerprint one partition, reusing the cache when it is still exact."""

    if table not in partition_spec(config):
        raise ValueError(
            f"unknown table {table!r} (not in the manifest spec)"
        )
    row_count, max_key = _partition_row_count_max(db, config, table, partition_key)
    hash_value = None
    if use_cache:
        hash_value = _cache_lookup(db, table, partition_key, row_count, max_key)
    if hash_value is None:
        hash_value = _compute_partition_hash(db, config, table, partition_key)
        _cache_store(db, table, partition_key, row_count, max_key, hash_value)
    return PartitionFingerprint(partition_key, row_count, max_key, hash_value)


def compute_table_fingerprint(
    db: AshareDB,
    config,
    table: str,
    *,
    use_cache: bool = True,
) -> TableFingerprint:
    """Fingerprint one table: schema digest + per-partition hashes."""

    if table not in partition_spec(config):
        raise ValueError(f"unknown table {table!r} (not in the manifest spec)")
    digest = schema_digest(db, table)
    partitions = tuple(
        compute_partition_fingerprint(
            db, config, table, key, use_cache=use_cache
        )
        for key in partition_keys(db, config, table)
    )
    key_column = partition_spec(config)[table][1]
    date_range = None
    if key_column is not None:
        row = db.query(f"SELECT MIN({key_column}) AS mn, MAX({key_column}) AS mx "
                       f"FROM {table}").iloc[0]
        if row["mn"] is not None and row["mx"] is not None:
            date_range = (str(row["mn"]), str(row["mx"]))
    return TableFingerprint(
        table=table,
        schema_digest=digest,
        row_count=sum(partition.row_count for partition in partitions),
        date_range=date_range,
        partitions=partitions,
    )


def build_dataset_manifest(
    db: AshareDB,
    config,
    *,
    source_versions: dict[str, str] | None = None,
    use_cache: bool = True,
) -> DatasetManifest:
    """Content-addressed manifest of the current database state."""

    tables = tuple(
        compute_table_fingerprint(db, config, table, use_cache=use_cache)
        for table in partition_spec(config)
    )
    merkle_root = hashlib.sha256(
        "\n".join(
            f"{table.table}:{table.merkle}"
            for table in sorted(tables, key=lambda t: t.table)
        ).encode("utf-8")
    ).hexdigest()
    content = json.dumps(
        {
            "manifest_version": MANIFEST_VERSION,
            "tables": {
                table.table: {
                    "merkle": table.merkle,
                    "row_count": table.row_count,
                    "schema_digest": table.schema_digest,
                }
                for table in sorted(tables, key=lambda t: t.table)
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    dataset_id = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return DatasetManifest(
        manifest_version=MANIFEST_VERSION,
        dataset_id=dataset_id,
        merkle_root=merkle_root,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_versions=dict(source_versions or {}),
        tables=tables,
    )


def save_manifest(db: AshareDB, config, manifest: DatasetManifest) -> None:
    """Persist the manifest; identical content is never duplicated.

    This is the content-addressed *library* insert only — it records that a
    dataset_id exists, not that it was published as a generation.  The
    publication entry is :func:`publish_snapshot` (A02 p26 §3/I7).
    """

    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_MANIFEST_TABLE} (
            dataset_id VARCHAR PRIMARY KEY,
            manifest_version VARCHAR NOT NULL,
            created_at VARCHAR NOT NULL,
            manifest VARCHAR NOT NULL
        )
        """
    )
    db.execute(
        f"INSERT INTO {_MANIFEST_TABLE} VALUES (?, ?, ?, ?) "
        f"ON CONFLICT (dataset_id) DO NOTHING",
        [
            manifest.dataset_id,
            manifest.manifest_version,
            manifest.created_at,
            json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True),
        ],
    )


# --- A02 publication protocol (p26 §3/I7) --------------------------------------
#
# One publication = one immutable, monotonic GENERATION of one member
# library.  The generation counter — never the second-granularity
# ``created_at`` — decides which manifest is the latest: the observed OB-1
# defect was that ``ORDER BY created_at DESC, dataset_id DESC`` resolved
# same-second ties with the content-arbitrary dataset_id hex, so a stale
# row could win, and ``ON CONFLICT (dataset_id) DO NOTHING`` then made the
# wrong selection permanent (republishing identical content can never move
# a row's created_at).  Publications are append-only events (I9): the
# same content may be published again under a new generation, and nothing
# is ever rewritten.

_PUBLICATION_TABLE = "snapshot_publication"


@dataclass(frozen=True)
class SnapshotPublication:
    """One published generation of one member library (A02 p26 §3/I7).

    ``publication_id`` is the monotonic generation counter shared by all
    member libraries of one database file; ``member_library`` names which
    library the generation belongs to (``primary`` or ``tushare_v2``).
    """

    publication_id: int
    member_library: str
    dataset_id: str
    manifest_version: str
    publication_version: str
    published_at: str
    manifest: DatasetManifest


def _member_library(config) -> str:
    """The member library a config's manifests belong to."""

    return "tushare_v2" if isinstance(config, _TushareV2TableNames) else "primary"


def _ensure_publication_table(db: AshareDB) -> None:
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PUBLICATION_TABLE} (
            publication_id BIGINT NOT NULL,
            member_library VARCHAR NOT NULL,
            dataset_id VARCHAR NOT NULL,
            manifest_version VARCHAR NOT NULL,
            publication_version VARCHAR NOT NULL,
            published_at VARCHAR NOT NULL,
            manifest VARCHAR NOT NULL,
            PRIMARY KEY (publication_id, member_library)
        )
        """
    )


def _append_publication(
    db: AshareDB, member_library: str, manifest: DatasetManifest
) -> SnapshotPublication:
    """Append one publication generation (transactional, monotonic).

    The generation id is allocated inside the transaction (DuckDB allows a
    single writer per file, so the MAX+1 read cannot race another writer).
    A failure rolls the generation back entirely — an interrupted
    publication leaves no row, so intermediate states never become visible
    as generations (p26 §3/I7).  The rollback is issued only for the
    transaction THIS frame opened: a caller that is already inside its own
    transaction gets the error propagated with its transaction untouched
    (never a silent rollback of the caller's work — review F2).
    """

    _ensure_publication_table(db)
    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = False
    try:
        db.execute("BEGIN TRANSACTION")
        started = True
        row = db.query(
            f"SELECT COALESCE(MAX(publication_id), 0) AS m "
            f"FROM {_PUBLICATION_TABLE}"
        )
        publication_id = int(row.iloc[0]["m"]) + 1
        db.execute(
            f"INSERT INTO {_PUBLICATION_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                publication_id,
                member_library,
                manifest.dataset_id,
                manifest.manifest_version,
                SNAPSHOT_PUBLICATION_VERSION,
                published_at,
                json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True),
            ],
        )
        db.execute("COMMIT")
    except Exception:
        if started:
            db.execute("ROLLBACK")
        raise
    return SnapshotPublication(
        publication_id=publication_id,
        member_library=member_library,
        dataset_id=manifest.dataset_id,
        manifest_version=manifest.manifest_version,
        publication_version=SNAPSHOT_PUBLICATION_VERSION,
        published_at=published_at,
        manifest=manifest,
    )


def publish_snapshot(
    db: AshareDB,
    config,
    *,
    source_versions: dict[str, str] | None = None,
) -> SnapshotPublication:
    """Publish ONE content-confirmed generation of this database (A02 I7).

    The manifest is built **uncached** — a published generation asserts the
    content it names, so it may never serve a stale partition hash from
    ``dataset_manifest_cache`` (a value mutation that preserves
    ``(count, max)`` would otherwise be published under the old identity).
    The build is then saved to the content-addressed library (idempotent)
    and appended to the publication ledger as a new monotonic generation.

    This is the sanctioned single-database publication entry; the
    dual-database composite generation is published through the composite
    authority in ``ashare_model.runspec`` (it must call the single hash
    authority, and ``ashare_data`` is the bottom §9 layer).
    """

    manifest = build_dataset_manifest(
        db, config, source_versions=source_versions, use_cache=False
    )
    save_manifest(db, config, manifest)
    return _append_publication(db, _member_library(config), manifest)


def latest_publication(db: AshareDB, config) -> SnapshotPublication | None:
    """The newest publication generation of this config's member library."""

    try:
        rows = db.query(
            f"SELECT publication_id, member_library, dataset_id, "
            f"manifest_version, publication_version, published_at, manifest "
            f"FROM {_PUBLICATION_TABLE} WHERE member_library = ? "
            "ORDER BY publication_id DESC LIMIT 1",
            [_member_library(config)],
        )
    except duckdb.CatalogException:
        return None
    if rows.empty:
        return None
    row = rows.iloc[0]
    return SnapshotPublication(
        publication_id=int(row["publication_id"]),
        member_library=str(row["member_library"]),
        dataset_id=str(row["dataset_id"]),
        manifest_version=str(row["manifest_version"]),
        publication_version=str(row["publication_version"]),
        published_at=str(row["published_at"]),
        manifest=DatasetManifest.from_dict(json.loads(str(row["manifest"]))),
    )


#: Member libraries that must ALL be present for one publication id to be a
#: complete dual-database composite generation (A02 p26 §3/I7): a formal run
#: may only consume a complete group — an intermediate state (one library
#: published, the other not) is never research-open.
COMPOSITE_PUBLICATION_MEMBERS: tuple[str, ...] = ("primary", "tushare_v2")


@dataclass(frozen=True)
class PublicationGroup:
    """One publication id's member manifests (A02 p26 §3/I7).

    ``complete`` is true exactly when every member of
    :data:`COMPOSITE_PUBLICATION_MEMBERS` is present — the atomicity unit
    for the cross-database generation (the two DuckDB files cannot commit
    atomically, so "all members in place under one id" is the boundary).
    ``publication_version`` is the publication-protocol identity the group
    was published under (R16/I3-EXT: a generation published under an older
    protocol must be treated as legacy when the protocol moves on, never
    silently re-identified).
    """

    publication_id: int
    published_at: str
    publication_version: str
    members: dict[str, DatasetManifest]

    @property
    def complete(self) -> bool:
        return all(member in self.members for member in COMPOSITE_PUBLICATION_MEMBERS)

    @property
    def missing_members(self) -> tuple[str, ...]:
        return tuple(
            member
            for member in COMPOSITE_PUBLICATION_MEMBERS
            if member not in self.members
        )


def append_publication_group(
    db: AshareDB, members: dict[str, DatasetManifest]
) -> PublicationGroup:
    """Append one publication id carrying ALL member manifests (atomic).

    The rows share one transactional generation id: either every member is
    in place or none is — an interrupted composite publication leaves no
    partial group, so intermediate states never become visible as
    generations (p26 §3/I7).  As with :func:`_append_publication`, the
    rollback only ever targets the transaction this frame opened (a
    caller's own open transaction is never rolled back — review F2).

    The composite *identity derivation* itself lives in
    ``ashare_model.runspec`` (single hash authority, §9 layering); this is
    only the ledger append.  Completeness is NOT validated here — an
    incomplete group (one member missing) is a legal ledger record of an
    intermediate state; ``latest_publication_group`` skips incomplete ids
    and the confirmation path refuses them (review F8: no silent
    fabrication).
    """

    _ensure_publication_table(db)
    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    started = False
    try:
        db.execute("BEGIN TRANSACTION")
        started = True
        row = db.query(
            f"SELECT COALESCE(MAX(publication_id), 0) AS m "
            f"FROM {_PUBLICATION_TABLE}"
        )
        publication_id = int(row.iloc[0]["m"]) + 1
        for member_library, manifest in members.items():
            db.execute(
                f"INSERT INTO {_PUBLICATION_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    publication_id,
                    member_library,
                    manifest.dataset_id,
                    manifest.manifest_version,
                    SNAPSHOT_PUBLICATION_VERSION,
                    published_at,
                    json.dumps(
                        manifest.to_dict(), ensure_ascii=False, sort_keys=True
                    ),
                ],
            )
        db.execute("COMMIT")
    except Exception:
        if started:
            db.execute("ROLLBACK")
        raise
    return PublicationGroup(
        publication_id, published_at, SNAPSHOT_PUBLICATION_VERSION, dict(members)
    )


def latest_publication_group(db: AshareDB) -> PublicationGroup | None:
    """The newest COMPLETE composite generation, or ``None``.

    Incomplete newer groups are skipped — they are intermediate states that
    must not be research-open — but they stay in the ledger (append-only
    audit trail); completeness is judged per publication id.  The group's
    ``publication_version`` is returned as recorded: gating it against the
    current protocol is the CONFIRMATION path's job (R16 — a group
    published under an older protocol is legacy, never silently
    re-identified).
    """

    try:
        rows = db.query(
            f"SELECT publication_id, member_library, publication_version, "
            f"published_at, manifest "
            f"FROM {_PUBLICATION_TABLE} ORDER BY publication_id DESC"
        )
    except duckdb.CatalogException:
        return None
    if rows.empty:
        return None
    groups: dict[int, dict[str, DatasetManifest]] = {}
    versions: dict[int, str] = {}
    published_at: dict[int, str] = {}
    for _, row in rows.iterrows():
        publication_id = int(row["publication_id"])
        groups.setdefault(publication_id, {})[str(row["member_library"])] = (
            DatasetManifest.from_dict(json.loads(str(row["manifest"])))
        )
        versions.setdefault(publication_id, str(row["publication_version"]))
        published_at.setdefault(publication_id, str(row["published_at"]))
    for publication_id in sorted(groups, reverse=True):
        group = PublicationGroup(
            publication_id,
            published_at[publication_id],
            versions[publication_id],
            groups[publication_id],
        )
        if group.complete:
            return group
    return None


def newest_publication_members(db: AshareDB) -> tuple[int, tuple[str, ...]] | None:
    """The newest publication id and its member libraries (I12: refusal
    messages must be able to name WHICH members of a group are missing)."""

    try:
        rows = db.query(
            f"SELECT publication_id, member_library FROM {_PUBLICATION_TABLE} "
            "ORDER BY publication_id DESC"
        )
    except duckdb.CatalogException:
        return None
    if rows.empty:
        return None
    newest = int(rows.iloc[0]["publication_id"])
    members = tuple(
        sorted(
            str(row["member_library"])
            for _, row in rows.iterrows()
            if int(row["publication_id"]) == newest
        )
    )
    return newest, members


def drifted_tables(expected: DatasetManifest, current: DatasetManifest) -> list[str]:
    """Tables whose fingerprints differ between two manifests (I12: the
    refusal message must name the drifted objects, never a bare boolean)."""

    expected_tables = {table.table: table for table in expected.tables}
    current_tables = {table.table: table for table in current.tables}
    drifted = [
        name
        for name, table in current_tables.items()
        if expected_tables.get(name) is None
        or expected_tables[name].merkle != table.merkle
    ]
    drifted.extend(
        f"{name} (missing)"
        for name in expected_tables
        if name not in current_tables
    )
    return drifted


def latest_manifest(db: AshareDB, config) -> DatasetManifest | None:
    """Most recently PUBLISHED manifest of this database, or ``None``.

    A02 (p26 §3/I7, the OB-1 fix): the newest publication generation
    decides — a monotonic counter, never the second-granularity
    ``created_at`` whose ``dataset_id DESC`` tiebreak is content-arbitrary.
    The legacy ``dataset_manifest`` ordering survives only as the fallback
    for databases with no publications (the pre-A02 audit shape), where it
    keeps its historical meaning: "the most recently created manifest
    row", unconfirmed by any generation.
    """

    publication = latest_publication(db, config)
    if publication is not None:
        return publication.manifest
    rows = db.query(
        f"SELECT manifest FROM {_MANIFEST_TABLE} "
        "ORDER BY created_at DESC, dataset_id DESC LIMIT 1"
    )
    if rows.empty:
        return None
    return DatasetManifest.from_dict(json.loads(str(rows.iloc[0]["manifest"])))


def resolve_dataset_id(db: AshareDB, config) -> str | None:
    """dataset_id of the latest manifest, or ``None`` when none exists.

    This is the **query** side of the A02 split (p26 I1): it reports what was
    last saved and never claims that the current content still matches it.
    Formal entries must consume the composite authority
    (``ashare_model.runspec.confirm_composite_snapshot``); the single-DB
    :func:`confirm_dataset_identity` below is the audit-level confirmation
    (its formal caller was superseded by the composite path in the H-1
    wiring).
    """

    manifest = latest_manifest(db, config)
    return manifest.dataset_id if manifest is not None else None


#: Publication-protocol identity (A02, p26 §4.2 captain ruling 2).  This is a
#: new artifact-facing version: it must stay wired into the owner index
#: (``ashare_model.runspec:_VERSION_IMPORTS``) so artifact classification can
#: see which publication protocol produced an identity.
SNAPSHOT_PUBLICATION_VERSION = "1"

#: Input domains that ARE part of the composite identity this round
#: (p26 §3/I3, as clarified by the A02 composite-coverage erratum).  The
#: list is data, not a second implementation: it documents what the
#: composite covers and feeds ``composite_snapshot_domains()``.
COMPOSITE_DOMAINS: tuple[str, ...] = (
    "primary:daily_bar",  # 行情
    "primary:trade_calendar",  # 交易日历
    "primary:constituents",  # PIT universe / 成分
    "primary:stocks",  # 证券主数据（上市/退市）
    "primary:fundamental_pit",  # 财务（含版本代际）
    "primary:margin_balance",  # 两融
    "primary:sw_industry_index",  # 行业指数
    "primary:sw_industry_member",  # 行业成员
    "v2:all_partitioned_tables",  # 复权/daily_basic/交易约束/价格等 v2 全域
)

#: Domains NOT admitted for research this round (captain ruling 3 + the
#: A02 composite-coverage erratum).  SEMANTICS (erratum, replacing the
#: pre-erratum ``NOT_INCLUDED_DOMAINS`` declaration): the composite binds
#: the FULL v2 dataset manifest, so these domains ARE identity-covered —
#: any content change in them rotates the composite identity and refuses
#: formal runs (fail-closed) — but they are NOT research-admitted: F3
#: (p25) and B01 have not adjudicated their data qualification, so no
#: factor may consume them yet.  This split is honest about what the
#: implementation does (identity coverage) versus what the research
#: process has decided (admission).  R15's silent-extension prohibition is
#: dispositioned by the erratum: changing the identity's table SCOPE still
#: requires an I3-EXT extension contract and a
#: ``SNAPSHOT_PUBLICATION_VERSION`` bump; admission decisions do not.
PENDING_ADMISSION_DOMAINS: tuple[str, ...] = (
    "v2:earnings_forecast_pit",
    "v2:earnings_express_pit",
    "v2:repurchase_event_pit",
    "v2:holder_number_pit",
    "v2:index_weight_pit(content)",
    "primary:legacy_unpartitioned_tables",
)


def composite_snapshot_domains() -> dict[str, tuple[str, ...]]:
    """The coverage statement of the composite identity (A02 I3/I3-EXT).

    ``included`` is the identity coverage (what a content change in it
    rotates); ``pending_admission`` lists the v2 domains that are
    identity-covered through the full v2 dataset manifest but not
    research-admitted this round (the A02 composite-coverage erratum).
    """

    return {
        "included": COMPOSITE_DOMAINS,
        "pending_admission": PENDING_ADMISSION_DOMAINS,
        "publication_version": (SNAPSHOT_PUBLICATION_VERSION,),
    }


# NOTE (A02 layering): the composite *derivation* lives in
# ``ashare_model.runspec.composite_snapshot_id`` because it must call the
# single hash authority ``ashare_model.identity.content_hash``, and
# ``ashare_data`` is the bottom layer of the AGENTS §9 dependency graph (the
# ``test_ashare_data_package_has_no_reverse_model_imports`` guard enforces
# this).  This module owns the publication version and the coverage lists
# only — one authority, no second hashing path.


#: Machine-readable capability statement of the verification modes (A02 I8).
_VERIFY_CAPABILITIES: dict[str, dict[str, object]] = {
    "fast": {
        "mode": "fast",
        # Recomputes schema digests and (row_count, max_key) partition keys.
        "detects": ("append", "delete", "schema_change"),
        "content_confirmation": False,
        "detail": (
            "cheap aggregate keys: detects appends/deletes/schema changes; "
            "does NOT detect value-only mutations that preserve count and max"
        ),
    },
    "full": {
        "mode": "full",
        "detects": ("append", "delete", "schema_change", "value_mutation"),
        "content_confirmation": True,
        "detail": "re-hashes every partition; detects value-only mutations",
    },
}


def verification_capability(verify: str) -> dict[str, object]:
    """What a verification mode actually proves (A02 I8, p26 §4/A02-A4).

    ``fast`` must never be presented as complete content confirmation: kept
    as an explicit, machine-readable statement so callers/logs cannot imply
    more than the mode proves.
    """

    if verify not in _VERIFY_CAPABILITIES:
        raise ValueError(
            f"unknown verify mode {verify!r}; expected 'fast' or 'full'"
        )
    return dict(_VERIFY_CAPABILITIES[verify])


def confirm_dataset_identity(
    db: AshareDB,
    config,
    *,
    verify: str = "fast",
) -> str:
    """The **confirmation** side of the A02 split (p26 I1/I2/I5).

    Returns the current ``dataset_id`` only when the database content still
    matches the latest published manifest; otherwise raises
    :class:`DatasetIdMismatch` naming the drifted tables (I12).  Content is
    recomputed without touching the manifest cache, so this stays a read-only
    operation on a read-only connection.

    ``verify`` is validated for signature compatibility but does NOT select
    a cheaper path: the confirmation is ALWAYS a full uncached re-hash — no
    fast mode may ever be presented as content confirmation (I8; the
    capability statements of the verification modes live in
    :func:`verification_capability` / :func:`verify_dataset`).  This API is
    audit-level since the formal loader moved to the composite authority
    (``ashare_model.runspec.confirm_composite_snapshot``).
    """

    if verify not in _VERIFY_CAPABILITIES:
        raise ValueError(
            f"unknown verify mode {verify!r}; expected 'fast' or 'full'"
        )
    expected = latest_manifest(db, config)
    if expected is None:
        raise DatasetIdMismatch(
            "no dataset manifest exists for the current database: run "
            "`python -m ashare_data.manifest` to publish one before a formal "
            "run (no content identity can be confirmed)"
        )
    # R16 (p26 §5.7): a generation published under an older publication
    # protocol is legacy once the protocol moves on — refuse it instead of
    # silently re-identifying the old content under the new protocol.
    publication = latest_publication(db, config)
    if (
        publication is not None
        and publication.publication_version != SNAPSHOT_PUBLICATION_VERSION
    ):
        raise DatasetIdMismatch(
            f"the newest publication {publication.publication_id} was "
            f"published under publication protocol "
            f"{publication.publication_version}, the current protocol is "
            f"{SNAPSHOT_PUBLICATION_VERSION}: the old generation is legacy "
            "(A02 I3-EXT/R16); publish a new generation under the current "
            "protocol"
        )
    current = build_dataset_manifest(db, config, use_cache=False)
    if current.dataset_id == expected.dataset_id:
        return expected.dataset_id

    detail = ", ".join(drifted_tables(expected, current))
    raise DatasetIdMismatch(
        f"latest manifest {expected.dataset_id[:12]}… (published "
        f"{expected.created_at}) no longer matches the current content "
        f"{current.dataset_id[:12]}…; drifted: {detail or 'unknown'}; "
        "publish a new manifest for the changed content before a formal run "
        "(the previous generation is kept as legacy)"
    )


@dataclass(frozen=True)
class TableVerify:
    table: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    tables: dict[str, TableVerify] = field(default_factory=dict)


def verify_dataset(
    db: AshareDB,
    config,
    *,
    expected: DatasetManifest,
    verify: str = "fast",
) -> VerifyReport:
    """Audit the database against an expected manifest.

    ``fast`` recomputes schema digests and partition (count, max) keys —
    cheap aggregates that catch appends, deletes and schema changes — and
    only re-hashes partitions whose cache entry no longer matches.
    ``full`` recomputes every partition hash, catching value-only
    mutations that preserve counts and max dates.
    """

    if verify not in ("fast", "full"):
        raise ValueError(f"unknown verify mode {verify!r}; expected 'fast' or 'full'")
    tables: dict[str, TableVerify] = {}
    for table in expected.tables:
        if table.table not in partition_spec(config):
            tables[table.table] = TableVerify(
                table.table, False, "table is no longer in the manifest spec"
            )
            continue
        try:
            digest = schema_digest(db, table.table)
        except ValueError as exc:
            tables[table.table] = TableVerify(table.table, False, str(exc))
            continue
        if digest != table.schema_digest:
            tables[table.table] = TableVerify(
                table.table, False, "schema digest changed"
            )
            continue
        current = compute_table_fingerprint(
            db, config, table.table, use_cache=(verify == "fast")
        )
        if current == table:
            tables[table.table] = TableVerify(table.table, True, "unchanged")
        else:
            changed = [
                partition.partition_key
                for partition in current.partitions
                if partition not in table.partitions
            ]
            tables[table.table] = TableVerify(
                table.table,
                False,
                "partition hash changed: " + ", ".join(changed),
            )
    return VerifyReport(ok=all(entry.ok for entry in tables.values()), tables=tables)


def check_dataset_id(payload_dataset_id: Any, expected: str | None) -> None:
    """Explicit artifact-migration policy.

    A payload without a dataset_id is a legacy (pre-T1-01) artifact and is
    accepted as such; a payload whose id differs from the current database
    is rejected, because mixing it into a run would silently compare
    measurements made on different data.
    """

    if expected is None or payload_dataset_id is None:
        return
    if str(payload_dataset_id) != str(expected):
        raise DatasetIdMismatch(
            f"artifact dataset_id {payload_dataset_id!r} does not match the "
            f"current database {expected!r}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI: publish the dataset manifest of the local database.

    Pre-T1-01 databases have no ``dataset_manifest`` table, so
    :func:`resolve_dataset_id` degrades to ``None`` and every formal run
    records ``dataset_id: null``.  This entry builds the manifest from the
    database content (a full uncached re-hash — a published generation
    asserts the content it names, A02 p26 §3/I7), persists it (one row per
    distinct dataset_id) and appends a monotonic publication generation, so
    the current data becomes identifiable:

        python -m ashare_data.manifest

    Note: this publishes ONE member library (the primary).  A formal run
    consumes the dual-database composite generation, which is published
    through the composite authority (``ashare_model.runspec``) — a
    primary-only publication leaves the composite group incomplete and
    formal loads refused by design (intermediate states are not research
    open).
    """

    import argparse

    from .config import load_config, make_data_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to ashare_config.yaml")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    raw = load_config(args.config, project_root=root)
    config = make_data_config(raw, root)
    with AshareDB(config.duckdb_path) as db:
        publication = publish_snapshot(
            db, config, source_versions={"manifest_cli": MANIFEST_VERSION}
        )
        manifest = publication.manifest
    print(f"dataset_id {manifest.dataset_id}")
    print(
        f"manifest v{manifest.manifest_version} published as generation "
        f"{publication.publication_id} at {publication.published_at} "
        f"({manifest.total_rows} rows across {len(manifest.tables)} tables)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
