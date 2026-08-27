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

MANIFEST_VERSION = "1"
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
    """

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
    """Persist the manifest; identical content is never duplicated."""

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


def latest_manifest(db: AshareDB, config) -> DatasetManifest | None:
    """Most recently created manifest of this database, or ``None``."""

    rows = db.query(
        f"SELECT manifest FROM {_MANIFEST_TABLE} "
        "ORDER BY created_at DESC, dataset_id DESC LIMIT 1"
    )
    if rows.empty:
        return None
    return DatasetManifest.from_dict(json.loads(str(rows.iloc[0]["manifest"])))


def resolve_dataset_id(db: AshareDB, config) -> str | None:
    """dataset_id of the latest manifest, or ``None`` when none exists."""

    manifest = latest_manifest(db, config)
    return manifest.dataset_id if manifest is not None else None


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
    """CLI: build and persist the dataset manifest of the local database.

    Pre-T1-01 databases have no ``dataset_manifest`` table, so
    :func:`resolve_dataset_id` degrades to ``None`` and every formal run
    records ``dataset_id: null``.  This entry builds the manifest from the
    database content (read-only queries) and persists it (one row per
    distinct dataset_id), so the current data becomes identifiable:

        python -m ashare_data.manifest
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
        manifest = build_dataset_manifest(
            db, config, source_versions={"manifest_cli": MANIFEST_VERSION}
        )
        save_manifest(db, config, manifest)
    print(f"dataset_id {manifest.dataset_id}")
    print(
        f"manifest v{manifest.manifest_version} persisted at "
        f"{manifest.created_at} ({manifest.total_rows} rows across "
        f"{len(manifest.tables)} tables)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
