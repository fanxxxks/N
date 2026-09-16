"""Loader-side identity derivation for the E2 factor-tensor cache.

Bridges ``ashare_model.factor_cache`` (the E2 content-addressed store,
t20) into ``AshareDataLoader.load_data`` (p21 §3, amended 2026-09-08,
captain-approved).  The loader supplies the eight mandatory identity
elements from its own authorities:

* ``dataset_id`` — the manifest binding the load already resolves;
* ``window_id`` — the loaded bar-axis range (first..last:count);
* ``universe_hash`` — ``identity.content_hash`` of the sorted universe
  codes (the single identity authority);
* ``factor_compute_version`` / ``feature_registry_version`` — the
  version-owner constants;
* ``industry_membership_version`` / ``financial_pit_version`` — the
  p17 §9 caller-supplied source generations, derived here from the v2
  store state (row count + latest ``available_date`` per family — a
  deterministic generation marker that changes on any source write);
* ``source_snapshot_ids`` — the ``source_ingest_runs`` ingest-id list.

Anything the store cannot derive (missing table, no ingest rows) turns
the identity into ``None``: the cache is **disabled** for that load and
the loader computes the tensor exactly as before — a disabled cache can
never produce a wrong reuse.  Dependency direction (§9): ashare_model
only; no CLI/Web imports.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger

from ashare_data.db import AshareDB
from ashare_model.factor_cache import FactorCache, FactorCacheIdentity
from ashare_model.feature_registry import FEATURE_REGISTRY_VERSION
from ashare_model.factors import FACTOR_COMPUTE_VERSION
from ashare_model.identity import content_hash

FACTOR_CACHE_DIRNAME = "factor_tensor_cache"

_FINANCIAL_TABLES = (
    "income_statement_pit",
    "balance_sheet_pit",
    "cashflow_statement_pit",
)


def factor_cache_root(data_dir: Path | str) -> Path:
    """The cache root under the loader's data directory (not a tuning
    knob — one location, like the RunStore's ``runs/``)."""

    return Path(data_dir) / FACTOR_CACHE_DIRNAME


def factor_cache_for(data_dir: Path | str) -> FactorCache:
    return FactorCache(factor_cache_root(data_dir))


def _source_generation_marker(db, table: str) -> str:
    row = db.query(
        f"SELECT COUNT(*) AS n, "
        f"COALESCE(CAST(MAX(available_date) AS VARCHAR), '-') AS m "
        f"FROM {table}"
    )
    record = row.iloc[0]
    return f"{table}:{int(record['n'])}:{record['m']}"


def _table_exists(db, table: str) -> bool:
    row = db.query(
        "SELECT COUNT(*) AS n FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table],
    )
    return int(row.iloc[0]["n"]) > 0


def loader_factor_cache_identity_with_reason(
    v2_db_path: Path | str,
    dataset_id: str | None,
    dates: list[str],
    ts_codes: list[str],
) -> tuple[FactorCacheIdentity | None, str | None]:
    """Derive the eight-element cache identity from store state, or
    ``(None, reason)`` when the store cannot support it (the cache is
    then disabled — never approximated).  ``dataset_id`` None (a legacy
    manifest) likewise disables the cache: a tensor without a dataset
    identity must not be cached.

    B04 (p33 I2): the reason is machine-readable so the loader can record
    WHY the cache was disabled — one of ``identity_inputs_missing``,
    ``ingest_source_missing``, ``ingest_source_empty``,
    ``marker_tables_missing`` or ``identity_derivation_failed``; ``None``
    on success.
    """

    if not dataset_id or not dates or not ts_codes:
        return None, "identity_inputs_missing"
    try:
        with AshareDB(v2_db_path, read_only=True) as db:
            if not _table_exists(db, "source_ingest_runs"):
                return None, "ingest_source_missing"
            ingested = db.query("SELECT ingest_id FROM source_ingest_runs")
            if ingested.empty:
                return None, "ingest_source_empty"
            snapshot_ids = tuple(
                str(value) for value in ingested["ingest_id"]
            )
            marker_tables = ("industry_membership_pit",) + _FINANCIAL_TABLES
            if not all(_table_exists(db, table) for table in marker_tables):
                return None, "marker_tables_missing"
            industry = _source_generation_marker(
                db, "industry_membership_pit"
            )
            financial = "|".join(
                _source_generation_marker(db, table)
                for table in _FINANCIAL_TABLES
            )
    except duckdb.Error as exc:
        # Loud degradation: the cache is off, the load path is unchanged.
        # Single pre-formatted argument — callers that intercept the
        # logger (tests, collapse guards) must never see a formatting
        # arity surprise.
        logger.warning(
            f"factor cache disabled: identity derivation failed ({exc})"
        )
        return None, "identity_derivation_failed"
    return (
        FactorCacheIdentity(
            dataset_id=str(dataset_id),
            window_id=f"{dates[0]}..{dates[-1]}:{len(dates)}",
            universe_hash=content_hash("universe_codes", sorted(ts_codes)),
            factor_compute_version=str(FACTOR_COMPUTE_VERSION),
            feature_registry_version=str(FEATURE_REGISTRY_VERSION),
            industry_membership_version=industry,
            financial_pit_version=financial,
            source_snapshot_ids=snapshot_ids,
        ),
        None,
    )


def loader_factor_cache_identity(
    v2_db_path: Path | str,
    dataset_id: str | None,
    dates: list[str],
    ts_codes: list[str],
) -> FactorCacheIdentity | None:
    """The derived identity, or ``None`` when the store cannot support it
    (compat projection of :func:`loader_factor_cache_identity_with_reason`
    — callers that need the disabled reason use the reasoned form)."""

    return loader_factor_cache_identity_with_reason(
        v2_db_path, dataset_id, dates, ts_codes
    )[0]
