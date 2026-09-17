"""Content-addressed persistent factor tensor cache (E2).

The factor tensor (``[feature, stock, date]``) produced from one
dataset/window/universe generation is cached on disk under a
content-addressed key built from the **full eight-element identity**
(any element mismatch is a different tensor):

1. ``dataset_id`` — the dataset identity the tensor was computed from;
2. ``window_id`` — the date window;
3. ``universe_hash`` — hash of the PIT universe (single identity
   authority: ``identity.content_hash`` of the sorted codes);
4. ``factor_compute_version`` — ``factors.FACTOR_COMPUTE_VERSION``;
5. ``feature_registry_version`` — ``feature_registry.FEATURE_REGISTRY_VERSION``;
6. ``industry_membership_version`` — the industry membership source
   generation (caller-supplied provenance, p17 §9/§12);
7. ``financial_pit_version`` — the financial PIT source generation
   (caller-supplied provenance, p17 §9/§12);
8. ``source snapshot ids`` — the ``source_ingest_runs`` ingest_id list
   of the snapshots the tensors were built from (stored normalized +
   aggregated; the key is order-independent).

Any element mismatch is a miss that must be recomputed; a tensor from
another dataset/window is **rejected even when a file sits at the
requested key** (the stored identity is verified against the requested
identity on every read — cross-key reuse cannot happen through a stale
or mislabeled file).  Entries are written atomically (tmp file +
``os.replace``) so concurrent readers never observe a half-written
tensor, and concurrent writers of the same key converge to one complete
entry.  Invalidation is observable: hits/misses/rejections/puts are
counted (``stats``) and logged through the standard loguru logging — no
second telemetry path.

This layer extends the project's single cache authority family
(``semantic_cache`` owns formula-score caching in memory; this module
owns the persistent factor tensor), and reuses the single identity
hashing authority (``identity.content_hash``) and the atomic-write
pattern of ``io_utils``/``run_store``.  It hashes identity blocks
opaquely: the industry/financial source generations are caller-supplied
provenance (p17 §9), never invented here.  Dependency direction (§9):
ashare_model only; no CLI/Web imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger

from ashare_model.identity import content_hash

# A02 (p26 §4.2): 1 -> 2. The ``dataset_id`` element of the cache identity
# now means the content-confirmed composite snapshot identity rather than
# "the latest saved id of the primary DB". The same key string therefore
# denotes a different tensor than before, so entries written under v1 must
# never be hit: the identity check classifies them as a miss (recompute),
# never as a reuse.
FACTOR_CACHE_VERSION = 2

_IDENTITY_STR_FIELDS = (
    "dataset_id",
    "window_id",
    "universe_hash",
    "factor_compute_version",
    "feature_registry_version",
    "industry_membership_version",
    "financial_pit_version",
)


@dataclass(frozen=True)
class FactorCacheIdentity:
    """The eight-element cache identity.  Every element is mandatory
    (a blank element would make the key blind to that dimension's
    changes) and participates in the content-addressed key."""

    dataset_id: str
    window_id: str
    universe_hash: str
    factor_compute_version: str
    feature_registry_version: str
    industry_membership_version: str
    financial_pit_version: str
    source_snapshot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in _IDENTITY_STR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"factor cache identity.{name} must be a non-empty string"
                )
        ids = tuple(self.source_snapshot_ids)
        if not ids:
            raise ValueError(
                "factor cache identity.source_snapshot_ids must not be "
                "empty: a key without snapshot ids is blind to source "
                "changes"
            )
        for value in ids:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"factor cache identity.source_snapshot_ids entries "
                    f"must be non-empty strings, got {value!r}"
                )


def factor_cache_payload(identity: FactorCacheIdentity) -> dict:
    """The canonical JSON-safe payload a cache entry is bound to (the
    key hash input AND the identity embedded in every stored file)."""

    ids = sorted(identity.source_snapshot_ids)
    return {
        "factor_cache_version": FACTOR_CACHE_VERSION,
        "dataset_id": identity.dataset_id,
        "window_id": identity.window_id,
        "universe_hash": identity.universe_hash,
        "factor_compute_version": identity.factor_compute_version,
        "feature_registry_version": identity.feature_registry_version,
        "industry_membership_version": identity.industry_membership_version,
        "financial_pit_version": identity.financial_pit_version,
        "source_snapshot_ids": ids,
        "source_snapshot_hash": hashlib.sha256(
            "\n".join(ids).encode("utf-8")
        ).hexdigest(),
    }


def factor_cache_key(identity: FactorCacheIdentity) -> str:
    """The content-addressed cache key (single identity hashing
    authority: ``identity.content_hash``)."""

    return content_hash("factor_cache", factor_cache_payload(identity))


class FactorCache:
    """Persistent content-addressed store of factor tensors.

    Layout: ``<root>/<key>.npz`` where the file embeds the tensor plus
    its canonical identity payload, written tmp + ``os.replace``.  Every
    read verifies the stored identity against the requested one — a
    mismatched file is rejected (counted + logged) and reported as a
    miss, so the caller recomputes instead of reusing a tensor that does
    not belong to the requested identity.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._n_puts = 0
        self._n_hits = 0
        self._n_misses = 0
        self._n_rejected = 0
        # B04 (p33 I2): how the LAST get() fared — "hit", "no_entry",
        # "unreadable" or "identity_mismatch".  Read-side observability
        # only: it never changes what may be reused (the identity check
        # stays the sole authority).
        self.last_outcome: str | None = None

    # --- entries -------------------------------------------------------------

    def entry_path(self, identity: FactorCacheIdentity) -> Path:
        """The store path for ``identity``'s entry — the single layout
        authority (``<root>/<content-hash>.npz``; put, get and external
        probes all resolve through this one pathlib construction, no
        string concatenation into a path component)."""

        return self.root.joinpath(factor_cache_key(identity)).with_suffix(
            ".npz"
        )

    def put(self, identity: FactorCacheIdentity, tensor: np.ndarray) -> str:
        """Atomically store ``tensor`` under ``identity``'s key and
        return the key.  Concurrent writers converge: each writes its own
        temp file and the final ``os.replace`` is atomic."""

        array = np.asarray(tensor)
        key = factor_cache_key(identity)
        payload = factor_cache_payload(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        # Path-traversal hardening: the cache key is a content hash, the
        # entry path is built purely through pathlib (see entry_path),
        # and the write-temp name comes from tempfile's own safe
        # generator (consumed through the already-open fd).
        final = self.entry_path(identity)
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=self.root, suffix=".npz.tmp"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as handle:
                np.savez(
                    handle,
                    tensor=array,
                    identity=json.dumps(payload, sort_keys=True),
                )
            # Windows: os.replace fails while a concurrent reader still
            # holds the target open — retry briefly (readers are short;
            # the tmp file keeps every attempt complete and isolated).
            for attempt in range(6):
                try:
                    os.replace(tmp_name, final)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._n_puts += 1
        logger.info(
            "factor_cache put key={} dataset={} window={} rows={}",
            key[:12],
            identity.dataset_id,
            identity.window_id,
            int(array.shape[0]) if array.ndim else 0,
        )
        return key

    def get(self, identity: FactorCacheIdentity) -> np.ndarray | None:
        """The stored tensor for ``identity``, or ``None`` (a miss, a
        mismatch, or a rejected cross-key file — the caller recomputes)."""

        key = factor_cache_key(identity)
        path = self.entry_path(identity)
        if not path.is_file():
            self._n_misses += 1
            self.last_outcome = "no_entry"
            logger.info(
                "factor_cache miss (no entry) key={} dataset={} window={}",
                key[:12],
                identity.dataset_id,
                identity.window_id,
            )
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                stored_identity = json.loads(str(data["identity"]))
                tensor = np.array(data["tensor"])
        except Exception as exc:  # noqa: BLE001 - a corrupt entry is a miss
            self._n_rejected += 1
            self.last_outcome = "unreadable"
            logger.warning(
                "factor_cache entry rejected (unreadable, recomputing) "
                "key={}: {}",
                key[:12],
                exc,
            )
            return None
        if stored_identity != factor_cache_payload(identity):
            self._n_rejected += 1
            self.last_outcome = "identity_mismatch"
            logger.warning(
                "factor_cache entry rejected (stored identity does not "
                "match the requested identity; recomputing) key={} "
                "stored_dataset={} requested_dataset={}",
                key[:12],
                stored_identity.get("dataset_id"),
                identity.dataset_id,
            )
            return None
        self._n_hits += 1
        self.last_outcome = "hit"
        logger.info(
            "factor_cache hit key={} dataset={} window={}",
            key[:12],
            identity.dataset_id,
            identity.window_id,
        )
        return tensor

    # --- observability -------------------------------------------------------

    @property
    def n_puts(self) -> int:
        return self._n_puts

    @property
    def n_hits(self) -> int:
        return self._n_hits

    @property
    def n_misses(self) -> int:
        return self._n_misses

    @property
    def n_rejected(self) -> int:
        return self._n_rejected

    def stats(self) -> dict[str, int]:
        """Invalidation/observability counters (the loguru logs carry the
        matching events; no second telemetry path)."""

        return {
            "factor_cache_version": FACTOR_CACHE_VERSION,
            "n_puts": self._n_puts,
            "n_hits": self._n_hits,
            "n_misses": self._n_misses,
            "n_rejected": self._n_rejected,
        }
