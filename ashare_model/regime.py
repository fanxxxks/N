"""Data-regime registry (T4-01).

Phase-4 rule: the 2021-2026 history has been viewed repeatedly, so it is
**development/validation data** — it must never again be presented as a
"final holdout".  The next true final evaluation may only consume

* **future** data — dates after the declared ``dev_cutoff`` (the last
  date that had been viewed when the regime was declared), or
* a **strictly locked slice** — a date range declared as locked *before*
  anyone looks at it, bound to the ``dataset_id`` it was declared on.

The :class:`RegimeRegistry` persists one :class:`DataRegime` (atomic
JSON via :mod:`ashare_data.io_utils`); a missing registry means "no
regime declared" (``None``), never a fabricated default.  Two gates are
enforced here:

* :meth:`RegimeRegistry.assert_folds_clear` — run **before any protocol
  trial**: every fold must stay entirely before the locked slice (fold
  train windows start at the beginning of history, so a fold that
  reaches the lock consumes locked data).  A lock declared on a
  different ``dataset_id`` than the current database is unverifiable and
  blocks the run.
* :meth:`RegimeRegistry.assert_final_evaluation` — a final (promotion)
  evaluation must use only future or locked windows.

Admin declaration: ``python -m ashare_model.regime --registry
data/holdout_registry.json --dev-cutoff 2026-12-31 [--lock START END
--note ...]``.  Locking twice without ``--force`` is refused: a lock is
a commitment, not a preference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ashare_data.io_utils import atomic_write_json, read_json_safe


class HoldoutViolation(Exception):
    """A protocol fold or final evaluation touches protected data."""


def _today() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class LockedSlice:
    """A strictly locked, never-viewed date range (closed interval)."""

    start: str  # YYYY-MM-DD, inclusive
    end: str  # YYYY-MM-DD, inclusive
    dataset_id: str | None = None
    locked_at: str = ""
    note: str = ""


@dataclass(frozen=True)
class DataRegime:
    """The data classification in force for all experiments."""

    dev_cutoff: str  # every date <= dev_cutoff is dev/validation data
    policy: str = "future_only"  # "future_only" | "locked_slice"
    locked_slice: LockedSlice | None = None
    declared_at: str = ""


def declare_dev_regime(dev_cutoff: str) -> DataRegime:
    """Declare everything up to ``dev_cutoff`` as dev/validation data."""

    return DataRegime(
        dev_cutoff=dev_cutoff,
        policy="future_only",
        declared_at=_today(),
    )


def lock_slice(
    regime: DataRegime,
    *,
    start: str,
    end: str,
    dataset_id: str | None = None,
    note: str = "",
) -> DataRegime:
    """Return ``regime`` with a strictly locked slice added.

    The lock must be a non-empty closed interval **strictly after** the
    dev cutoff — locking already-viewed (dev) data would present it as
    unseen, which is exactly the fraud this registry exists to prevent.
    """

    if not start or not end or start > end:
        raise ValueError(
            f"locked slice must be a non-empty interval, got {start}..{end}"
        )
    if start <= regime.dev_cutoff:
        raise ValueError(
            f"locked slice {start}..{end} overlaps dev data (dev_cutoff "
            f"{regime.dev_cutoff}); locked data must never have been viewed"
        )
    return DataRegime(
        dev_cutoff=regime.dev_cutoff,
        policy="locked_slice",
        locked_slice=LockedSlice(
            start=start,
            end=end,
            dataset_id=dataset_id,
            locked_at=_today(),
            note=note,
        ),
        declared_at=regime.declared_at,
    )


def _fold_window(fold) -> tuple[str, str]:
    """``(train_end, test_end)`` from a FoldConfig or a plain dict (the
    shape protocol artifacts record their folds in)."""

    if isinstance(fold, dict):
        return str(fold["train_end"]), str(fold["test_end"])
    return str(fold.train_end), str(fold.test_end)


class RegimeRegistry:
    """One persisted :class:`DataRegime` with the two experiment gates."""

    def __init__(self, path: str | Path, regime: DataRegime | None = None):
        self.path = Path(path)
        self.regime = regime
        if self.regime is None and self.path.exists():
            payload = read_json_safe(self.path)
            self.regime = self._regime_from_payload(payload)

    # -- persistence ---------------------------------------------------------

    @staticmethod
    def _regime_from_payload(payload) -> DataRegime | None:
        if not isinstance(payload, dict) or "dev_cutoff" not in payload:
            return None
        locked = payload.get("locked_slice")
        locked_slice = None
        if isinstance(locked, dict):
            locked_slice = LockedSlice(
                start=str(locked["start"]),
                end=str(locked["end"]),
                dataset_id=locked.get("dataset_id"),
                locked_at=str(locked.get("locked_at") or ""),
                note=str(locked.get("note") or ""),
            )
        return DataRegime(
            dev_cutoff=str(payload["dev_cutoff"]),
            policy=str(payload.get("policy") or "future_only"),
            locked_slice=locked_slice,
            declared_at=str(payload.get("declared_at") or ""),
        )

    def save(self) -> None:
        regime = self.regime
        locked = regime.locked_slice if regime else None
        payload = None
        if regime is not None:
            payload = {
                "declared_at": regime.declared_at,
                "dev_cutoff": regime.dev_cutoff,
                "policy": regime.policy,
                "locked_slice": (
                    {
                        "start": locked.start,
                        "end": locked.end,
                        "dataset_id": locked.dataset_id,
                        "locked_at": locked.locked_at,
                        "note": locked.note,
                    }
                    if locked is not None
                    else None
                ),
            }
        atomic_write_json(self.path, payload)

    # -- classification ------------------------------------------------------

    def classify_window(self, start: str, end: str) -> str:
        """Classify the closed interval ``[start, end]``.

        Returns ``"locked_holdout"`` when it overlaps the locked slice,
        ``"dev"`` when it ends at or before ``dev_cutoff``, and
        ``"future"`` otherwise.  Without a declared regime everything is
        ``"dev"`` (the legacy stance: all history is viewed data).
        """

        regime = self.regime
        if regime is None:
            return "dev"
        locked = regime.locked_slice
        if locked is not None and not (end < locked.start or start > locked.end):
            return "locked_holdout"
        if end <= regime.dev_cutoff:
            return "dev"
        return "future"

    # -- gates ---------------------------------------------------------------

    def assert_folds_clear(self, folds, dataset_id: str | None = None) -> None:
        """Reject any fold that would consume locked data (raise
        :class:`HoldoutViolation`).  Called before the first trial of a
        protocol run.

        A fold is clear only when it ends entirely before the locked
        slice: its train window starts at the beginning of history, so a
        fold reaching the lock trains on locked data even when its test
        window is clear.  A lock bound to a dataset id that differs from
        the current database is unverifiable and blocks the run.
        """

        regime = self.regime
        if regime is None:
            return
        locked = regime.locked_slice
        if locked is None:
            return
        if (
            dataset_id
            and locked.dataset_id
            and dataset_id != locked.dataset_id
        ):
            raise HoldoutViolation(
                f"locked slice {locked.start}..{locked.end} was declared on "
                f"dataset {locked.dataset_id}, current dataset is {dataset_id}; "
                "the lock cannot be verified on this data"
            )
        for fold in folds:
            train_end, test_end = _fold_window(fold)
            if test_end >= locked.start:
                raise HoldoutViolation(
                    f"fold {train_end} -> {test_end} reaches the "
                    f"locked holdout {locked.start}..{locked.end}; locked data "
                    "must never be consumed by train or test"
                )

    def assert_final_evaluation(self, folds) -> None:
        """Reject a final (promotion) evaluation whose windows are dev
        data: only future or locked windows qualify."""

        for fold in folds:
            train_end, test_end = _fold_window(fold)
            kind = self.classify_window(train_end, test_end)
            if kind == "dev":
                raise HoldoutViolation(
                    f"fold {train_end} -> {test_end} is dev/validation "
                    f"data (dev_cutoff {self.regime.dev_cutoff if self.regime else None}); "
                    "a final evaluation may only use future or strictly locked data"
                )


def main(argv=None) -> int:
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", required=True, help="registry JSON path")
    parser.add_argument("--dev-cutoff", required=True, help="last viewed date (YYYY-MM-DD)")
    parser.add_argument(
        "--lock", nargs=2, metavar=("START", "END"),
        help="additionally lock the closed interval START..END",
    )
    parser.add_argument("--dataset-id", default=None, help="dataset id the lock binds to")
    parser.add_argument("--note", default="", help="declaration note")
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing locked slice (a lock is a commitment; "
        "refused without --force)",
    )
    args = parser.parse_args(argv)

    path = Path(args.registry)
    registry = RegimeRegistry(path)
    if registry.regime is not None and registry.regime.locked_slice is not None:
        if args.lock and not args.force:
            print(
                "refusing to overwrite an existing locked slice "
                f"{registry.regime.locked_slice.start}..{registry.regime.locked_slice.end}; "
                "pass --force to replace it",
                file=sys.stderr,
            )
            return 1
    regime = declare_dev_regime(args.dev_cutoff)
    if args.lock:
        regime = lock_slice(
            regime,
            start=args.lock[0],
            end=args.lock[1],
            dataset_id=args.dataset_id,
            note=args.note,
        )
    RegimeRegistry(path, regime).save()
    print(f"regime declared: dev_cutoff={args.dev_cutoff} policy={regime.policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
