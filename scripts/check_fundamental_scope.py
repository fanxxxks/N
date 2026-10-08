"""P2-01: fundamentals-table scope audit and purge.

The ``fundamental_pit`` table must only hold rows for valid A-share codes
inside the persisted stock scope (stocks ∪ PIT constituents ∪ cached
daily bars).  ``--report`` prints the audit; ``--purge`` first dumps every
row it is about to delete (the out-of-scope ``fundamental_pit`` rows and
the invalid ``stocks`` rows) to a JSON backup under ``data/backups/``
(override with ``--backup-dir``) and aborts without deleting when the
backup cannot be written (fail-closed), then deletes and writes the
measured before/after counts to ``data/fundamental_scope.json`` (also
reports the ``stocks``-table rows that fail the A-share validator —
legacy B-shares etc.).

Usage:
    python scripts/check_fundamental_scope.py --report
    python scripts/check_fundamental_scope.py --purge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# When invoked as `python scripts/check_fundamental_scope.py`, sys.path[0]
# is the scripts dir; make the package roots importable first.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_data.config import load_config, make_data_config
from ashare_data.db import AshareDB
from ashare_data.fundamentals import (
    fundamental_scope_stats,
    persisted_scope_codes,
    purge_out_of_scope_fundamentals,
)
from ashare_data.io_utils import atomic_write_json
from ashare_data.processor import is_valid_a_share_code


class BackupError(RuntimeError):
    """A pre-purge backup could not be written; the purge must not run
    (fail-closed)."""


def _resolve_backup_dir(raw: str | Path | None) -> Path:
    """Backup root: ``data/backups/`` under the project root by default
    (same convention as the constituents-rebuild backup in
    ``scripts/import_pit_universe.py``)."""
    if raw is None:
        return _ROOT / "data" / "backups"
    path = Path(raw)
    return path if path.is_absolute() else _ROOT / path


def _json_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame rows as JSON-safe dicts (NULL/NaN become ``None``)."""
    if df is None or len(df) == 0:
        return []
    cleaned = df.astype(object).where(pd.notna(df), None)
    return cleaned.to_dict("records")


def write_backup_json(path: Path, payload: dict) -> None:
    """Atomically persist ``payload`` and verify it reads back.

    Any failure raises :class:`BackupError` so callers abort BEFORE any
    DELETE runs (fail-closed).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(text):
            raise BackupError(f"backup verification mismatch for {path}")
    except BackupError:
        raise
    except (OSError, ValueError) as exc:  # noqa: BLE001 - fail closed
        raise BackupError(f"backup write failed for {path}: {exc}") from exc


def _fundamentals_rows_to_purge(
    db, config, scope: set[str]
) -> pd.DataFrame:
    """The ``fundamental_pit`` rows ``--purge`` will delete.

    Mirrors the exact predicate of ``purge_out_of_scope_fundamentals``
    (SQL-injection hardening: verbatim for the registered default table,
    scope through a bound parameter).
    """
    if config.fundamentals_table != "fundamental_pit":
        raise ValueError(
            "no literal purge statements for fundamentals table "
            f"{config.fundamentals_table!r}"
        )
    try:
        if scope:
            return db.query(
                "SELECT * FROM fundamental_pit "
                "WHERE ts_code NOT IN (SELECT unnest(?))",
                [sorted(scope)],
            )
        return db.query("SELECT * FROM fundamental_pit")
    except Exception:  # noqa: BLE001 - table may not exist yet.
        return pd.DataFrame()


def _stocks_rows_to_delete(
    db, config, codes: list[str]
) -> pd.DataFrame:
    """The ``stocks`` rows ``--purge`` will delete (invalid codes)."""
    if not codes:
        return pd.DataFrame()
    try:
        return db.query(
            f"SELECT * FROM {config.stocks_table} "
            "WHERE ts_code IN (SELECT unnest(?))",
            [sorted(codes)],
        )
    except Exception:  # noqa: BLE001 - table may not exist yet.
        return pd.DataFrame()


def run_purge(
    db,
    config,
    *,
    scope: set[str],
    invalid_stocks: list[str],
    backup_dir: Path,
) -> dict[str, object]:
    """Backup the rows ``--purge`` will delete, then delete them.

    Fail-closed (P2 review follow-up): the backup file must be
    verifiably on disk before any DELETE runs — when
    :func:`write_backup_json` raises :class:`BackupError` neither the
    fundamentals purge nor the stocks deletes are executed.
    """
    fundamentals_backup = _json_records(
        _fundamentals_rows_to_purge(db, config, scope)
    )
    stocks_backup = _json_records(_stocks_rows_to_delete(db, config, invalid_stocks))
    backup_path = backup_dir / (
        "fundamental_purge_backup_"
        + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    write_backup_json(
        backup_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "fundamental_pit": {
                "table": config.fundamentals_table,
                "row_count": len(fundamentals_backup),
                "rows": fundamentals_backup,
            },
            "stocks": {
                "table": config.stocks_table,
                "row_count": len(stocks_backup),
                "rows": stocks_backup,
            },
            "invalid_stocks_codes": list(invalid_stocks),
        },
    )
    print(
        f"backup: {len(fundamentals_backup)} fundamental rows + "
        f"{len(stocks_backup)} stocks rows -> {backup_path}"
    )

    result = purge_out_of_scope_fundamentals(db, config)
    # The stocks table is the persisted stock scope: legacy
    # non-A-share rows (B-shares) are deleted, never kept.
    stocks_before = int(
        db.query(
            f"SELECT count(*) AS n FROM {config.stocks_table}"
        )["n"].iloc[0]
    )
    stocks_removed = 0
    for code in invalid_stocks:
        db.execute(
            f"DELETE FROM {config.stocks_table} WHERE ts_code = ?",
            [code],
        )
        stocks_removed += 1
    stocks_after = stocks_before - stocks_removed
    return {
        "purge": result,
        "stocks_purge": {
            "removed_rows": stocks_removed,
            "rows_before": stocks_before,
            "rows_after": stocks_after,
        },
        "backup_path": str(backup_path),
        "backup_rows": {
            "fundamental_pit": len(fundamentals_backup),
            "stocks": len(stocks_backup),
        },
    }


def _stocks_invalid_codes(db, config) -> list[str]:
    try:
        df = db.query(f"SELECT DISTINCT ts_code FROM {config.stocks_table}")
    except Exception:  # noqa: BLE001 - table may not exist yet.
        return []
    return sorted(
        str(code) for code in df["ts_code"] if not is_valid_a_share_code(str(code))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the scope audit (default when --purge is absent)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="delete out-of-scope fundamentals rows and write "
        "data/fundamental_scope.json with before/after counts",
    )
    parser.add_argument(
        "--output", default="data/fundamental_scope.json"
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="directory for the pre-purge JSON backup of the rows --purge "
        "will delete (default: data/backups/ under the project root); "
        "the purge aborts when the backup cannot be written",
    )
    args = parser.parse_args()

    raw = load_config(args.config, project_root=_ROOT)
    data_config = make_data_config(raw, _ROOT)
    with AshareDB(data_config.duckdb_path) as db:
        stats = fundamental_scope_stats(db, data_config)
        invalid_stocks = _stocks_invalid_codes(db, data_config)
        stats["invalid_stocks_rows"] = len(invalid_stocks)
        stats["invalid_stocks_codes"] = invalid_stocks
        scope = persisted_scope_codes(db, data_config)
        scope_count = len(scope)

        print(
            f"fundamental_pit: {stats['total_rows']} rows / "
            f"{stats['out_of_scope_codes_count']} out-of-scope codes "
            f"({stats['out_of_scope_rows']} rows); scope = {scope_count} codes"
        )
        if invalid_stocks:
            print(
                f"stocks table: {len(invalid_stocks)} non-A-share rows "
                f"(e.g. {invalid_stocks[:5]})"
            )
        if stats["out_of_scope_rows"] or invalid_stocks:
            print("run with --purge to remove them")
        else:
            print("scope OK: no out-of-scope rows, no invalid stocks rows")

        if args.purge:
            try:
                purged = run_purge(
                    db,
                    data_config,
                    scope=scope,
                    invalid_stocks=invalid_stocks,
                    backup_dir=_resolve_backup_dir(args.backup_dir),
                )
            except BackupError as exc:
                raise SystemExit(
                    f"purge aborted — backup failed: {exc}"
                ) from exc
            payload = {
                "scope": {
                    "total_rows": stats["total_rows"],
                    "in_scope_rows": stats["in_scope_rows"],
                    "out_of_scope_rows": stats["out_of_scope_rows"],
                    "out_of_scope_codes": stats["out_of_scope_codes"],
                    "scope_codes_count": scope_count,
                },
                "purge": purged["purge"],
                "stocks_purge": purged["stocks_purge"],
                "backup": {
                    "path": purged["backup_path"],
                    "rows": purged["backup_rows"],
                },
            }
            out_path = _ROOT / args.output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out_path, payload)
            print(
                f"purged {purged['purge']['removed_rows']} fundamental rows "
                f"({purged['purge']['rows_before']} -> "
                f"{purged['purge']['rows_after']}) and "
                f"{purged['stocks_purge']['removed_rows']} invalid stocks "
                f"rows ({purged['stocks_purge']['rows_before']} -> "
                f"{purged['stocks_purge']['rows_after']}); "
                f"backup: {purged['backup_path']}; "
                f"audit written to {out_path}"
            )
        return 0


if __name__ == "__main__":
    sys.exit(main())
