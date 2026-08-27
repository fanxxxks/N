"""P2-01: fundamentals-table scope audit and purge.

The ``fundamental_pit`` table must only hold rows for valid A-share codes
inside the persisted stock scope (stocks ∪ PIT constituents ∪ cached
daily bars).  ``--report`` prints the audit; ``--purge`` deletes the
out-of-scope rows and writes the measured before/after counts to
``data/fundamental_scope.json`` (also reports the ``stocks``-table rows
that fail the A-share validator — legacy B-shares etc.).

Usage:
    python scripts/check_fundamental_scope.py --report
    python scripts/check_fundamental_scope.py --purge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    args = parser.parse_args()

    raw = load_config(args.config, project_root=_ROOT)
    data_config = make_data_config(raw, _ROOT)
    with AshareDB(data_config.duckdb_path) as db:
        stats = fundamental_scope_stats(db, data_config)
        invalid_stocks = _stocks_invalid_codes(db, data_config)
        stats["invalid_stocks_rows"] = len(invalid_stocks)
        stats["invalid_stocks_codes"] = invalid_stocks
        scope_count = len(persisted_scope_codes(db, data_config))

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
            result = purge_out_of_scope_fundamentals(db, data_config)
            # The stocks table is the persisted stock scope: legacy
            # non-A-share rows (B-shares) are deleted, never kept.
            stocks_before = int(
                db.query(
                    f"SELECT count(*) AS n FROM {data_config.stocks_table}"
                )["n"].iloc[0]
            )
            stocks_removed = 0
            for code in invalid_stocks:
                db.execute(
                    f"DELETE FROM {data_config.stocks_table} WHERE ts_code = ?",
                    [code],
                )
                stocks_removed += 1
            stocks_after = stocks_before - stocks_removed
            payload = {
                "scope": {
                    "total_rows": stats["total_rows"],
                    "in_scope_rows": stats["in_scope_rows"],
                    "out_of_scope_rows": stats["out_of_scope_rows"],
                    "out_of_scope_codes": stats["out_of_scope_codes"],
                    "scope_codes_count": scope_count,
                },
                "purge": result,
                "stocks_purge": {
                    "removed_rows": stocks_removed,
                    "rows_before": stocks_before,
                    "rows_after": stocks_after,
                },
            }
            out_path = _ROOT / args.output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(out_path, payload)
            print(
                f"purged {result['removed_rows']} fundamental rows "
                f"({result['rows_before']} -> {result['rows_after']}) and "
                f"{stocks_removed} invalid stocks rows "
                f"({stocks_before} -> {stocks_after}); "
                f"audit written to {out_path}"
            )
        return 0


if __name__ == "__main__":
    sys.exit(main())
