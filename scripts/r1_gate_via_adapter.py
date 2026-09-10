"""Empirical formal G1-G8 check via the gate adapter (p21 runbook step).

Read-only with respect to research data: the adapter materializes
_gate_* tables in the v2 store (the sanctioned second-workspace pattern,
captain ruling) and drops them afterwards.  Not a research run.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ashare_data.config import load_config, make_data_config, resolve_runtime_paths
from ashare_data.db import AshareDB
from ashare_data.gates import ProductionGateRunner
from ashare_data.tushare_store import TushareStore

# A01 (t45): both database paths come from the single config resolver instead
# of machine-absolute constants.  The previous V2 constant pointed at a retired
# second-workspace database path that no longer exists.  resolve_runtime_paths
# refuses a missing required input (InputPathMissing) before anything is
# opened, so this adapter fails closed instead of silently building a store
# over a nonexistent path.
_preflight = resolve_runtime_paths(
    _REPO_ROOT / "configs" / "r1_medium_a.yaml", project_root=_REPO_ROOT
)
LEGACY = str(_preflight.duckdb_path)
V2 = str(_preflight.v2_duckdb_path)


def main() -> int:
    raw = load_config("configs/r1_medium_a.yaml", project_root=_REPO_ROOT)
    base = make_data_config(raw, _REPO_ROOT)
    db = AshareDB(V2)
    store = TushareStore(
        db,
        raw_root=None,
        session_resolver=None,
        code_commit="r1-gate-check",
        sdk_version="n/a",
    )
    try:
        gate_config = store.prepare_gate_tables(LEGACY, base)
    finally:
        # DuckDB cannot mix read-write and read-only configurations on one
        # file inside a process: close the adapter's write connection so the
        # formal gate runner can open the same file read-only.
        db.close()

    try:
        runner = ProductionGateRunner(gate_config)
        result = runner.run("formal")
        print("formal G1-G8 via adapter ok =", result.ok)
        for check in result.checks:
            mark = "PASS" if check.ok else "FAIL"
            print(f"[{mark}] {check.detail[:150]}")
        ok = result.ok
    finally:
        db2 = AshareDB(V2)
        store2 = TushareStore(
            db2,
            raw_root=None,
            session_resolver=None,
            code_commit="r1-gate-check",
            sdk_version="n/a",
        )
        try:
            store2.cleanup_gate_tables()
        except Exception as exc:  # noqa: BLE001 - cleanup evidence, not gate
            print("cleanup error:", exc)
        db2.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
