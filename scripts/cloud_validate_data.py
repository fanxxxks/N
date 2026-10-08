#!/usr/bin/env python3
"""Read-only snapshot confirmation, followed by historical G1-G8 and D7."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate(settings, snapshot_only=False):
    # Keep full manifest hashing ahead of the torch runtime import.
    from ashare_data.config import DataConfig
    from ashare_data.db import AshareDB
    from ashare_model.runspec import confirm_composite_snapshot
    from ashare_data.universe import resolve_universe_contract
    windows = settings["windows"]
    config = DataConfig(duckdb_path=Path(settings["primary"]), v2_duckdb_path=Path(settings["v2"]),
        data_dir=Path(settings["output"]) / "validation-cache",
        start_date=min(w["start"] for w in windows), end_date=max(w["test_end"] for w in windows),
        index_codes=settings["index_codes"], min_listed_sessions=settings.get("min_listed_sessions", 60))
    with AshareDB(config.duckdb_path, read_only=True) as primary, AshareDB(config.v2_duckdb_path, read_only=True) as v2:
        snapshot = confirm_composite_snapshot(primary, config, v2)
    if snapshot.composite_id != settings["dataset_id"]:
        raise ValueError(f"dataset identity mismatch: confirmed {snapshot.composite_id}")
    source_end = snapshot.v2.table("daily_bar_raw").date_range[1]
    contract = resolve_universe_contract(config, allow_development_fallback=False)
    cutoff = max(s for s in contract.sessions if s <= config.end_date.replace("-", ""))
    if config.end_date.replace("-", "") > source_end.replace("-", ""):
        raise ValueError("test boundary exceeds confirmed source history")
    result = {"dataset_id": snapshot.composite_id, "publication_id": snapshot.publication_id,
        "primary_manifest": snapshot.primary.to_dict(), "v2_manifest": snapshot.v2.to_dict(),
        "source_cutoff": source_end, "data_cutoff": cutoff, "purpose": "historical_research"}
    if snapshot_only:
        return result
    from ashare_data.gates import ProductionGateRunner
    from ashare_model.data_eligibility import check_runspec_data_eligibility
    gate = ProductionGateRunner(config, min_eligible=settings.get("min_eligible", 100)).run(
        mode="formal", purpose="historical_research", snapshot_cutoff=cutoff)
    result["gates"] = asdict(gate)
    if not gate.ok:
        raise ValueError("historical gates failed: " + "; ".join(c.detail for c in gate.checks if not c.ok))
    checks = []
    for window in windows:
        spec = SimpleNamespace(spec_id=window["name"], purpose="historical_research", data_cutoff=cutoff,
            dev_window=SimpleNamespace(start=window["start"].replace("-", ""), end=window["train_end"].replace("-", "")))
        verdict = check_runspec_data_eligibility(spec, settings["feature_names"], list(contract.codes), config)
        checks.append(asdict(verdict))
        if not verdict.ok:
            raise ValueError("source eligibility failed: " + "; ".join(verdict.reasons))
    result["source_eligibility"] = checks
    result["ok"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings-json")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()
    if args.request:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        settings = {**request["settings"], "windows": [request["identity"]["window"]]}
    else:
        settings = json.loads(args.settings_json)
    try:
        result = validate(settings, args.snapshot_only)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
