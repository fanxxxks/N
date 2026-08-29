"""Stamp legacy strategy/protocol/bare-factor artifacts in the data directory.

Migration tool (P0-04): old artifacts (reward-v10 strategy, protocol-v12
result) get an explicit ``legacy: true`` marker with reasons so nothing
treats them as the current champion.  The classification rules live in
``ashare_model.artifact_versions``; this script only loads the config and
applies them.  Idempotent — safe to run any time after a data change.

Usage:
    python scripts/stamp_legacy_artifacts.py [--config PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Scripts live one level below the package root: make `ashare_*` importable
# regardless of the caller's sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ashare_data.config import load_config, make_data_config  # noqa: E402
from ashare_model.artifact_versions import stamp_legacy_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="path to ashare_config.yaml")
    args = parser.parse_args()

    config = make_data_config(load_config(args.config, project_root=_ROOT), _ROOT)
    outcomes = stamp_legacy_artifacts(config.data_dir)
    for outcome in outcomes:
        status = (
            "stamped legacy"
            if outcome["stamped"]
            else "already legacy"
            if outcome["legacy"]
            else "current (untouched)"
        )
        print(f"{outcome['name']:8s} {outcome['path']}: {status}")
        if outcome["reasons"]:
            for reason in outcome["reasons"]:
                print(f"           - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
