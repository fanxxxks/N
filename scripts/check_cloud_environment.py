#!/usr/bin/env python3
"""Read-only runtime checks before a cloud campaign."""
import argparse
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--require-python312", action="store_true")
    args = parser.parse_args()
    import torch
    import optuna
    import duckdb
    import psutil
    report = {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "optuna": optuna.__version__, "duckdb": duckdb.__version__, "cpu_count": psutil.cpu_count(),
        "ram_gib": round(psutil.virtual_memory().total / 2**30, 1), "cuda_available": torch.cuda.is_available(),
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
            "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
            for i in range(torch.cuda.device_count())]}
    print(json.dumps(report, indent=2))
    pip_status = subprocess.call([sys.executable, "-m", "pip", "check"])
    ok = pip_status == 0 and len(report["gpus"]) == args.expected_gpus
    if args.expected_gpus:
        ok = ok and report["cuda_available"]
        for i in range(args.expected_gpus):
            if i < torch.cuda.device_count():
                with torch.cuda.device(i):
                    torch.empty(1, device=f"cuda:{i}").add_(1)
                    torch.cuda.synchronize(i)
    if args.require_python312:
        ok = ok and sys.version_info[:2] == (3, 12)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
