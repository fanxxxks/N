#!/usr/bin/env python3
"""Isolated cloud jobs. Planning imports no torch and starts no training."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from ashare_model.identity import content_hash

CLOUD_VERSION = 1
METHODS = {"random", "gp", "tpe", "rl"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # Cover effective source even in the development checkout. No result files.
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    names += ["scripts/run_cloud_search.py", "scripts/cloud_validate_data.py", "ashare_model/cloud_search.py"]
    files = {name: file_sha(ROOT / name) for name in sorted(set(names)) if name and
             (name.endswith(".py") or name.startswith("requirements")) and (ROOT / name).is_file()}
    return {"commit": commit, "source_digest": content_hash("cloud-source", files), "cloud_version": CLOUD_VERSION}


def load_settings(path, args):
    config_path = Path(path).resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if settings.get("version") != CLOUD_VERSION:
        raise ValueError("unsupported cloud configuration version")
    settings["_config_path"] = str(config_path)
    if args.validation_only:
        settings["evaluation_mode"] = "validation_only"
    for key in ("primary", "v2", "output"):
        value = getattr(args, key, None) or settings.get(key)
        if not value:
            raise ValueError(f"--{key} must name an actual path")
        settings[key] = str(Path(value).expanduser().resolve())
    for key in ("budget", "time_limit", "threads", "device", "horizon"):
        if getattr(args, key, None) is not None:
            settings[key] = getattr(args, key)
    if args.methods:
        settings["methods"] = args.methods.split(",")
    if args.seeds:
        settings["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.window:
        settings["windows"] = [w for w in settings["windows"] if w["name"] in args.window.split(",")]
    if not settings["windows"] or not settings["methods"] or not settings["seeds"]:
        raise ValueError("empty method/seed/window selection")
    if not set(settings["methods"]) <= METHODS:
        raise ValueError("unknown search method")
    if settings["budget"] <= 0 or settings["time_limit"] <= 0 or settings["threads"] <= 0:
        raise ValueError("budget, time limit and threads must be positive")
    if not 1 <= settings.get("candidates", 15) <= 20:
        raise ValueError("candidates must be between 1 and 20")
    if settings.get("attempt_multiplier", 20) < 1:
        raise ValueError("attempt_multiplier must be >= 1")
    if settings["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; assign physical cards with --gpus")
    if settings["horizon"] not in (1, 5, 10):
        raise ValueError("predefined matched prediction/rebalance horizons are 1, 5, 10")
    if settings.get("model", {}).get("max_formula_len", 12) < 2:
        raise ValueError("formula length must leave room for an operator")
    inputs = [Path(settings[k]) for k in ("primary", "v2")]
    if inputs[0] == inputs[1]:
        raise ValueError("primary and v2 must be different databases")
    for path in inputs:
        if not path.is_file():
            raise ValueError(f"input path missing: {path}")
        if Path(str(path) + ".wal").exists():
            raise ValueError(f"checkpoint a private copy before use: {path}.wal")
    output = Path(settings["output"])
    for path in inputs:
        if output == path.parent or output in path.parents or path.parent in output.parents:
            raise ValueError("output must be outside shared input directories")
    for window in settings["windows"]:
        bounds = [date.fromisoformat(str(window[k])) for k in
                  ("start", "train_end", "validation_end", "test_end")]
        if not all(a < b for a, b in zip(bounds, bounds[1:])):
            raise ValueError(f"non-chronological window: {window['name']}")
    return settings


def make_plan(settings):
    data = {key: {"sha256": file_sha(settings[key]), "bytes": Path(settings[key]).stat().st_size}
            for key in ("primary", "v2")}
    code = code_identity()
    excluded = {"primary", "v2", "output", "_config_path", "methods", "seeds", "windows"}
    common = {k: v for k, v in settings.items() if k not in excluded}
    tasks = []
    for window in settings["windows"]:
        for seed in settings["seeds"]:
            for method in settings["methods"]:
                identity = {"method": method, "seed": seed, "window": window,
                            "budget": settings["budget"], "data": data, "config": common, "code": code}
                task_id = content_hash("cloud-task", identity)
                tasks.append({"task_id": task_id, "identity": identity, "settings": settings,
                              "output": str(Path(settings["output"]) / task_id)})
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("duplicate task identities in plan")
    return tasks


def state_for(task):
    path = Path(task["output"]) / "state.json"
    if not path.exists():
        directory = path.parent
        if directory.exists() and any(p.name != "run.lock" for p in directory.iterdir()):
            raise ValueError(f"unowned non-empty task output: {directory}")
        return {"status": "pending", "attempts": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("identity") != task["identity"]:
        raise ValueError(f"task output identity conflict: {task['output']}")
    return state


def process_alive(pid):
    if os.name == "nt":
        # os.kill(pid, 0) terminates a Windows process; it is not a liveness
        # probe there. Use the declared psutil dependency instead.
        import psutil
        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@contextmanager
def claim_task(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "run.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # An interrupted parent leaves a lock. Never steal a live parent's lock.
        pids = json.loads(lock.read_text())
        pids = pids if isinstance(pids, list) else [pids]
        if not any(process_alive(pid) for pid in pids):
            lock.unlink()
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise ValueError(f"task already running: {directory}")
    os.write(fd, json.dumps([os.getpid()]).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def completed_valid(task, state):
    if state["status"] != "completed":
        return False
    attempt = Path(task["output"]) / state["latest_attempt"]
    return all((attempt / name).is_file() and file_sha(attempt / name) == checksum
               for name, checksum in state["artifacts"].items())


def kill_worker(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_task(task, gpu, retry=False):
    directory = Path(task["output"])
    with claim_task(directory):
        state = state_for(task)
        if completed_valid(task, state):
            return {"task_id": task["task_id"], "status": "skipped_completed"}
        if state["status"] == "completed":
            raise ValueError(f"completed artifacts missing or changed: {directory}")
        if state["status"] in ("failed", "timeout", "running") and not retry:
            return {"task_id": task["task_id"], "status": "retry_required"}
        attempt_name = f"attempt-{len(state['attempts']) + 1:04d}"
        attempt = directory / attempt_name
        attempt.mkdir(exist_ok=False)
        write_json(directory / "task.json", task)
        write_json(attempt / "request.json", {**task, "attempt_output": str(attempt), "gpu": gpu})
        state.update(identity=task["identity"], task_id=task["task_id"], status="running",
                     latest_attempt=attempt_name)
        state["attempts"].append({"directory": attempt_name, "started": time.time()})
        write_json(directory / "state.json", state)
        environment = os.environ.copy()
        threads = str(task["settings"]["threads"])
        environment.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                           OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
                           PYTHONUNBUFFERED="1", PYTHONUTF8="1")
        if gpu is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.monotonic()
        with (attempt / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(ROOT / "scripts/run_cloud_search.py"),
                "--worker", str(attempt / "request.json")], cwd=attempt, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"))
            if hasattr(process, "pid"):
                (directory / "run.lock").write_text(json.dumps([os.getpid(), process.pid]))
            try:
                code = process.wait(timeout=task["settings"]["time_limit"])
                status = "completed" if code == 0 and (attempt / "result.json").is_file() else "failed"
            except subprocess.TimeoutExpired:
                kill_worker(process)
                status = "timeout"
                code = None
        state.update(status=status, exit_code=code, elapsed_seconds=time.monotonic() - started)
        state["attempts"][-1].update(status=status, elapsed_seconds=state["elapsed_seconds"])
        if status == "completed":
            state["artifacts"] = {str(p.relative_to(attempt)): file_sha(p)
                for p in attempt.rglob("*") if p.is_file() and p.suffix in (".json", ".pt")}
        write_json(directory / "state.json", state)
        return {"task_id": task["task_id"], "status": status, "elapsed_seconds": state["elapsed_seconds"]}


def summarize(output):
    rows = []
    for path in Path(output).glob("*/state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        task = json.loads((path.parent / "task.json").read_text(encoding="utf-8"))
        attempt = path.parent / state["latest_attempt"]
        if state["status"] != "completed":
            partial = attempt / "search_progress.json"
            progress = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
            search = progress.get("search", {"requested_budget": task["identity"]["budget"],
                "consumed_budget": 0, "proposal_count": 0, "invalid_proposals": 0, "semantic_duplicates": 0})
            search["stop_reason"] = state["status"]
            rows.append({"task_id": state["task_id"], "identity": task["identity"],
                "method": task["identity"]["method"], "seed": task["identity"]["seed"],
                "window": task["identity"]["window"], "status": state["status"],
                "search": search, "curve": progress.get("curve", []), "partial_progress": True,
                "elapsed_seconds": state.get("elapsed_seconds"), "search_seconds": progress.get("search_seconds", 0)})
            continue
        if not completed_valid(task, state):
            raise ValueError(f"changed completed artifact: {path.parent}")
        result = json.loads((path.parent / state["latest_attempt"] / "result.json").read_text(encoding="utf-8"))
        rows.append({**result, "status": state["status"], "wall_seconds": state["elapsed_seconds"]})
    # Paired comparisons only: method is the sole varying identity field.
    groups = {}
    for row in rows:
        if "search" not in row:
            continue
        identity = dict(row["identity"])
        identity.pop("method")
        groups.setdefault(content_hash("cloud-paired", identity), []).append(row)
    comparisons = []
    for key, group in groups.items():
        common = min(r["search"]["consumed_budget"] for r in group)
        common_time = min(r["search_seconds"] for r in group)
        def at(curve, column, limit):
            values = [p["best_validation_score"] for p in curve if p[column] <= limit]
            return values[-1] if values else None
        comparisons.append({"paired_identity": key, "methods_present": [r["method"] for r in group],
            "all_four_present": {r["method"] for r in group} == METHODS,
            "common_completed_budget": common, "common_search_seconds": common_time,
            "rows": [{"method": r["method"],
                "best_at_common_budget": at(r["curve"], "effective_evaluations", common),
                "best_at_common_time": at(r["curve"], "elapsed_seconds", common_time)} for r in group]})
    payload = {"version": CLOUD_VERSION, "historical_research": True, "rows": rows, "comparisons": comparisons}
    write_json(Path(output) / "summary.json", payload)
    with (Path(output) / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["task_id", "method", "seed", "window", "status", "outcome", "requested_budget",
            "consumed_budget", "proposal_count", "semantic_duplicates", "invalid_proposals", "stop_reason",
            "search_seconds", "wall_seconds", "capital", "net_return", "benchmark_return", "net_excess_return",
            "max_drawdown", "actual_turnover", "actual_order_count", "total_fees", "no_trade_reason"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            common = {**row, **row["search"], "window": row["window"]["name"]}
            for account in row.get("test_accounts", []) or [{}]:
                report = account.get("account", {})
                writer.writerow({**common, "capital": account.get("capital"), **report,
                    "actual_turnover": report.get("actual_turnover_total"),
                    "total_fees": report.get("fees", {}).get("total")})
    return payload


def frequency_tasks(output, frequencies):
    """Select cadence using completed validation accounts, before any test."""
    groups = {}
    for path in Path(output).glob("*/state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        task = json.loads((path.parent / "task.json").read_text(encoding="utf-8"))
        if task["settings"].get("evaluation_mode") != "validation_only":
            continue
        if not completed_valid(task, state):
            raise ValueError(f"frequency selection requires every validation task completed: {path.parent}")
        result_path = path.parent / state["latest_attempt"] / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        paired = {**task["identity"], "config": dict(task["identity"]["config"])}
        paired["config"].pop("horizon")
        groups.setdefault(content_hash("frequency-paired", paired), []).append((task, result, result_path))
    jobs = []
    for group in groups.values():
        if sorted(t["settings"]["horizon"] for t, _, _ in group) != sorted(frequencies):
            raise ValueError("frequency group is missing or duplicates a predefined horizon")
        viable = [(t, r, p) for t, r, p in group if r.get("selected")]
        if not viable:
            continue
        task, result, source = sorted(viable, key=lambda x: (
            -x[1]["selected"]["validation_account"]["net_excess_return"], x[0]["settings"]["horizon"]))[0]
        selection = {"selected_task_id": task["task_id"], "selected_horizon": task["settings"]["horizon"],
            "criterion": "validation net excess at 50000; tie uses shorter horizon", "frozen_before_test": True,
            "validation_tasks": [{"task_id": t["task_id"], "result_sha256": file_sha(p),
                "horizon": t["settings"]["horizon"], "selected": r.get("selected")} for t, r, p in group],
            "search_cost": {"requested_budget": sum(r["search"]["requested_budget"] for _, r, _ in group),
                "consumed_budget": sum(r["search"]["consumed_budget"] for _, r, _ in group),
                "search_seconds": sum(r["search_seconds"] for _, r, _ in group)}}
        identity = {**task["identity"], "config": {**task["identity"]["config"], "evaluation_mode": "frequency_test"},
                    "frequency_selection": selection}
        job_id = content_hash("frequency-frozen-test", identity)
        jobs.append({**task, "settings": {**task["settings"], "evaluation_mode": "frequency_test"},
            "task_id": job_id, "identity": identity, "output": str(Path(output) / job_id),
            "test_from": {"path": str(source), "sha256": file_sha(source)}})
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/cloud_four_search.yaml"))
    for key in ("primary", "v2", "output", "device", "methods", "seeds", "window", "task"):
        parser.add_argument("--" + key)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--horizon", type=int, choices=(1, 5, 10))
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--select-frequency", action="store_true")
    parser.add_argument("--frequencies", default="1,5,10")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        from ashare_model.cloud_search import run_worker
        return run_worker(Path(args.worker))
    if args.summarize:
        if not args.output:
            parser.error("--summarize requires --output")
        payload = summarize(args.output)
        print(json.dumps({"rows": len(payload["rows"]), "summary": str(Path(args.output) / "summary.json")}))
        return 0
    if args.select_frequency:
        if not args.output:
            parser.error("--select-frequency requires --output")
        tasks = frequency_tasks(args.output, [int(n) for n in args.frequencies.split(",")])
        if not tasks:
            print(json.dumps({"task_count": 0, "outcome": "no_qualified_frequency_candidate"}))
            return 0
        settings = tasks[0]["settings"]
    else:
        settings = load_settings(args.config, args)
        tasks = make_plan(settings)
    if args.task:
        tasks = [t for t in tasks if t["task_id"].startswith(args.task)]
        if len(tasks) != 1:
            raise ValueError("--task must identify exactly one task in this plan")
    for task in tasks:
        state_for(task)  # refuse every output conflict before starting any job
    if args.rerun_failed:
        tasks = [t for t in tasks if state_for(t)["status"] in ("failed", "timeout", "running")]
    if args.plan:
        print(json.dumps({"task_count": len(tasks), "training_started": False,
            "data_validation": "pending (use --validate-data)", "tasks": [
            {k: t[k] for k in ("task_id", "identity", "output")} for t in tasks]}, ensure_ascii=False, indent=2))
        return 0
    if args.validate_data:
        command = [sys.executable, str(ROOT / "scripts/cloud_validate_data.py"),
                   "--settings-json", json.dumps(settings)]
        return subprocess.call(command, cwd=ROOT)
    if args.max_concurrent <= 0:
        raise ValueError("max-concurrent must be positive")
    gpus = args.gpus.split(",") if settings["device"] == "cuda" else [None] * args.max_concurrent
    if len(set(gpus)) != len(gpus) and settings["device"] == "cuda":
        raise ValueError("independent jobs require distinct GPUs")
    if len(gpus) < args.max_concurrent:
        raise ValueError("not enough GPU slots for max-concurrent")
    # One serial queue per GPU avoids concurrent processes sharing a card.
    queues = [tasks[i::args.max_concurrent] for i in range(args.max_concurrent)]
    def run_queue(index):
        return [run_task(t, gpus[index], args.retry or args.rerun_failed) for t in queues[index]]
    results = []
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        for future in as_completed([pool.submit(run_queue, i) for i in range(args.max_concurrent)]):
            results.extend(future.result())
    print(json.dumps(results, indent=2))
    summarize(settings["output"])
    return 1 if any(r["status"] in ("failed", "timeout", "retry_required") for r in results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
