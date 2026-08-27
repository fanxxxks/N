"""Searcher cost benchmark: time & peak RSS per unique semantic
evaluation (P1-04) and the small-budget smoke test (P1-05).

The four searchers (gp / tpe / random / rl) each run to completion on
the SAME capped window (``prepare_window`` window_cap — by default the
300x400 head of fold 0's training window), the SAME nominal budget and
the SAME seed.  The budget unit is the unique semantic formula
evaluation (T2-01 ledger; duplicates, degenerate and invalid formulas
never bill).  Non-RL searchers run through ``train_search`` with
(steps=budget, batch_size=1); RL runs ``train`` with the fixed split
(steps=4, batch=budget/4), so all four share the same nominal budget.

Per searcher the report records the actual ``unique_semantic_evals``,
wall-clock seconds, wall seconds **per 1000 evaluations**, peak RSS in
MB (stdlib polling: ``resource.ru_maxrss`` on POSIX, ctypes
``GetProcessMemoryInfo`` on Windows — zero new dependencies), the
completion flag and the selected validation reward.  The wall time
includes the identical ``prepare_window`` cost every searcher pays, so
per-searcher comparisons are fair by construction; the peak RSS is a
process-wide polling maximum, so a later searcher inherits the earlier
allocations (conservative; comparisons are within-process relative).

The smoke acceptance (P1-05): all four searchers complete under the
same small budget on the 300x400 capped window, one fold, one seed.

See docs/phase5_measurement_log.md §2 (SEARCHER_BENCH_VERSION 1).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from pathlib import Path

from loguru import logger

from ashare_data.config import (
    load_config,
    make_backtest_config,
    make_data_config,
    make_model_config,
    make_protocol_config,
    make_reward_config,
    make_sim_config,
)
from ashare_data.gates import ProductionGateRunner
from ashare_execution import validate_execution_config
from ashare_logging import export_log_txt, setup_run_logging

from .data_loader import AshareDataLoader
from .semantic_cache import SEMANTIC_CACHE_VERSION
from .train import AshareTrainer, resolve_device

SEARCHER_BENCH_VERSION = 1

SEARCHERS = ("gp", "tpe", "random", "rl")
DEFAULT_WINDOW_CAP = (300, 400)
MIN_BUDGET = 16  # the RL split needs budget >= 16 with budget % 4 == 0
RL_STEPS = 4


def current_rss_mb() -> float | None:
    """Process RSS in MB right now (stdlib only), or ``None`` on
    unsupported platforms.  POSIX reads ``resource.ru_maxrss`` (the
    peak so far — exactly the "peak RSS" the report wants); Windows
    reads the working-set size via ``GetProcessMemoryInfo``."""

    if sys.platform == "win32":
        try:

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            # GetProcessMemoryInfo is exported by psapi.dll (kernel32 only
            # re-exports the K32-prefixed alias on modern Windows).
            get_mem = getattr(
                ctypes.windll.psapi, "GetProcessMemoryInfo", None
            ) or getattr(
                ctypes.windll.kernel32, "GetProcessMemoryInfo", None
            )
            if get_mem is None:
                return None
            # Explicit argtypes are required: without them the pseudo-handle
            # is passed as a 32-bit int and the call fails with 0.
            get_mem.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_Counters),
                ctypes.c_ulong,
            ]
            get_mem.restype = ctypes.c_int
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.argtypes = []
            get_current_process.restype = ctypes.c_void_p
            handle = get_current_process()
            if get_mem(handle, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024.0 * 1024.0)
        except Exception:  # pragma: no cover - defensive
            return None
        return None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        divisor = 1024.0 if sys.platform.startswith("linux") else 1024.0 * 1024.0
        return rss / divisor
    except Exception:  # pragma: no cover - defensive
        return None


def measure_peak_rss(
    fn, interval: float = 0.02
) -> tuple[object, float | None]:
    """Run ``fn`` while polling RSS on a daemon thread; returns
    ``(fn_result, peak_rss_mb)`` (``None`` peak on unsupported
    platforms)."""

    peak: dict[str, float | None] = {"value": None}
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            rss = current_rss_mb()
            if rss is not None:
                peak["value"] = (
                    rss if peak["value"] is None else max(peak["value"], rss)
                )
            stop.wait(interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join()
    return result, peak["value"]


def rl_split(budget: int) -> tuple[int, int]:
    """(steps, batch_size) that multiply to ``budget`` for the RL
    searcher: the fixed (4, budget/4) split keeps the same nominal
    budget for every searcher."""

    budget = int(budget)
    if budget < MIN_BUDGET or budget % RL_STEPS != 0:
        raise ValueError(
            f"budget must be >= {MIN_BUDGET} and divisible by {RL_STEPS}, "
            f"got {budget}"
        )
    return RL_STEPS, budget // RL_STEPS


def benchmark_searchers(
    data_config,
    model_config,
    backtest_config,
    reward_config,
    loader: AshareDataLoader,
    *,
    searchers: tuple[str, ...] = SEARCHERS,
    budget: int,
    seed: int,
    train_end_date: str | None,
    window_cap: tuple[int, int],
    device: str | None = None,
) -> dict:
    """Run every searcher on the identical capped window under the
    identical nominal budget and seed, measuring wall time and peak RSS.

    Returns the versioned, JSON-serializable report payload with one row
    per searcher.  A searcher crash is recorded (``completed`` False with
    the exception text), never raised — a measurement must not lose a
    failed row.
    """

    if not searchers:
        raise ValueError("searchers must not be empty")
    unknown = [s for s in searchers if s not in SEARCHERS]
    if unknown:
        raise ValueError(f"unknown searchers: {unknown} (choose from {SEARCHERS})")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    rl_steps, rl_batch = rl_split(budget)
    vm_device = resolve_device(device)

    rows: dict[str, dict] = {}
    for searcher in searchers:
        trainer = AshareTrainer(
            data_config,
            model_config,
            backtest_config,
            loader=loader,
            reward_config=reward_config,
        )
        if searcher == "rl":
            steps, batch = rl_steps, rl_batch

            def run() -> list[int] | None:
                return trainer.train(
                    steps=steps,
                    batch_size=batch,
                    seed=seed,
                    save_artifacts=False,
                    train_end_date=train_end_date,
                    window_cap=window_cap,
                    device=str(vm_device),
                )

        else:
            steps, batch = budget, 1

            def run() -> list[int] | None:
                return trainer.train_search(
                    searcher=searcher,
                    steps=steps,
                    batch_size=batch,
                    seed=seed,
                    save_artifacts=False,
                    train_end_date=train_end_date,
                    window_cap=window_cap,
                    device=str(vm_device),
                )

        logger.info(
            "benchmark searcher={} budget={} seed={} window_cap={}",
            searcher,
            budget,
            seed,
            window_cap,
        )
        error: str | None = None
        wall = 0.0
        peak = None
        try:
            started = time.perf_counter()
            _, peak = measure_peak_rss(run)
            wall = time.perf_counter() - started
        except Exception as exc:  # a failed row is a measurement too
            error = f"{type(exc).__name__}: {exc}"
            logger.error("searcher {} failed: {}", searcher, error)
        evals = int(getattr(trainer, "semantic_cache", None).budget_used or 0)
        best = getattr(trainer, "best_val_reward", None)
        if best is None:
            best = getattr(trainer, "best_reward", None)
        rows[searcher] = {
            "searcher": searcher,
            "budget": budget,
            "steps": steps,
            "batch_size": batch,
            "unique_semantic_evals": evals,
            "wall_seconds": wall,
            "wall_per_1000_evals": (
                wall / evals * 1000.0 if evals > 0 else None
            ),
            "peak_rss_mb": peak,
            "completed": error is None,
            "selected_val_reward": (
                float(best) if best is not None else None
            ),
            "error": error,
        }
        logger.info(
            "benchmark searcher={} evals={} wall={:.1f}s "
            "per_1000={} peak_rss_mb={} completed={}",
            searcher,
            evals,
            wall,
            rows[searcher]["wall_per_1000_evals"],
            peak,
            error is None,
        )

    policy = getattr(loader, "universe_policy", None)
    return {
        "version": SEARCHER_BENCH_VERSION,
        "provenance": {
            "dataset_id": loader.dataset_id,
            "window_cap": [int(window_cap[0]), int(window_cap[1])],
            "train_end_date": train_end_date,
            "seed": seed,
            "budget": budget,
            "searchers": list(searchers),
            "device": str(vm_device),
            "max_formula_len": int(model_config.max_formula_len),
            "semantic_cache_version": SEMANTIC_CACHE_VERSION,
            "backtest": {
                "top_n": int(backtest_config.top_n),
                "single_weight_cap": float(backtest_config.single_weight_cap),
                "initial_capital": float(backtest_config.initial_capital),
            },
            "universe_policy": (
                {
                    "index_codes": [str(code) for code in policy.index_codes],
                    "min_listed_sessions": int(policy.min_listed_sessions),
                    "membership_end_inclusive": bool(
                        policy.membership_end_inclusive
                    ),
                    "degraded": (
                        bool(loader.universe_status.degraded)
                        if loader.universe_status is not None
                        else None
                    ),
                }
                if policy is not None
                else None
            ),
        },
        "rows": rows,
    }


def _parse_window_cap(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"window-cap must be '<stocks>x<dates>', got {text!r}")
    stocks, dates = int(parts[0]), int(parts[1])
    if stocks <= 0 or dates <= 0:
        raise ValueError("window-cap must be positive (stocks, dates)")
    return stocks, dates


def main(argv=None) -> int:
    setup_run_logging(run_name="searcher_bench")
    parser = argparse.ArgumentParser(
        description="Searcher cost benchmark: time & peak RSS per 1000 "
        "unique semantic evaluations (P1-04) and the small-budget smoke "
        "test (P1-05)"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data/searcher_bench.json")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fold", type=int, default=0, help="protocol fold index (default 0)"
    )
    parser.add_argument(
        "--window-cap",
        default="300x400",
        help="capped window '<stocks>x<dates>' (default 300x400)",
    )
    parser.add_argument(
        "--searchers",
        default=",".join(SEARCHERS),
        help="comma-separated searchers (default gp,tpe,random,rl)",
    )
    parser.add_argument(
        "--min-eligible",
        type=int,
        default=None,
        help="production gate G6: minimum eligible stocks per major window "
        "(default: 100)",
    )
    args = parser.parse_args(argv)

    try:
        root = Path(__file__).resolve().parents[1]
        raw = load_config(args.config, project_root=root)
        data_config = make_data_config(raw, root)
        ProductionGateRunner(
            data_config, min_eligible=args.min_eligible
        ).require_production()
        model_config = make_model_config(raw)
        backtest_config = make_backtest_config(raw)
        reward_config = make_reward_config(raw)
        sim_config = make_sim_config(raw, root)
        validate_execution_config(backtest_config, sim_config)
        proto_cfg = make_protocol_config(raw)
        fold_index = int(args.fold)
        if not 0 <= fold_index < len(proto_cfg.folds):
            raise ValueError(
                f"fold index out of range: {fold_index} "
                f"(config has {len(proto_cfg.folds)} folds)"
            )
        train_end_date = proto_cfg.folds[fold_index].train_end
        window_cap = _parse_window_cap(args.window_cap)
        searchers = tuple(s.strip() for s in args.searchers.split(",") if s.strip())

        loader = AshareDataLoader(data_config, model_config)
        loader.load_data()
        payload = benchmark_searchers(
            data_config,
            model_config,
            backtest_config,
            reward_config,
            loader,
            searchers=searchers,
            budget=args.budget,
            seed=args.seed,
            train_end_date=train_end_date,
            window_cap=window_cap,
        )
        out_path = root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.success(f"Searcher bench written to {out_path}")
        print(
            f"{'searcher':<8} {'evals':>6} {'wall_s':>9} "
            f"{'per_1000':>9} {'rss_mb':>9} {'done':>6}"
        )
        for searcher in searchers:
            row = payload["rows"][searcher]
            per_1000 = (
                f"{row['wall_per_1000_evals']:.1f}"
                if row["wall_per_1000_evals"] is not None
                else "-"
            )
            rss = f"{row['peak_rss_mb']:.0f}" if row["peak_rss_mb"] else "-"
            print(
                f"{searcher:<8} {row['unique_semantic_evals']:>6} "
                f"{row['wall_seconds']:>9.1f} {per_1000:>9} {rss:>9} "
                f"{str(row['completed']):>6}"
            )
    finally:
        export_log_txt(run_name="searcher_bench")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
