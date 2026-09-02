"""Searcher cost benchmark: time & peak RSS per unique semantic
evaluation (P1-04) and the small-budget smoke test (P1-05), extended in
P10 with the pre-registered multi-seed matched campaign runner.

The four searchers (gp / tpe / random / rl) each run to completion on
the SAME capped window (``prepare_window`` window_cap — by default the
300x400 head of fold 0's training window), the SAME nominal budget and
the SAME seed.  The budget unit is the unique semantic formula
evaluation (T2-01 ledger; duplicates, degenerate and invalid formulas
never bill).  Non-RL searchers run through ``train_search`` with
(steps=budget, batch_size=1); RL runs ``train`` with a fixed step/batch
split (steps x batch = budget), so all four share the same nominal
budget.

Per searcher the report records the actual ``unique_semantic_evals``,
wall-clock seconds, wall seconds **per 1000 evaluations**, peak RSS in
MB (stdlib polling), the completion flag, the selected validation
reward, the termination/stagnation reason and the evaluated-formula
length histogram (content tokens + EOS).  The wall time includes the
identical ``prepare_window`` cost every searcher pays, so per-searcher
comparisons are fair by construction.

P10 campaign mode (``benchmark_campaign``; SEARCHER_BENCH_VERSION 3):
seeds x searchers rows with a per-row fresh trainer/cache, an
append-only ``ExperimentLedger`` trial per row attempt, atomic per-row
campaign JSON persistence with resume support, and the fail-closed
circuit breakers of docs/p10_searcher_fairness_contract.md §4.5/§6
(wall cap checked between rows, post-seed-block calibration-deviation
stop, identity-drift refusal).  Campaign identity (budget/seeds/window/
fold/rl-steps/dataset_id) must match on resume; failed rows are retried
as new ledger trials while every prior entry stays in the chain.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import math
import os
import subprocess
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

from .admission import ADMISSION_WINDOW
from .data_loader import AshareDataLoader
from .data_tier import (
    DATA_TIER_VERSION,
    feature_tier,
    formula_data_tier_report,
)
from .ir import canonical_tokens
from .ledger import ExperimentLedger
from .runspec import new_run_id
from .search_length_prior import LENGTH_PRIOR_PROFILE as P14_LENGTH_PRIOR_PROFILE
from .semantic_cache import SEMANTIC_CACHE_VERSION
from .train import AshareTrainer, resolve_device
from .vocab import FORMULA_VOCAB

# v4 (P14): campaign rows carry the research/promotion_tier_a track
# dimension with per-track budgets and the proposal length-prior profile
# (docs/p14_search_digest_preregistration.md §5.3/§5.4); v3 payloads stay
# read-only legacy and are never matched against v4 campaigns.
SEARCHER_BENCH_VERSION = 4

SEARCHERS = ("gp", "tpe", "random", "rl")
DEFAULT_WINDOW_CAP = (300, 400)
MIN_BUDGET = 16  # the RL split needs budget >= 16 with budget % steps == 0
RL_STEPS = 4

# ---------------------------------------------------------------------------
# P14 pre-registered search-digest constants
# (docs/p14_search_digest_preregistration.md §5.4 — the research/promotion
# budget separation; the contract is the single authority).
# ---------------------------------------------------------------------------
P14_TRACKS = ("research", "promotion_tier_a")
P14_RESEARCH_BUDGET = 1200
P14_PROMOTION_BUDGET = 800

# ---------------------------------------------------------------------------
# P10 pre-registered campaign constants
# (docs/p10_searcher_fairness_contract.md §4.1 — approved 2026-09-01; the
# contract is the single authority, these are its executable values).
# ---------------------------------------------------------------------------
P10_COMPARE_BUDGET = 2000
P10_COMPARE_SEEDS = (42, 7, 2024)
P10_COMPARE_WINDOW = ADMISSION_WINDOW  # (300, 400), the P4 admission cap
P10_COMPARE_FOLD = 0
P10_RL_STEPS = 8  # ADMISSION_STEPS precedent (bench v2 fixed 4)
P10_ROW_ORDER = SEARCHERS
# Appendix A engineering calibration (CPU host, budget 128, seed 42):
# seconds per unique semantic evaluation per searcher.
P10_CALIBRATION_S_PER_EVAL: dict[str, float] = {
    "gp": 0.657,
    "tpe": 1.012,
    "random": 0.562,
    "rl": 0.676,
}
P10_DEVIATION_STOP_RATIO = 2.0
P10_STAGE_A_WALL_CAP_S = 7 * 3600


def _git_commit(repo_root: Path) -> str | None:
    """HEAD commit of the repo (provenance nicety; ``None`` when the
    command is unavailable — never a gate)."""

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


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


def rl_split(budget: int, steps: int = RL_STEPS) -> tuple[int, int]:
    """(steps, batch_size) that multiply to ``budget`` for the RL
    searcher: the fixed split keeps the same nominal budget for every
    searcher.  The bench default stays the P4-era ``RL_STEPS = 4``; the
    P10 campaign passes ``P10_RL_STEPS = 8`` (the ADMISSION_STEPS
    precedent, contract §4.1)."""

    budget = int(budget)
    steps = int(steps)
    if steps <= 0:
        raise ValueError(f"rl steps must be positive, got {steps}")
    if budget < MIN_BUDGET or budget % steps != 0:
        raise ValueError(
            f"budget must be >= {MIN_BUDGET} and divisible by {steps}, "
            f"got {budget}"
        )
    return steps, budget // steps


def _formula_len_histogram(search_result, vocab) -> dict[str, int]:
    """Length histogram of the scored proposals' distinct canonical
    token forms (content + EOS, PADs stripped — ``ir.canonical_tokens``,
    the AST dedup key of the search pipeline).  Scores may repeat
    semantic duplicates (RL batches score every proposal), so canonicals
    are deduplicated here.  The billing ledger may dedupe further still
    (numerically equivalent formulas share a calibration fingerprint
    without sharing an AST), so this total upper-bounds the billed
    unique-semantic-evaluation count.  The shared EOS-inclusive cap of
    contract §4.3 makes ``max(keys) <= max_formula_len`` a directly
    assertable invariant for every backend."""

    counter: collections.Counter[int] = collections.Counter()
    seen: set[tuple[int, ...]] = set()
    for score in search_result.scores:
        tokens = getattr(score, "tokens", None)
        if not tokens:
            continue
        canonical = canonical_tokens(tokens, vocab)
        if not canonical:
            continue
        key = tuple(int(token) for token in canonical)
        if key in seen:
            continue
        seen.add(key)
        counter[len(key)] += 1
    return {str(length): counter[length] for length in sorted(counter)}


def tier_a_feature_ids(vocab) -> list[int]:
    """Samplable tier-A feature token ids (p14 §5.4 promotion track).

    The intersection of the single tier authority
    (``ashare_model.data_tier.feature_tier``) with the samplable
    vocabulary — deprecated features are never legal to sample (grammar
    v4+ rule), so they are excluded here.  No second tier mapping exists.
    """

    return [
        vocab.feature_offset + index
        for index, name in enumerate(vocab.feature_names)
        if name not in vocab.deprecated_names and feature_tier(name).value == "A"
    ]


def _validate_promotion_tier_purity(search_result, vocab) -> None:
    """p14 §5.4.4 (fail-closed): every billed candidate of a
    ``promotion_tier_a`` row must trace to data tier A.  A violation fails
    the row and the campaign — it is a search-space leak bug, never a
    research result."""

    for score in getattr(search_result, "scores", ()) or ():
        tokens = getattr(score, "tokens", None)
        if not tokens:
            raise ValueError(
                "promotion_tier_a purity violation: a billed candidate "
                "carries no token list"
            )
        report = formula_data_tier_report(tokens=tokens)
        if report is None or report["max_tier"] != "A":
            raise ValueError(
                "promotion_tier_a purity violation: candidate "
                f"{getattr(score, 'formula_text', None)!r} traces to "
                f"{report['max_tier'] if report else 'unknown'} tier "
                f"(per-feature {report['per_feature'] if report else 'n/a'})"
            )


def _run_row(
    data_config,
    model_config,
    backtest_config,
    reward_config,
    loader: AshareDataLoader,
    *,
    searcher: str,
    seed: int,
    budget: int,
    steps: int,
    batch: int,
    train_end_date: str | None,
    window_cap: tuple[int, int],
    vm_device,
    track: str = "research",
    feature_ids: list[int] | None = None,
) -> dict:
    """Run one (searcher, seed, track) row with a fresh trainer (fresh
    semantic-cache ledger) and return the recorded row dict.  A crash is
    a recorded measurement (``completed`` False), never a raise.  A
    ``promotion_tier_a`` row additionally fails closed when any billed
    candidate traces beyond tier A (p14 §5.4.4)."""

    trainer = AshareTrainer(
        data_config,
        model_config,
        backtest_config,
        loader=loader,
        reward_config=reward_config,
        feature_ids=feature_ids,
    )

    def run():
        return trainer.search(
            searcher=searcher,
            steps=steps,
            batch_size=batch,
            seed=seed,
            save_artifacts=False,
            train_end_date=train_end_date,
            window_cap=window_cap,
            device=str(vm_device),
            # The benchmark's RL row is the explicitly named random-init
            # arm; imitation is compared separately by P4 admission.
            rl_initialization="random" if searcher == "rl" else None,
        )

    logger.info(
        "benchmark searcher={} seed={} budget={} steps={} batch={} "
        "window_cap={}",
        searcher,
        seed,
        budget,
        steps,
        batch,
        window_cap,
    )
    error: str | None = None
    wall = 0.0
    peak = None
    started = time.perf_counter()
    search_result = None
    try:
        search_result, peak = measure_peak_rss(run)
        if track == "promotion_tier_a":
            _validate_promotion_tier_purity(search_result, trainer.vocab)
    except Exception as exc:  # a failed row is a measurement too
        error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "searcher {} seed {} track {} failed: {}", searcher, seed, track, error
        )
    wall = time.perf_counter() - started
    cache = getattr(trainer, "semantic_cache", None)
    evals = (
        int(search_result.consumed_budget)
        if search_result is not None
        else int(cache.budget_used) if cache is not None else 0
    )
    best = getattr(trainer, "best_val_reward", None)
    if best is None:
        best = getattr(trainer, "best_reward", None)
    completed = error is None
    row = {
        "searcher": searcher,
        "seed": int(seed),
        "track": track,
        "budget": budget,
        "requested_budget": budget,
        "consumed_budget": evals,
        "steps": steps,
        "batch_size": batch,
        "unique_semantic_evals": evals,
        "length_prior_profile": P14_LENGTH_PRIOR_PROFILE,
        "tier_restriction": (
            {
                "tier": "A",
                "feature_count": len(tier_a_feature_ids(trainer.vocab)),
                "data_tier_version": DATA_TIER_VERSION,
            }
            if track == "promotion_tier_a"
            else None
        ),
        "wall_seconds": wall,
        "wall_per_1000_evals": (
            wall / evals * 1000.0 if completed and evals > 0 else None
        ),
        "peak_rss_mb": peak,
        "completed": completed,
        "selected_val_reward": (
            float(best) if best is not None and math.isfinite(best) else None
        ),
        "error": error,
        "termination_reason": (
            search_result.termination_reason
            if search_result is not None
            else "backend_error"
        ),
        "stagnation_reason": (
            search_result.stagnation_reason
            if search_result is not None
            else None
        ),
        # P10 §4.4 keeps the AGENTS §7 accounting fields visible at the
        # top level of every row (they also live in ``search_result``).
        "proposal_count": (
            int(search_result.proposal_count)
            if search_result is not None
            else None
        ),
        "invalid_proposals": (
            int(search_result.invalid_proposals)
            if search_result is not None
            else None
        ),
        "semantic_duplicates": (
            int(search_result.semantic_duplicates)
            if search_result is not None
            else None
        ),
        "best_so_far": (
            [[x, reward] for x, reward in search_result.best_so_far]
            if search_result is not None
            else []
        ),
        "formula_len_histogram": (
            _formula_len_histogram(search_result, getattr(trainer, "vocab", None))
            if search_result is not None
            else {}
        ),
        "search_result": (
            search_result.to_dict() if search_result is not None else None
        ),
        # Row-carried identity fields: on the campaign payload every row
        # is self-contained, so a drift between rows is directly visible.
        "dataset_id": loader.dataset_id,
        "window_cap": [int(window_cap[0]), int(window_cap[1])],
        "train_end_date": train_end_date,
        "device": str(vm_device),
        "max_formula_len": int(model_config.max_formula_len),
    }
    logger.info(
        "benchmark searcher={} seed={} evals={} wall={:.1f}s "
        "per_1000={} peak_rss_mb={} completed={}",
        searcher,
        seed,
        evals,
        wall,
        row["wall_per_1000_evals"],
        peak,
        error is None,
    )
    return row


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
        steps, batch = (rl_steps, rl_batch) if searcher == "rl" else (budget, 1)
        rows[searcher] = _run_row(
            data_config,
            model_config,
            backtest_config,
            reward_config,
            loader,
            searcher=searcher,
            seed=seed,
            budget=budget,
            steps=steps,
            batch=batch,
            train_end_date=train_end_date,
            window_cap=window_cap,
            vm_device=vm_device,
            track="research",  # single-run mode searches the full vocabulary
            feature_ids=None,
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


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Durably persist the campaign payload (tmp file + atomic replace)
    so a killed process never leaves a torn artifact."""

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)


def _campaign_identity(
    *,
    budget: int,
    seeds: tuple[int, ...],
    searchers: tuple[str, ...],
    tracks: tuple[str, ...],
    research_budget: int,
    promotion_budget: int,
    rl_steps: int,
    window_cap: tuple[int, int],
    train_end_date: str | None,
    fold_index: int,
    dataset_id: str,
    max_formula_len: int,
) -> dict:
    """The semantic identity a resume must match exactly (contract §6.4)."""

    return {
        "budget": int(budget),
        "seeds": [int(seed) for seed in seeds],
        "searchers": list(searchers),
        "tracks": list(tracks),
        "research_budget": int(research_budget),
        "promotion_budget": int(promotion_budget),
        "length_prior_profile": P14_LENGTH_PRIOR_PROFILE,
        "rl_steps": int(rl_steps),
        "window_cap": [int(window_cap[0]), int(window_cap[1])],
        "train_end_date": train_end_date,
        "fold_index": int(fold_index),
        "dataset_id": dataset_id,
        "max_formula_len": int(max_formula_len),
    }


def benchmark_campaign(
    data_config,
    model_config,
    backtest_config,
    reward_config,
    loader: AshareDataLoader,
    *,
    seeds,
    budget: int,
    train_end_date: str | None,
    window_cap: tuple[int, int],
    run_dir,
    research_budget: int | None = None,
    promotion_budget: int | None = None,
    fold_index: int = P10_COMPARE_FOLD,
    rl_steps: int = P10_RL_STEPS,
    device: str | None = None,
    wall_cap_s: float | None = P10_STAGE_A_WALL_CAP_S,
    calibration_s_per_eval: dict[str, float] | None = None,
    config_hash: str | None = None,
    searchers: tuple[str, ...] = P10_ROW_ORDER,
) -> dict:
    """P14 matched campaign (p14 §5.4; circuit breakers per p10 §4/§6/§7):
    ``seeds × tracks × searchers`` rows — a ``research`` track over the
    full samplable vocabulary and a ``promotion_tier_a`` track over the
    tier-A samplable vocabulary — on the identical window/evaluator with

    * per-track budgets (research 1200 / promotion 800 by default; the
      split must sum to ``budget`` — p14 §7.3);
    * a fresh trainer (fresh semantic-cache ledger) per row;
    * one append-only :class:`ExperimentLedger` trial per row attempt —
      failures stay in the chain, retries open new trials;
    * atomic per-row persistence of ``<run_dir>/campaign.json`` and
      resume that skips only rows whose latest attempt completed;
    * fail-closed circuit breakers: between-row wall cap, post-seed-block
      calibration-deviation stop, identity-drift refusal on resume, and
      the promotion-track tier-purity validation (p14 §5.4.4).

    The returned payload is the campaign artifact (SEARCHER_BENCH_VERSION
    4).  ``campaign_status == "completed"`` requires every planned row to
    have completed; anything else must not be read as a matched
    comparison.  Matched comparisons are valid within a track only.
    """

    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be non-empty and unique, got {seeds!r}")
    unknown = [s for s in searchers if s not in SEARCHERS]
    if unknown:
        raise ValueError(f"unknown searchers: {unknown} (choose from {SEARCHERS})")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be positive")
    research_budget = (
        P14_RESEARCH_BUDGET if research_budget is None else int(research_budget)
    )
    promotion_budget = (
        P14_PROMOTION_BUDGET if promotion_budget is None else int(promotion_budget)
    )
    if (
        research_budget <= 0
        or promotion_budget <= 0
        or research_budget + promotion_budget != budget
    ):
        raise ValueError(
            f"campaign budget split mismatch: research_budget "
            f"({research_budget}) + promotion_budget ({promotion_budget}) "
            f"must both be positive and sum to budget ({budget}) (p14 §7.3)"
        )
    if budget < MIN_BUDGET:
        raise ValueError(f"campaign budget must be >= {MIN_BUDGET}, got {budget}")
    rates = dict(calibration_s_per_eval or P10_CALIBRATION_S_PER_EVAL)
    missing = [s for s in searchers if s not in rates]
    if missing:
        raise ValueError(f"no calibration rate for searchers: {missing}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    campaign_path = run_dir / "campaign.json"
    ledger_path = run_dir / "ledger.jsonl"
    vm_device = resolve_device(device)
    dataset_id = loader.dataset_id

    def row_budget(track: str) -> int:
        return research_budget if track == "research" else promotion_budget

    identity = _campaign_identity(
        budget=budget,
        seeds=seeds,
        searchers=searchers,
        tracks=P14_TRACKS,
        research_budget=research_budget,
        promotion_budget=promotion_budget,
        rl_steps=rl_steps,
        window_cap=window_cap,
        train_end_date=train_end_date,
        fold_index=fold_index,
        dataset_id=dataset_id,
        max_formula_len=int(model_config.max_formula_len),
    )

    plan: list[tuple[int, str, str]] = [
        (seed, track, searcher)
        for seed in seeds
        for track in P14_TRACKS
        for searcher in searchers
    ]
    if campaign_path.exists():
        with campaign_path.open(encoding="utf-8") as handle:
            recorded = json.load(handle)
        recorded_identity = recorded.get("identity")
        if recorded_identity != identity:
            drift = {
                key: (
                    (recorded_identity or {}).get(key),
                    (identity or {}).get(key),
                )
                for key in sorted(set(recorded_identity or {}) | set(identity))
                if (recorded_identity or {}).get(key)
                != (identity or {}).get(key)
            }
            raise RuntimeError(
                "campaign identity drift on resume; refusing to run "
                f"(fail-closed, contract §6.4): {drift}"
            )
        run_id = str(recorded["run_id"])
        payload = recorded
        latest: dict[tuple[int, str, str], dict] = {
            (int(row["seed"]), str(row["track"]), str(row["searcher"])): row
            for row in recorded.get("rows", [])
        }
        logger.info(
            "campaign resume run_id={} rows-done={} of {}",
            run_id,
            sum(1 for row in latest.values() if row.get("completed") is True),
            len(plan),
        )
    elif ledger_path.exists():
        raise RuntimeError(
            f"ledger {ledger_path} exists without campaign.json; cannot "
            "verify campaign identity (fail-closed, contract §6.4)"
        )
    else:
        run_id = new_run_id()
        payload = {
            "version": SEARCHER_BENCH_VERSION,
            "run_id": run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "campaign_status": "running",
            "identity": identity,
            "campaign": {
                **identity,
                "device": str(vm_device),
                "semantic_cache_version": SEMANTIC_CACHE_VERSION,
                "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
                "config_hash": config_hash,
                "wall_cap_s": wall_cap_s,
                "calibration_s_per_eval": dict(rates),
                "deviation_stop_ratio": P10_DEVIATION_STOP_RATIO,
                "row_order": [
                    f"{seed}:{track}:{searcher}"
                    for seed, track, searcher in plan
                ],
                "tier_restriction_provenance": {
                    "track": "promotion_tier_a",
                    "tier": "A",
                    "authority": "ashare_model.data_tier.feature_tier",
                    "data_tier_version": DATA_TIER_VERSION,
                    "feature_count": len(tier_a_feature_ids(FORMULA_VOCAB)),
                },
                "rl_initialization": "random",
            },
            "rows": [],
            "not_run": [],
        }
        latest = {}

    ledger = ExperimentLedger(ledger_path, run_id=run_id)
    stopped_reason: str | None = None
    started_at = time.perf_counter()

    def persist() -> None:
        payload["rows"] = [latest[key] for key in plan if key in latest]
        _atomic_write_json(campaign_path, payload)

    def block_check(block_seed: int) -> bool:
        """True when the finished seed block's mean wall exceeds the
        calibrated projection by the pre-registered ratio."""

        block_rows = [
            row
            for (row_seed, _, _), row in latest.items()
            if row_seed == block_seed
        ]
        if not block_rows:
            return False
        block_mean = sum(row["wall_seconds"] for row in block_rows) / len(
            block_rows
        )
        block_planned = [
            row_budget(track) * float(rates[s])
            for s_seed, track, s in plan
            if s_seed == block_seed
        ]
        projected_mean = sum(block_planned) / len(block_planned)
        if block_mean > P10_DEVIATION_STOP_RATIO * projected_mean:
            logger.error(
                "campaign calibration deviation after seed {}: mean wall "
                "{:.1f}s vs projected {:.1f}s (ratio limit {:.1f})",
                block_seed,
                block_mean,
                projected_mean,
                P10_DEVIATION_STOP_RATIO,
            )
            return True
        return False

    for index, (seed, track, searcher) in enumerate(plan):
        done = latest.get((seed, track, searcher))
        if done is not None and done.get("completed") is True:
            continue  # resume: this row already has a terminal success

        if (
            wall_cap_s is not None
            and time.perf_counter() - started_at
            + row_budget(track) * float(rates[searcher])
            > float(wall_cap_s)
        ):
            stopped_reason = "stopped_wall_cap"
            logger.error(
                "campaign wall cap hit before row {}:{}:{} (elapsed={:.1f}s "
                "projected={:.1f}s cap={:.1f}s)",
                seed,
                track,
                searcher,
                time.perf_counter() - started_at,
                row_budget(track) * float(rates[searcher]),
                wall_cap_s,
            )
            break

        if index > 0 and plan[index - 1][0] != seed:
            # Contract §4.5: after finishing a full seed block, stop when
            # the block's mean wall exceeds the calibrated projection by
            # the pre-registered ratio (fail-closed, never run silently).
            if block_check(plan[index - 1][0]):
                stopped_reason = "stopped_calibration_deviation"
                break

        if searcher == "rl" and row_budget(track) % int(rl_steps) != 0:
            raise ValueError(
                f"rl row budget {row_budget(track)} for track {track!r} is "
                f"not divisible by rl_steps={rl_steps}"
            )
        steps, batch = (
            (rl_steps, row_budget(track) // int(rl_steps))
            if searcher == "rl"
            else (row_budget(track), 1)
        )
        track_feature_ids = (
            tier_a_feature_ids(FORMULA_VOCAB)
            if track == "promotion_tier_a"
            else None
        )
        trial_id = ledger.record_trial(
            algorithm=f"searcher:{searcher}",
            candidate=f"{searcher}:{seed}:{track}",
            seed=seed,
            fold_train_end=train_end_date,
            payload={
                "row_id": f"{seed}:{track}:{searcher}",
                "track": track,
                "requested_budget": row_budget(track),
                "steps": steps,
                "batch_size": batch,
            },
        )
        try:
            row = _run_row(
                data_config,
                model_config,
                backtest_config,
                reward_config,
                loader,
                searcher=searcher,
                seed=seed,
                budget=row_budget(track),
                steps=steps,
                batch=batch,
                train_end_date=train_end_date,
                window_cap=window_cap,
                vm_device=vm_device,
                track=track,
                feature_ids=track_feature_ids,
            )
        except Exception as exc:
            # _run_row records crashes as rows; reaching this handler
            # means the recording itself failed — fail the trial and stop
            # the campaign (fail-closed, no silent continuation).
            ledger.fail_trial(trial_id, f"{type(exc).__name__}: {exc}")
            raise
        if row["dataset_id"] != dataset_id:
            ledger.fail_trial(trial_id, "dataset_id drifted mid-campaign")
            raise RuntimeError(
                "dataset_id drifted mid-campaign "
                f"({dataset_id} -> {row['dataset_id']}); fail-closed "
                "(contract §6.4)"
            )
        latest[(seed, track, searcher)] = row
        if row["completed"]:
            ledger.complete_trial(
                trial_id,
                metrics={
                    "consumed_budget": row["consumed_budget"],
                    "wall_seconds": row["wall_seconds"],
                    "termination_reason": row["termination_reason"],
                    "stagnation_reason": row["stagnation_reason"],
                },
            )
        else:
            ledger.fail_trial(trial_id, row["error"] or "row failed")
        persist()

    if stopped_reason is None and plan:
        # The final seed block must clear the deviation guard too.
        if block_check(plan[-1][0]):
            stopped_reason = "stopped_calibration_deviation"

    if stopped_reason is not None:
        payload["campaign_status"] = stopped_reason
        payload["not_run"] = [
            {
                "seed": seed,
                "track": track,
                "searcher": searcher,
                "status": "not_run",
                "reason": stopped_reason,
            }
            for seed, track, searcher in plan
            if latest.get((seed, track, searcher), {}).get("completed")
            is not True
        ]
    else:
        payload["not_run"] = []
        payload["campaign_status"] = (
            "completed"
            if all(latest[key].get("completed") is True for key in plan)
            else "failed"
        )
    payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["wall_seconds_total"] = time.perf_counter() - started_at
    persist()
    logger.success(
        "campaign {} status={} rows={} not_run={}",
        run_id,
        payload["campaign_status"],
        len(payload["rows"]),
        len(payload["not_run"]),
    )
    return payload


def _parse_window_cap(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"window-cap must be '<stocks>x<dates>', got {text!r}")
    stocks, dates = int(parts[0]), int(parts[1])
    if stocks <= 0 or dates <= 0:
        raise ValueError("window-cap must be positive (stocks, dates)")
    return stocks, dates


def _config_hash(config_arg: str | None, root: Path) -> str | None:
    """SHA-256 over the effective config inputs (YAML file bytes plus the
    runtime-overrides file when present).  ``config_arg=None`` resolves
    to :func:`ashare_data.config.load_config`'s default path.  ``None``
    is returned only when the config file cannot be read — the campaign
    still runs, provenance records the gap."""

    from ashare_data.config import RUNTIME_OVERRIDES_FILENAME

    try:
        path = Path(config_arg) if config_arg is not None else None
        if path is None:
            path = root / "config" / "ashare_config.yaml"
        elif not path.is_absolute():
            path = root / path
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        overrides = path.parent / RUNTIME_OVERRIDES_FILENAME
        if overrides.exists():
            digest.update(b"\x00--runtime-overrides--\x00")
            digest.update(overrides.read_bytes())
    except OSError:
        return None
    return digest.hexdigest()


def main(argv=None) -> int:
    setup_run_logging(run_name="searcher_bench")
    parser = argparse.ArgumentParser(
        description="Searcher cost benchmark: time & peak RSS per 1000 "
        "unique semantic evaluations (P1-04) and the small-budget smoke "
        "test (P1-05); P10 campaign mode with --run-dir"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="data/searcher_bench.json")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="requested unique semantic evaluations per row "
        "(default: 1000 legacy bench / P10_COMPARE_BUDGET campaign)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="enable P10 campaign mode: seeds x searchers rows with "
        "ledger, resume and circuit breakers under this directory",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="campaign mode: comma-separated paired seeds "
        "(default: P10_COMPARE_SEEDS)",
    )
    parser.add_argument(
        "--rl-steps",
        type=int,
        default=None,
        help="campaign mode: RL optimizer steps (default: P10_RL_STEPS=8)",
    )
    parser.add_argument(
        "--wall-cap-hours",
        type=float,
        default=None,
        help="campaign mode: hard Stage-A wall cap in hours "
        "(default: P10_STAGE_A_WALL_CAP_S)",
    )
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

        if args.run_dir is not None:
            seeds = (
                tuple(int(s) for s in args.seeds.split(",") if s.strip())
                if args.seeds
                else P10_COMPARE_SEEDS
            )
            payload = benchmark_campaign(
                data_config,
                model_config,
                backtest_config,
                reward_config,
                loader,
                seeds=seeds,
                budget=args.budget or P10_COMPARE_BUDGET,
                train_end_date=train_end_date,
                window_cap=window_cap,
                fold_index=fold_index,
                rl_steps=args.rl_steps or P10_RL_STEPS,
                device=None,
                wall_cap_s=(
                    args.wall_cap_hours * 3600.0
                    if args.wall_cap_hours is not None
                    else P10_STAGE_A_WALL_CAP_S
                ),
                run_dir=root / args.run_dir,
                config_hash=_config_hash(args.config, root),
                searchers=searchers,
            )
            print(
                f"campaign run_id={payload['run_id']} "
                f"status={payload['campaign_status']} "
                f"rows={len(payload['rows'])} "
                f"not_run={len(payload['not_run'])}"
            )
            print(
                f"{'seed':>6} {'searcher':<8} {'evals':>6} {'req':>6} "
                f"{'wall_s':>9} {'per_1000':>9} {'rss_mb':>9} {'done':>6}"
            )
            for row in payload["rows"]:
                per_1000 = (
                    f"{row['wall_per_1000_evals']:.1f}"
                    if row["wall_per_1000_evals"] is not None
                    else "-"
                )
                rss = f"{row['peak_rss_mb']:.0f}" if row["peak_rss_mb"] else "-"
                print(
                    f"{row['seed']:>6} {row['searcher']:<8} "
                    f"{row['unique_semantic_evals']:>6} "
                    f"{row['requested_budget']:>6} "
                    f"{row['wall_seconds']:>9.1f} {per_1000:>9} {rss:>9} "
                    f"{str(row['completed']):>6}"
                )
            for row in payload["not_run"]:
                print(
                    f"{row['seed']:>6} {row['searcher']:<8} "
                    f"{'-':>6} {'-':>6} {'-':>9} {'-':>9} {'-':>9} "
                    f"not_run({row['reason']})"
                )
            return 0

        payload = benchmark_searchers(
            data_config,
            model_config,
            backtest_config,
            reward_config,
            loader,
            searchers=searchers,
            budget=args.budget if args.budget is not None else 1000,
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
