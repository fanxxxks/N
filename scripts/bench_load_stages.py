"""B04 (p33 I4): loader stage benchmark — cold / hot / invalidate protocol.

Engineering measurement only (AGENTS §4.1): stage wall-times, peak RSS and
the bitwise cold/hot equivalence of the loader artifacts.  It supports
boundary/performance conclusions only — never a research claim.

Protocol (docs/p33_b04_load_cache_performance_draft.md §3.4):

* synthetic fixture (default): deterministic dual-database store built with
  the shared test fixture helpers (single implementation, tests/conftest);
  phases: cold (miss) -> hot (hit) -> invalidate (one more ingest row +
  republication; must be a miss) -> hot2;
* ``--real``: the configured real databases through the real config
  resolution, a FIXED window (sorted universe head x last open sessions at
  or before the data cutoff) — read-only loads plus the local factor-cache
  artifact the loader itself writes; cold/hot tensors must be bitwise
  equal or the benchmark exits non-zero.

Stage attribution: the benchmark wraps the loader's stage functions in its
own process (never in production code).  ``torch.tensor`` calls are
attributed to the ``copy`` stage by wrapping the torch module attribute
for the duration of the run (script-local; restored in ``finally``).
Unattributed residual is reported as ``other`` (target_ret assembly,
encode, numpy glue).

Usage:
  python scripts/bench_load_stages.py [--json PATH] [--codes 60] [--sessions 240]
  python scripts/bench_load_stages.py --real [--real-config configs/r1_medium_a.yaml]
      [--max-codes 300] [--last-sessions 260] [--json PATH]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil
import torch

from ashare_data.io_utils import atomic_write_json
from ashare_model import data_loader as data_loader_module

STAGES = (
    "confirm",
    "identity",
    "read",
    "normalize_scale",
    "universe",
    "tensor_maps",
    "freshness",
    "pit",
    "capital",
    "industry",
    "factor",
    "copy",
)


class StageClock:
    """Accumulates per-stage wall seconds via wrapped callables.

    Wrap ONCE (the outer ``with``) and :meth:`reset` before each phase —
    the wrappers themselves must never stack.
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = {name: 0.0 for name in STAGES}
        self._wrapped: list = []
        self._torch_real = None
        self._installed = False

    def reset(self) -> None:
        for name in STAGES:
            self.totals[name] = 0.0

    def wrap_module_fn(self, module, name: str, stage: str) -> None:
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return real(*args, **kwargs)
            finally:
                self.totals[stage] += time.perf_counter() - start

        setattr(module, name, wrapper)
        self._wrapped.append((module, name, real))

    def wrap_method(self, cls, name: str, stage: str) -> None:
        real = getattr(cls, name)

        def wrapper(self_loader, *args, **kwargs):
            start = time.perf_counter()
            try:
                return real(self_loader, *args, **kwargs)
            finally:
                self.totals[stage] += time.perf_counter() - start

        setattr(cls, name, wrapper)
        self._wrapped.append((cls, name, real))

    def wrap_torch_tensor(self) -> None:
        real = torch.tensor

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return real(*args, **kwargs)
            finally:
                self.totals["copy"] += time.perf_counter() - start

        torch.tensor = wrapper
        self._torch_real = real

    def __enter__(self) -> "StageClock":
        if self._installed:
            return self
        try:
            # Class methods (the loader's internal stages).
            self.wrap_method(
                data_loader_module.AshareDataLoader,
                "confirm_input_snapshot",
                "confirm",
            )
            self.wrap_method(
                data_loader_module.AshareDataLoader, "_load_v2_bars", "read"
            )
            # Module-level stage functions as the loader sees them.
            self.wrap_module_fn(
                data_loader_module,
                "loader_factor_cache_identity_with_reason",
                "identity",
            )
            self.wrap_module_fn(
                data_loader_module, "adjusted_price_scale", "normalize_scale"
            )
            self.wrap_module_fn(
                data_loader_module, "normalize_daily_bars", "normalize_scale"
            )
            self.wrap_module_fn(
                data_loader_module, "apply_adjusted_scale", "normalize_scale"
            )
            self.wrap_module_fn(
                data_loader_module, "build_universe_mask", "universe"
            )
            self.wrap_module_fn(data_loader_module, "pivot_wide", "tensor_maps")
            self.wrap_module_fn(
                data_loader_module, "industry_index_freshness", "freshness"
            )
            self.wrap_module_fn(
                data_loader_module, "build_pit_frames", "pit"
            )
            self.wrap_module_fn(
                data_loader_module, "build_capital_frames", "capital"
            )
            self.wrap_module_fn(
                data_loader_module, "build_industry_member_frame", "industry"
            )
            self.wrap_module_fn(
                data_loader_module, "compute_factor_tensor", "factor"
            )
            self.wrap_torch_tensor()
        except BaseException:
            # Dual-review F3: a partial install must never leak wrappers —
            # unwind everything applied so far before propagating.
            self.__exit__(None, None, None)
            raise
        self._installed = True
        return self

    def __exit__(self, *exc) -> None:
        for owner, name, real in reversed(self._wrapped):
            setattr(owner, name, real)
        if self._torch_real is not None:
            torch.tensor = self._torch_real


class RssPeak:
    """Samples this process's RSS in a daemon thread; peak in MB."""

    def __init__(self, interval: float = 0.05) -> None:
        self._interval = interval
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            self._peak = max(self._peak, process.memory_info().rss)
            self._stop.wait(self._interval)

    def __enter__(self) -> "RssPeak":
        self._peak = psutil.Process().memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._peak = max(self._peak, psutil.Process().memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024.0 * 1024.0)


def _snapshot(loader) -> dict[str, np.ndarray]:
    payload = {
        "factor_tensor": loader.factor_tensor.numpy().copy(),
        "target_ret": loader.target_ret.numpy().copy(),
        "industry_codes": loader.industry_codes.numpy().copy(),
    }
    for key, value in loader.raw_data_cache.items():
        payload[f"raw:{key}"] = value.numpy().copy()
    for key, value in loader.adjusted_data_cache.items():
        payload[f"adjusted:{key}"] = value.numpy().copy()
    return payload


def _equivalent(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    if set(left) != set(right):
        return False
    return all(
        np.array_equal(left[key], right[key], equal_nan=True) for key in left
    )


def _run_phase(
    label: str,
    loader_factory,
    clock: StageClock,
    load_kwargs: dict | None = None,
) -> dict:
    clock.reset()
    with RssPeak() as rss:
        start = time.perf_counter()
        loader = loader_factory().load_data(**(load_kwargs or {}))
        wall = time.perf_counter() - start
    stages = dict(clock.totals)
    attributed = sum(stages.values())
    return {
        "phase": label,
        "wall_s": round(wall, 3),
        "peak_rss_mb": round(rss.peak_mb, 1),
        "factor_cache_outcome": loader.factor_cache_outcome,
        "factor_cache_reason": loader.factor_cache_reason,
        "stages_s": {key: round(value, 3) for key, value in stages.items()},
        "other_s": round(max(wall - attributed, 0.0), 3),
        "snapshot": _snapshot(loader),
    }


def _finalize(result: dict) -> dict:
    result.pop("snapshot", None)
    return result


def _synthetic_codes(n: int) -> list[str]:
    half = n // 2
    return [f"6{i:05d}.SH" for i in range(half)] + [
        f"0{i + 1:05d}.SZ" for i in range(n - half)
    ]


def build_synthetic_fixture(data_dir: Path, n_codes: int, n_sessions: int):
    from ashare_data.config import DataConfig
    from ashare_data.db import AshareDB
    from tests.conftest import (
        make_bars,
        populate_v2_industry_tables,
        populate_v2_margin_table,
        populate_v2_price_tables,
        populate_v2_statement_tables,
        publish_manifest,
    )
    from ashare_data.config import resolve_v2_duckdb_path

    cfg = DataConfig(
        data_dir=data_dir,
        duckdb_path=data_dir / "ashare.duckdb",
        parquet_dir=data_dir / "parquet",
        start_date="2024-01-01",
        end_date="2026-12-31",
        index_codes=["000300.SH"],
        index_names=["沪深300"],
        min_listed_sessions=1,
    )
    codes = _synthetic_codes(n_codes)
    dates, _, bars = make_bars(n_dates=n_sessions, ts_codes=codes)
    # Primary side through the SAME public db API tests/conftest._populate_db
    # uses — that helper's constituent in_date f-string only admits nine
    # codes, so the benchmark assembles the identical rows itself (the v2
    # side and the publication reuse the shared helpers unchanged).
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.parquet_dir.mkdir(parents=True, exist_ok=True)
    with AshareDB(cfg.duckdb_path) as db:
        db.create_schema(cfg)
        db.upsert_daily(bars.to_dict("records"), cfg)
        db.upsert_stocks(
            [
                {
                    "ts_code": code,
                    "name": code,
                    "industry": None,
                    "list_date": "20200101",
                    "is_st": False,
                }
                for code in codes
            ],
            cfg,
        )
        db.upsert_calendar(
            [{"trade_date": date, "is_open": True} for date in dates], cfg
        )
        db.upsert_constituents(
            [
                {
                    "index_code": "000300.SH",
                    "ts_code": code,
                    # Staggered PIT in-dates (the production universe
                    # contract rejects one uniform full-history interval
                    # as a stretched current snapshot).
                    "in_date": f"2020{position % 12 + 1:02d}01",
                    "out_date": "99991231",
                }
                for position, code in enumerate(codes)
            ],
            cfg,
        )
    v2_path = resolve_v2_duckdb_path(cfg)
    populate_v2_price_tables(v2_path, dates, codes, bars)
    # Ten Shenwan L1 industries so the industry stage and INDUSTRY_MOMENTUM
    # do real work; the canonical v2 form carries the .SI suffix.
    industries = [f"{801000 + i}.SI" for i in range(10)]
    populate_v2_industry_tables(
        v2_path,
        membership_rows=[
            {
                "ts_code": code,
                "index_code": industries[position % 10],
                "level": 1,
                "in_date": "20200101",
                "out_date": "99991231",
                "is_new": "N",
                "industry_name": f"bench_{position % 10}",
                "available_date": "20200102",
            }
            for position, code in enumerate(codes)
        ],
        taxonomy_rows=[
            {
                "index_code": index_code,
                "industry_name": f"bench_{position}",
                "level": "L1",
                "industry_code": index_code,
                "src": "bench",
            }
            for position, index_code in enumerate(industries)
        ],
    )
    with AshareDB(cfg.duckdb_path) as db:
        db.upsert_sw_index(
            [
                {
                    "index_code": index_code[:-3],
                    "industry_name": f"bench_{position}",
                    "trade_date": session,
                    "close": 50.0 + position + 0.1 * index,
                }
                for position, index_code in enumerate(industries)
                for index, session in enumerate(dates)
            ],
            cfg,
        )
    # Real-as-of financial PIT rows: eight report periods per code, visible
    # from their f_ann_date-driven available_date (p17 §6.2 shapes).
    quarters = [f"{year}{suffix}" for year in (2022, 2023) for suffix in ("0331", "0630", "0930", "1231")]
    income_rows, balance_rows, cashflow_rows = [], [], []
    for position, code in enumerate(codes):
        for index, end in enumerate(quarters):
            available = f"2024{index + 1:02d}15"
            base = 1.0 + 0.01 * position
            common = {
                "ts_code": code, "end_date": end,
                "ann_date": available, "f_ann_date": available,
                "available_date": available,
                "report_type": "1", "comp_type": "1",
                "update_flag": "1", "ingest_id": "bench-ing-1",
            }
            income_rows.append(
                {
                    **common,
                    "n_income_attr_p": 1e7 * base, "revenue": 4e7 * base,
                    "oper_cost": 3e7 * base, "total_profit": 1.2e7 * base,
                }
            )
            balance_rows.append(
                {
                    **common,
                    "total_assets": 5e8 * base, "total_liab": 2.5e8 * base,
                }
            )
            cashflow_rows.append(
                {**common, "n_cashflow_act": 8e6 * base}
            )
    populate_v2_statement_tables(
        v2_path,
        income_rows=income_rows,
        balance_rows=balance_rows,
        cashflow_rows=cashflow_rows,
    )
    margin_rows = [
        {
            "ts_code": code,
            "trade_date": session,
            "available_date": session,
            "rzye": 1e6 + position,
            "rzmre": 1e4 + position,
        }
        for position, code in enumerate(codes)
        for session in dates
    ]
    populate_v2_margin_table(v2_path, margin_rows)
    with AshareDB(v2_path) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS source_ingest_runs (ingest_id VARCHAR)"
        )
        db.execute("DELETE FROM source_ingest_runs")
        db.execute("INSERT INTO source_ingest_runs VALUES ('bench-ing-1')")
    publish_manifest(cfg)

    def invalidate() -> None:
        with AshareDB(v2_path) as db:
            db.execute("INSERT INTO source_ingest_runs VALUES ('bench-ing-2')")
        publish_manifest(cfg)

    return cfg, dates, codes, invalidate


def run_synthetic(args, json_out: dict) -> int:
    from ashare_data.config import ModelConfig
    from ashare_model.data_loader import AshareDataLoader

    data_dir = Path(args.synthetic_dir) if args.synthetic_dir else None
    if data_dir is None:
        import tempfile

        data_dir = Path(tempfile.mkdtemp(prefix="b04_bench_"))
    build_start = time.perf_counter()
    cfg, dates, codes, invalidate = build_synthetic_fixture(
        data_dir, args.codes, args.sessions
    )
    build_s = time.perf_counter() - build_start
    print(
        f"fixture: {len(codes)} codes x {len(dates)} sessions in "
        f"{build_s:.1f}s at {data_dir}"
    )
    del dates, codes

    with StageClock() as clock:
        cold = _run_phase("cold", lambda: AshareDataLoader(cfg, ModelConfig()), clock)
        hot = _run_phase("hot", lambda: AshareDataLoader(cfg, ModelConfig()), clock)
        invalidate()
        invalidated = _run_phase(
            "invalidate", lambda: AshareDataLoader(cfg, ModelConfig()), clock
        )
        hot2 = _run_phase(
            "hot2", lambda: AshareDataLoader(cfg, ModelConfig()), clock
        )

    checks = {
        "cold_is_miss": cold["factor_cache_outcome"] == "miss",
        "hot_is_hit": hot["factor_cache_outcome"] == "hit",
        "hot_tensor_equals_cold": _equivalent(hot["snapshot"], cold["snapshot"]),
        "invalidate_is_miss": invalidated["factor_cache_outcome"] == "miss",
        "hot2_is_hit": hot2["factor_cache_outcome"] == "hit",
        "hot2_tensor_equals_cold": _equivalent(hot2["snapshot"], cold["snapshot"]),
    }
    json_out.update(
        {
            "mode": "synthetic",
            "fixture": {
                "codes": args.codes,
                "sessions": args.sessions,
                "build_s": round(build_s, 1),
                "dir": str(data_dir),
            },
            "phases": [
                _finalize(cold),
                _finalize(hot),
                _finalize(invalidated),
                _finalize(hot2),
            ],
            "checks": checks,
        }
    )
    _report(json_out)
    return 0 if all(checks.values()) else 1


def run_real(args, json_out: dict) -> int:
    import duckdb

    from ashare_data.config import resolve_v2_duckdb_path
    from ashare_model.data_loader import (
        AshareDataLoader,
        build_loader_from_config,
    )

    config_path = args.real_config or "configs/r1_medium_a.yaml"
    probe = AshareDataLoader(_loader_args(config_path))
    universe = sorted(probe.load_universe())
    v2_path = str(resolve_v2_duckdb_path(probe.config))
    with duckdb.connect(v2_path, read_only=True) as con:
        cutoff = str(
            con.execute("SELECT MAX(trade_date) FROM daily_bar_raw").fetchone()[0]
        ).replace("-", "")
    sessions = [s for s in probe._contract().sessions if s <= cutoff]
    window = sessions[-args.last_sessions :]
    codes = universe[: args.max_codes]
    print(
        f"real window: {len(codes)} codes x {len(window)} sessions "
        f"(cutoff {cutoff}, config {config_path})"
    )

    def factory() -> "AshareDataLoader":
        return build_loader_from_config(REPO_ROOT, config_path)

    with StageClock() as clock:
        # The pre-registered fixed window is PASSED to load_data — codes
        # must sit inside the resolved universe contract, dates inside its
        # session axis (validated fail-closed by the loader).
        cold = _run_phase(
            "cold", factory, clock, {"ts_codes": codes, "dates": window}
        )
        hot = _run_phase(
            "hot", factory, clock, {"ts_codes": codes, "dates": window}
        )

    checks = {
        "cold_is_miss": cold["factor_cache_outcome"] == "miss",
        "hot_is_hit": hot["factor_cache_outcome"] == "hit",
        "hot_tensor_equals_cold": _equivalent(hot["snapshot"], cold["snapshot"]),
    }
    json_out.update(
        {
            "mode": "real",
            "config": config_path,
            "window": {
                "codes": len(codes),
                "sessions": len(window),
                "first": window[0] if window else None,
                "last": window[-1] if window else None,
                "data_cutoff": cutoff,
            },
            "phases": [_finalize(cold), _finalize(hot)],
            "checks": checks,
        }
    )
    _report(json_out)
    return 0 if all(checks.values()) else 1


def _loader_args(config_path: str):
    from ashare_data.config import load_config, make_data_config

    raw = load_config(config_path, project_root=REPO_ROOT)
    return make_data_config(raw, REPO_ROOT)


def _report(json_out: dict) -> None:
    phases = json_out["phases"]
    print(f"\nmode={json_out['mode']}  checks={json_out['checks']}")
    header = (
        f"{'phase':<12}{'outcome':<13}{'wall_s':>9}{'RSS_MB':>10}"
        + "".join(f"{name:>12}" for name in STAGES)
        + f"{'other':>10}"
    )
    print(header)
    for phase in phases:
        row = (
            f"{phase['phase']:<12}{phase['factor_cache_outcome']:<13}"
            f"{phase['wall_s']:>9.2f}{phase['peak_rss_mb']:>10.0f}"
            + "".join(f"{phase['stages_s'][name]:>12.2f}" for name in STAGES)
            + f"{phase['other_s']:>10.2f}"
        )
        print(row)
    if len(phases) >= 2:
        print(
            f"hot/cold wall ratio: "
            f"{phases[1]['wall_s'] / phases[0]['wall_s']:.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the raw measurement JSON here")
    parser.add_argument("--codes", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=240)
    parser.add_argument("--synthetic-dir", help="reuse a fixture directory")
    parser.add_argument(
        "--real", action="store_true", help="benchmark the real databases"
    )
    parser.add_argument("--real-config", default="configs/r1_medium_a.yaml")
    parser.add_argument("--max-codes", type=int, default=300)
    parser.add_argument("--last-sessions", type=int, default=260)
    args = parser.parse_args()

    json_out: dict = {}
    code = run_real(args, json_out) if args.real else run_synthetic(args, json_out)
    json_out["meta"] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "stage_attribution": "wrapped loader callables; residual in other_s",
    }
    if args.json:
        atomic_write_json(Path(args.json), json_out)
        print(f"json written: {args.json}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
