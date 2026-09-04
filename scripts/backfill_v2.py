"""D3 backfill orchestrator (p17 §10 eight steps, §11 scope, §13 pacing).

Executes the full Tushare v2 backfill into a NEW ``data/ashare_v2.duckdb``
(never the production database — p17 §1.3/§8) through the tested
primitives only:

* fetch  — :class:`ashare_data.tushare_client.TushareClient` (typed
  fail-closed contracts, 180 req/min sliding-window limiter, 429/timeout
  exponential backoff, ``needs_pagination`` signal);
* store  — :class:`ashare_data.tushare_store.TushareStore` (L0 snapshot
  BEFORE the database transaction, ``source_ingest_runs`` ledger,
  KEYED/APPEND semantics, single-batch atomicity);
* calendar — the PRODUCTION ``trade_calendar`` read-only is the only
  session authority (p17 §6.1: no second calendar);
* scope  — ``persisted_scope_codes`` from the production database
  (p17 §11.1): frames are filtered to the scope before any formal-table
  insert (§15.5); full-market raw frames are preserved in L0 only;
* resume — the ledger is the journal: a planned batch whose canonical
  parameters_json already has a succeeded run is skipped, so re-running
  the orchestrator after a stop is safe and cheap.

Pagination: date-windowed endpoints (forecast/express/repurchase/
stk_holdernumber/disclosure_date) bisect their window on the
``needs_pagination`` signal; per-day / per-stock single-window endpoints
raise the §15.8 stop condition when a single call cannot fit the
interface cap (documented margins in the baseline record).

Stop conditions (p17 §15) propagate as :class:`BackfillStopCondition`
after the failed batch is ledgered; the process exits non-zero.  AkShare
is never consulted (§15.2).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ashare_data.config import load_config  # noqa: E402
from ashare_data.db import AshareDB  # noqa: E402
from ashare_data.manifest import build_v2_dataset_manifest, save_manifest  # noqa: E402
from ashare_data.tushare_client import TushareClient  # noqa: E402
from ashare_data.tushare_store import (  # noqa: E402
    TUSHARE_STORE_SCHEMA_VERSION,
    TushareStore,
)

MARKET_START = "20140101"
FUNDAMENTAL_START = "20121231"


class BackfillStopCondition(RuntimeError):
    """p17 §15.8-family stop: a single call cannot fit the interface cap."""


def canonical_params(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _mid(lo: str, hi: str) -> str:
    """Midpoint of [lo, hi] for date-window bisection (YYYYMMDD strings)."""

    a = datetime.strptime(lo, "%Y%m%d")
    b = datetime.strptime(hi, "%Y%m%d")
    mid = a + (b - a) / 2
    return max(min(mid.strftime("%Y%m%d"), hi), lo)


def add_days(date: str, days: int) -> str:
    return (datetime.strptime(date, "%Y%m%d") + pd.Timedelta(days=days)).strftime("%Y%m%d")


def quarter_ends(start: str, end: str) -> list[str]:
    """Quarter-end dates in [start, end] ascending (YYYYMMDD)."""

    days = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end),
                         freq="QE")
    return [d.strftime("%Y%m%d") for d in days]


def year_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Consecutive year windows covering [start, end]."""

    windows = []
    cursor = start
    while cursor <= end:
        year_end = (pd.Timestamp(cursor) + pd.offsets.YearEnd(0)).strftime("%Y%m%d")
        window_end = min(year_end, end)
        windows.append((cursor, window_end))
        cursor = add_days(window_end, 1)
    return windows


class BackfillRunner:
    """Sequential eight-step executor with ledger-based resume."""

    def __init__(self, args):
        self.args = args
        self.market_start = args.market_start
        self.fundamental_start = args.fundamental_start
        self.db_path = Path(args.db)
        self.raw_root = Path(args.raw_root)
        self.prod_db_path = Path(args.prod_db)
        self.today = datetime.now().strftime("%Y%m%d")
        self.progress_every = args.progress_every

        load_config(args.config, project_root=ROOT)
        self.client = TushareClient(retries=3)
        self.session_resolver = self._make_session_resolver()
        self.code_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.store = TushareStore(
            AshareDB(self.db_path),
            self.raw_root,
            session_resolver=self.session_resolver,
            code_commit=self.code_commit,
            sdk_version=_pkg_version("tushare"),
        )
        self.store.create_schema()
        self.scope_codes = self._load_scope_codes()
        self.resume_keys = self._load_resume_keys()
        self.counts: dict[str, int] = {"batches": 0, "skipped": 0, "rows": 0}
        self.started = time.monotonic()
        self._session_cache: dict[str, str] = {}

    # -- authorities ---------------------------------------------------------

    def _make_session_resolver(self):
        """The production trade_calendar (read-only) is the only session
        authority (p17 §6.1); the resolver returns the first open session
        strictly AFTER the given date."""

        conn = AshareDB(self.prod_db_path, read_only=True)
        sessions = [
            str(r[0]) for r in conn.query(
                "SELECT trade_date FROM trade_calendar WHERE is_open = 1 "
                "ORDER BY trade_date"
            ).itertuples(index=False)
        ]
        conn.close()
        index = {s: i for i, s in enumerate(sessions)}

        def resolve(date: str) -> str:
            if date in self._session_cache:
                return self._session_cache[date]
            for session in sessions:
                if session > date:
                    self._session_cache[date] = session
                    return session
            raise BackfillStopCondition(
                f"no open session after {date} in the production calendar "
                f"(horizon {sessions[-1]})"
            )

        return resolve

    def _load_scope_codes(self) -> frozenset[str]:
        conn = AshareDB(self.prod_db_path, read_only=True)
        try:
            codes = {
                str(r[0]) for r in conn.query(
                    "SELECT DISTINCT ts_code FROM ("
                    " SELECT ts_code FROM stocks"
                    " UNION SELECT ts_code FROM constituents"
                    " UNION SELECT ts_code FROM sw_industry_member"
                    " UNION SELECT ts_code FROM daily_bar)"
                ).itertuples(index=False)
            }
        finally:
            conn.close()
        return frozenset(codes)

    def _load_resume_keys(self) -> set[tuple[str, str]]:
        # Query through the store's own connection: DuckDB refuses a
        # second connection to the same file with a different mode.
        rows = self.store.db.query(
            "SELECT endpoint, parameters_json FROM source_ingest_runs "
            "WHERE status = 'succeeded'"
        )
        return {(str(r[0]), str(r[1])) for r in rows.itertuples(index=False)}

    # -- batch plumbing -------------------------------------------------------

    def _skip_key(self, endpoint: str, params: dict) -> tuple[str, str]:
        return (endpoint, canonical_params(params))

    def _should_skip(self, endpoint: str, params: dict) -> bool:
        return self._skip_key(endpoint, params) in self.resume_keys

    def _run_batch(self, endpoint: str, frame: pd.DataFrame, params: dict,
                   fields: list[str], *, row_limit: int | None,
                   formal: bool, scope_column: str | None = None) -> None:
        key = self._skip_key(endpoint, params)
        if key in self.resume_keys:
            self.counts["skipped"] += 1
            return
        if formal and scope_column and scope_column in frame.columns:
            before = len(frame)
            frame = frame[frame[scope_column].isin(self.scope_codes)]
            dropped = before - len(frame)
            if dropped:
                print(f"  scope-filter {endpoint} {params}: dropped "
                      f"{dropped} out-of-scope rows (§15.5)")
        if formal:
            result = self.store.ingest_batch(
                endpoint, frame, params=params, fields=fields,
                row_limit=row_limit,
            )
        else:
            result = self.store.snapshot_only(
                endpoint, frame, params=params, fields=fields,
                row_limit=row_limit,
            )
        self.resume_keys.add(key)
        self.counts["batches"] += 1
        self.counts["rows"] += int(len(frame))
        if self.counts["batches"] % self.progress_every == 0:
            self._progress(endpoint)

    def _progress(self, endpoint: str) -> None:
        elapsed = time.monotonic() - self.started
        rate = self.counts["batches"] / elapsed * 60 if elapsed else 0.0
        print(f"[progress] batches={self.counts['batches']} "
              f"rows={self.counts['rows']} skipped={self.counts['skipped']} "
              f"rate={rate:.0f}/min elapsed={elapsed/60:.1f}min "
              f"last={endpoint}", flush=True)

    # -- fetch plumbing --------------------------------------------------------

    def _fetch(self, endpoint: str, **params) -> pd.DataFrame:
        if endpoint == "suspend_d":
            # additive source column suspend_timing rides the discovery
            # path (p17 §5.0); the store still fail-closes the DDL columns.
            return self.client.query_raw(endpoint, **params)
        return self.client.query(endpoint, **params).data

    def _fetch_single_guarded(self, endpoint: str, params: dict,
                              frame: pd.DataFrame,
                              row_limit: int | None) -> None:
        """Per-day / per-stock windows cannot be paginated further: a cap
        hit here is the §15.8 stop condition (documented margins in the
        t9 baseline record)."""

        if row_limit is not None and len(frame) >= row_limit:
            raise BackfillStopCondition(
                f"§15.8 {endpoint} hit the interface row cap "
                f"({len(frame)} >= {row_limit}) and the per-day/per-stock "
                f"window cannot be paginated further: params={params}"
            )

    # -- steps -------------------------------------------------------------------

    def step2(self) -> None:
        print("== step 2: trade_cal parity + stock_basic (L0-only)")
        cal = self._fetch("trade_cal", exchange="SSE",
                          start_date="19901219", end_date="20261231")
        self._fetch_single_guarded("trade_cal", {"exchange": "SSE"}, cal, 20000)
        self._run_batch("trade_cal", cal,
                        params={"exchange": "SSE", "start_date": "19901219",
                                "end_date": "20261231"},
                        fields=list(cal.columns), row_limit=20000,
                        formal=False)
        self._record_parity(cal)
        for status in ("L", "D", "P"):
            params = {"list_status": status}
            if self._should_skip("stock_basic", params):
                self.counts["skipped"] += 1
                continue
            frame = self._fetch("stock_basic", **params)
            self._fetch_single_guarded("stock_basic", params, frame, 6000)
            self._run_batch("stock_basic", frame, params=params,
                            fields=list(frame.columns), row_limit=6000,
                            formal=False)

    def _record_parity(self, tushare_cal: pd.DataFrame) -> None:
        conn = AshareDB(self.prod_db_path, read_only=True)
        try:
            prod = {
                str(r[0]) for r in conn.query(
                    "SELECT trade_date FROM trade_calendar WHERE is_open = 1"
                ).itertuples(index=False)
            }
        finally:
            conn.close()
        ts = {
            str(d) for d in tushare_cal.loc[
                tushare_cal["is_open"] == 1, "cal_date"
            ].astype(str)
        }
        overlap_lo, overlap_hi = max(min(ts), min(prod)), min(max(ts), max(prod))
        ts_in_window = {d for d in ts if overlap_lo <= d <= overlap_hi}
        prod_in_window = {d for d in prod if overlap_lo <= d <= overlap_hi}
        only_ts = sorted(ts_in_window - prod_in_window)
        only_prod = sorted(prod_in_window - ts_in_window)
        print(f"[parity] overlap {overlap_lo}..{overlap_hi}: "
              f"tushare={len(ts_in_window)} prod={len(prod_in_window)} "
              f"only_tushare={only_ts[:10]}{'…' if len(only_ts) > 10 else ''} "
              f"only_prod={only_prod[:10]}{'…' if len(only_prod) > 10 else ''} "
              "(recorded; production calendar remains the only authority)")

    def step3(self) -> None:
        print("== step 3: daily / adj_factor / daily_basic / stk_limit / "
              "suspend_d per trade_date (2014-01-01 .. today)")
        for date in self._market_sessions():
            for endpoint, table, limit in (
                ("daily", "daily_bar_raw", 6000),
                ("adj_factor", "daily_adj_factor", 6000),
                ("daily_basic", "daily_basic_pit", 6000),
                ("stk_limit", "daily_price_limit", 7800),
                ("suspend_d", "daily_suspension", 3000),
            ):
                params = {"trade_date": date}
                if self._should_skip(table, params):
                    self.counts["skipped"] += 1
                    continue
                frame = self._fetch(endpoint, **params)
                self._fetch_single_guarded(endpoint, params, frame, limit)
                self._run_batch(table, frame, params=params,
                                fields=list(frame.columns), row_limit=limit,
                                formal=True,
                                scope_column="ts_code")

    def step4(self) -> None:
        print("== step 4: margin_detail per trade_date (T+1 availability "
              "is encoded by the writer per §6.1)")
        for date in self._market_sessions():
            params = {"trade_date": date}
            if self._should_skip("margin_detail_pit", params):
                self.counts["skipped"] += 1
                continue
            frame = self._fetch("margin_detail", **params)
            self._fetch_single_guarded("margin_detail", params, frame, 5000)
            self._run_batch("margin_detail_pit", frame, params=params,
                            fields=list(frame.columns), row_limit=5000,
                            formal=True, scope_column="ts_code")

    def _market_sessions(self) -> list[str]:
        conn = AshareDB(self.prod_db_path, read_only=True)
        try:
            rows = conn.query(
                "SELECT trade_date FROM trade_calendar WHERE is_open = 1 "
                f"AND trade_date >= '{self.market_start}' "
                f"AND trade_date <= '{self.today}' ORDER BY trade_date"
            )
        finally:
            conn.close()
        return [str(r[0]) for r in rows.itertuples(index=False)]

    def step5(self) -> None:
        print("== step 5: SW taxonomy (both generations) + membership per "
              "scope stock")
        for src in ("SW2021", "SW"):
            for level in ("L1", "L2", "L3"):
                params = {"src": src, "level": level}
                if self._should_skip("industry_taxonomy", params):
                    self.counts["skipped"] += 1
                    continue
                frame = self._fetch("index_classify", **params)
                self._fetch_single_guarded("index_classify", params, frame, 500)
                self._run_batch("industry_taxonomy", frame, params=params,
                                fields=list(frame.columns), row_limit=500,
                                formal=True, scope_column=None)
        for code in sorted(self.scope_codes):
            params = {"ts_code": code}
            if self._should_skip("industry_membership_pit", params):
                self.counts["skipped"] += 1
                continue
            frame = self._fetch("index_member_all", **params)
            self._fetch_single_guarded("index_member_all", params, frame, 2000)
            self._run_batch("industry_membership_pit", frame, params=params,
                            fields=list(frame.columns), row_limit=2000,
                            formal=True, scope_column="ts_code")

    def step6(self) -> None:
        print("== step 6: statements per stock (full history) + fina_audit "
              "+ disclosure_date per quarter")
        for endpoint, table, limit in (
            ("income", "income_statement_pit", 3000),
            ("balancesheet", "balance_sheet_pit", 3000),
            ("cashflow", "cashflow_statement_pit", 3000),
            ("fina_indicator", "financial_indicator_audit", 100),
        ):
            for code in sorted(self.scope_codes):
                params = {"ts_code": code}
                if self._should_skip(table, params):
                    self.counts["skipped"] += 1
                    continue
                if endpoint == "fina_indicator":
                    # 100-row per-call cap: bisect the report-period window.
                    self._fetch_statement_windowed(
                        endpoint, table, code, limit,
                        self.fundamental_start, self.today)
                    continue
                frame = self._fetch(endpoint, **params)
                self._fetch_single_guarded(endpoint, params, frame, limit)
                self._run_batch(table, frame, params=params,
                                fields=list(frame.columns), row_limit=limit,
                                formal=True, scope_column="ts_code")
        for quarter in quarter_ends(self.fundamental_start, self.today):
            params = {"end_date": quarter}
            if self._should_skip("financial_disclosure_schedule", params):
                self.counts["skipped"] += 1
                continue
            frame = self._fetch("disclosure_date", **params)
            self._fetch_single_guarded("disclosure_date", params, frame, 6000)
            self._run_batch("financial_disclosure_schedule", frame,
                            params=params, fields=list(frame.columns),
                            row_limit=6000, formal=True, scope_column="ts_code")

    def _fetch_statement_windowed(self, endpoint: str, table: str, code: str,
                                  limit: int, start: str, end: str) -> None:
        """fina_indicator: bisect the report-period window on the cap
        signal; each fetch is its own batch."""

        stack: list[tuple[str, str]] = [(start, end)]
        while stack:
            lo, hi = stack.pop()
            params = {"ts_code": code, "start_date": lo, "end_date": hi}
            frame = self._fetch(endpoint, **params)
            if len(frame) >= limit and lo != hi:
                mid = _mid(lo, hi)
                if mid < hi:
                    print(f"  paginate {endpoint} {code} {lo}..{hi}: "
                          f"{len(frame)} rows == cap, bisecting")
                    stack.append((lo, mid))
                    stack.append((add_days(mid, 1), hi))
                    continue
            self._fetch_single_guarded(endpoint, params, frame, limit)
            self._run_batch(table, frame, params=params,
                            fields=list(frame.columns), row_limit=limit,
                            formal=True, scope_column="ts_code")

    def step7(self) -> None:
        print("== step 7: forecast / express / repurchase / "
              "stk_holdernumber by year windows (bisected on cap) + "
              "index_weight per configured index")
        # repurchase / stk_holdernumber accept bare start_date/end_date
        # windows (docs 124/166); year windows bisected on the cap signal.
        windowed = (
            ("repurchase", "repurchase_event_pit", 3000),
            ("stk_holdernumber", "holder_number_pit", 3000),
        )
        for endpoint, table, limit in windowed:
            for lo, hi in year_windows(self.fundamental_start, self.today):
                stack: list[tuple[str, str]] = [(lo, hi)]
                while stack:
                    wlo, whi = stack.pop()
                    params = {"start_date": wlo, "end_date": whi}
                    frame = self._fetch(endpoint, **params)
                    if len(frame) >= limit and wlo != whi:
                        mid = _mid(wlo, whi)
                        if mid < whi:
                            print(f"  paginate {endpoint} {wlo}..{whi}: "
                                  f"{len(frame)} rows == cap, bisecting")
                            stack.append((wlo, mid))
                            stack.append((add_days(mid, 1), whi))
                            continue
                    self._fetch_single_guarded(endpoint, params, frame, limit)
                    self._run_batch(table, frame, params=params,
                                    fields=list(frame.columns), row_limit=limit,
                                    formal=True, scope_column="ts_code")
        # forecast / express REQUIRE ann_date or ts_code (bare start/end
        # windows are rejected with a parameter-validation error, observed
        # live 2026-09-04) — per-stock full history, docs 45/46.
        per_stock = (
            ("forecast", "earnings_forecast_pit", 3500),
            ("express", "earnings_express_pit", 3000),
        )
        for endpoint, table, limit in per_stock:
            for code in sorted(self.scope_codes):
                params = {"ts_code": code}
                if self._should_skip(table, params):
                    self.counts["skipped"] += 1
                    continue
                frame = self._fetch(endpoint, **params)
                self._fetch_single_guarded(endpoint, params, frame, limit)
                self._run_batch(table, frame, params=params,
                                fields=list(frame.columns), row_limit=limit,
                                formal=True, scope_column="ts_code")
        for index_code in ("000300.SH", "000905.SH", "000852.SH"):
            params = {"index_code": index_code}
            if self._should_skip("index_weight_pit", params):
                self.counts["skipped"] += 1
                continue
            frame = self._fetch("index_weight", **params)
            self._fetch_single_guarded("index_weight", params, frame, 3000)
            self._run_batch("index_weight_pit", frame, params=params,
                            fields=list(frame.columns), row_limit=3000,
                            formal=True, scope_column="con_code")

    def step8(self) -> None:
        print("== step 8: v2 manifest")
        manifest = build_v2_dataset_manifest(
            self.store.db,
            source_versions={
                "tushare_store": TUSHARE_STORE_SCHEMA_VERSION,
                "backfill": "d3:1",
            },
        )
        save_manifest(self.store.db, None, manifest)
        print(f"[manifest] dataset_id={manifest.dataset_id} "
              f"tables={len(manifest.tables)} rows={manifest.total_rows} "
              f"range={manifest.date_range}")

    # -- driver ------------------------------------------------------------

    def run(self, steps: list[str]) -> None:
        handlers = {
            "2": self.step2, "3": self.step3, "4": self.step4,
            "5": self.step5, "6": self.step6, "7": self.step7,
            "8": self.step8,
        }
        for step in steps:
            handlers[step]()
            self._progress(f"step{step}-done")
        print("[done] batches=%d rows=%d skipped=%d elapsed=%.1fmin" % (
            self.counts["batches"], self.counts["rows"],
            self.counts["skipped"], (time.monotonic() - self.started) / 60,
        ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", default="2,3,4,5,6,7,8",
                        help="ordered step list, e.g. '2,3' or '3' to resume")
    parser.add_argument("--market-start", default=MARKET_START)
    parser.add_argument("--fundamental-start", default=FUNDAMENTAL_START)
    parser.add_argument("--db", default="data/ashare_v2.duckdb")
    parser.add_argument("--raw-root", default="data/raw/tushare")
    parser.add_argument("--prod-db",
                        default=str(Path(r"D:\minequant\AlphaGPT\data\ashare.duckdb")))
    parser.add_argument("--config", default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)

    runner = BackfillRunner(args)
    try:
        runner.run([s.strip() for s in args.steps.split(",") if s.strip()])
    except BackfillStopCondition as exc:
        print(f"[STOP-CONDITION] {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
