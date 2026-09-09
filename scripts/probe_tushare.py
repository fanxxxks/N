"""Read-only Tushare permission probe (D1, branch codex/tushare-client).

``--probe`` requests a small sample from every registered interface and
prints (optionally archives via ``--output``) each interface's permission
status, response columns, row count and date range.  It NEVER writes any
database, cache or repository file by default — the only optional write is
the sanitized report file given to ``--output``.

Coverage (task t3): the 20 core interfaces must report permission OK; the
trio ``stock_st`` / ``share_float`` / ``bak_basic`` is probed with the
explicit expectation of NO permission on a 2000-point account — a reported
permission failure there is the expected outcome and never affects the exit
code or the migration plan.

Token discipline: ``TUSHARE_TOKEN`` is read only from the process
environment.  The git-ignored ``config/.env`` may inject it through the
existing configuration mechanism (``ashare_data.config.load_config`` runs
``load_dotenv(override=False)``).  The token value never enters this
script's output: every report is scrubbed through the client's sanitizer
before printing or writing.  This probe is a manual, local tool — it is not
part of CI and must not be run with real credentials there.

Exit codes: 0 — all core interfaces OK (trio reported informationally);
1 — missing token or probe could not start; 2 — at least one core
interface failed (permission, schema contract or transport error).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ashare_data.tushare_client import (  # noqa: E402
    CORE_INTERFACES,
    EXPECTED_NO_PERMISSION_INTERFACES,
    INTERFACE_FIELDS,
    TushareClient,
    TushareError,
    TushareTokenMissing,
    _sanitize,
)

PROBE_SAMPLE_CODE = "000001.SZ"
SAMPLE_INDEX_CODE = "000300.SH"
SAMPLE_L1_INDUSTRY = "801010.SI"
_PERMISSION_MARKERS = ("权限", "积分")


@dataclass(frozen=True)
class ProbeSpec:
    """One interface's minimal sample request (read-only, few rows)."""

    api: str
    params: dict[str, str] = field(default_factory=dict)
    #: Columns whose min/max are reported as the sample's date range.
    date_columns: tuple[str, ...] = ()
    expect_no_permission: bool = False


def _latest_quarter_end(today: datetime) -> str:
    quarter_ends = [
        (today.year, 3, 31),
        (today.year, 6, 30),
        (today.year, 9, 30),
        (today.year, 12, 31),
    ]
    past = [q for q in quarter_ends if datetime(*q) <= today]
    if not past:
        return f"{today.year - 1}1231"
    year, month, day = past[-1]
    return f"{year:04d}{month:02d}{day:02d}"


def build_probe_specs(today: datetime, sessions: list[str]) -> dict[str, ProbeSpec]:
    """Sample specs derived from the actual trade calendar.

    ``sessions`` is the probed open-session list (ascending, ``YYYYMMDD``);
    the most recent session anchors every date-keyed sample so the probe
    works on trading days, weekends and holidays alike.
    """

    latest = sessions[-1] if sessions else today.strftime("%Y%m%d")
    latest_dt = datetime.strptime(latest, "%Y%m%d")
    week_ago = (latest_dt - timedelta(days=7)).strftime("%Y%m%d")
    month_ago = (latest_dt - timedelta(days=31)).strftime("%Y%m%d")
    year_ago = f"{latest_dt.year - 1}0101"
    quarter_end = _latest_quarter_end(latest_dt)
    prior_year = {
        "start_date": year_ago,
        "end_date": latest,
    }
    return {
        # --- core interfaces (permission expected) ---
        "trade_cal": ProbeSpec(
            "trade_cal",
            {"exchange": "SSE", "start_date": week_ago, "end_date": latest},
            ("cal_date",),
        ),
        "daily": ProbeSpec(
            "daily",
            {"ts_code": PROBE_SAMPLE_CODE, "start_date": week_ago, "end_date": latest},
            ("trade_date",),
        ),
        "adj_factor": ProbeSpec(
            "adj_factor",
            {"ts_code": PROBE_SAMPLE_CODE, "start_date": week_ago, "end_date": latest},
            ("trade_date",),
        ),
        "daily_basic": ProbeSpec(
            "daily_basic",
            {"ts_code": PROBE_SAMPLE_CODE, "start_date": week_ago, "end_date": latest},
            ("trade_date",),
        ),
        "stk_limit": ProbeSpec("stk_limit", {"trade_date": latest}, ("trade_date",)),
        "suspend_d": ProbeSpec("suspend_d", {"trade_date": latest}, ("trade_date",)),
        "margin_detail": ProbeSpec(
            "margin_detail", {"trade_date": latest}, ("trade_date",)
        ),
        "index_classify": ProbeSpec(
            "index_classify", {"src": "SW2021", "level": "L1"}, ()
        ),
        "index_member_all": ProbeSpec(
            "index_member_all", {"l1_code": SAMPLE_L1_INDUSTRY}, ("in_date", "out_date")
        ),
        "income": ProbeSpec(
            "income",
            {"ts_code": PROBE_SAMPLE_CODE, **prior_year},
            ("ann_date", "f_ann_date", "end_date"),
        ),
        "balancesheet": ProbeSpec(
            "balancesheet",
            {"ts_code": PROBE_SAMPLE_CODE, **prior_year},
            ("ann_date", "f_ann_date", "end_date"),
        ),
        "cashflow": ProbeSpec(
            "cashflow",
            {"ts_code": PROBE_SAMPLE_CODE, **prior_year},
            ("ann_date", "f_ann_date", "end_date"),
        ),
        "fina_indicator": ProbeSpec(
            "fina_indicator",
            {"ts_code": PROBE_SAMPLE_CODE, **prior_year},
            ("ann_date", "end_date"),
        ),
        "disclosure_date": ProbeSpec(
            "disclosure_date", {"end_date": quarter_end}, ("pre_date", "actual_date")
        ),
        "forecast": ProbeSpec(
            "forecast", {"ts_code": PROBE_SAMPLE_CODE, **prior_year}, ("ann_date", "end_date")
        ),
        "express": ProbeSpec(
            "express", {"ts_code": PROBE_SAMPLE_CODE, **prior_year}, ("ann_date", "end_date")
        ),
        "repurchase": ProbeSpec(
            "repurchase",
            {"start_date": month_ago, "end_date": latest},
            ("ann_date", "end_date"),
        ),
        "stk_holdernumber": ProbeSpec(
            "stk_holdernumber",
            {"ts_code": PROBE_SAMPLE_CODE, **prior_year},
            ("ann_date", "end_date"),
        ),
        "stock_basic": ProbeSpec("stock_basic", {}, ("list_date", "delist_date")),
        "index_weight": ProbeSpec(
            "index_weight",
            {
                "index_code": SAMPLE_INDEX_CODE,
                "start_date": month_ago,
                "end_date": latest,
            },
            ("trade_date",),
        ),
        # --- expected NO permission on a 2000-point account ---
        "stock_st": ProbeSpec(
            "stock_st", {"trade_date": latest}, ("trade_date",), True
        ),
        "share_float": ProbeSpec(
            "share_float",
            {"ts_code": PROBE_SAMPLE_CODE, "start_date": month_ago, "end_date": latest},
            ("float_date",),
            True,
        ),
        "bak_basic": ProbeSpec(
            "bak_basic", {"trade_date": latest}, ("trade_date",), True
        ),
    }


@dataclass
class ProbeOutcome:
    """One interface's probe result (never contains the token)."""

    api: str
    status: str
    rows: int | None = None
    columns: tuple[str, ...] = ()
    date_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    contract_match: str | None = None
    needs_pagination: bool | None = None
    detail: str = ""
    #: Raw discovery frame (internal; used to derive the calendar anchor,
    #: excluded from the rendered report).
    frame: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def render(self) -> str:
        lines = [f"[{self.api}] status: {self.status}"]
        if self.rows is not None:
            lines.append(f"  rows: {self.rows}")
        if self.columns:
            lines.append(f"  columns ({len(self.columns)}): {', '.join(self.columns)}")
        for column, (low, high) in self.date_ranges.items():
            lines.append(f"  range {column}: {low} .. {high}")
        if self.contract_match is not None:
            lines.append(f"  contract_match: {self.contract_match}")
        if self.needs_pagination is not None:
            lines.append(f"  needs_pagination: {self.needs_pagination}")
        if self.detail:
            lines.append(f"  detail: {self.detail}")
        return "\n".join(lines)


def _date_ranges(frame: pd.DataFrame, date_columns: tuple[str, ...]) -> dict:
    ranges: dict[str, tuple[str, str]] = {}
    for column in date_columns:
        if column in frame.columns and not frame.empty:
            values = frame[column].dropna().astype(str)
            if not values.empty:
                ranges[column] = (values.min(), values.max())
    return ranges


def _permission_denied(message: str) -> bool:
    return any(marker in message for marker in _PERMISSION_MARKERS)


def _probe_interface(client: TushareClient, spec: ProbeSpec) -> ProbeOutcome:
    """Discovery call + typed contract verification for one interface."""

    try:
        frame = client.query_raw(spec.api, **spec.params)
    except TushareError as exc:
        if spec.expect_no_permission and _permission_denied(str(exc)):
            return ProbeOutcome(
                spec.api,
                "no_permission (expected)",
                detail=f"{exc} — reported to the plan as expected; not a failure",
            )
        status = (
            "no_permission (UNEXPECTED for a core interface — affects the plan)"
            if not spec.expect_no_permission and _permission_denied(str(exc))
            else "error"
        )
        return ProbeOutcome(spec.api, status, detail=str(exc))

    outcome = ProbeOutcome(
        spec.api,
        "ok" if not spec.expect_no_permission else "unexpected_success (has permission)",
        rows=int(len(frame)),
        columns=tuple(str(column) for column in frame.columns),
        date_ranges=_date_ranges(frame, spec.date_columns),
        frame=frame,
    )
    if spec.api in INTERFACE_FIELDS:
        try:
            typed = client.query(spec.api, **spec.params)
        except TushareError as exc:
            outcome.contract_match = f"FAILED: {exc}"
        else:
            outcome.contract_match = "ok"
            outcome.needs_pagination = typed.needs_pagination
    return outcome


def _load_config_safely(config_path: str | None) -> str:
    """Run the existing configuration load so a git-ignored config/.env can
    inject ``TUSHARE_TOKEN`` into the process environment (override=False:
    pre-existing environment variables win).  A missing/broken config only
    degrades to the raw process environment."""

    from ashare_data.config import load_config

    try:
        load_config(config_path, project_root=ROOT)
        return "config loaded (config/.env injection active)"
    except Exception as exc:  # noqa: BLE001 - probe must still run
        return f"config not loaded ({exc}); using the raw process environment"


def _open_sessions_from_calendar(
    outcome: ProbeOutcome, today: datetime
) -> list[str]:
    """Open sessions (ascending, ``YYYYMMDD``) of the probed calendar window.

    Prefers the actual ``is_open`` flag; the latest open session not after
    today anchors every date-keyed sample.
    """

    frame = outcome.frame
    if frame is None or frame.empty or "cal_date" not in frame.columns:
        return []
    dates = frame["cal_date"].astype(str)
    if "is_open" in frame.columns:
        open_mask = pd.to_numeric(frame["is_open"], errors="coerce").fillna(0) == 1
        dates = dates[open_mask]
    today_str = today.strftime("%Y%m%d")
    return sorted({d for d in dates if d <= today_str})


def _contract_check_failed(outcome: ProbeOutcome) -> bool:
    return bool(
        outcome.contract_match and outcome.contract_match.startswith("FAILED")
    )


def run_probe(config_path: str | None = None) -> tuple[int, str]:
    """Run the read-only probe; return ``(exit_code, report_text)``."""

    lines = [
        "Tushare read-only permission probe (D1)",
        f"ran at: {datetime.now().isoformat(timespec='seconds')}",
        _load_config_safely(config_path),
        "writes: none (no database, no cache; only this report)",
        "",
    ]
    try:
        client = TushareClient(retries=2)
    except TushareTokenMissing as exc:
        lines.append(f"TOKEN MISSING: {exc}")
        return 1, _finalize("\n".join(lines), None)

    today = datetime.now()
    specs = build_probe_specs(today, sessions=[])
    # trade_cal first: the real open sessions anchor every date-keyed sample.
    calendar_outcome = _probe_interface(client, specs["trade_cal"])
    sessions = _open_sessions_from_calendar(calendar_outcome, today)
    specs = build_probe_specs(today, sessions=sessions)
    if sessions:
        lines.append(f"anchor open sessions from trade_cal: {sessions[0]} .. {sessions[-1]}")
    else:
        lines.append("trade_cal anchor unavailable; using calendar-independent sample dates")
    lines.append("")

    order = list(CORE_INTERFACES) + list(EXPECTED_NO_PERMISSION_INTERFACES)
    core_failures: list[str] = []
    for api in order:
        outcome = _probe_interface(client, specs[api])
        lines.append(outcome.render())
        lines.append("")
        if api in CORE_INTERFACES and (
            outcome.status != "ok" or _contract_check_failed(outcome)
        ):
            core_failures.append(api)
    summary = (
        f"summary: {len(order) - len(core_failures)}/{len(order)} probes OK; "
        f"core failures: {core_failures or 'none'}; "
        f"trio {EXPECTED_NO_PERMISSION_INTERFACES} reported above "
        "(no-permission is the expected outcome there)"
    )
    lines.append(summary)
    exit_code = 2 if core_failures else 0
    return exit_code, _finalize("\n".join(lines), client)


def _finalize(report: str, client: TushareClient | None) -> str:
    """Scrub the token out of the full report (belt and braces: the client
    already sanitizes every exception message)."""

    if client is not None:
        secret = getattr(client, "_token", None)
        if secret:
            report = _sanitize(report, secret)
    return report


def print_plan() -> str:
    """The no-network probe plan (interfaces + sample params)."""

    specs = build_probe_specs(datetime.now(), sessions=[])
    lines = ["probe plan (no requests made):"]
    for api in list(CORE_INTERFACES) + list(EXPECTED_NO_PERMISSION_INTERFACES):
        spec = specs[api]
        marker = " [expected no permission]" if spec.expect_no_permission else ""
        lines.append(f"  {api}{marker}: params={spec.params}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run the read-only probe (small samples, no database writes)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to ashare_config.yaml (default: config/ashare_config.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="also write the sanitized report to this UTF-8 text file",
    )
    args = parser.parse_args(argv)
    if not args.probe:
        print(__doc__.strip())
        print()
        print(print_plan())
        return 0
    exit_code, report = run_probe(args.config)
    print(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")
        print(f"report written: {output_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
