"""Centralized production gate runner (T0-03).

Every formal entry — training, the evaluation protocol, backtest,
simulation, archiving and the web API — is gated on the same
:class:`ProductionGateRunner`.  The runner merges the historical G6/G7
gates with the strict PIT universe contract into one auditable check list:

* G1  every configured index has historical membership intervals (not a
       current snapshot stretched over history)
* G2  every participating stock has a valid listing date
* G3  the open-session calendar covers the daily-bar window
* G4  membership intervals are non-overlapping and non-duplicate
* G5  the strict PIT universe contract resolves (dev mode may fall back)
* G6  every major backtest window has at least ``min_eligible`` eligible
       stocks (yearly slices of the PIT eligibility mask)
* G7  every membership interval that overlaps the daily-bar horizon has
       daily bars — a zero-bar historical member (delisted/merged/never
       synced) fails the gate in formal mode (survivorship audit)
* G8  daily bars are fresh: the bar window ends within
       ``FRESHNESS_TOLERANCE_SESSIONS`` open sessions of the most recent
       completed open session (p16 data-freshness contract; lag counts
       open sessions, never calendar days)

Modes:

* ``formal`` — :meth:`ProductionGateRunner.require_production` raises
  :class:`~ashare_data.universe.UniverseContractError` listing every
  failing check; no representative run may proceed with a red gate.
* ``dev`` — :meth:`ProductionGateRunner.run` performs the same checks with
  the universe-contract development fallback enabled; failures degrade the
  result instead of raising, and the result always carries
  ``degraded=True`` so no consumer can mistake a dev run for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import DataConfig
from .db import AshareDB, sql_quoted_list
from .universe import (
    ResolvedUniverse,
    UniverseContractError,
    UniverseContractStatus,
    UniversePolicy,
    build_universe_mask,
    dev_fallback_membership_records,
    member_bar_coverage,
    membership_interval_issues,
    resolve_universe_contract,
)

MIN_ELIGIBLE_DEFAULT = 100

# p16 §2.1: how many open sessions the daily-bar window may lag behind the
# most recent completed open session.  Deployment tuning knob (constructor
# override), never a bypass — 3 tolerates one missed overnight sync cycle
# and catches the 8-session drift observed in the p15 freshness log.
FRESHNESS_TOLERANCE_SESSIONS = 3

_SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class GateCheck:
    """One auditable gate check outcome."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class GateResult:
    """Full gate outcome.  ``degraded`` is always True in dev mode, and
    also True when any check failed, so a degraded result can never be
    mistaken for a production-clean one."""

    mode: str
    ok: bool
    degraded: bool
    checks: tuple[GateCheck, ...]
    status: UniverseContractStatus | None
    warnings: tuple[str, ...]

    def check(self, name: str) -> GateCheck:
        for item in self.checks:
            if item.name == name:
                return item
        raise KeyError(name)


def _dev_membership_records(
    config: DataConfig, contract: ResolvedUniverse
) -> list[dict]:
    """All-period membership records for the development fallback.

    Delegates to the single definition in :mod:`ashare_data.universe`
    (arch-review F4).
    """

    return dev_fallback_membership_records(config.index_codes[0], contract)


def _contract_eligible_mask(
    config: DataConfig, contract: ResolvedUniverse
) -> tuple[np.ndarray, list[str]]:
    """Build the ``[stock, date]`` PIT eligibility mask directly from the
    database (the same ``build_universe_mask`` contract the loader uses,
    without computing any factor tensors).  Returns ``(mask, dates)`` with
    ``dates`` the horizon-capped session axis the mask columns align to.
    """

    with AshareDB(config.duckdb_path, read_only=True) as db:
        code_list = sql_quoted_list(contract.codes)
        bars = db.query(
            f"SELECT ts_code, trade_date FROM {config.daily_table} "
            f"WHERE ts_code IN ({code_list})"
        )
        horizon = db.query(
            f"SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx "
            f"FROM {config.daily_table}"
        ).iloc[0]
    if bars.empty or pd.isna(horizon["mn"]):
        raise UniverseContractError("daily table has no bars for the universe")
    dates = [
        date
        for date in contract.sessions
        if str(horizon["mn"]) <= date <= str(horizon["mx"])
    ]
    presence = bars.assign(_present=1.0).pivot_table(
        index="ts_code", columns="trade_date", values="_present", aggfunc="last"
    )
    presence = presence.reindex(index=contract.codes, columns=dates)
    presence = presence.notna().to_numpy(dtype=bool)
    membership = (
        contract.constituents.to_dict("records")
        if not contract.status.degraded
        else _dev_membership_records(config, contract)
    )
    list_dates = {
        str(row["ts_code"]): row["list_date"]
        for row in contract.stocks.to_dict("records")
    }
    policy = UniversePolicy(
        index_codes=tuple(str(code) for code in config.index_codes),
        min_listed_sessions=config.min_listed_sessions,
        membership_end_inclusive=False,
    )
    mask = build_universe_mask(
        contract.codes,
        dates,
        contract.sessions,
        membership,
        list_dates,
        presence,
        policy,
    )
    return mask.eligible, dates


class ProductionGateRunner:
    """One gate implementation for every formal entry point.

    The runner reads the database read-only; it never mutates data and
    never loads factor tensors, so calling it at the start of a formal
    run costs a bar scan plus a mask build, not a full training data load.
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        min_eligible: int | None = None,
        freshness_tolerance: int | None = None,
    ):
        self.config = config
        # Deployment tuning knob: how many eligible stocks each major
        # (yearly) window must have.  Never a bypass — the other gates
        # (G7 zero-bar survivorship, G5 strict contract) still hold.
        self.min_eligible = (
            MIN_ELIGIBLE_DEFAULT if min_eligible is None else int(min_eligible)
        )
        # Deployment tuning knob (p16 §2.1): open-session freshness
        # tolerance for G8.  Never a bypass — there is no way to disable
        # the freshness check, only to widen the tolerated lag.
        self.freshness_tolerance = (
            FRESHNESS_TOLERANCE_SESSIONS
            if freshness_tolerance is None
            else int(freshness_tolerance)
        )

    def run(self, mode: str = "formal", *, today: str | None = None) -> GateResult:
        """Execute all gates and return the auditable result.

        ``run`` itself never raises: formal callers inspect ``result.ok``
        (or use :meth:`require_production`, which raises on failure); dev
        callers get a degraded result instead of an exception.

        ``today`` is a test-only injection of the G8 evaluation day (p16
        §2.1); production callers never pass it, and no config surface
        can override it — that would be a bypass, not an injection.
        """

        if mode not in ("formal", "dev"):
            raise ValueError(f"unknown gate mode {mode!r}; expected 'formal' or 'dev'")
        dev = mode == "dev"
        checks: list[GateCheck] = []
        status: UniverseContractStatus | None = None
        evaluation_day = (
            today if today is not None else pd.Timestamp.now(tz=_SHANGHAI).strftime("%Y%m%d")
        )

        with AshareDB(self.config.duckdb_path, read_only=True) as db:
            constituents = db.query(f"SELECT * FROM {self.config.constituents_table}")
            stocks = db.query(f"SELECT * FROM {self.config.stocks_table}")
            calendar = db.query(
                f"SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx "
                f"FROM {self.config.calendar_table} WHERE is_open=true"
            ).iloc[0]
            daily = db.query(
                f"SELECT MIN(trade_date) AS mn, MAX(trade_date) AS mx "
                f"FROM {self.config.daily_table}"
            ).iloc[0]

        # G1: historical intervals per configured index, not one stretched
        # snapshot (the strict contract also rejects snapshot shapes; this
        # keeps the per-index row audit visible in the gate output).
        in_dates = constituents["in_date"].astype(str).str.replace("-", "")
        out_dates = constituents["out_date"].astype(str).str.replace("-", "")
        snapshot_shaped = (
            len(constituents) >= 2
            and in_dates.nunique() == 1
            and out_dates.nunique() == 1
            and out_dates.iloc[0] == "99991231"
        )
        per_index = {
            code: int((constituents["index_code"].astype(str) == code).sum())
            for code in self.config.index_codes
        }
        g1_ok = not snapshot_shaped and all(
            per_index.get(code, 0) > 0 for code in self.config.index_codes
        )
        checks.append(GateCheck(
            "G1 historical member intervals per configured index",
            g1_ok,
            f"rows per index: {per_index}; snapshot-shaped={snapshot_shaped}",
        ))

        # G2: valid list dates for every participating stock.
        member_codes = set(constituents["ts_code"].astype(str))
        stock_table = stocks.copy()
        stock_table["ts_code"] = stock_table["ts_code"].astype(str)
        missing_rows = sorted(member_codes - set(stock_table["ts_code"]))
        null_dates = sorted(
            stock_table.loc[
                stock_table["ts_code"].isin(member_codes)
                & stock_table["list_date"].isna(),
                "ts_code",
            ].tolist()
        )
        g2_ok = not missing_rows and not null_dates
        checks.append(GateCheck(
            "G2 valid list_date for all participating stocks",
            g2_ok,
            f"missing stock rows: {len(missing_rows)}, "
            f"null list_date: {len(null_dates)}",
        ))

        # G3: calendar covers the bar window.
        g3_ok = (
            not pd.isna(calendar["mn"])
            and str(calendar["mn"]) <= str(daily["mn"])
            and str(calendar["mx"]) >= str(daily["mx"])
        )
        checks.append(GateCheck(
            "G3 open calendar covers the daily-bar window",
            g3_ok,
            f"calendar {calendar['mn']}..{calendar['mx']} vs "
            f"daily {daily['mn']}..{daily['mx']}",
        ))

        # G4: interval structure (half-open, non-overlapping, no dupes).
        issues = membership_interval_issues(constituents)
        checks.append(GateCheck(
            "G4 no duplicate/overlapping intervals",
            not issues,
            "; ".join(issues[:2]) if issues else "clean",
        ))

        # G5: the PIT universe contract resolves (strict in formal mode;
        # the explicit development fallback in dev mode).
        try:
            contract = resolve_universe_contract(
                self.config, allow_development_fallback=dev
            )
            status = contract.status
            checks.append(GateCheck(
                "G5 strict-mode PIT universe contract" if not dev
                else "G5 PIT universe contract (dev fallback allowed)",
                True,
                f"mode={status.mode}, constituent_rows={status.constituent_rows}, "
                f"stock_rows={status.stock_rows}, "
                f"open_sessions={status.open_sessions}",
            ))
        except UniverseContractError as exc:
            checks.append(GateCheck(
                "G5 strict-mode PIT universe contract" if not dev
                else "G5 PIT universe contract (dev fallback allowed)",
                False,
                str(exc),
            ))
            contract = None

        # G6 + the mask build (shared by the eligibility audit).
        if contract is not None:
            try:
                mask, mask_dates = _contract_eligible_mask(self.config, contract)
                eligible_per_day = mask.sum(axis=0)
                year_min = {}
                for index, day in enumerate(eligible_per_day):
                    year_min.setdefault(mask_dates[index][:4], []).append(int(day))
                minima = {
                    year: min(values) for year, values in sorted(year_min.items())
                }
                g6_ok = all(value >= self.min_eligible for value in minima.values())
                checks.append(GateCheck(
                    f"G6 >= {self.min_eligible} eligible stocks per major window",
                    g6_ok,
                    "min eligible per year: " + ", ".join(
                        f"{year}:{value}" for year, value in minima.items()
                    ),
                ))
            except UniverseContractError as exc:
                checks.append(GateCheck(
                    f"G6 >= {self.min_eligible} eligible stocks per major window",
                    False,
                    f"mask unavailable: {exc}",
                ))
        else:
            checks.append(GateCheck(
                f"G6 >= {self.min_eligible} eligible stocks per major window",
                False,
                "mask unavailable (G5 failed)",
            ))

        # G7: survivorship audit — every membership interval overlapping
        # the daily-bar horizon must have bars (half-open intervals, capped
        # to the horizon by member_bar_coverage).
        with AshareDB(self.config.duckdb_path, read_only=True) as db:
            coverage = member_bar_coverage(db, self.config)
        observed = coverage[coverage["sessions"] > 0]
        zero_bar = observed[observed["bars"] == 0]
        median_cov = (
            float(observed["coverage"].median()) if len(observed) else float("nan")
        )
        g7_ok = len(zero_bar) == 0
        checks.append(GateCheck(
            "G7 every PIT member interval has daily bars",
            g7_ok,
            f"{len(observed)} observed intervals, {len(zero_bar)} zero-bar, "
            f"median coverage {median_cov:.2%}; zero-bar codes: "
            + (
                ", ".join(sorted(zero_bar["ts_code"].astype(str).unique())[:10])
                if len(zero_bar)
                else "none"
            ),
        ))

        # G8: data freshness (p16) — the daily-bar window must end within
        # the open-session tolerance of the most recent completed open
        # session.  Sources: daily_bar max(trade_date), trade_calendar
        # open sessions strictly before the evaluation day, and the
        # evaluation day itself.  Fail-closed boundaries (no daily rows,
        # no reference session, exhausted calendar horizon) never pass.
        with AshareDB(self.config.duckdb_path, read_only=True) as db:
            open_sessions = db.query(
                f"SELECT trade_date FROM {self.config.calendar_table} "
                f"WHERE is_open=true AND trade_date < ? "
                f"ORDER BY trade_date",
                [evaluation_day],
            )
        daily_max = None if pd.isna(daily["mx"]) else str(daily["mx"])
        reference = (
            str(open_sessions["trade_date"].iloc[-1])
            if not open_sessions.empty
            else None
        )
        calendar_max_open = (
            None if pd.isna(calendar["mx"]) else str(calendar["mx"])
        )
        g8_reason: str | None = None
        lag: int | None = None
        if daily_max is None:
            g8_reason = "daily table has no rows"
        elif reference is None:
            g8_reason = (
                f"no open calendar session strictly before today "
                f"{evaluation_day}: no usable reference"
            )
        elif calendar_max_open is None or calendar_max_open < evaluation_day:
            g8_reason = (
                f"calendar pre-listed horizon exhausted: max open session "
                f"{calendar_max_open} < today {evaluation_day}"
            )
        else:
            lag = int(
                (open_sessions["trade_date"].astype(str) > daily_max).sum()
            )
        g8_ok = (
            g8_reason is None and lag is not None and lag <= self.freshness_tolerance
        )
        if g8_reason is not None:
            g8_detail = f"{g8_reason}; today={evaluation_day}"
        else:
            g8_detail = (
                f"daily {daily_max} vs reference {reference}, "
                f"lag={lag} {'<=' if g8_ok else '>'} "
                f"N={self.freshness_tolerance}, today={evaluation_day}"
            )
        checks.append(GateCheck(
            "G8 daily bars fresh within the open-session tolerance",
            g8_ok,
            g8_detail,
        ))

        ok = all(check.ok for check in checks)
        warnings = tuple(
            check.detail for check in checks if not check.ok
        ) if not ok else ()
        result = GateResult(
            mode=mode,
            ok=ok,
            degraded=dev or not ok,
            checks=tuple(checks),
            status=status,
            warnings=warnings,
        )
        return result

    def require_production(
        self, *, today: str | None = None
    ) -> UniverseContractStatus:
        """Formal entry: run every gate and raise on any failure.

        ``today`` mirrors :meth:`run` — test-only injection of the G8
        evaluation day; production callers never pass it.
        """
        result = self.run(mode="formal", today=today)
        if not result.ok:
            failures = [
                f"{check.name} ({check.detail})" for check in result.checks if not check.ok
            ]
            raise UniverseContractError(
                "production gate check failed: " + "; ".join(failures)
            )
        return result.status
