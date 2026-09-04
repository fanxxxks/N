"""Tushare Pro client adapter (D1, branch codex/tushare-client).

The typed entry point for the Tushare migration (AlphaGPT PIT roadmap D1):
token handling, lazy SDK loading, fail-closed response-schema validation,
rate limiting, retry/backoff and the per-call row-cap pagination signal.

Discipline (task t3 work package):

* **Token** — ``TUSHARE_TOKEN`` is read ONLY from the process environment.
  The git-ignored ``config/.env`` may inject it through the existing
  configuration mechanism (``ashare_data.config.load_config`` runs
  ``load_dotenv(override=False)`` before the client is built); the client
  itself never reads files.  A missing token fails closed.  The token value
  never appears in ``repr``, exception messages, logs or any other output:
  every message derived from a transport error is scrubbed through
  :func:`_sanitize`, and the raised error is constructed outside the
  handling block with a sanitized cause copy, so neither ``__cause__``,
  ``__context__`` nor any traceback rendering can leak the secret (D1a).
* **Lazy SDK** — ``import tushare`` happens on the first API call, so
  non-Tushare paths never require the package or a token.
* **Fail-closed schema validation** — every registered interface has an
  exact column contract (:data:`INTERFACE_FIELDS`, frozen from the Tushare
  Pro docs).  The typed path requests exactly those fields and validates
  the response against the same set: missing columns, unexpected columns
  (a renamed/ambiguous field shows up as both) and duplicate columns raise
  :class:`TushareSchemaError` — nothing is silently skipped or renamed.
  Empty responses (zero rows, no columns) are canonicalized to the contract
  columns; an empty response that still carries unexpected columns is a
  schema violation and raises.
* **Rate limit** — sliding-window limiter, default
  :data:`MAX_REQUESTS_PER_MINUTE` (180/min, below the official 200/min cap
  of a 2000-point account; the ceiling is enforced, not advisory).
* **Retry/backoff** — 429/rate-limit and timeout failures retry with
  exponential backoff (``backoff_base * 2**attempt``, capped); permission
  and other non-retryable failures raise immediately.
* **Row cap signal** — when a response's row count reaches the interface's
  documented per-call cap (:data:`INTERFACE_ROW_LIMITS`, conservative
  lower bounds in the pagination-safe direction), the typed result carries
  ``needs_pagination=True``; the caller must paginate instead of assuming
  completeness.

The transport is reached through ``api.query(api_name, fields=..., **params)``
(the generic Tushare Pro SDK surface), which is injectable via
``api_factory`` for tests — CI never touches the network.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable

import pandas as pd

#: Hard ceiling of the client's rate limiter: below the official
#: 200 req/min of a 2000-point account (tushare.pro 积分频次对应表,
#: doc_id=290).  Constructor values above the ceiling are rejected.
MAX_REQUESTS_PER_MINUTE = 180

_WINDOW_SECONDS = 60.0
_TOKEN_ENV_VAR = "TUSHARE_TOKEN"
_REDACTED = "<redacted>"

#: The 20 core interfaces probed by ``scripts/probe_tushare.py`` (task t3).
CORE_INTERFACES = (
    "daily",
    "adj_factor",
    "daily_basic",
    "stk_limit",
    "suspend_d",
    "margin_detail",
    "index_classify",
    "index_member_all",
    "income",
    "balancesheet",
    "cashflow",
    "fina_indicator",
    "disclosure_date",
    "forecast",
    "express",
    "repurchase",
    "stk_holdernumber",
    "stock_basic",
    "index_weight",
    "trade_cal",
)

#: Interfaces probed with explicit permission expectations on a 2000-point
#: account (stock_st: 3000积分起 doc_id=397; bak_basic: 需要5000积分
#: doc_id=262; share_float: 2000积分接口).  The D1 probe (2026-09-04)
#: confirmed stock_st and bak_basic are denied; share_float unexpectedly
#: HAS permission on this account and is reported as ``unexpected_success``
#: (a planning bonus, never a failure).
EXPECTED_NO_PERMISSION_INTERFACES = ("stock_st", "share_float", "bak_basic")

#: Exact column contract per interface (frozen from the Tushare Pro docs,
#: 2026-09; extended to the p17 §5 DDL writer set in D1a where the live API
#: verifiably returns the field — margin_detail 10 columns, income
#: +oper_cost, cashflow +c_cash_equ_end_period, index_classify +src, all
#: confirmed against the real endpoint 2026-09-04).  The typed path
#: requests exactly these fields and validates the response against the
#: same set.  A new upstream column can never silently enter the pipeline:
#: extending a contract is a deliberate, reviewed edit.  Deliberate
#: NON-extension: suspend_d keeps its 3 API-supplied fields — the live API
#: silently drops resume_date/ann_date from explicit field requests (5
#: requested -> 3 returned, verified 2026-09-04), so they must stay out of
#: the typed contract (p17 §5.2 keeps both columns nullable in L1; the
#: p17 §15.6 erratum records the mapping).
INTERFACE_FIELDS: dict[str, tuple[str, ...]] = {
    "daily": (
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "pre_close",
        "close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ),
    "adj_factor": (
        "ts_code",
        "trade_date",
        "adj_factor",
    ),
    "daily_basic": (
        "ts_code",
        "trade_date",
        "close",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ),
    "stk_limit": (
        "trade_date",
        "ts_code",
        "pre_close",
        "up_limit",
        "down_limit",
    ),
    "suspend_d": (
        "ts_code",
        "trade_date",
        "suspend_type",
    ),
    "margin_detail": (
        "trade_date",
        "ts_code",
        "rzye",
        "rqye",
        "rzmre",
        "rqyl",
        "rzche",
        # API field name is `rqchl` (融券偿还量, confirmed by the D1 raw
        # probe and by explicit-request evidence 2026-09-04: requesting
        # `rqche` is silently dropped).  p17 §5.2's "rqche" is a mapping
        # typo — flagged for the p17 §15.6 erratum (t42 line); the typed
        # contract uses the source's own name, never a silent rename.
        "rqchl",
        "rqmcl",
        "rzrqye",
    ),
    "index_classify": (
        "index_code",
        "industry_name",
        "level",
        "industry_code",
        "is_pub",
        "parent_code",
        "src",
    ),
    "index_member_all": (
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "ts_code",
        "name",
        "in_date",
        "out_date",
        "is_new",
    ),
    "income": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "total_revenue",
        "revenue",
        "oper_cost",
        "operate_profit",
        "total_profit",
        "n_income",
        "n_income_attr_p",
        "basic_eps",
        "diluted_eps",
        "update_flag",
    ),
    "balancesheet": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "total_assets",
        "total_liab",
        "total_hldr_eqy_exc_min_int",
        "total_hldr_eqy_inc_min_int",
        "undistr_porfit",
        "cap_rese",
        "update_flag",
    ),
    "cashflow": (
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "report_type",
        "comp_type",
        "end_type",
        "net_profit",
        "n_cashflow_act",
        "n_cashflow_inv_act",
        "n_cash_flows_fnc_act",
        "c_cash_equ_end_period",
        "free_cashflow",
        "update_flag",
    ),
    "fina_indicator": (
        "ts_code",
        "ann_date",
        "end_date",
        "eps",
        "dt_eps",
        "roe",
        "roe_waa",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "bps",
        "ocfps",
        "update_flag",
    ),
    "disclosure_date": (
        "ts_code",
        "ann_date",
        "end_date",
        "pre_date",
        "actual_date",
        "modify_date",
    ),
    "forecast": (
        "ts_code",
        "ann_date",
        "end_date",
        "type",
        "p_change_min",
        "p_change_max",
        "net_profit_min",
        "net_profit_max",
        "last_parent_net",
        "first_ann_date",
        "summary",
        "change_reason",
    ),
    "express": (
        "ts_code",
        "ann_date",
        "end_date",
        "revenue",
        "operate_profit",
        "total_profit",
        "n_income",
        "total_assets",
        "total_hldr_eqy_exc_min_int",
        "diluted_eps",
        "diluted_roe",
        "bps",
    ),
    "repurchase": (
        "ts_code",
        "ann_date",
        "end_date",
        "proc",
        "exp_date",
        "vol",
        "amount",
        "high_limit",
        "low_limit",
    ),
    "stk_holdernumber": (
        "ts_code",
        "ann_date",
        "end_date",
        "holder_num",
    ),
    "stock_basic": (
        "ts_code",
        "symbol",
        "name",
        "area",
        "industry",
        "market",
        "exchange",
        "list_status",
        "list_date",
        "delist_date",
    ),
    "index_weight": (
        "index_code",
        "con_code",
        "trade_date",
        "weight",
    ),
    "trade_cal": (
        "exchange",
        "cal_date",
        "is_open",
        "pretrade_date",
    ),
}

#: Documented per-call row caps (conservative lower bounds — a value BELOW
#: the true cap only triggers pagination one call early, which is safe; a
#: value ABOVE it would silently accept a truncated response).  Sources:
#: daily/adj_factor/daily_basic 6000 (doc_id=27/28/32), stk_limit 7800
#: (2026-08 mirrors; the D1 probe observed a complete 7768-row full-day
#: cross-section, proving the cap is above the outdated 5800 figure),
#: fina_indicator 100 (doc_id=79), index_member_all 2000 (doc_id=335),
#: disclosure_date 6000 (doc_id=162), forecast 3500 (doc_id=45),
#: stk_holdernumber 3000 (doc_id=166); the rest are conservative estimates
#: pending real-probe evidence (adjust only on evidence).
INTERFACE_ROW_LIMITS: dict[str, int] = {
    "daily": 6000,
    "adj_factor": 6000,
    "daily_basic": 6000,
    "stk_limit": 7800,
    "suspend_d": 3000,
    "margin_detail": 5000,
    "index_classify": 500,
    "index_member_all": 2000,
    "income": 3000,
    "balancesheet": 3000,
    "cashflow": 3000,
    "fina_indicator": 100,
    "disclosure_date": 6000,
    "forecast": 3500,
    "express": 3000,
    "repurchase": 3000,
    "stk_holdernumber": 3000,
    "stock_basic": 6000,
    "index_weight": 3000,
    "trade_cal": 20000,
}

#: Substrings (case-insensitive) that mark a transport error as transient:
#: HTTP 429 / provider rate-limit messages (including the SDK's Chinese
#: wording) and socket timeouts.  Anything else — notably permission
#: errors — fails immediately.
_RETRYABLE_MARKERS = (
    "429",
    "too many requests",
    "每分钟最多",
    "每天最多",
    "超出最大",
    "超过最大",
    "timeout",
    "timed out",
)


def _sanitize(message: object, secret: str | None) -> str:
    """Scrub the secret token out of an arbitrary error message."""

    text = str(message)
    if secret:
        text = text.replace(secret, _REDACTED)
    return text


def _is_retryable(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


class TushareError(Exception):
    """Base class of every typed Tushare client failure."""


class TushareTokenMissing(TushareError):
    """No ``TUSHARE_TOKEN`` in the process environment (fail closed)."""


class TushareContractError(TushareError):
    """The requested interface has no registered column contract."""


class TushareSchemaError(TushareError):
    """A response violated the interface's exact column contract."""


class TushareRequestError(TushareError):
    """The transport call failed (message sanitized of the token)."""


class TushareRateLimiter:
    """Thread-safe sliding-window limiter (one instance per client)."""

    def __init__(
        self,
        max_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not 1 <= max_per_minute <= MAX_REQUESTS_PER_MINUTE:
            raise ValueError(
                f"max_per_minute must be within [1, {MAX_REQUESTS_PER_MINUTE}] "
                "(the 2000-point tier's official 200/min limit minus a "
                "safety margin), got "
                f"{max_per_minute}"
            )
        self.max_per_minute = int(max_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._window: deque[float] = deque()

    def acquire(self) -> None:
        """Block until one request slot is available in the 60s window."""

        while True:
            with self._lock:
                now = self._clock()
                cutoff = now - _WINDOW_SECONDS
                while self._window and self._window[0] <= cutoff:
                    self._window.popleft()
                if len(self._window) < self.max_per_minute:
                    self._window.append(now)
                    return
                wait = self._window[0] + _WINDOW_SECONDS - now
            self._sleep(max(wait, 0.0))


@dataclass(frozen=True)
class TushareResponse:
    """Typed result of one validated interface call."""

    api: str
    data: pd.DataFrame
    needs_pagination: bool
    row_limit: int


class TushareClient:
    """Fail-closed adapter over the Tushare Pro HTTP API.

    Parameters mirror the discipline above; ``api_factory(token) -> api``
    replaces the lazy SDK construction in tests (the fake must expose
    ``query(api_name, fields=..., **params)``).
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        max_per_minute: int = MAX_REQUESTS_PER_MINUTE,
        retries: int = 3,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        api_factory: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if retries < 0:
            raise ValueError(f"retries must be >= 0, got {retries}")
        if backoff_base <= 0 or backoff_cap < backoff_base:
            raise ValueError(
                "backoff_base must be > 0 and backoff_cap >= backoff_base, "
                f"got base={backoff_base} cap={backoff_cap}"
            )
        self._token = token if token is not None else self._read_token()
        self._limiter = TushareRateLimiter(
            max_per_minute, clock=clock, sleep=sleep
        )
        self._retries = int(retries)
        self._backoff_base = float(backoff_base)
        self._backoff_cap = float(backoff_cap)
        self._sleep = sleep
        self._api_factory = api_factory or self._default_api_factory
        self._api_obj: Any = None

    # --- token & SDK lifecycle ------------------------------------------

    @staticmethod
    def _read_token() -> str:
        value = os.environ.get(_TOKEN_ENV_VAR, "")
        if not value.strip():
            raise TushareTokenMissing(
                f"{_TOKEN_ENV_VAR} is not set. Provide it via the process "
                "environment, or via the git-ignored config/.env (injected "
                "into the process environment by ashare_data.config."
                "load_config before building the client). The token value "
                "must never be written into code, logs, commits or outputs."
            )
        return value.strip()

    @staticmethod
    def _default_api_factory(token: str) -> Any:
        import tushare as ts  # lazy: only real Tushare paths pay for this

        return ts.pro_api(token)

    @property
    def _api(self) -> Any:
        if self._api_obj is None:
            self._api_obj = self._api_factory(self._token)
        return self._api_obj

    def __repr__(self) -> str:  # token must never leak through repr
        return (
            f"TushareClient(max_per_minute={self._limiter.max_per_minute}, "
            f"retries={self._retries}, token={_REDACTED})"
        )

    # --- typed query path -------------------------------------------------

    def query(self, api_name: str, **params: Any) -> TushareResponse:
        """One validated interface call.

        Requests exactly the registered contract fields, validates the
        response fail-closed and flags pagination when the row count hits
        the interface's per-call cap.
        """

        fields = self._contract_fields(api_name)
        frame = self._call(api_name, fields=fields, params=params)
        frame = self._validate_frame(api_name, fields, frame)
        row_limit = INTERFACE_ROW_LIMITS[api_name]
        needs_pagination = bool(row_limit and len(frame) >= row_limit)
        return TushareResponse(
            api=api_name,
            data=frame,
            needs_pagination=needs_pagination,
            row_limit=int(row_limit),
        )

    def query_raw(self, api_name: str, **params: Any) -> pd.DataFrame:
        """Discovery path (probe tooling): the raw transport frame without
        contract validation.  Rate limiting, retry/backoff and token
        sanitization still apply.  Not for production ingestion — the typed
        :meth:`query` is the only sanctioned data path."""

        return self._call(api_name, fields=(), params=params)

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _contract_fields(api_name: str) -> tuple[str, ...]:
        try:
            return INTERFACE_FIELDS[api_name]
        except KeyError:
            registered = ", ".join(sorted(INTERFACE_FIELDS))
            raise TushareContractError(
                f"no column contract registered for interface "
                f"{api_name!r}; registered interfaces: {registered}"
            ) from None

    def _call(
        self,
        api_name: str,
        *,
        fields: tuple[str, ...],
        params: dict[str, Any],
    ) -> pd.DataFrame:
        fields_arg = ",".join(fields)
        failure: tuple[str, str] | None = None
        for attempt in range(self._retries + 1):
            self._limiter.acquire()
            try:
                return self._api.query(api_name, fields=fields_arg, **params)
            except Exception as exc:  # noqa: BLE001 - transport errors are
                # heterogeneous (requests, SDK, server); only the sanitized
                # message may ever leave this block (D1a: the original
                # exception object itself never escapes — it still carries
                # the unsanitized message and would leak through
                # __cause__/__context__/traceback rendering).
                message = _sanitize(exc, self._token)
                if attempt >= self._retries or not _is_retryable(exc):
                    failure = (type(exc).__name__, message)
                    break
                delay = min(
                    self._backoff_base * (2**attempt), self._backoff_cap
                )
                self._sleep(delay)
        if failure is not None:
            # Raised OUTSIDE the except block: the implicit __context__
            # chain is broken (no handled exception is attached), and the
            # explicit cause is a freshly built copy carrying only the
            # sanitized message plus the original type name for diagnosis.
            exc_type, message = failure
            cause = TushareError(f"{exc_type}: {message}")
            raise TushareRequestError(
                f"tushare {api_name} request failed: {message}"
            ) from cause
        raise AssertionError(  # pragma: no cover - defensive
            f"tushare {api_name}: retry loop exited without a response "
            "or a recorded failure"
        )

    def _validate_frame(
        self,
        api_name: str,
        fields: tuple[str, ...],
        frame: Any,
    ) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TushareSchemaError(
                f"tushare {api_name}: response is "
                f"{type(frame).__name__}, expected a pandas DataFrame"
            )
        if len(frame) == 0 and len(frame.columns) == 0:
            # Legitimately empty result (e.g. no suspensions on a session):
            # canonicalize to the contract columns.  No data is invented.
            return frame.reindex(columns=list(fields))
        actual = [str(column) for column in frame.columns]
        missing = [column for column in fields if column not in actual]
        unexpected = [column for column in actual if column not in fields]
        duplicated = sorted(
            {column for column in actual if actual.count(column) > 1}
        )
        if missing or unexpected or duplicated:
            raise TushareSchemaError(
                f"tushare {api_name}: response schema violated the contract "
                f"({len(fields)} fields). "
                f"missing: {missing or 'none'}; "
                f"unexpected: {unexpected or 'none'}; "
                f"duplicated: {duplicated or 'none'}; "
                + self._rename_hints(missing, unexpected)
            )
        return frame

    @staticmethod
    def _rename_hints(missing: list[str], unexpected: list[str]) -> str:
        """Name likely renames (ambiguous columns) for diagnosis."""

        hints: list[str] = []
        for expected in missing:
            for candidate in unexpected:
                ratio = SequenceMatcher(None, expected, candidate).ratio()
                if ratio >= 0.6:
                    hints.append(
                        f"{expected!r} -> {candidate!r} "
                        f"(similarity {ratio:.2f})"
                    )
        return f"possible renames: {hints or 'none'}"
