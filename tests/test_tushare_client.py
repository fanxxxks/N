"""D1 contract tests for the Tushare client (branch codex/tushare-client).

Behavior contract (task t3 work package, AGENTS §3.1 engineering change):

* ``TUSHARE_TOKEN`` is read ONLY from the process environment (the git-ignored
  ``config/.env`` injection happens through the existing
  ``ashare_data.config.load_config`` mechanism, before the client is built);
  a missing token fails closed.  The token value never appears in ``repr``,
  exception messages, logs or any other output.
* The tushare SDK is imported lazily: importing the module neither requires
  the ``tushare`` package nor a configured token.
* Interface responses are validated fail-closed against a registered column
  contract (``INTERFACE_FIELDS``): missing columns, unexpected/renamed
  (ambiguous) columns and duplicate columns raise ``TushareSchemaError`` —
  never silently skipped or renamed.  The typed path requests exactly the
  contract fields from the API.
* A response whose row count reaches the interface's per-call cap
  (``INTERFACE_ROW_LIMITS``) carries ``needs_pagination=True`` so the caller
  paginates instead of assuming completeness.
* Requests are rate-limited to <= 180 req/min (below the official 200/min
  limit of a 2000-point account) and 429/timeout failures retry with
  exponential backoff; non-retryable failures (e.g. permission errors) raise
  immediately.

All fixtures are frozen in-file (frozen small response fixtures; CI never
touches the network).  Fake tokens ("TOK_...") are placeholders — a real
token must never appear in tests, logs or artifacts.
"""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from ashare_data.tushare_client import (
    CORE_INTERFACES,
    EXPECTED_NO_PERMISSION_INTERFACES,
    INTERFACE_FIELDS,
    INTERFACE_ROW_LIMITS,
    MAX_REQUESTS_PER_MINUTE,
    TushareClient,
    TushareRequestError,
    TushareSchemaError,
    TushareTokenMissing,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAKE_TOKEN = "TOK_SECRET_SHOULD_NEVER_APPEAR"

# Documented column contracts (frozen from the Tushare Pro interface docs,
# 2026-09): the typed path requests exactly these fields per interface.
_DATE_LIKE = {
    "trade_date",
    "ann_date",
    "f_ann_date",
    "end_date",
    "cal_date",
    "pretrade_date",
    "in_date",
    "out_date",
    "pre_date",
    "actual_date",
    "modify_date",
    "enddate",
    "start_date",
    "float_date",
    "exp_date",
}
_CODE_LIKE = {
    "ts_code",
    "con_code",
    "index_code",
    "exchange",
    "src",
}
_NAME_LIKE = {
    "name",
    "industry_name",
    "l1_name",
    "l2_name",
    "l3_name",
    "suspend_type",
    "proc",
    "type",
    "is_new",
}


def _sample_value(column: str, row: int) -> object:
    if column in _DATE_LIKE:
        return f"202401{row:02d}"
    if column in _CODE_LIKE:
        return {
            "ts_code": "000001.SZ",
            "con_code": "000001.SZ",
            "index_code": "000300.SH",
            "exchange": "SSE",
            "src": "SW2021",
        }.get(column, "SSE")
    if column in _NAME_LIKE:
        return {
            "name": "平安银行",
            "industry_name": "银行",
            "l1_name": "金融",
            "l2_name": "银行",
            "l3_name": "国有大型银行",
            "suspend_type": "S",
            "proc": "实施",
            "type": "预增",
            "is_new": "Y",
        }.get(column, f"v{row}")
    if column == "is_open":
        return 1
    return 1.0 * row + 1.0


def frame_for(api: str, rows: int = 2) -> pd.DataFrame:
    """Frozen-style fixture: a frame with exactly the contract columns."""

    columns = INTERFACE_FIELDS[api]
    data = {
        column: [_sample_value(column, i) for i in range(1, rows + 1)]
        for column in columns
    }
    return pd.DataFrame(data)


class FakeApi:
    """Scripted stand-in for the lazy tushare pro_api transport object.

    Records every ``query`` call and pops one scripted outcome per call; an
    outcome that is an ``Exception`` subclass is raised, anything else is
    returned as the response frame.
    """

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, dict]] = []

    def query(self, api_name: str, fields: str = "", **params):
        self.calls.append((api_name, fields, dict(params)))
        if not self.outcomes:
            raise AssertionError("FakeApi ran out of scripted outcomes")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(
    outcomes: list,
    *,
    token: str | None = FAKE_TOKEN,
    monkeypatch: pytest.MonkeyPatch | None = None,
    **kwargs,
) -> tuple[TushareClient, FakeApi]:
    """Build a TushareClient over a scripted FakeApi (no network, no SDK)."""

    api = FakeApi(outcomes)
    kwargs.setdefault("api_factory", lambda tok: api)
    kwargs.setdefault("sleep", lambda seconds: None)
    client = TushareClient(token=token, **kwargs)
    return client, api


def load_probe_module():
    """Load ``scripts/probe_tushare.py`` as a module (no network)."""

    spec = importlib.util.spec_from_file_location(
        "probe_tushare_under_test",
        PROJECT_ROOT / "scripts" / "probe_tushare.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The module must be registered before exec: dataclass string-annotation
    # resolution looks the module up in sys.modules while it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestContractRegistry:
    def test_core_interfaces_complete(self):
        assert CORE_INTERFACES == (
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

    def test_expected_no_permission_interfaces(self):
        assert EXPECTED_NO_PERMISSION_INTERFACES == (
            "stock_st",
            "share_float",
            "bak_basic",
        )

    def test_every_core_interface_has_field_and_row_limit_contract(self):
        assert set(INTERFACE_FIELDS) == set(CORE_INTERFACES)
        assert set(INTERFACE_ROW_LIMITS) == set(CORE_INTERFACES)
        for api, fields in INTERFACE_FIELDS.items():
            assert len(fields) == len(set(fields)), (
                f"{api}: duplicate columns in the contract would make the "
                "exact-set validation ambiguous"
            )
            assert (
                "ts_code" in fields
                or "con_code" in fields
                or api in ("trade_cal", "index_classify")
            ), f"{api}: contract must carry an instrument key"

    def test_rate_limit_ceiling_is_below_official_limit(self):
        assert MAX_REQUESTS_PER_MINUTE == 180


class TestLazyLoadingAndToken:
    def test_module_import_neither_needs_sdk_nor_token(self):
        """Importing the module must not import the tushare SDK nor read a
        token (lazy SDK + env-only token)."""

        env = {k: v for k, v in os.environ.items() if k != "TUSHARE_TOKEN"}
        script = (
            "import sys\n"
            "import ashare_data.tushare_client\n"
            "assert 'tushare' not in sys.modules, 'SDK imported eagerly'\n"
            "print('ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().endswith("ok")

    def test_missing_token_fails_closed(self, monkeypatch):
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        with pytest.raises(TushareTokenMissing) as excinfo:
            TushareClient(api_factory=lambda tok: None)
        message = str(excinfo.value)
        assert "TUSHARE_TOKEN" in message
        # The message points at the injection mechanism without echoing any
        # secret material.
        assert "process environment" in message or "config/.env" in message

    def test_token_from_process_environment(self, monkeypatch):
        monkeypatch.setenv("TUSHARE_TOKEN", "TOK_FROM_ENV")
        client, api = make_client([frame_for("daily")], token=None)
        response = client.query("daily", ts_code="000001.SZ")
        assert not response.data.empty
        assert api.calls[0][0] == "daily"

    def test_repr_never_contains_token(self):
        client, _ = make_client([])
        assert FAKE_TOKEN not in repr(client)

    def test_request_error_messages_are_sanitized(self):
        client, _ = make_client(
            [Exception(f"boom {FAKE_TOKEN}")]
        )
        with pytest.raises(TushareRequestError) as excinfo:
            client.query("daily", ts_code="000001.SZ")
        message = str(excinfo.value)
        assert FAKE_TOKEN not in message
        assert "<redacted>" in message


class TestFailClosedSchemaValidation:
    def test_requests_exactly_the_contract_fields(self):
        client, api = make_client([frame_for("daily")])
        client.query("daily", ts_code="000001.SZ")
        api_name, fields, params = api.calls[0]
        assert api_name == "daily"
        assert fields == ",".join(INTERFACE_FIELDS["daily"])
        assert params == {"ts_code": "000001.SZ"}

    def test_missing_column_raises(self):
        broken = frame_for("daily").drop(columns=["amount"])
        client, _ = make_client([broken])
        with pytest.raises(TushareSchemaError) as excinfo:
            client.query("daily", ts_code="000001.SZ")
        message = str(excinfo.value)
        assert "amount" in message
        assert "missing" in message.lower()

    def test_unexpected_ambiguous_column_raises(self):
        renamed = frame_for("daily").rename(columns={"vol": "volume"})
        client, _ = make_client([renamed])
        with pytest.raises(TushareSchemaError) as excinfo:
            client.query("daily", ts_code="000001.SZ")
        message = str(excinfo.value)
        # Both sides of the ambiguity are named: the expected column that is
        # missing and the unexpected column that appeared.
        assert "vol" in message
        assert "volume" in message

    def test_unknown_interface_raises_contract_error(self):
        from ashare_data.tushare_client import TushareContractError

        client, _ = make_client([])
        with pytest.raises(TushareContractError):
            client.query("not_a_registered_interface")

    def test_empty_response_is_canonicalized(self):
        client, _ = make_client([pd.DataFrame()])
        response = client.query("suspend_d", trade_date="20240102")
        assert response.data.empty
        assert list(response.data.columns) == list(INTERFACE_FIELDS["suspend_d"])
        assert response.needs_pagination is False

    def test_empty_response_with_wrong_columns_raises(self):
        client, _ = make_client([pd.DataFrame({"oops": []})])
        with pytest.raises(TushareSchemaError):
            client.query("suspend_d", trade_date="20240102")

    def test_query_raw_returns_unvalidated_discovery_frame(self):
        """The probe's discovery path: raw transport response, no contract
        validation, same rate limiting / retry / sanitization as the typed
        path."""

        raw = pd.DataFrame({"anything": [1, 2]})
        client, api = make_client([raw])
        frame = client.query_raw("stock_st", trade_date="20240102")
        assert frame.equals(raw)
        api_name, fields, _ = api.calls[0]
        assert api_name == "stock_st"
        assert fields == ""


class TestRowCapSignal:
    def test_rows_at_interface_limit_signal_pagination(self):
        limit = INTERFACE_ROW_LIMITS["fina_indicator"]
        client, _ = make_client([frame_for("fina_indicator", rows=limit)])
        response = client.query("fina_indicator", ts_code="000001.SZ")
        assert len(response.data) == limit
        assert response.needs_pagination is True
        assert response.row_limit == limit

    def test_rows_below_limit_are_assumed_complete(self):
        limit = INTERFACE_ROW_LIMITS["fina_indicator"]
        client, _ = make_client([frame_for("fina_indicator", rows=limit - 1)])
        response = client.query("fina_indicator", ts_code="000001.SZ")
        assert response.needs_pagination is False


class TestRateLimitingAndBackoff:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    def test_rate_limiter_spreads_calls_within_one_minute(self):
        clock = self.FakeClock()
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock.advance(seconds)

        client, _ = make_client(
            [frame_for("daily") for _ in range(3)],
            max_per_minute=2,
            clock=clock,
            sleep=sleep,
        )
        for _ in range(3):
            client.query("daily", ts_code="000001.SZ")
        # The first two calls fit the window; the third must wait until the
        # window slides (60s window at 2 req/min => a full 60s wait here).
        assert sleeps == [60.0]

    def test_rate_limit_ceiling_is_enforced(self):
        with pytest.raises(ValueError):
            make_client([], max_per_minute=MAX_REQUESTS_PER_MINUTE + 1)

    def test_rate_limit_error_retries_with_exponential_backoff(self):
        rate_limited = Exception("抱歉，您每分钟最多访问该接口200次 (429)")
        sleeps: list[float] = []
        client, api = make_client(
            [rate_limited, rate_limited, frame_for("daily")],
            retries=3,
            backoff_base=1.0,
            sleep=sleeps.append,
        )
        response = client.query("daily", ts_code="000001.SZ")
        assert not response.data.empty
        assert len(api.calls) == 3
        assert sleeps == [1.0, 2.0]

    def test_timeout_retries_with_exponential_backoff(self):
        sleeps: list[float] = []
        client, api = make_client(
            [TimeoutError("The read operation timed out"), frame_for("daily")],
            retries=3,
            backoff_base=1.0,
            sleep=sleeps.append,
        )
        response = client.query("daily", ts_code="000001.SZ")
        assert not response.data.empty
        assert len(api.calls) == 2
        assert sleeps == [1.0]

    def test_backoff_is_capped(self):
        rate_limited = Exception("抱歉，您每分钟最多访问该接口200次")
        sleeps: list[float] = []
        client, _ = make_client(
            [rate_limited for _ in range(4)],
            retries=4,
            backoff_base=2.0,
            backoff_cap=4.0,
            sleep=sleeps.append,
        )
        with pytest.raises(TushareRequestError):
            client.query("daily", ts_code="000001.SZ")
        assert sleeps == [2.0, 4.0, 4.0, 4.0]

    def test_permission_error_fails_immediately_without_retry(self):
        forbidden = Exception("抱歉，您没有访问该接口的权限")
        sleeps: list[float] = []
        client, api = make_client(
            [forbidden],
            retries=3,
            sleep=sleeps.append,
        )
        with pytest.raises(TushareRequestError) as excinfo:
            client.query("daily", ts_code="000001.SZ")
        assert len(api.calls) == 1
        assert sleeps == []
        assert FAKE_TOKEN not in str(excinfo.value)

    def test_retries_exhausted_raises_sanitized_error(self):
        rate_limited = Exception(f"每分钟最多 {FAKE_TOKEN}")
        client, api = make_client(
            [rate_limited for _ in range(3)],
            retries=2,
        )
        with pytest.raises(TushareRequestError) as excinfo:
            client.query("daily", ts_code="000001.SZ")
        assert len(api.calls) == 3
        assert FAKE_TOKEN not in str(excinfo.value)


class TestProbeSpecs:
    """Structural checks of scripts/probe_tushare.py (no network, no DB)."""

    def test_probe_specs_cover_core_and_expected_trio(self):
        probe = load_probe_module()
        specs = probe.build_probe_specs(datetime(2026, 9, 7), sessions=["20260904"])
        assert set(specs) == set(CORE_INTERFACES) | set(
            EXPECTED_NO_PERMISSION_INTERFACES
        )
        for api, spec in specs.items():
            assert isinstance(spec.params, dict)
            if api in INTERFACE_FIELDS:
                assert set(spec.date_columns) <= set(INTERFACE_FIELDS[api]), (
                    f"{api}: date range columns must exist in the contract"
                )
        for api in EXPECTED_NO_PERMISSION_INTERFACES:
            assert specs[api].expect_no_permission is True
        for api in CORE_INTERFACES:
            assert specs[api].expect_no_permission is False

    def test_probe_sample_dates_are_yyyymmdd_strings(self):
        probe = load_probe_module()
        specs = probe.build_probe_specs(datetime(2026, 9, 7), sessions=["20260904"])
        for spec in specs.values():
            for key, value in spec.params.items():
                if key.endswith("_date") or key == "enddate":
                    assert isinstance(value, str) and len(value) == 8, (
                        f"{spec.api}.{key}={value!r} is not a YYYYMMDD sample"
                    )

    def test_print_plan_lists_every_interface_without_network(self, capsys):
        probe = load_probe_module()
        plan = probe.print_plan()
        for api in list(CORE_INTERFACES) + list(EXPECTED_NO_PERMISSION_INTERFACES):
            assert api in plan
        assert "expected no permission" in plan
