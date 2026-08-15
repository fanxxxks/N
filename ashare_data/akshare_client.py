"""AkShare client with graceful fallbacks and optional offline fixtures."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from .processor import is_valid_a_share_code


class AkShareUnavailable(Exception):
    """Raised when an AkShare endpoint is unavailable and no fallback exists."""


def _call_with_timeout(fn: Callable, timeout: float | None) -> Any:
    """Run ``fn`` bounded by ``timeout`` seconds.

    A hung HTTP call inside AkShare blocks forever without this: the
    function runs in a daemon thread and a timeout raises TimeoutError so
    the retry loop can move on (the abandoned thread dies with the
    process).
    """

    if not timeout:
        return fn()
    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - forwarded to the caller.
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"AkShare call timed out after {timeout}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _retry(
    fn: Callable,
    retries: int = 3,
    delay: float = 1.0,
    timeout: float | None = None,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return _call_with_timeout(fn, timeout)
        except Exception as exc:  # noqa: BLE001 - AkShare errors are heterogeneous.
            last_exc = exc
            logger.warning(f"AkShare call failed ({attempt + 1}/{retries}): {exc}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise AkShareUnavailable(str(last_exc))


def _symbol_from_ts_code(ts_code: str) -> str:
    return ts_code.split(".")[0]


def _ts_code_from_symbol(symbol: str) -> str:
    """Map a bare 6-digit symbol to an exchange-qualified code.

    The result is validated with :func:`is_valid_a_share_code` by callers, so
    B-shares and other instruments are dropped even if a suffix is attached.
    """

    symbol = str(symbol).strip()
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    if symbol.startswith(("0", "3")):
        return f"{symbol}.SZ"
    if symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    if symbol.startswith("9"):
        return f"{symbol}.SH"  # B-shares; rejected by the validator.
    return f"{symbol}.SZ"


def _sina_symbol(ts_code: str) -> str:
    """Convert ``000001.SZ`` to Sina's ``sz000001`` format."""
    symbol, suffix = ts_code.split(".", 1)
    market = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix, "sz")
    return f"{market}{symbol}"


class AkShareClient:
    """Thin AkShare adapter used by ashare_data.sync.

    Set ``offline=True`` to use deterministic local fixtures for tests and
    development without network access.
    """

    def __init__(
        self,
        config,
        offline: bool | None = None,
        fixture_dir: str | Path | None = None,
    ):
        self.config = config
        self.offline = (
            offline
            if offline is not None
            else os.getenv("ASHARE_OFFLINE", "0").lower() in {"1", "true", "yes"}
        )
        self.fixture_dir = Path(fixture_dir) if fixture_dir else Path("tests/fixtures")
        # Circuit breaker for the Eastmoney daily endpoint: after a few
        # consecutive failures we skip it and go straight to the fallback so
        # a blocked host does not stall the whole sync with retries.
        self._em_fail_streak = 0
        self._em_broken = False

    def _fetch(
        self, fn: Callable, retries: int | None = None, delay: float = 1.0
    ) -> Any:
        """Bounded AkShare call using the configured retries and timeout.

        Every network call in this client goes through here so a hung
        endpoint can never stall the sync indefinitely.
        """

        return _retry(
            fn,
            retries if retries is not None else self.config.request_retries,
            delay,
            self.config.request_timeout,
        )

    def _em_trip(self, exc: Exception) -> None:
        self._em_fail_streak += 1
        if self._em_fail_streak >= 3 and not self._em_broken:
            self._em_broken = True
            logger.warning(
                "Eastmoney daily endpoint failing repeatedly; "
                f"switching to Sina for the rest of this run ({exc})"
            )

    def _fixture_path(self, name: str) -> Path:
        return self.fixture_dir / f"{name}.json"

    def _load_fixture(self, name: str) -> pd.DataFrame | None:
        path = self._fixture_path(name)
        if not path.exists():
            return None
        return pd.read_json(path, orient="records", dtype=False)

    def get_trade_calendar(self) -> list[str]:
        if self.offline:
            df = self._load_fixture("calendar")
            if df is not None:
                return sorted(df["trade_date"].astype(str).tolist())
            return pd.bdate_range(self.config.start_date, self.config.end_date).strftime(
                "%Y%m%d"
            ).tolist()

        try:
            import akshare as ak

            df = self._fetch(ak.tool_trade_date_hist_sina)
            dates = sorted(pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d").tolist())
            start = self.config.start_date.replace("-", "")
            end = self.config.end_date.replace("-", "")
            dates = [d for d in dates if start <= d <= end]
            if dates:
                return dates
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Trade calendar unavailable, using business days: {exc}")
        return pd.bdate_range(self.config.start_date, self.config.end_date).strftime(
            "%Y%m%d"
        ).tolist()

    def get_stock_list(self) -> pd.DataFrame:
        if self.offline:
            fixture = self._load_fixture("stocks")
            if fixture is not None and not fixture.empty:
                fixture = fixture[
                    fixture["ts_code"].astype(str).apply(is_valid_a_share_code)
                ]
                if not fixture.empty:
                    return fixture
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "name": "平安银行",
                        "industry": "银行",
                        "list_date": "19910403",
                        "is_st": False,
                    },
                    {
                        "ts_code": "600000.SH",
                        "name": "浦发银行",
                        "industry": "银行",
                        "list_date": "19991110",
                        "is_st": False,
                    },
                ]
            )

        import akshare as ak

        try:
            df = self._fetch(ak.stock_info_a_code_name)
        except Exception:
            df = self._fetch(ak.stock_zh_a_spot_em)
        df = df.rename(columns={"code": "symbol", "名称": "name", "name": "name"})
        df["ts_code"] = df["symbol"].apply(_ts_code_from_symbol)
        df = df[df["ts_code"].apply(is_valid_a_share_code)]
        names = df["name"].astype(str)
        df["is_st"] = names.str.upper().str.contains("ST", na=False)
        df["industry"] = None
        df["list_date"] = None
        return df[["ts_code", "name", "industry", "list_date", "is_st"]]

    def get_constituents(self, index_code: str) -> list[str]:
        if self.offline:
            df = self._load_fixture(f"constituents_{index_code}")
            if df is not None:
                codes = [
                    c
                    for c in df["ts_code"].astype(str).tolist()
                    if is_valid_a_share_code(c)
                ]
                return codes
            return ["000001.SZ", "600000.SH"]

        import akshare as ak

        symbol = index_code.split(".")[0]
        errors: list[str] = []
        for fn, kwargs in (
            (ak.index_stock_cons_csindex, {"symbol": symbol}),
            (ak.index_stock_cons, {"symbol": symbol}),
            (ak.index_stock_cons_weight_csindex, {"symbol": symbol}),
        ):
            try:
                df = self._fetch(lambda: fn(**kwargs))
                cols = {str(c).lower() for c in df.columns}
                code_col = next((c for c in df.columns if "代码" in str(c) or "code" in str(c).lower()), None)
                if code_col is None:
                    continue
                codes = df[code_col].astype(str).tolist()
                if not codes:
                    continue
                codes = [
                    c if "." in c else _ts_code_from_symbol(c) for c in codes
                ]
                # Keep only real A-share stocks; the index itself and other
                # instruments must never enter the trading universe.
                codes = [c for c in codes if is_valid_a_share_code(c)]
                if not codes:
                    continue
                return codes
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        logger.warning(
            f"Constituents unavailable for {index_code}: {'; '.join(errors[-2:])}"
        )
        return []

    def get_daily_bar(
        self,
        ts_code: str,
        start_date: str = "20150101",
        end_date: str = "20261231",
    ) -> pd.DataFrame:
        if self.offline:
            fixture = self._load_fixture(f"daily_{ts_code}")
            if fixture is not None:
                return fixture
            return pd.DataFrame()

        import akshare as ak

        symbol = _symbol_from_ts_code(ts_code)
        provider = getattr(self.config, "daily_provider", "auto") or "auto"
        df: pd.DataFrame | None = None
        source = ""

        if provider in ("auto", "eastmoney") and not self._em_broken:
            try:
                df = self._fetch(
                    lambda: ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=self.config.adjust,
                    ),
                    retries=1,
                )
                source = "eastmoney"
            except AkShareUnavailable as exc:
                self._em_trip(exc)

        if (df is None or df.empty) and provider in ("auto", "sina"):
            try:
                df = self._fetch(
                    lambda: ak.stock_zh_a_daily(
                        symbol=_sina_symbol(ts_code),
                        start_date=start_date,
                        end_date=end_date,
                        adjust=self.config.adjust,
                    ),
                    self.config.request_retries,
                )
                source = "sina"
            except AkShareUnavailable as exc:
                raise AkShareUnavailable(
                    f"All daily providers failed for {ts_code}: {exc}"
                )

        if df is None or df.empty:
            return pd.DataFrame()

        if source == "eastmoney":
            rename = {
                "日期": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover_rate",
            }
            df = df.rename(columns=rename)
            if "turnover_rate" in df.columns:
                # Eastmoney reports turnover in percent; store as fraction so
                # both providers share one unit.
                df["turnover_rate"] = (
                    pd.to_numeric(df["turnover_rate"], errors="coerce") / 100.0
                )
        else:  # sina
            rename = {
                "date": "trade_date",
                "turnover": "turnover_rate",
            }
            df = df.rename(columns=rename)

        if "trade_date" not in df.columns:
            return pd.DataFrame()
        df["ts_code"] = ts_code
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
        return df

    def get_earnings_report(self, quarter: str) -> pd.DataFrame:
        """Whole-market earnings report for one quarter (Eastmoney 业绩报表).

        ``quarter`` is a period-end date like ``20250331``.  Returns
        canonical rows with the cumulative year-to-date values and the YoY
        growth rates.  The endpoint's own announcement column carries
        restatement dates (every quarter of a fiscal year is re-published
        when the annual report lands), so the point-in-time date is set
        deterministically to the END of the quarter's statutory disclosure
        season: Q1 -> 04-30, H1 -> 08-31, Q3 -> 10-31, annual -> 04-30 of
        the next year.  This is conservative (a value is never visible
        before its season ends) and needs no per-stock announcement feed.
        Rows for non-A-share instruments are dropped by the code validator.
        """

        if self.offline:
            df = self._load_fixture(f"earnings_{quarter}")
            if df is None or df.empty:
                return pd.DataFrame()
        else:
            import akshare as ak

            df = self._fetch(lambda: ak.stock_yjbb_em(date=quarter))
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "股票代码": "symbol",
                "每股收益": "eps_cum",
                "营业总收入-营业总收入": "revenue_cum",
                "营业总收入-同比增长": "revenue_yoy",
                "净利润-净利润": "profit_cum",
                "净利润-同比增长": "profit_yoy",
                "每股净资产": "bvps",
                "净资产收益率": "roe",
                "销售毛利率": "gross_margin",
            }
        )
        if "symbol" not in df.columns:
            return pd.DataFrame()
        df["ts_code"] = df["symbol"].astype(str).apply(_ts_code_from_symbol)
        df = df[df["ts_code"].apply(is_valid_a_share_code)]
        year = int(quarter[:4])
        season_end = {
            "0331": f"{year}0430",
            "0630": f"{year}0831",
            "0930": f"{year}1031",
            "1231": f"{year + 1}0430",
        }.get(quarter[4:], quarter)
        df["announce_date"] = season_end
        df["report_date"] = quarter
        numeric = [
            "eps_cum",
            "bvps",
            "roe",
            "gross_margin",
            "revenue_cum",
            "profit_cum",
            "revenue_yoy",
            "profit_yoy",
        ]
        for col in numeric:
            df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df["net_margin"] = (
            df["profit_cum"] / df["revenue_cum"] * 100.0
        ).replace([np.inf, -np.inf], np.nan)
        return df[
            [
                "ts_code",
                "report_date",
                "announce_date",
                "eps_cum",
                "bvps",
                "roe",
                "gross_margin",
                "net_margin",
                "revenue_cum",
                "profit_cum",
                "revenue_yoy",
                "profit_yoy",
            ]
        ]

    def get_financial_indicator(
        self, ts_code: str, start_year: int
    ) -> pd.DataFrame:
        """Per-stock Sina financial indicators supplementing the earnings
        report with ROA and the debt ratio (the fields the bulk endpoint
        does not carry).  Announcement dates are matched later against the
        earnings-report table, so the rows carry ``report_date`` only."""

        if self.offline:
            df = self._load_fixture(f"financial_{ts_code}")
            if df is None or df.empty:
                return pd.DataFrame()
        else:
            import akshare as ak

            symbol = _symbol_from_ts_code(ts_code)
            df = self._fetch(
                lambda: ak.stock_financial_analysis_indicator(
                    symbol=symbol, start_year=str(start_year)
                )
            )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "日期": "report_date",
                "总资产净利润率(%)": "roa",
                "资产负债率(%)": "debt_ratio",
            }
        )
        if "report_date" not in df.columns:
            return pd.DataFrame()
        df["report_date"] = pd.to_datetime(
            df["report_date"], errors="coerce"
        ).dt.strftime("%Y%m%d")
        df["ts_code"] = ts_code
        df["roa"] = pd.to_numeric(df.get("roa"), errors="coerce")
        df["debt_ratio"] = pd.to_numeric(df.get("debt_ratio"), errors="coerce")
        return df.dropna(subset=["report_date"])[["ts_code", "report_date", "roa", "debt_ratio"]]

    def get_dividend_detail(self, ts_code: str) -> pd.DataFrame:
        """Per-stock dividend records (Eastmoney 分红送配详情).

        The yield is the reported ``现金分红-股息率`` (percent) and the
        point-in-time key is the ex-dividend date (``除权除息日``): before
        that date the dividend is not yet an actionable fact.
        """

        if self.offline:
            df = self._load_fixture(f"dividend_{ts_code}")
            if df is None or df.empty:
                return pd.DataFrame()
        else:
            import akshare as ak

            symbol = _symbol_from_ts_code(ts_code)
            df = self._fetch(lambda: ak.stock_fhps_detail_em(symbol=symbol))
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(
            columns={
                "报告期": "report_date",
                "现金分红-股息率": "dividend_yield",
                "除权除息日": "announce_date",
            }
        )
        if "announce_date" not in df.columns:
            return pd.DataFrame()
        df["report_date"] = pd.to_datetime(
            df["report_date"], errors="coerce"
        ).dt.strftime("%Y%m%d")
        df["announce_date"] = pd.to_datetime(
            df["announce_date"], errors="coerce"
        ).dt.strftime("%Y%m%d")
        # Percent -> fraction.
        df["dividend_yield"] = (
            pd.to_numeric(df["dividend_yield"], errors="coerce") / 100.0
        )
        df["ts_code"] = ts_code
        return df.dropna(subset=["announce_date"])[
            ["ts_code", "report_date", "announce_date", "dividend_yield"]
        ]
