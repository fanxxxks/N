"""AkShare client with graceful fallbacks and optional offline fixtures."""

from __future__ import annotations

import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger

from .processor import is_valid_a_share_code


class AkShareUnavailable(Exception):
    """Raised when an AkShare endpoint is unavailable and no fallback exists."""


def _retry(fn: Callable, retries: int = 3, delay: float = 1.0) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
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

            df = _retry(ak.tool_trade_date_hist_sina, self.config.request_retries)
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
            df = _retry(ak.stock_info_a_code_name, self.config.request_retries)
        except Exception:
            df = _retry(ak.stock_zh_a_spot_em, self.config.request_retries)
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
                df = _retry(lambda: fn(**kwargs), self.config.request_retries)
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
                df = _retry(
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
                df = _retry(
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

    def get_fundamental_snapshot(self, ts_code: str) -> dict[str, float | None]:
        """Best-effort valuation/financial snapshot.

        AkShare endpoints vary over time; this function never raises and returns
        ``None`` for fields that are unavailable, matching the objective's
        degrade-and-log policy.
        """

        if self.offline:
            return {}
        import akshare as ak

        result: dict[str, float | None] = {}
        symbol = _symbol_from_ts_code(ts_code)
        try:
            df = _retry(
                lambda: ak.stock_a_indicator_lg(symbol=symbol),
                self.config.request_retries,
            )
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                result = {
                    "pe_ttm": _safe_float(latest.get("pe_ttm")),
                    "pb": _safe_float(latest.get("pb")),
                    "ps_ttm": _safe_float(latest.get("ps_ttm")),
                    "roe": _safe_float(latest.get("roe_ttm")),
                    "dividend_yield": _safe_float(latest.get("dv_ratio")),
                }
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Fundamental snapshot skipped for {ts_code}: {exc}")
        return result


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
