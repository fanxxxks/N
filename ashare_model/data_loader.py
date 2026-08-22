"""Load A-share data from DuckDB into model-ready tensors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import torch

from ashare_data.capital_flow import build_capital_frames, build_industry_member_frame
from ashare_data.config import DataConfig, ModelConfig, make_data_config
from ashare_data.db import AshareDB
from ashare_data.fundamentals import build_pit_frames
from ashare_data.processor import (
    encode_industry_frame,
    is_valid_a_share_code,
    normalize_daily_bars,
    open_to_open_returns,
    pivot_wide,
    tradability_blocked_matrix,
)
from ashare_data.universe import (
    ResolvedUniverse,
    UniverseContractError,
    UniverseContractStatus,
    UniverseMask,
    UniversePolicy,
    build_universe_mask,
    resolve_universe_contract,
)

from .factors import compute_factor_tensor


def date_index(dates: list[str], date_str: str) -> int:
    """First column index in ``dates`` at or after ``date_str``.

    ``date_str`` accepts ``YYYY-MM-DD`` or ``YYYYMMDD``.  Returns
    ``len(dates)`` when the date lies past the last column and clamps to at
    least 1, mirroring the historical training-window behavior (a window
    must never collapse to index 0).  Single code path shared by the
    trainer and the evaluation protocol.
    """

    key = date_str.replace("-", "")
    for idx, date in enumerate(dates):
        if date >= key:
            return max(idx, 1)
    return len(dates)


class AshareDataLoader:
    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig | None = None,
        *,
        allow_development_universe_fallback: bool = False,
    ):
        self.config = data_config
        self.model_config = model_config or ModelConfig()
        self.allow_development_universe_fallback = bool(
            allow_development_universe_fallback
        )
        self._reset_loaded_state()

    def _reset_loaded_state(self) -> None:
        """Clear every cache whose shape or provenance depends on a load."""

        self.bars: pd.DataFrame | None = None
        self.ts_codes: list[str] = []
        self.dates: list[str] = []
        self.factor_tensor: torch.Tensor | None = None
        self.industry_codes: torch.Tensor | None = None
        self.raw_data_cache: dict[str, torch.Tensor] = {}
        self.target_ret: torch.Tensor | None = None
        self.stock_names: dict[str, str] = {}
        self.stock_list_dates: dict[str, object] = {}
        self.universe: UniverseMask | None = None
        self.universe_policy: UniversePolicy | None = None
        self.universe_status: UniverseContractStatus | None = None
        self._universe_contract: ResolvedUniverse | None = None
        self._tradability_cache: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def universe_mask(self) -> np.ndarray | None:
        """Read-only eligibility view owned by :attr:`universe`."""

        return None if self.universe is None else self.universe.eligible

    @property
    def universe_reason_codes(self) -> np.ndarray | None:
        """Read-only reason-code view owned by :attr:`universe`."""

        return None if self.universe is None else self.universe.reasons

    def _contract(self) -> ResolvedUniverse:
        if self._universe_contract is None:
            self._universe_contract = resolve_universe_contract(
                self.config,
                allow_development_fallback=(
                    self.allow_development_universe_fallback
                ),
            )
            self.universe_status = self._universe_contract.status
        return self._universe_contract

    def load_stock_meta(self) -> None:
        """Load names and listing dates without interpreting current ST state.

        Failures (e.g. the table does not exist yet) leave the maps empty so
        the loader still works on a bare daily-bar database.
        """

        self.stock_names = {}
        self.stock_list_dates = {}
        try:
            with AshareDB(self.config.duckdb_path, read_only=True) as db:
                df = db.query(
                    f"SELECT ts_code, name, list_date "
                    f"FROM {self.config.stocks_table}"
                )
        except Exception:  # noqa: BLE001 - table may not exist in minimal DBs.
            return
        for row in df.to_dict("records"):
            code = str(row.get("ts_code") or "")
            if not code:
                continue
            name = str(row.get("name") or code)
            self.stock_names[code] = name
            self.stock_list_dates[code] = row.get("list_date")

    def load_universe(self) -> list[str]:
        """Validated union of configured PIT constituent intervals.

        This method only selects configured indices and returns every valid
        A-share code that appears in any interval.  Per-date eligibility is
        built exactly once by :func:`build_universe_mask` in ``load_data``.
        A stock-list/all-period fallback exists only when the loader was
        explicitly constructed in development mode.
        """

        contract = self._contract()
        return sorted(c for c in contract.codes if is_valid_a_share_code(c))

    def _membership_records(self, contract: ResolvedUniverse) -> list[dict]:
        if not contract.status.degraded:
            return contract.constituents.to_dict("records")
        # The explicit development fallback is an in-memory all-period
        # membership only.  It is never persisted and its degraded provenance
        # remains visible through ``universe_status``.
        index_code = str(self.config.index_codes[0])
        return [
            {
                "index_code": index_code,
                "ts_code": code,
                "in_date": contract.sessions[0],
                "out_date": "99991231",
            }
            for code in contract.codes
        ]

    def _build_universe(
        self,
        contract: ResolvedUniverse,
        bars: pd.DataFrame,
    ) -> None:
        presence = pivot_wide(
            bars.assign(_bar_present=1.0),
            self.ts_codes,
            self.dates,
            "_bar_present",
        ).notna().to_numpy(dtype=bool)
        policy = UniversePolicy(
            index_codes=tuple(str(code) for code in self.config.index_codes),
            min_listed_sessions=self.config.min_listed_sessions,
            membership_end_inclusive=False,
        )
        self.universe_policy = policy
        self.universe = build_universe_mask(
            self.ts_codes,
            self.dates,
            contract.sessions,
            self._membership_records(contract),
            self.stock_list_dates,
            presence,
            policy,
        )

    def load_data(
        self,
        ts_codes: list[str] | None = None,
        dates: list[str] | None = None,
    ) -> "AshareDataLoader":
        self._reset_loaded_state()
        contract = self._contract()
        self.load_stock_meta()
        if ts_codes is None:
            ts_codes = self.load_universe()
        requested_codes = sorted({str(code) for code in ts_codes})
        if not requested_codes:
            raise ValueError("No universe symbols loaded")
        uncovered = sorted(set(requested_codes) - set(contract.codes))
        if uncovered:
            raise UniverseContractError(
                "requested ts_codes are not covered by the resolved universe: "
                + ", ".join(uncovered[:5])
            )

        requested_dates = (
            sorted({str(date).replace("-", "") for date in dates})
            if dates
            else None
        )
        if requested_dates is not None:
            non_sessions = sorted(set(requested_dates) - set(contract.sessions))
            if non_sessions:
                raise UniverseContractError(
                    "requested dates are not trade_calendar.is_open=True sessions: "
                    + ", ".join(non_sessions[:5])
                )

        with AshareDB(self.config.duckdb_path, read_only=True) as db:
            if requested_dates:
                where_dates = "AND trade_date IN (" + ",".join(
                    f"'{date}'" for date in requested_dates
                ) + ")"
            else:
                where_dates = ""
            code_list = ",".join(f"'{c}'" for c in requested_codes)
            df = db.query(
                f"""
                SELECT ts_code, trade_date, open, high, low, close, pre_close,
                       volume, amount, turnover_rate, adj_factor
                FROM {self.config.daily_table}
                WHERE ts_code IN ({code_list}) {where_dates}
                ORDER BY trade_date
                """
            )

        df = normalize_daily_bars(df)
        if df.empty:
            raise ValueError("No daily bars found")
        self.bars = df
        self.ts_codes = requested_codes
        if requested_dates is not None:
            self.dates = requested_dates
        else:
            first_bar = str(df["trade_date"].min())
            last_bar = str(df["trade_date"].max())
            self.dates = [
                date for date in contract.sessions if first_bar <= date <= last_bar
            ]
        if not self.dates:
            raise UniverseContractError(
                "no trade_calendar.is_open=True sessions overlap the loaded bars"
            )
        self._build_universe(contract, df)

        tensor_map: dict[str, torch.Tensor] = {}
        for column in (
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "amount",
            "turnover_rate",
            "adj_factor",
        ):
            wide = pivot_wide(df, self.ts_codes, self.dates, column)
            tensor_map[column] = torch.tensor(
                np.nan_to_num(wide.values, nan=0.0, posinf=0.0, neginf=0.0),
                dtype=torch.float32,
            )
        self.raw_data_cache = tensor_map

        # Announce-date-aligned point-in-time fundamentals and capital-flow
        # frames; any failure (e.g. the tables do not exist yet) degrades
        # to neutral frames.
        close_wide = pivot_wide(df, self.ts_codes, self.dates, "close")
        pit = build_pit_frames(self.config, self.ts_codes, self.dates, close_wide)
        capital = build_capital_frames(self.config, self.ts_codes, self.dates)
        industry_frame = build_industry_member_frame(
            self.config, self.ts_codes, self.dates
        )

        factors = compute_factor_tensor(
            df,
            self.ts_codes,
            self.dates,
            pit_fundamentals=pit,
            extra_frames=capital,
            industry_frame=industry_frame,
            universe_mask=self.universe_mask,
        )
        self.factor_tensor = torch.tensor(factors, dtype=torch.float32)
        # Dense industry group ids for the VM's CS_NEUTRALIZE operator
        # (unmapped stocks stay NaN so no group is fabricated for them).
        self.industry_codes = torch.tensor(
            encode_industry_frame(industry_frame), dtype=torch.float32
        )

        # Next-open to next-open forward return with a missing-data mask, so
        # suspended / unlisted days can never fabricate huge fake targets.
        open_tensor = self.raw_data_cache["open"]
        target = self.mask_by_universe(open_to_open_returns(open_tensor.numpy()))
        self.target_ret = torch.tensor(target, dtype=torch.float32)
        return self

    def mask_by_universe(
        self,
        values: np.ndarray,
        *,
        start: int = 0,
    ) -> np.ndarray:
        """Replace values outside the PIT eligibility mask with ``NaN``."""

        if self.universe_mask is None:
            raise ValueError("universe mask requires loaded data")
        array = np.asarray(values, dtype=np.float64).copy()
        mask = self.universe_mask[:, start : start + array.shape[1]]
        if array.shape != mask.shape:
            raise ValueError(
                f"values shape {array.shape} does not match universe mask {mask.shape}"
            )
        array[~mask] = np.nan
        return array

    def tradability_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """``(blocked_buy, blocked_sell)`` ``[stock x date]`` bool matrices.

        Built once from the raw OHLCV cache with the exact rule the backtest
        engine applies per execution day (0-filled missing cells mark
        suspension, one-word limit-up opens block buys, one-word limit-down
        opens block sells).  The reward path consumes these masks so the
        training basket can never trade what the engine could not have
        traded; callers slice them to the same date window as the signals.
        """

        cached = self._tradability_cache
        if cached is not None:
            return cached
        if not self.raw_data_cache:
            raise ValueError("tradability masks require loaded raw bars")
        raw = self.raw_data_cache
        blocked_buy = tradability_blocked_matrix(
            raw["open"].numpy(),
            raw["high"].numpy(),
            raw["low"].numpy(),
            raw["pre_close"].numpy(),
            raw["volume"].numpy(),
            self.ts_codes,
            self.stock_names,
            "buy",
        )
        blocked_sell = tradability_blocked_matrix(
            raw["open"].numpy(),
            raw["high"].numpy(),
            raw["low"].numpy(),
            raw["pre_close"].numpy(),
            raw["volume"].numpy(),
            self.ts_codes,
            self.stock_names,
            "sell",
        )
        if self.universe_mask is None:
            raise ValueError("tradability masks require loaded universe mask")
        # Ineligible stocks cannot be newly bought.  Sell eligibility remains
        # market-driven so an index exit never traps an existing position.
        blocked_buy |= ~self.universe_mask
        self._tradability_cache = (blocked_buy, blocked_sell)
        return self._tradability_cache


def build_loader_from_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> AshareDataLoader:
    from ashare_data.config import load_config, make_model_config

    root = Path(project_root)
    raw = load_config(config_path, project_root=root)
    data_config = make_data_config(raw, root)
    model_config = make_model_config(raw)
    return AshareDataLoader(data_config, model_config)
