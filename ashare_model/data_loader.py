"""Load A-share data from DuckDB into model-ready tensors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import torch

from ashare_data.capital_flow import build_capital_frames
from ashare_data.config import DataConfig, ModelConfig, make_data_config
from ashare_data.db import AshareDB
from ashare_data.fundamentals import build_pit_frames
from ashare_data.processor import (
    is_valid_a_share_code,
    normalize_daily_bars,
    open_to_open_returns,
    pivot_wide,
)

from .factors import compute_factor_tensor


class AshareDataLoader:
    def __init__(self, data_config: DataConfig, model_config: ModelConfig | None = None):
        self.config = data_config
        self.model_config = model_config or ModelConfig()
        self.bars: pd.DataFrame | None = None
        self.ts_codes: list[str] = []
        self.dates: list[str] = []
        self.factor_tensor: torch.Tensor | None = None
        self.raw_data_cache: dict[str, torch.Tensor] = {}
        self.target_ret: torch.Tensor | None = None
        self.stock_names: dict[str, str] = {}
        self.st_stocks: set[str] = set()

    def load_stock_meta(self) -> None:
        """Load ts_code -> name and the ST flag set from the stocks table.

        Failures (e.g. the table does not exist yet) leave the maps empty so
        the loader still works on a bare daily-bar database.
        """

        self.stock_names = {}
        self.st_stocks = set()
        try:
            with AshareDB(self.config.duckdb_path, read_only=True) as db:
                df = db.query(
                    f"SELECT ts_code, name, is_st FROM {self.config.stocks_table}"
                )
        except Exception:  # noqa: BLE001 - table may not exist in minimal DBs.
            return
        for row in df.to_dict("records"):
            code = str(row.get("ts_code") or "")
            if not code:
                continue
            name = str(row.get("name") or code)
            self.stock_names[code] = name
            if bool(row.get("is_st")) or "ST" in name.upper():
                self.st_stocks.add(code)

    def load_universe(self) -> list[str]:
        """Default universe: validated constituents (fallback: stock list).

        Index codes and other non-stock instruments are dropped, codes are
        intersected with the stock list when available (authoritative
        membership check), and ST stocks are excluded, matching the
        sync-time universe policy.
        """

        self.load_stock_meta()
        with AshareDB(self.config.duckdb_path, read_only=True) as db:
            try:
                constituents = db.query(
                    f"SELECT DISTINCT ts_code FROM {self.config.constituents_table}"
                )
                codes = sorted(constituents["ts_code"].astype(str).tolist())
            except Exception:  # noqa: BLE001
                codes = []
            try:
                stocks = db.query(f"SELECT ts_code FROM {self.config.stocks_table}")
                stock_codes = set(stocks["ts_code"].astype(str).tolist())
            except Exception:  # noqa: BLE001
                stock_codes = set()
            if not codes:
                codes = sorted(stock_codes)
        codes = [c for c in codes if is_valid_a_share_code(c)]
        if stock_codes:
            codes = [c for c in codes if c in stock_codes]
        codes = [c for c in codes if c not in self.st_stocks]
        return codes

    def load_data(
        self,
        ts_codes: list[str] | None = None,
        dates: list[str] | None = None,
    ) -> "AshareDataLoader":
        self.load_stock_meta()
        if ts_codes is None:
            ts_codes = self.load_universe()
        requested_codes = list(ts_codes)
        if not requested_codes:
            raise ValueError("No universe symbols loaded")

        with AshareDB(self.config.duckdb_path, read_only=True) as db:
            if dates:
                where_dates = "AND trade_date IN (" + ",".join(f"'{d}'" for d in dates) + ")"
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
        self.ts_codes = sorted(df["ts_code"].unique().tolist())
        self.dates = sorted(df["trade_date"].unique().tolist())

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

        factors = compute_factor_tensor(
            df, self.ts_codes, self.dates, pit_fundamentals=pit, extra_frames=capital
        )
        self.factor_tensor = torch.tensor(factors, dtype=torch.float32)

        # Next-open to next-open forward return with a missing-data mask, so
        # suspended / unlisted days can never fabricate huge fake targets.
        open_tensor = self.raw_data_cache["open"]
        self.target_ret = torch.tensor(
            open_to_open_returns(open_tensor.numpy()), dtype=torch.float32
        )
        return self


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
